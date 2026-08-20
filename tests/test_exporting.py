from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from voice_dataset_pipeline.asr import ASRProfile
from voice_dataset_pipeline.cli import (
    _dataset_config_fingerprint,
    _verify_current_export,
)
from voice_dataset_pipeline.config import load_config
from voice_dataset_pipeline.exporting import (
    TrainingRecord,
    export_training_dataset,
)
from voice_dataset_pipeline.models import (
    ASRRecord,
    ClipRecord,
    MediaKind,
    QualityRecord,
    ReviewDecision,
    ReviewState,
    SourceRecord,
)
from voice_dataset_pipeline.workspace import Workspace


def _wav_clip(tmp_path: Path, number: int) -> ClipRecord:
    path = tmp_path / f"clip-{number}.wav"
    sample_rate = 16_000
    frames = 800
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x00\x00" * frames)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ClipRecord(
        clip_id=f"{number:064x}",
        source_id=f"{number + 10:064x}",
        audio_path=path,
        start_ms=0,
        end_ms=50,
        sample_rate=sample_rate,
        frames=frames,
        sha256=digest,
    )


def _source(clip: ClipRecord) -> SourceRecord:
    return SourceRecord(
        source_id=clip.source_id,
        content_sha256=clip.sha256,
        original_path=clip.audio_path,
        normalized_path=clip.audio_path,
        media_kind=MediaKind.AUDIO,
        size_bytes=clip.audio_path.stat().st_size,
        sample_rate=clip.sample_rate,
        channels=1,
        frames=clip.frames,
        duration_seconds=clip.frames / clip.sample_rate,
    )


def _reviewed_workspace(tmp_path: Path) -> tuple[Workspace, ClipRecord]:
    workspace = Workspace.create(tmp_path / "workspace")
    clip = _wav_clip(tmp_path, 1).model_copy(update={"text": "字幕提示。"})
    workspace.write_jsonl(workspace.paths.sources_jsonl, [_source(clip)])
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    workspace.save_review(
        ReviewState(
            cursor=1,
            order=[clip.clip_id],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id,
                    transcript="人工确认台词。",
                    emotion="neutral",
                    cluster="conversational",
                    confirmed=True,
                )
            },
        )
    )
    return workspace, clip


def _quality_record(
    clip: ClipRecord,
    *,
    profile_sha256: str,
    accepted: bool = True,
) -> QualityRecord:
    return QualityRecord(
        clip_id=clip.clip_id,
        audio_path=clip.audio_path,
        audio_sha256=clip.sha256,
        profile_sha256=profile_sha256,
        duration_seconds=clip.frames / clip.sample_rate,
        rms_dbfs=-20,
        peak_dbfs=-3,
        clipping_ratio=0,
        silence_ratio=0,
        accepted=accepted,
        reasons=[] if accepted else ["quality_rejected"],
    )


def _asr_profile() -> ASRProfile:
    return ASRProfile(
        model="sensevoice",
        vad_model="vad",
        language="auto",
        replacements={},
        minimum_similarity=0.72,
        require_expected_match=True,
    )


def _asr_record(
    clip: ClipRecord,
    profile: ASRProfile,
    *,
    transcript: str = "人工确认台词。",
    expected_text: str | None = None,
    accepted: bool = True,
    profile_sha256: str | None = None,
) -> ASRRecord:
    expected = clip.text if expected_text is None else expected_text
    return ASRRecord(
        clip_id=clip.clip_id,
        audio_sha256=clip.sha256,
        profile_sha256=profile_sha256 or profile.fingerprint(expected),
        transcript=transcript,
        model=profile.model,
        expected_text=expected,
        transcript_similarity=1,
        accepted=accepted,
        reasons=[] if accepted else ["asr_rejected"],
    )


def test_export_materializes_explorer_text_gpt_list_and_flat_rvc_dataset(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    included = _wav_clip(tmp_path, 1)
    excluded = _wav_clip(tmp_path, 2)
    workspace.write_jsonl(
        workspace.paths.sources_jsonl,
        [_source(included), _source(excluded)],
    )
    workspace.write_jsonl(workspace.paths.clips_jsonl, [included, excluded])
    workspace.save_review(
        ReviewState(
            cursor=2,
            order=[included.clip_id, excluded.clip_id],
            decisions={
                included.clip_id: ReviewDecision(
                    clip_id=included.clip_id,
                    transcript="你好，我是测试语音。",
                    emotion="happy",
                    cluster="bright_voice",
                    confirmed=True,
                ),
                excluded.clip_id: ReviewDecision(
                    clip_id=excluded.clip_id,
                    transcript="不要导出。",
                    emotion="sad",
                    cluster="soft_voice",
                    excluded=True,
                    confirmed=True,
                ),
            },
        )
    )

    result = export_training_dataset(
        workspace,
        output_root=tmp_path / "exports",
        speaker="character",
        language="zh",
    )

    assert result.included == 1
    assert result.excluded == 1
    explorer_wav = result.root / "reviewed" / "happy" / "bright_voice" / f"{included.clip_id}.wav"
    assert explorer_wav.is_file()
    assert explorer_wav.with_suffix(".txt").read_text(encoding="utf-8") == (
        "你好，我是测试语音。\n"
    )

    fields = result.gpt_sovits_list.read_text(encoding="utf-8").strip().split("|")
    assert fields == [
        str(explorer_wav.resolve()),
        "character",
        "zh",
        "你好，我是测试语音。",
    ]
    rvc_entries = list(result.rvc_dataset.iterdir())
    assert rvc_entries == [result.rvc_dataset / f"{included.clip_id}.wav"]
    assert rvc_entries[0].is_file()

    manifest = workspace.read_jsonl(result.manifest, TrainingRecord)
    assert len(manifest) == 1
    assert manifest[0].clip_id == included.clip_id
    assert manifest[0].reviewed is True


def test_export_rejects_any_unreviewed_clip_by_default(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clip = _wav_clip(tmp_path, 1)
    workspace.write_jsonl(workspace.paths.sources_jsonl, [_source(clip)])
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])

    with pytest.raises(ValueError, match="has not been reviewed"):
        export_training_dataset(
            workspace,
            output_root=tmp_path / "exports",
        )


@pytest.mark.parametrize("state", ["missing", "rejected", "stale"])
def test_export_quality_gate_fails_closed(tmp_path: Path, state: str) -> None:
    workspace, clip = _reviewed_workspace(tmp_path)
    profile_sha256 = "a" * 64
    if state != "missing":
        record = _quality_record(
            clip,
            profile_sha256="b" * 64 if state == "stale" else profile_sha256,
            accepted=state != "rejected",
        )
        workspace.write_jsonl(workspace.paths.quality_jsonl, [record])

    with pytest.raises(ValueError, match="no included clips"):
        export_training_dataset(
            workspace,
            output_root=tmp_path / "exports",
            quality_enabled=True,
            quality_profile_sha256=profile_sha256,
        )


@pytest.mark.parametrize("state", ["missing", "rejected", "stale"])
def test_export_required_asr_gate_fails_closed(tmp_path: Path, state: str) -> None:
    workspace, clip = _reviewed_workspace(tmp_path)
    profile = _asr_profile()
    if state != "missing":
        record = _asr_record(
            clip,
            profile,
            transcript="" if state == "rejected" else "人工确认台词。",
            accepted=state != "rejected",
            profile_sha256="c" * 64 if state == "stale" else None,
        )
        workspace.write_jsonl(workspace.paths.asr_jsonl, [record])

    with pytest.raises(ValueError, match="no included clips"):
        export_training_dataset(
            workspace,
            output_root=tmp_path / "exports",
            require_asr=True,
            asr_profile=profile,
        )


def test_export_respects_disabled_gates_and_accepts_current_strict_records(
    tmp_path: Path,
) -> None:
    workspace, clip = _reviewed_workspace(tmp_path)
    quality_profile = "a" * 64
    asr_profile = _asr_profile()
    workspace.write_jsonl(
        workspace.paths.quality_jsonl,
        [_quality_record(clip, profile_sha256=quality_profile, accepted=False)],
    )
    workspace.write_jsonl(
        workspace.paths.asr_jsonl,
        [_asr_record(clip, asr_profile, accepted=False)],
    )
    disabled = export_training_dataset(
        workspace,
        output_root=tmp_path / "disabled",
    )
    assert disabled.included == 1

    workspace.write_jsonl(
        workspace.paths.quality_jsonl,
        [_quality_record(clip, profile_sha256=quality_profile)],
    )
    workspace.write_jsonl(
        workspace.paths.asr_jsonl,
        [_asr_record(clip, asr_profile)],
    )
    strict = export_training_dataset(
        workspace,
        output_root=tmp_path / "strict",
        quality_enabled=True,
        quality_profile_sha256=quality_profile,
        require_asr=True,
        asr_profile=asr_profile,
    )
    assert strict.included == 1


def test_export_strict_asr_compares_reviewed_text_when_clip_has_no_subtitle(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clip = _wav_clip(tmp_path, 1)
    workspace.write_jsonl(workspace.paths.sources_jsonl, [_source(clip)])
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    workspace.save_review(
        ReviewState(
            cursor=1,
            order=[clip.clip_id],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id,
                    transcript="人工确认台词。",
                    emotion="neutral",
                    cluster="conversational",
                    confirmed=True,
                )
            },
        )
    )
    profile = _asr_profile()
    workspace.write_jsonl(
        workspace.paths.asr_jsonl,
        [
            _asr_record(
                clip,
                profile,
                transcript="人工确认台词。",
                expected_text="",
                accepted=False,
            ).model_copy(update={"reasons": ["missing_expected_text"]})
        ],
    )

    result = export_training_dataset(
        workspace,
        output_root=tmp_path / "exports",
        require_asr=True,
        asr_profile=profile,
    )

    assert result.included == 1


def test_export_strict_asr_rejects_review_text_that_does_not_match_audio_asr(
    tmp_path: Path,
) -> None:
    workspace, clip = _reviewed_workspace(tmp_path)
    profile = _asr_profile()
    workspace.write_jsonl(
        workspace.paths.asr_jsonl,
        [_asr_record(clip, profile, transcript="字幕提示。", accepted=True)],
    )

    with pytest.raises(ValueError, match="no included clips"):
        export_training_dataset(
            workspace,
            output_root=tmp_path / "exports",
            require_asr=True,
            asr_profile=profile,
        )


def test_training_rejects_export_after_review_changes(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clip = _wav_clip(tmp_path, 1)
    workspace.write_jsonl(workspace.paths.sources_jsonl, [_source(clip)])
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    workspace.save_review(
        ReviewState(
            cursor=1,
            order=[clip.clip_id],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id,
                    transcript="最初的文本。",
                    emotion="neutral",
                    cluster="conversational",
                    confirmed=True,
                )
            },
        )
    )
    config = load_config()
    result = export_training_dataset(
        workspace,
        output_root=tmp_path / "exports",
        config_fingerprint=_dataset_config_fingerprint(config),
    )
    workspace.save_review(
        ReviewState(
            cursor=1,
            order=[clip.clip_id],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id,
                    transcript="人工修订后的文本。",
                    emotion="happy",
                    cluster="bright_playful",
                    confirmed=True,
                )
            },
        )
    )

    with pytest.raises(ValueError, match="过期"):
        _verify_current_export(workspace, config, result.root)


@pytest.mark.parametrize("target", ["manifest", "dataset_list"])
def test_training_rejects_export_after_materialized_content_is_modified(
    tmp_path: Path,
    target: str,
) -> None:
    workspace, _clip = _reviewed_workspace(tmp_path)
    config = load_config()
    result = export_training_dataset(
        workspace,
        output_root=tmp_path / "exports",
        config_fingerprint=_dataset_config_fingerprint(config),
    )
    path = result.manifest if target == "manifest" else result.gpt_sovits_list
    path.write_text(path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        _verify_current_export(workspace, config, result.root)


def test_training_rejects_legacy_export_without_manifest_content_hash(tmp_path: Path) -> None:
    workspace, _clip = _reviewed_workspace(tmp_path)
    config = load_config()
    result = export_training_dataset(
        workspace,
        output_root=tmp_path / "exports",
        config_fingerprint=_dataset_config_fingerprint(config),
    )
    metadata_path = result.root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("manifest_sha256")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="missing the manifest content hash"):
        _verify_current_export(workspace, config, result.root)


def test_dataset_fingerprint_ignores_training_only_configuration() -> None:
    config = load_config()
    enabled = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "enabled": True,
                    "gpt_sovits": config.training.gpt_sovits.model_copy(
                        update={"enabled": True, "gpt_epochs": 99}
                    ),
                }
            )
        }
    )

    assert _dataset_config_fingerprint(enabled) == _dataset_config_fingerprint(config)


def test_dataset_fingerprint_includes_default_speaker_and_language() -> None:
    config = load_config()
    changed = config.model_copy(
        update={
            "training": config.training.model_copy(
                update={
                    "gpt_sovits": config.training.gpt_sovits.model_copy(
                        update={"speaker": "new_character", "language": "en"}
                    )
                }
            )
        }
    )

    assert _dataset_config_fingerprint(changed) != _dataset_config_fingerprint(config)
