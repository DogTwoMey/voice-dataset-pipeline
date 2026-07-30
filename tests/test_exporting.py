from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest

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
    ClipRecord,
    MediaKind,
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
