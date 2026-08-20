from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from voice_dataset_pipeline.models import ClipRecord, LabelRecord, ReviewState
from voice_dataset_pipeline.review import merge_adjacent_clips, review_workspace
from voice_dataset_pipeline.workspace import Workspace


def _merge_workspace(
    tmp_path: Path, *, include_third: bool = False
) -> tuple[Workspace, ClipRecord, ClipRecord]:
    import numpy as np
    import soundfile as sf

    from voice_dataset_pipeline.media import sha256_file
    from voice_dataset_pipeline.models import (
        ASRRecord,
        QualityRecord,
        ReviewDecision,
        Segment,
        SourceRecord,
    )

    workspace = Workspace.create(tmp_path / "workspace")
    source_wav = workspace.paths.normalized_audio / "source.wav"
    source_frames = 96_000 if include_third else 64_000
    sf.write(source_wav, np.zeros(source_frames, dtype=np.float32), 32_000)
    source = SourceRecord(
        source_id="s" * 64,
        content_sha256="c" * 64,
        original_path=source_wav,
        normalized_path=source_wav,
        media_kind="audio",
        size_bytes=source_wav.stat().st_size,
        sample_rate=32_000,
        channels=1,
        frames=source_frames,
        duration_seconds=source_frames / 32_000,
    )
    workspace.write_jsonl(workspace.paths.sources_jsonl, [source])
    pair: list[ClipRecord] = []
    intervals = [("a", 0, 1_000), ("b", 1_000, 2_000)]
    if include_third:
        intervals.append(("c", 2_000, 3_000))
    for name, start_ms, end_ms in intervals:
        path = workspace.paths.clips / f"{name}.wav"
        sf.write(path, np.zeros(32_000, dtype=np.float32), 32_000)
        pair.append(
            ClipRecord(
                clip_id=name * 64,
                source_id=source.source_id,
                audio_path=path,
                start_ms=start_ms,
                end_ms=end_ms,
                sample_rate=32_000,
                frames=32_000,
                sha256=sha256_file(path),
            )
        )
    left, right = pair[:2]
    workspace.write_jsonl(workspace.paths.clips_jsonl, pair)
    workspace.write_jsonl(
        workspace.paths.segments_jsonl,
        [
            Segment(
                source_id=source.source_id,
                start_seconds=clip.start_seconds,
                end_seconds=clip.end_seconds,
            )
            for clip in pair
        ],
    )
    workspace.write_jsonl(
        workspace.paths.labels_jsonl,
        [LabelRecord(clip_id=clip.clip_id, transcript=f"{clip.clip_id[0]}。") for clip in pair],
    )
    workspace.write_jsonl(
        workspace.paths.quality_jsonl,
        [
            QualityRecord(
                clip_id=clip.clip_id,
                audio_path=clip.audio_path,
                audio_sha256=clip.sha256,
                profile_sha256="q" * 64,
                duration_seconds=1,
                rms_dbfs=-20,
                peak_dbfs=-3,
                clipping_ratio=0,
                silence_ratio=0,
                accepted=True,
            )
            for clip in pair
        ],
    )
    workspace.write_jsonl(
        workspace.paths.asr_jsonl,
        [ASRRecord(clip_id=clip.clip_id, audio_sha256=clip.sha256) for clip in pair],
    )
    workspace.save_review(
        ReviewState(
            order=[clip.clip_id for clip in pair],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id, transcript=f"{clip.clip_id[0]}。"
                )
                for clip in pair
            },
        )
    )
    return workspace, left, right


def test_merge_adjacent_clips_replaces_active_pair(tmp_path: Path) -> None:
    workspace, left, right = _merge_workspace(tmp_path)
    merged, label = merge_adjacent_clips(
        workspace,
        left,
        right,
        transcript="完整台词。",
        emotion="neutral",
        cluster="neutral",
    )
    active = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert [row.clip_id for row in active] == [merged.clip_id]
    assert merged.frames == 64_000
    assert label.transcript == "完整台词。"


def test_merge_rejects_tui_neighbors_that_skip_an_active_timeline_clip(tmp_path: Path) -> None:
    workspace, left, _middle = _merge_workspace(tmp_path, include_third=True)
    active_before = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert isinstance(active_before, list)
    right = active_before[2]

    review = workspace.load_review()
    review.order = [left.clip_id, right.clip_id, active_before[1].clip_id]
    workspace.save_review(review)

    with pytest.raises(ValueError, match="direct timeline neighbors"):
        merge_adjacent_clips(
            workspace,
            left,
            right,
            transcript="不能跨过中间片段。",
            emotion="neutral",
            cluster="neutral",
        )

    active_after = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert active_after == active_before
    assert active_before[1].clip_id in {clip.clip_id for clip in active_after}


def test_merge_rejects_overlapping_active_timeline_neighbors(tmp_path: Path) -> None:
    workspace, left, right = _merge_workspace(tmp_path)
    overlapping = right.model_copy(update={"start_ms": left.end_ms - 1})
    workspace.write_jsonl(workspace.paths.clips_jsonl, [left, overlapping])

    with pytest.raises(ValueError, match="overlap"):
        merge_adjacent_clips(
            workspace,
            left,
            overlapping,
            transcript="重叠片段。",
            emotion="neutral",
            cluster="neutral",
        )


def test_merge_rejects_pair_that_was_already_replaced(tmp_path: Path) -> None:
    workspace, left, right = _merge_workspace(tmp_path)
    merged, _ = merge_adjacent_clips(
        workspace,
        left,
        right,
        transcript="第一次合并。",
        emotion="neutral",
        cluster="neutral",
    )

    with pytest.raises(ValueError, match="canonical active manifest"):
        merge_adjacent_clips(
            workspace,
            left,
            right,
            transcript="不应再次合并。",
            emotion="neutral",
            cluster="neutral",
        )

    active = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert [clip.clip_id for clip in active] == [merged.clip_id]
    assert not workspace.paths.pending_review_merge_json.exists()


@pytest.mark.parametrize(
    "target_name",
    ["clips.jsonl", "segments.jsonl", "quality.jsonl", "asr.jsonl", "labels.jsonl", "review.json"],
)
def test_review_start_recovers_merge_interrupted_at_each_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_name: str
) -> None:
    import voice_dataset_pipeline.workspace as workspace_module

    workspace, left, right = _merge_workspace(tmp_path)
    original_atomic_replace = workspace_module._atomic_replace
    interrupted = False

    def fail_at_projection(path: Path, payload: bytes) -> None:
        nonlocal interrupted
        if path.name == target_name and not interrupted:
            interrupted = True
            raise OSError("simulated power loss")
        original_atomic_replace(path, payload)

    monkeypatch.setattr(workspace_module, "_atomic_replace", fail_at_projection)
    with pytest.raises(OSError, match="simulated power loss"):
        merge_adjacent_clips(
            workspace,
            left,
            right,
            transcript="前半句，后半句。",
            emotion="neutral",
            cluster="neutral",
        )

    receipt = workspace.paths.pending_review_merge_json
    assert receipt.is_file()
    monkeypatch.setattr(workspace_module, "_atomic_replace", original_atomic_replace)

    recovered = review_workspace(
        workspace,
        emotions=["neutral"],
        key_reader=_keys("q"),
        clear_screen=False,
    )

    assert not receipt.exists()
    assert len(recovered.order) == 1
    assert recovered.order[0] in recovered.decisions
    assert recovered.decisions[recovered.order[0]].transcript == "前半句，后半句。"
    active = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert [row.clip_id for row in active] == recovered.order


def test_review_start_replays_receipt_left_after_all_projections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, left, right = _merge_workspace(tmp_path)
    receipt = workspace.paths.pending_review_merge_json
    original_unlink = Path.unlink
    interrupted = False

    def fail_receipt_cleanup(path: Path, *args, **kwargs) -> None:
        nonlocal interrupted
        if path == receipt and not interrupted:
            interrupted = True
            raise OSError("simulated power loss after commit")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_receipt_cleanup)
    with pytest.raises(OSError, match="after commit"):
        merge_adjacent_clips(
            workspace,
            left,
            right,
            transcript="前半句，后半句。",
            emotion="neutral",
            cluster="neutral",
        )
    assert receipt.is_file()
    committed = workspace.load_review()
    assert len(committed.order) == 1

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = review_workspace(
        workspace,
        emotions=["neutral"],
        key_reader=_keys("q"),
        clear_screen=False,
    )

    assert recovered == workspace.load_review()
    assert not receipt.exists()
    assert workspace.recover_review_merge() is False


def _clip(tmp_path: Path, number: int) -> ClipRecord:
    audio = tmp_path / f"clip-{number}.wav"
    audio.write_bytes(b"not-played")
    return ClipRecord(
        clip_id=f"{number:064x}",
        source_id=f"source-{number}",
        audio_path=audio,
        start_ms=0,
        end_ms=500,
        sample_rate=16_000,
        frames=8_000,
        sha256=f"{number + 100:064x}",
    )


def _keys(*values: str):
    iterator: Iterator[str] = iter(values)
    return lambda: next(iterator)


def test_review_numeric_choice_and_exclude_resume_from_saved_cursor(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clips = [_clip(tmp_path, 1), _clip(tmp_path, 2)]
    workspace.write_jsonl(workspace.paths.clips_jsonl, clips)
    workspace.write_jsonl(
        workspace.paths.labels_jsonl,
        [
            LabelRecord(
                clip_id=clips[0].clip_id,
                transcript="first line",
                emotion="unknown",
                cluster="unknown",
            ),
            LabelRecord(
                clip_id=clips[1].clip_id,
                transcript="second line",
                emotion="sad",
                cluster="soft",
            ),
        ],
    )

    interrupted = review_workspace(
        workspace,
        emotions=["neutral", "happy", "sad"],
        key_reader=_keys("2", "q"),
        clear_screen=False,
    )

    assert interrupted.cursor == 1
    assert interrupted.decisions[clips[0].clip_id].emotion == "happy"
    assert interrupted.decisions[clips[0].clip_id].cluster == "happy"
    persisted = workspace.load_review()
    assert persisted.cursor == 1
    assert clips[1].clip_id not in persisted.decisions

    resumed = review_workspace(
        Workspace.open(workspace.root, create=False),
        emotions=["neutral", "happy", "sad"],
        key_reader=_keys("x"),
        clear_screen=False,
    )

    assert resumed.cursor == 2
    assert resumed.decisions[clips[0].clip_id].excluded is False
    assert resumed.decisions[clips[1].clip_id].excluded is True
    assert resumed.decisions[clips[1].clip_id].transcript == "second line"
    assert Workspace.open(workspace.root, create=False).load_review() == resumed


def test_review_undo_restores_cursor_and_previous_decision(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clips = [_clip(tmp_path, 1), _clip(tmp_path, 2)]
    workspace.write_jsonl(workspace.paths.clips_jsonl, clips)

    state = review_workspace(
        workspace,
        emotions=["neutral", "happy"],
        key_reader=_keys("2", "b", "1", "x"),
        clear_screen=False,
    )

    assert state.cursor == 2
    assert state.decisions[clips[0].clip_id].emotion == "neutral"
    assert state.decisions[clips[0].clip_id].excluded is False
    assert state.decisions[clips[1].clip_id].excluded is True
    assert len(state.history) == 2
