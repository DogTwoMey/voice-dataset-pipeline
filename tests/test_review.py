from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from voice_dataset_pipeline.models import ClipRecord, LabelRecord
from voice_dataset_pipeline.review import review_workspace
from voice_dataset_pipeline.workspace import Workspace


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
