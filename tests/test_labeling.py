from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from voice_dataset_pipeline.labeling import label_clips
from voice_dataset_pipeline.models import ClipRecord, LabelRecord
from voice_dataset_pipeline.workspace import Workspace


def _clip(tmp_path: Path, number: int) -> ClipRecord:
    audio = tmp_path / f"clip-{number}.wav"
    audio.write_bytes(b"test")
    return ClipRecord(
        clip_id=f"{number:064x}",
        source_id=f"source-{number}",
        audio_path=audio,
        start_ms=0,
        end_ms=1000,
        sample_rate=16_000,
        frames=16_000,
        sha256=f"{number + 100:064x}",
    )


class _FailOnSecondLabeler:
    def __init__(self, failing_clip_id: str) -> None:
        self.failing_clip_id = failing_clip_id
        self.calls: list[str] = []

    def label(
        self,
        *,
        path: Path,
        clip_id: str,
        emotions: Sequence[str],
        clusters: Sequence[str],
        language_hint: str = "auto",
    ) -> LabelRecord:
        self.calls.append(clip_id)
        if clip_id == self.failing_clip_id:
            raise RuntimeError("simulated interruption")
        return LabelRecord(
            clip_id=clip_id,
            transcript=f"text for {path.stem}",
            emotion=emotions[0],
            cluster="cluster_a",
            confidence=0.9,
            model="fake",
        )


class _RecordingLabeler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def label(
        self,
        *,
        path: Path,
        clip_id: str,
        emotions: Sequence[str],
        clusters: Sequence[str],
        language_hint: str = "auto",
    ) -> LabelRecord:
        self.calls.append(clip_id)
        return LabelRecord(
            clip_id=clip_id,
            transcript=f"resumed {path.stem}",
            emotion=emotions[-1],
            cluster="cluster_b",
            confidence=0.8,
            model="fake-resume",
        )


def test_label_clips_persists_each_success_and_resumes_after_failure(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clips = [_clip(tmp_path, 1), _clip(tmp_path, 2)]
    workspace.write_jsonl(workspace.paths.clips_jsonl, clips)
    interrupted = _FailOnSecondLabeler(clips[1].clip_id)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        label_clips(
            workspace,
            interrupted,
            emotions=["neutral", "happy"],
            clusters=["cluster_a", "cluster_b", "unknown"],
            language_hint="zh",
        )

    persisted = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert [row.clip_id for row in persisted] == [clips[0].clip_id]
    assert interrupted.calls == [clips[0].clip_id, clips[1].clip_id]
    assert not list(workspace.paths.manifests.glob("*.tmp"))

    resumed = _RecordingLabeler()
    summary = label_clips(
        workspace,
        resumed,
        emotions=["neutral", "happy"],
        clusters=["cluster_a", "cluster_b", "unknown"],
        language_hint="zh",
    )

    assert summary.total == 2
    assert summary.labelled == 1
    assert summary.skipped == 1
    assert resumed.calls == [clips[1].clip_id]
    final = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert [row.clip_id for row in final] == [
        clips[0].clip_id,
        clips[1].clip_id,
    ]
    assert final[0].model == "fake"
    assert final[1].model == "fake-resume"


def test_label_clips_upgrades_provisional_sensevoice_seed_without_force(
    tmp_path: Path,
) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    clip = _clip(tmp_path, 3)
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    workspace.write_jsonl(
        workspace.paths.labels_jsonl,
        [
            LabelRecord(
                clip_id=clip.clip_id,
                transcript="ASR draft",
                model="iic/SenseVoiceSmall",
                rationale="local SenseVoice seed; requires review",
            )
        ],
    )
    labeler = _RecordingLabeler()

    summary = label_clips(
        workspace,
        labeler,
        emotions=["neutral", "happy"],
        clusters=["cluster_a", "cluster_b", "unknown"],
        language_hint="zh",
    )

    assert summary.labelled == 1
    assert summary.skipped == 0
    final = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert len(final) == 1
    assert final[0].model == "fake-resume"
