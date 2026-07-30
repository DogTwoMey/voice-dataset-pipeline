"""Resumable labelling orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import ClipRecord, LabelRecord
from .workspace import Workspace


class ClipLabeler(Protocol):
    def label(
        self,
        *,
        path: Path,
        clip_id: str,
        emotions: Sequence[str],
        clusters: Sequence[str],
        language_hint: str = "auto",
    ) -> LabelRecord: ...


@dataclass(frozen=True, slots=True)
class LabelSummary:
    total: int
    labelled: int
    skipped: int


def label_clips(
    workspace: Workspace,
    labeler: ClipLabeler,
    *,
    emotions: Sequence[str],
    clusters: Sequence[str],
    language_hint: str = "auto",
    force: bool = False,
    limit: int | None = None,
) -> LabelSummary:
    """Label pending clips and atomically persist every successful result."""

    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert isinstance(clips, list)
    assert isinstance(labels, list)
    existing = {row.clip_id for row in labels}
    labelled = 0
    skipped = 0
    attempted = 0
    for clip in clips:
        if not force and clip.clip_id in existing:
            skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        result = labeler.label(
            path=clip.audio_path,
            clip_id=clip.clip_id,
            emotions=emotions,
            clusters=clusters,
            language_hint=language_hint,
        )
        if result.clip_id != clip.clip_id:
            raise ValueError(
                f"labeler returned clip_id {result.clip_id!r}, expected {clip.clip_id!r}"
            )
        workspace.upsert_jsonl(
            workspace.paths.labels_jsonl,
            result,
            key="clip_id",
        )
        existing.add(clip.clip_id)
        labelled += 1
    return LabelSummary(total=len(clips), labelled=labelled, skipped=skipped)
