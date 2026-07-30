"""Materialize a reviewed, immutable training dataset."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from .models import ClipRecord, LabelRecord, ReviewDecision, SourceRecord, StrictModel
from .workspace import Workspace


class TrainingRecord(StrictModel):
    clip_id: str
    source_id: str
    wav_path: Path
    text: str
    language: str
    speaker: str
    emotion: str
    cluster: str
    include: bool = True
    reviewed: bool = True
    duration_seconds: float = Field(gt=0)
    sha256: str
    pipeline_config_sha256: str = ""


@dataclass(frozen=True, slots=True)
class ExportResult:
    root: Path
    manifest: Path
    gpt_sovits_list: Path
    rvc_dataset: Path
    included: int
    excluded: int
    fingerprint: str


_UNSAFE_COMPONENT = re.compile(r"[^\w.-]+", flags=re.UNICODE)
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def _component(value: str, fallback: str) -> str:
    result = _UNSAFE_COMPONENT.sub("_", value.strip()).strip(" ._")
    if not result or result.lower() in _WINDOWS_RESERVED:
        result = fallback
    return result[:64]


def _copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) == _sha256(source):
            return
        raise FileExistsError(f"export target already exists with different content: {destination}")
    shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_fingerprint(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_text(text: str, clip_id: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError(f"{clip_id}: transcript is empty")
    if any(character in normalized for character in ("\r", "\n", "|")):
        raise ValueError(f"{clip_id}: transcript contains newline or '|'")
    return normalized


def _decision_for(
    clip: ClipRecord,
    labels: dict[str, LabelRecord],
    decisions: dict[str, ReviewDecision],
    *,
    allow_unreviewed: bool,
) -> tuple[ReviewDecision, bool]:
    decision = decisions.get(clip.clip_id)
    if decision is not None:
        if not decision.confirmed and not allow_unreviewed:
            raise ValueError(f"clip review is not confirmed: {clip.clip_id}")
        return decision, decision.confirmed
    if not allow_unreviewed:
        raise ValueError(f"clip has not been reviewed: {clip.clip_id}")
    label = labels.get(clip.clip_id)
    return (
        ReviewDecision(
            clip_id=clip.clip_id,
            emotion=label.emotion if label else clip.emotion,
            cluster=label.cluster if label else clip.cluster,
            transcript=label.transcript if label else clip.text,
        ),
        False,
    )


def _prepare_export_plan(
    workspace: Workspace,
    *,
    speaker: str,
    language: str,
    allow_unreviewed: bool,
    config_fingerprint: str,
) -> tuple[list[tuple[ClipRecord, ReviewDecision, bool]], str]:
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    sources = workspace.read_jsonl(workspace.paths.sources_jsonl, SourceRecord)
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    state = workspace.load_review()
    assert isinstance(clips, list)
    assert isinstance(sources, list)
    assert isinstance(labels, list)
    source_ids = {row.source_id for row in sources}
    stale_sources = sorted({row.source_id for row in clips if row.source_id not in source_ids})
    if stale_sources:
        raise ValueError(f"clips refer to inactive sources: {stale_sources[:3]}")
    labels_by_id = {row.clip_id: row for row in labels}

    planned: list[tuple[ClipRecord, ReviewDecision, bool]] = []
    fingerprint_rows: list[dict[str, object]] = []
    for clip in clips:
        decision, reviewed = _decision_for(
            clip,
            labels_by_id,
            state.decisions,
            allow_unreviewed=allow_unreviewed,
        )
        planned.append((clip, decision, reviewed))
        fingerprint_rows.append(
            {
                "clip_id": clip.clip_id,
                "sha256": clip.sha256,
                "text": decision.transcript,
                "emotion": decision.emotion,
                "cluster": decision.cluster,
                "excluded": decision.excluded,
                "reviewed": reviewed,
                "speaker": speaker,
                "language": language,
                "pipeline_config_sha256": config_fingerprint,
            }
        )
    return planned, _canonical_fingerprint(fingerprint_rows)


def current_training_fingerprint(
    workspace: Workspace,
    *,
    speaker: str,
    language: str,
    allow_unreviewed: bool,
    config_fingerprint: str,
) -> str:
    """Return the fingerprint an export of the current workspace would receive."""

    _, fingerprint = _prepare_export_plan(
        workspace,
        speaker=speaker,
        language=language,
        allow_unreviewed=allow_unreviewed,
        config_fingerprint=config_fingerprint,
    )
    return fingerprint


def export_training_dataset(
    workspace: Workspace,
    *,
    output_root: str | Path | None = None,
    speaker: str = "speaker",
    language: str = "zh",
    allow_unreviewed: bool = False,
    config_fingerprint: str = "",
) -> ExportResult:
    """Create a content-addressed export without moving source clips."""

    planned, fingerprint = _prepare_export_plan(
        workspace,
        speaker=speaker,
        language=language,
        allow_unreviewed=allow_unreviewed,
        config_fingerprint=config_fingerprint,
    )
    base = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else workspace.paths.training / "exports"
    )
    root = base / f"dataset-{fingerprint[:12]}"
    manifest_path = root / "manifest.jsonl"
    gpt_list = root / "gpt-sovits" / "dataset.list"
    rvc_dataset = root / "rvc" / "dataset"

    records: list[TrainingRecord] = []
    gpt_lines: list[str] = []
    excluded = 0
    basenames: set[str] = set()
    for clip, decision, reviewed in planned:
        if decision.excluded:
            excluded += 1
            continue
        text = _validate_text(decision.transcript, clip.clip_id)
        emotion = _component(decision.emotion, "unknown")
        cluster = _component(decision.cluster, "unknown")
        basename = f"{clip.clip_id}.wav"
        if basename in basenames:
            raise ValueError(f"duplicate exported WAV basename: {basename}")
        basenames.add(basename)
        explorer_wav = root / "reviewed" / emotion / cluster / basename
        _copy_immutable(clip.audio_path.resolve(), explorer_wav)
        explorer_wav.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        rvc_wav = rvc_dataset / basename
        _copy_immutable(clip.audio_path.resolve(), rvc_wav)
        record = TrainingRecord(
            clip_id=clip.clip_id,
            source_id=clip.source_id,
            wav_path=explorer_wav.resolve(),
            text=text,
            language=language,
            speaker=speaker,
            emotion=decision.emotion,
            cluster=decision.cluster,
            reviewed=reviewed,
            duration_seconds=clip.frames / clip.sample_rate,
            sha256=clip.sha256,
            pipeline_config_sha256=config_fingerprint,
        )
        records.append(record)
        gpt_lines.append(f"{record.wav_path}|{record.speaker}|{record.language}|{record.text}")

    if not records:
        raise ValueError("no included clips to export")
    workspace.write_jsonl(manifest_path, records)
    gpt_list.parent.mkdir(parents=True, exist_ok=True)
    gpt_list.write_text("\n".join(gpt_lines) + "\n", encoding="utf-8")
    metadata = {
        "fingerprint": fingerprint,
        "included": len(records),
        "excluded": excluded,
        "speaker": speaker,
        "language": language,
        "allow_unreviewed": allow_unreviewed,
        "pipeline_config_sha256": config_fingerprint,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ExportResult(
        root=root,
        manifest=manifest_path,
        gpt_sovits_list=gpt_list,
        rvc_dataset=rvc_dataset,
        included=len(records),
        excluded=excluded,
        fingerprint=fingerprint,
    )
