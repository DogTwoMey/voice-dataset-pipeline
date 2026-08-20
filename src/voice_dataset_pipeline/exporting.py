"""Materialize a reviewed, immutable training dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from .asr import ASRProfile, transcript_similarity
from .media import sha256_file
from .models import (
    ASRRecord,
    ClipRecord,
    LabelRecord,
    QualityRecord,
    ReviewDecision,
    SourceRecord,
    StrictModel,
)
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


def _atomic_text(path: Path, value: str) -> None:
    """Durably publish a text file without exposing a partial payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_export_content(root: str | Path, metadata: dict[str, object]) -> None:
    """Fail closed when a materialized training manifest was modified in place."""

    dataset = Path(root).expanduser().resolve()
    manifest = dataset / "manifest.jsonl"
    declared_manifest = metadata.get("manifest_sha256")
    if not isinstance(declared_manifest, str) or len(declared_manifest) != 64:
        raise ValueError("export metadata is missing the manifest content hash; re-export")
    if not manifest.is_file():
        raise ValueError(f"export manifest does not exist: {manifest}")
    if _sha256(manifest) != declared_manifest:
        raise ValueError("export manifest content hash mismatch; re-export")

    dataset_list = dataset / "gpt-sovits" / "dataset.list"
    declared_list = metadata.get("dataset_list_sha256")
    if dataset_list.exists():
        if not dataset_list.is_file():
            raise ValueError(f"export dataset.list is not a file: {dataset_list}")
        if not isinstance(declared_list, str) or len(declared_list) != 64:
            raise ValueError("export metadata is missing the dataset.list content hash; re-export")
        if _sha256(dataset_list) != declared_list:
            raise ValueError("export dataset.list content hash mismatch; re-export")
    elif declared_list not in (None, ""):
        raise ValueError(f"export dataset.list does not exist: {dataset_list}")


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
    quality_enabled: bool,
    quality_profile_sha256: str,
    require_asr: bool,
    asr_profile: ASRProfile | None,
) -> tuple[list[tuple[ClipRecord, ReviewDecision, bool, bool]], str]:
    if quality_enabled and len(quality_profile_sha256) != 64:
        raise ValueError("quality gate requires the current quality profile fingerprint")
    if require_asr and asr_profile is None:
        raise ValueError("strict ASR gate requires the current ASR profile")
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    sources = workspace.read_jsonl(workspace.paths.sources_jsonl, SourceRecord)
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    quality = workspace.read_jsonl(workspace.paths.quality_jsonl, QualityRecord)
    asr = workspace.read_jsonl(workspace.paths.asr_jsonl, ASRRecord)
    state = workspace.load_review()
    assert isinstance(clips, list)
    assert isinstance(sources, list)
    assert isinstance(labels, list)
    assert isinstance(quality, list)
    assert isinstance(asr, list)
    source_ids = {row.source_id for row in sources}
    stale_sources = sorted({row.source_id for row in clips if row.source_id not in source_ids})
    if stale_sources:
        raise ValueError(f"clips refer to inactive sources: {stale_sources[:3]}")
    labels_by_id = {row.clip_id: row for row in labels}
    quality_by_id = {row.clip_id: row for row in quality}
    asr_by_id = {row.clip_id: row for row in asr}

    planned: list[tuple[ClipRecord, ReviewDecision, bool, bool]] = []
    fingerprint_rows: list[dict[str, object]] = []
    for clip in clips:
        decision, reviewed = _decision_for(
            clip,
            labels_by_id,
            state.decisions,
            allow_unreviewed=allow_unreviewed,
        )
        quality_record = quality_by_id.get(clip.clip_id)
        asr_record = asr_by_id.get(clip.clip_id)
        actual_sha256: str | None = None
        quality_state = "disabled"
        asr_state = "disabled"
        gate_accepted = True
        if quality_enabled or require_asr:
            actual_sha256 = sha256_file(clip.audio_path)
        if quality_enabled:
            if quality_record is None:
                quality_state = "missing"
            elif (
                actual_sha256 != clip.sha256
                or quality_record.audio_sha256 != actual_sha256
                or quality_record.profile_sha256 != quality_profile_sha256
            ):
                quality_state = "stale"
            elif not quality_record.accepted:
                quality_state = "rejected"
            else:
                quality_state = "accepted"
            gate_accepted = quality_state == "accepted"
        expected_asr_profile = (
            asr_profile.fingerprint(asr_record.expected_text)
            if require_asr and asr_profile and asr_record
            else ""
        )
        reviewed_asr_similarity: float | None = None
        if require_asr:
            assert asr_profile is not None
            if asr_record is None:
                asr_state = "missing"
            elif (
                actual_sha256 != clip.sha256
                or asr_record.audio_sha256 != actual_sha256
                or asr_record.profile_sha256 != expected_asr_profile
            ):
                asr_state = "stale"
            else:
                reviewed_asr_similarity = transcript_similarity(
                    decision.transcript, asr_record.transcript
                )
            if asr_state == "disabled" and (
                reviewed_asr_similarity is None
                or reviewed_asr_similarity < asr_profile.minimum_similarity
            ):
                asr_state = "rejected"
            elif asr_state == "disabled":
                asr_state = "accepted"
            gate_accepted = gate_accepted and asr_state == "accepted"
        planned.append((clip, decision, reviewed, gate_accepted))
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
                "gate": {
                    "accepted": gate_accepted,
                    "actual_audio_sha256": actual_sha256,
                    "quality_enabled": quality_enabled,
                    "quality_profile_sha256": (quality_profile_sha256 if quality_enabled else ""),
                    "quality_state": quality_state,
                    "require_asr": require_asr,
                    "asr_profile_sha256": expected_asr_profile,
                    "asr_state": asr_state,
                    "reviewed_asr_similarity": reviewed_asr_similarity,
                },
                "quality": (
                    {
                        "accepted": quality_record.accepted,
                        "profile_sha256": quality_record.profile_sha256,
                        "reasons": quality_record.reasons,
                    }
                    if quality_enabled and quality_record
                    else None
                ),
                "asr": (
                    {
                        "accepted": asr_record.accepted,
                        "audio_sha256": asr_record.audio_sha256,
                        "transcript": asr_record.transcript,
                        "transcription_similarity": asr_record.transcript_similarity,
                        "reviewed_similarity": reviewed_asr_similarity,
                        "reasons": asr_record.reasons,
                    }
                    if require_asr and asr_record
                    else None
                ),
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
    quality_enabled: bool = False,
    quality_profile_sha256: str = "",
    require_asr: bool = False,
    asr_profile: ASRProfile | None = None,
) -> str:
    """Return the fingerprint an export of the current workspace would receive."""

    _, fingerprint = _prepare_export_plan(
        workspace,
        speaker=speaker,
        language=language,
        allow_unreviewed=allow_unreviewed,
        config_fingerprint=config_fingerprint,
        quality_enabled=quality_enabled,
        quality_profile_sha256=quality_profile_sha256,
        require_asr=require_asr,
        asr_profile=asr_profile,
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
    quality_enabled: bool = False,
    quality_profile_sha256: str = "",
    require_asr: bool = False,
    asr_profile: ASRProfile | None = None,
) -> ExportResult:
    """Create a content-addressed export without moving source clips."""

    planned, fingerprint = _prepare_export_plan(
        workspace,
        speaker=speaker,
        language=language,
        allow_unreviewed=allow_unreviewed,
        config_fingerprint=config_fingerprint,
        quality_enabled=quality_enabled,
        quality_profile_sha256=quality_profile_sha256,
        require_asr=require_asr,
        asr_profile=asr_profile,
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
    for clip, decision, reviewed, gate_accepted in planned:
        if decision.excluded or not gate_accepted:
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
    _atomic_text(gpt_list, "\n".join(gpt_lines) + "\n")
    metadata = {
        "format_version": 2,
        "fingerprint": fingerprint,
        "included": len(records),
        "excluded": excluded,
        "speaker": speaker,
        "language": language,
        "allow_unreviewed": allow_unreviewed,
        "pipeline_config_sha256": config_fingerprint,
        "manifest_sha256": _sha256(manifest_path),
        "dataset_list_sha256": _sha256(gpt_list),
    }
    _atomic_text(
        root / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
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
