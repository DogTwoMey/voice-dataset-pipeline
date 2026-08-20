"""Streaming acoustic metrics and reproducible clip quality decisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .media import sha256_file
from .models import ClipRecord, QualityRecord
from .workspace import Workspace


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_duration_seconds: float = 0.8
    max_duration_seconds: float = 15.0
    min_rms_dbfs: float = -55.0
    max_rms_dbfs: float = -6.0
    min_peak_dbfs: float = -45.0
    clipping_amplitude: float = 0.999
    max_clipping_ratio: float = 0.01
    silence_threshold_dbfs: float = -50.0
    max_silence_ratio: float = 0.65

    def __post_init__(self) -> None:
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_duration_seconds < self.min_duration_seconds:
            raise ValueError("max_duration_seconds must not be shorter than the minimum")
        if self.max_rms_dbfs <= self.min_rms_dbfs:
            raise ValueError("max_rms_dbfs must exceed min_rms_dbfs")
        if not 0 < self.clipping_amplitude <= 1:
            raise ValueError("clipping_amplitude must be in (0, 1]")
        if not 0 <= self.max_clipping_ratio <= 1:
            raise ValueError("max_clipping_ratio must be in [0, 1]")
        if not 0 <= self.max_silence_ratio <= 1:
            raise ValueError("max_silence_ratio must be in [0, 1]")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AudioMetrics:
    duration_seconds: float
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: float
    silence_ratio: float


@dataclass(frozen=True, slots=True)
class QualitySummary:
    total: int
    evaluated: int
    reused: int
    accepted: int
    rejected: int


def _dbfs(amplitude: float) -> float:
    return round(20.0 * math.log10(max(amplitude, 1e-6)), 3)


def measure_audio(
    path: str | Path,
    *,
    silence_threshold_dbfs: float = -50.0,
    clipping_amplitude: float = 0.999,
    block_frames: int = 65_536,
) -> AudioMetrics:
    """Measure a decodable audio file without loading it wholly into memory."""

    source = Path(path).expanduser().resolve()
    if block_frames <= 0:
        raise ValueError("block_frames must be positive")
    silence_amplitude = 10.0 ** (silence_threshold_dbfs / 20.0)
    sample_count = 0
    sum_squares = 0.0
    peak = 0.0
    clipped = 0
    silent = 0
    with sf.SoundFile(source) as audio:
        if audio.samplerate <= 0 or len(audio) <= 0:
            raise ValueError(f"audio is empty or has an invalid sample rate: {source}")
        duration = len(audio) / audio.samplerate
        while True:
            block = audio.read(block_frames, dtype="float32", always_2d=True)
            if not len(block):
                break
            absolute = np.abs(block.astype(np.float64, copy=False))
            sample_count += absolute.size
            sum_squares += float(np.square(absolute).sum(dtype=np.float64))
            peak = max(peak, float(absolute.max(initial=0.0)))
            clipped += int(np.count_nonzero(absolute >= clipping_amplitude))
            silent += int(np.count_nonzero(absolute < silence_amplitude))
    if sample_count <= 0:
        raise ValueError(f"audio contains no samples: {source}")
    rms = math.sqrt(sum_squares / sample_count)
    return AudioMetrics(
        duration_seconds=round(duration, 6),
        rms_dbfs=_dbfs(rms),
        peak_dbfs=_dbfs(peak),
        clipping_ratio=round(clipped / sample_count, 8),
        silence_ratio=round(silent / sample_count, 8),
    )


def evaluate_audio(
    path: str | Path,
    *,
    clip_id: str,
    thresholds: QualityThresholds | None = None,
    audio_sha256: str | None = None,
) -> QualityRecord:
    """Apply one threshold profile and return an inspectable gate decision."""

    profile = thresholds or QualityThresholds()
    source = Path(path).expanduser().resolve()
    metrics = measure_audio(
        source,
        silence_threshold_dbfs=profile.silence_threshold_dbfs,
        clipping_amplitude=profile.clipping_amplitude,
    )
    reasons: list[str] = []
    if metrics.duration_seconds < profile.min_duration_seconds:
        reasons.append("duration_too_short")
    if metrics.duration_seconds > profile.max_duration_seconds:
        reasons.append("duration_too_long")
    if metrics.rms_dbfs < profile.min_rms_dbfs:
        reasons.append("rms_too_low")
    if metrics.rms_dbfs > profile.max_rms_dbfs:
        reasons.append("rms_too_high")
    if metrics.peak_dbfs < profile.min_peak_dbfs:
        reasons.append("peak_too_low")
    if metrics.clipping_ratio > profile.max_clipping_ratio:
        reasons.append("clipping_ratio_too_high")
    if metrics.silence_ratio > profile.max_silence_ratio:
        reasons.append("silence_ratio_too_high")
    return QualityRecord(
        clip_id=clip_id,
        audio_path=source,
        audio_sha256=audio_sha256 or sha256_file(source),
        profile_sha256=profile.fingerprint,
        duration_seconds=metrics.duration_seconds,
        rms_dbfs=metrics.rms_dbfs,
        peak_dbfs=metrics.peak_dbfs,
        clipping_ratio=metrics.clipping_ratio,
        silence_ratio=metrics.silence_ratio,
        accepted=not reasons,
        reasons=reasons,
    )


def evaluate_workspace(
    workspace: Workspace,
    *,
    thresholds: QualityThresholds | None = None,
    force: bool = False,
) -> QualitySummary:
    """Evaluate immutable clips, atomically persisting every completed decision."""

    profile = thresholds or QualityThresholds()
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    records = workspace.read_jsonl(workspace.paths.quality_jsonl, QualityRecord)
    assert isinstance(clips, list)
    assert isinstance(records, list)
    existing = {record.clip_id: record for record in records}
    evaluated = 0
    reused = 0
    final: dict[str, QualityRecord] = {}
    for clip in clips:
        actual_sha256 = sha256_file(clip.audio_path)
        prior = existing.get(clip.clip_id)
        if (
            not force
            and prior is not None
            and prior.audio_sha256 == actual_sha256
            and prior.profile_sha256 == profile.fingerprint
        ):
            final[clip.clip_id] = prior
            reused += 1
            continue
        record = evaluate_audio(
            clip.audio_path,
            clip_id=clip.clip_id,
            thresholds=profile,
            audio_sha256=actual_sha256,
        )
        workspace.upsert_jsonl(workspace.paths.quality_jsonl, record, key="clip_id")
        final[clip.clip_id] = record
        evaluated += 1
    # A completed pass also drops decisions for clips no longer in the active
    # manifest; immutable WAVs may remain on disk, but stale gates must not.
    workspace.write_jsonl(
        workspace.paths.quality_jsonl,
        [final[clip.clip_id] for clip in clips],
    )
    accepted = sum(record.accepted for record in final.values())
    return QualitySummary(
        total=len(clips),
        evaluated=evaluated,
        reused=reused,
        accepted=accepted,
        rejected=len(final) - accepted,
    )
