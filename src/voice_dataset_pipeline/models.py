"""Shared, serialisable domain models for the voice dataset pipeline.

The models deliberately contain no orchestration logic.  Keeping the records
strict and JSON-friendly makes the on-disk manifests usable by both the local
energy splitter and optional model-backed implementations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp suitable for persisted records."""

    return datetime.now(UTC)


class StrictModel(BaseModel):
    """Base class for persisted records.

    Unknown fields are rejected so a typo in a TOML file or a stale manifest is
    surfaced immediately instead of silently changing a training run.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MediaKind(StrEnum):
    AUDIO = "audio"
    VIDEO = "video"


class SplitBackend(StrEnum):
    ENERGY = "energy"
    GEMINI = "gemini"


class InputMode(StrEnum):
    AUTO = "auto"
    AUDIO = "audio"
    VIDEO = "video"


class Emotion(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    UNKNOWN = "unknown"


class SourceRecord(StrictModel):
    """One content-deduplicated input after audio normalisation."""

    source_id: str = Field(min_length=64, max_length=64)
    content_sha256: str = Field(min_length=64, max_length=64)
    original_path: Path
    normalized_path: Path
    media_kind: MediaKind
    size_bytes: Annotated[int, Field(ge=0)]
    sample_rate: Annotated[int, Field(gt=0)]
    channels: Annotated[int, Field(gt=0)]
    frames: Annotated[int, Field(ge=0)]
    duration_seconds: Annotated[float, Field(ge=0)]
    ingested_at: datetime = Field(default_factory=utc_now)


class Segment(StrictModel):
    """A proposed speech interval in a normalized source."""

    source_id: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(gt=0)]
    average_dbfs: float | None = None
    backend: SplitBackend = SplitBackend.ENERGY
    text_hint: str = ""
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def end_after_start(self) -> Segment:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class ClipRecord(StrictModel):
    """An immutable WAV clip materialized from a segment."""

    clip_id: str = Field(min_length=64, max_length=64)
    source_id: str
    audio_path: Path
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(gt=0)]
    text: str = ""
    emotion: str = Emotion.UNKNOWN.value
    cluster: str = "unknown"
    status: str = "pending"
    sample_rate: Annotated[int, Field(gt=0)]
    frames: Annotated[int, Field(gt=0)]
    sha256: str = Field(min_length=64, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def end_after_start(self) -> ClipRecord:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self

    @property
    def path(self) -> Path:
        """Compatibility shorthand for code that handles generic assets."""

        return self.audio_path

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1_000

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1_000


class LabelRecord(StrictModel):
    """A model-generated label that remains reviewable."""

    clip_id: str
    transcript: str = ""
    emotion: str = Emotion.UNKNOWN.value
    cluster: str = "unknown"
    confidence: Annotated[float | None, Field(ge=0, le=1)] = None
    rationale: str = ""
    model: str = ""
    labelled_at: datetime = Field(default_factory=utc_now)


class QualityRecord(StrictModel):
    """Reproducible acoustic quality decision for one immutable clip."""

    clip_id: str
    audio_path: Path
    audio_sha256: str = Field(min_length=64, max_length=64)
    profile_sha256: str = Field(min_length=64, max_length=64)
    duration_seconds: Annotated[float, Field(gt=0)]
    rms_dbfs: float
    peak_dbfs: float
    clipping_ratio: Annotated[float, Field(ge=0, le=1)]
    silence_ratio: Annotated[float, Field(ge=0, le=1)]
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=utc_now)


class ASRRecord(StrictModel):
    """Local transcription and speech-emotion result for one clip."""

    clip_id: str
    audio_sha256: str = Field(min_length=64, max_length=64)
    # Empty keeps manifests written before cache profiles were introduced
    # readable.  Such records never match a current profile and are therefore
    # recomputed (or rejected by a strict export gate).
    profile_sha256: str = ""
    transcript: str = ""
    raw_text: str = ""
    language: str = "auto"
    emotion: str = Emotion.UNKNOWN.value
    model: str = ""
    expected_text: str = ""
    transcript_similarity: Annotated[float | None, Field(ge=0, le=1)] = None
    accepted: bool = True
    reasons: list[str] = Field(default_factory=list)
    transcribed_at: datetime = Field(default_factory=utc_now)


class ReviewDecision(StrictModel):
    """The latest human decision for one clip."""

    clip_id: str
    emotion: str = Emotion.UNKNOWN.value
    cluster: str = "unknown"
    transcript: str = ""
    excluded: bool = False
    confirmed: bool = False
    reviewed_at: datetime = Field(default_factory=utc_now)


class ReviewState(StrictModel):
    """Crash-resumable state for the terminal review workflow."""

    version: Annotated[int, Field(ge=1)] = 1
    cursor: Annotated[int, Field(ge=0)] = 0
    order: list[str] = Field(default_factory=list)
    decisions: dict[str, ReviewDecision] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)


class ReviewMergeReceipt(StrictModel):
    """Redo log for one review merge spanning manifests and review state."""

    version: Annotated[int, Field(ge=1)] = 1
    left_clip_id: str
    right_clip_id: str
    merged_clip_id: str
    clips: list[ClipRecord]
    segments: list[Segment]
    labels: list[LabelRecord]
    quality: list[QualityRecord]
    asr: list[ASRRecord]
    review_state: ReviewState
    created_at: datetime = Field(default_factory=utc_now)
