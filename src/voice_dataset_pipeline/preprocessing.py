"""Subtitle-first segmentation with explicit, inspectable fallback provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .models import MediaKind, Segment, SourceRecord
from .splitting import SpeechSplitter
from .subtitles import (
    FFmpegSubtitleExtractor,
    find_sidecar,
    parse_subtitle,
    segments_from_subtitles,
)


class SegmentationStrategy(StrEnum):
    SIDECAR = "sidecar_subtitle"
    EMBEDDED = "embedded_subtitle"
    VISION = "vision_model"
    SILENCE = "silence_splitter"


class EmbeddedSubtitleExtractor(Protocol):
    def extract(self, media_path: str | Path, output_dir: str | Path) -> Path | None: ...


class SourceBoundaryProvider(Protocol):
    def split(self, source: SourceRecord) -> Sequence[Segment]: ...


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    strategy: SegmentationStrategy
    segments: tuple[Segment, ...]
    attempts: tuple[SegmentationStrategy, ...]
    failures: dict[str, str] = field(default_factory=dict)
    subtitle_path: Path | None = None


SidecarLocator = Callable[[str | Path], Path | None]


def _annotate_segments(
    segments: Sequence[Segment],
    *,
    source_id: str,
    strategy: SegmentationStrategy,
) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        if segment.source_id != source_id:
            raise ValueError(
                "fallback splitter returned source_id "
                f"{segment.source_id!r}, expected {source_id!r}"
            )
        result.append(
            segment.model_copy(
                update={
                    "provenance": {
                        **segment.provenance,
                        "strategy": strategy.value,
                    }
                }
            )
        )
    return result


class PreprocessingPipeline:
    """Try deterministic text-bearing boundaries before semantic and VAD fallbacks.

    The public interface returns proposed segments only.  It never materializes
    clips, mutates a workspace, calls ASR, or silently converts subtitle text
    into a confirmed transcript.
    """

    def __init__(
        self,
        *,
        silence_splitter: SpeechSplitter,
        embedded_extractor: EmbeddedSubtitleExtractor | None = None,
        vision_splitter: SourceBoundaryProvider | None = None,
        sidecar_locator: SidecarLocator = find_sidecar,
    ) -> None:
        self.silence_splitter = silence_splitter
        self.embedded_extractor = embedded_extractor
        self.vision_splitter = vision_splitter
        self.sidecar_locator = sidecar_locator

    def split(self, source: SourceRecord, *, scratch_dir: str | Path) -> SegmentationResult:
        attempts: list[SegmentationStrategy] = []
        failures: dict[str, str] = {}

        attempts.append(SegmentationStrategy.SIDECAR)
        sidecar = self.sidecar_locator(source.original_path)
        if sidecar is not None:
            try:
                segments = segments_from_subtitles(
                    parse_subtitle(sidecar),
                    source_id=source.source_id,
                    subtitle_path=sidecar,
                    strategy=SegmentationStrategy.SIDECAR.value,
                    duration_seconds=source.duration_seconds,
                )
                if segments:
                    return SegmentationResult(
                        strategy=SegmentationStrategy.SIDECAR,
                        segments=tuple(segments),
                        attempts=tuple(attempts),
                        failures=failures,
                        subtitle_path=sidecar.resolve(),
                    )
                failures[SegmentationStrategy.SIDECAR.value] = "subtitle contained no usable cues"
            except (OSError, ValueError) as exc:
                failures[SegmentationStrategy.SIDECAR.value] = str(exc)

        if source.media_kind is MediaKind.VIDEO and self.embedded_extractor is not None:
            attempts.append(SegmentationStrategy.EMBEDDED)
            try:
                embedded = self.embedded_extractor.extract(source.original_path, scratch_dir)
                if embedded is not None:
                    segments = segments_from_subtitles(
                        parse_subtitle(embedded),
                        source_id=source.source_id,
                        subtitle_path=embedded,
                        strategy=SegmentationStrategy.EMBEDDED.value,
                        duration_seconds=source.duration_seconds,
                    )
                    if segments:
                        return SegmentationResult(
                            strategy=SegmentationStrategy.EMBEDDED,
                            segments=tuple(segments),
                            attempts=tuple(attempts),
                            failures=failures,
                            subtitle_path=embedded.resolve(),
                        )
                failures[SegmentationStrategy.EMBEDDED.value] = (
                    "no usable embedded text subtitle stream"
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures[SegmentationStrategy.EMBEDDED.value] = str(exc)

        if source.media_kind is MediaKind.VIDEO and self.vision_splitter is not None:
            attempts.append(SegmentationStrategy.VISION)
            try:
                segments = _annotate_segments(
                    self.vision_splitter.split(source),
                    source_id=source.source_id,
                    strategy=SegmentationStrategy.VISION,
                )
                if segments:
                    return SegmentationResult(
                        strategy=SegmentationStrategy.VISION,
                        segments=tuple(segments),
                        attempts=tuple(attempts),
                        failures=failures,
                    )
                failures[SegmentationStrategy.VISION.value] = "vision model returned no boundaries"
            except Exception as exc:  # external model adapters define their own error types
                failures[SegmentationStrategy.VISION.value] = str(exc)

        attempts.append(SegmentationStrategy.SILENCE)
        segments = _annotate_segments(
            self.silence_splitter.split(
                source.normalized_path,
                source_id=source.source_id,
            ),
            source_id=source.source_id,
            strategy=SegmentationStrategy.SILENCE,
        )
        return SegmentationResult(
            strategy=SegmentationStrategy.SILENCE,
            segments=tuple(segments),
            attempts=tuple(attempts),
            failures=failures,
        )


def default_subtitle_extractor(
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
) -> FFmpegSubtitleExtractor:
    """Construct the production embedded-subtitle adapter without hidden work."""

    return FFmpegSubtitleExtractor(
        ffmpeg_binary=ffmpeg_binary,
        ffprobe_binary=ffprobe_binary,
    )
