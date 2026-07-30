"""Speech-only segmentation with a replaceable splitter protocol."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf

from .config import SplittingConfig
from .media import sha256_file
from .models import ClipRecord, Segment, SplitBackend


@runtime_checkable
class SpeechSplitter(Protocol):
    """The seam shared by local and future multimodal splitters."""

    backend: SplitBackend

    def split(self, audio_path: str | Path, *, source_id: str) -> list[Segment]: ...


# A shorter alias is convenient for third-party implementations.
Splitter = SpeechSplitter


@dataclass(frozen=True, slots=True)
class _EnergyFrame:
    start: float
    end: float
    dbfs: float

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True, slots=True)
class _Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class EnergySplitter:
    """Streaming RMS VAD with conservative, local boundary decisions."""

    backend = SplitBackend.ENERGY

    def __init__(self, config: SplittingConfig) -> None:
        self.config = config

    def split(self, audio_path: str | Path, *, source_id: str) -> list[Segment]:
        path = Path(audio_path)
        frames, duration = self._energy_frames(path)
        if not frames or duration <= 0:
            return []

        levels = np.asarray([frame.dbfs for frame in frames], dtype=np.float64)
        adaptive = float(
            np.percentile(levels, self.config.adaptive_percentile) + self.config.threshold_offset_db
        )
        threshold = float(
            np.clip(
                adaptive,
                self.config.threshold_floor_db,
                self.config.threshold_ceiling_db,
            )
        )

        speech = [frame for frame in frames if frame.dbfs >= threshold]
        cores = self._speech_runs(speech)
        cores = self._merge_short_neighbours(cores)

        padding = self.config.padding_ms / 1_000
        core_maximum = self.config.max_segment_seconds - 2 * padding
        split_cores: list[_Interval] = []
        for interval in cores:
            split_cores.extend(self._split_long(interval, frames, core_maximum))

        padded = [
            _Interval(
                max(0.0, interval.start - padding),
                min(duration, interval.end + padding),
            )
            for interval in split_cores
        ]
        contracted = self._contract_padding_overlaps(padded, split_cores)

        return [
            Segment(
                source_id=source_id,
                start_seconds=round(interval.start, 6),
                end_seconds=round(interval.end, 6),
                average_dbfs=self._average_dbfs(frames, interval),
                backend=self.backend,
            )
            for interval in contracted
            if interval.end > interval.start
        ]

    def materialize(
        self,
        audio_path: str | Path,
        segments: Sequence[Segment],
        output_dir: str | Path,
        *,
        source_id: str | None = None,
    ) -> list[ClipRecord]:
        return materialize_clips(audio_path, segments, output_dir, source_id=source_id)

    def _energy_frames(self, path: Path) -> tuple[list[_EnergyFrame], float]:
        result: list[_EnergyFrame] = []
        with sf.SoundFile(path) as audio:
            sample_rate = audio.samplerate
            total_frames = len(audio)
            frame_length = max(1, round(sample_rate * self.config.frame_ms / 1_000))
            hop_length = max(1, round(sample_rate * self.config.hop_ms / 1_000))
            buffer = np.empty(0, dtype=np.float32)
            buffer_start = 0

            while True:
                block = audio.read(
                    self.config.stream_block_frames,
                    dtype="float32",
                    always_2d=True,
                )
                if not len(block):
                    break
                mono = np.mean(block, axis=1, dtype=np.float32)
                buffer = np.concatenate((buffer, mono))
                while len(buffer) >= frame_length:
                    values = buffer[:frame_length]
                    result.append(self._make_frame(values, buffer_start, frame_length, sample_rate))
                    buffer = buffer[hop_length:]
                    buffer_start += hop_length

            # Include one final partial window so speech at EOF is not dropped.
            if len(buffer) and (not result or buffer_start > round(result[-1].start * sample_rate)):
                result.append(self._make_frame(buffer, buffer_start, len(buffer), sample_rate))
        return result, total_frames / sample_rate

    @staticmethod
    def _make_frame(
        values: np.ndarray, start_sample: int, length: int, sample_rate: int
    ) -> _EnergyFrame:
        rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
        dbfs = 20 * math.log10(max(rms, 1e-6))
        return _EnergyFrame(
            start=start_sample / sample_rate,
            end=(start_sample + length) / sample_rate,
            dbfs=dbfs,
        )

    def _speech_runs(self, speech: Sequence[_EnergyFrame]) -> list[_Interval]:
        if not speech:
            return []
        maximum_gap = self.config.min_silence_ms / 1_000
        minimum_speech = self.config.min_speech_ms / 1_000
        runs: list[_Interval] = []
        start = speech[0].start
        end = speech[0].end
        for frame in speech[1:]:
            if frame.start - end <= maximum_gap:
                end = max(end, frame.end)
            else:
                if end - start >= minimum_speech:
                    runs.append(_Interval(start, end))
                start, end = frame.start, frame.end
        if end - start >= minimum_speech:
            runs.append(_Interval(start, end))
        return runs

    def _merge_short_neighbours(self, intervals: Sequence[_Interval]) -> list[_Interval]:
        """Merge a nearby pair only when at least one member is short."""

        if not intervals:
            return []
        maximum_gap = self.config.merge_gap_ms / 1_000
        minimum = self.config.min_segment_seconds
        merged: list[_Interval] = [intervals[0]]
        for right in intervals[1:]:
            left = merged[-1]
            gap = right.start - left.end
            if gap <= maximum_gap and (left.duration < minimum or right.duration < minimum):
                merged[-1] = _Interval(left.start, right.end)
            else:
                merged.append(right)
        return merged

    def _split_long(
        self,
        interval: _Interval,
        frames: Sequence[_EnergyFrame],
        maximum: float,
    ) -> list[_Interval]:
        if interval.duration <= maximum:
            return [interval]

        minimum = min(self.config.min_segment_seconds, maximum / 2)
        search = self.config.boundary_search_ms / 1_000
        result: list[_Interval] = []
        cursor = interval.start
        while interval.end - cursor > maximum:
            target = cursor + maximum
            lower = max(cursor + minimum, target - search)
            upper = min(target, interval.end - minimum)
            candidates = [frame for frame in frames if lower <= frame.center <= upper]
            boundary = min(candidates, key=lambda frame: frame.dbfs).center if candidates else upper
            if boundary <= cursor:
                boundary = min(target, interval.end)
            result.append(_Interval(cursor, boundary))
            cursor = boundary
        if interval.end > cursor:
            result.append(_Interval(cursor, interval.end))
        return result

    @staticmethod
    def _contract_padding_overlaps(
        padded: Sequence[_Interval], cores: Sequence[_Interval]
    ) -> list[_Interval]:
        if not padded:
            return []
        starts = [interval.start for interval in padded]
        ends = [interval.end for interval in padded]
        for index in range(len(padded) - 1):
            if ends[index] <= starts[index + 1]:
                continue
            # Both padding regions yield to the midpoint of the original gap.
            boundary = (cores[index].end + cores[index + 1].start) / 2
            ends[index] = boundary
            starts[index + 1] = boundary
        return [
            _Interval(start, end) for start, end in zip(starts, ends, strict=True) if end > start
        ]

    @staticmethod
    def _average_dbfs(frames: Sequence[_EnergyFrame], interval: _Interval) -> float | None:
        values = [frame.dbfs for frame in frames if interval.start <= frame.center <= interval.end]
        if not values:
            return None
        # Average in the power domain, then convert back to dBFS.
        power = float(np.mean(np.power(10.0, np.asarray(values) / 10.0)))
        return round(10 * math.log10(max(power, 1e-12)), 3)


def materialize_clips(
    audio_path: str | Path,
    segments: Sequence[Segment],
    output_dir: str | Path,
    *,
    source_id: str | None = None,
    subtype: str = "PCM_16",
) -> list[ClipRecord]:
    """Write deterministic clips once; existing clip IDs are never overwritten."""

    source = Path(audio_path)
    source_digest = source_id or sha256_file(source)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    records: list[ClipRecord] = []

    with sf.SoundFile(source) as audio:
        sample_rate = audio.samplerate
        total_frames = len(audio)
        for segment in segments:
            if segment.source_id != source_digest:
                raise ValueError(
                    f"segment source_id {segment.source_id!r} does not match {source_digest!r}"
                )
            start = max(0, round(segment.start_seconds * sample_rate))
            end = min(total_frames, round(segment.end_seconds * sample_rate))
            if end <= start:
                raise ValueError(
                    f"segment rounds to an empty clip: {segment.start_seconds}.."
                    f"{segment.end_seconds}"
                )
            identity = f"{source_digest}:{sample_rate}:{start}:{end}:{subtype}".encode()
            clip_id = hashlib.sha256(identity).hexdigest()
            destination = destination_dir / f"{clip_id}.wav"

            if not destination.exists():
                audio.seek(start)
                values = audio.read(end - start, dtype="float32", always_2d=True)
                temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.wav")
                try:
                    sf.write(
                        temporary,
                        values,
                        sample_rate,
                        format="WAV",
                        subtype=subtype,
                    )
                    # Publishing via a hard link is atomic and never replaces an
                    # immutable clip another worker may have completed first.
                    with suppress(FileExistsError):
                        os.link(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

            info = sf.info(destination)
            if info.samplerate != sample_rate or info.frames != end - start:
                raise ValueError(f"immutable clip conflicts with requested segment: {destination}")
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    source_id=source_digest,
                    audio_path=destination,
                    start_ms=round(start * 1_000 / sample_rate),
                    end_ms=round(end * 1_000 / sample_rate),
                    sample_rate=info.samplerate,
                    frames=info.frames,
                    sha256=sha256_file(destination),
                )
            )
    return records
