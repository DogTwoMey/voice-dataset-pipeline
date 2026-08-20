"""Reliable local preparation for Gemini segmentation of long videos.

The remote adapter intentionally knows nothing about workspaces or FFmpeg.
This module adds that orchestration at a separate seam: it plans contiguous
windows around locally detected silence, transcodes content-addressed preview
files, invokes ``GeminiInteractions.split`` once per window, and maps the
returned boundaries back onto the source timeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import GeminiChunkingConfig
from .errors import ConfigurationError, ExternalToolError
from .models import Segment, SourceRecord


class GeminiSplitClient(Protocol):
    def split(
        self,
        *,
        path: Path,
        modality: str,
        source_id: str,
        duration_seconds: float,
        min_segment_seconds: float,
        max_segment_seconds: float,
    ) -> list[Segment]: ...


CommandRunner = Callable[[Sequence[str]], Any]


@dataclass(frozen=True, slots=True)
class ChunkWindow:
    index: int
    start_ms: int
    end_ms: int
    path: Path

    @property
    def duration_seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1_000


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    window: ChunkWindow
    reused: bool


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _result_text(result: Any) -> str:
    return f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"


def _require_success(result: Any, operation: str) -> None:
    returncode = int(getattr(result, "returncode", 1))
    if returncode:
        detail = " ".join(_result_text(result).split())[-1_000:]
        raise ExternalToolError(f"{operation}失败 (exit {returncode}): {detail}")


def parse_silence_midpoints(output: str) -> list[int]:
    """Parse completed FFmpeg silencedetect intervals into midpoint milliseconds."""

    starts: list[float] = []
    result: list[int] = []
    for line in output.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            starts.append(float(start_match.group(1)))
        end_match = _SILENCE_END_RE.search(line)
        if end_match and starts:
            start = starts.pop(0)
            end = float(end_match.group(1))
            if end > start:
                result.append(round((start + end) * 500))
    return sorted(set(result))


def plan_chunk_boundaries(
    duration_ms: int,
    silence_midpoints_ms: Sequence[int],
    *,
    target_ms: int,
    maximum_ms: int,
    search_ms: int,
) -> list[tuple[int, int]]:
    """Plan exact source coverage, preferring a nearby silence for each cut."""

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if target_ms <= 0 or maximum_ms <= 0 or target_ms > maximum_ms:
        raise ValueError("chunk target must be positive and not exceed maximum")

    silences = sorted({point for point in silence_midpoints_ms if 0 < point < duration_ms})
    boundaries = [0]
    start = 0
    while duration_ms - start > maximum_ms:
        desired = start + target_ms
        hard_end = min(duration_ms, start + maximum_ms)
        # Avoid pathological tiny windows even if boundary_search is configured
        # wider than the target duration.
        lower = max(start + max(1, target_ms // 2), desired - search_ms)
        upper = min(hard_end, desired + search_ms)
        candidates = [point for point in silences if lower <= point <= upper]
        end = (
            min(candidates, key=lambda point: (abs(point - desired), point))
            if candidates
            else desired
        )
        end = min(max(end, start + 1), hard_end)
        boundaries.append(end)
        start = end
    boundaries.append(duration_ms)

    windows = list(zip(boundaries, boundaries[1:], strict=False))
    _validate_chunk_coverage(windows, duration_ms=duration_ms, maximum_ms=maximum_ms)
    return windows


def _validate_chunk_coverage(
    windows: Sequence[tuple[int, int]],
    *,
    duration_ms: int,
    maximum_ms: int,
) -> None:
    if not windows or windows[0][0] != 0 or windows[-1][1] != duration_ms:
        raise ExternalToolError("Gemini 本地分块未完整覆盖源媒体")
    previous_end = 0
    for start_ms, end_ms in windows:
        if start_ms != previous_end or end_ms <= start_ms:
            raise ExternalToolError("Gemini 本地分块存在空洞、重叠或乱序")
        if end_ms - start_ms > maximum_ms:
            raise ExternalToolError("Gemini 本地分块超过配置的最大时长")
        previous_end = end_ms


class ChunkedGeminiSplitter:
    """Use the original request for short media and chunk only long video."""

    def __init__(
        self,
        client: GeminiSplitClient,
        *,
        config: GeminiChunkingConfig,
        ffmpeg_binary: str,
        scratch_dir: str | Path,
        min_segment_seconds: float,
        max_segment_seconds: float,
        runner: CommandRunner = _default_runner,
    ) -> None:
        self.client = client
        self.config = config
        self.ffmpeg_binary = ffmpeg_binary
        self.scratch_dir = Path(scratch_dir).expanduser().resolve()
        self.min_segment_seconds = min_segment_seconds
        self.max_segment_seconds = max_segment_seconds
        self.runner = runner

    def split(self, source: SourceRecord, *, modality: str = "video") -> list[Segment]:
        path = source.original_path if modality == "video" else source.normalized_path
        if (
            modality != "video"
            or not self.config.enabled
            or source.duration_seconds <= self.config.threshold_seconds
        ):
            return self._split_one(
                path=path,
                modality=modality,
                source_id=source.source_id,
                duration_seconds=source.duration_seconds,
            )

        prepared, profile = self._prepare_chunks(source)
        retained = self.config.keep_chunks
        remapped: list[Segment] = []
        try:
            for item in prepared:
                window = item.window
                local = (
                    self._read_boundary_cache(
                        window,
                        profile=profile,
                        source_id=source.source_id,
                    )
                    if self.config.reuse_chunks
                    else None
                )
                boundary_reused = local is not None
                if local is None:
                    local = self._split_one(
                        path=window.path,
                        modality="video",
                        source_id=source.source_id,
                        duration_seconds=window.duration_seconds,
                    )
                mapped = self._remap(
                    local,
                    window=window,
                    source=source,
                    profile=profile,
                    reused=item.reused,
                    boundary_reused=boundary_reused,
                    retained=retained,
                )
                if not boundary_reused:
                    self._write_boundary_cache(
                        window,
                        profile=profile,
                        segments=local,
                    )
                remapped.extend(mapped)
            self._validate_remapped(remapped, source)
            return remapped
        finally:
            if not retained:
                for item in prepared:
                    item.window.path.unlink(missing_ok=True)

    def _split_one(
        self,
        *,
        path: Path,
        modality: str,
        source_id: str,
        duration_seconds: float,
    ) -> list[Segment]:
        return self.client.split(
            path=path,
            modality=modality,
            source_id=source_id,
            duration_seconds=duration_seconds,
            min_segment_seconds=self.min_segment_seconds,
            max_segment_seconds=self.max_segment_seconds,
        )

    def _profile(self, source: SourceRecord) -> str:
        chunking = self.config.model_dump(mode="json")
        # These flags control cache policy rather than chunk bytes or Gemini
        # semantics.  Excluding them lets a later retry opt into reuse.
        chunking.pop("reuse_chunks", None)
        chunking.pop("keep_chunks", None)
        payload = {
            "version": 1,
            "source_id": source.source_id,
            "content_sha256": source.content_sha256,
            "duration_ms": round(source.duration_seconds * 1_000),
            "ffmpeg_binary": self.ffmpeg_binary,
            "chunking": chunking,
            "gemini_client": {
                "type": type(self.client).__qualname__,
                "model": str(getattr(self.client, "model", "")),
            },
            "minimum_segment_seconds": self.min_segment_seconds,
            "maximum_segment_seconds": self.max_segment_seconds,
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _prepare_chunks(self, source: SourceRecord) -> tuple[list[PreparedChunk], str]:
        duration_ms = round(source.duration_seconds * 1_000)
        maximum_ms = round(self.config.max_seconds * 1_000)
        profile = self._profile(source)
        cache_dir = self.scratch_dir / source.source_id / profile[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"

        windows = self._read_manifest(
            manifest_path,
            profile=profile,
            duration_ms=duration_ms,
            maximum_ms=maximum_ms,
        )
        if windows is None:
            silences = self._detect_silences(source.normalized_path)
            planned = plan_chunk_boundaries(
                duration_ms,
                silences,
                target_ms=round(self.config.target_seconds * 1_000),
                maximum_ms=maximum_ms,
                search_ms=round(self.config.boundary_search_seconds * 1_000),
            )
            windows = [
                ChunkWindow(
                    index=index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    path=cache_dir / f"chunk_{index:04d}_{start_ms:09d}_{end_ms:09d}.mp4",
                )
                for index, (start_ms, end_ms) in enumerate(planned)
            ]
            self._write_manifest(manifest_path, profile=profile, windows=windows)

        prepared: list[PreparedChunk] = []
        for window in windows:
            reused = self.config.reuse_chunks and window.path.is_file()
            if not reused:
                self._render_chunk(source.original_path, window)
            prepared.append(PreparedChunk(window=window, reused=reused))
        return prepared, profile

    def _detect_silences(self, normalized_audio: Path) -> list[int]:
        if not normalized_audio.is_file():
            raise ConfigurationError(f"用于分块的规范化音频不存在: {normalized_audio}")
        result = self.runner(
            [
                self.ffmpeg_binary,
                "-hide_banner",
                "-nostats",
                "-i",
                str(normalized_audio),
                "-af",
                (
                    f"silencedetect=noise={self.config.silence_noise_db:g}dB:"
                    f"d={self.config.min_silence_seconds:g}"
                ),
                "-f",
                "null",
                "-",
            ]
        )
        _require_success(result, "FFmpeg 静音检测")
        return parse_silence_midpoints(_result_text(result))

    def _render_chunk(self, source_video: Path, window: ChunkWindow) -> None:
        if not source_video.is_file():
            raise ConfigurationError(f"源视频不存在: {source_video}")
        temporary = window.path.with_name(
            f".{window.path.stem}.{uuid.uuid4().hex}.tmp{window.path.suffix}"
        )
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{window.start_ms / 1_000:.3f}",
            "-t",
            f"{window.duration_seconds:.3f}",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale=-2:{self.config.video_height}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            f"{self.config.video_bitrate_kbps}k",
            "-maxrate",
            f"{self.config.video_bitrate_kbps}k",
            "-bufsize",
            f"{self.config.video_bitrate_kbps * 2}k",
            "-c:a",
            "aac",
            "-b:a",
            f"{self.config.audio_bitrate_kbps}k",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            result = self.runner(command)
            _require_success(result, "FFmpeg Gemini 视频分块")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ExternalToolError("FFmpeg Gemini 视频分块未生成输出文件")
            os.replace(temporary, window.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_manifest(path: Path, *, profile: str, windows: Sequence[ChunkWindow]) -> None:
        payload = {
            "version": 1,
            "profile_sha256": profile,
            "chunks": [
                {
                    "index": window.index,
                    "start_ms": window.start_ms,
                    "end_ms": window.end_ms,
                    "file": window.path.name,
                }
                for window in windows
            ],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _boundary_cache_path(window: ChunkWindow) -> Path:
        return window.path.with_suffix(".segments.json")

    @classmethod
    def _write_boundary_cache(
        cls,
        window: ChunkWindow,
        *,
        profile: str,
        segments: Sequence[Segment],
    ) -> None:
        path = cls._boundary_cache_path(window)
        payload = {
            "version": 1,
            "profile_sha256": profile,
            "chunk": {
                "index": window.index,
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
            },
            "segments": [segment.model_dump(mode="json") for segment in segments],
        }
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @classmethod
    def _read_boundary_cache(
        cls,
        window: ChunkWindow,
        *,
        profile: str,
        source_id: str,
    ) -> list[Segment] | None:
        path = cls._boundary_cache_path(window)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected_chunk = {
                "index": window.index,
                "start_ms": window.start_ms,
                "end_ms": window.end_ms,
            }
            if (
                payload.get("version") != 1
                or payload.get("profile_sha256") != profile
                or payload.get("chunk") != expected_chunk
            ):
                return None
            segments = [Segment.model_validate(item) for item in payload["segments"]]
            if not segments or any(segment.source_id != source_id for segment in segments):
                return None
            previous_end = 0.0
            for segment in segments:
                if (
                    segment.start_seconds < previous_end
                    or segment.end_seconds > window.duration_seconds + 0.1
                ):
                    return None
                previous_end = segment.end_seconds
            return segments
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _read_manifest(
        path: Path,
        *,
        profile: str,
        duration_ms: int,
        maximum_ms: int,
    ) -> list[ChunkWindow] | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or payload.get("profile_sha256") != profile:
                return None
            windows = [
                ChunkWindow(
                    index=int(item["index"]),
                    start_ms=int(item["start_ms"]),
                    end_ms=int(item["end_ms"]),
                    path=path.parent / Path(str(item["file"])).name,
                )
                for item in payload["chunks"]
            ]
            if [window.index for window in windows] != list(range(len(windows))):
                return None
            _validate_chunk_coverage(
                [(window.start_ms, window.end_ms) for window in windows],
                duration_ms=duration_ms,
                maximum_ms=maximum_ms,
            )
            return windows
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ExternalToolError):
            return None

    @staticmethod
    def _remap(
        segments: Sequence[Segment],
        *,
        window: ChunkWindow,
        source: SourceRecord,
        profile: str,
        reused: bool,
        boundary_reused: bool,
        retained: bool,
    ) -> list[Segment]:
        result: list[Segment] = []
        previous_end = 0.0
        for segment in segments:
            if segment.source_id != source.source_id:
                raise ExternalToolError("Gemini 分块返回了错误的 source_id")
            if segment.start_seconds < previous_end:
                raise ExternalToolError("Gemini 分块内边界重叠或乱序")
            if segment.end_seconds > window.duration_seconds + 0.1:
                raise ExternalToolError("Gemini 分块边界越过本地块时长")
            local_end = min(segment.end_seconds, window.duration_seconds)
            if local_end <= segment.start_seconds:
                raise ExternalToolError("Gemini 分块返回了空片段")
            offset = window.start_ms / 1_000
            result.append(
                segment.model_copy(
                    update={
                        "start_seconds": round(offset + segment.start_seconds, 6),
                        "end_seconds": round(offset + local_end, 6),
                        "provenance": {
                            **segment.provenance,
                            "gemini_chunked": "true",
                            "gemini_chunk_profile": profile,
                            "gemini_chunk_index": str(window.index),
                            "gemini_chunk_start_ms": str(window.start_ms),
                            "gemini_chunk_end_ms": str(window.end_ms),
                            "gemini_chunk_path": str(window.path),
                            "gemini_chunk_cache": "reused" if reused else "rendered",
                            "gemini_boundary_cache": ("reused" if boundary_reused else "generated"),
                            "gemini_chunk_retained": str(retained).lower(),
                        },
                    }
                )
            )
            previous_end = local_end
        return result

    @staticmethod
    def _validate_remapped(segments: Sequence[Segment], source: SourceRecord) -> None:
        if not segments:
            raise ExternalToolError("Gemini 所有视频块均未返回语音片段")
        previous_end = 0.0
        for segment in segments:
            if segment.start_seconds < previous_end:
                raise ExternalToolError("Gemini 回映后的全局边界重叠或乱序")
            if segment.end_seconds > source.duration_seconds + 0.1:
                raise ExternalToolError("Gemini 回映后的边界越过源媒体时长")
            previous_end = segment.end_seconds
