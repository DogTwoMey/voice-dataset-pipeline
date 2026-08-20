"""Subtitle discovery, parsing, and embedded-stream extraction.

The module deliberately returns ordinary :class:`Segment` records so subtitle
timings can feed the same immutable clip materializer as local or model-backed
splitters.  Subtitle text is a hint until ASR and human review confirm it.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Segment, SplitBackend

SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa")
_TEXT_SUBTITLE_CODECS = {"ass", "mov_text", "ssa", "subrip", "text", "webvtt"}
_TIMING_RE = re.compile(
    r"^\s*(?P<start>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d+)?)\s*-->\s*"
    r"(?P<end>(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d+)?)(?:\s+.*)?$"
)
_ASS_TAG_RE = re.compile(r"\{[^{}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError(f"invalid subtitle interval: {self.start_seconds}..{self.end_seconds}")
        if not self.text.strip():
            raise ValueError("subtitle cue text must not be empty")


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: str) -> str:
    value = value.replace(r"\N", " ").replace(r"\n", " ")
    value = _ASS_TAG_RE.sub("", value)
    value = _HTML_TAG_RE.sub("", value)
    return " ".join(html.unescape(value).split())


def _timestamp(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"invalid subtitle timestamp: {value!r}")
    result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if result < 0:
        raise ValueError(f"negative subtitle timestamp: {value!r}")
    return result


def _parse_srt_or_vtt(text: str) -> list[SubtitleCue]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cues: list[SubtitleCue] = []
    index = 0
    while index < len(lines):
        timing = _TIMING_RE.match(lines[index])
        if timing is None:
            index += 1
            continue
        index += 1
        payload: list[str] = []
        while index < len(lines) and lines[index].strip():
            payload.append(lines[index])
            index += 1
        cleaned = _clean_text(" ".join(payload))
        if cleaned:
            cues.append(
                SubtitleCue(
                    start_seconds=_timestamp(timing.group("start")),
                    end_seconds=_timestamp(timing.group("end")),
                    text=cleaned,
                )
            )
    return cues


def _parse_ass(text: str) -> list[SubtitleCue]:
    fields: list[str] = []
    in_events = False
    cues: list[SubtitleCue] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.casefold() == "[events]"
            continue
        if not in_events:
            continue
        if line.casefold().startswith("format:"):
            fields = [item.strip().casefold() for item in line.split(":", 1)[1].split(",")]
            continue
        if not line.casefold().startswith("dialogue:") or not fields:
            continue
        values = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values, strict=True))
        if not {"start", "end", "text"}.issubset(row):
            continue
        cleaned = _clean_text(row["text"])
        if cleaned:
            cues.append(
                SubtitleCue(
                    start_seconds=_timestamp(row["start"]),
                    end_seconds=_timestamp(row["end"]),
                    text=cleaned,
                )
            )
    return cues


def parse_subtitle(path: str | Path) -> list[SubtitleCue]:
    """Parse SRT, WebVTT, ASS, or SSA into chronological non-empty cues."""

    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() not in SUBTITLE_EXTENSIONS:
        raise ValueError(f"unsupported subtitle format: {source.suffix}")
    text = source.read_text(encoding="utf-8-sig", errors="replace")
    cues = (
        _parse_ass(text)
        if source.suffix.casefold() in {".ass", ".ssa"}
        else _parse_srt_or_vtt(text)
    )
    return sorted(cues, key=lambda cue: (cue.start_seconds, cue.end_seconds, cue.text))


def find_sidecar(media_path: str | Path) -> Path | None:
    """Find an exact or language-suffixed subtitle beside a media file."""

    media = Path(media_path).expanduser().resolve()
    stem = media.stem.casefold()
    extension_rank = {extension: index for index, extension in enumerate(SUBTITLE_EXTENSIONS)}
    candidates: list[Path] = []
    if media.parent.is_dir():
        for candidate in media.parent.iterdir():
            suffix = candidate.suffix.casefold()
            candidate_stem = candidate.stem.casefold()
            if (
                candidate.is_file()
                and suffix in extension_rank
                and (candidate_stem == stem or candidate_stem.startswith(f"{stem}."))
            ):
                candidates.append(candidate.resolve())
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda path: (
            0 if path.stem.casefold() == stem else 1,
            extension_rank[path.suffix.casefold()],
            path.name.casefold(),
        ),
    )


class FFmpegSubtitleExtractor:
    """Extract the first text subtitle stream as a stable UTF-8 SRT sidecar."""

    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        runner: CommandRunner = _run_command,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self._runner = runner

    @staticmethod
    def _check(result: subprocess.CompletedProcess[str], action: str) -> None:
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-2_000:]
            raise RuntimeError(f"{action} failed with exit {result.returncode}: {detail}")

    def extract(self, media_path: str | Path, output_dir: str | Path) -> Path | None:
        media = Path(media_path).expanduser().resolve()
        if not media.is_file():
            raise FileNotFoundError(f"media file does not exist: {media}")
        probe = self._runner(
            [
                self.ffprobe_binary,
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream=index,codec_name:stream_tags=language,title",
                "-of",
                "json",
                str(media),
            ]
        )
        self._check(probe, "ffprobe subtitle discovery")
        try:
            payload: dict[str, Any] = json.loads(probe.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("ffprobe returned invalid subtitle metadata") from exc
        streams = [
            stream
            for stream in payload.get("streams", [])
            if str(stream.get("codec_name", "")).casefold() in _TEXT_SUBTITLE_CODECS
            and isinstance(stream.get("index"), int)
        ]
        if not streams:
            return None
        stream = min(streams, key=lambda item: int(item["index"]))
        stream_index = int(stream["index"])
        destination_dir = Path(output_dir).expanduser().resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        content_sha256 = _sha256_file(media)
        destination = destination_dir / f"{content_sha256}.embedded.{stream_index}.srt"
        if destination.is_file() and destination.stat().st_size:
            return destination

        temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.srt")
        try:
            result = self._runner(
                [
                    self.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(media),
                    "-map",
                    f"0:{stream_index}",
                    "-c:s",
                    "srt",
                    str(temporary),
                ]
            )
            self._check(result, "ffmpeg subtitle extraction")
            if not temporary.is_file() or not temporary.stat().st_size:
                return None
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)


def segments_from_subtitles(
    cues: Sequence[SubtitleCue],
    *,
    source_id: str,
    subtitle_path: str | Path,
    strategy: str,
    duration_seconds: float | None = None,
) -> list[Segment]:
    """Convert subtitle cues into traceable split proposals."""

    source = Path(subtitle_path).expanduser().resolve()
    result: list[Segment] = []
    for index, cue in enumerate(cues):
        start = max(0.0, cue.start_seconds)
        end = cue.end_seconds
        if duration_seconds is not None:
            end = min(end, duration_seconds)
        if end <= start:
            continue
        result.append(
            Segment(
                source_id=source_id,
                start_seconds=round(start, 6),
                end_seconds=round(end, 6),
                backend=SplitBackend.ENERGY,
                text_hint=cue.text,
                provenance={
                    "strategy": strategy,
                    "subtitle_path": str(source),
                    "cue_index": str(index),
                },
            )
        )
    return result
