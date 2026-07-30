"""Media discovery and content-deduplicated FFmpeg ingestion."""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

import soundfile as sf

from .config import MediaConfig
from .models import (
    ClipRecord,
    InputMode,
    LabelRecord,
    MediaKind,
    ReviewState,
    Segment,
    SourceRecord,
)
from .workspace import Workspace


@runtime_checkable
class AudioDecoder(Protocol):
    """Normalise any supported media into a mono PCM WAV.

    Tests and embedders can satisfy this seam without installing FFmpeg.
    """

    def normalize(self, source: Path, destination: Path, *, sample_rate: int) -> None: ...


class FFmpegDecoder:
    def __init__(self, binary: str = "ffmpeg") -> None:
        self.binary = binary

    def normalize(self, source: Path, destination: Path, *, sample_rate: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"FFmpeg executable was not found: {self.binary!r}") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg failed for {source} (exit {exc.returncode})") from exc


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def source_identity(content_sha256: str, *, sample_rate: int) -> str:
    """Fingerprint the source bytes together with normalization semantics."""

    payload = f"{content_sha256}\0{sample_rate}\0mono\0pcm_s16le".encode()
    return hashlib.sha256(payload).hexdigest()


def _normalise_extensions(extensions: Iterable[str]) -> set[str]:
    return {
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension in extensions
    }


def media_kind(
    path: str | Path, *, audio_extensions: Iterable[str], video_extensions: Iterable[str]
) -> MediaKind | None:
    suffix = Path(path).suffix.casefold()
    if suffix in _normalise_extensions(audio_extensions):
        return MediaKind.AUDIO
    if suffix in _normalise_extensions(video_extensions):
        return MediaKind.VIDEO
    return None


def discover_media(
    inputs: str | Path | Iterable[str | Path],
    *,
    audio_extensions: Iterable[str],
    video_extensions: Iterable[str],
    input_mode: InputMode = InputMode.AUTO,
) -> list[Path]:
    """Recursively discover media in deterministic path order."""

    input_mode = InputMode(input_mode)
    roots = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
    discovered: dict[str, Path] = {}
    missing: list[Path] = []

    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            missing.append(root)
            continue
        candidates = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for candidate in candidates:
            kind = media_kind(
                candidate,
                audio_extensions=audio_extensions,
                video_extensions=video_extensions,
            )
            if kind is None:
                continue
            if input_mode is InputMode.AUDIO and kind is not MediaKind.AUDIO:
                continue
            if input_mode is InputMode.VIDEO and kind is not MediaKind.VIDEO:
                continue
            resolved = candidate.resolve()
            discovered[str(resolved).casefold()] = resolved

    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"media input does not exist: {missing_text}")
    return sorted(discovered.values(), key=lambda path: str(path).casefold())


class MediaIngestor:
    def __init__(
        self,
        workspace: Workspace,
        config: MediaConfig,
        *,
        decoder: AudioDecoder | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.decoder = decoder or FFmpegDecoder(config.ffmpeg_binary)

    def discover(
        self,
        inputs: str | Path | Iterable[str | Path],
        *,
        input_mode: InputMode = InputMode.AUTO,
    ) -> list[Path]:
        return discover_media(
            inputs,
            audio_extensions=self.config.audio_extensions,
            video_extensions=self.config.video_extensions,
            input_mode=input_mode,
        )

    def ingest(
        self,
        inputs: str | Path | Iterable[str | Path],
        *,
        input_mode: InputMode = InputMode.AUTO,
    ) -> list[SourceRecord]:
        """Normalize every unique file content once and update the manifest."""

        existing_records = self.workspace.read_jsonl(
            self.workspace.paths.sources_jsonl, SourceRecord
        )
        existing = {record.source_id: record for record in existing_records}
        results: list[SourceRecord] = []
        seen: set[str] = set()

        for source in self.discover(inputs, input_mode=input_mode):
            digest = sha256_file(source)
            source_id = source_identity(digest, sample_rate=self.config.sample_rate)
            if source_id in seen:
                continue
            seen.add(source_id)

            kind = media_kind(
                source,
                audio_extensions=self.config.audio_extensions,
                video_extensions=self.config.video_extensions,
            )
            assert kind is not None
            prior = existing.get(source_id)
            if (
                prior is not None
                and prior.normalized_path.exists()
                and prior.sample_rate == self.config.sample_rate
                and prior.channels == 1
            ):
                self._supersede_other_normalizations(
                    content_sha256=digest,
                    keep_source_id=source_id,
                )
                if prior.original_path != source or prior.media_kind != kind:
                    prior = prior.model_copy(
                        update={
                            "original_path": source,
                            "media_kind": kind,
                            "size_bytes": source.stat().st_size,
                        }
                    )
                    self.workspace.upsert_jsonl(
                        self.workspace.paths.sources_jsonl,
                        prior,
                        key="source_id",
                    )
                results.append(prior)
                continue

            destination = (
                self.workspace.paths.normalized_audio
                / f"{source_id}_{self.config.sample_rate}_mono.wav"
            )
            if not destination.exists():
                temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.wav")
                try:
                    self.decoder.normalize(source, temporary, sample_rate=self.config.sample_rate)
                    info = sf.info(temporary)
                    if info.channels != 1:
                        raise ValueError(
                            f"decoder produced {info.channels} channels, expected mono"
                        )
                    if info.samplerate != self.config.sample_rate:
                        raise ValueError(
                            "decoder produced "
                            f"{info.samplerate} Hz, expected {self.config.sample_rate} Hz"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)

            info = sf.info(destination)
            if info.channels != 1 or info.samplerate != self.config.sample_rate:
                raise ValueError(
                    f"normalized asset has unexpected format: {destination} "
                    f"({info.channels} ch, {info.samplerate} Hz)"
                )
            record = SourceRecord(
                source_id=source_id,
                content_sha256=digest,
                original_path=source,
                normalized_path=destination,
                media_kind=kind,
                size_bytes=source.stat().st_size,
                sample_rate=info.samplerate,
                channels=info.channels,
                frames=info.frames,
                duration_seconds=info.duration,
            )
            self._supersede_other_normalizations(
                content_sha256=digest,
                keep_source_id=source_id,
            )
            self.workspace.upsert_jsonl(
                self.workspace.paths.sources_jsonl,
                record,
                key="source_id",
            )
            existing[source_id] = record
            results.append(record)
        return results

    def _supersede_other_normalizations(
        self,
        *,
        content_sha256: str,
        keep_source_id: str,
    ) -> None:
        """Invalidate manifests derived from an older normalization profile.

        Immutable WAV files remain on disk for recovery, but stale rows cannot
        silently flow into a new export.
        """

        sources = self.workspace.read_jsonl(
            self.workspace.paths.sources_jsonl,
            SourceRecord,
        )
        assert isinstance(sources, list)
        obsolete_sources = {
            row.source_id
            for row in sources
            if row.content_sha256 == content_sha256 and row.source_id != keep_source_id
        }
        if not obsolete_sources:
            return
        segments = self.workspace.read_jsonl(
            self.workspace.paths.segments_jsonl,
            Segment,
        )
        clips = self.workspace.read_jsonl(
            self.workspace.paths.clips_jsonl,
            ClipRecord,
        )
        labels = self.workspace.read_jsonl(
            self.workspace.paths.labels_jsonl,
            LabelRecord,
        )
        assert isinstance(segments, list)
        assert isinstance(clips, list)
        assert isinstance(labels, list)
        obsolete_clips = {row.clip_id for row in clips if row.source_id in obsolete_sources}
        self.workspace.write_jsonl(
            self.workspace.paths.sources_jsonl,
            [row for row in sources if row.source_id not in obsolete_sources],
        )
        self.workspace.write_jsonl(
            self.workspace.paths.segments_jsonl,
            [row for row in segments if row.source_id not in obsolete_sources],
        )
        remaining_clips = [row for row in clips if row.source_id not in obsolete_sources]
        self.workspace.write_jsonl(self.workspace.paths.clips_jsonl, remaining_clips)
        self.workspace.write_jsonl(
            self.workspace.paths.labels_jsonl,
            [row for row in labels if row.clip_id not in obsolete_clips],
        )
        state = self.workspace.load_review()
        if state.order or state.decisions:
            remaining_order = [clip_id for clip_id in state.order if clip_id not in obsolete_clips]
            completed_before_cursor = sum(
                1 for clip_id in state.order[: state.cursor] if clip_id not in obsolete_clips
            )
            self.workspace.save_review(
                ReviewState(
                    cursor=min(completed_before_cursor, len(remaining_order)),
                    order=remaining_order,
                    decisions={
                        clip_id: decision
                        for clip_id, decision in state.decisions.items()
                        if clip_id not in obsolete_clips
                    },
                )
            )
