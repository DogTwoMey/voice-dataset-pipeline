from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_dataset_pipeline.config import GeminiChunkingConfig
from voice_dataset_pipeline.gemini_chunking import (
    ChunkedGeminiSplitter,
    parse_silence_midpoints,
    plan_chunk_boundaries,
)
from voice_dataset_pipeline.models import MediaKind, Segment, SourceRecord, SplitBackend


def _source(tmp_path: Path, *, duration: float) -> SourceRecord:
    original = tmp_path / "episode.mp4"
    normalized = tmp_path / "normalized.wav"
    original.write_bytes(b"video")
    normalized.write_bytes(b"audio")
    return SourceRecord(
        source_id="a" * 64,
        content_sha256="b" * 64,
        original_path=original,
        normalized_path=normalized,
        media_kind=MediaKind.VIDEO,
        size_bytes=5,
        sample_rate=48_000,
        channels=1,
        frames=round(duration * 48_000),
        duration_seconds=duration,
    )


class FakeRunner:
    def __init__(self, silence_output: str = "") -> None:
        self.silence_output = silence_output
        self.commands: list[list[str]] = []

    def __call__(self, command):
        command = list(command)
        self.commands.append(command)
        if any("silencedetect=" in item for item in command):
            return SimpleNamespace(returncode=0, stdout="", stderr=self.silence_output)
        Path(command[-1]).write_bytes(b"preview")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class FakeGemini:
    model = "gemini-test"

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_at = fail_at

    def split(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if self.fail_at == index:
            raise RuntimeError("fixture interruption")
        return [
            Segment(
                source_id=str(kwargs["source_id"]),
                start_seconds=1.0,
                end_seconds=2.0,
                backend=SplitBackend.GEMINI,
            )
        ]


def _config(**updates: object) -> GeminiChunkingConfig:
    return GeminiChunkingConfig(
        threshold_seconds=100,
        target_seconds=60,
        max_seconds=75,
        boundary_search_seconds=10,
        **updates,
    )


def _splitter(
    tmp_path: Path,
    client: FakeGemini,
    runner: FakeRunner,
    **config_updates: object,
) -> ChunkedGeminiSplitter:
    return ChunkedGeminiSplitter(
        client,
        config=_config(**config_updates),
        ffmpeg_binary="fixture-ffmpeg",
        scratch_dir=tmp_path / "chunks",
        min_segment_seconds=0.8,
        max_segment_seconds=15,
        runner=runner,
    )


def test_silence_parser_and_full_length_plan_cover_956_seconds() -> None:
    output = """
[silencedetect] silence_start: 74.2
[silencedetect] silence_end: 76.2 | silence_duration: 2
[silencedetect] silence_start: 149.5
[silencedetect] silence_end: 150.5 | silence_duration: 1
"""
    assert parse_silence_midpoints(output) == [75_200, 150_000]

    windows = plan_chunk_boundaries(
        956_053,
        [75_200, 150_000],
        target_ms=75_000,
        maximum_ms=100_000,
        search_ms=15_000,
    )

    assert 10 <= len(windows) <= 13
    assert windows[0][0] == 0
    assert windows[-1][1] == 956_053
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:], strict=False))
    assert all(0 < end - start <= 100_000 for start, end in windows)


def test_short_video_keeps_single_original_media_call(tmp_path: Path) -> None:
    source = _source(tmp_path, duration=30)
    client = FakeGemini()
    runner = FakeRunner()

    segments = _splitter(tmp_path, client, runner).split(source)

    assert len(segments) == 1
    assert client.calls[0]["path"] == source.original_path
    assert client.calls[0]["duration_seconds"] == 30
    assert runner.commands == []
    assert segments[0].provenance == {}


def test_long_video_chunks_near_silence_remaps_and_reuses_both_caches(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, duration=200)
    silence = """
silence_start: 55
silence_end: 57 | silence_duration: 2
silence_start: 115
silence_end: 117 | silence_duration: 2
"""
    first_client = FakeGemini()
    first_runner = FakeRunner(silence)

    segments = _splitter(tmp_path, first_client, first_runner).split(source)

    assert [(row.start_seconds, row.end_seconds) for row in segments] == [
        (1.0, 2.0),
        (57.0, 58.0),
        (117.0, 118.0),
        (177.0, 178.0),
    ]
    assert len(first_client.calls) == 4
    render_commands = [
        command
        for command in first_runner.commands
        if not any("silencedetect=" in x for x in command)
    ]
    assert len(render_commands) == 4
    assert all("500k" in command and "scale=-2:360" in command for command in render_commands)
    assert segments[0].provenance["gemini_chunk_cache"] == "rendered"
    assert segments[0].provenance["gemini_boundary_cache"] == "generated"

    second_client = FakeGemini()
    second_runner = FakeRunner("should not be parsed")
    cached = _splitter(tmp_path, second_client, second_runner).split(source)

    assert second_runner.commands == []
    assert second_client.calls == []
    assert cached == [
        row.model_copy(
            update={
                "provenance": {
                    **row.provenance,
                    "gemini_chunk_cache": "reused",
                    "gemini_boundary_cache": "reused",
                }
            }
        )
        for row in segments
    ]


def test_completed_chunk_boundaries_survive_later_api_failure(tmp_path: Path) -> None:
    source = _source(tmp_path, duration=200)
    runner = FakeRunner()

    with pytest.raises(RuntimeError, match="fixture interruption"):
        _splitter(tmp_path, FakeGemini(fail_at=1), runner).split(source)

    resumed_client = FakeGemini()
    resumed = _splitter(tmp_path, resumed_client, FakeRunner()).split(source)

    assert len(resumed_client.calls) == 3
    assert len(resumed) == 4
    assert resumed[0].provenance["gemini_boundary_cache"] == "reused"
    assert all(row.provenance["gemini_boundary_cache"] == "generated" for row in resumed[1:])


def test_cleanup_removes_preview_media_but_keeps_boundary_cache(tmp_path: Path) -> None:
    source = _source(tmp_path, duration=200)
    segments = _splitter(
        tmp_path,
        FakeGemini(),
        FakeRunner(),
        keep_chunks=False,
    ).split(source)

    chunk_root = tmp_path / "chunks"
    assert not list(chunk_root.rglob("*.mp4"))
    assert len(list(chunk_root.rglob("*.segments.json"))) == len(segments)
    assert all(row.provenance["gemini_chunk_retained"] == "false" for row in segments)
