from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from voice_dataset_pipeline.subtitles import (
    FFmpegSubtitleExtractor,
    find_sidecar,
    parse_subtitle,
    segments_from_subtitles,
)


@pytest.mark.parametrize(
    ("suffix", "content", "expected"),
    [
        (
            ".srt",
            "1\n00:00:01,000 --> 00:00:02,500\n<b>你好</b>，世界\n\n",
            "你好，世界",
        ),
        (
            ".vtt",
            "WEBVTT\n\n00:01.000 --> 00:02.500 align:start\n你好，世界\n\n",
            "你好，世界",
        ),
        (
            ".ass",
            "[Events]\nFormat: Layer, Start, End, Style, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:02.50,Default,{\\i1}你好\\N世界\n",
            "你好 世界",
        ),
    ],
)
def test_parse_supported_subtitle_formats(tmp_path, suffix, content, expected):
    path = tmp_path / f"episode{suffix}"
    path.write_text(content, encoding="utf-8")

    cues = parse_subtitle(path)

    assert len(cues) == 1
    assert cues[0].start_seconds == 1
    assert cues[0].end_seconds == 2.5
    assert cues[0].text == expected


def test_sidecar_prefers_exact_stem_and_segments_keep_text_provenance(tmp_path):
    media = tmp_path / "episode.mp4"
    media.write_bytes(b"video")
    (tmp_path / "episode.zh-CN.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n语言版本\n\n", encoding="utf-8"
    )
    exact = tmp_path / "episode.srt"
    exact.write_text("1\n00:00:00,000 --> 00:00:01,500\n准确台词\n\n", encoding="utf-8")

    selected = find_sidecar(media)
    segments = segments_from_subtitles(
        parse_subtitle(selected),
        source_id="source",
        subtitle_path=selected,
        strategy="sidecar_subtitle",
        duration_seconds=1.25,
    )

    assert selected == exact.resolve()
    assert segments[0].end_seconds == 1.25
    assert segments[0].text_hint == "准确台词"
    assert segments[0].provenance["strategy"] == "sidecar_subtitle"
    assert segments[0].provenance["subtitle_path"] == str(exact.resolve())


def test_ffmpeg_extractor_selects_first_text_stream_and_publishes_srt(tmp_path):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"container")
    commands: list[list[str]] = []

    def runner(argv):
        command = list(argv)
        commands.append(command)
        if command[0] == "probe-bin":
            payload = {
                "streams": [
                    {"index": 1, "codec_name": "hdmv_pgs_subtitle"},
                    {"index": 3, "codec_name": "subrip"},
                    {"index": 5, "codec_name": "ass"},
                ]
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        Path(command[-1]).write_text(
            "1\n00:00:00,000 --> 00:00:01,000\n内嵌字幕\n\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    extractor = FFmpegSubtitleExtractor(
        ffmpeg_binary="ffmpeg-bin",
        ffprobe_binary="probe-bin",
        runner=runner,
    )

    output = extractor.extract(media, tmp_path / "scratch")

    assert output is not None and output.is_file()
    assert parse_subtitle(output)[0].text == "内嵌字幕"
    assert commands[1][commands[1].index("-map") + 1] == "0:3"


def test_ffmpeg_extractor_cache_isolated_for_same_name_different_media(tmp_path):
    first = tmp_path / "one" / "episode.mkv"
    second = tmp_path / "two" / "episode.mkv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first-container")
    second.write_bytes(b"second-container")
    extraction_inputs: list[Path] = []

    def runner(argv):
        command = list(argv)
        if command[0] == "probe-bin":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"streams": [{"index": 3, "codec_name": "subrip"}]}),
                stderr="",
            )
        media = Path(command[command.index("-i") + 1])
        extraction_inputs.append(media)
        text = "第一份" if media == first.resolve() else "第二份"
        Path(command[-1]).write_text(
            f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    extractor = FFmpegSubtitleExtractor(ffprobe_binary="probe-bin", runner=runner)
    scratch = tmp_path / "scratch"

    first_output = extractor.extract(first, scratch)
    second_output = extractor.extract(second, scratch)

    assert first_output != second_output
    assert parse_subtitle(first_output)[0].text == "第一份"
    assert parse_subtitle(second_output)[0].text == "第二份"
    assert extraction_inputs == [first.resolve(), second.resolve()]


def test_ffmpeg_extractor_cache_invalidates_when_media_content_changes(tmp_path):
    media = tmp_path / "episode.mkv"
    media.write_bytes(b"version-one")
    extraction_count = 0

    def runner(argv):
        nonlocal extraction_count
        command = list(argv)
        if command[0] == "probe-bin":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"streams": [{"index": 3, "codec_name": "subrip"}]}),
                stderr="",
            )
        extraction_count += 1
        text = "第一版" if media.read_bytes() == b"version-one" else "第二版"
        Path(command[-1]).write_text(
            f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    extractor = FFmpegSubtitleExtractor(ffprobe_binary="probe-bin", runner=runner)
    scratch = tmp_path / "scratch"

    first_output = extractor.extract(media, scratch)
    assert parse_subtitle(first_output)[0].text == "第一版"
    assert extractor.extract(media, scratch) == first_output
    assert extraction_count == 1

    media.write_bytes(b"version-two")
    second_output = extractor.extract(media, scratch)

    assert second_output != first_output
    assert parse_subtitle(second_output)[0].text == "第二版"
    assert extraction_count == 2
