from __future__ import annotations

import numpy as np
import soundfile as sf

from voice_dataset_pipeline.config import load_config
from voice_dataset_pipeline.media import sha256_file
from voice_dataset_pipeline.models import Segment
from voice_dataset_pipeline.splitting import EnergySplitter, materialize_clips

SAMPLE_RATE = 8_000


def _tone(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    count = round(seconds * SAMPLE_RATE)
    time = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * 220 * time)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.float32)


def _write(path, parts) -> None:
    sf.write(path, np.concatenate(parts), SAMPLE_RATE, subtype="PCM_16")


def _split_config(**updates):
    values = {
        "frame_ms": 20,
        "hop_ms": 10,
        "min_speech_ms": 100,
        "min_silence_ms": 100,
        "min_segment_seconds": 0.5,
        "merge_gap_ms": 400,
        "padding_ms": 200,
        "max_segment_seconds": 3.0,
        "boundary_search_ms": 100,
    }
    values.update(updates)
    return load_config().splitting.model_copy(update=values)


def test_merge_requires_at_least_one_short_neighbour_and_padding_contracts(tmp_path):
    audio = tmp_path / "two.wav"
    _write(
        audio,
        [_silence(0.2), _tone(1.0), _silence(0.3), _tone(1.0), _silence(0.2)],
    )

    segments = EnergySplitter(_split_config()).split(audio, source_id="source")

    assert len(segments) == 2
    assert segments[0].end_seconds <= segments[1].start_seconds
    # 200 ms padding on each side overlaps in a 300 ms gap and yields to its midpoint.
    assert abs(segments[0].end_seconds - 1.35) < 0.05
    assert abs(segments[1].start_seconds - 1.35) < 0.05


def test_short_neighbour_is_merged(tmp_path):
    audio = tmp_path / "short.wav"
    _write(
        audio,
        [_silence(0.2), _tone(0.2), _silence(0.2), _tone(1.0), _silence(0.2)],
    )

    segments = EnergySplitter(_split_config(padding_ms=50, max_segment_seconds=3)).split(
        audio, source_id="source"
    )

    assert len(segments) == 1
    assert segments[0].duration_seconds > 1.3


def test_max_duration_and_immutable_materialization(tmp_path, monkeypatch):
    audio = tmp_path / "long.wav"
    _write(audio, [_silence(0.1), _tone(3.2), _silence(0.1)])
    source_id = sha256_file(audio)
    splitter = EnergySplitter(
        _split_config(
            padding_ms=100,
            max_segment_seconds=1.0,
            min_segment_seconds=0.3,
        )
    )

    segments = splitter.split(audio, source_id=source_id)

    assert len(segments) >= 4
    assert max(segment.duration_seconds for segment in segments) <= 1.001
    records = splitter.materialize(audio, segments, tmp_path / "clips")

    def fail_write(*args, **kwargs):
        raise AssertionError("an immutable existing clip must not be rewritten")

    monkeypatch.setattr("voice_dataset_pipeline.splitting.sf.write", fail_write)
    repeated = splitter.materialize(audio, segments, tmp_path / "clips")

    assert [record.clip_id for record in records] == [record.clip_id for record in repeated]
    assert all(record.audio_path.is_file() for record in repeated)


def test_materialization_preserves_subtitle_text_hint(tmp_path):
    audio = tmp_path / "hint.wav"
    _write(audio, [_tone(1.0)])
    source_id = sha256_file(audio)

    records = materialize_clips(
        audio,
        [
            Segment(
                source_id=source_id,
                start_seconds=0,
                end_seconds=1,
                text_hint="字幕提示",
                provenance={"strategy": "sidecar_subtitle"},
            )
        ],
        tmp_path / "clips",
    )

    assert records[0].text == "字幕提示"
