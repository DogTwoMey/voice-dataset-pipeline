from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voice_dataset_pipeline.config import load_config
from voice_dataset_pipeline.media import MediaIngestor, discover_media, sha256_file
from voice_dataset_pipeline.models import (
    ClipRecord,
    InputMode,
    LabelRecord,
    ReviewDecision,
    ReviewState,
    Segment,
)
from voice_dataset_pipeline.workspace import Workspace


class FakeDecoder:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def normalize(self, source: Path, destination: Path, *, sample_rate: int) -> None:
        self.calls.append(source)
        values = np.linspace(-0.2, 0.2, sample_rate // 10, dtype=np.float32)
        sf.write(destination, values, sample_rate, format="WAV", subtype="PCM_16")


def test_discover_media_recurses_filters_and_sorts(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.MP4").write_bytes(b"video")
    (nested / "a.wav").write_bytes(b"audio")
    (nested / "ignore.txt").write_text("no", encoding="utf-8")

    all_media = discover_media(
        tmp_path,
        audio_extensions=[".wav"],
        video_extensions=[".mp4"],
    )
    audio_only = discover_media(
        tmp_path,
        audio_extensions=[".wav"],
        video_extensions=[".mp4"],
        input_mode=InputMode.AUDIO,
    )

    assert [path.name for path in all_media] == ["a.wav", "b.MP4"]
    assert [path.name for path in audio_only] == ["a.wav"]


def test_ingest_deduplicates_by_input_sha_without_real_ffmpeg(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    # Different names and media types, identical bytes.
    (inputs / "same.wav").write_bytes(b"identical-media-content")
    (inputs / "same.mp4").write_bytes(b"identical-media-content")

    config = load_config().media.model_copy(update={"sample_rate": 8_000})
    workspace = Workspace.create(tmp_path / "workspace")
    decoder = FakeDecoder()
    ingestor = MediaIngestor(workspace, config, decoder=decoder)

    first = ingestor.ingest(inputs)
    second = ingestor.ingest(inputs)

    assert len(first) == len(second) == 1
    assert len(decoder.calls) == 1
    assert first[0].source_id == second[0].source_id
    assert first[0].normalized_path == second[0].normalized_path
    info = sf.info(first[0].normalized_path)
    assert info.channels == 1
    assert info.samplerate == 8_000
    assert len(workspace.read_jsonl(workspace.paths.sources_jsonl)) == 1

    changed_decoder = FakeDecoder()
    changed_config = config.model_copy(update={"sample_rate": 16_000})
    changed = MediaIngestor(workspace, changed_config, decoder=changed_decoder).ingest(inputs)

    assert len(changed_decoder.calls) == 1
    assert changed[0].source_id != first[0].source_id
    assert changed[0].normalized_path != first[0].normalized_path
    assert sf.info(changed[0].normalized_path).samplerate == 16_000
    assert len(workspace.read_jsonl(workspace.paths.sources_jsonl)) == 1


def test_reingest_refreshes_moved_original_path(tmp_path):
    first_path = tmp_path / "first.wav"
    first_path.write_bytes(b"same-media")
    config = load_config().media.model_copy(update={"sample_rate": 8_000})
    workspace = Workspace.create(tmp_path / "workspace")
    ingestor = MediaIngestor(workspace, config, decoder=FakeDecoder())
    first = ingestor.ingest(first_path)[0]

    moved_path = tmp_path / "moved.wav"
    first_path.rename(moved_path)
    second = ingestor.ingest(moved_path)[0]

    assert second.source_id == first.source_id
    assert second.original_path == moved_path.resolve()
    assert second.normalized_path == first.normalized_path


def test_changed_normalization_invalidates_old_downstream_manifests(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"same-media")
    config = load_config().media.model_copy(update={"sample_rate": 8_000})
    workspace = Workspace.create(tmp_path / "workspace")
    first = MediaIngestor(workspace, config, decoder=FakeDecoder()).ingest(source)[0]
    clip = ClipRecord(
        clip_id="a" * 64,
        source_id=first.source_id,
        audio_path=first.normalized_path,
        start_ms=0,
        end_ms=100,
        sample_rate=8_000,
        frames=800,
        sha256=sha256_file(first.normalized_path),
    )
    workspace.write_jsonl(
        workspace.paths.segments_jsonl,
        [Segment(source_id=first.source_id, start_seconds=0, end_seconds=0.1)],
    )
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    workspace.write_jsonl(
        workspace.paths.labels_jsonl,
        [LabelRecord(clip_id=clip.clip_id, transcript="old")],
    )
    workspace.save_review(
        ReviewState(
            cursor=1,
            order=[clip.clip_id],
            decisions={
                clip.clip_id: ReviewDecision(
                    clip_id=clip.clip_id,
                    transcript="old",
                    confirmed=True,
                )
            },
        )
    )

    changed_config = config.model_copy(update={"sample_rate": 16_000})
    second = MediaIngestor(
        workspace,
        changed_config,
        decoder=FakeDecoder(),
    ).ingest(source)[0]

    assert second.source_id != first.source_id
    assert workspace.read_jsonl(workspace.paths.segments_jsonl) == []
    assert workspace.read_jsonl(workspace.paths.clips_jsonl) == []
    assert workspace.read_jsonl(workspace.paths.labels_jsonl) == []
    assert workspace.load_review().order == []
