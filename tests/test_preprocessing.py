from __future__ import annotations

from voice_dataset_pipeline.models import MediaKind, Segment, SourceRecord, SplitBackend
from voice_dataset_pipeline.preprocessing import PreprocessingPipeline, SegmentationStrategy


class StubSilenceSplitter:
    backend = SplitBackend.ENERGY

    def __init__(self):
        self.calls = 0

    def split(self, audio_path, *, source_id):
        self.calls += 1
        return [Segment(source_id=source_id, start_seconds=0.2, end_seconds=1.2)]


class StubEmbeddedExtractor:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def extract(self, media_path, output_dir):
        self.calls += 1
        return self.result


class StubVisionSplitter:
    def __init__(self, result=()):
        self.result = result
        self.calls = 0

    def split(self, source):
        self.calls += 1
        return self.result


def _source(tmp_path, *, kind=MediaKind.VIDEO):
    original = tmp_path / ("episode.mp4" if kind is MediaKind.VIDEO else "episode.wav")
    normalized = tmp_path / "normalized.wav"
    original.write_bytes(b"media")
    normalized.write_bytes(b"wave")
    return SourceRecord(
        source_id="a" * 64,
        content_sha256="b" * 64,
        original_path=original,
        normalized_path=normalized,
        media_kind=kind,
        size_bytes=5,
        sample_rate=48_000,
        channels=1,
        frames=96_000,
        duration_seconds=2,
    )


def test_sidecar_is_preferred_and_downstream_fallbacks_are_not_called(tmp_path):
    source = _source(tmp_path)
    sidecar = source.original_path.with_suffix(".srt")
    sidecar.write_text("1\n00:00:00,000 --> 00:00:01,000\n字幕台词\n\n", encoding="utf-8")
    silence = StubSilenceSplitter()
    embedded = StubEmbeddedExtractor()
    vision = StubVisionSplitter()
    pipeline = PreprocessingPipeline(
        silence_splitter=silence,
        embedded_extractor=embedded,
        vision_splitter=vision,
    )

    result = pipeline.split(source, scratch_dir=tmp_path / "scratch")

    assert result.strategy is SegmentationStrategy.SIDECAR
    assert result.attempts == (SegmentationStrategy.SIDECAR,)
    assert result.segments[0].text_hint == "字幕台词"
    assert result.segments[0].provenance["strategy"] == "sidecar_subtitle"
    assert embedded.calls == vision.calls == silence.calls == 0


def test_embedded_subtitle_precedes_vision(tmp_path):
    source = _source(tmp_path)
    embedded_path = tmp_path / "embedded.srt"
    embedded_path.write_text("1\n00:00:00,250 --> 00:00:01,250\n内嵌台词\n\n", encoding="utf-8")
    silence = StubSilenceSplitter()
    vision = StubVisionSplitter()
    pipeline = PreprocessingPipeline(
        silence_splitter=silence,
        embedded_extractor=StubEmbeddedExtractor(embedded_path),
        vision_splitter=vision,
    )

    result = pipeline.split(source, scratch_dir=tmp_path / "scratch")

    assert result.strategy is SegmentationStrategy.EMBEDDED
    assert result.segments[0].text_hint == "内嵌台词"
    assert vision.calls == silence.calls == 0


def test_empty_subtitle_and_vision_results_fall_back_to_silence(tmp_path):
    source = _source(tmp_path)
    empty = source.original_path.with_suffix(".vtt")
    empty.write_text("WEBVTT\n\n", encoding="utf-8")
    silence = StubSilenceSplitter()
    vision = StubVisionSplitter()
    pipeline = PreprocessingPipeline(
        silence_splitter=silence,
        embedded_extractor=StubEmbeddedExtractor(None),
        vision_splitter=vision,
    )

    result = pipeline.split(source, scratch_dir=tmp_path / "scratch")

    assert result.strategy is SegmentationStrategy.SILENCE
    assert result.attempts == (
        SegmentationStrategy.SIDECAR,
        SegmentationStrategy.EMBEDDED,
        SegmentationStrategy.VISION,
        SegmentationStrategy.SILENCE,
    )
    assert result.segments[0].provenance["strategy"] == "silence_splitter"
    assert "sidecar_subtitle" in result.failures
    assert "embedded_subtitle" in result.failures
    assert "vision_model" in result.failures
    assert silence.calls == 1


def test_vision_boundaries_are_used_before_silence(tmp_path):
    source = _source(tmp_path)
    silence = StubSilenceSplitter()
    vision = StubVisionSplitter(
        [Segment(source_id=source.source_id, start_seconds=0.1, end_seconds=0.9)]
    )
    pipeline = PreprocessingPipeline(
        silence_splitter=silence,
        embedded_extractor=StubEmbeddedExtractor(None),
        vision_splitter=vision,
    )

    result = pipeline.split(source, scratch_dir=tmp_path / "scratch")

    assert result.strategy is SegmentationStrategy.VISION
    assert result.segments[0].provenance["strategy"] == "vision_model"
    assert silence.calls == 0
