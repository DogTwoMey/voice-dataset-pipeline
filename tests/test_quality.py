from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from voice_dataset_pipeline.cli import _preprocess
from voice_dataset_pipeline.media import sha256_file
from voice_dataset_pipeline.models import (
    ClipRecord,
    MediaKind,
    QualityRecord,
    Segment,
    SourceRecord,
)
from voice_dataset_pipeline.quality import (
    QualityThresholds,
    evaluate_audio,
    evaluate_workspace,
    measure_audio,
)
from voice_dataset_pipeline.workspace import Workspace

SAMPLE_RATE = 8_000


def _write(path, values):
    sf.write(path, np.asarray(values, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def test_audio_metrics_report_duration_rms_peak_clipping_and_silence(tmp_path):
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    audio = np.concatenate(
        (np.zeros(SAMPLE_RATE, dtype=np.float32), 0.5 * np.sin(2 * np.pi * 220 * time))
    )
    path = tmp_path / "mixed.wav"
    _write(path, audio)

    metrics = measure_audio(path, silence_threshold_dbfs=-45)

    assert metrics.duration_seconds == 2
    assert -12.1 < metrics.rms_dbfs < -12.0
    assert -6.1 < metrics.peak_dbfs < -5.9
    assert metrics.clipping_ratio == 0
    assert 0.49 < metrics.silence_ratio < 0.52


def test_quality_gate_records_all_rejection_reasons(tmp_path):
    path = tmp_path / "clipped.wav"
    _write(path, np.ones(SAMPLE_RATE // 2, dtype=np.float32))
    thresholds = QualityThresholds(
        min_duration_seconds=1,
        max_duration_seconds=2,
        max_clipping_ratio=0,
    )

    record = evaluate_audio(path, clip_id="clip", thresholds=thresholds)

    assert record.accepted is False
    assert "duration_too_short" in record.reasons
    assert "clipping_ratio_too_high" in record.reasons
    assert record.peak_dbfs > -0.01


def test_workspace_quality_evaluation_is_persisted_and_reused(tmp_path):
    workspace = Workspace.create(tmp_path / "job")
    audio = tmp_path / "tone.wav"
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    _write(audio, 0.2 * np.sin(2 * np.pi * 220 * time))
    digest = sha256_file(audio)
    clip = ClipRecord(
        clip_id="c" * 64,
        source_id="s" * 64,
        audio_path=audio,
        start_ms=0,
        end_ms=1_000,
        sample_rate=SAMPLE_RATE,
        frames=SAMPLE_RATE,
        sha256=digest,
    )
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])

    first = evaluate_workspace(workspace)
    second = evaluate_workspace(workspace)
    records = workspace.read_jsonl(workspace.paths.quality_jsonl, QualityRecord)

    assert first.evaluated == 1 and first.reused == 0
    assert second.evaluated == 0 and second.reused == 1
    assert first.accepted == second.accepted == 1
    assert len(records) == 1
    assert records[0].audio_sha256 == digest


def test_preprocess_skips_implicit_quality_when_disabled(tmp_path, monkeypatch, capsys):
    workspace = Workspace.create(tmp_path / "job")
    audio = tmp_path / "tone.wav"
    _write(audio, np.zeros(SAMPLE_RATE, dtype=np.float32))
    digest = sha256_file(audio)
    source_id = "s" * 64
    workspace.write_jsonl(
        workspace.paths.sources_jsonl,
        [
            SourceRecord(
                source_id=source_id,
                content_sha256=digest,
                original_path=audio,
                normalized_path=audio,
                media_kind=MediaKind.AUDIO,
                size_bytes=audio.stat().st_size,
                sample_rate=SAMPLE_RATE,
                channels=1,
                frames=SAMPLE_RATE,
                duration_seconds=1,
            )
        ],
    )
    workspace.write_jsonl(
        workspace.paths.segments_jsonl,
        [Segment(source_id=source_id, start_seconds=0, end_seconds=1)],
    )
    workspace.write_jsonl(
        workspace.paths.clips_jsonl,
        [
            ClipRecord(
                clip_id="c" * 64,
                source_id=source_id,
                audio_path=audio,
                start_ms=0,
                end_ms=1_000,
                sample_rate=SAMPLE_RATE,
                frames=SAMPLE_RATE,
                sha256=digest,
            )
        ],
    )
    config = tmp_path / "pipeline.toml"
    config.write_text("[quality]\nenabled = false\n", encoding="utf-8")

    import voice_dataset_pipeline.quality as quality_module

    def unexpected_quality(*_args, **_kwargs):
        raise AssertionError("preprocess must not evaluate disabled quality")

    monkeypatch.setattr(quality_module, "evaluate_workspace", unexpected_quality)
    args = argparse.Namespace(
        workspace=workspace.root,
        config=config,
        inputs=[],
        mode="auto",
        secrets=None,
        replace=False,
        force_quality=False,
        skip_asr=True,
        asr=False,
        force_asr=False,
    )

    assert _preprocess(args) == 0
    assert "质量门禁已禁用，跳过" in capsys.readouterr().out
