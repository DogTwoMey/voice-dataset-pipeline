from __future__ import annotations

import hashlib
import json
from pathlib import Path

from voice_dataset_pipeline.emotion import plan_for
from voice_dataset_pipeline.references import ReferenceSelector


def test_selector_prefers_reviewed_same_emotion(tmp_path: Path) -> None:
    neutral = tmp_path / "neutral.wav"
    happy = tmp_path / "happy.wav"
    neutral.write_bytes(b"wav")
    happy.write_bytes(b"wav")
    rows = [
        {
            "wav_path": str(neutral),
            "sha256": hashlib.sha256(neutral.read_bytes()).hexdigest(),
            "text": "中性",
            "emotion": "neutral",
            "duration_seconds": 6,
            "reviewed": True,
        },
        {
            "wav_path": str(happy),
            "sha256": hashlib.sha256(happy.read_bytes()).hexdigest(),
            "text": "开心",
            "emotion": "happy",
            "duration_seconds": 5,
            "reviewed": True,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    selected = ReferenceSelector(manifest).select(plan_for("happy"))
    assert selected.audio_path == happy
    assert selected.transcript == "开心"


def test_selector_rejects_tampered_selected_reference(tmp_path: Path) -> None:
    audio = tmp_path / "happy.wav"
    audio.write_bytes(b"original")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "wav_path": str(audio),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "text": "开心",
                "emotion": "happy",
                "duration_seconds": 5,
                "reviewed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audio.write_bytes(b"tampered")

    try:
        ReferenceSelector(manifest).select(plan_for("happy"))
    except ValueError as exc:
        assert "reference audio SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered reference audio was accepted")


def test_selector_honors_configured_reference_clip(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    pinned = tmp_path / "pinned.wav"
    first.write_bytes(b"first")
    pinned.write_bytes(b"pinned")
    rows = [
        {
            "clip_id": "first-clip",
            "wav_path": str(first),
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            "text": "第一条",
            "emotion": "neutral",
            "duration_seconds": 6.0,
            "reviewed": True,
        },
        {
            "clip_id": "pinned-clip",
            "wav_path": str(pinned),
            "sha256": hashlib.sha256(pinned.read_bytes()).hexdigest(),
            "text": "固定参考",
            "emotion": "neutral",
            "duration_seconds": 6.1,
            "reviewed": True,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    selected = ReferenceSelector(manifest, preferred_clip_ids={"neutral": "pinned-clip"}).select(
        plan_for("neutral")
    )

    assert selected.audio_path == pinned
    assert selected.transcript == "固定参考"
    assert "configured-clip" in selected.reason


def test_selector_rejects_missing_configured_clip(tmp_path: Path) -> None:
    audio = tmp_path / "neutral.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "available",
                "wav_path": str(audio),
                "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "text": "可用参考",
                "emotion": "neutral",
                "duration_seconds": 6.0,
                "reviewed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        ReferenceSelector(manifest, preferred_clip_ids={"neutral": "missing"}).select(
            plan_for("neutral")
        )
    except ValueError as exc:
        assert "configured reference clip" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing configured reference clip was silently ignored")


def test_selector_applies_scene_cluster_preference(tmp_path: Path) -> None:
    conversational = tmp_path / "z-conversational.wav"
    whisper = tmp_path / "a-whisper.wav"
    conversational.write_bytes(b"conversational")
    whisper.write_bytes(b"whisper")
    rows = [
        {
            "wav_path": str(conversational),
            "sha256": hashlib.sha256(conversational.read_bytes()).hexdigest(),
            "text": "普通参考",
            "emotion": "neutral",
            "cluster": "conversational",
            "duration_seconds": 6.0,
            "reviewed": True,
        },
        {
            "wav_path": str(whisper),
            "sha256": hashlib.sha256(whisper.read_bytes()).hexdigest(),
            "text": "耳语参考",
            "emotion": "neutral",
            "cluster": "whisper",
            "duration_seconds": 6.0,
            "reviewed": True,
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )

    selected = ReferenceSelector(manifest, preferred_clusters=("whisper", "calm_soft")).select(
        plan_for("neutral")
    )

    assert selected.audio_path == whisper
    assert "scene-cluster:whisper" in selected.reason
