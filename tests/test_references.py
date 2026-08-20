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
