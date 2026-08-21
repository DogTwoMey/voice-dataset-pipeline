from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from voice_dataset_pipeline.emotion import EmotionPlan, RuleBasedEmotionAnalyzer
from voice_dataset_pipeline.references import ReferenceChoice
from voice_dataset_pipeline.registry import VoiceModelRecord
from voice_dataset_pipeline.synthesis import GPTSoVITSRuntime, SynthesisService


class _FakeRuntime:
    def __init__(self) -> None:
        self.requests = []

    def synthesize(self, **kwargs):
        self.requests.append(kwargs)
        output = Path(kwargs["output"])
        output.write_bytes(b"wave")
        return 32000, output


class _CountingAnalyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, _text: str) -> EmotionPlan:
        self.calls += 1
        return EmotionPlan(emotion="neutral")


def test_service_accepts_explicit_reference(tmp_path: Path) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    python = tmp_path / "python.exe"
    gpt = tmp_path / "gpt.ckpt"
    sovits = tmp_path / "sovits.pth"
    manifest = tmp_path / "manifest.jsonl"
    reference = tmp_path / "reference.wav"
    for path in (python, gpt, sovits, manifest, reference):
        path.write_bytes(b"x")
    service = SynthesisService(
        VoiceModelRecord(
            name="test",
            repository=repository,
            python=python,
            gpt_weights=gpt,
            sovits_weights=sovits,
            reference_manifest=manifest,
        ),
        RuleBasedEmotionAnalyzer(),
    )
    service.runtime = _FakeRuntime()
    result = service.synthesize(
        "你好",
        tmp_path / "out.wav",
        reference_audio=reference,
        reference_text="参考台词",
    )
    assert result.output.is_file()
    assert result.reference.reason == "explicit"


def test_service_applies_validated_persona_emotion_override(tmp_path: Path) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "python": "python.exe",
            "gpt": "gpt.ckpt",
            "sovits": "sovits.pth",
            "manifest": "manifest.jsonl",
            "reference": "reference.wav",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"x")
    service = SynthesisService(
        VoiceModelRecord(
            name="test",
            repository=repository,
            python=paths["python"],
            gpt_weights=paths["gpt"],
            sovits_weights=paths["sovits"],
            reference_manifest=paths["manifest"],
        ),
        RuleBasedEmotionAnalyzer(),
        emotion_overrides={
            "neutral": {"top_k": 20, "top_p": 0.96, "temperature": 1.0, "pace": 1.04}
        },
    )
    runtime = _FakeRuntime()
    service.runtime = runtime

    result = service.synthesize(
        "你好",
        tmp_path / "out.wav",
        reference_audio=paths["reference"],
        reference_text="参考台词",
        emotion="neutral",
    )

    assert result.emotion.top_k == 20
    assert result.emotion.top_p == 0.96
    assert result.emotion.temperature == 1.0
    assert result.emotion.pace == 1.04
    assert runtime.requests[0]["plan"] == result.emotion


def test_service_revalidates_emotion_override(tmp_path: Path) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "python": "python.exe",
            "gpt": "gpt.ckpt",
            "sovits": "sovits.pth",
            "manifest": "manifest.jsonl",
            "reference": "reference.wav",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"x")
    service = SynthesisService(
        VoiceModelRecord(
            name="test",
            repository=repository,
            python=paths["python"],
            gpt_weights=paths["gpt"],
            sovits_weights=paths["sovits"],
            reference_manifest=paths["manifest"],
        ),
        RuleBasedEmotionAnalyzer(),
        emotion_overrides={"neutral": {"pace": 0}},
    )

    with pytest.raises(ValueError, match="greater than or equal to 0.5"):
        service.synthesize(
            "你好",
            tmp_path / "out.wav",
            reference_audio=paths["reference"],
            reference_text="参考台词",
            emotion="neutral",
        )


def test_service_applies_scene_tone_and_scene_reference_pin(tmp_path: Path) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    python = tmp_path / "python.exe"
    gpt = tmp_path / "gpt.ckpt"
    sovits = tmp_path / "sovits.pth"
    speech = tmp_path / "speech.wav"
    whisper = tmp_path / "whisper.wav"
    for path in (python, gpt, sovits, speech, whisper):
        path.write_bytes(path.name.encode())
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "clip_id": "speech",
            "wav_path": str(speech),
            "sha256": hashlib.sha256(speech.read_bytes()).hexdigest(),
            "text": "普通参考",
            "emotion": "neutral",
            "cluster": "conversational",
            "duration_seconds": 6.0,
            "reviewed": True,
        },
        {
            "clip_id": "whisper",
            "wav_path": str(whisper),
            "sha256": hashlib.sha256(whisper.read_bytes()).hexdigest(),
            "text": "耳语参考",
            "emotion": "neutral",
            "cluster": "whisper",
            "duration_seconds": 6.0,
            "reviewed": True,
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    service = SynthesisService(
        VoiceModelRecord(
            name="test",
            repository=repository,
            python=python,
            gpt_weights=gpt,
            sovits_weights=sovits,
            reference_manifest=manifest,
        ),
        RuleBasedEmotionAnalyzer(),
        preferred_reference_clip_ids={"neutral": "speech"},
        scene_reference_clip_ids={"asmr": {"neutral": "whisper"}},
        emotion_overrides={
            "neutral": {"top_k": 20, "top_p": 0.96, "temperature": 1.0, "pace": 1.04}
        },
    )
    runtime = _FakeRuntime()
    service.runtime = runtime

    result = service.synthesize("你好", tmp_path / "out.wav", emotion="neutral", scene="asmr")

    assert result.scene == "asmr"
    assert result.reference.audio_path == whisper
    assert result.emotion.pace == pytest.approx(0.97)
    assert result.emotion.top_k == 18
    assert result.emotion.top_p == pytest.approx(0.94)
    assert result.emotion.temperature == pytest.approx(0.92)


def test_service_reuses_one_emotion_plan_across_scenes(tmp_path: Path) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    python = tmp_path / "python.exe"
    gpt = tmp_path / "gpt.ckpt"
    sovits = tmp_path / "sovits.pth"
    manifest = tmp_path / "manifest.jsonl"
    reference = tmp_path / "reference.wav"
    for path in (python, gpt, sovits, manifest, reference):
        path.write_bytes(b"x")
    analyzer = _CountingAnalyzer()
    service = SynthesisService(
        VoiceModelRecord(
            name="test",
            repository=repository,
            python=python,
            gpt_weights=gpt,
            sovits_weights=sovits,
            reference_manifest=manifest,
        ),
        analyzer,
    )
    service.runtime = _FakeRuntime()

    base_plan = service.resolve_emotion_plan("同一台词")
    service.synthesize(
        "同一台词",
        tmp_path / "speech.wav",
        reference_audio=reference,
        reference_text="参考台词",
        scene="speech",
        base_plan=base_plan,
    )
    service.synthesize(
        "同一台词",
        tmp_path / "stage.wav",
        reference_audio=reference,
        reference_text="参考台词",
        scene="stage",
        base_plan=base_plan,
    )

    assert analyzer.calls == 1


def test_runtime_uses_registered_provider_python(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    python = tmp_path / "provider-python.exe"
    gpt = tmp_path / "gpt.ckpt"
    sovits = tmp_path / "sovits.pth"
    manifest = tmp_path / "manifest.jsonl"
    reference = tmp_path / "reference.wav"
    for path in (python, gpt, sovits, manifest, reference):
        path.write_bytes(b"x")
    record = VoiceModelRecord(
        name="test",
        repository=repository,
        python=python,
        gpt_weights=gpt,
        sovits_weights=sovits,
        reference_manifest=manifest,
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        payload = json.loads(kwargs["input"])
        Path(payload["output"]).write_bytes(b"R" * 45)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout='provider log\n{"sample_rate":32000,"frames":100}\n',
            stderr="",
        )

    monkeypatch.setattr("voice_dataset_pipeline.synthesis.subprocess.run", fake_run)
    monkeypatch.setattr(VoiceModelRecord, "verify_synthesis_integrity", lambda self: None)
    output = tmp_path / "output.wav"
    rate, destination = GPTSoVITSRuntime(
        record,
        text_split_method="cut0",
        fragment_interval=0.3,
    ).synthesize(
        text="你好",
        reference=ReferenceChoice(reference, "参考台词", "neutral", 1.0, "test"),
        plan=EmotionPlan(),
        output=output,
        seed=2333,
    )

    assert captured["argv"][0] == str(python.resolve())
    assert Path(captured["argv"][1]).name == "_provider_worker.py"
    assert captured["argv"][2] == "gpt-sovits"
    assert captured["kwargs"]["cwd"] == repository.resolve()
    assert captured["kwargs"]["shell"] is False
    pythonpath = captured["kwargs"]["env"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath == [str(repository.resolve()), str(repository.resolve() / "GPT_SoVITS")]
    assert str(Path(__file__).resolve().parents[1] / "src") not in pythonpath
    payload = json.loads(captured["kwargs"]["input"])
    assert payload["seed"] == 2333
    assert payload["text_split_method"] == "cut0"
    assert payload["fragment_interval"] == 0.3
    assert rate == 32000
    assert destination == output.resolve()
    assert output.read_bytes() == b"R" * 45
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_runtime_removes_provider_partial_after_failure(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "provider"
    repository.mkdir()
    paths = {
        name: tmp_path / filename
        for name, filename in {
            "python": "provider-python.exe",
            "gpt": "gpt.ckpt",
            "sovits": "sovits.pth",
            "manifest": "manifest.jsonl",
            "reference": "reference.wav",
        }.items()
    }
    for path in paths.values():
        path.write_bytes(b"x")
    record = VoiceModelRecord(
        name="test",
        repository=repository,
        python=paths["python"],
        gpt_weights=paths["gpt"],
        sovits_weights=paths["sovits"],
        reference_manifest=paths["manifest"],
    )

    def fake_run(argv, **kwargs):
        Path(json.loads(kwargs["input"])["output"]).write_bytes(b"partial")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="provider failed")

    monkeypatch.setattr("voice_dataset_pipeline.synthesis.subprocess.run", fake_run)
    monkeypatch.setattr(VoiceModelRecord, "verify_synthesis_integrity", lambda self: None)
    output = tmp_path / "output.wav"
    with pytest.raises(ValueError, match="provider failed"):
        GPTSoVITSRuntime(record).synthesize(
            text="你好",
            reference=ReferenceChoice(paths["reference"], "参考台词", "neutral", 1.0, "test"),
            plan=EmotionPlan(),
            output=output,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))
