from __future__ import annotations

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
    def synthesize(self, **kwargs):
        output = Path(kwargs["output"])
        output.write_bytes(b"wave")
        return 32000, output


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
    rate, destination = GPTSoVITSRuntime(record).synthesize(
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
    assert json.loads(captured["kwargs"]["input"])["seed"] == 2333
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
