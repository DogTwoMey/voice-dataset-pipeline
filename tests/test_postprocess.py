from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_dataset_pipeline.errors import ExternalToolError
from voice_dataset_pipeline.postprocess import (
    AudioArtifactPlan,
    RVCPostprocessor,
    SoXMasterer,
    _unpack_rvc_payload,
    sox_profiles,
)


def _provider(tmp_path: Path) -> RVCPostprocessor:
    repository = tmp_path / "rvc"
    repository.mkdir()
    python = tmp_path / "python.exe"
    model = tmp_path / "voice.pth"
    index = tmp_path / "voice.index"
    for path in (python, model, index):
        path.write_bytes(b"provider")
    return RVCPostprocessor(
        repository=repository,
        python=python,
        model=model,
        index=index,
    )


def test_convert_publishes_only_completed_output(tmp_path: Path, monkeypatch) -> None:
    processor = _provider(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    destination = tmp_path / "final.wav"

    def fake_run(argv, **_kwargs):
        assert Path(argv[1]).name == "_provider_worker.py"
        assert argv[2] == "rvc"
        output = Path(argv[argv.index("--output") + 1])
        assert output != destination
        output.write_bytes(b"0" * 64)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "voice_dataset_pipeline.postprocess.verify_provider_snapshot", lambda **_kwargs: None
    )
    assert processor.convert(source, destination) == destination
    assert destination.read_bytes() == b"0" * 64
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_convert_removes_partial_output_after_failure(tmp_path: Path, monkeypatch) -> None:
    processor = _provider(tmp_path)
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    destination = tmp_path / "final.wav"

    def fake_run(argv, **_kwargs):
        Path(argv[argv.index("--output") + 1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(argv, 1, "", "bad model")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "voice_dataset_pipeline.postprocess.verify_provider_snapshot", lambda **_kwargs: None
    )
    with pytest.raises(ValueError, match="bad model"):
        processor.convert(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_rvc_failure_payload_preserves_provider_status() -> None:
    with pytest.raises(ValueError, match="provider diagnostic"):
        _unpack_rvc_payload("provider diagnostic", None)


def test_audio_artifact_plan_preserves_each_stage(tmp_path: Path) -> None:
    output = tmp_path / "voice.wav"

    raw = AudioArtifactPlan.build(output, use_rvc=False, use_sox=False)
    sox = AudioArtifactPlan.build(output, use_rvc=False, use_sox=True)
    rvc = AudioArtifactPlan.build(output, use_rvc=True, use_sox=False)
    both = AudioArtifactPlan.build(output, use_rvc=True, use_sox=True)

    assert raw.sovits == output.resolve()
    assert sox.sovits.name == "voice.sovits.wav"
    assert sox.mastering_input == sox.sovits
    assert rvc.rvc == output.resolve()
    assert both.sovits.name == "voice.sovits.wav"
    assert both.rvc is not None and both.rvc.name == "voice.rvc.wav"
    assert both.mastering_input == both.rvc


def test_audio_artifact_plan_fails_before_overwriting_any_stage(tmp_path: Path) -> None:
    plan = AudioArtifactPlan.build(tmp_path / "voice.wav", use_rvc=True, use_sox=True)
    plan.sovits.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="voice.sovits.wav"):
        plan.preflight()


def test_audio_artifact_plan_rejects_non_wav_before_rendering(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must use the .wav extension"):
        AudioArtifactPlan.build(tmp_path / "voice.mp3", use_rvc=False, use_sox=True)


def test_sox_masterer_uses_scene_profile_and_atomic_output(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "sox_ng.exe"
    binary.write_bytes(b"binary")
    source = tmp_path / "source.wav"
    destination = tmp_path / "final.wav"
    sf.write(source, np.linspace(-0.2, 0.2, 3200, dtype=np.float32), 32000, subtype="PCM_16")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        output = Path(argv[argv.index("-b") + 2])
        values, sample_rate = sf.read(source, dtype="float32")
        sf.write(output, values * 2, sample_rate, subtype="PCM_16")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    processor = SoXMasterer(binary=binary, profile="speech")

    assert processor.process(source, destination) == destination.resolve()
    assert captured["kwargs"]["shell"] is False
    assert captured["argv"][:4] == [str(binary.resolve()), "-R", "-D", "-V1"]
    assert captured["argv"][-4:] == ["gain", "-n", "-3", "dither"]
    assert sf.info(destination).frames == sf.info(source).frames
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_sox_masterer_removes_partial_output_after_failure(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "sox_ng.exe"
    binary.write_bytes(b"binary")
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(320, dtype=np.float32), 32000, subtype="PCM_16")
    destination = tmp_path / "final.wav"

    def fake_run(argv, **_kwargs):
        Path(argv[argv.index("-b") + 2]).write_bytes(b"partial")
        return subprocess.CompletedProcess(argv, 2, "", "invalid effect")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ExternalToolError, match="invalid effect"):
        SoXMasterer(binary=binary, profile="asmr").process(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_sox_masterer_rejects_success_exit_with_clipping_warning(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "sox_ng.exe"
    binary.write_bytes(b"binary")
    source = tmp_path / "source.wav"
    destination = tmp_path / "final.wav"
    sf.write(source, np.zeros(320, dtype=np.float32), 32000, subtype="PCM_16")

    def fake_run(argv, **_kwargs):
        output = Path(argv[argv.index("-b") + 2])
        sf.write(output, np.zeros(320, dtype=np.float32), 32000, subtype="PCM_16")
        return subprocess.CompletedProcess(
            argv,
            0,
            "",
            "sox WARN gain: gain clipped 4 samples; decrease volume?",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ExternalToolError, match="clipped samples"):
        SoXMasterer(binary=binary, profile="speech").process(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_sox_masterer_rejects_binary_output_with_wrong_bit_depth(
    tmp_path: Path, monkeypatch
) -> None:
    binary = tmp_path / "sox_ng.exe"
    binary.write_bytes(b"binary")
    source = tmp_path / "source.wav"
    destination = tmp_path / "final.wav"
    sf.write(source, np.zeros(320, dtype=np.float32), 32000, subtype="PCM_16")

    def fake_run(argv, **_kwargs):
        output = Path(argv[argv.index("-b") + 2])
        sf.write(output, np.zeros(320, dtype=np.float32), 32000, subtype="PCM_16")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ExternalToolError, match="expected PCM_24, got PCM_16"):
        SoXMasterer(binary=binary, profile="speech", output_bits=24).process(source, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_sox_profiles_cover_all_scenes() -> None:
    assert sox_profiles() == ("speech", "singing", "audiobook", "asmr", "stage")
