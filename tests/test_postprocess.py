from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from voice_dataset_pipeline.postprocess import RVCPostprocessor, _unpack_rvc_payload


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
