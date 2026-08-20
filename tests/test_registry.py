from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from voice_dataset_pipeline.registry import ModelRegistry, VoiceModelRecord


def _commit(repository: Path, message: str) -> None:
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Voice Dataset Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
        check=True,
    )


def _record(tmp_path: Path, name: str) -> VoiceModelRecord:
    repository = tmp_path / f"provider-{name}"
    repository.mkdir(exist_ok=True)
    marker = repository / "tracked.txt"
    marker.write_text(name, encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    _commit(repository, "fixture")
    pretrained = repository / "GPT_SoVITS" / "pretrained_models"
    for path in (
        pretrained / "chinese-roberta-wwm-ext-large" / "model.bin",
        pretrained / "chinese-hubert-base" / "model.bin",
        repository / "GPT_SoVITS" / "text" / "G2PWModel" / "g2pW.onnx",
        pretrained / "fast_langdetect" / "lid.176.bin",
        pretrained / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset-{path.name}".encode())
    python = tmp_path / "python.exe"
    gpt = tmp_path / f"{name}.ckpt"
    sovits = tmp_path / f"{name}.pth"
    manifest = tmp_path / "manifest.jsonl"
    for path in (python, gpt, sovits, manifest):
        path.write_bytes(b"test")
    return VoiceModelRecord(
        name=name,
        repository=repository,
        python=python,
        gpt_weights=gpt,
        sovits_weights=sovits,
        reference_manifest=manifest,
    )


def test_register_and_activate_are_atomic(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models/registry.json")
    first = registry.register(_record(tmp_path, "first"), activate=True)
    registry.register(_record(tmp_path, "second"))
    assert first.gpt_weights_sha256 == hashlib.sha256(b"test").hexdigest()
    assert first.sovits_weights_sha256 == hashlib.sha256(b"test").hexdigest()
    assert first.reference_manifest_sha256 == hashlib.sha256(b"test").hexdigest()
    assert first.provider_commit
    assert first.provider_dirty_sha256 == hashlib.sha256(b"").hexdigest()
    assert set(first.provider_assets_sha256) == {
        "bert",
        "g2pw",
        "hubert",
        "language_detector",
        "sv",
    }
    assert first.provider_code_sha256 == hashlib.sha256(b"").hexdigest()
    assert registry.get().name == "first"
    registry.activate("second")
    assert registry.get().name == "second"
    assert sum(row.active for row in registry.list()) == 1
    assert registry.get().python == (tmp_path / "python.exe").resolve()


def test_reregister_preserves_active_model_without_activate_flag(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models/registry.json")
    original = _record(tmp_path, "character")
    registry.register(original, activate=True)
    updated = registry.register(original)
    assert updated.active is True
    assert registry.get().name == "character"


def test_register_requires_gpt_sovits_provider_python(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models/registry.json")
    record = _record(tmp_path, "missing-python").model_copy(update={"python": None})
    with pytest.raises(ValueError, match="provider Python"):
        registry.register(record)


def test_integrity_rejects_tampered_weights(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models/registry.json")
    sealed = registry.register(_record(tmp_path, "tampered"))
    assert sealed.gpt_weights is not None
    sealed.gpt_weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="GPT weights SHA-256 mismatch"):
        sealed.verify_synthesis_integrity()


def test_legacy_record_requires_reregistration(tmp_path: Path) -> None:
    record = _record(tmp_path, "legacy")
    with pytest.raises(ValueError, match="re-register"):
        record.verify_synthesis_integrity()


def test_integrity_rejects_provider_head_and_dirty_drift(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models/registry.json")
    sealed = registry.register(_record(tmp_path, "provider-drift"))
    marker = sealed.repository / "tracked.txt"
    marker.write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked dirty fingerprint mismatch"):
        sealed.verify_synthesis_integrity()

    _commit(sealed.repository, "provider update")
    with pytest.raises(ValueError, match="HEAD mismatch"):
        sealed.verify_synthesis_integrity()


def test_integrity_rejects_tampered_provider_asset(tmp_path: Path) -> None:
    sealed = ModelRegistry(tmp_path / "models/registry.json").register(
        _record(tmp_path, "provider-asset")
    )
    asset = sealed.repository / "GPT_SoVITS" / "text" / "G2PWModel" / "g2pW.onnx"
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="g2pw asset SHA-256 mismatch"):
        sealed.verify_synthesis_integrity()


@pytest.mark.parametrize(
    ("version", "relative"),
    [
        ("v3", "models--nvidia--bigvgan_v2_24khz_100band_256x/model.bin"),
        ("v4", "gsv-v4-pretrained/vocoder.pth"),
    ],
)
def test_registry_seals_version_specific_vocoder(
    tmp_path: Path, version: str, relative: str
) -> None:
    record = _record(tmp_path, f"provider-{version}").model_copy(update={"version": version})
    vocoder = record.repository / "GPT_SoVITS" / "pretrained_models" / relative
    vocoder.parent.mkdir(parents=True, exist_ok=True)
    vocoder.write_bytes(b"vocoder")
    sealed = ModelRegistry(tmp_path / "models/registry.json").register(record)
    assert "vocoder" in sealed.provider_assets_sha256


def test_integrity_rejects_untracked_provider_python_drift(tmp_path: Path) -> None:
    sealed = ModelRegistry(tmp_path / "models/registry.json").register(
        _record(tmp_path, "provider-code")
    )
    (sealed.repository / "untracked_override.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Python source fingerprint mismatch"):
        sealed.verify_synthesis_integrity()


def test_rvc_integrity_rejects_tampered_index(tmp_path: Path) -> None:
    base = _record(tmp_path, "with-rvc")
    rvc_repository = tmp_path / "rvc-provider"
    rvc_repository.mkdir()
    (rvc_repository / "tracked.txt").write_text("rvc", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(rvc_repository)], check=True)
    _commit(rvc_repository, "fixture")
    rvc_python = tmp_path / "rvc-python.exe"
    rvc_model = tmp_path / "voice.pth"
    rvc_index = tmp_path / "voice.index"
    for path in (rvc_python, rvc_model, rvc_index):
        path.write_bytes(b"rvc")
    for path in (
        rvc_repository / "assets" / "hubert_base" / "model.bin",
        rvc_repository / "assets" / "rmvpe" / "rmvpe.pt",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rvc-asset")
    record = base.model_copy(
        update={
            "vc_backend": "rvc",
            "vc_repository": rvc_repository,
            "vc_python": rvc_python,
            "vc_model": rvc_model,
            "vc_index": rvc_index,
        }
    )
    sealed = ModelRegistry(tmp_path / "models/registry.json").register(record)
    rvc_index.write_bytes(b"changed")
    with pytest.raises(ValueError, match="RVC index SHA-256 mismatch"):
        sealed.verify_rvc_integrity()


def test_rvc_integrity_rejects_tampered_base_asset(tmp_path: Path) -> None:
    base = _record(tmp_path, "with-rvc-asset")
    rvc_repository = tmp_path / "rvc-provider-assets"
    rvc_repository.mkdir()
    (rvc_repository / "tracked.txt").write_text("rvc", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(rvc_repository)], check=True)
    _commit(rvc_repository, "fixture")
    rvc_python = tmp_path / "rvc-asset-python.exe"
    rvc_model = tmp_path / "voice-asset.pth"
    rvc_index = tmp_path / "voice-asset.index"
    for path in (rvc_python, rvc_model, rvc_index):
        path.write_bytes(b"rvc")
    hubert = rvc_repository / "assets" / "hubert_base" / "model.bin"
    rmvpe = rvc_repository / "assets" / "rmvpe" / "rmvpe.pt"
    for path in (hubert, rmvpe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rvc-asset")
    sealed = ModelRegistry(tmp_path / "models/registry.json").register(
        base.model_copy(
            update={
                "vc_backend": "rvc",
                "vc_repository": rvc_repository,
                "vc_python": rvc_python,
                "vc_model": rvc_model,
                "vc_index": rvc_index,
            }
        )
    )
    rmvpe.write_bytes(b"changed")
    with pytest.raises(ValueError, match="rmvpe asset SHA-256 mismatch"):
        sealed.verify_rvc_integrity()
