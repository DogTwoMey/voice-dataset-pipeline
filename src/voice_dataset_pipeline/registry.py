"""Atomic, workspace-local registry for trained character voice models."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import Field, model_validator

from .models import StrictModel, utc_now


class VoiceModelRecord(StrictModel):
    name: str = Field(min_length=1)
    persona: str = ""
    backend: str = "gpt-sovits"
    version: str = "v2ProPlus"
    repository: Path
    python: Path | None = None
    gpt_weights: Path | None = None
    sovits_weights: Path | None = None
    gpt_weights_sha256: str = ""
    sovits_weights_sha256: str = ""
    reference_manifest: Path
    reference_manifest_sha256: str = ""
    dataset_fingerprint: str = ""
    provider_commit: str = ""
    provider_dirty_sha256: str = ""
    provider_code_sha256: str = ""
    provider_assets_sha256: dict[str, str] = Field(default_factory=dict)
    vc_backend: str = "none"
    vc_repository: Path | None = None
    vc_python: Path | None = None
    vc_model: Path | None = None
    vc_index: Path | None = None
    vc_model_sha256: str = ""
    vc_index_sha256: str = ""
    vc_provider_commit: str = ""
    vc_provider_dirty_sha256: str = ""
    vc_provider_code_sha256: str = ""
    vc_provider_assets_sha256: dict[str, str] = Field(default_factory=dict)
    active: bool = False
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_artifacts(self) -> VoiceModelRecord:
        if self.backend == "gpt-sovits" and not (self.gpt_weights and self.sovits_weights):
            raise ValueError("GPT-SoVITS registry entry requires GPT and SoVITS weights")
        if self.vc_backend == "rvc" and not all(
            (self.vc_repository, self.vc_python, self.vc_model, self.vc_index)
        ):
            raise ValueError("RVC postprocessing requires repository, Python, model and index")
        return self

    def verify_synthesis_integrity(self) -> None:
        assert self.gpt_weights is not None
        assert self.sovits_weights is not None
        verify_provider_snapshot(
            label=f"GPT-SoVITS model '{self.name}'",
            repository=self.repository,
            commit=self.provider_commit,
            dirty_sha256=self.provider_dirty_sha256,
            code_sha256=self.provider_code_sha256,
            artifacts=(
                ("GPT weights", self.gpt_weights, self.gpt_weights_sha256),
                ("SoVITS weights", self.sovits_weights, self.sovits_weights_sha256),
                (
                    "reference manifest",
                    self.reference_manifest,
                    self.reference_manifest_sha256,
                ),
            ),
            provider_assets=tuple(
                (name, path, self.provider_assets_sha256.get(name, ""))
                for name, path in _gpt_provider_assets(self).items()
            ),
        )

    def verify_rvc_integrity(self) -> None:
        if self.vc_backend != "rvc" or not all((self.vc_repository, self.vc_model, self.vc_index)):
            raise ValueError(f"model '{self.name}' does not have a complete RVC registration")
        assert self.vc_repository is not None
        assert self.vc_model is not None
        assert self.vc_index is not None
        verify_provider_snapshot(
            label=f"RVC model '{self.name}'",
            repository=self.vc_repository,
            commit=self.vc_provider_commit,
            dirty_sha256=self.vc_provider_dirty_sha256,
            code_sha256=self.vc_provider_code_sha256,
            artifacts=(
                ("RVC model", self.vc_model, self.vc_model_sha256),
                ("RVC index", self.vc_index, self.vc_index_sha256),
            ),
            provider_assets=tuple(
                (name, path, self.vc_provider_assets_sha256.get(name, ""))
                for name, path in _rvc_provider_assets(self.vc_repository).items()
            ),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return _sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"provider asset does not exist: {resolved}")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"provider asset directory is empty: {resolved}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_python_tree(repository: Path) -> str:
    resolved = repository.expanduser().resolve()
    excluded = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "assets",
        "logs",
        "outputs",
        "pretrained_models",
    }
    files: list[Path] = []
    for directory, directories, names in os.walk(resolved):
        directories[:] = sorted(name for name in directories if name.lower() not in excluded)
        root = Path(directory)
        files.extend(root / name for name in sorted(names) if name.lower().endswith(".py"))
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _gpt_provider_assets(record: VoiceModelRecord) -> dict[str, Path]:
    root = record.repository.expanduser().resolve() / "GPT_SoVITS" / "pretrained_models"
    assets = {
        "bert": root / "chinese-roberta-wwm-ext-large",
        "g2pw": record.repository.expanduser().resolve() / "GPT_SoVITS" / "text" / "G2PWModel",
        "hubert": root / "chinese-hubert-base",
        "language_detector": root / "fast_langdetect",
    }
    if "Pro" in record.version:
        assets["sv"] = root / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
    if record.version == "v3":
        assets["vocoder"] = root / "models--nvidia--bigvgan_v2_24khz_100band_256x"
    elif record.version == "v4":
        assets["vocoder"] = root / "gsv-v4-pretrained" / "vocoder.pth"
    return assets


def _rvc_provider_assets(repository: Path) -> dict[str, Path]:
    root = repository.expanduser().resolve() / "assets"
    return {
        "hubert": root / "hubert_base",
        "rmvpe": root / "rmvpe" / "rmvpe.pt",
    }


def _hash_provider_assets(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256_tree(path) for name, path in sorted(paths.items())}


def _git_state(repository: Path, label: str) -> tuple[str, str]:
    repository = repository.expanduser().resolve()
    base = [
        "git",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
    ]
    head = subprocess.run(
        [*base, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if head.returncode != 0 or not head.stdout.strip():
        raise ValueError(f"{label} is not a verifiable Git checkout: {repository}")
    diff = subprocess.run(
        [*base, "diff", "--binary", "HEAD", "--"],
        check=False,
        capture_output=True,
        shell=False,
    )
    if diff.returncode != 0:
        raise ValueError(f"cannot fingerprint tracked changes in {label}: {repository}")
    return head.stdout.strip(), hashlib.sha256(diff.stdout).hexdigest()


def verify_provider_snapshot(
    *,
    label: str,
    repository: Path,
    commit: str,
    dirty_sha256: str,
    code_sha256: str,
    artifacts: Sequence[tuple[str, Path, str]],
    provider_assets: Sequence[tuple[str, Path, str]],
) -> None:
    missing: list[str] = []
    if not commit:
        missing.append("provider commit")
    if not dirty_sha256:
        missing.append("provider dirty fingerprint")
    if not code_sha256:
        missing.append("provider Python source fingerprint")
    missing.extend(f"{name} SHA-256" for name, _, value in artifacts if not value)
    missing.extend(f"{name} asset SHA-256" for name, _, value in provider_assets if not value)
    if missing:
        raise ValueError(
            f"{label} registry entry lacks integrity metadata ({', '.join(missing)}); "
            "re-register it with `voice-dataset model register`"
        )
    for name, path, expected in artifacts:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{name} does not exist: {resolved}")
        actual = _sha256_file(resolved)
        if actual != expected:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected}, got {actual}; "
                "re-register only after verifying the artifact"
            )
    actual_code = _sha256_python_tree(repository)
    if actual_code != code_sha256:
        raise ValueError(
            f"{label} Python source fingerprint mismatch; "
            "restore the checkout or re-register intentionally"
        )
    for name, path, expected in provider_assets:
        actual = _sha256_tree(path)
        if actual != expected:
            raise ValueError(
                f"{label} {name} asset SHA-256 mismatch: expected {expected}, got {actual}; "
                "restore the provider asset or re-register intentionally"
            )
    actual_commit, actual_dirty = _git_state(repository, label)
    if actual_commit != commit:
        raise ValueError(
            f"{label} HEAD mismatch: registered {commit}, current {actual_commit}; "
            "restore the checkout or re-register intentionally"
        )
    if actual_dirty != dirty_sha256:
        raise ValueError(
            f"{label} tracked dirty fingerprint mismatch; "
            "restore the checkout or re-register intentionally"
        )


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def list(self) -> list[VoiceModelRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model registry: {self.path}") from exc
        return [VoiceModelRecord.model_validate(row) for row in raw.get("models", [])]

    def get(self, name: str | None = None) -> VoiceModelRecord:
        records = self.list()
        matches = (
            [row for row in records if row.name == name]
            if name
            else [row for row in records if row.active]
        )
        if not matches:
            requested = name or "active model"
            raise ValueError(f"model is not registered: {requested}")
        return matches[-1]

    def register(self, record: VoiceModelRecord, *, activate: bool = False) -> VoiceModelRecord:
        self._validate_paths(record)
        sealed = self._seal_integrity(record)
        current = self.list()
        was_active = any(row.name == record.name and row.active for row in current)
        records = [row for row in current if row.name != record.name]
        selected = sealed.model_copy(update={"active": activate or record.active or was_active})
        if selected.active:
            records = [row.model_copy(update={"active": False}) for row in records]
        records.append(selected)
        self._write(records)
        return selected

    def activate(self, name: str) -> VoiceModelRecord:
        records = self.list()
        if not any(row.name == name for row in records):
            raise ValueError(f"model is not registered: {name}")
        updated = [row.model_copy(update={"active": row.name == name}) for row in records]
        self._write(updated)
        return next(row for row in updated if row.name == name)

    @staticmethod
    def _validate_paths(record: VoiceModelRecord) -> None:
        if record.backend == "gpt-sovits" and record.python is None:
            raise ValueError("GPT-SoVITS registry entry requires its provider Python")
        for label, path in (
            ("provider repository", record.repository),
            ("provider Python", record.python),
            ("GPT weights", record.gpt_weights),
            ("SoVITS weights", record.sovits_weights),
            ("reference manifest", record.reference_manifest),
            ("VC repository", record.vc_repository),
            ("VC Python", record.vc_python),
            ("VC model", record.vc_model),
            ("VC index", record.vc_index),
        ):
            if path is not None and not path.expanduser().resolve().exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    @staticmethod
    def _seal(expected: str, actual: str, label: str) -> str:
        if expected and expected != actual:
            raise ValueError(
                f"{label} changed before registration: expected {expected}, got {actual}"
            )
        return actual

    @classmethod
    def _seal_integrity(cls, record: VoiceModelRecord) -> VoiceModelRecord:
        assert record.gpt_weights is not None
        assert record.sovits_weights is not None
        head, dirty = _git_state(record.repository, "GPT-SoVITS provider")
        updates: dict[str, object] = {
            "provider_commit": cls._seal(record.provider_commit, head, "provider HEAD"),
            "provider_dirty_sha256": cls._seal(
                record.provider_dirty_sha256,
                dirty,
                "provider tracked dirty fingerprint",
            ),
            "provider_code_sha256": cls._seal(
                record.provider_code_sha256,
                _sha256_python_tree(record.repository),
                "provider Python source fingerprint",
            ),
            "gpt_weights_sha256": cls._seal(
                record.gpt_weights_sha256,
                _sha256_file(record.gpt_weights),
                "GPT weights SHA-256",
            ),
            "sovits_weights_sha256": cls._seal(
                record.sovits_weights_sha256,
                _sha256_file(record.sovits_weights),
                "SoVITS weights SHA-256",
            ),
            "reference_manifest_sha256": cls._seal(
                record.reference_manifest_sha256,
                _sha256_file(record.reference_manifest),
                "reference manifest SHA-256",
            ),
        }
        provider_assets = _hash_provider_assets(_gpt_provider_assets(record))
        if record.provider_assets_sha256 and record.provider_assets_sha256 != provider_assets:
            raise ValueError(
                "provider inference assets changed before registration: "
                f"expected {record.provider_assets_sha256}, got {provider_assets}"
            )
        updates["provider_assets_sha256"] = provider_assets
        if record.vc_backend == "rvc":
            assert record.vc_repository is not None
            assert record.vc_model is not None
            assert record.vc_index is not None
            vc_head, vc_dirty = _git_state(record.vc_repository, "RVC provider")
            updates.update(
                {
                    "vc_provider_commit": cls._seal(
                        record.vc_provider_commit, vc_head, "RVC provider HEAD"
                    ),
                    "vc_provider_dirty_sha256": cls._seal(
                        record.vc_provider_dirty_sha256,
                        vc_dirty,
                        "RVC provider tracked dirty fingerprint",
                    ),
                    "vc_provider_code_sha256": cls._seal(
                        record.vc_provider_code_sha256,
                        _sha256_python_tree(record.vc_repository),
                        "RVC provider Python source fingerprint",
                    ),
                    "vc_model_sha256": cls._seal(
                        record.vc_model_sha256,
                        _sha256_file(record.vc_model),
                        "RVC model SHA-256",
                    ),
                    "vc_index_sha256": cls._seal(
                        record.vc_index_sha256,
                        _sha256_file(record.vc_index),
                        "RVC index SHA-256",
                    ),
                }
            )
            vc_provider_assets = _hash_provider_assets(_rvc_provider_assets(record.vc_repository))
            if (
                record.vc_provider_assets_sha256
                and record.vc_provider_assets_sha256 != vc_provider_assets
            ):
                raise ValueError(
                    "RVC provider inference assets changed before registration: "
                    f"expected {record.vc_provider_assets_sha256}, got {vc_provider_assets}"
                )
            updates["vc_provider_assets_sha256"] = vc_provider_assets
        return record.model_copy(update=updates)

    def _write(self, records: list[VoiceModelRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                {"version": 3, "models": [row.model_dump(mode="json") for row in records]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        handle, name = tempfile.mkstemp(dir=self.path.parent, prefix=".registry-", suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
