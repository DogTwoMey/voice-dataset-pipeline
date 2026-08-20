"""Optional RVC conversion in its own pinned provider environment."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .registry import verify_provider_snapshot


@dataclass(frozen=True, slots=True)
class RVCOptions:
    f0_method: str = "rmvpe"
    transpose: int = 0
    index_rate: float = 0.45
    rms_mix_rate: float = 0.25
    protect: float = 0.33


class RVCPostprocessor:
    def __init__(
        self,
        *,
        repository: str | Path,
        python: str | Path,
        model: str | Path,
        index: str | Path,
        options: RVCOptions | None = None,
        model_sha256: str = "",
        index_sha256: str = "",
        provider_commit: str = "",
        provider_dirty_sha256: str = "",
        provider_code_sha256: str = "",
        provider_assets_sha256: Mapping[str, str] | None = None,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.python = Path(python).expanduser().resolve()
        self.model = Path(model).expanduser().resolve()
        self.index = Path(index).expanduser().resolve()
        self.options = options or RVCOptions()
        self.model_sha256 = model_sha256
        self.index_sha256 = index_sha256
        self.provider_commit = provider_commit
        self.provider_dirty_sha256 = provider_dirty_sha256
        self.provider_code_sha256 = provider_code_sha256
        self.provider_assets_sha256 = dict(provider_assets_sha256 or {})
        for label, path in (
            ("RVC repository", self.repository),
            ("RVC Python", self.python),
            ("RVC model", self.model),
            ("RVC index", self.index),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    def convert(self, input_audio: str | Path, output_audio: str | Path) -> Path:
        verify_provider_snapshot(
            label="RVC postprocessor",
            repository=self.repository,
            commit=self.provider_commit,
            dirty_sha256=self.provider_dirty_sha256,
            code_sha256=self.provider_code_sha256,
            artifacts=(
                ("RVC model", self.model, self.model_sha256),
                ("RVC index", self.index, self.index_sha256),
            ),
            provider_assets=(
                (
                    "hubert",
                    self.repository / "assets" / "hubert_base",
                    self.provider_assets_sha256.get("hubert", ""),
                ),
                (
                    "rmvpe",
                    self.repository / "assets" / "rmvpe" / "rmvpe.pt",
                    self.provider_assets_sha256.get("rmvpe", ""),
                ),
            ),
        )
        source = Path(input_audio).expanduser().resolve()
        destination = Path(output_audio).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"postprocess input does not exist: {source}")
        if destination.exists():
            raise FileExistsError(f"postprocess output already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp.wav",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        worker = Path(__file__).with_name("_provider_worker.py").resolve()
        if not worker.is_file():
            raise FileNotFoundError(f"provider worker does not exist: {worker}")
        environment = os.environ.copy()
        environment["weight_root"] = str(self.model.parent)
        environment["rmvpe_root"] = str(self.repository / "assets" / "rmvpe")
        environment["index_root"] = str(self.repository / "logs")
        environment["outside_index_root"] = str(self.repository / "assets" / "indices")
        environment["PYTHONPATH"] = str(self.repository)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        argv = [
            str(self.python),
            str(worker),
            "rvc",
            "--repository",
            str(self.repository),
            "--model",
            str(self.model),
            "--index",
            str(self.index),
            "--input",
            str(source),
            "--output",
            str(temporary),
            "--f0-method",
            self.options.f0_method,
            "--transpose",
            str(self.options.transpose),
            "--index-rate",
            str(self.options.index_rate),
            "--rms-mix-rate",
            str(self.options.rms_mix_rate),
            "--protect",
            str(self.options.protect),
        ]
        try:
            result = subprocess.run(
                argv,
                cwd=self.repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout)[-4000:]
                raise ValueError(f"RVC postprocess failed ({result.returncode}): {detail}")
            if not temporary.is_file() or temporary.stat().st_size <= 44:
                raise ValueError("RVC postprocess did not create a valid WAV")
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)


def _unpack_rvc_payload(status: object, payload: object) -> tuple[int, object]:
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"RVC inference failed: {status}")
    sample_rate, waveform = payload
    if sample_rate is None or waveform is None:
        raise ValueError(f"RVC inference failed: {status}")
    return int(sample_rate), waveform
