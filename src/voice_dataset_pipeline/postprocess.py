"""Optional RVC conversion in its own pinned provider environment."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import ExternalToolError
from .registry import verify_provider_snapshot
from .scenes import SceneName

_SOX_EFFECTS: Mapping[SceneName, tuple[str, ...]] = {
    SceneName.SPEECH: ("gain", "-n", "-3"),
    SceneName.SINGING: ("highpass", "30", "0.707q", "gain", "-n", "-4"),
    SceneName.AUDIOBOOK: (
        "highpass",
        "40",
        "0.707q",
        "equalizer",
        "180",
        "0.8q",
        "0.5",
        "equalizer",
        "6500",
        "1.1q",
        "-0.5",
        "gain",
        "-n",
        "-3.5",
    ),
    SceneName.ASMR: ("gain", "-n", "-6"),
    SceneName.STAGE: (
        "highpass",
        "65",
        "0.707q",
        "equalizer",
        "2800",
        "1.0q",
        "0.8",
        "gain",
        "-n",
        "-2.5",
    ),
}


@dataclass(frozen=True, slots=True)
class AudioArtifactPlan:
    final: Path
    sovits: Path
    rvc: Path | None
    use_sox: bool

    @classmethod
    def build(
        cls,
        output: str | Path,
        *,
        use_rvc: bool,
        use_sox: bool,
    ) -> AudioArtifactPlan:
        final = Path(output).expanduser().resolve()
        if final.suffix.lower() != ".wav":
            raise ValueError("synthesis output must use the .wav extension")
        sovits = (
            final.with_name(f"{final.stem}.sovits{final.suffix}") if use_rvc or use_sox else final
        )
        rvc = None
        if use_rvc:
            rvc = final.with_name(f"{final.stem}.rvc{final.suffix}") if use_sox else final
        return cls(final=final, sovits=sovits, rvc=rvc, use_sox=use_sox)

    @property
    def mastering_input(self) -> Path:
        return self.rvc or self.sovits

    def paths(self) -> tuple[Path, ...]:
        ordered = (self.sovits, self.rvc, self.final)
        return tuple(dict.fromkeys(path for path in ordered if path is not None))

    def preflight(self) -> None:
        conflicts = [path for path in self.paths() if path.exists()]
        if conflicts:
            joined = ", ".join(str(path) for path in conflicts)
            raise FileExistsError(f"audio output already exists: {joined}")


class SoXMasterer:
    """Apply a fixed, auditable scene preset through a SoX-compatible executable."""

    def __init__(
        self,
        *,
        binary: str | Path = "sox",
        profile: SceneName | str = SceneName.SPEECH,
        output_bits: int = 16,
    ) -> None:
        self.binary = _resolve_executable(binary)
        self.profile = (
            profile if isinstance(profile, SceneName) else SceneName(profile.strip().lower())
        )
        if output_bits not in {16, 24}:
            raise ValueError("SoX output_bits must be 16 or 24")
        self.output_bits = output_bits

    @property
    def effects(self) -> tuple[str, ...]:
        effects = ("gain", "-h", *_SOX_EFFECTS[self.profile])
        return (*effects, "dither") if self.output_bits == 16 else effects

    def process(self, input_audio: str | Path, output_audio: str | Path) -> Path:
        source = Path(input_audio).expanduser().resolve()
        destination = Path(output_audio).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"SoX input does not exist: {source}")
        if destination.exists():
            raise FileExistsError(f"SoX output already exists: {destination}")
        if destination.suffix.lower() != ".wav":
            raise ValueError("SoX mastering output must use the .wav extension")
        source_info = sf.info(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp.wav",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink(missing_ok=True)
        argv = [
            str(self.binary),
            "-R",
            "-D",
            "-V1",
            str(source),
            "-b",
            str(self.output_bits),
            str(temporary),
            *self.effects,
        ]
        try:
            try:
                result = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    shell=False,
                )
            except OSError as exc:
                raise ExternalToolError(f"cannot start SoX: {self.binary}: {exc}") from exc
            diagnostics = "\n".join(part for part in (result.stdout, result.stderr) if part)
            if result.returncode != 0:
                raise ExternalToolError(
                    f"SoX mastering failed ({result.returncode}): {diagnostics[-4000:]}"
                )
            clipped = [
                int(before or after)
                for before, after in re.findall(
                    r"(?:(\d+)\s+samples?\s+clipped|clipped\s+(\d+)\s+samples?)",
                    diagnostics,
                    flags=re.IGNORECASE,
                )
            ]
            if any(value > 0 for value in clipped):
                raise ExternalToolError(f"SoX reported clipped samples: {diagnostics[-4000:]}")
            _validate_mastered_audio(
                source_info,
                temporary,
                expected_output_bits=self.output_bits,
            )
            os.replace(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)


def sox_profiles() -> tuple[str, ...]:
    return tuple(scene.value for scene in _SOX_EFFECTS)


def _resolve_executable(binary: str | Path) -> Path:
    value = str(binary).strip()
    if not value:
        raise ValueError("SoX binary is empty")
    explicit = Path(value).expanduser()
    if explicit.is_absolute() or explicit.parent != Path("."):
        resolved = explicit.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SoX binary does not exist: {resolved}")
        return resolved
    discovered = shutil.which(value)
    if not discovered:
        raise FileNotFoundError(
            f"SoX binary '{value}' was not found; install SoX/SoX_ng or configure "
            "[postprocess.sox].binary"
        )
    return Path(discovered).resolve()


def _validate_mastered_audio(
    source_info: sf.SoundFile,
    output: Path,
    *,
    expected_output_bits: int,
) -> None:
    if not output.is_file() or output.stat().st_size <= 44:
        raise ExternalToolError("SoX did not create a valid WAV")
    output_info = sf.info(output)
    if output_info.format != "WAV":
        raise ExternalToolError(f"SoX output is not WAV: {output_info.format}")
    expected_subtype = {16: "PCM_16", 24: "PCM_24"}[expected_output_bits]
    if output_info.subtype != expected_subtype:
        raise ExternalToolError(
            f"SoX output subtype mismatch: expected {expected_subtype}, got {output_info.subtype}"
        )
    if output_info.samplerate != source_info.samplerate:
        raise ExternalToolError(
            f"SoX changed sample rate: {source_info.samplerate} -> {output_info.samplerate}"
        )
    if output_info.channels != source_info.channels:
        raise ExternalToolError(
            f"SoX changed channel count: {source_info.channels} -> {output_info.channels}"
        )
    tolerance = max(1, round(source_info.samplerate * 0.01))
    if abs(output_info.frames - source_info.frames) > tolerance:
        raise ExternalToolError(
            f"SoX changed duration: {source_info.frames} -> {output_info.frames} frames"
        )
    peak = 0.0
    for block in sf.blocks(output, blocksize=65536, dtype="float32", always_2d=True):
        if not np.isfinite(block).all():
            raise ExternalToolError("SoX output contains non-finite samples")
        if block.size:
            peak = max(peak, float(np.max(np.abs(block))))
    if peak >= 1.0:
        raise ExternalToolError(f"SoX output clips at peak {peak:.6f}")


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
