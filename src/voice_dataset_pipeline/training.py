"""Safe command planning for external GPT-SoVITS and RVC training repositories.

The project deliberately does not vendor either trainer.  The adapters in this
module validate a reviewed dataset, derive provider-native manifests/configs,
and return an inspectable :class:`TrainingPlan`.  No provider command runs
until the caller explicitly invokes :meth:`TrainingPlan.execute`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ExternalToolError

_MISSING = object()
_SAFE_EXPERIMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TRUE = {True, "1", "true", "yes", "y", "reviewed", "approved", "accepted"}
_FALSE = {False, "0", "false", "no", "n", "excluded", "rejected"}


def _get(value: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        if key in value:
            return value[key]
    elif value is not None and hasattr(value, key):
        return getattr(value, key)
    if default is _MISSING:
        raise ConfigurationError(f"missing training configuration value: {key}")
    return default


def _nested(value: Any, *keys: str, default: Any = _MISSING) -> Any:
    current = value
    for key in keys:
        current = _get(current, key, default)
        if current is default:
            return default
    return current


def _backend_config(value: Any, backend: str) -> Any:
    training = _get(value, "training", value)
    return _get(training, backend, _get(value, backend, value))


def _option(config: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        try:
            result = _get(config, name)
        except ConfigurationError:
            continue
        if result not in (None, ""):
            return result
    if default is _MISSING:
        raise ConfigurationError(f"missing training configuration value: {'/'.join(names)}")
    return default


def _bool(value: Any, *, name: str) -> bool:
    normalized = value.strip().lower() if isinstance(value, str) else value
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {value!r}")


def _resolve(path: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    candidate = Path(os.path.expandvars(os.fspath(path))).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise ConfigurationError(f"required provider artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ConfigurationError(f"required provider directory is empty: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_snapshot(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): _config_snapshot(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_config_snapshot(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _adapter_config_snapshot(value: Any) -> dict[str, Any]:
    snapshot = _config_snapshot(value)
    if not isinstance(snapshot, dict):
        return {"value": snapshot}
    snapshot.pop("enabled", None)
    return snapshot


def _guard_existing_plan(run_dir: Path, input_fingerprint: str) -> None:
    """Reject a stale experiment before rewriting any generated input."""

    plan_path = run_dir / "training-plan.json"
    if not plan_path.exists():
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ConfigurationError(
                f"untracked training files already exist; use a new experiment: {run_dir}"
            )
        return
    try:
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        previous_input = previous["metadata"]["input_fingerprint"]
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise ConfigurationError(f"existing training plan is invalid: {plan_path}") from exc
    if previous_input != input_fingerprint:
        raise ConfigurationError(
            f"experiment input/config/provider fingerprint changed; use a new experiment: {run_dir}"
        )


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "rev-parse",
            "HEAD",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_tracked_diff_hash(repository: Path) -> str:
    """Hash staged and unstaged changes to tracked provider files."""

    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "diff",
            "--binary",
            "HEAD",
            "--",
        ],
        check=False,
        capture_output=True,
        shell=False,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else ""


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ConfigurationError("required files are missing:\n" + "\n".join(missing))


def _require_directories(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise ConfigurationError("required directories are missing:\n" + "\n".join(missing))


def _probe_python(
    python: Path,
    repository: Path,
    modules: Sequence[str],
    *,
    require_cuda: bool,
    pythonpath: Sequence[Path],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(item) for item in pythonpath)
    code = "\n".join(
        [
            "import importlib,json,sys",
            f"names={list(modules)!r}",
            "missing=[]",
            "import_errors={}",
            "for name in names:",
            " try:",
            "  importlib.import_module(name)",
            " except ModuleNotFoundError:",
            "  missing.append(name)",
            " except Exception as exc:",
            "  import_errors[name]=repr(exc)",
            "result={'executable':sys.executable,'version':sys.version.split()[0],"
            "'missing':missing,'import_errors':import_errors}",
            "try:",
            " import torch",
            " result.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),"
            "gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')",
            "except Exception as exc:",
            " result['torch_error']=repr(exc)",
            "print(json.dumps(result,ensure_ascii=False))",
        ]
    )
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=repository,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise ConfigurationError(f"target Python probe failed ({python}): {details}")
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"target Python probe returned invalid JSON: {python}") from exc
    if report.get("missing"):
        raise ConfigurationError(
            f"target Python is missing modules: {', '.join(report['missing'])}"
        )
    if report.get("import_errors"):
        details = ", ".join(f"{name}: {error}" for name, error in report["import_errors"].items())
        raise ConfigurationError(f"target Python has broken imports: {details}")
    if report.get("torch_error"):
        raise ConfigurationError(f"target Python cannot import torch: {report['torch_error']}")
    if require_cuda and not report.get("cuda_available"):
        raise ConfigurationError(f"CUDA is not available in target Python: {python}")
    return report


def _normalise_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    return _bool(value, name="manifest boolean")


@dataclass(frozen=True, slots=True)
class _TrainingItem:
    clip_id: str
    audio_path: Path
    text: str
    language: str
    speaker: str
    emotion: str
    audio_sha256: str
    frames: int
    sample_rate: int

    def serialise(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "audio_path": str(self.audio_path),
            "text": self.text,
            "language": self.language,
            "speaker": self.speaker,
            "emotion": self.emotion,
            "audio_sha256": self.audio_sha256,
            "frames": self.frames,
            "sample_rate": self.sample_rate,
        }


def _record_get(record: Any, *keys: str, default: Any = _MISSING) -> Any:
    for key in keys:
        try:
            return _get(record, key)
        except ConfigurationError:
            continue
    if default is _MISSING:
        raise ConfigurationError(f"manifest row is missing {'/'.join(keys)}")
    return default


def _load_rows(source: Path | Iterable[Any]) -> tuple[list[Any], Path | None]:
    if not isinstance(source, (str, os.PathLike, Path)):
        return list(source), None
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"training manifest does not exist: {path}")
    rows: list[Any] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ConfigurationError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else list(payload.get("items", []))
    elif path.suffix.lower() in {".tsv", ".csv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter=delimiter))
    else:
        raise ConfigurationError("training manifest must be JSONL, JSON, TSV, or CSV")
    return rows, path.parent


def _audio_info(path: Path) -> tuple[int, int]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
    except Exception as exc:
        raise ConfigurationError(f"cannot decode training WAV {path}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ConfigurationError(f"invalid or empty training WAV: {path}")
    return int(info.frames), int(info.samplerate)


def _validated_items(
    source: Path | Iterable[Any],
    *,
    require_text: bool,
    default_speaker: str,
    require_reviewed: bool,
) -> list[_TrainingItem]:
    rows, base = _load_rows(source)
    selected: list[_TrainingItem] = []
    seen_ids: set[str] = set()
    seen_basenames: set[str] = set()
    for index, row in enumerate(rows, 1):
        excluded = _normalise_bool(_record_get(row, "excluded", default=False), default=False)
        include = _normalise_bool(_record_get(row, "include", default=True), default=True)
        status = str(_record_get(row, "status", default="")).strip().lower()
        reviewed_raw = _record_get(row, "reviewed", default=None)
        reviewed = (
            _normalise_bool(reviewed_raw, default=False)
            if reviewed_raw is not None
            else status in {"reviewed", "approved", "accepted", "included"}
        )
        if excluded or not include:
            continue
        if require_reviewed and not reviewed:
            raise ConfigurationError(f"manifest row {index} is included but not reviewed")
        clip_id = str(_record_get(row, "clip_id", "id")).strip()
        if not clip_id or clip_id in seen_ids:
            raise ConfigurationError(
                f"duplicate or empty clip_id in training manifest: {clip_id!r}"
            )
        raw_audio = _record_get(row, "audio_path", "wav_path", "path")
        audio_path = _resolve(raw_audio, base=base)
        if audio_path.suffix.lower() != ".wav" or not audio_path.is_file():
            raise ConfigurationError(f"training audio must be an existing WAV: {audio_path}")
        if audio_path.name.casefold() in seen_basenames:
            raise ConfigurationError(
                "training WAV basenames must be unique because GPT-SoVITS collapses "
                f"feature paths by basename: {audio_path.name}"
            )
        text = str(_record_get(row, "text", "transcript", default="")).strip()
        if require_text and not text:
            raise ConfigurationError(f"training text is empty for {clip_id}")
        if any(character in text for character in ("\n", "\r", "|")):
            raise ConfigurationError(
                f"training text contains a forbidden newline or '|': {clip_id}"
            )
        language = str(_record_get(row, "language", default="zh")).strip().lower()
        if language not in {"zh", "ja", "en", "ko", "yue"}:
            raise ConfigurationError(f"unsupported GPT-SoVITS language {language!r}: {clip_id}")
        speaker = str(_record_get(row, "speaker", "speaker_id", default=default_speaker)).strip()
        if not speaker or any(character in speaker for character in ("\n", "\r", "|")):
            raise ConfigurationError(f"invalid speaker field for {clip_id}")
        frames, sample_rate = _audio_info(audio_path)
        actual_sha256 = _sha256_file(audio_path)
        declared_sha256 = (
            str(_record_get(row, "sha256", "audio_sha256", default="")).strip().lower()
        )
        if declared_sha256 and declared_sha256 != actual_sha256:
            raise ConfigurationError(f"audio SHA-256 mismatch for {clip_id}: {audio_path}")
        selected.append(
            _TrainingItem(
                clip_id=clip_id,
                audio_path=audio_path,
                text=text,
                language=language,
                speaker=speaker,
                emotion=str(_record_get(row, "emotion", default="unknown")).strip().lower(),
                audio_sha256=actual_sha256,
                frames=frames,
                sample_rate=sample_rate,
            )
        )
        seen_ids.add(clip_id)
        seen_basenames.add(audio_path.name.casefold())
    if not selected:
        raise ConfigurationError("reviewed training dataset is empty")
    return selected


@dataclass(slots=True)
class CommandSpec:
    """One external command with all execution context made explicit."""

    name: str
    argv: list[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    expected_files: list[Path] = field(default_factory=list)
    expected_directories: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.argv:
            raise ValueError("CommandSpec requires a name and a non-empty argv list")
        self.cwd = Path(self.cwd).resolve()
        self.argv = [str(value) for value in self.argv]
        self.env = {str(key): str(value) for key, value in self.env.items()}
        self.expected_files = [Path(value).resolve() for value in self.expected_files]
        self.expected_directories = [Path(value).resolve() for value in self.expected_directories]

    def serialise(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": self.argv,
            "cwd": str(self.cwd),
            "env": self.env,
            "shell": False,
            "expected_files": [str(path) for path in self.expected_files],
            "expected_directories": [str(path) for path in self.expected_directories],
        }


@dataclass(slots=True)
class TrainingPlan:
    """An inert, serialisable plan; execution is always an explicit operation."""

    backend: str
    experiment: str
    run_dir: Path
    fingerprint: str
    commands: list[CommandSpec]
    metadata: dict[str, Any]
    plan_manifest: Path
    artifact_manifest: Path

    def serialise(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backend": self.backend,
            "experiment": self.experiment,
            "run_dir": str(self.run_dir),
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
            "commands": [command.serialise() for command in self.commands],
            "artifact_manifest": str(self.artifact_manifest),
        }

    def write_manifest(self) -> Path:
        _atomic_json(self.plan_manifest, self.serialise())
        return self.plan_manifest

    def execute(self) -> dict[str, Any]:
        """Execute this already-inspected plan with ``shell=False``.

        Output is streamed to one log per command.  A zero exit code is not
        considered sufficient: each command's declared files/directories are
        checked before the next command is started.
        """

        current = json.loads(self.plan_manifest.read_text(encoding="utf-8"))
        if current.get("fingerprint") != self.fingerprint:
            raise ConfigurationError("training plan manifest fingerprint changed")
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        completed: list[dict[str, Any]] = []
        for command in self.commands:
            log_path = logs_dir / f"{command.name}.log"
            environment = os.environ.copy()
            environment.update(command.env)
            started = time.time()
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                result = subprocess.run(
                    command.argv,
                    cwd=command.cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    shell=False,
                )
            if result.returncode != 0:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise ExternalToolError(
                    f"{command.name} failed with exit {result.returncode}\n{tail}"
                )
            _require_files(command.expected_files)
            _require_directories(command.expected_directories)
            completed.append(
                {
                    "name": command.name,
                    "returncode": result.returncode,
                    "seconds": round(time.time() - started, 3),
                    "log": str(log_path),
                }
            )
        if not self.artifact_manifest.is_file():
            raise ExternalToolError(
                f"training completed without an artifact manifest: {self.artifact_manifest}"
            )
        artifacts = json.loads(self.artifact_manifest.read_text(encoding="utf-8"))
        for artifact in artifacts.get("artifacts", []):
            path = Path(artifact["path"]).resolve()
            if not path.is_file():
                raise ExternalToolError(f"recorded training artifact is missing: {path}")
            actual = _sha256_file(path)
            if actual != artifact.get("sha256"):
                raise ExternalToolError(f"recorded training artifact hash changed: {path}")
        result_payload = {
            "schema_version": 1,
            "backend": self.backend,
            "experiment": self.experiment,
            "fingerprint": self.fingerprint,
            "commands": completed,
            "artifacts": artifacts.get("artifacts", []),
        }
        _atomic_json(self.run_dir / "training-result.json", result_payload)
        return result_payload


class _AdapterBase:
    backend: str

    def __init__(
        self,
        config: Any,
        workspace: str | os.PathLike[str] | None = None,
        *,
        python_probe: Callable[..., dict[str, Any]] | None = _probe_python,
    ) -> None:
        self.root_config = config
        self.config = _backend_config(config, self.backend)
        if workspace is None:
            workspace_value = _option(
                self.config,
                "workspace",
                "work_dir",
                default=_nested(config, "workspace", "root", default="voice-workspace"),
            )
            workspace = workspace_value
        self.workspace = _resolve(workspace)
        self.python_probe = python_probe

    def _experiment(self, explicit: str | None) -> str:
        value = explicit or str(
            _option(self.config, "experiment", "run_name", default=f"voice_{self.backend}")
        )
        if not _SAFE_EXPERIMENT.fullmatch(value):
            raise ConfigurationError(
                "experiment must be 1-128 ASCII letters, digits, '.', '_' or '-'"
            )
        return value

    def _commit_plan(
        self,
        *,
        experiment: str,
        run_dir: Path,
        commands: list[CommandSpec],
        metadata: dict[str, Any],
        artifact_manifest: Path,
    ) -> TrainingPlan:
        fingerprint_payload = {
            "backend": self.backend,
            "experiment": experiment,
            "metadata": metadata,
            "commands": [command.serialise() for command in commands],
        }
        fingerprint = _fingerprint(fingerprint_payload)
        plan_path = run_dir / "training-plan.json"
        if plan_path.exists():
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fingerprint:
                raise ConfigurationError(
                    "experiment directory contains a different plan; "
                    f"use a new experiment: {run_dir}"
                )
        plan = TrainingPlan(
            backend=self.backend,
            experiment=experiment,
            run_dir=run_dir,
            fingerprint=fingerprint,
            commands=commands,
            metadata=metadata,
            plan_manifest=plan_path,
            artifact_manifest=artifact_manifest,
        )
        plan.write_manifest()
        return plan


class GPTSoVITSAdapter(_AdapterBase):
    """Adapter for GPT-SoVITS v2/v2Pro/v2ProPlus command-line trainers."""

    backend = "gpt_sovits"
    _MODELS = {
        "v2": {
            "s2_config": "GPT_SoVITS/configs/s2.json",
            "s2g": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
            "s2d": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
            "s1": (
                "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/"
                "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
            ),
        },
        "v2Pro": {
            "s2_config": "GPT_SoVITS/configs/s2v2Pro.json",
            "s2g": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "s2d": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
            "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        },
        "v2ProPlus": {
            "s2_config": "GPT_SoVITS/configs/s2v2ProPlus.json",
            "s2g": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
            "s2d": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
            "s1": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        },
    }

    def plan(
        self,
        manifest: Path | Iterable[Any],
        *,
        experiment: str | None = None,
    ) -> TrainingPlan:
        config = self.config
        repository = _resolve(_option(config, "repository", "root", "gpt_sovits_root"))
        python = _resolve(
            _option(config, "python", "python_exe", default=repository / ".venv/Scripts/python.exe")
        )
        experiment_name = self._experiment(experiment)
        model_version = str(_option(config, "model_version", default="v2ProPlus"))
        if model_version not in self._MODELS:
            raise ConfigurationError(f"unsupported GPT-SoVITS model_version: {model_version}")
        if not repository.is_dir() or not python.is_file():
            raise ConfigurationError("GPT-SoVITS repository or Python executable is missing")
        mapping = self._MODELS[model_version]
        paths = {name: repository / relative for name, relative in mapping.items()}
        paths.update(
            {
                "bert": repository / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                "hubert": repository / "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "g2pw": repository / "GPT_SoVITS/text/G2PWModel",
                "language_detector": repository / "GPT_SoVITS/pretrained_models/fast_langdetect",
                "sv": repository
                / "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
                "s1_config": repository / "GPT_SoVITS/configs/s1longer-v2.yaml",
            }
        )
        scripts = [
            repository / "GPT_SoVITS/prepare_datasets/1-get-text.py",
            repository / "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
            repository / "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
            repository / "GPT_SoVITS/s2_train.py",
            repository / "GPT_SoVITS/s1_train.py",
        ]
        if "Pro" in model_version:
            scripts.append(repository / "GPT_SoVITS/prepare_datasets/2-get-sv.py")
        _require_files(
            [
                python,
                paths["s2_config"],
                paths["s2g"],
                paths["s2d"],
                paths["s1"],
                paths["s1_config"],
                *scripts,
            ]
        )
        _require_directories(
            [paths["bert"], paths["hubert"], paths["g2pw"], paths["language_detector"]]
        )
        if "Pro" in model_version:
            _require_files([paths["sv"]])
        pythonpath = [
            repository,
            repository / "GPT_SoVITS",
            repository / "tools",
            repository / "tools/asr",
            Path(__file__).resolve().parents[1],
        ]
        probe_report = {}
        if self.python_probe is not None:
            probe_report = self.python_probe(
                python,
                repository,
                ["torch", "yaml", "transformers", "soundfile"],
                require_cuda=True,
                pythonpath=pythonpath,
            )
        items = _validated_items(
            manifest,
            require_text=True,
            default_speaker=str(_option(config, "speaker", default="speaker")),
            require_reviewed=_bool(
                _option(config, "require_reviewed", default=True),
                name="require_reviewed",
            ),
        )
        provider_hashes = {
            name: _sha256_tree(path)
            for name, path in paths.items()
            if name
            in {
                "s2_config",
                "s2g",
                "s2d",
                "s1",
                "bert",
                "hubert",
                "g2pw",
                "language_detector",
                "sv",
                "s1_config",
            }
            and (name != "sv" or "Pro" in model_version)
        }
        provider_hashes.update(
            {str(path.relative_to(repository)): _sha256_file(path) for path in scripts}
        )
        input_fingerprint = _fingerprint(
            {
                "dataset": [item.serialise() for item in items],
                "config": _adapter_config_snapshot(config),
                "provider": {
                    "repository": str(repository),
                    "git_head": _git_head(repository),
                    "git_tracked_diff_sha256": _git_tracked_diff_hash(repository),
                    "hashes": provider_hashes,
                },
            }
        )
        run_dir = self.workspace / "training" / self.backend / experiment_name
        _guard_existing_plan(run_dir, input_fingerprint)
        exp_dir = run_dir / "experiment"
        config_dir = run_dir / "generated-configs"
        sovits_weights = run_dir / "weights" / "sovits"
        gpt_weights = run_dir / "weights" / "gpt"
        for directory in (exp_dir, config_dir, sovits_weights, gpt_weights):
            directory.mkdir(parents=True, exist_ok=True)
        selected_path = run_dir / "selected-training-rows.jsonl"
        _atomic_text(
            selected_path,
            "".join(
                json.dumps(item.serialise(), ensure_ascii=False, sort_keys=True) + "\n"
                for item in items
            ),
        )
        dataset_list = exp_dir / "dataset.list"
        _atomic_text(
            dataset_list,
            "".join(
                f"{item.audio_path}|{item.speaker}|{item.language}|{item.text}\n" for item in items
            ),
        )
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError(
                "GPT-SoVITS planning requires PyYAML; install the project's 'train' extra"
            ) from exc
        s2_config = json.loads(paths["s2_config"].read_text(encoding="utf-8"))
        full_precision = _bool(
            _option(config, "full_precision", default=True), name="full_precision"
        )
        gpu = str(_option(config, "gpu", default=0))
        s2_config["train"].update(
            {
                "batch_size": int(_option(config, "sovits_batch_size", default=6)),
                "epochs": int(_option(config, "sovits_epochs", default=12)),
                "text_low_lr_rate": float(_option(config, "text_low_lr_rate", default=0.4)),
                "pretrained_s2G": str(paths["s2g"]),
                "pretrained_s2D": str(paths["s2d"]),
                "if_save_latest": True,
                "if_save_every_weights": True,
                "save_every_epoch": int(_option(config, "sovits_save_every", default=4)),
                "gpu_numbers": gpu,
                "grad_ckpt": _bool(
                    _option(config, "grad_checkpoint", default=False),
                    name="grad_checkpoint",
                ),
                "lora_rank": str(_option(config, "lora_rank", default=32)),
                "fp16_run": not full_precision,
            }
        )
        s2_config["model"]["version"] = model_version
        s2_config["data"]["exp_dir"] = str(exp_dir)
        s2_config["s2_ckpt_dir"] = str(exp_dir)
        s2_config["save_weight_dir"] = str(sovits_weights)
        s2_config["name"] = experiment_name
        s2_config["version"] = model_version
        s2_path = config_dir / "s2_train.json"
        _atomic_text(s2_path, json.dumps(s2_config, ensure_ascii=False, indent=2) + "\n")
        s1_config = yaml.safe_load(paths["s1_config"].read_text(encoding="utf-8"))
        s1_config["train"].update(
            {
                "batch_size": int(_option(config, "gpt_batch_size", default=6)),
                "epochs": int(_option(config, "gpt_epochs", default=20)),
                "precision": "32" if full_precision else "16-mixed",
                "save_every_n_epoch": int(_option(config, "gpt_save_every", default=5)),
                "if_save_every_weights": True,
                "if_save_latest": True,
                "if_dpo": False,
                "half_weights_save_dir": str(gpt_weights),
                "exp_name": experiment_name,
            }
        )
        s1_config["pretrained_s1"] = str(paths["s1"])
        s1_config["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
        s1_config["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
        s1_config["output_dir"] = str(exp_dir / f"logs_s1_{model_version}")
        s1_path = config_dir / "s1_train.yaml"
        _atomic_text(s1_path, yaml.safe_dump(s1_config, allow_unicode=True, sort_keys=False))
        environment = {
            "PYTHONPATH": os.pathsep.join(str(path) for path in pythonpath),
            "version": model_version,
            "inp_text": str(dataset_list),
            "inp_wav_dir": "",
            "exp_name": experiment_name,
            "opt_dir": str(exp_dir),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": gpu,
            "is_half": str(not full_precision),
            "bert_pretrained_dir": str(paths["bert"]),
            "cnhubert_base_dir": str(paths["hubert"]),
            "pretrained_s2G": str(paths["s2g"]),
            "s2config_path": str(paths["s2_config"]),
            "sv_path": str(paths["sv"]),
        }
        worker_module = ["-m", "voice_dataset_pipeline.training"]
        artifact_manifest = run_dir / "artifacts.json"
        commands = [
            CommandSpec(
                "prepare-text",
                [str(python), "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
                repository,
                environment,
            ),
            CommandSpec(
                "prepare-hubert",
                [str(python), "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
                repository,
                environment,
            ),
        ]
        if "Pro" in model_version:
            commands.append(
                CommandSpec(
                    "prepare-speaker",
                    [str(python), "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"],
                    repository,
                    environment,
                )
            )
        commands.extend(
            [
                CommandSpec(
                    "prepare-semantic",
                    [str(python), "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
                    repository,
                    environment,
                ),
                CommandSpec(
                    "verify-prepared-dataset",
                    [
                        str(python),
                        *worker_module,
                        "_verify-gpt-dataset",
                        "--exp-dir",
                        str(exp_dir),
                        "--selected",
                        str(selected_path),
                        "--pro",
                        str("Pro" in model_version),
                    ],
                    repository,
                    {"PYTHONPATH": environment["PYTHONPATH"]},
                    expected_files=[
                        exp_dir / "2-name2text.txt",
                        exp_dir / "6-name2semantic.tsv",
                    ],
                    expected_directories=[exp_dir / "4-cnhubert", exp_dir / "5-wav32k"],
                ),
                CommandSpec(
                    "train-sovits",
                    [
                        str(python),
                        "-s",
                        "GPT_SoVITS/s2_train.py",
                        "--config",
                        str(s2_path),
                    ],
                    repository,
                    {**environment, "hz": "25hz"},
                ),
                CommandSpec(
                    "train-gpt",
                    [
                        str(python),
                        "-s",
                        "GPT_SoVITS/s1_train.py",
                        "--config_file",
                        str(s1_path),
                    ],
                    repository,
                    {**environment, "hz": "25hz"},
                ),
                CommandSpec(
                    "verify-models",
                    [
                        str(python),
                        *worker_module,
                        "_verify-gpt-models",
                        "--repository",
                        str(repository),
                        "--experiment",
                        experiment_name,
                        "--model-version",
                        model_version,
                        "--sovits-dir",
                        str(sovits_weights),
                        "--gpt-dir",
                        str(gpt_weights),
                        "--output",
                        str(artifact_manifest),
                    ],
                    repository,
                    {"PYTHONPATH": environment["PYTHONPATH"]},
                    expected_files=[artifact_manifest],
                ),
            ]
        )
        metadata = {
            "input_fingerprint": input_fingerprint,
            "provider": {
                "repository": str(repository),
                "git_head": _git_head(repository),
                "git_tracked_diff_sha256": _git_tracked_diff_hash(repository),
                "python": str(python),
                "probe": probe_report,
                "hashes": provider_hashes,
            },
            "dataset": {
                "items": len(items),
                "selected_manifest": str(selected_path),
                "selected_sha256": _sha256_file(selected_path),
                "dataset_list": str(dataset_list),
                "dataset_list_sha256": _sha256_file(dataset_list),
            },
            "model_version": model_version,
            "generated_configs": {
                str(s2_path): _sha256_file(s2_path),
                str(s1_path): _sha256_file(s1_path),
            },
        }
        return self._commit_plan(
            experiment=experiment_name,
            run_dir=run_dir,
            commands=commands,
            metadata=metadata,
            artifact_manifest=artifact_manifest,
        )


class RVCAdapter(_AdapterBase):
    """Adapter for RVC v1/v2 preprocessing, F0/HuBERT, training and index build."""

    backend = "rvc"

    def plan(
        self,
        manifest: Path | Iterable[Any],
        *,
        experiment: str | None = None,
    ) -> TrainingPlan:
        config = self.config
        repository = _resolve(_option(config, "repository", "root"))
        python = _resolve(
            _option(config, "python", "python_exe", default=repository / ".venv/Scripts/python.exe")
        )
        experiment_name = self._experiment(experiment)
        version = str(_option(config, "version", default="v2"))
        rate = str(_option(config, "sample_rate", default="48k"))
        if version not in {"v1", "v2"} or rate not in {"32k", "40k", "48k"}:
            raise ConfigurationError("RVC version/rate must be v1|v2 and 32k|40k|48k")
        if version == "v2" and rate == "40k":
            raise ConfigurationError("RVC 40k uses the v1 configuration; choose version='v1'")
        if not repository.is_dir() or not python.is_file():
            raise ConfigurationError("RVC repository or Python executable is missing")
        pretrained_dir = repository / (
            "assets/pretrained_v2" if version == "v2" else "assets/pretrained"
        )
        generator = pretrained_dir / f"f0G{rate}.pth"
        discriminator = pretrained_dir / f"f0D{rate}.pth"
        config_template = (
            repository
            / "configs"
            / ("v1" if version == "v1" or rate == "40k" else "v2")
            / f"{rate}.json"
        )
        required_files = [
            repository / "train/preprocess.py",
            repository / "train/dataset/extract_f0.py",
            repository / "train/dataset/extract_hubert_feature.py",
            repository / "train/train.py",
            repository / "train/train_index.py",
            generator,
            discriminator,
            config_template,
            repository / "assets/rmvpe/rmvpe.pt",
            repository / "assets/hubert_base/config.json",
            repository / "assets/hubert_base/preprocessor_config.json",
            repository / "assets/hubert_base/pytorch_model.bin",
        ]
        _require_files([python, *required_files])
        probe_report = {}
        if self.python_probe is not None:
            probe_report = self.python_probe(
                python,
                repository,
                [
                    "torch",
                    "transformers",
                    "faiss",
                    "soundfile",
                    "librosa",
                    "sklearn",
                    "parselmouth",
                ],
                require_cuda=True,
                pythonpath=[repository],
            )
        items = _validated_items(
            manifest,
            require_text=False,
            default_speaker="0",
            require_reviewed=_bool(
                _option(config, "require_reviewed", default=True),
                name="require_reviewed",
            ),
        )
        provider_hashes = {
            str(path.relative_to(repository)): _sha256_file(path) for path in required_files
        }
        input_fingerprint = _fingerprint(
            {
                "dataset": [item.serialise() for item in items],
                "config": _adapter_config_snapshot(config),
                "provider": {
                    "repository": str(repository),
                    "git_head": _git_head(repository),
                    "git_tracked_diff_sha256": _git_tracked_diff_hash(repository),
                    "hashes": provider_hashes,
                },
            }
        )
        run_dir = self.workspace / "training" / self.backend / experiment_name
        _guard_existing_plan(run_dir, input_fingerprint)
        exp_dir = repository / "logs" / experiment_name
        weights_dir = repository / "assets/weights"
        indices_dir = repository / "assets/indices"
        if exp_dir.is_dir() and any(exp_dir.iterdir()):
            raise ConfigurationError(
                "RVC provider experiment directory is not empty; "
                f"use a new experiment to avoid stale features: {exp_dir}"
            )
        existing_model = weights_dir / f"{experiment_name}.pth"
        existing_indices = list(indices_dir.glob(f"*{experiment_name}*.index"))
        if existing_model.exists() or existing_indices:
            raise ConfigurationError(
                "RVC provider artifacts already use this experiment name; choose a new experiment"
            )
        dataset_dir = run_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_rows: list[dict[str, Any]] = []
        for item in items:
            target = dataset_dir / f"{item.clip_id}.wav"
            if target.exists():
                if _sha256_file(target) != item.audio_sha256:
                    raise ConfigurationError(
                        f"stale RVC dataset file conflicts with {item.clip_id}"
                    )
                mode = "existing"
            else:
                try:
                    os.link(item.audio_path, target)
                    mode = "hardlink"
                except OSError:
                    shutil.copy2(item.audio_path, target)
                    mode = "copy"
            dataset_rows.append(
                {
                    **item.serialise(),
                    "dataset_wav": str(target),
                    "materialization": mode,
                }
            )
        dataset_manifest = run_dir / "dataset-manifest.json"
        _atomic_json(dataset_manifest, dataset_rows)
        for directory in (exp_dir, weights_dir, indices_dir):
            directory.mkdir(parents=True, exist_ok=True)
        workers = int(_option(config, "workers", default=max(1, os.cpu_count() or 1)))
        gpu = str(_option(config, "gpu", default=0))
        batch_size = int(_option(config, "batch_size", default=6))
        epochs = int(_option(config, "epochs", default=200))
        save_every = int(_option(config, "save_every", default=25))
        if min(workers, batch_size, epochs, save_every) <= 0:
            raise ConfigurationError(
                "RVC workers, batch size, epochs and save_every must be positive"
            )
        numeric_rate = {"32k": 32000, "40k": 40000, "48k": 48000}[rate]
        pythonpath = os.pathsep.join([str(repository), str(Path(__file__).resolve().parents[1])])
        worker_module = ["-m", "voice_dataset_pipeline.training"]
        artifact_manifest = run_dir / "artifacts.json"
        commands = [
            CommandSpec(
                "preprocess",
                [
                    str(python),
                    "-m",
                    "train.preprocess",
                    str(dataset_dir),
                    str(numeric_rate),
                    str(workers),
                    str(exp_dir),
                    "False",
                    str(_option(config, "preprocess_seconds", default=3.7)),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_directories=[exp_dir / "0_gt_wavs", exp_dir / "1_16k_wavs"],
            ),
            CommandSpec(
                "extract-f0",
                [
                    str(python),
                    "-m",
                    "train.dataset.extract_f0",
                    "cuda",
                    "1",
                    "0",
                    gpu,
                    str(exp_dir),
                    str(_bool(_option(config, "half", default=True), name="half")),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_directories=[exp_dir / "2a_f0", exp_dir / "2b-f0nsf"],
            ),
            CommandSpec(
                "extract-hubert",
                [
                    str(python),
                    "-m",
                    "train.dataset.extract_hubert_feature",
                    f"cuda:{gpu}",
                    "1",
                    "0",
                    gpu,
                    str(exp_dir),
                    version,
                    str(_bool(_option(config, "half", default=True), name="half")),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_directories=[
                    exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
                ],
            ),
            CommandSpec(
                "build-training-manifest",
                [
                    str(python),
                    *worker_module,
                    "_init-rvc-experiment",
                    "--repository",
                    str(repository),
                    "--experiment",
                    experiment_name,
                    "--version",
                    version,
                    "--sample-rate",
                    rate,
                    "--config-template",
                    str(config_template),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_files=[exp_dir / "filelist.txt", exp_dir / "config.json"],
            ),
            CommandSpec(
                "train",
                [
                    str(python),
                    "-m",
                    "train.train",
                    "-e",
                    experiment_name,
                    "-sr",
                    rate,
                    "-f0",
                    "1",
                    "-bs",
                    str(batch_size),
                    "-g",
                    gpu,
                    "-te",
                    str(epochs),
                    "-se",
                    str(save_every),
                    "-pg",
                    str(generator),
                    "-pd",
                    str(discriminator),
                    "-l",
                    "1",
                    "-c",
                    "0",
                    "-sw",
                    "1",
                    "-v",
                    version,
                ],
                repository,
                {"PYTHONPATH": pythonpath, "RVC_CUDA_GRAPH": "0"},
                expected_files=[weights_dir / f"{experiment_name}.pth"],
            ),
            CommandSpec(
                "verify-inference-model",
                [
                    str(python),
                    *worker_module,
                    "_verify-rvc-model",
                    "--model",
                    str(weights_dir / f"{experiment_name}.pth"),
                    "--version",
                    version,
                    "--sample-rate",
                    rate,
                    "--output",
                    str(artifact_manifest),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_files=[artifact_manifest],
            ),
            CommandSpec(
                "train-index",
                [
                    str(python),
                    "-m",
                    "train.train_index",
                    experiment_name,
                    version,
                    str(indices_dir),
                    str(workers),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
            ),
            CommandSpec(
                "verify-index",
                [
                    str(python),
                    *worker_module,
                    "_verify-rvc-index",
                    "--repository",
                    str(repository),
                    "--experiment",
                    experiment_name,
                    "--version",
                    version,
                    "--output",
                    str(artifact_manifest),
                ],
                repository,
                {"PYTHONPATH": pythonpath},
                expected_files=[artifact_manifest],
            ),
        ]
        metadata = {
            "input_fingerprint": input_fingerprint,
            "provider": {
                "repository": str(repository),
                "git_head": _git_head(repository),
                "git_tracked_diff_sha256": _git_tracked_diff_hash(repository),
                "python": str(python),
                "probe": probe_report,
                "hashes": provider_hashes,
            },
            "dataset": {
                "items": len(items),
                "manifest": str(dataset_manifest),
                "manifest_sha256": _sha256_file(dataset_manifest),
            },
            "version": version,
            "sample_rate": rate,
            "epochs": epochs,
            "pretrained": {
                "generator": str(generator),
                "generator_sha256": _sha256_file(generator),
                "discriminator": str(discriminator),
                "discriminator_sha256": _sha256_file(discriminator),
            },
        }
        return self._commit_plan(
            experiment=experiment_name,
            run_dir=run_dir,
            commands=commands,
            metadata=metadata,
            artifact_manifest=artifact_manifest,
        )


def _read_selected(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _names_in_text(path: Path, *, header: bool = False) -> set[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if header and lines and lines[0].startswith("item_name\t"):
        lines = lines[1:]
    return {line.split("\t", 1)[0] for line in lines}


def _verify_gpt_dataset(exp_dir: Path, selected: Path, pro: bool) -> None:
    rows = _read_selected(selected)
    expected = {Path(row["audio_path"]).name for row in rows}
    zh_expected = {Path(row["audio_path"]).name for row in rows if row.get("language") == "zh"}
    text_parts = sorted(exp_dir.glob("2-name2text-*.txt"))
    semantic_parts = sorted(exp_dir.glob("6-name2semantic-*.tsv"))
    if not text_parts or not semantic_parts:
        raise ConfigurationError("GPT-SoVITS preparation did not create shard manifests")
    text_lines = [
        line
        for part in text_parts
        for line in part.read_text(encoding="utf-8").splitlines()
        if line
    ]
    semantic_lines = [
        line
        for part in semantic_parts
        for line in part.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("item_name\t")
    ]
    _atomic_text(exp_dir / "2-name2text.txt", "\n".join(text_lines) + "\n")
    _atomic_text(
        exp_dir / "6-name2semantic.tsv",
        "item_name\tsemantic_audio\n" + "\n".join(semantic_lines) + "\n",
    )
    checks = {
        "text": _names_in_text(exp_dir / "2-name2text.txt"),
        "hubert": {path.name.removesuffix(".pt") for path in (exp_dir / "4-cnhubert").glob("*.pt")},
        "wav32k": {path.name for path in (exp_dir / "5-wav32k").glob("*.wav")},
        "semantic": _names_in_text(exp_dir / "6-name2semantic.tsv", header=True),
    }
    if pro:
        checks["speaker"] = {
            path.name.removesuffix(".pt") for path in (exp_dir / "7-sv_cn").glob("*.pt")
        }
    checks["bert"] = {path.name.removesuffix(".pt") for path in (exp_dir / "3-bert").glob("*.pt")}
    failures = {
        name: sorted((zh_expected if name == "bert" else expected) - names)
        for name, names in checks.items()
        if (zh_expected if name == "bert" else expected) != names
    }
    if failures:
        raise ConfigurationError(f"GPT-SoVITS prepared dataset mismatch: {failures}")


def _latest_epoch_file(directory: Path, pattern: str, expression: str) -> Path:
    regex = re.compile(expression)
    candidates: list[tuple[int, Path]] = []
    for path in directory.glob(pattern):
        match = regex.search(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ConfigurationError(f"no compact inference weight found in {directory}")
    return max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime_ns))[1]


def _verify_gpt_models(
    repository: Path,
    experiment: str,
    model_version: str,
    sovits_dir: Path,
    gpt_dir: Path,
    output: Path,
) -> None:
    sys.path.insert(0, str(repository / "GPT_SoVITS"))
    import torch
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

    sovits = _latest_epoch_file(
        sovits_dir,
        f"{experiment}_e*_s*.pth",
        rf"^{re.escape(experiment)}_e(\d+)_s\d+\.pth$",
    )
    gpt = _latest_epoch_file(
        gpt_dir,
        f"{experiment}-e*.ckpt",
        rf"^{re.escape(experiment)}-e(\d+)\.ckpt$",
    )
    sovits_payload = load_sovits_new(str(sovits))
    gpt_payload = torch.load(gpt, map_location="cpu", weights_only=False)
    required = {"weight", "config", "info"}
    if not required.issubset(sovits_payload) or not required.issubset(gpt_payload):
        raise ConfigurationError("GPT-SoVITS compact weight is structurally invalid")
    _, detected, _ = get_sovits_version_from_path_fast(str(sovits))
    if detected != model_version:
        raise ConfigurationError(
            f"SoVITS model version mismatch: expected {model_version}, got {detected}"
        )
    _atomic_json(
        output,
        {
            "artifacts": [
                {"kind": "sovits", "path": str(sovits), "sha256": _sha256_file(sovits)},
                {"kind": "gpt", "path": str(gpt), "sha256": _sha256_file(gpt)},
            ]
        },
    )


def _init_rvc_experiment(
    repository: Path,
    experiment: str,
    version: str,
    sample_rate: str,
    config_template: Path,
) -> None:
    exp_dir = repository / "logs" / experiment
    feature = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    directories = {
        "gt": exp_dir / "0_gt_wavs",
        "feature": feature,
        "f0": exp_dir / "2a_f0",
        "f0nsf": exp_dir / "2b-f0nsf",
    }
    _require_directories(directories.values())
    names = {path.stem for path in directories["gt"].glob("*.wav")}
    corresponding = {
        "feature": {path.stem for path in directories["feature"].glob("*.npy")},
        "f0": {path.name.removesuffix(".wav.npy") for path in directories["f0"].glob("*.wav.npy")},
        "f0nsf": {
            path.name.removesuffix(".wav.npy") for path in directories["f0nsf"].glob("*.wav.npy")
        },
    }
    if not names or any(values != names for values in corresponding.values()):
        raise ConfigurationError(
            f"RVC preprocessing/F0/HuBERT outputs do not have identical stems: "
            f"gt={len(names)}, { {key: len(value) for key, value in corresponding.items()} }"
        )
    rows = [
        (
            f"{directories['gt'] / (name + '.wav')}|"
            f"{directories['feature'] / (name + '.npy')}|"
            f"{directories['f0'] / (name + '.wav.npy')}|"
            f"{directories['f0nsf'] / (name + '.wav.npy')}|0"
        )
        for name in sorted(names)
    ]
    dimension = 256 if version == "v1" else 768
    mute = repository / "logs/mute"
    mute_parts = [
        mute / "0_gt_wavs" / f"mute{sample_rate}.wav",
        mute / f"3_feature{dimension}" / "mute.npy",
        mute / "2a_f0/mute.wav.npy",
        mute / "2b-f0nsf/mute.wav.npy",
    ]
    if all(path.is_file() for path in mute_parts):
        mute_row = "|".join(str(path) for path in mute_parts) + "|0"
        rows.extend([mute_row, mute_row])
    random.Random(42).shuffle(rows)
    _atomic_text(exp_dir / "filelist.txt", "\n".join(rows) + "\n")
    payload = json.loads(config_template.read_text(encoding="utf-8"))
    _atomic_text(
        exp_dir / "config.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _verify_rvc_model(model: Path, version: str, sample_rate: str, output: Path) -> None:
    if not model.is_file():
        raise ConfigurationError(
            f"RVC inference model is missing: {model}; "
            "G_*.pth/D_*.pth checkpoints are not sufficient"
        )
    import torch

    payload = torch.load(model, map_location="cpu", weights_only=False)
    required = {"weight", "config", "info", "sr", "f0", "version"}
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ConfigurationError("RVC .pth is not an inference model")
    if str(payload["version"]) != version or str(payload["sr"]) != sample_rate:
        raise ConfigurationError(
            f"RVC inference metadata mismatch: version={payload['version']}, sr={payload['sr']}"
        )
    if int(payload["f0"]) != 1:
        raise ConfigurationError("RVC inference model was not trained with F0")
    _atomic_json(
        output,
        {"artifacts": [{"kind": "rvc_model", "path": str(model), "sha256": _sha256_file(model)}]},
    )


def _verify_rvc_index(repository: Path, experiment: str, version: str, output: Path) -> None:
    candidates = [
        Path(path)
        for pattern in (
            str(repository / "assets/indices" / f"*{experiment}*{version}.index"),
            str(repository / "logs" / experiment / f"added_*{experiment}_{version}.index"),
        )
        for path in glob.glob(pattern)
        if Path(path).is_file() and "added_" in Path(path).name
    ]
    if not candidates:
        raise ConfigurationError(f"RVC added FAISS index is missing for {experiment}")
    index_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    import faiss

    index = faiss.read_index(str(index_path))
    expected_dimension = 256 if version == "v1" else 768
    if index.d != expected_dimension or index.ntotal <= 0:
        raise ConfigurationError(f"RVC FAISS index is invalid: d={index.d}, ntotal={index.ntotal}")
    payload = (
        json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"artifacts": []}
    )
    payload.setdefault("artifacts", []).append(
        {"kind": "rvc_index", "path": str(index_path), "sha256": _sha256_file(index_path)}
    )
    _atomic_json(output, payload)


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("worker")
    parser.add_argument("--exp-dir", type=Path)
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--pro")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--model-version")
    parser.add_argument("--sovits-dir", type=Path)
    parser.add_argument("--gpt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--sample-rate")
    parser.add_argument("--config-template", type=Path)
    parser.add_argument("--model", type=Path)
    return parser


def _run_worker(argv: Sequence[str] | None = None) -> int:
    args = _worker_parser().parse_args(argv)
    if args.worker == "_verify-gpt-dataset":
        _verify_gpt_dataset(args.exp_dir, args.selected, _bool(args.pro, name="pro"))
    elif args.worker == "_verify-gpt-models":
        _verify_gpt_models(
            args.repository,
            args.experiment,
            args.model_version,
            args.sovits_dir,
            args.gpt_dir,
            args.output,
        )
    elif args.worker == "_init-rvc-experiment":
        _init_rvc_experiment(
            args.repository,
            args.experiment,
            args.version,
            args.sample_rate,
            args.config_template,
        )
    elif args.worker == "_verify-rvc-model":
        _verify_rvc_model(args.model, args.version, args.sample_rate, args.output)
    elif args.worker == "_verify-rvc-index":
        _verify_rvc_index(args.repository, args.experiment, args.version, args.output)
    else:
        raise ConfigurationError(f"unknown internal training worker: {args.worker}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_run_worker())
    except (ConfigurationError, OSError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
