"""构建并校验可移植的 GPT-SoVITS 语音包。

这个模块刻意只接受已经完成训练和显式评估选择的产物。它不会读取、复制或
导出原始训练集；唯一会随模型复制的音频是调用方明确列出的推理参考音频。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_BUNDLE_ID_PATTERN = r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$"
_PROFILE_ID_PATTERN = r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$"
_WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}

Sha256 = Annotated[str, Field(pattern=_SHA256_PATTERN)]
GitCommit = Annotated[str, Field(pattern=_COMMIT_PATTERN)]
PositiveBytes = Annotated[int, Field(gt=0)]


class VoiceBundleError(ValueError):
    """语音包输入、构建或完整性校验失败。"""


class VoiceBundleIntegrityError(VoiceBundleError):
    """已生成语音包的路径或内容与清单不一致。"""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _bundle_relative_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("语音包资产路径必须是使用 '/' 的非空相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("语音包资产路径不得为绝对路径或包含 '.'/'..'")
    if path.as_posix() != value:
        raise ValueError("语音包资产路径必须是规范化的 POSIX 相对路径")
    for part in path.parts:
        windows_stem = part.split(".", 1)[0].casefold()
        if ":" in part or part.endswith((" ", ".")) or windows_stem in _WINDOWS_RESERVED:
            raise ValueError(f"语音包资产路径包含 Windows 不支持的名称: {part}")
    return path.as_posix()


def _profile_id(value: str) -> str:
    if not re.fullmatch(_PROFILE_ID_PATTERN, value):
        raise ValueError("参考 profile ID 只能包含小写字母、数字、'_' 和 '-'")
    if value.casefold() in _WINDOWS_RESERVED:
        raise ValueError(f"参考 profile ID 是 Windows 保留名称: {value}")
    return value


class BundleFile(_StrictModel):
    path: str
    sha256: Sha256
    bytes: PositiveBytes

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _bundle_relative_path(value)


class BundleAssets(_StrictModel):
    gpt: BundleFile
    sovits: BundleFile


class BundleEngine(_StrictModel):
    name: Literal["gpt-sovits"]
    api: Literal["api_v2"]
    model_version: Annotated[str, Field(min_length=1, max_length=64)]


class BundleReference(_StrictModel):
    description: Annotated[str, Field(min_length=1, max_length=500)]
    auto_enabled: bool
    audio: str
    prompt_text: Annotated[str, Field(min_length=1, max_length=10_000)]
    prompt_lang: Literal["zh", "ja", "en", "ko", "yue"]
    sha256: Sha256
    bytes: PositiveBytes

    @field_validator("audio")
    @classmethod
    def validate_audio(cls, value: str) -> str:
        return _bundle_relative_path(value)

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("参考文本必须是非空单行文本")
        return normalized


class BundleReferences(_StrictModel):
    default: str
    items: dict[str, BundleReference]

    @model_validator(mode="after")
    def validate_profiles(self) -> BundleReferences:
        if not self.items:
            raise ValueError("语音包必须至少包含一个参考 profile")
        normalized_names = {_profile_id(name).casefold() for name in self.items}
        if len(normalized_names) != len(self.items):
            raise ValueError("参考 profile ID 在 Windows 上发生大小写冲突")
        _profile_id(self.default)
        if self.default not in self.items:
            raise ValueError("references.default 必须指向 references.items 中的 profile")
        return self


class CodeProvenance(_StrictModel):
    repository: Annotated[str, Field(min_length=1, max_length=500)]
    commit: GitCommit
    dirty_diff_sha256: Sha256 | None = None
    source_snapshot_sha256: Sha256


class DatasetProvenance(_StrictModel):
    fingerprint: Sha256
    metadata_sha256: Sha256
    manifest_sha256: Sha256
    dataset_list_sha256: Sha256
    included_items: Annotated[int, Field(gt=0)]
    reviewed_only: bool


class TrainingProvenance(_StrictModel):
    plan_fingerprint: Sha256
    plan_sha256: Sha256
    result_sha256: Sha256 | None = None
    artifacts_sha256: Sha256 | None = None


class SelectionProvenance(_StrictModel):
    source_sha256: Sha256
    method: Annotated[str, Field(min_length=1, max_length=500)]
    training_plan_fingerprint: Sha256
    gpt_sha256: Sha256
    sovits_sha256: Sha256
    evaluation_fingerprint: Sha256 | None = None


class BundleProvenance(_StrictModel):
    provider: CodeProvenance
    pipeline: CodeProvenance
    dataset: DatasetProvenance
    training: TrainingProvenance
    selection: SelectionProvenance


class BundleRights(_StrictModel):
    attestation_id: Annotated[str, Field(min_length=1, max_length=200)]
    attestation_sha256: Sha256
    dataset_fingerprint: Sha256
    training_plan_fingerprint: Sha256
    rights_basis: Literal["self_recorded", "licensed", "public_domain", "explicit_permission"]
    voice_subject_authorization: Literal["self", "explicit_permission", "not_applicable"]
    training_allowed: Literal[True]
    local_inference_allowed: Literal[True]
    model_distribution_allowed: bool
    reference_audio_distribution_allowed: bool
    source_dataset_included: Literal[False]


class BundleDistribution(_StrictModel):
    model_allowed: bool
    reference_audio_allowed: bool
    source_dataset_included: Literal[False]


class VoiceBundleManifest(_StrictModel):
    """插件与训练器共享的严格 voice-bundle v2 schema。"""

    schema_version: Literal[2]
    bundle_id: Annotated[str, Field(pattern=_BUNDLE_ID_PATTERN)]
    display_name: Annotated[str, Field(min_length=1, max_length=200)]
    engine: BundleEngine
    assets: BundleAssets
    references: BundleReferences
    provenance: BundleProvenance
    rights: BundleRights
    distribution: BundleDistribution
    files: list[BundleFile]

    @model_validator(mode="after")
    def validate_inventory(self) -> VoiceBundleManifest:
        declared_assets = [
            ("assets.gpt", self.assets.gpt.path, self.assets.gpt.sha256, self.assets.gpt.bytes),
            (
                "assets.sovits",
                self.assets.sovits.path,
                self.assets.sovits.sha256,
                self.assets.sovits.bytes,
            ),
            *[
                (
                    f"references.items.{profile}",
                    reference.audio,
                    reference.sha256,
                    reference.bytes,
                )
                for profile, reference in self.references.items.items()
            ],
        ]
        expected: dict[str, tuple[str, int]] = {}
        portable_paths: dict[str, str] = {}
        for label, path, sha256, size in declared_assets:
            folded = path.casefold()
            if folded in portable_paths:
                raise ValueError(
                    f"资产路径在 Windows 上冲突: {label} 与 {portable_paths[folded]} -> {path}"
                )
            portable_paths[folded] = label
            expected[path] = (sha256, size)

        inventory: dict[str, tuple[str, int]] = {}
        for item in self.files:
            if item.path in inventory:
                raise ValueError(f"files 中存在重复路径: {item.path}")
            inventory[item.path] = (item.sha256, item.bytes)
        if inventory != expected:
            raise ValueError("files 必须且只能完整列出 GPT、SoVITS 和参考音频资产")
        if self.distribution.model_allowed != self.rights.model_distribution_allowed:
            raise ValueError("模型分发声明与 rights attestation 不一致")
        if (
            self.distribution.reference_audio_allowed
            != self.rights.reference_audio_distribution_allowed
        ):
            raise ValueError("参考音频分发声明与 rights attestation 不一致")
        return self


class ReferenceProfileSource(_StrictModel):
    description: Annotated[str, Field(min_length=1, max_length=500)]
    auto_enabled: bool = True
    audio_path: Annotated[str, Field(min_length=1, max_length=2_000)]
    prompt_text: Annotated[str, Field(min_length=1, max_length=10_000)]
    prompt_lang: Literal["zh", "ja", "en", "ko", "yue"]
    sha256: Sha256 | None = None

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\n" in normalized or "\r" in normalized:
            raise ValueError("参考文本必须是非空单行文本")
        return normalized


class ReferenceProfileDocument(_StrictModel):
    schema_version: Literal[1] = 1
    default: str
    items: dict[str, ReferenceProfileSource]

    @model_validator(mode="after")
    def validate_profiles(self) -> ReferenceProfileDocument:
        if not self.items:
            raise ValueError("参考 profile 文件必须至少包含一个条目")
        normalized_names = {_profile_id(name).casefold() for name in self.items}
        if len(normalized_names) != len(self.items):
            raise ValueError("参考 profile ID 在 Windows 上发生大小写冲突")
        _profile_id(self.default)
        if self.default not in self.items:
            raise ValueError("default 必须指向 items 中的 profile")
        return self


class RightsAttestation(_StrictModel):
    schema_version: Literal[1] = 1
    attestation_id: Annotated[str, Field(min_length=1, max_length=200)]
    subject: Annotated[str, Field(min_length=1, max_length=500)]
    dataset_fingerprint: Sha256
    training_plan_fingerprint: Sha256
    rights_basis: Literal["self_recorded", "licensed", "public_domain", "explicit_permission"]
    rights_holder: Annotated[str, Field(min_length=1, max_length=500)]
    evidence_reference: Annotated[str, Field(min_length=1, max_length=2_000)]
    voice_subject_authorization: Literal["self", "explicit_permission", "not_applicable"]
    training_allowed: bool
    local_inference_allowed: bool
    model_distribution_allowed: bool = False
    reference_audio_distribution_allowed: bool = False
    source_dataset_distribution_allowed: Literal[False] = False
    attested_by: Annotated[str, Field(min_length=1, max_length=500)]
    attested_at: Annotated[str, Field(min_length=1, max_length=100)]

    @field_validator("attested_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        normalized = value.strip()
        if "T" not in normalized:
            raise ValueError("attested_at 必须是带时间的 ISO-8601 字符串")
        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("attested_at 不是有效的 ISO-8601 时间") from exc
        if parsed.utcoffset() is None:
            raise ValueError("attested_at 必须包含时区偏移")
        return normalized

    @model_validator(mode="after")
    def require_local_rights(self) -> RightsAttestation:
        if not self.training_allowed:
            raise ValueError("rights attestation 必须明确允许训练")
        if not self.local_inference_allowed:
            raise ValueError("rights attestation 必须明确允许本地推理")
        return self


@dataclass(frozen=True, slots=True)
class BundleBuildRequest:
    bundle_id: str
    display_name: str
    output_dir: Path
    selection_path: Path
    reference_profile_path: Path
    rights_attestation_path: Path
    training_plan_path: Path
    dataset_metadata_path: Path
    pipeline: CodeProvenance
    training_result_path: Path | None = None
    artifacts_path: Path | None = None
    provider_repository: str | None = None


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    root: Path
    manifest_path: Path
    manifest: VoiceBundleManifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise VoiceBundleError(f"{label}不存在: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VoiceBundleError(f"无法读取{label}: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VoiceBundleError(f"{label}顶层必须是 JSON 对象: {resolved}")
    return payload


def _read_model(path: Path, model: type[_StrictModel], label: str) -> _StrictModel:
    payload = _read_json(path, label)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise VoiceBundleError(f"{label}校验失败: {exc}") from exc


def _source_file(value: str, *, base: Path, label: str, suffix: str) -> Path:
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = base / source
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise VoiceBundleError(f"{label}不存在: {source}") from exc
    if not resolved.is_file() or resolved.suffix.lower() != suffix:
        raise VoiceBundleError(f"{label}必须是现有的 {suffix} 文件: {resolved}")
    if resolved.stat().st_size <= 0:
        raise VoiceBundleError(f"{label}不能为空文件: {resolved}")
    if suffix == ".wav":
        try:
            info = sf.info(str(resolved))
        except Exception as exc:
            raise VoiceBundleError(f"{label}不是可解码的 WAV: {resolved}: {exc}") from exc
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise VoiceBundleError(f"{label}不是有效的非空 WAV: {resolved}")
    return resolved


def _selected_paths(selection_path: Path) -> tuple[Path, Path, SelectionProvenance]:
    payload = _read_json(selection_path, "checkpoint selection")
    selected = payload.get("selected", payload)
    if not isinstance(selected, dict):
        raise VoiceBundleError("checkpoint selection.selected 必须是对象")
    gpt_value = selected.get("gpt_path")
    sovits_value = selected.get("sovits_path")
    if not isinstance(gpt_value, str) or not isinstance(sovits_value, str):
        raise VoiceBundleError("checkpoint selection 必须显式包含 gpt_path 和 sovits_path")
    base = selection_path.expanduser().resolve().parent
    gpt = _source_file(gpt_value, base=base, label="selected GPT", suffix=".ckpt")
    sovits = _source_file(sovits_value, base=base, label="selected SoVITS", suffix=".pth")
    declared_hashes: dict[str, str] = {}
    for key, source in (("gpt_sha256", gpt), ("sovits_sha256", sovits)):
        declared = selected.get(key)
        if not isinstance(declared, str) or not re.fullmatch(_SHA256_PATTERN, declared):
            raise VoiceBundleError(f"checkpoint selection 必须显式包含有效的 {key}")
        if declared != _sha256_file(source):
            raise VoiceBundleError(f"checkpoint selection 的 {key} 与文件不一致")
        declared_hashes[key] = declared
    training_plan_fingerprint = payload.get("training_plan_fingerprint")
    if not isinstance(training_plan_fingerprint, str) or not re.fullmatch(
        _SHA256_PATTERN, training_plan_fingerprint
    ):
        raise VoiceBundleError("selection.training_plan_fingerprint 必须是有效 SHA-256")
    evaluation_fingerprint = payload.get("evaluation_fingerprint")
    if evaluation_fingerprint is not None and (
        not isinstance(evaluation_fingerprint, str)
        or not re.fullmatch(_SHA256_PATTERN, evaluation_fingerprint)
    ):
        raise VoiceBundleError("selection.evaluation_fingerprint 必须是 SHA-256")
    method_value = payload.get("goal") or payload.get("method") or "explicit-selection"
    if not isinstance(method_value, str) or not method_value.strip():
        raise VoiceBundleError("selection method/goal 必须是非空字符串")
    method = method_value.strip()
    return (
        gpt,
        sovits,
        SelectionProvenance(
            source_sha256=_sha256_file(selection_path.expanduser().resolve()),
            method=method,
            training_plan_fingerprint=training_plan_fingerprint,
            gpt_sha256=declared_hashes["gpt_sha256"],
            sovits_sha256=declared_hashes["sovits_sha256"],
            evaluation_fingerprint=evaluation_fingerprint,
        ),
    )


def _normalise_repository_url(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    elif normalized.startswith("ssh://git@github.com/"):
        normalized = "https://github.com/" + normalized.removeprefix("ssh://git@github.com/")
    return normalized.removesuffix(".git").rstrip("/")


def _git_output(root: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        shell=False,
    )
    if process.returncode != 0:
        error = process.stderr.decode("utf-8", errors="replace").strip()
        raise VoiceBundleError(f"Git 命令失败 ({' '.join(arguments)}): {error}")
    return process.stdout.decode("utf-8", errors="strict").strip()


def _repository_from_path(path: Path) -> str:
    return _normalise_repository_url(_git_output(path, "remote", "get-url", "origin"))


def _source_snapshot(root: Path) -> str:
    candidates = sorted(
        [path for path in (root / "src").rglob("*.py") if path.is_file()]
        + [path for path in (root / "schemas").rglob("*.json") if path.is_file()]
        + ([root / "pyproject.toml"] if (root / "pyproject.toml").is_file() else []),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not candidates:
        raise VoiceBundleError(f"无法在管线仓库中找到可哈希的源码: {root}")
    digest = hashlib.sha256()
    for path in candidates:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def discover_pipeline_provenance(
    root: Path,
    *,
    repository: str | None = None,
    commit: str | None = None,
) -> CodeProvenance:
    """从 Git checkout 和源码快照生成不含本地绝对路径的管线来源记录。"""

    resolved = root.expanduser().resolve()
    repository_url = (
        _normalise_repository_url(repository) if repository else _repository_from_path(resolved)
    )
    commit_value = commit or _git_output(resolved, "rev-parse", "HEAD")
    diff = subprocess.run(
        ["git", "-C", str(resolved), "diff", "--binary", "HEAD", "--"],
        capture_output=True,
        check=False,
        shell=False,
    )
    if diff.returncode != 0:
        error = diff.stderr.decode("utf-8", errors="replace").strip()
        raise VoiceBundleError(f"无法计算管线 tracked diff: {error}")
    dirty_sha256 = hashlib.sha256(diff.stdout).hexdigest() if diff.stdout else None
    return CodeProvenance(
        repository=repository_url,
        commit=commit_value,
        dirty_diff_sha256=dirty_sha256,
        source_snapshot_sha256=_source_snapshot(resolved),
    )


def _provider_provenance(
    plan: dict[str, Any],
    repository_override: str | None,
) -> tuple[CodeProvenance, str, str]:
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        raise VoiceBundleError("training-plan.json 缺少 metadata")
    provider = metadata.get("provider")
    if not isinstance(provider, dict):
        raise VoiceBundleError("training-plan.json 缺少 metadata.provider")
    repository_value = provider.get("repository")
    if repository_override:
        repository = _normalise_repository_url(repository_override)
    elif isinstance(repository_value, str) and re.match(
        r"^(?:https?|ssh)://|^git@", repository_value
    ):
        repository = _normalise_repository_url(repository_value)
    elif isinstance(repository_value, str):
        provider_root = Path(repository_value).expanduser().resolve()
        repository = _repository_from_path(provider_root)
    else:
        raise VoiceBundleError("无法从 training plan 解析 provider repository")
    hashes = provider.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise VoiceBundleError("training-plan.json 缺少 provider 文件哈希")
    for value in hashes.values():
        if not isinstance(value, str) or not re.fullmatch(_SHA256_PATTERN, value):
            raise VoiceBundleError("training-plan.json 包含无效的 provider 文件哈希")
    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise VoiceBundleError("training-plan.json 缺少 model_version")
    fingerprint = plan.get("fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(_SHA256_PATTERN, fingerprint):
        raise VoiceBundleError("training-plan.json 缺少有效 fingerprint")
    provenance = CodeProvenance(
        repository=repository,
        commit=provider.get("git_head"),
        dirty_diff_sha256=provider.get("git_tracked_diff_sha256") or None,
        source_snapshot_sha256=_canonical_sha256(hashes),
    )
    return provenance, model_version, fingerprint


def _dataset_provenance(path: Path) -> DatasetProvenance:
    payload = _read_json(path, "dataset metadata")
    allow_unreviewed = payload.get("allow_unreviewed", False)
    if not isinstance(allow_unreviewed, bool):
        raise VoiceBundleError("dataset metadata.allow_unreviewed 必须是布尔值")
    try:
        return DatasetProvenance(
            fingerprint=payload.get("fingerprint"),
            metadata_sha256=_sha256_file(path.expanduser().resolve()),
            manifest_sha256=payload.get("manifest_sha256"),
            dataset_list_sha256=payload.get("dataset_list_sha256"),
            included_items=payload.get("included"),
            reviewed_only=not allow_unreviewed,
        )
    except ValidationError as exc:
        raise VoiceBundleError(f"dataset metadata 校验失败: {exc}") from exc


def _validate_plan_dataset(plan: dict[str, Any], dataset: DatasetProvenance) -> None:
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        raise VoiceBundleError("training-plan.json 缺少 metadata")
    plan_dataset = metadata.get("dataset")
    if not isinstance(plan_dataset, dict):
        raise VoiceBundleError("training-plan.json 缺少 metadata.dataset")
    if plan_dataset.get("dataset_list_sha256") != dataset.dataset_list_sha256:
        raise VoiceBundleError("dataset metadata.dataset_list_sha256 与 training plan 不一致")
    plan_items = plan_dataset.get("items")
    if (
        not isinstance(plan_items, int)
        or isinstance(plan_items, bool)
        or plan_items != dataset.included_items
    ):
        raise VoiceBundleError("dataset metadata.included 与 training plan dataset.items 不一致")


def _optional_training_file(path: Path | None, label: str, fingerprint: str) -> str | None:
    if path is None:
        return None
    payload = _read_json(path, label)
    declared = payload.get("fingerprint")
    if declared is not None and declared != fingerprint:
        raise VoiceBundleError(f"{label} fingerprint 与 training plan 不一致")
    if label == "artifacts.json":
        rows = payload.get("artifacts")
        if not isinstance(rows, list) or not rows:
            raise VoiceBundleError("artifacts.json 必须包含非空 artifacts 数组")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise VoiceBundleError("artifacts.json 包含无效条目")
            source = Path(row["path"]).expanduser().resolve()
            declared_sha256 = row.get("sha256")
            if not source.is_file() or declared_sha256 != _sha256_file(source):
                raise VoiceBundleError(f"artifacts.json 记录的产物不存在或哈希不符: {source}")
    return _sha256_file(path.expanduser().resolve())


def _copy_file(source: Path, destination: Path) -> BundleFile:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    relative = destination.relative_to(destination.parents[1]).as_posix()
    return BundleFile(
        path=relative,
        sha256=_sha256_file(destination),
        bytes=destination.stat().st_size,
    )


def _write_manifest(path: Path, manifest: VoiceBundleManifest) -> None:
    path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_voice_bundle(request: BundleBuildRequest) -> BundleBuildResult:
    """从显式选择、参考 profile 和权利声明构建一个原子语音包。"""

    output = request.output_dir.expanduser().resolve()
    if output.exists():
        raise VoiceBundleError(f"输出目录已存在；为避免覆盖请使用新目录: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    selection_path = request.selection_path.expanduser().resolve()
    gpt_source, sovits_source, selection = _selected_paths(selection_path)
    plan_path = request.training_plan_path.expanduser().resolve()
    plan = _read_json(plan_path, "training-plan.json")
    if plan.get("backend") != "gpt_sovits":
        raise VoiceBundleError("training plan backend 必须是 gpt_sovits")
    provider, model_version, plan_fingerprint = _provider_provenance(
        plan,
        request.provider_repository,
    )
    if selection.training_plan_fingerprint != plan_fingerprint:
        raise VoiceBundleError(
            "selection.training_plan_fingerprint 与 training plan fingerprint 不一致"
        )
    dataset = _dataset_provenance(request.dataset_metadata_path)
    _validate_plan_dataset(plan, dataset)
    references = _read_model(
        request.reference_profile_path,
        ReferenceProfileDocument,
        "reference profile",
    )
    assert isinstance(references, ReferenceProfileDocument)
    rights = _read_model(
        request.rights_attestation_path,
        RightsAttestation,
        "rights attestation",
    )
    assert isinstance(rights, RightsAttestation)
    if rights.dataset_fingerprint != dataset.fingerprint:
        raise VoiceBundleError(
            "rights attestation.dataset_fingerprint 与 dataset metadata.fingerprint 不一致"
        )
    if rights.training_plan_fingerprint != plan_fingerprint:
        raise VoiceBundleError(
            "rights attestation.training_plan_fingerprint 与 training plan fingerprint 不一致"
        )
    result_sha256 = _optional_training_file(
        request.training_result_path,
        "training-result.json",
        plan_fingerprint,
    )
    artifacts_sha256 = _optional_training_file(
        request.artifacts_path,
        "artifacts.json",
        plan_fingerprint,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        gpt = _copy_file(gpt_source, temporary / "models" / "gpt.ckpt")
        sovits = _copy_file(sovits_source, temporary / "models" / "sovits.pth")
        if gpt.sha256 != selection.gpt_sha256 or sovits.sha256 != selection.sovits_sha256:
            raise VoiceBundleError("selected checkpoint 在校验与复制之间发生变化")
        bundled_references: dict[str, BundleReference] = {}
        files = [gpt, sovits]
        reference_base = request.reference_profile_path.expanduser().resolve().parent
        for profile_name in sorted(references.items):
            source_profile = references.items[profile_name]
            audio_source = _source_file(
                source_profile.audio_path,
                base=reference_base,
                label=f"参考音频 {profile_name}",
                suffix=".wav",
            )
            audio_sha256 = _sha256_file(audio_source)
            if source_profile.sha256 is not None and source_profile.sha256 != audio_sha256:
                raise VoiceBundleError(f"参考音频 {profile_name} 的声明 SHA-256 不一致")
            audio = _copy_file(
                audio_source,
                temporary / "references" / f"{profile_name}.wav",
            )
            files.append(audio)
            bundled_references[profile_name] = BundleReference(
                description=source_profile.description,
                auto_enabled=source_profile.auto_enabled,
                audio=audio.path,
                prompt_text=source_profile.prompt_text,
                prompt_lang=source_profile.prompt_lang,
                sha256=audio.sha256,
                bytes=audio.bytes,
            )

        attestation_path = request.rights_attestation_path.expanduser().resolve()
        manifest = VoiceBundleManifest(
            schema_version=2,
            bundle_id=request.bundle_id,
            display_name=request.display_name,
            engine=BundleEngine(
                name="gpt-sovits",
                api="api_v2",
                model_version=model_version,
            ),
            assets=BundleAssets(gpt=gpt, sovits=sovits),
            references=BundleReferences(
                default=references.default,
                items=bundled_references,
            ),
            provenance=BundleProvenance(
                provider=provider,
                pipeline=request.pipeline,
                dataset=dataset,
                training=TrainingProvenance(
                    plan_fingerprint=plan_fingerprint,
                    plan_sha256=_sha256_file(plan_path),
                    result_sha256=result_sha256,
                    artifacts_sha256=artifacts_sha256,
                ),
                selection=selection,
            ),
            rights=BundleRights(
                attestation_id=rights.attestation_id,
                attestation_sha256=_sha256_file(attestation_path),
                dataset_fingerprint=rights.dataset_fingerprint,
                training_plan_fingerprint=rights.training_plan_fingerprint,
                rights_basis=rights.rights_basis,
                voice_subject_authorization=rights.voice_subject_authorization,
                training_allowed=True,
                local_inference_allowed=True,
                model_distribution_allowed=rights.model_distribution_allowed,
                reference_audio_distribution_allowed=(rights.reference_audio_distribution_allowed),
                source_dataset_included=False,
            ),
            distribution=BundleDistribution(
                model_allowed=rights.model_distribution_allowed,
                reference_audio_allowed=rights.reference_audio_distribution_allowed,
                source_dataset_included=False,
            ),
            files=files,
        )
        manifest_path = temporary / "voice-bundle.json"
        _write_manifest(manifest_path, manifest)
        load_voice_bundle(manifest_path)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    final_manifest = output / "voice-bundle.json"
    return BundleBuildResult(
        root=output,
        manifest_path=final_manifest,
        manifest=load_voice_bundle(final_manifest),
    )


def _asset_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(_bundle_relative_path(relative))
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise VoiceBundleIntegrityError(f"语音包资产不得是符号链接: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise VoiceBundleIntegrityError(f"语音包资产不存在: {relative}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VoiceBundleIntegrityError(f"语音包资产逃逸出 bundle 根目录: {relative}") from exc
    if not resolved.is_file():
        raise VoiceBundleIntegrityError(f"语音包资产不是普通文件: {relative}")
    return resolved


def load_voice_bundle(manifest_path: Path | str) -> VoiceBundleManifest:
    """解析 schema 并核验全部资产的路径、长度、SHA-256 与白名单。"""

    path = Path(manifest_path).expanduser().resolve()
    payload = _read_json(path, "voice-bundle.json")
    try:
        manifest = VoiceBundleManifest.model_validate(payload)
    except ValidationError as exc:
        raise VoiceBundleIntegrityError(f"voice-bundle schema 校验失败: {exc}") from exc
    root = path.parent.resolve(strict=True)
    declared_paths: set[str] = set()
    for item in manifest.files:
        asset = _asset_path(root, item.path)
        if asset.stat().st_size != item.bytes:
            raise VoiceBundleIntegrityError(f"资产字节数不符: {item.path}")
        if _sha256_file(asset) != item.sha256:
            raise VoiceBundleIntegrityError(f"资产 SHA-256 不符: {item.path}")
        declared_paths.add(item.path)

    actual_paths: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise VoiceBundleIntegrityError(f"语音包内不得包含符号链接: {candidate}")
        if candidate.is_file() and candidate != path:
            actual_paths.add(candidate.relative_to(root).as_posix())
    if actual_paths != declared_paths:
        missing = sorted(declared_paths - actual_paths)
        unexpected = sorted(actual_paths - declared_paths)
        raise VoiceBundleIntegrityError(
            f"语音包文件白名单不一致: missing={missing}, unexpected={unexpected}"
        )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-bundle", description="构建和校验本地 GPT-SoVITS 语音包"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="从显式 checkpoint selection 构建语音包")
    build.add_argument("--bundle-id", required=True)
    build.add_argument("--display-name", required=True)
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--reference-profile", type=Path, required=True)
    build.add_argument("--rights-attestation", type=Path, required=True)
    build.add_argument("--training-plan", type=Path, required=True)
    build.add_argument("--training-result", type=Path)
    build.add_argument("--artifacts", type=Path)
    build.add_argument("--dataset-metadata", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--provider-repository")
    build.add_argument("--pipeline-root", type=Path, default=Path(__file__).resolve().parents[2])
    build.add_argument("--pipeline-repository")
    build.add_argument("--pipeline-commit")

    verify = commands.add_parser("verify", help="校验已有 voice-bundle.json 及全部资产")
    verify.add_argument("manifest", type=Path)
    commands.add_parser("schema", help="输出权威 JSON Schema")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            print(json.dumps(VoiceBundleManifest.model_json_schema(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "verify":
            manifest = load_voice_bundle(args.manifest)
            print(f"语音包校验通过: {manifest.bundle_id} ({len(manifest.files)} 个资产)")
            return 0
        pipeline = discover_pipeline_provenance(
            args.pipeline_root,
            repository=args.pipeline_repository,
            commit=args.pipeline_commit,
        )
        result = build_voice_bundle(
            BundleBuildRequest(
                bundle_id=args.bundle_id,
                display_name=args.display_name,
                output_dir=args.output,
                selection_path=args.selection,
                reference_profile_path=args.reference_profile,
                rights_attestation_path=args.rights_attestation,
                training_plan_path=args.training_plan,
                training_result_path=args.training_result,
                artifacts_path=args.artifacts,
                dataset_metadata_path=args.dataset_metadata,
                provider_repository=args.provider_repository,
                pipeline=pipeline,
            )
        )
        print(f"语音包已构建并校验: {result.manifest_path}")
        return 0
    except (OSError, ValidationError, VoiceBundleError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
