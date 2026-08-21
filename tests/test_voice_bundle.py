from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from voice_dataset_pipeline.voice_bundle import (
    BundleBuildRequest,
    CodeProvenance,
    VoiceBundleError,
    VoiceBundleIntegrityError,
    VoiceBundleManifest,
    build_voice_bundle,
    load_voice_bundle,
    main,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(32_000)
        stream.writeframes(b"\x00\x00" * 320)


def _request(
    tmp_path: Path,
    *,
    training_allowed: bool = True,
    local_inference_allowed: bool = True,
    rights_dataset_fingerprint: str = "7" * 64,
    rights_training_plan_fingerprint: str = "1" * 64,
) -> BundleBuildRequest:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    gpt = inputs / "selected.ckpt"
    sovits = inputs / "selected.pth"
    reference = inputs / "neutral.wav"
    gpt.write_bytes(b"gpt-selected-checkpoint")
    sovits.write_bytes(b"sovits-selected-checkpoint")
    _write_wav(reference)

    selection = inputs / "selection.json"
    _write_json(
        selection,
        {
            "goal": "人工与固定参数评估后显式选择",
            "training_plan_fingerprint": "1" * 64,
            "evaluation_fingerprint": "6" * 64,
            "selected": {
                "gpt_path": str(gpt),
                "gpt_sha256": _sha256(gpt),
                "sovits_path": str(sovits),
                "sovits_sha256": _sha256(sovits),
            },
        },
    )
    profile = inputs / "reference-profile.json"
    _write_json(
        profile,
        {
            "schema_version": 1,
            "default": "neutral",
            "items": {
                "neutral": {
                    "description": "中性日常对话",
                    "auto_enabled": True,
                    "audio_path": "neutral.wav",
                    "prompt_text": "你好，很高兴见到你。",
                    "prompt_lang": "zh",
                    "sha256": _sha256(reference),
                }
            },
        },
    )
    rights = inputs / "rights-attestation.json"
    _write_json(
        rights,
        {
            "schema_version": 1,
            "attestation_id": "local-test-001",
            "subject": "测试声音",
            "dataset_fingerprint": rights_dataset_fingerprint,
            "training_plan_fingerprint": rights_training_plan_fingerprint,
            "rights_basis": "explicit_permission",
            "rights_holder": "测试权利人",
            "evidence_reference": "private://rights/local-test-001",
            "voice_subject_authorization": "explicit_permission",
            "training_allowed": training_allowed,
            "local_inference_allowed": local_inference_allowed,
            "attested_by": "测试人员",
            "attested_at": "2026-08-21T12:00:00+08:00",
        },
    )
    plan = inputs / "training-plan.json"
    _write_json(
        plan,
        {
            "schema_version": 1,
            "backend": "gpt_sovits",
            "fingerprint": "1" * 64,
            "metadata": {
                "model_version": "v2ProPlus",
                "provider": {
                    "repository": "https://github.com/RVC-Boss/GPT-SoVITS",
                    "git_head": "2" * 40,
                    "git_tracked_diff_sha256": "3" * 64,
                    "hashes": {"GPT_SoVITS/s1_train.py": "4" * 64},
                },
                "dataset": {
                    "dataset_list_sha256": "9" * 64,
                    "items": 1,
                },
            },
        },
    )
    metadata = inputs / "metadata.json"
    _write_json(
        metadata,
        {
            "format_version": 2,
            "fingerprint": "7" * 64,
            "manifest_sha256": "8" * 64,
            "dataset_list_sha256": "9" * 64,
            "included": 1,
            "allow_unreviewed": False,
        },
    )
    return BundleBuildRequest(
        bundle_id="example.voice.v1",
        display_name="示例角色语音",
        output_dir=tmp_path / "bundle",
        selection_path=selection,
        reference_profile_path=profile,
        rights_attestation_path=rights,
        training_plan_path=plan,
        dataset_metadata_path=metadata,
        pipeline=CodeProvenance(
            repository="https://github.com/example/voice-dataset-pipeline",
            commit="a" * 40,
            dirty_diff_sha256=None,
            source_snapshot_sha256="b" * 64,
        ),
    )


def test_builds_portable_bundle_from_explicit_selection(tmp_path: Path) -> None:
    request = _request(tmp_path)

    result = build_voice_bundle(request)

    manifest = load_voice_bundle(result.manifest_path)
    assert manifest.schema_version == 2
    assert manifest.bundle_id == "example.voice.v1"
    assert manifest.display_name == "示例角色语音"
    assert manifest.engine.api == "api_v2"
    assert manifest.engine.model_version == "v2ProPlus"
    assert manifest.assets.gpt.path == "models/gpt.ckpt"
    assert manifest.assets.sovits.path == "models/sovits.pth"
    assert manifest.references.default == "neutral"
    assert manifest.references.items["neutral"].audio == "references/neutral.wav"
    assert manifest.references.items["neutral"].auto_enabled is True
    assert manifest.distribution.model_allowed is False
    assert manifest.distribution.reference_audio_allowed is False
    assert manifest.distribution.source_dataset_included is False
    assert manifest.rights.dataset_fingerprint == manifest.provenance.dataset.fingerprint
    assert (
        manifest.rights.training_plan_fingerprint == manifest.provenance.training.plan_fingerprint
    )
    assert (
        manifest.provenance.selection.training_plan_fingerprint
        == manifest.provenance.training.plan_fingerprint
    )
    assert {item.path for item in manifest.files} == {
        "models/gpt.ckpt",
        "models/sovits.pth",
        "references/neutral.wav",
    }
    serialized = result.manifest_path.read_text(encoding="utf-8")
    assert json.loads(serialized)["engine"]["api"] == "api_v2"
    assert str(tmp_path) not in serialized
    assert "evidence_reference" not in serialized
    assert "rights_holder" not in serialized
    assert '"subject"' not in serialized


def test_rejects_rights_without_local_inference_authorization(tmp_path: Path) -> None:
    request = _request(tmp_path, local_inference_allowed=False)

    with pytest.raises(VoiceBundleError, match="本地推理"):
        build_voice_bundle(request)

    assert not request.output_dir.exists()


def test_rejects_rights_without_training_authorization(tmp_path: Path) -> None:
    request = _request(tmp_path, training_allowed=False)

    with pytest.raises(VoiceBundleError, match="允许训练"):
        build_voice_bundle(request)

    assert not request.output_dir.exists()


def test_rejects_rights_attestation_for_another_dataset(tmp_path: Path) -> None:
    request = _request(tmp_path, rights_dataset_fingerprint="0" * 64)

    with pytest.raises(VoiceBundleError, match="dataset_fingerprint"):
        build_voice_bundle(request)

    assert not request.output_dir.exists()


def test_rejects_rights_attestation_for_another_training_plan(tmp_path: Path) -> None:
    request = _request(tmp_path, rights_training_plan_fingerprint="0" * 64)

    with pytest.raises(VoiceBundleError, match="training_plan_fingerprint"):
        build_voice_bundle(request)

    assert not request.output_dir.exists()


def test_rejects_implicit_or_incomplete_checkpoint_selection(tmp_path: Path) -> None:
    request = _request(tmp_path)
    _write_json(request.selection_path, {"selected": {"gpt_path": "selected.ckpt"}})

    with pytest.raises(VoiceBundleError, match="显式包含 gpt_path 和 sovits_path"):
        build_voice_bundle(request)


def test_rejects_selection_without_checkpoint_hashes(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = json.loads(request.selection_path.read_text(encoding="utf-8"))
    del payload["selected"]["gpt_sha256"]
    _write_json(request.selection_path, payload)

    with pytest.raises(VoiceBundleError, match="有效的 gpt_sha256"):
        build_voice_bundle(request)


def test_rejects_selection_for_another_training_plan(tmp_path: Path) -> None:
    request = _request(tmp_path)
    payload = json.loads(request.selection_path.read_text(encoding="utf-8"))
    payload["training_plan_fingerprint"] = "0" * 64
    _write_json(request.selection_path, payload)

    with pytest.raises(VoiceBundleError, match="selection.training_plan_fingerprint"):
        build_voice_bundle(request)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_list_sha256", "0" * 64, "dataset_list_sha256"),
        ("included", 2, "dataset.items"),
    ],
)
def test_rejects_dataset_metadata_not_bound_to_training_plan(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    request = _request(tmp_path)
    payload = json.loads(request.dataset_metadata_path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(request.dataset_metadata_path, payload)

    with pytest.raises(VoiceBundleError, match=message):
        build_voice_bundle(request)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.ckpt",
        "C:/outside.ckpt",
        "models\\gpt.ckpt",
        "models/../gpt.ckpt",
        "models//gpt.ckpt",
        "models/con.ckpt",
        "models/gpt.ckpt:stream",
    ],
)
def test_loader_rejects_nonportable_or_escaping_paths(tmp_path: Path, unsafe_path: str) -> None:
    result = build_voice_bundle(_request(tmp_path))
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["assets"]["gpt"]["path"] = unsafe_path
    payload["files"][0]["path"] = unsafe_path
    _write_json(result.manifest_path, payload)

    with pytest.raises(VoiceBundleIntegrityError, match="schema 校验失败"):
        load_voice_bundle(result.manifest_path)


def test_loader_detects_tampering_and_unlisted_files(tmp_path: Path) -> None:
    result = build_voice_bundle(_request(tmp_path))
    gpt = result.root / "models" / "gpt.ckpt"
    gpt.write_bytes(b"tampered")

    with pytest.raises(VoiceBundleIntegrityError, match="字节数不符|SHA-256 不符"):
        load_voice_bundle(result.manifest_path)

    gpt.write_bytes(b"gpt-selected-checkpoint")
    (result.root / "raw-dataset.wav").write_bytes(b"must-not-be-packaged")
    with pytest.raises(VoiceBundleIntegrityError, match="文件白名单不一致"):
        load_voice_bundle(result.manifest_path)


@pytest.mark.parametrize("conflicting_path", ["models/sovits.pth", "MODELS/SOVITS.PTH"])
def test_loader_rejects_model_asset_path_collisions(tmp_path: Path, conflicting_path: str) -> None:
    result = build_voice_bundle(_request(tmp_path))
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["assets"]["gpt"]["path"] = conflicting_path
    payload["files"][0]["path"] = conflicting_path
    _write_json(result.manifest_path, payload)

    with pytest.raises(VoiceBundleIntegrityError, match="Windows 上冲突"):
        load_voice_bundle(result.manifest_path)


@pytest.mark.parametrize("missing_engine_field", [None, "name", "api"])
def test_loader_rejects_legacy_version_or_missing_fixed_engine_fields(
    tmp_path: Path, missing_engine_field: str | None
) -> None:
    result = build_voice_bundle(_request(tmp_path))
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if missing_engine_field is None:
        payload["schema_version"] = 1
    else:
        del payload["engine"][missing_engine_field]
    _write_json(result.manifest_path, payload)

    with pytest.raises(VoiceBundleIntegrityError, match="schema 校验失败"):
        load_voice_bundle(result.manifest_path)


def test_schema_and_verify_cli_are_strict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = build_voice_bundle(_request(tmp_path))

    schema = VoiceBundleManifest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) >= {
        "schema_version",
        "bundle_id",
        "display_name",
        "engine",
        "assets",
        "references",
        "provenance",
        "rights",
        "distribution",
        "files",
    }
    assert set(schema["$defs"]["BundleEngine"]["required"]) == {
        "name",
        "api",
        "model_version",
    }
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "voice-bundle.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == schema
    assert main(["verify", str(result.manifest_path)]) == 0
    assert "语音包校验通过" in capsys.readouterr().out
