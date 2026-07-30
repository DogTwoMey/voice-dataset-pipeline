from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from voice_dataset_pipeline.errors import ConfigurationError
from voice_dataset_pipeline.training import (
    CommandSpec,
    GPTSoVITSAdapter,
    RVCAdapter,
    TrainingPlan,
    _verify_rvc_model,
)


def _file(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _wav(path: Path, *, seconds: float = 0.05, sample_rate: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(b"\x01\x00" * frames)
    return path


def _row(path: Path, *, clip_id: str = "clip-a", text: str = "你好") -> dict[str, object]:
    return {
        "clip_id": clip_id,
        "audio_path": str(path),
        "text": text,
        "language": "zh",
        "speaker": "tester",
        "emotion": "neutral",
        "reviewed": True,
        "include": True,
    }


def _gpt_repository(root: Path) -> Path:
    _file(root / "python.exe")
    _file(
        root / "GPT_SoVITS/configs/s2v2ProPlus.json",
        json.dumps({"train": {}, "model": {}, "data": {}}).encode(),
    )
    _file(
        root / "GPT_SoVITS/configs/s1longer-v2.yaml",
        b"train:\n  batch_size: 1\n",
    )
    for relative in (
        "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
        "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
        "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
        "GPT_SoVITS/prepare_datasets/1-get-text.py",
        "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py",
        "GPT_SoVITS/prepare_datasets/2-get-sv.py",
        "GPT_SoVITS/prepare_datasets/3-get-semantic.py",
        "GPT_SoVITS/s2_train.py",
        "GPT_SoVITS/s1_train.py",
    ):
        _file(root / relative)
    _file(root / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/config.json")
    _file(root / "GPT_SoVITS/pretrained_models/chinese-hubert-base/config.json")
    return root


def _rvc_repository(root: Path) -> Path:
    _file(root / "python.exe")
    for relative in (
        "train/preprocess.py",
        "train/dataset/extract_f0.py",
        "train/dataset/extract_hubert_feature.py",
        "train/train.py",
        "train/train_index.py",
        "assets/pretrained_v2/f0G48k.pth",
        "assets/pretrained_v2/f0D48k.pth",
        "assets/rmvpe/rmvpe.pt",
        "assets/hubert_base/config.json",
        "assets/hubert_base/preprocessor_config.json",
        "assets/hubert_base/pytorch_model.bin",
    ):
        _file(root / relative)
    _file(root / "configs/v2/48k.json", b'{"train":{},"data":{},"model":{}}')
    return root


def test_training_plan_is_inert_until_execute(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    artifact_manifest = tmp_path / "artifacts.json"
    code = (
        "from pathlib import Path; import hashlib,json,sys;"
        "p=Path(sys.argv[1]); p.write_bytes(b'ok');"
        "Path(sys.argv[2]).write_text(json.dumps({'artifacts':["
        "{'kind':'fixture','path':str(p),'sha256':hashlib.sha256(b'ok').hexdigest()}]}))"
    )
    command = CommandSpec(
        "fixture",
        [sys.executable, "-c", code, str(output), str(artifact_manifest)],
        tmp_path,
        expected_files=[output, artifact_manifest],
    )
    plan = TrainingPlan(
        backend="fixture",
        experiment="fixture",
        run_dir=tmp_path,
        fingerprint="a" * 64,
        commands=[command],
        metadata={},
        plan_manifest=tmp_path / "training-plan.json",
        artifact_manifest=artifact_manifest,
    )
    plan.write_manifest()

    assert not output.exists()
    result = plan.execute()

    assert output.read_bytes() == b"ok"
    assert result["artifacts"][0]["sha256"] == __import__("hashlib").sha256(b"ok").hexdigest()
    command_manifest = json.loads(plan.plan_manifest.read_text(encoding="utf-8"))
    assert command_manifest["commands"][0]["shell"] is False
    assert command_manifest["commands"][0]["argv"] == command.argv


def test_manifest_gate_rejects_unreviewed_rows(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    wav = _wav(tmp_path / "clip.wav")
    row = _row(wav)
    row["reviewed"] = False
    adapter = GPTSoVITSAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "fixture",
        },
        tmp_path / "workspace",
        python_probe=None,
    )

    with pytest.raises(ConfigurationError, match="not reviewed"):
        adapter.plan([row])


def test_manifest_gate_rejects_duplicate_basenames(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    first = _wav(tmp_path / "a/clip.wav")
    second = _wav(tmp_path / "b/clip.wav")
    adapter = GPTSoVITSAdapter(
        {"repository": repository, "python": repository / "python.exe"},
        tmp_path / "workspace",
        python_probe=None,
    )

    with pytest.raises(ConfigurationError, match="basenames must be unique"):
        adapter.plan([_row(first, clip_id="one"), _row(second, clip_id="two")])


def test_gpt_sovits_plan_uses_provider_cwd_env_and_argv_lists(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    wav = _wav(tmp_path / "clip.wav")
    adapter = GPTSoVITSAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "character_v2proplus",
            "model_version": "v2ProPlus",
        },
        tmp_path / "workspace",
        python_probe=None,
    )

    plan = adapter.plan([_row(wav)])

    assert plan.backend == "gpt_sovits"
    assert [command.name for command in plan.commands] == [
        "prepare-text",
        "prepare-hubert",
        "prepare-speaker",
        "prepare-semantic",
        "verify-prepared-dataset",
        "train-sovits",
        "train-gpt",
        "verify-models",
    ]
    assert all(command.cwd == repository.resolve() for command in plan.commands)
    assert all(isinstance(command.argv, list) for command in plan.commands)
    assert plan.commands[0].argv[-1].endswith("1-get-text.py")
    assert str(repository / "GPT_SoVITS") in plan.commands[0].env["PYTHONPATH"]
    verifier = next(
        command for command in plan.commands if command.name == "verify-prepared-dataset"
    )
    assert verifier.argv[1:3] == ["-m", "voice_dataset_pipeline.training"]
    dataset_list = Path(plan.metadata["dataset"]["dataset_list"])
    assert dataset_list.read_text(encoding="utf-8") == (f"{wav.resolve()}|tester|zh|你好\n")
    assert plan.plan_manifest.is_file()
    assert not plan.artifact_manifest.exists()


def test_rvc_plan_uses_module_entrypoints_and_requires_compact_export(tmp_path: Path) -> None:
    repository = _rvc_repository(tmp_path / "rvc")
    wav = _wav(tmp_path / "clip.wav")
    adapter = RVCAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "character_rvc_v2",
            "version": "v2",
            "sample_rate": "48k",
            "epochs": 10,
            "save_every": 5,
        },
        tmp_path / "workspace",
        python_probe=None,
    )

    plan = adapter.plan([_row(wav)])

    module_commands = {
        command.name: command.argv for command in plan.commands if "-m" in command.argv
    }
    assert module_commands["preprocess"][1:3] == ["-m", "train.preprocess"]
    assert module_commands["extract-f0"][1:3] == ["-m", "train.dataset.extract_f0"]
    assert module_commands["extract-hubert"][1:3] == [
        "-m",
        "train.dataset.extract_hubert_feature",
    ]
    assert module_commands["train"][1:3] == ["-m", "train.train"]
    assert "-sw" in module_commands["train"]
    assert module_commands["train"][module_commands["train"].index("-sw") + 1] == "1"
    train = next(command for command in plan.commands if command.name == "train")
    assert train.expected_files == [(repository / "assets/weights/character_rvc_v2.pth").resolve()]
    assert train.cwd == repository.resolve()
    assert str(repository.resolve()) in train.env["PYTHONPATH"].split(os.pathsep)
    manifest_builder = next(
        command for command in plan.commands if command.name == "build-training-manifest"
    )
    assert manifest_builder.argv[1:3] == ["-m", "voice_dataset_pipeline.training"]
    dataset = json.loads(Path(plan.metadata["dataset"]["manifest"]).read_text(encoding="utf-8"))
    assert Path(dataset[0]["dataset_wav"]).is_file()
    assert plan.plan_manifest.is_file()
    assert not plan.artifact_manifest.exists()


def test_rvc_verifier_rejects_training_checkpoint_without_inference_model(
    tmp_path: Path,
) -> None:
    checkpoint = _file(tmp_path / "G_2333333.pth")

    with pytest.raises(ConfigurationError, match="inference model is missing"):
        _verify_rvc_model(
            tmp_path / "character.pth",
            "v2",
            "48k",
            tmp_path / "artifacts.json",
        )

    assert checkpoint.is_file()


def test_changed_gpt_input_is_rejected_before_rewriting_staging(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    wav = _wav(tmp_path / "clip.wav")
    adapter = GPTSoVITSAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "guarded",
        },
        tmp_path / "workspace",
        python_probe=None,
    )
    first = adapter.plan([_row(wav, text="first")])
    selected = Path(first.metadata["dataset"]["selected_manifest"])
    before = selected.read_bytes()

    with pytest.raises(ConfigurationError, match="fingerprint changed"):
        adapter.plan([_row(wav, text="second")])

    assert selected.read_bytes() == before


def test_changed_gpt_provider_script_invalidates_existing_plan(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    wav = _wav(tmp_path / "clip.wav")
    adapter = GPTSoVITSAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "guarded-provider",
        },
        tmp_path / "workspace",
        python_probe=None,
    )
    adapter.plan([_row(wav)])
    script = repository / "GPT_SoVITS/prepare_datasets/1-get-text.py"
    script.write_bytes(b"locally modified provider script")

    with pytest.raises(ConfigurationError, match="fingerprint changed"):
        adapter.plan([_row(wav)])


def test_changed_tracked_provider_helper_invalidates_existing_plan(tmp_path: Path) -> None:
    repository = _gpt_repository(tmp_path / "gpt")
    helper = _file(repository / "GPT_SoVITS/provider_helper.py", b"original")
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    wav = _wav(tmp_path / "clip.wav")
    adapter = GPTSoVITSAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "guarded-helper",
        },
        tmp_path / "workspace",
        python_probe=None,
    )
    adapter.plan([_row(wav)])
    helper.write_bytes(b"locally modified tracked helper")

    with pytest.raises(ConfigurationError, match="fingerprint changed"):
        adapter.plan([_row(wav)])


def test_rvc_plan_rejects_nonempty_provider_experiment(tmp_path: Path) -> None:
    repository = _rvc_repository(tmp_path / "rvc")
    wav = _wav(tmp_path / "clip.wav")
    _file(repository / "logs/stale/0_gt_wavs/old.wav")
    adapter = RVCAdapter(
        {
            "repository": repository,
            "python": repository / "python.exe",
            "experiment": "stale",
        },
        tmp_path / "workspace",
        python_probe=None,
    )

    with pytest.raises(ConfigurationError, match="not empty"):
        adapter.plan([_row(wav)])
