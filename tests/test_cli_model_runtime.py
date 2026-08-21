from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from voice_dataset_pipeline import cli
from voice_dataset_pipeline import training as training_module
from voice_dataset_pipeline.scenes import SceneName


def test_model_register_parser_accepts_provider_python(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "model",
            "register",
            str(tmp_path),
            "--name",
            "character",
            "--repository",
            str(tmp_path / "provider"),
            "--python",
            str(tmp_path / "provider-python.exe"),
            "--gpt",
            str(tmp_path / "gpt.ckpt"),
            "--sovits",
            str(tmp_path / "sovits.pth"),
        ]
    )

    assert args.python == tmp_path / "provider-python.exe"


def test_synthesize_parser_accepts_scene_batch_and_sox(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        [
            "synthesize",
            str(tmp_path),
            "--text",
            "测试",
            "--output",
            str(tmp_path / "voice.wav"),
            "--scene",
            "all",
            "--mastering",
            "sox",
        ]
    )

    assert args.scene == "all"
    assert args.mastering == "sox"
    assert cli._scene_output(tmp_path / "voice.wav", SceneName.ASMR, multiple=True).name == (
        "voice.asmr.wav"
    )


def test_train_auto_registration_records_provider_python(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    dataset = tmp_path / "dataset"
    repository = tmp_path / "provider"
    workspace_root.mkdir()
    dataset.mkdir()
    repository.mkdir()
    manifest = dataset / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    provider_python = tmp_path / "provider-python.exe"
    gpt = tmp_path / "gpt.ckpt"
    sovits = tmp_path / "sovits.pth"
    for path in (provider_python, gpt, sovits):
        path.write_bytes(b"test")

    trainer = SimpleNamespace(
        repository=repository,
        python=provider_python,
        speaker="character",
        model_version="v2ProPlus",
        enabled=True,
        experiment_name="character-v1",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            enabled=True,
            gpt_sovits=trainer,
            rvc=SimpleNamespace(),
        )
    )

    class FakePlan:
        metadata = {
            "dataset": {"selected_sha256": "dataset-sha"},
            "provider": {
                "git_head": "provider-head",
                "git_tracked_diff_sha256": "provider-dirty",
                "hashes": {
                    "bert": "bert-sha",
                    "hubert": "hubert-sha",
                    "g2pw": "g2pw-sha",
                    "language_detector": "language-detector-sha",
                    "sv": "sv-sha",
                },
            },
        }

        @staticmethod
        def serialise():
            return {"commands": []}

        @staticmethod
        def execute():
            return {
                "artifacts": [
                    {"kind": "gpt", "path": str(gpt), "sha256": "gpt-sha"},
                    {"kind": "sovits", "path": str(sovits), "sha256": "sovits-sha"},
                ]
            }

    class FakeAdapter:
        def __init__(self, *_args):
            pass

        @staticmethod
        def plan(_manifest, *, experiment):
            assert experiment == "character-v1"
            return FakePlan()

    registered = {}

    class FakeRegistry:
        @staticmethod
        def register(record, *, activate):
            registered["record"] = record
            registered["activate"] = activate

    monkeypatch.setattr(cli, "_load_for", lambda *_args: config)
    monkeypatch.setattr(
        cli.Workspace,
        "open",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace(root=workspace_root)),
    )
    monkeypatch.setattr(cli, "_latest_export", lambda _workspace: dataset)
    monkeypatch.setattr(cli, "_verify_current_export", lambda *_args: None)
    monkeypatch.setattr(cli, "_registry_for", lambda *_args: FakeRegistry())
    monkeypatch.setattr(training_module, "GPTSoVITSAdapter", FakeAdapter)
    args = SimpleNamespace(
        workspace=workspace_root,
        config=None,
        dataset=None,
        trainer="gpt-sovits",
        execute=True,
        register_as="character",
        activate=True,
    )

    assert cli._train(args) == 0
    assert registered["record"].python == provider_python.resolve()
    assert registered["record"].dataset_fingerprint == "dataset-sha"
    assert registered["record"].gpt_weights_sha256 == "gpt-sha"
    assert registered["record"].sovits_weights_sha256 == "sovits-sha"
    assert registered["record"].provider_dirty_sha256 == "provider-dirty"
    assert registered["record"].provider_assets_sha256 == {
        "bert": "bert-sha",
        "hubert": "hubert-sha",
        "g2pw": "g2pw-sha",
        "language_detector": "language-detector-sha",
        "sv": "sv-sha",
    }
    assert registered["activate"] is True
