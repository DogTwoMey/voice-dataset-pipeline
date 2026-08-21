from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_dataset_pipeline.cli import main
from voice_dataset_pipeline.config import (
    InferenceConfig,
    config_layout,
    generate_default_config_layout,
    load_config,
    load_secrets,
    write_secrets_gitignore,
)
from voice_dataset_pipeline.models import ReviewDecision, ReviewState
from voice_dataset_pipeline.scenes import SceneName
from voice_dataset_pipeline.workspace import Workspace


def test_workspace_has_stable_layout_and_atomic_jsonl(tmp_path):
    workspace = Workspace.create(tmp_path / "job")

    assert workspace.paths.normalized_audio == workspace.root / "normalized"
    assert workspace.paths.clips_jsonl == workspace.root / "manifests" / "clips.jsonl"
    assert workspace.paths.review_json == workspace.root / "state" / "review.json"
    assert workspace.paths.training.is_dir()

    workspace.append_jsonl(workspace.paths.labels_jsonl, {"clip_id": "a", "value": 1})
    workspace.append_jsonl(workspace.paths.labels_jsonl, {"clip_id": "b", "value": 2})
    workspace.upsert_jsonl(
        workspace.paths.labels_jsonl,
        {"clip_id": "a", "value": 3},
        key="clip_id",
    )

    assert workspace.read_jsonl(workspace.paths.labels_jsonl) == [
        {"clip_id": "a", "value": 3},
        {"clip_id": "b", "value": 2},
    ]
    assert not list(workspace.root.rglob("*.tmp"))


def test_review_state_round_trips_and_open_can_require_existing(tmp_path):
    root = tmp_path / "job"
    workspace = Workspace.open(root, create=True)
    state = ReviewState(
        cursor=1,
        order=["clip-a", "clip-b"],
        decisions={
            "clip-a": ReviewDecision(
                clip_id="clip-a",
                emotion="calm_custom",
                cluster="soft",
                transcript="你好",
            )
        },
    )

    workspace.save_review(state)
    loaded = Workspace.open(root, create=False).load_review()

    assert loaded.cursor == 1
    assert loaded.decisions["clip-a"].emotion == "calm_custom"
    assert loaded.decisions["clip-a"].transcript == "你好"


def test_config_forbids_unknown_fields_and_api_keys(tmp_path):
    config_path = tmp_path / "unsafe.toml"
    config_path.write_text(
        '[gemini]\napi_key = "must-not-be-persisted"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="api_key"):
        load_config(config_path)


def test_init_generates_separate_project_and_gitignored_secret_configs(tmp_path):
    workspace = tmp_path / "job"

    assert main(["init", str(workspace)]) == 0

    layout = config_layout(workspace)
    assert layout.project == workspace.resolve() / "config/pipeline.toml"
    assert layout.secrets == workspace.resolve() / "secrets/credentials.toml"
    assert layout.project.is_file()
    assert layout.secrets.is_file()
    assert layout.project.parent != layout.secrets.parent
    assert load_config(layout.project).gemini.api_key_env == "GEMINI_API_KEY"
    assert load_secrets(layout.secrets).get("GEMINI_API_KEY") is None
    rules = layout.secrets_gitignore.read_text(encoding="utf-8").splitlines()
    assert "*" in rules
    assert "!.gitignore" in rules


def test_config_generator_preserves_files_unless_overwrite_is_explicit(tmp_path):
    layout = generate_default_config_layout(tmp_path)
    layout.project.write_text("# retained project\n", encoding="utf-8")
    layout.secrets.write_text(
        '[environment]\nGEMINI_API_KEY = "fixture-token"\n',
        encoding="utf-8",
    )

    generated = generate_default_config_layout(tmp_path)

    assert generated == layout
    assert layout.project.read_text(encoding="utf-8") == "# retained project\n"
    secrets = load_secrets(layout.secrets)
    assert secrets.get("GEMINI_API_KEY") == "fixture-token"
    assert "fixture-token" not in repr(secrets)


def test_secrets_config_rejects_non_environment_variable_names(tmp_path):
    path = tmp_path / "credentials.toml"
    path.write_text('[environment]\n"gemini.api-key" = "bad"\n', encoding="utf-8")

    with pytest.raises(ValidationError, match="invalid environment variable"):
        load_secrets(path)


def test_secrets_gitignore_rejects_rules_that_reinclude_credentials(tmp_path):
    path = tmp_path / "secrets/.gitignore"
    path.parent.mkdir()
    path.write_text("*\n!.gitignore\n!credentials.toml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain only"):
        write_secrets_gitignore(path)


def test_repository_project_and_secret_examples_remain_valid():
    root = Path(__file__).resolve().parents[1]

    project = load_config(root / "examples/project/pipeline.toml")
    secrets = load_secrets(root / "examples/secrets/credentials.toml.example")

    assert project.gemini.api_key_env == "GEMINI_API_KEY"
    assert secrets.get("GEMINI_API_KEY") is None
    assert project.scenes.default == SceneName.SPEECH
    assert project.postprocess.sox.enabled is False
    assert load_config(root / "examples/project/dracaene.toml").scenes.default == SceneName.SPEECH


def test_inference_emotion_overrides_are_validated() -> None:
    config = InferenceConfig.model_validate(
        {"emotion_overrides": {"neutral": {"top_k": 20, "top_p": 0.96, "pace": 1.04}}}
    )

    assert config.emotion_overrides["neutral"].top_k == 20
    assert config.emotion_overrides["neutral"].pace == 1.04
    with pytest.raises(ValidationError, match="less than or equal to 2"):
        InferenceConfig.model_validate({"emotion_overrides": {"neutral": {"temperature": 3}}})
