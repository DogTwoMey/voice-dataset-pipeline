from __future__ import annotations

import pytest
from pydantic import ValidationError

from voice_dataset_pipeline.config import load_config
from voice_dataset_pipeline.models import ReviewDecision, ReviewState
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
