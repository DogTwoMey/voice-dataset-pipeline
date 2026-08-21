from __future__ import annotations

import pytest

from voice_dataset_pipeline.emotion import EmotionPlan
from voice_dataset_pipeline.scenes import SCENE_ORDER, SceneName, apply_scene, scene_profile


def test_scene_catalog_is_complete_and_stable() -> None:
    assert SCENE_ORDER == (
        SceneName.SPEECH,
        SceneName.SINGING,
        SceneName.AUDIOBOOK,
        SceneName.ASMR,
        SceneName.STAGE,
    )
    assert all(scene_profile(scene).preferred_clusters for scene in SCENE_ORDER)


@pytest.mark.parametrize(
    ("scene", "pace", "top_k", "top_p", "temperature"),
    [
        ("speech", 1.04, 20, 0.96, 1.0),
        ("singing", 1.0, 18, 0.95, 0.95),
        ("audiobook", 0.98, 18, 0.95, 0.95),
        ("asmr", 0.97, 18, 0.94, 0.92),
        ("stage", 1.02, 20, 0.96, 1.0),
    ],
)
def test_scene_applies_only_effective_gpt_controls(
    scene: str,
    pace: float,
    top_k: int,
    top_p: float,
    temperature: float,
) -> None:
    base = EmotionPlan(
        emotion="neutral",
        pace=1.04,
        pitch=0.2,
        energy=0.3,
        pause_style="expressive",
        top_k=20,
        top_p=0.96,
        temperature=1.0,
    )

    result = apply_scene(base, scene)

    assert result.pace == pytest.approx(pace)
    assert result.top_k == top_k
    assert result.top_p == pytest.approx(top_p)
    assert result.temperature == pytest.approx(temperature)
    assert result.pitch == base.pitch
    assert result.energy == base.energy
    assert result.pause_style == base.pause_style
    assert f"scene={scene}" in result.rationale


def test_singing_profile_states_provider_limit() -> None:
    assert "does not generate melody" in scene_profile(SceneName.SINGING).capability_note
