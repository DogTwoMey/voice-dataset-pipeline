"""Scene profiles spanning synthesis sampling, references, and mastering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .emotion import EmotionPlan


class SceneName(StrEnum):
    SPEECH = "speech"
    SINGING = "singing"
    AUDIOBOOK = "audiobook"
    ASMR = "asmr"
    STAGE = "stage"


SCENE_ORDER = (
    SceneName.SPEECH,
    SceneName.SINGING,
    SceneName.AUDIOBOOK,
    SceneName.ASMR,
    SceneName.STAGE,
)


@dataclass(frozen=True, slots=True)
class SceneProfile:
    name: SceneName
    pace_scale: float
    top_k_scale: float
    top_p_delta: float
    temperature_scale: float
    preferred_clusters: tuple[str, ...]
    description: str
    capability_note: str = ""

    def apply(self, plan: EmotionPlan) -> EmotionPlan:
        values = plan.model_dump()
        values.update(
            pace=max(0.5, min(2.0, plan.pace * self.pace_scale)),
            top_k=max(1, min(100, round(plan.top_k * self.top_k_scale))),
            top_p=max(0.01, min(1.0, plan.top_p + self.top_p_delta)),
            temperature=max(0.01, min(2.0, plan.temperature * self.temperature_scale)),
            rationale=_append_rationale(plan.rationale, f"scene={self.name.value}"),
        )
        return EmotionPlan.model_validate(values)


_SCENE_PROFILES = {
    SceneName.SPEECH: SceneProfile(
        name=SceneName.SPEECH,
        pace_scale=1.0,
        top_k_scale=1.0,
        top_p_delta=0.0,
        temperature_scale=1.0,
        preferred_clusters=("conversational", "calm_soft"),
        description="Natural conversational speech; the default scene.",
    ),
    SceneName.SINGING: SceneProfile(
        name=SceneName.SINGING,
        pace_scale=1.0 / 1.04,
        top_k_scale=0.90,
        top_p_delta=-0.01,
        temperature_scale=0.95,
        preferred_clusters=("singing", "calm_soft", "conversational"),
        description="Conservative vocal delivery and mastering for a supplied singing reference.",
        capability_note=(
            "GPT-SoVITS text inference does not generate melody or score timing; "
            "without a singing reference this remains speech prosody."
        ),
    ),
    SceneName.AUDIOBOOK: SceneProfile(
        name=SceneName.AUDIOBOOK,
        pace_scale=0.98 / 1.04,
        top_k_scale=0.90,
        top_p_delta=-0.01,
        temperature_scale=0.95,
        preferred_clusters=("calm_soft", "conversational"),
        description="Stable, slightly slower long-form narration.",
    ),
    SceneName.ASMR: SceneProfile(
        name=SceneName.ASMR,
        pace_scale=0.97 / 1.04,
        top_k_scale=0.90,
        top_p_delta=-0.02,
        temperature_scale=0.92,
        preferred_clusters=("calm_soft", "whisper"),
        description=(
            "Soft-spoken intimate delivery that preserves clarity, breath, and micro-dynamics."
        ),
    ),
    SceneName.STAGE: SceneProfile(
        name=SceneName.STAGE,
        pace_scale=1.02 / 1.04,
        top_k_scale=1.0,
        top_p_delta=0.0,
        temperature_scale=1.0,
        preferred_clusters=("bright_playful", "conversational"),
        description="Projected dramatic delivery with conservative presence mastering.",
    ),
}


def scene_profile(scene: SceneName | str) -> SceneProfile:
    name = scene if isinstance(scene, SceneName) else SceneName(scene.strip().lower())
    return _SCENE_PROFILES[name]


def apply_scene(plan: EmotionPlan, scene: SceneName | str) -> EmotionPlan:
    return scene_profile(scene).apply(plan)


def _append_rationale(current: str, value: str) -> str:
    return f"{current}; {value}" if current else value
