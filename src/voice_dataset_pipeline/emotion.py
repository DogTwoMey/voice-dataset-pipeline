"""Target-text emotion planning for backend-neutral synthesis."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import Field

from .models import StrictModel


class EmotionPlan(StrictModel):
    """Backend-neutral speaking intent plus conservative GPT sampling hints."""

    emotion: str = "neutral"
    intensity: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.5, ge=0, le=1)
    pace: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-1, le=1)
    energy: float = Field(default=0.0, ge=-1, le=1)
    pause_style: str = "natural"
    top_k: int = Field(default=15, ge=1, le=100)
    top_p: float = Field(default=0.95, gt=0, le=1)
    temperature: float = Field(default=0.9, gt=0, le=2)
    rationale: str = ""


class TextEmotionAnalyzer(Protocol):
    def analyze(self, text: str) -> EmotionPlan: ...


_PROFILES: dict[str, dict[str, float | int | str]] = {
    "neutral": {
        "pace": 1.0,
        "pitch": 0.0,
        "energy": 0.0,
        "top_k": 15,
        "top_p": 0.95,
        "temperature": 0.9,
    },
    "happy": {
        "pace": 1.04,
        "pitch": 0.2,
        "energy": 0.2,
        "top_k": 20,
        "top_p": 0.96,
        "temperature": 1.0,
    },
    "sad": {
        "pace": 0.92,
        "pitch": -0.15,
        "energy": -0.25,
        "top_k": 12,
        "top_p": 0.92,
        "temperature": 0.82,
    },
    "angry": {
        "pace": 1.02,
        "pitch": 0.05,
        "energy": 0.35,
        "top_k": 18,
        "top_p": 0.94,
        "temperature": 0.95,
    },
    "surprised": {
        "pace": 1.03,
        "pitch": 0.3,
        "energy": 0.15,
        "top_k": 20,
        "top_p": 0.97,
        "temperature": 1.05,
    },
    "fearful": {
        "pace": 1.06,
        "pitch": 0.2,
        "energy": -0.1,
        "top_k": 16,
        "top_p": 0.94,
        "temperature": 0.95,
    },
    "disgusted": {
        "pace": 0.96,
        "pitch": -0.1,
        "energy": 0.1,
        "top_k": 15,
        "top_p": 0.93,
        "temperature": 0.9,
    },
}


def plan_for(
    emotion: str, *, intensity: float = 0.5, confidence: float = 1.0, rationale: str = ""
) -> EmotionPlan:
    normalized = emotion.strip().lower()
    profile = _PROFILES.get(normalized, _PROFILES["neutral"])
    return EmotionPlan(
        emotion=normalized if normalized in _PROFILES else "neutral",
        intensity=intensity,
        confidence=confidence,
        rationale=rationale,
        **profile,
    )


class RuleBasedEmotionAnalyzer:
    """Offline fallback. It is intentionally conservative and deterministic."""

    _KEYWORDS = {
        "happy": ("开心", "高兴", "太好了", "真棒", "谢谢", "欢迎", "喜欢", "哈哈"),
        "sad": ("难过", "悲伤", "遗憾", "对不起", "失去", "哭", "离开"),
        "angry": ("生气", "愤怒", "可恶", "住手", "闭嘴", "绝不", "混蛋"),
        "fearful": ("害怕", "恐惧", "小心", "危险", "别过来"),
        "surprised": ("竟然", "怎么会", "真的吗", "什么", "居然"),
        "disgusted": ("恶心", "讨厌", "厌恶"),
    }

    def analyze(self, text: str) -> EmotionPlan:
        value = text.strip()
        if not value:
            raise ValueError("target text is empty")
        scores = {
            emotion: sum(value.count(word) for word in words)
            for emotion, words in self._KEYWORDS.items()
        }
        emotion, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            emotion = "surprised" if "？" in value or "?" in value else "neutral"
        punctuation = len(re.findall(r"[!?！？]", value))
        intensity = min(1.0, 0.35 + score * 0.2 + punctuation * 0.1)
        return plan_for(
            emotion,
            intensity=intensity,
            confidence=0.45 if score == 0 else min(0.85, 0.55 + score * 0.1),
            rationale="offline deterministic fallback",
        )


@dataclass(frozen=True, slots=True)
class OpenAICompatibleEmotionAnalyzer:
    """Small JSON-only client for OpenAI-compatible chat gateways."""

    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 60.0

    def analyze(self, text: str) -> EmotionPlan:
        value = text.strip()
        if not value:
            raise ValueError("target text is empty")
        if not self.api_key.strip():
            raise ValueError("emotion API key is empty")
        url = self.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        schema = {
            "emotion": "neutral|happy|sad|angry|surprised|fearful|disgusted",
            "intensity": "number 0..1",
            "confidence": "number 0..1",
            "pace": "number 0.75..1.25",
            "pitch": "number -1..1",
            "energy": "number -1..1",
            "pause_style": "natural|short|long|dramatic",
            "rationale": "short string",
        }
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You plan acting style for Chinese character TTS. Return JSON only. "
                    f"Schema: {json.dumps(schema, ensure_ascii=False)}",
                },
                {"role": "user", "content": value},
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise ValueError(f"emotion gateway HTTP {exc.code}: {detail}") from exc
        content = body["choices"][0]["message"]["content"]
        raw = json.loads(content)
        if not isinstance(raw, dict):
            raise ValueError("invalid emotion gateway response: content must be a JSON object")
        try:
            base = plan_for(
                str(raw.get("emotion", "neutral")),
                intensity=raw.get("intensity", 0.5),
                confidence=raw.get("confidence", 0.5),
                rationale=str(raw.get("rationale", "")),
            )
            validated = base.model_dump()
            validated.update(
                {key: raw[key] for key in ("pace", "pitch", "energy", "pause_style") if key in raw}
            )
            return EmotionPlan.model_validate(validated)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid emotion gateway response: {exc}") from exc
