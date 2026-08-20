from __future__ import annotations

import json

import pytest

from voice_dataset_pipeline.emotion import (
    OpenAICompatibleEmotionAnalyzer,
    RuleBasedEmotionAnalyzer,
    plan_for,
)


class _GatewayResponse:
    def __init__(self, content: object) -> None:
        self.payload = json.dumps(
            {"choices": [{"message": {"content": json.dumps(content)}}]}
        ).encode()

    def __enter__(self) -> _GatewayResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_rule_analyzer_is_deterministic() -> None:
    plan = RuleBasedEmotionAnalyzer().analyze("太好了，谢谢你！")
    assert plan.emotion == "happy"
    assert 0 <= plan.intensity <= 1
    assert plan.pace > 1


def test_unknown_profile_falls_back_to_neutral() -> None:
    assert plan_for("not-a-label").emotion == "neutral"


@pytest.mark.parametrize("pace", [0, "not-a-number", 3])
def test_openai_analyzer_rejects_invalid_gateway_pace(
    monkeypatch: pytest.MonkeyPatch,
    pace: object,
) -> None:
    response = _GatewayResponse({"emotion": "happy", "pace": pace})
    monkeypatch.setattr(
        "voice_dataset_pipeline.emotion.urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )
    analyzer = OpenAICompatibleEmotionAnalyzer(
        base_url="https://example.invalid/v1",
        model="fixture",
        api_key="test-key",
    )

    with pytest.raises(ValueError, match="invalid emotion gateway response") as raised:
        analyzer.analyze("你好。")
    assert "pace" in str(raised.value)
