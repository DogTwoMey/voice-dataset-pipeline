from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice_dataset_pipeline.errors import ExternalToolError
from voice_dataset_pipeline.gemini import (
    GeminiInteractions,
    GeminiLabelResult,
    GeminiSplitResult,
)
from voice_dataset_pipeline.models import SplitBackend


@dataclass
class _Uploaded:
    name: str
    uri: str
    mime_type: str
    state: object


class _FakeFiles:
    def __init__(self, *, initially_active: bool = False) -> None:
        state = "ACTIVE" if initially_active else "PROCESSING"
        self.uploaded = _Uploaded(
            name="files/test-upload",
            uri="gemini://files/test-upload",
            mime_type="audio/wav",
            state=SimpleNamespace(name=state),
        )
        self.upload_calls: list[str] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []

    def upload(self, *, file: str) -> _Uploaded:
        self.upload_calls.append(file)
        return self.uploaded

    def get(self, *, name: str) -> _Uploaded:
        self.get_calls.append(name)
        return _Uploaded(
            name=self.uploaded.name,
            uri=self.uploaded.uri,
            mime_type=self.uploaded.mime_type,
            state=SimpleNamespace(name="ACTIVE"),
        )

    def delete(self, *, name: str) -> None:
        self.delete_calls.append(name)


class _FakeInteractions:
    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.outputs.pop(0)))


class _FakeClient:
    def __init__(
        self,
        outputs: list[dict[str, object]],
        *,
        initially_active: bool = False,
    ) -> None:
        self.files = _FakeFiles(initially_active=initially_active)
        self.interactions = _FakeInteractions(outputs)


def _media_file(tmp_path: Path) -> Path:
    path = tmp_path / "voice.wav"
    path.write_bytes(b"not-decoded-by-this-test")
    return path


def test_split_uploads_polls_uses_schema_and_always_deletes(tmp_path: Path) -> None:
    path = _media_file(tmp_path)
    client = _FakeClient(
        [
            {
                "segments": [
                    {"start_ms": 100, "end_ms": 900},
                    {"start_ms": 1200, "end_ms": 2500},
                ]
            }
        ]
    )
    gemini = GeminiInteractions(
        model="gemini-test",
        client=client,
        max_retries=0,
        sleep=lambda _seconds: None,
    )

    segments = gemini.split(
        path=path,
        modality="audio",
        source_id="source-1",
        duration_seconds=3.0,
        min_segment_seconds=0.5,
        max_segment_seconds=10.0,
    )

    assert [(row.start_seconds, row.end_seconds) for row in segments] == [
        (0.1, 0.9),
        (1.2, 2.5),
    ]
    assert all(row.backend is SplitBackend.GEMINI for row in segments)
    assert client.files.upload_calls == [str(path.resolve())]
    assert client.files.get_calls == ["files/test-upload"]
    assert client.files.delete_calls == ["files/test-upload"]

    call = client.interactions.calls[0]
    assert call["model"] == "gemini-test"
    assert call["store"] is False
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": GeminiSplitResult.model_json_schema(),
    }
    media_part = call["input"][0]  # type: ignore[index]
    assert media_part["uri"] == "gemini://files/test-upload"
    assert media_part["type"] == "audio"


def test_split_rejects_overlapping_boundaries_and_deletes_upload(
    tmp_path: Path,
) -> None:
    path = _media_file(tmp_path)
    client = _FakeClient(
        [
            {
                "segments": [
                    {"start_ms": 100, "end_ms": 1000},
                    {"start_ms": 900, "end_ms": 1500},
                ]
            }
        ],
        initially_active=True,
    )
    gemini = GeminiInteractions(
        model="gemini-test",
        client=client,
        max_retries=0,
    )

    with pytest.raises(ExternalToolError):
        gemini.split(
            path=path,
            modality="audio",
            source_id="source-1",
            duration_seconds=2.0,
            min_segment_seconds=0.5,
            max_segment_seconds=10.0,
        )

    assert client.files.delete_calls == ["files/test-upload"]


def test_label_falls_back_to_unknown_emotion_and_sanitizes_cluster(
    tmp_path: Path,
) -> None:
    path = _media_file(tmp_path)
    client = _FakeClient(
        [
            {
                "transcript": "你好。",
                "emotion": "ecstatic",
                "cluster": "Warm-Voice !!",
                "confidence": 0.75,
                "rationale": "bright delivery",
            }
        ],
        initially_active=True,
    )
    gemini = GeminiInteractions(
        model="gemini-test",
        client=client,
        max_retries=0,
    )

    label = gemini.label(
        path=path,
        clip_id="clip-1",
        emotions=["neutral", "happy"],
        clusters=["warm_voice", "unknown"],
        language_hint="zh",
    )

    assert label.transcript == "你好。"
    assert label.emotion == "unknown"
    assert label.cluster == "warm_voice"
    assert label.model == "gemini-test"
    call = client.interactions.calls[0]
    assert call["store"] is False
    assert call["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": GeminiLabelResult.model_json_schema(),
    }
    assert client.files.delete_calls == ["files/test-upload"]
