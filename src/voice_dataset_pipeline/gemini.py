"""Gemini Interactions API adapters for segmentation and clip labelling.

The SDK is imported lazily so every local-only command remains usable without
the optional ``gemini`` dependency.  Uploaded media is deleted in ``finally``
and interactions are stateless by default.
"""

from __future__ import annotations

import mimetypes
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigurationError, ExternalToolError
from .models import LabelRecord, Segment, SplitBackend


class _GeminiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeminiBoundary(_GeminiModel):
    """One speech-only interval returned by Gemini."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> GeminiBoundary:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        return self


class GeminiSplitResult(_GeminiModel):
    segments: list[GeminiBoundary]


class GeminiLabelResult(_GeminiModel):
    transcript: str
    emotion: str
    cluster: str
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""


def _mime_for(path: Path, modality: str) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "video/mp4" if modality == "video" else "audio/wav"


def _state_name(uploaded: Any) -> str:
    state = getattr(uploaded, "state", None)
    name = getattr(state, "name", state)
    return str(name or "").upper()


def _retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        status_number = int(status)
    except (TypeError, ValueError):
        status_number = 0
    if 400 <= status_number < 500 and status_number not in {408, 429}:
        return False
    text = str(error).lower()
    permanent_markers = (
        "invalid_request",
        "api key not valid",
        "current location",
        "available regions",
    )
    return not any(marker in text for marker in permanent_markers)


class GeminiInteractions:
    """Small, injectable wrapper around ``google.genai``.

    Parameters are plain values instead of the full pipeline configuration to
    keep this adapter easy to test and reuse.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        api_key: str | None = None,
        timeout_seconds: float = 120,
        max_retries: int = 3,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep
        if client is not None:
            self.client = client
            return

        resolved_api_key = (api_key or os.environ.get(api_key_env, "")).strip()
        if not resolved_api_key:
            raise ConfigurationError(
                f"未在 secrets/credentials.toml 或环境变量 {api_key_env} 中提供 API Key。"
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ConfigurationError(
                "缺少 Gemini 依赖；请执行 pip install 'voice-dataset-pipeline[gemini]'"
            ) from exc
        self.client = genai.Client(api_key=resolved_api_key)

    def _wait_until_active(self, uploaded: Any) -> Any:
        deadline = time.monotonic() + self.timeout_seconds
        current = uploaded
        while True:
            state = _state_name(current)
            if state in {"", "ACTIVE"}:
                return current
            if state == "FAILED":
                raise ExternalToolError(f"Gemini 文件处理失败: {getattr(current, 'name', '')}")
            if time.monotonic() >= deadline:
                raise ExternalToolError("等待 Gemini 文件处理超时")
            self._sleep(min(2.0, max(0.0, deadline - time.monotonic())))
            current = self.client.files.get(name=current.name)

    def _structured_media_call(
        self,
        *,
        path: Path,
        modality: str,
        prompt: str,
        result_type: type[_GeminiModel],
        validate_result: Callable[[_GeminiModel], None] | None = None,
    ) -> _GeminiModel:
        path = path.resolve()
        if not path.is_file():
            raise ConfigurationError(f"媒体文件不存在: {path}")

        uploaded: Any | None = None
        primary_error: BaseException | None = None
        try:
            uploaded = self.client.files.upload(file=str(path))
            uploaded = self._wait_until_active(uploaded)
            media_part = {
                "type": modality,
                "uri": uploaded.uri,
                "mime_type": getattr(uploaded, "mime_type", None) or _mime_for(path, modality),
            }
            attempt_prompt = prompt
            for attempt in range(self.max_retries + 1):
                try:
                    interaction = self.client.interactions.create(
                        model=self.model,
                        input=[media_part, {"type": "text", "text": attempt_prompt}],
                        response_format={
                            "type": "text",
                            "mime_type": "application/json",
                            "schema": result_type.model_json_schema(),
                        },
                        store=False,
                    )
                    parsed = result_type.model_validate_json(interaction.output_text)
                    if validate_result is not None:
                        validate_result(parsed)
                    return parsed
                except Exception as exc:
                    if attempt >= self.max_retries or not _retryable(exc):
                        raise
                    feedback = " ".join(str(exc).split())[:500]
                    attempt_prompt = (
                        f"{prompt}\n\nYour previous JSON response was rejected by the "
                        "deterministic "
                        f"validator: {feedback}. Return a corrected complete JSON response; "
                        "do not repeat the invalid boundary pattern."
                    )
                    self._sleep(min(8.0, 2.0**attempt))
            raise AssertionError("unreachable")
        except (ConfigurationError, ExternalToolError) as exc:
            primary_error = exc
            raise
        except Exception as exc:
            primary_error = exc
            raise ExternalToolError(f"Gemini 调用失败: {exc}") from exc
        finally:
            if uploaded is not None:
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception as cleanup_error:
                    if primary_error is None:
                        raise ExternalToolError(
                            f"Gemini 临时文件删除失败: {cleanup_error}"
                        ) from cleanup_error

    def split(
        self,
        *,
        path: Path,
        modality: str,
        source_id: str,
        duration_seconds: float,
        min_segment_seconds: float,
        max_segment_seconds: float,
    ) -> list[Segment]:
        """Return boundaries only; no model text is trusted or persisted here."""

        prompt = f"""
Identify only contiguous regions containing one speaker utterance in this {modality}.
Return millisecond boundaries; do not transcribe, classify, summarize, or rewrite speech.
Split when a speaker, conversational turn, scene, topic, or complete utterance changes.
Do not cut in the middle of a word or a grammatical continuation.
Prefer {min_segment_seconds:.2f}-{max_segment_seconds:.2f} seconds per region.
The media duration is {duration_seconds:.3f} seconds.
Exclude music-only, effects-only, and silence-only regions.
Return boundaries in chronological order without overlaps.
""".strip()
        result = self._structured_media_call(
            path=path,
            modality=modality,
            prompt=prompt,
            result_type=GeminiSplitResult,
            validate_result=lambda value: _validated_segments(
                value.segments,  # type: ignore[attr-defined]
                source_id=source_id,
                duration_seconds=duration_seconds,
                max_segment_seconds=max_segment_seconds,
            ),
        )
        assert isinstance(result, GeminiSplitResult)
        return _validated_segments(
            result.segments,
            source_id=source_id,
            duration_seconds=duration_seconds,
            max_segment_seconds=max_segment_seconds,
        )

    def label(
        self,
        *,
        path: Path,
        clip_id: str,
        emotions: Sequence[str],
        clusters: Sequence[str],
        language_hint: str = "auto",
    ) -> LabelRecord:
        """Transcribe and classify one already-split clip."""

        allowed = [item.strip() for item in emotions if item.strip()]
        if "unknown" not in allowed:
            allowed.append("unknown")
        allowed_clusters = list(dict.fromkeys(_safe_cluster(item) for item in clusters if item))
        if "unknown" not in allowed_clusters:
            allowed_clusters.append("unknown")
        prompt = f"""
Analyze this single speech clip. Transcribe it faithfully in language {language_hint}.
Do not invent missing words. Choose exactly one emotion from: {", ".join(allowed)}.
Choose exactly one shared acoustic/style cluster from: {", ".join(allowed_clusters)}.
Do not create a new cluster; the cluster is a grouping label, not a character identity.
Confidence is 0 to 1. Keep rationale short and based only on audible evidence.
""".strip()
        result = self._structured_media_call(
            path=path,
            modality="audio",
            prompt=prompt,
            result_type=GeminiLabelResult,
        )
        assert isinstance(result, GeminiLabelResult)
        emotion = result.emotion.strip().lower()
        if emotion not in allowed:
            emotion = "unknown"
        cluster = _safe_cluster(result.cluster)
        if cluster not in allowed_clusters:
            cluster = "unknown"
        return LabelRecord(
            clip_id=clip_id,
            transcript=result.transcript.strip(),
            emotion=emotion,
            cluster=cluster,
            confidence=result.confidence,
            rationale=result.rationale.strip(),
            model=self.model,
        )


def _validated_segments(
    boundaries: Sequence[GeminiBoundary],
    *,
    source_id: str,
    duration_seconds: float,
    max_segment_seconds: float | None = None,
) -> list[Segment]:
    duration_ms = int(round(duration_seconds * 1_000))
    result: list[Segment] = []
    previous_end = 0
    for boundary in boundaries:
        if boundary.end_ms > duration_ms + 100:
            raise ExternalToolError(
                f"Gemini 边界越过媒体时长: {boundary.end_ms} > {duration_ms} ms"
            )
        if boundary.start_ms < previous_end:
            raise ExternalToolError("Gemini 返回了重叠或乱序边界")
        end_ms = min(boundary.end_ms, duration_ms)
        if end_ms <= boundary.start_ms:
            raise ExternalToolError("Gemini 返回了空片段")
        if max_segment_seconds is not None and end_ms - boundary.start_ms > round(
            max_segment_seconds * 1_000
        ):
            raise ExternalToolError(f"Gemini 返回了超过 {max_segment_seconds:g}s 的片段")
        result.append(
            Segment(
                source_id=source_id,
                start_seconds=boundary.start_ms / 1_000,
                end_seconds=end_ms / 1_000,
                backend=SplitBackend.GEMINI,
            )
        )
        previous_end = end_ms
    if not result:
        raise ExternalToolError("Gemini 未返回任何语音片段")
    return result


_CLUSTER_RE = re.compile(r"[^a-z0-9_]+")


def _safe_cluster(value: str) -> str:
    cluster = _CLUSTER_RE.sub("_", value.strip().lower().replace("-", "_"))
    cluster = re.sub(r"_+", "_", cluster).strip("_")
    return cluster[:48] or "unknown"
