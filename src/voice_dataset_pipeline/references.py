"""Auditable reference-audio selection from a reviewed training export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .emotion import EmotionPlan
from .media import sha256_file


@dataclass(frozen=True, slots=True)
class ReferenceChoice:
    audio_path: Path
    transcript: str
    emotion: str
    score: float
    reason: str
    sha256: str = ""


class ReferenceSelector:
    def __init__(
        self,
        manifest: str | Path,
        *,
        preferred_min: float = 3.0,
        preferred_max: float = 10.0,
        preferred_clip_ids: dict[str, str] | None = None,
        preferred_clusters: tuple[str, ...] = (),
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.preferred_min = preferred_min
        self.preferred_max = preferred_max
        self.preferred_clip_ids = {
            emotion.strip().lower(): clip_id.strip()
            for emotion, clip_id in (preferred_clip_ids or {}).items()
            if emotion.strip() and clip_id.strip()
        }
        self.preferred_clusters = tuple(
            cluster.strip().lower() for cluster in preferred_clusters if cluster.strip()
        )

    def select(self, plan: EmotionPlan) -> ReferenceChoice:
        if not self.manifest.is_file():
            raise FileNotFoundError(f"reference manifest does not exist: {self.manifest}")
        rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        configured_clip_id = self.preferred_clip_ids.get(plan.emotion.strip().lower(), "")
        candidates: list[ReferenceChoice] = []
        for row in rows:
            clip_id = str(row.get("clip_id", "")).strip()
            if configured_clip_id and clip_id != configured_clip_id:
                continue
            path = Path(row.get("wav_path") or row.get("audio_path") or "").expanduser()
            text = str(row.get("text") or row.get("transcript") or "").strip()
            if not path.is_file() or not text or row.get("include", True) is False:
                continue
            expected_sha256 = str(row.get("sha256") or "").strip().lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha256
            ):
                raise ValueError(
                    f"reference manifest has no valid SHA-256 for usable audio: {path}"
                )
            duration = float(row.get("duration_seconds", 0.0))
            emotion = str(row.get("emotion", "neutral"))
            score = 0.0
            reasons: list[str] = []
            if configured_clip_id:
                reasons.append("configured-clip")
            if emotion == plan.emotion:
                score += 100
                reasons.append("same-emotion")
            if self.preferred_min <= duration <= self.preferred_max:
                score += 30
                reasons.append("preferred-duration")
            elif 1.5 <= duration <= 15:
                score += 10
            if bool(row.get("reviewed", False)):
                score += 20
                reasons.append("human-reviewed")
            cluster = str(row.get("cluster", "")).strip().lower()
            if cluster not in {"", "unknown"}:
                score += 5
            if cluster in self.preferred_clusters:
                rank = self.preferred_clusters.index(cluster)
                score += max(10, 40 - rank * 10)
                reasons.append(f"scene-cluster:{cluster}")
            score -= abs(duration - 6.0)
            candidates.append(
                ReferenceChoice(
                    path.resolve(),
                    text,
                    emotion,
                    score,
                    ",".join(reasons),
                    expected_sha256,
                )
            )
        if not candidates:
            if configured_clip_id:
                raise ValueError(
                    f"configured reference clip '{configured_clip_id}' for emotion "
                    f"'{plan.emotion}' is not usable in {self.manifest}"
                )
            raise ValueError(f"reference manifest contains no usable audio: {self.manifest}")
        selected = max(candidates, key=lambda item: (item.score, item.audio_path.name))
        actual_sha256 = sha256_file(selected.audio_path)
        if actual_sha256 != selected.sha256:
            raise ValueError(
                f"reference audio SHA-256 mismatch: expected {selected.sha256}, "
                f"got {actual_sha256}; restore the reviewed export or re-export it"
            )
        return selected
