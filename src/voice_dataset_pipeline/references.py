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
        self, manifest: str | Path, *, preferred_min: float = 3.0, preferred_max: float = 10.0
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.preferred_min = preferred_min
        self.preferred_max = preferred_max

    def select(self, plan: EmotionPlan) -> ReferenceChoice:
        if not self.manifest.is_file():
            raise FileNotFoundError(f"reference manifest does not exist: {self.manifest}")
        rows = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates: list[ReferenceChoice] = []
        for row in rows:
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
            if str(row.get("cluster", "")) not in {"", "unknown"}:
                score += 5
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
            raise ValueError(f"reference manifest contains no usable audio: {self.manifest}")
        selected = max(candidates, key=lambda item: (item.score, item.audio_path.name))
        actual_sha256 = sha256_file(selected.audio_path)
        if actual_sha256 != selected.sha256:
            raise ValueError(
                f"reference audio SHA-256 mismatch: expected {selected.sha256}, "
                f"got {actual_sha256}; restore the reviewed export or re-export it"
            )
        return selected
