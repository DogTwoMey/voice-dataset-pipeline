"""Stable workspace layout and crash-safe local state persistence."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .models import ReviewState

RecordT = TypeVar("RecordT", bound=BaseModel)
_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path
    normalized_audio: Path
    clips: Path
    manifests: Path
    state: Path
    training: Path
    sources_jsonl: Path
    segments_jsonl: Path
    clips_jsonl: Path
    labels_jsonl: Path
    review_json: Path

    @classmethod
    def under(cls, root: Path) -> WorkspacePaths:
        manifests = root / "manifests"
        state = root / "state"
        return cls(
            root=root,
            normalized_audio=root / "normalized",
            clips=root / "clips",
            manifests=manifests,
            state=state,
            training=root / "training",
            sources_jsonl=manifests / "sources.jsonl",
            segments_jsonl=manifests / "segments.jsonl",
            clips_jsonl=manifests / "clips.jsonl",
            labels_jsonl=manifests / "labels.jsonl",
            review_json=state / "review.json",
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    indent = 2 if pretty else None
    text = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        indent=indent,
        sort_keys=pretty,
        separators=None if pretty else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _atomic_replace(path: Path, payload: bytes) -> None:
    """Durably replace one file using a temporary sibling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Workspace:
    """A deterministic directory layout shared by all pipeline commands."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        self.paths = WorkspacePaths.under(Path(root).expanduser().resolve())
        if create:
            self._create_directories()
        elif not self.paths.root.is_dir():
            raise FileNotFoundError(f"workspace does not exist: {self.paths.root}")

    @property
    def root(self) -> Path:
        return self.paths.root

    @classmethod
    def create(cls, root: str | Path) -> Workspace:
        return cls(root, create=True)

    @classmethod
    def open(cls, root: str | Path, *, create: bool = True) -> Workspace:
        return cls(root, create=create)

    # An explicit verb reads naturally in integrations that call ``init``.
    initialize = create

    def _create_directories(self) -> None:
        for path in (
            self.paths.root,
            self.paths.normalized_audio,
            self.paths.clips,
            self.paths.manifests,
            self.paths.state,
            self.paths.training,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def read_jsonl(
        self, path: str | Path, model: type[RecordT] | None = None
    ) -> list[RecordT] | list[dict[str, Any]]:
        source = self._path(path)
        if not source.exists():
            return []
        rows: list[Any] = []
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    rows.append(model.model_validate(value) if model else value)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
        return rows

    def write_jsonl(self, path: str | Path, records: Iterable[BaseModel | dict[str, Any]]) -> Path:
        destination = self._path(path)
        payload = b"".join(
            _json_bytes(record.model_dump(mode="json") if isinstance(record, BaseModel) else record)
            for record in records
        )
        with _WRITE_LOCK:
            _atomic_replace(destination, payload)
        return destination

    def append_jsonl(self, path: str | Path, record: BaseModel | dict[str, Any]) -> Path:
        """Append logically, while replacing the file atomically on disk."""

        destination = self._path(path)
        line = _json_bytes(
            record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        )
        with _WRITE_LOCK:
            previous = destination.read_bytes() if destination.exists() else b""
            if previous and not previous.endswith(b"\n"):
                previous += b"\n"
            _atomic_replace(destination, previous + line)
        return destination

    def upsert_jsonl(
        self,
        path: str | Path,
        record: BaseModel | dict[str, Any],
        *,
        key: str,
    ) -> Path:
        value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        if key not in value:
            raise KeyError(f"record has no {key!r} field")
        with _WRITE_LOCK:
            rows = self.read_jsonl(path)
            assert isinstance(rows, list)
            replaced = False
            result: list[dict[str, Any]] = []
            for row in rows:
                if row.get(key) == value[key]:
                    if not replaced:
                        result.append(value)
                        replaced = True
                else:
                    result.append(row)
            if not replaced:
                result.append(value)
            return self.write_jsonl(path, result)

    def load_review(self) -> ReviewState:
        if not self.paths.review_json.exists():
            return ReviewState()
        try:
            raw = json.loads(self.paths.review_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid review state {self.paths.review_json}: {exc}") from exc
        return ReviewState.model_validate(raw)

    def save_review(self, state: ReviewState) -> Path:
        state.updated_at = datetime.now(UTC)
        payload = _json_bytes(state.model_dump(mode="json"), pretty=True)
        with _WRITE_LOCK:
            _atomic_replace(self.paths.review_json, payload)
        return self.paths.review_json
