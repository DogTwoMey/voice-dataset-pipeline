"""Lazy SenseVoice adapter for local transcript and speech-emotion validation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from importlib import metadata
from pathlib import Path
from typing import Any

from .media import sha256_file
from .models import ASRRecord, ClipRecord, LabelRecord
from .workspace import Workspace

_TAG = re.compile(r"<\|([^|>]+)\|>")
_KNOWN_EMOTIONS = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "FEARFUL": "fearful",
    "DISGUSTED": "disgusted",
    "SURPRISED": "surprised",
}
_CONTINUATION_STARTS = (
    "一位",
    "但是",
    "不过",
    "而且",
    "所以",
    "然后",
    "的话",
    "呢",
    "吗",
    "吧",
)
_INCOMPLETE_ENDS = (
    "的确是有",
    "毕竟",
    "因为",
    "所以",
    "但是",
    "不过",
    "而且",
    "我们",
    "一个",
    "这个",
    "的",
    "和",
    "与",
)


def normalize_transcript(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in value if character.isalnum())


def transcript_similarity(expected: str, actual: str) -> float | None:
    left = normalize_transcript(expected)
    right = normalize_transcript(actual)
    if not left or not right:
        return None
    return round(SequenceMatcher(None, left, right, autojunk=False).ratio(), 6)


def parse_sensevoice_text(raw: str) -> tuple[str, str, str]:
    tags = _TAG.findall(raw)
    language = next(
        (tag.casefold() for tag in tags if tag.casefold() in {"zh", "en", "ja", "yue", "ko"}),
        "auto",
    )
    emotion = next(
        (_KNOWN_EMOTIONS[tag.upper()] for tag in tags if tag.upper() in _KNOWN_EMOTIONS), "unknown"
    )
    transcript = _TAG.sub("", raw).strip()
    return transcript, language, emotion


def boundary_warnings(records: list[ASRRecord], clips: list[ClipRecord]) -> dict[str, list[str]]:
    """Flag likely cross-clip grammatical continuations for focused TUI review."""

    by_id = {row.clip_id: row for row in records}
    warnings: dict[str, list[str]] = {}
    for left_clip, right_clip in zip(clips, clips[1:], strict=False):
        if left_clip.source_id != right_clip.source_id:
            continue
        left = by_id.get(left_clip.clip_id)
        right = by_id.get(right_clip.clip_id)
        if left is None or right is None or not left.transcript or not right.transcript:
            continue
        left_text = left.transcript.rstrip("。！？!?…，,；;：: ")
        right_text = right.transcript.lstrip("。！？!?…，,；;：: ")
        continuation = any(right_text.startswith(value) for value in _CONTINUATION_STARTS)
        incomplete = any(left_text.endswith(value) for value in _INCOMPLETE_ENDS)
        no_terminal = left.transcript.rstrip().endswith(("，", ",", "；", ";", "：", ":"))
        if not (continuation and (incomplete or no_terminal)):
            continue
        warnings.setdefault(left.clip_id, []).append("possible_boundary_continuation_to_next")
        warnings.setdefault(right.clip_id, []).append(
            "possible_boundary_continuation_from_previous"
        )
    return warnings


@dataclass(frozen=True, slots=True)
class ASRSummary:
    total: int
    transcribed: int
    reused: int
    accepted: int


@dataclass(frozen=True, slots=True)
class ASRProfile:
    """Configuration that can change a persisted ASR decision."""

    model: str
    vad_model: str
    language: str
    replacements: dict[str, str]
    minimum_similarity: float
    require_expected_match: bool
    model_revision: str = ""
    vad_revision: str = ""
    funasr_version: str = ""
    modelscope_version: str = ""

    def fingerprint(self, expected_text: str) -> str:
        payload = {
            "model": self.model,
            "vad_model": self.vad_model,
            "language": self.language,
            "replacements": self.replacements,
            "minimum_similarity": self.minimum_similarity,
            "require_expected_match": self.require_expected_match,
            "expected_text": expected_text,
            "model_revision": self.model_revision,
            "vad_revision": self.vad_revision,
            "funasr_version": self.funasr_version,
            "modelscope_version": self.modelscope_version,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


class SenseVoiceTranscriber:
    """Load FunASR only when the user explicitly runs local ASR."""

    def __init__(
        self,
        *,
        model: str,
        vad_model: str,
        device: str = "cuda:0",
        model_revision: str = "master",
        vad_revision: str = "master",
        expected_funasr_version: str = "",
        expected_modelscope_version: str = "",
        language: str = "auto",
        replacements: dict[str, str] | None = None,
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ValueError("SenseVoice requires the 'asr' optional dependencies") from exc
        self.model_name = model
        self.vad_model_name = vad_model
        self.model_revision = model_revision
        self.vad_revision = vad_revision
        self.funasr_version = _package_version("funasr")
        self.modelscope_version = _package_version("modelscope")
        for package, expected, actual in (
            ("funasr", expected_funasr_version, self.funasr_version),
            ("modelscope", expected_modelscope_version, self.modelscope_version),
        ):
            if expected and expected != actual:
                raise ValueError(
                    f"{package} version mismatch: configured {expected}, installed {actual}; "
                    "use the pinned ASR environment or update the project config intentionally"
                )
        self.language = language
        self.replacements = dict(replacements or {})
        self._model = AutoModel(
            model=model,
            model_revision=model_revision,
            vad_model=vad_model,
            vad_model_revision=vad_revision,
            vad_kwargs={"max_single_segment_time": 30_000},
            device=device,
            trust_remote_code=True,
            disable_update=True,
        )

    def transcribe(self, path: str | Path) -> tuple[str, str, str, str]:
        rows: list[dict[str, Any]] = self._model.generate(
            input=str(Path(path).resolve()),
            cache={},
            language=self.language,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        if not rows:
            return "", "", "auto", "unknown"
        raw = str(rows[0].get("text", ""))
        transcript, language, emotion = parse_sensevoice_text(raw)
        for source, target in self.replacements.items():
            transcript = transcript.replace(source, target)
        return transcript, raw, language, emotion


def transcribe_workspace(
    workspace: Workspace,
    transcriber: SenseVoiceTranscriber,
    *,
    minimum_similarity: float = 0.72,
    require_expected_match: bool = False,
    force: bool = False,
    seed_labels: bool = True,
) -> ASRSummary:
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    previous = workspace.read_jsonl(workspace.paths.asr_jsonl, ASRRecord)
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert isinstance(clips, list)
    assert isinstance(previous, list)
    assert isinstance(labels, list)
    existing = {row.clip_id: row for row in previous}
    label_map = {row.clip_id: row for row in labels}
    final: list[ASRRecord] = []
    transcribed = 0
    reused = 0
    profile = ASRProfile(
        model=transcriber.model_name,
        vad_model=transcriber.vad_model_name,
        language=transcriber.language,
        replacements=transcriber.replacements,
        minimum_similarity=minimum_similarity,
        require_expected_match=require_expected_match,
        model_revision=getattr(transcriber, "model_revision", ""),
        vad_revision=getattr(transcriber, "vad_revision", ""),
        funasr_version=getattr(transcriber, "funasr_version", ""),
        modelscope_version=getattr(transcriber, "modelscope_version", ""),
    )
    for clip in clips:
        digest = sha256_file(clip.audio_path)
        profile_sha256 = profile.fingerprint(clip.text)
        prior = existing.get(clip.clip_id)
        if (
            not force
            and prior is not None
            and prior.audio_sha256 == digest
            and prior.profile_sha256 == profile_sha256
        ):
            record = prior
            reused += 1
        else:
            transcript, raw, language, emotion = transcriber.transcribe(clip.audio_path)
            similarity = transcript_similarity(clip.text, transcript)
            reasons: list[str] = []
            if not transcript:
                reasons.append("empty_transcript")
            if require_expected_match and similarity is None:
                reasons.append("missing_expected_text")
            if similarity is not None and similarity < minimum_similarity:
                reasons.append("transcript_similarity_too_low")
            record = ASRRecord(
                clip_id=clip.clip_id,
                audio_sha256=digest,
                profile_sha256=profile_sha256,
                transcript=transcript,
                raw_text=raw,
                language=language,
                emotion=emotion,
                model=transcriber.model_name,
                expected_text=clip.text,
                transcript_similarity=similarity,
                accepted=not reasons,
                reasons=reasons,
            )
            workspace.upsert_jsonl(workspace.paths.asr_jsonl, record, key="clip_id")
            transcribed += 1
        final.append(record)
        if seed_labels and record.transcript and (force or clip.clip_id not in label_map):
            label_map[clip.clip_id] = LabelRecord(
                clip_id=clip.clip_id,
                transcript=record.transcript,
                emotion=record.emotion,
                cluster="unknown",
                confidence=None,
                rationale="local SenseVoice seed; requires review",
                model=record.model,
            )
            workspace.upsert_jsonl(
                workspace.paths.labels_jsonl, label_map[clip.clip_id], key="clip_id"
            )
    warnings = boundary_warnings(final, clips)
    final = [
        row.model_copy(
            update={
                "reasons": list(
                    dict.fromkeys(
                        [
                            reason
                            for reason in row.reasons
                            if not reason.startswith("possible_boundary_continuation_")
                        ]
                        + warnings.get(row.clip_id, [])
                    )
                )
            }
        )
        for row in final
    ]
    for row in final:
        if row.clip_id not in label_map or not warnings.get(row.clip_id):
            continue
        label = label_map[row.clip_id]
        rationale = label.rationale.split(" | boundary warning:", 1)[0]
        label_map[row.clip_id] = label.model_copy(
            update={
                "rationale": f"{rationale} | boundary warning: " + ", ".join(warnings[row.clip_id])
            }
        )
    workspace.write_jsonl(workspace.paths.labels_jsonl, label_map.values())
    workspace.write_jsonl(workspace.paths.asr_jsonl, final)
    return ASRSummary(
        total=len(clips),
        transcribed=transcribed,
        reused=reused,
        accepted=sum(row.accepted for row in final),
    )
