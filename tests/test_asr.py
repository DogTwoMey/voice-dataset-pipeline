from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import replace

import pytest

from voice_dataset_pipeline.asr import (
    ASRProfile,
    SenseVoiceTranscriber,
    boundary_warnings,
    parse_sensevoice_text,
    transcribe_workspace,
    transcript_similarity,
)
from voice_dataset_pipeline.models import ASRRecord, ClipRecord
from voice_dataset_pipeline.workspace import Workspace


class _FakeTranscriber:
    def __init__(self) -> None:
        self.model_name = "sensevoice"
        self.vad_model_name = "vad-1"
        self.model_revision = "model-rev-1"
        self.vad_revision = "vad-rev-1"
        self.funasr_version = "funasr-test"
        self.modelscope_version = "modelscope-test"
        self.language = "auto"
        self.replacements: dict[str, str] = {}
        self.calls = 0

    def transcribe(self, _path) -> tuple[str, str, str, str]:
        self.calls += 1
        return "expected text", "expected text", "en", "neutral"


def test_parse_sensevoice_tags() -> None:
    text, language, emotion = parse_sensevoice_text(
        "<|zh|><|HAPPY|><|Speech|><|woitn|>你好，旅行者。"
    )
    assert text == "你好，旅行者。"
    assert language == "zh"
    assert emotion == "happy"


def test_similarity_normalizes_punctuation_and_width() -> None:
    assert transcript_similarity("你好，Ａ！", "你好A") == 1.0
    assert transcript_similarity("", "你好") is None


def test_boundary_warning_catches_classifier_continuation(tmp_path) -> None:
    clips = [
        ClipRecord(
            clip_id="a" * 64,
            source_id="source",
            audio_path=tmp_path / "a.wav",
            start_ms=0,
            end_ms=1000,
            sample_rate=32000,
            frames=32000,
            sha256="1" * 64,
        ),
        ClipRecord(
            clip_id="b" * 64,
            source_id="source",
            audio_path=tmp_path / "b.wav",
            start_ms=1000,
            end_ms=2000,
            sample_rate=32000,
            frames=32000,
            sha256="2" * 64,
        ),
    ]
    rows = [
        ASRRecord(clip_id=clips[0].clip_id, audio_sha256="1" * 64, transcript="的确是有。"),
        ASRRecord(clip_id=clips[1].clip_id, audio_sha256="2" * 64, transcript="一位，不过。"),
    ]
    warnings = boundary_warnings(rows, clips)
    assert warnings[clips[0].clip_id] == ["possible_boundary_continuation_to_next"]
    assert warnings[clips[1].clip_id] == ["possible_boundary_continuation_from_previous"]


def test_asr_cache_profile_covers_every_decision_input(tmp_path) -> None:
    workspace = Workspace.create(tmp_path / "workspace")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"immutable audio")
    clip = ClipRecord(
        clip_id="a" * 64,
        source_id="source",
        audio_path=audio,
        start_ms=0,
        end_ms=1000,
        sample_rate=16_000,
        frames=16_000,
        sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        text="expected text",
    )
    workspace.write_jsonl(workspace.paths.clips_jsonl, [clip])
    transcriber = _FakeTranscriber()

    first = transcribe_workspace(workspace, transcriber, seed_labels=False)
    same = transcribe_workspace(workspace, transcriber, seed_labels=False)
    assert (first.transcribed, same.reused, transcriber.calls) == (1, 1, 1)

    transcriber.vad_model_name = "vad-2"
    transcribe_workspace(workspace, transcriber, seed_labels=False)
    transcriber.language = "en"
    transcribe_workspace(workspace, transcriber, seed_labels=False)
    transcriber.replacements = {"old": "new"}
    transcribe_workspace(workspace, transcriber, seed_labels=False)
    transcribe_workspace(
        workspace,
        transcriber,
        minimum_similarity=0.8,
        seed_labels=False,
    )
    transcribe_workspace(
        workspace,
        transcriber,
        minimum_similarity=0.8,
        require_expected_match=True,
        seed_labels=False,
    )
    changed_clip = clip.model_copy(update={"text": "changed expected text"})
    workspace.write_jsonl(workspace.paths.clips_jsonl, [changed_clip])
    transcribe_workspace(
        workspace,
        transcriber,
        minimum_similarity=0.8,
        require_expected_match=True,
        seed_labels=False,
    )

    assert transcriber.calls == 7
    records = workspace.read_jsonl(workspace.paths.asr_jsonl, ASRRecord)
    assert len(records) == 1
    expected_profile = ASRProfile(
        model=transcriber.model_name,
        vad_model=transcriber.vad_model_name,
        language=transcriber.language,
        replacements=transcriber.replacements,
        minimum_similarity=0.8,
        require_expected_match=True,
        model_revision=transcriber.model_revision,
        vad_revision=transcriber.vad_revision,
        funasr_version=transcriber.funasr_version,
        modelscope_version=transcriber.modelscope_version,
    )
    assert records[0].profile_sha256 == expected_profile.fingerprint(changed_clip.text)


def test_asr_profile_covers_provider_revision_and_library_versions() -> None:
    base = ASRProfile(
        model="sensevoice",
        vad_model="vad",
        language="auto",
        replacements={},
        minimum_similarity=0.72,
        require_expected_match=True,
        model_revision="model-rev-1",
        vad_revision="vad-rev-1",
        funasr_version="1.3.23",
        modelscope_version="1.38.1",
    )
    baseline = base.fingerprint("台词")
    for field in ("model_revision", "vad_revision", "funasr_version", "modelscope_version"):
        changed = replace(base, **{field: "changed"})
        assert changed.fingerprint("台词") != baseline


def test_sensevoice_passes_pinned_model_revisions(monkeypatch) -> None:
    captured = {}
    fake_module = types.ModuleType("funasr")

    def fake_auto_model(**kwargs):
        captured.update(kwargs)
        return object()

    fake_module.AutoModel = fake_auto_model
    monkeypatch.setitem(sys.modules, "funasr", fake_module)
    monkeypatch.setattr(
        "voice_dataset_pipeline.asr._package_version",
        lambda name: {"funasr": "1.3.23", "modelscope": "1.38.1"}[name],
    )

    SenseVoiceTranscriber(
        model="sensevoice",
        vad_model="vad",
        model_revision="model-rev",
        vad_revision="vad-rev",
        expected_funasr_version="1.3.23",
        expected_modelscope_version="1.38.1",
    )

    assert captured["model_revision"] == "model-rev"
    assert captured["vad_model_revision"] == "vad-rev"


def test_sensevoice_rejects_library_version_drift(monkeypatch) -> None:
    fake_module = types.ModuleType("funasr")
    fake_module.AutoModel = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "funasr", fake_module)
    monkeypatch.setattr(
        "voice_dataset_pipeline.asr._package_version",
        lambda name: {"funasr": "changed", "modelscope": "1.38.1"}[name],
    )

    with pytest.raises(ValueError, match="funasr version mismatch"):
        SenseVoiceTranscriber(
            model="sensevoice",
            vad_model="vad",
            expected_funasr_version="1.3.23",
            expected_modelscope_version="1.38.1",
        )
