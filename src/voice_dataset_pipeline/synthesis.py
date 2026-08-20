"""End-to-end character synthesis without requiring a WebUI or API server."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .emotion import EmotionPlan, TextEmotionAnalyzer
from .references import ReferenceChoice, ReferenceSelector
from .registry import VoiceModelRecord


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    output: Path
    sample_rate: int
    emotion: EmotionPlan
    reference: ReferenceChoice


class GPTSoVITSRuntime:
    """Run upstream inference through the Python recorded with the model."""

    def __init__(self, model: VoiceModelRecord, *, device: str = "cuda", half: bool = True) -> None:
        if model.backend != "gpt-sovits":
            raise ValueError(f"unsupported synthesis backend: {model.backend}")
        if model.python is None:
            raise ValueError(
                "GPT-SoVITS Python is not registered; re-register the model with --python"
            )
        assert model.gpt_weights is not None
        assert model.sovits_weights is not None
        self.model = model
        self.python = model.python.expanduser().resolve()
        self.repository = model.repository.expanduser().resolve()
        self.gpt_weights = model.gpt_weights.expanduser().resolve()
        self.sovits_weights = model.sovits_weights.expanduser().resolve()
        self.device = device
        self.half = half
        for label, path in (
            ("GPT-SoVITS repository", self.repository),
            ("GPT-SoVITS Python", self.python),
            ("GPT weights", self.gpt_weights),
            ("SoVITS weights", self.sovits_weights),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")

    def synthesize(
        self,
        *,
        text: str,
        reference: ReferenceChoice,
        plan: EmotionPlan,
        output: str | Path,
        text_language: str = "zh",
        seed: int = -1,
    ) -> tuple[int, Path]:
        self.model.verify_synthesis_integrity()
        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"output already exists: {destination}")
        reference_path = reference.audio_path.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference audio does not exist: {reference_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=".tmp.wav",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        worker = Path(__file__).with_name("_provider_worker.py").resolve()
        if not worker.is_file():
            raise FileNotFoundError(f"provider worker does not exist: {worker}")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(self.repository), str(self.repository / "GPT_SoVITS")]
        )
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        argv = [
            str(self.python),
            str(worker),
            "gpt-sovits",
            "--repository",
            str(self.repository),
            "--gpt-weights",
            str(self.gpt_weights),
            "--sovits-weights",
            str(self.sovits_weights),
            "--version",
            self.model.version,
            "--device",
            self.device,
            "--half",
            "true" if self.half else "false",
        ]
        payload = {
            "text": text,
            "reference_audio": str(reference_path),
            "reference_text": reference.transcript,
            "text_language": text_language,
            "top_k": plan.top_k,
            "top_p": plan.top_p,
            "temperature": plan.temperature,
            "pace": plan.pace,
            "seed": seed,
            "output": str(temporary),
        }
        try:
            result = subprocess.run(
                argv,
                cwd=self.repository,
                env=environment,
                input=json.dumps(payload, ensure_ascii=False),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout)[-4000:]
                raise ValueError(f"GPT-SoVITS synthesis failed ({result.returncode}): {detail}")
            summary = _worker_summary(result.stdout)
            if not temporary.is_file() or temporary.stat().st_size <= 44:
                raise ValueError("GPT-SoVITS worker did not create a valid WAV")
            os.replace(temporary, destination)
            return int(summary["sample_rate"]), destination
        finally:
            temporary.unlink(missing_ok=True)


def _worker_summary(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and int(payload.get("sample_rate", 0)) > 0:
            return payload
    raise ValueError("GPT-SoVITS worker returned no result metadata")


class SynthesisService:
    def __init__(
        self,
        model: VoiceModelRecord,
        analyzer: TextEmotionAnalyzer,
        *,
        device: str = "cuda",
        half: bool = True,
        preferred_reference_min: float = 3.0,
        preferred_reference_max: float = 10.0,
    ) -> None:
        self.model = model
        self.analyzer = analyzer
        self.runtime = GPTSoVITSRuntime(model, device=device, half=half)
        self.preferred_reference_min = preferred_reference_min
        self.preferred_reference_max = preferred_reference_max

    def synthesize(
        self,
        text: str,
        output: str | Path,
        *,
        reference_audio: str | Path | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
        intensity: float = 0.5,
        language: str = "zh",
        seed: int = -1,
    ) -> SynthesisResult:
        from .emotion import plan_for

        plan = plan_for(emotion, intensity=intensity) if emotion else self.analyzer.analyze(text)
        if reference_audio is not None:
            path = Path(reference_audio).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"reference audio does not exist: {path}")
            transcript = (reference_text or "").strip()
            if not transcript:
                raise ValueError("--reference-text is required with an explicit reference audio")
            choice = ReferenceChoice(path, transcript, plan.emotion, float("inf"), "explicit")
        else:
            choice = ReferenceSelector(
                self.model.reference_manifest,
                preferred_min=self.preferred_reference_min,
                preferred_max=self.preferred_reference_max,
            ).select(plan)
        sample_rate, destination = self.runtime.synthesize(
            text=text,
            reference=choice,
            plan=plan,
            output=output,
            text_language=language,
            seed=seed,
        )
        return SynthesisResult(destination, sample_rate, plan, choice)
