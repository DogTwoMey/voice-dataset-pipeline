"""Strict TOML configuration with a complete, usable default profile."""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .models import InputMode, SplitBackend, StrictModel

DEFAULT_CONFIG = r"""
[media]
ffmpeg_binary = "ffmpeg"
sample_rate = 48000
audio_extensions = [
  ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff",
]
video_extensions = [
  ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".ts", ".mts", ".m2ts",
]

[splitting]
backend = "energy"
input_mode = "auto"
frame_ms = 30
hop_ms = 10
adaptive_percentile = 20.0
threshold_offset_db = 10.0
threshold_floor_db = -55.0
threshold_ceiling_db = -25.0
min_speech_ms = 180
min_silence_ms = 180
min_segment_seconds = 0.8
merge_gap_ms = 300
padding_ms = 120
max_segment_seconds = 15.0
boundary_search_ms = 750
stream_block_frames = 65536

[gemini]
api_key_env = "GEMINI_API_KEY"
model = "gemini-3.6-flash"
timeout_seconds = 120.0
max_retries = 3

[review]
emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted", "unknown"]
clusters = [
  "calm_soft", "conversational", "bright_playful", "sad_soft",
  "tense_loud", "whisper", "unknown",
]
play_command = ""

[training]
enabled = false

[training.gpt_sovits]
enabled = false
experiment_name = ""
model_version = "v2ProPlus"
speaker = ""
language = "zh"
require_reviewed = true
full_precision = true
gpu = 0
sovits_batch_size = 6
sovits_epochs = 12
text_low_lr_rate = 0.4
sovits_save_every = 4
grad_checkpoint = false
lora_rank = 32
gpt_batch_size = 6
gpt_epochs = 20
gpt_save_every = 5

[training.rvc]
enabled = false
experiment_name = ""
version = "v2"
sample_rate = "48k"
speaker = ""
language = "zh"
require_reviewed = true
gpu = 0
workers = 4
batch_size = 6
epochs = 200
save_every = 25
preprocess_seconds = 3.7
half = true

""".lstrip()


class MediaConfig(StrictModel):
    ffmpeg_binary: str = "ffmpeg"
    sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    audio_extensions: list[str]
    video_extensions: list[str]


class SplittingConfig(StrictModel):
    backend: SplitBackend = SplitBackend.ENERGY
    input_mode: InputMode = InputMode.AUTO
    frame_ms: int = Field(default=30, gt=0, le=1_000)
    hop_ms: int = Field(default=10, gt=0, le=1_000)
    adaptive_percentile: float = Field(default=20, ge=0, le=100)
    threshold_offset_db: float = Field(default=10, ge=0, le=60)
    threshold_floor_db: float = Field(default=-55, le=0)
    threshold_ceiling_db: float = Field(default=-25, le=0)
    min_speech_ms: int = Field(default=180, ge=0)
    min_silence_ms: int = Field(default=180, ge=0)
    min_segment_seconds: float = Field(default=0.8, gt=0)
    merge_gap_ms: int = Field(default=300, ge=0)
    padding_ms: int = Field(default=120, ge=0)
    max_segment_seconds: float = Field(default=15, gt=0)
    boundary_search_ms: int = Field(default=750, ge=0)
    stream_block_frames: int = Field(default=65_536, gt=0)

    @model_validator(mode="after")
    def validate_timing(self) -> SplittingConfig:
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must not exceed frame_ms")
        if self.max_segment_seconds * 1_000 <= 2 * self.padding_ms:
            raise ValueError("max_segment_seconds must be longer than twice padding_ms")
        return self


class GeminiConfig(StrictModel):
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-3.6-flash"
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ReviewConfig(StrictModel):
    emotions: list[str]
    clusters: list[str]
    play_command: str = ""


class TrainerConfig(StrictModel):
    enabled: bool = False
    repository: Path | None = None
    python: Path | None = None
    experiment_name: str = ""
    speaker: str = ""
    language: str = "zh"


class GPTSoVITSConfig(TrainerConfig):
    model_version: str = "v2ProPlus"
    require_reviewed: bool = True
    full_precision: bool = True
    gpu: int = Field(default=0, ge=0)
    sovits_batch_size: int = Field(default=6, gt=0)
    sovits_epochs: int = Field(default=12, gt=0)
    text_low_lr_rate: float = Field(default=0.4, gt=0)
    sovits_save_every: int = Field(default=4, gt=0)
    grad_checkpoint: bool = False
    lora_rank: int = Field(default=32, gt=0)
    gpt_batch_size: int = Field(default=6, gt=0)
    gpt_epochs: int = Field(default=20, gt=0)
    gpt_save_every: int = Field(default=5, gt=0)


class RVCTrainingConfig(TrainerConfig):
    version: str = "v2"
    sample_rate: Literal["32k", "40k", "48k"] = "48k"
    require_reviewed: bool = True
    gpu: int = Field(default=0, ge=0)
    workers: int = Field(default=4, gt=0)
    batch_size: int = Field(default=6, gt=0)
    epochs: int = Field(default=200, gt=0)
    save_every: int = Field(default=25, gt=0)
    preprocess_seconds: float = Field(default=3.7, gt=0)
    half: bool = True


class TrainingConfig(StrictModel):
    enabled: bool = False
    gpt_sovits: GPTSoVITSConfig = Field(default_factory=GPTSoVITSConfig)
    rvc: RVCTrainingConfig = Field(default_factory=RVCTrainingConfig)


class PipelineConfig(StrictModel):
    media: MediaConfig
    splitting: SplittingConfig
    gemini: GeminiConfig
    review: ReviewConfig
    training: TrainingConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load defaults, optionally overlay a TOML file, then validate strictly."""

    defaults = tomllib.loads(DEFAULT_CONFIG)
    if path is None:
        raw = defaults
    else:
        config_path = Path(path)
        with config_path.open("rb") as stream:
            raw = _deep_merge(defaults, tomllib.load(stream))
    return PipelineConfig.model_validate(raw)


def write_default_config(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create a documented starter config without clobbering user changes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode: Literal["w", "x"] = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(DEFAULT_CONFIG)
    return destination
