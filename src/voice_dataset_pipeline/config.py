"""Strict TOML configuration with a complete, usable default profile."""

from __future__ import annotations

import copy
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator

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

[gemini.chunking]
# Long videos are locally divided around silence before remote analysis.
enabled = true
threshold_seconds = 180.0
target_seconds = 90.0
max_seconds = 120.0
boundary_search_seconds = 15.0
min_silence_seconds = 0.30
silence_noise_db = -40.0
video_height = 360
video_bitrate_kbps = 500
audio_bitrate_kbps = 64
reuse_chunks = true
keep_chunks = true

[asr]
enabled = false
provider = "sensevoice"
model = "iic/SenseVoiceSmall"
vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
model_revision = "master"
vad_revision = "master"
funasr_version = "1.4.2"
modelscope_version = "1.39.1"
device = "cuda:0"
language = "auto"
replacements = {}

[quality]
enabled = true
min_duration_seconds = 1.0
max_duration_seconds = 12.0
min_rms_dbfs = -45.0
max_rms_dbfs = -8.0
max_clipping_ratio = 0.001
max_silence_ratio = 0.45
min_transcript_similarity = 0.72
require_asr = false

[emotion]
provider = "rules"
base_url = "https://example.invalid/v1"
model = "replace-with-model-id"
api_key_env = "OPENAI_COMPAT_API_KEY"
timeout_seconds = 60.0

[reference]
preferred_min_seconds = 3.0
preferred_max_seconds = 10.0

[inference]
default_model = ""
device = "cuda"
half = true
language = "zh"
seed = -1

[registry]
path = "models/registry.json"

[postprocess]
enabled = false
backend = "rvc"
f0_method = "rmvpe"
transpose = 0
index_rate = 0.45
rms_mix_rate = 0.25
protect = 0.33

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
sovits_epochs = 8
text_low_lr_rate = 0.4
sovits_save_every = 4
grad_checkpoint = false
lora_rank = 32
gpt_batch_size = 6
gpt_epochs = 12
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

DEFAULT_SECRETS = r"""
# Sensitive local values only. Never commit this file.
# Keys are environment-variable names referenced by the project configuration.
[environment]
GEMINI_API_KEY = ""
OPENAI_COMPAT_API_KEY = ""
""".lstrip()

SECRETS_GITIGNORE = r"""
# Keep every local credential in this directory out of Git.
*
!.gitignore
""".lstrip()

PROJECT_CONFIG_RELATIVE = Path("config/pipeline.toml")
SECRETS_CONFIG_RELATIVE = Path("secrets/credentials.toml")
LEGACY_PROJECT_CONFIG_RELATIVE = Path("pipeline.toml")


@dataclass(frozen=True, slots=True)
class ConfigLayout:
    project: Path
    secrets: Path
    secrets_gitignore: Path


def config_layout(root: str | Path) -> ConfigLayout:
    base = Path(root).expanduser().resolve()
    return ConfigLayout(
        project=base / PROJECT_CONFIG_RELATIVE,
        secrets=base / SECRETS_CONFIG_RELATIVE,
        secrets_gitignore=base / SECRETS_CONFIG_RELATIVE.parent / ".gitignore",
    )


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


class GeminiChunkingConfig(StrictModel):
    """Local, content-addressed preparation for long Gemini video calls."""

    enabled: bool = True
    threshold_seconds: float = Field(default=180, gt=0)
    target_seconds: float = Field(default=90, gt=0)
    max_seconds: float = Field(default=120, gt=0)
    boundary_search_seconds: float = Field(default=15, ge=0)
    min_silence_seconds: float = Field(default=0.3, gt=0)
    silence_noise_db: float = Field(default=-40, ge=-100, le=0)
    video_height: int = Field(default=360, ge=144, le=2160)
    video_bitrate_kbps: int = Field(default=500, ge=64, le=20_000)
    audio_bitrate_kbps: int = Field(default=64, ge=16, le=512)
    reuse_chunks: bool = True
    keep_chunks: bool = True

    @model_validator(mode="after")
    def validate_durations(self) -> GeminiChunkingConfig:
        if self.target_seconds > self.max_seconds:
            raise ValueError("target_seconds must not exceed max_seconds")
        return self


class GeminiConfig(StrictModel):
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-3.6-flash"
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)
    chunking: GeminiChunkingConfig = Field(default_factory=GeminiChunkingConfig)


class ASRConfig(StrictModel):
    enabled: bool = False
    provider: Literal["sensevoice"] = "sensevoice"
    model: str = "iic/SenseVoiceSmall"
    vad_model: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    model_revision: str = "master"
    vad_revision: str = "master"
    funasr_version: str = "1.4.2"
    modelscope_version: str = "1.39.1"
    device: str = "cuda:0"
    language: str = "auto"
    replacements: dict[str, str] = Field(default_factory=dict)


class QualityConfig(StrictModel):
    enabled: bool = True
    min_duration_seconds: float = Field(default=1.0, gt=0)
    max_duration_seconds: float = Field(default=12.0, gt=0)
    min_rms_dbfs: float = Field(default=-45.0, le=0)
    max_rms_dbfs: float = Field(default=-8.0, le=0)
    max_clipping_ratio: float = Field(default=0.001, ge=0, le=1)
    max_silence_ratio: float = Field(default=0.45, ge=0, le=1)
    min_transcript_similarity: float = Field(default=0.72, ge=0, le=1)
    require_asr: bool = False

    @model_validator(mode="after")
    def validate_ranges(self) -> QualityConfig:
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must exceed min_duration_seconds")
        if self.max_rms_dbfs <= self.min_rms_dbfs:
            raise ValueError("max_rms_dbfs must exceed min_rms_dbfs")
        return self


class EmotionConfig(StrictModel):
    provider: Literal["rules", "openai-compatible"] = "rules"
    base_url: str = "https://example.invalid/v1"
    model: str = "replace-with-model-id"
    api_key_env: str = "OPENAI_COMPAT_API_KEY"
    timeout_seconds: float = Field(default=60, gt=0)


class ReferenceConfig(StrictModel):
    preferred_min_seconds: float = Field(default=3.0, gt=0)
    preferred_max_seconds: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> ReferenceConfig:
        if self.preferred_max_seconds <= self.preferred_min_seconds:
            raise ValueError("preferred_max_seconds must exceed preferred_min_seconds")
        return self


class InferenceConfig(StrictModel):
    default_model: str = ""
    device: str = "cuda"
    half: bool = True
    language: str = "zh"
    seed: int = -1


class RegistryConfig(StrictModel):
    path: Path = Path("models/registry.json")


class PostprocessConfig(StrictModel):
    enabled: bool = False
    backend: Literal["rvc"] = "rvc"
    f0_method: str = "rmvpe"
    transpose: int = Field(default=0, ge=-24, le=24)
    index_rate: float = Field(default=0.45, ge=0, le=1)
    rms_mix_rate: float = Field(default=0.25, ge=0, le=1)
    protect: float = Field(default=0.33, ge=0, le=0.5)


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
    asr: ASRConfig
    quality: QualityConfig
    emotion: EmotionConfig
    reference: ReferenceConfig
    inference: InferenceConfig
    registry: RegistryConfig
    postprocess: PostprocessConfig
    review: ReviewConfig
    training: TrainingConfig


class SecretsConfig(StrictModel):
    """Local credentials keyed by their configured environment-variable name."""

    environment: dict[str, SecretStr] = Field(default_factory=dict)

    @field_validator("environment")
    @classmethod
    def validate_environment_names(
        cls,
        value: dict[str, SecretStr],
    ) -> dict[str, SecretStr]:
        invalid = sorted(name for name in value if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name))
        if invalid:
            raise ValueError(f"invalid environment variable names: {invalid}")
        return value

    def get(self, name: str) -> str | None:
        value = self.environment.get(name)
        if value is None:
            return None
        revealed = value.get_secret_value().strip()
        return revealed or None


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


def load_secrets(path: str | Path | None = None) -> SecretsConfig:
    """Load a distinct local credential file; a missing default means no secrets."""

    if path is None:
        return SecretsConfig()
    secrets_path = Path(path).expanduser().resolve()
    with secrets_path.open("rb") as stream:
        return SecretsConfig.model_validate(tomllib.load(stream))


def write_default_config(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create a documented starter config without clobbering user changes."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode: Literal["w", "x"] = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(DEFAULT_CONFIG)
    return destination


def write_default_secrets(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an empty local credential file without exposing any real token."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode: Literal["w", "x"] = "w" if overwrite else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(DEFAULT_SECRETS)
    return destination


def write_secrets_gitignore(path: str | Path) -> Path:
    """Protect the complete secrets directory, preserving only its ignore rule."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        rules = [
            line.strip()
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if rules != ["*", "!.gitignore"]:
            raise ValueError(
                f"secrets .gitignore must contain only '*' followed by '!.gitignore': {destination}"
            )
        return destination
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(SECRETS_GITIGNORE)
    return destination


def generate_default_config_layout(
    root: str | Path,
    *,
    overwrite_project: bool = False,
    overwrite_secrets: bool = False,
) -> ConfigLayout:
    """Generate separate project and credential files under one workspace root."""

    layout = config_layout(root)
    write_secrets_gitignore(layout.secrets_gitignore)
    if overwrite_project or not layout.project.exists():
        write_default_config(layout.project, overwrite=overwrite_project)
    if overwrite_secrets or not layout.secrets.exists():
        write_default_secrets(layout.secrets, overwrite=overwrite_secrets)
    return layout
