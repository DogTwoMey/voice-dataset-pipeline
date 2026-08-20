"""Command-line entry point for the headless pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    LEGACY_PROJECT_CONFIG_RELATIVE,
    PipelineConfig,
    SecretsConfig,
    config_layout,
    load_config,
    load_secrets,
    write_default_config,
    write_default_secrets,
    write_secrets_gitignore,
)
from .errors import VoiceDatasetError
from .exporting import (
    current_training_fingerprint,
    export_training_dataset,
    verify_export_content,
)
from .gemini import GeminiInteractions
from .gemini_chunking import ChunkedGeminiSplitter
from .labeling import label_clips
from .media import MediaIngestor
from .models import (
    ClipRecord,
    InputMode,
    LabelRecord,
    ReviewState,
    Segment,
    SourceRecord,
    SplitBackend,
)
from .review import review_workspace
from .splitting import EnergySplitter, materialize_clips
from .workspace import Workspace


def _add_common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("workspace", type=Path, help="工作区目录")
    subparser.add_argument(
        "--config",
        type=Path,
        help="非敏感 TOML 配置；默认读取 <workspace>/config/pipeline.toml",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-dataset",
        description="音视频拆分、Gemini 标注、TUI 复核与训练编排",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="初始化本地工作区和默认配置")
    _add_common(init)
    init.add_argument("--overwrite-config", action="store_true")
    init.add_argument("--overwrite-secrets", action="store_true")

    ingest = commands.add_parser("ingest", help="递归导入并规范化音视频")
    _add_common(ingest)
    ingest.add_argument("inputs", nargs="+", type=Path)
    ingest.add_argument("--mode", choices=[item.value for item in InputMode], default="auto")

    split = commands.add_parser("split", help="仅拆分语音片段")
    _add_common(split)
    split.add_argument("--backend", choices=[item.value for item in SplitBackend])
    split.add_argument("--modality", choices=[item.value for item in InputMode])
    split.add_argument(
        "--secrets",
        type=Path,
        help="敏感配置；默认读取 <workspace>/secrets/credentials.toml",
    )
    split.add_argument("--limit", type=int)
    split.add_argument(
        "--replace",
        action="store_true",
        help="显式替换已有边界；保留不可变 WAV，但重置受影响复核状态",
    )

    preprocess = commands.add_parser(
        "preprocess",
        help="字幕优先回退拆分、音质门禁，并可选运行本地 SenseVoice",
    )
    _add_common(preprocess)
    preprocess.add_argument("inputs", nargs="*", type=Path)
    preprocess.add_argument("--mode", choices=[item.value for item in InputMode], default="auto")
    preprocess.add_argument("--replace", action="store_true")
    preprocess_asr = preprocess.add_mutually_exclusive_group()
    preprocess_asr.add_argument(
        "--asr",
        action="store_true",
        help="即使配置未启用，也运行 SenseVoice",
    )
    preprocess_asr.add_argument(
        "--skip-asr",
        action="store_true",
        help="只完成切分和质量门禁；即使配置已启用 ASR 也不在当前进程转写",
    )
    preprocess.add_argument("--force-asr", action="store_true")
    preprocess.add_argument("--force-quality", action="store_true")
    preprocess.add_argument(
        "--secrets",
        type=Path,
        help="Gemini 视觉回退所需敏感配置",
    )

    quality = commands.add_parser("quality", help="计算并持久化声学质量门禁")
    _add_common(quality)
    quality.add_argument("--force", action="store_true")

    transcribe = commands.add_parser("transcribe", help="本地 SenseVoice 转写与语音情绪识别")
    _add_common(transcribe)
    transcribe.add_argument("--force", action="store_true")
    transcribe.add_argument("--no-seed-labels", action="store_true")

    label = commands.add_parser("label", help="调用 Gemini 转写、情绪判断和聚类")
    _add_common(label)
    label.add_argument("--provider", choices=["gemini"], default="gemini")
    label.add_argument("--language", default="auto")
    label.add_argument(
        "--secrets",
        type=Path,
        help="敏感配置；默认读取 <workspace>/secrets/credentials.toml",
    )
    label.add_argument("--limit", type=int)
    label.add_argument("--force", action="store_true", help="重新标注已有条目")

    review = commands.add_parser("review", help="启动可断点恢复的按键式 TUI")
    _add_common(review)

    export = commands.add_parser("export", help="物化已复核的训练集")
    _add_common(export)
    export.add_argument("--output", type=Path)
    export.add_argument("--speaker")
    export.add_argument("--language")
    export.add_argument("--allow-unreviewed", action="store_true")

    train = commands.add_parser("train", help="生成训练计划；--execute 才执行")
    _add_common(train)
    train.add_argument("trainer", choices=["gpt-sovits", "rvc"])
    train.add_argument("--dataset", type=Path, help="导出数据集根目录；默认选择最新导出")
    train.add_argument("--execute", action="store_true")
    train.add_argument("--register-as", help="GPT-SoVITS 训练成功后登记模型名")
    train.add_argument("--activate", action="store_true", help="同时设为默认模型")

    emotion = commands.add_parser("emotion", help="分析目标台词并生成后端无关的情绪计划")
    _add_common(emotion)
    emotion.add_argument("--text", required=True)
    emotion.add_argument(
        "--secrets",
        type=Path,
        help="敏感配置；默认读取 <workspace>/secrets/credentials.toml",
    )

    model = commands.add_parser("model", help="登记、查看或激活角色模型")
    model_actions = model.add_subparsers(dest="model_action", required=True)
    model_register = model_actions.add_parser("register", help="登记一组已训练权重")
    _add_common(model_register)
    model_register.add_argument("--name", required=True)
    model_register.add_argument("--persona", default="")
    model_register.add_argument("--repository", type=Path, required=True)
    model_register.add_argument(
        "--python",
        type=Path,
        required=True,
        help="GPT-SoVITS 专属 Python 解释器",
    )
    model_register.add_argument("--gpt", type=Path, required=True)
    model_register.add_argument("--sovits", type=Path, required=True)
    model_register.add_argument("--manifest", type=Path)
    model_register.add_argument("--version", default="v2ProPlus")
    model_register.add_argument("--dataset-fingerprint", default="")
    model_register.add_argument("--provider-commit", default="")
    model_register.add_argument("--rvc-repository", type=Path)
    model_register.add_argument("--rvc-python", type=Path)
    model_register.add_argument("--rvc-model", type=Path)
    model_register.add_argument("--rvc-index", type=Path)
    model_register.add_argument("--activate", action="store_true")
    model_list = model_actions.add_parser("list", help="列出模型")
    _add_common(model_list)
    model_activate = model_actions.add_parser("activate", help="切换默认模型")
    _add_common(model_activate)
    model_activate.add_argument("name")

    synthesize = commands.add_parser("synthesize", help="文本 + 可选参考音频 -> 角色语音 WAV")
    _add_common(synthesize)
    synthesize.add_argument("--text", required=True)
    synthesize.add_argument("--output", type=Path, required=True)
    synthesize.add_argument("--model", help="模型注册名；省略时使用 active/default_model")
    synthesize.add_argument("--reference", type=Path)
    synthesize.add_argument("--reference-text")
    synthesize.add_argument("--emotion")
    synthesize.add_argument("--intensity", type=float, default=0.5)
    synthesize.add_argument("--language")
    synthesize.add_argument("--seed", type=int)
    synthesize.add_argument(
        "--postprocess",
        choices=["none", "rvc"],
        default="none",
        help="可选后处理；始终保留 .sovits.wav 原始输出",
    )
    synthesize.add_argument(
        "--secrets",
        type=Path,
        help="敏感配置；默认读取 <workspace>/secrets/credentials.toml",
    )

    status = commands.add_parser("status", help="只读显示工作区阶段计数")
    status.add_argument("workspace", type=Path, help="工作区目录")
    return parser


def _load_for(workspace: Path, explicit: Path | None) -> PipelineConfig:
    if explicit is not None:
        return load_config(explicit)
    root = workspace.expanduser().resolve()
    layout = config_layout(root)
    if layout.project.is_file():
        return load_config(layout.project)
    legacy = root / LEGACY_PROJECT_CONFIG_RELATIVE
    return load_config(legacy if legacy.is_file() else None)


def _load_secrets_for(workspace: Path, explicit: Path | None) -> SecretsConfig:
    if explicit is not None:
        return load_secrets(explicit)
    local = config_layout(workspace).secrets
    return load_secrets(local if local.is_file() else None)


def _init(args: argparse.Namespace) -> int:
    workspace = Workspace.create(args.workspace)
    layout = config_layout(workspace.root)
    legacy = workspace.root / LEGACY_PROJECT_CONFIG_RELATIVE
    if layout.project.exists() and not args.overwrite_config:
        print(f"保留项目配置: {layout.project}")
    elif args.config is not None:
        source = args.config.expanduser().resolve()
        load_config(source)
        layout.project.parent.mkdir(parents=True, exist_ok=True)
        if source != layout.project:
            shutil.copy2(source, layout.project)
        print(f"已复制配置模板: {source}")
    elif legacy.is_file() and not args.overwrite_config:
        load_config(legacy)
        layout.project.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, layout.project)
        print(f"已迁移旧项目配置: {legacy}")
    else:
        write_default_config(layout.project, overwrite=args.overwrite_config)

    write_secrets_gitignore(layout.secrets_gitignore)
    if layout.secrets.exists() and not args.overwrite_secrets:
        print(f"保留敏感配置: {layout.secrets}")
    else:
        write_default_secrets(layout.secrets, overwrite=args.overwrite_secrets)

    print(f"工作区: {workspace.root}")
    print(f"项目配置（可提交）: {layout.project}")
    print(f"敏感配置（Git 忽略）: {layout.secrets}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace)
    ingestor = MediaIngestor(workspace, config.media)
    records = ingestor.ingest(args.inputs, input_mode=InputMode(args.mode))
    print(f"已导入/复用 {len(records)} 个唯一媒体文件")
    print(f"来源清单: {workspace.paths.sources_jsonl}")
    return 0


def _gemini(
    config: PipelineConfig,
    workspace: Path,
    secrets_path: Path | None,
) -> GeminiInteractions:
    secrets = _load_secrets_for(workspace, secrets_path)
    return GeminiInteractions(
        model=config.gemini.model,
        api_key_env=config.gemini.api_key_env,
        api_key=secrets.get(config.gemini.api_key_env),
        timeout_seconds=config.gemini.timeout_seconds,
        max_retries=config.gemini.max_retries,
    )


def _gemini_splitter(
    config: PipelineConfig,
    workspace: Workspace,
    secrets_path: Path | None,
) -> ChunkedGeminiSplitter:
    return ChunkedGeminiSplitter(
        _gemini(config, workspace.root, secrets_path),
        config=config.gemini.chunking,
        ffmpeg_binary=config.media.ffmpeg_binary,
        scratch_dir=workspace.paths.state / "gemini_chunks",
        min_segment_seconds=config.splitting.min_segment_seconds,
        max_segment_seconds=config.splitting.max_segment_seconds,
    )


def _select_modality(source: SourceRecord, requested: InputMode) -> str | None:
    if requested is InputMode.AUDIO:
        return "audio"
    if requested is InputMode.VIDEO:
        return "video" if source.media_kind.value == "video" else None
    return "video" if source.media_kind.value == "video" else "audio"


def _reset_downstream_for_replaced_clips(
    workspace: Workspace,
    *,
    old_clip_ids: set[str],
    current_clips: list[ClipRecord],
) -> None:
    current_ids = {row.clip_id for row in current_clips}
    obsolete = old_clip_ids - current_ids
    if not obsolete:
        return
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert isinstance(labels, list)
    workspace.write_jsonl(
        workspace.paths.labels_jsonl,
        [row for row in labels if row.clip_id not in obsolete],
    )
    old_state = workspace.load_review()
    decisions = {
        clip_id: decision
        for clip_id, decision in old_state.decisions.items()
        if clip_id in current_ids
    }
    workspace.save_review(
        ReviewState(
            order=[row.clip_id for row in current_clips],
            decisions=decisions,
        )
    )


def _split(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    sources = workspace.read_jsonl(workspace.paths.sources_jsonl, SourceRecord)
    segments = workspace.read_jsonl(workspace.paths.segments_jsonl, Segment)
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert isinstance(sources, list)
    assert isinstance(segments, list)
    assert isinstance(clips, list)
    if not sources:
        raise ValueError("没有来源媒体；请先执行 ingest")

    backend = SplitBackend(args.backend or config.splitting.backend.value)
    requested = InputMode(args.modality or config.splitting.input_mode.value)
    local = EnergySplitter(config.splitting)
    remote = (
        _gemini_splitter(config, workspace, args.secrets)
        if backend is SplitBackend.GEMINI
        else None
    )
    processed = 0
    skipped = 0
    incompatible = 0

    for source in sources:
        if args.limit is not None and processed >= args.limit:
            break
        existing = [row for row in segments if row.source_id == source.source_id]
        if existing and not args.replace:
            skipped += 1
            continue
        modality = _select_modality(source, requested)
        if modality is None:
            incompatible += 1
            continue
        if backend is SplitBackend.ENERGY:
            proposed = local.split(source.normalized_path, source_id=source.source_id)
        else:
            assert remote is not None
            proposed = remote.split(source, modality=modality)
        if not proposed:
            if args.replace:
                raise ValueError(f"替换拆分未产生任何片段，旧结果已保留: {source.original_path}")
            print(f"[WARN] 未发现语音: {source.original_path}")
            processed += 1
            continue
        materialized = materialize_clips(
            source.normalized_path,
            proposed,
            workspace.paths.clips,
            source_id=source.source_id,
        )
        old_clip_ids = {row.clip_id for row in clips if row.source_id == source.source_id}
        segments = [row for row in segments if row.source_id != source.source_id] + proposed
        clips = [row for row in clips if row.source_id != source.source_id] + materialized
        workspace.write_jsonl(workspace.paths.clips_jsonl, clips)
        workspace.write_jsonl(workspace.paths.segments_jsonl, segments)
        if args.replace:
            _reset_downstream_for_replaced_clips(
                workspace,
                old_clip_ids=old_clip_ids,
                current_clips=clips,
            )
        processed += 1
        print(f"{source.original_path}: {len(materialized)} 段")

    print(
        f"完成：处理 {processed} 个来源，跳过已有 {skipped} 个，"
        f"模式不兼容 {incompatible} 个；当前共 {len(clips)} 段"
    )
    print(f"片段清单: {workspace.paths.clips_jsonl}")
    return 0


def _quality_thresholds(config: PipelineConfig):
    from .quality import QualityThresholds

    return QualityThresholds(
        min_duration_seconds=config.quality.min_duration_seconds,
        max_duration_seconds=config.quality.max_duration_seconds,
        min_rms_dbfs=config.quality.min_rms_dbfs,
        max_rms_dbfs=config.quality.max_rms_dbfs,
        max_clipping_ratio=config.quality.max_clipping_ratio,
        max_silence_ratio=config.quality.max_silence_ratio,
    )


def _asr_profile(config: PipelineConfig):
    from .asr import ASRProfile

    return ASRProfile(
        model=config.asr.model,
        vad_model=config.asr.vad_model,
        language=config.asr.language,
        replacements=config.asr.replacements,
        minimum_similarity=config.quality.min_transcript_similarity,
        require_expected_match=config.quality.require_asr,
        model_revision=config.asr.model_revision,
        vad_revision=config.asr.vad_revision,
        funasr_version=config.asr.funasr_version,
        modelscope_version=config.asr.modelscope_version,
    )


def _quality(args: argparse.Namespace) -> int:
    from .quality import evaluate_workspace

    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    result = evaluate_workspace(
        workspace,
        thresholds=_quality_thresholds(config),
        force=args.force,
    )
    print(
        f"质量门禁：总计 {result.total}，本次 {result.evaluated}，复用 {result.reused}，"
        f"通过 {result.accepted}，拒绝 {result.rejected}"
    )
    print(f"清单: {workspace.paths.quality_jsonl}")
    return 0


def _transcribe_workspace(
    config: PipelineConfig,
    workspace: Workspace,
    *,
    force: bool,
    seed_labels: bool,
) -> None:
    from .asr import SenseVoiceTranscriber, transcribe_workspace

    transcriber = SenseVoiceTranscriber(
        model=config.asr.model,
        vad_model=config.asr.vad_model,
        model_revision=config.asr.model_revision,
        vad_revision=config.asr.vad_revision,
        expected_funasr_version=config.asr.funasr_version,
        expected_modelscope_version=config.asr.modelscope_version,
        device=config.asr.device,
        language=config.asr.language,
        replacements=config.asr.replacements,
    )
    result = transcribe_workspace(
        workspace,
        transcriber,
        minimum_similarity=config.quality.min_transcript_similarity,
        require_expected_match=config.quality.require_asr,
        force=force,
        seed_labels=seed_labels,
    )
    print(
        f"SenseVoice：总计 {result.total}，本次 {result.transcribed}，"
        f"复用 {result.reused}，通过 {result.accepted}"
    )
    print(f"清单: {workspace.paths.asr_jsonl}")


def _transcribe(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    _transcribe_workspace(
        config,
        workspace,
        force=args.force,
        seed_labels=not args.no_seed_labels,
    )
    return 0


def _preprocess(args: argparse.Namespace) -> int:
    from .preprocessing import PreprocessingPipeline, default_subtitle_extractor
    from .quality import evaluate_workspace

    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace)
    if args.inputs:
        MediaIngestor(workspace, config.media).ingest(
            args.inputs,
            input_mode=InputMode(args.mode),
        )
    sources = workspace.read_jsonl(workspace.paths.sources_jsonl, SourceRecord)
    segments = workspace.read_jsonl(workspace.paths.segments_jsonl, Segment)
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert isinstance(sources, list)
    assert isinstance(segments, list)
    assert isinstance(clips, list)
    if not sources:
        raise ValueError("没有来源媒体；请传入输入路径或先执行 ingest")
    vision = None
    if config.splitting.backend is SplitBackend.GEMINI:
        vision = _gemini_splitter(config, workspace, args.secrets)
    pipeline = PreprocessingPipeline(
        silence_splitter=EnergySplitter(config.splitting),
        embedded_extractor=default_subtitle_extractor(
            ffmpeg_binary=config.media.ffmpeg_binary,
            ffprobe_binary="ffprobe",
        ),
        vision_splitter=vision,
    )
    processed = 0
    skipped = 0
    scratch = workspace.paths.state / "subtitles"
    for source in sources:
        existing = [row for row in segments if row.source_id == source.source_id]
        if existing and not args.replace:
            skipped += 1
            continue
        result = pipeline.split(source, scratch_dir=scratch)
        if not result.segments:
            print(f"[WARN] 未发现语音: {source.original_path}")
            continue
        materialized = materialize_clips(
            source.normalized_path,
            result.segments,
            workspace.paths.clips,
            source_id=source.source_id,
        )
        old_ids = {row.clip_id for row in clips if row.source_id == source.source_id}
        segments = [row for row in segments if row.source_id != source.source_id] + list(
            result.segments
        )
        clips = [row for row in clips if row.source_id != source.source_id] + materialized
        workspace.write_jsonl(workspace.paths.segments_jsonl, segments)
        workspace.write_jsonl(workspace.paths.clips_jsonl, clips)
        if args.replace:
            _reset_downstream_for_replaced_clips(
                workspace,
                old_clip_ids=old_ids,
                current_clips=clips,
            )
        processed += 1
        print(
            f"{source.original_path}: {len(materialized)} 段，"
            f"策略 {result.strategy.value}，尝试 "
            f"{', '.join(item.value for item in result.attempts)}"
        )
        for strategy, failure in result.failures.items():
            print(f"[WARN] {strategy}: {failure}")
    if config.quality.enabled:
        quality = evaluate_workspace(
            workspace,
            thresholds=_quality_thresholds(config),
            force=args.force_quality,
        )
        quality_status = f"质量通过 {quality.accepted}/{quality.total}"
    else:
        quality_status = "质量门禁已禁用，跳过"
    print(f"预处理完成：处理 {processed}，跳过 {skipped}；当前 {len(clips)} 段；{quality_status}")
    if not args.skip_asr and (args.asr or config.asr.enabled):
        _transcribe_workspace(
            config,
            workspace,
            force=args.force_asr,
            seed_labels=True,
        )
    return 0


def _label(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    result = label_clips(
        workspace,
        _gemini(config, args.workspace, args.secrets),
        emotions=config.review.emotions,
        clusters=config.review.clusters,
        language_hint=args.language,
        force=args.force,
        limit=args.limit,
        replacements=config.asr.replacements,
    )
    print(f"标注完成：总计 {result.total}，本次 {result.labelled}，复用 {result.skipped}")
    print(f"标注清单: {workspace.paths.labels_jsonl}")
    return 0


def _review(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    state = review_workspace(
        workspace,
        emotions=config.review.emotions,
        play_command=config.review.play_command,
    )
    print(f"复核进度: {state.cursor}/{len(state.order)}")
    print(f"状态: {workspace.paths.review_json}")
    return 0


def _dataset_config_fingerprint(config: PipelineConfig) -> str:
    """Hash only configuration that can affect dataset contents."""

    payload = config.model_dump(
        mode="json",
        include={"media", "splitting", "gemini", "asr", "quality", "review"},
    )
    payload["export_defaults"] = {
        "speaker": config.training.gpt_sovits.speaker or "speaker",
        "language": config.training.gpt_sovits.language or "zh",
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _export(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    trainer = config.training.gpt_sovits
    result = export_training_dataset(
        workspace,
        output_root=args.output,
        speaker=args.speaker or trainer.speaker or "speaker",
        language=args.language or trainer.language or "zh",
        allow_unreviewed=args.allow_unreviewed,
        config_fingerprint=_dataset_config_fingerprint(config),
        quality_enabled=config.quality.enabled,
        quality_profile_sha256=_quality_thresholds(config).fingerprint,
        require_asr=config.quality.require_asr,
        asr_profile=_asr_profile(config),
    )
    workspace.upsert_jsonl(
        workspace.paths.state / "exports.jsonl",
        {
            "fingerprint": result.fingerprint,
            "root": str(result.root),
            "created_at": datetime.now(UTC).isoformat(),
        },
        key="fingerprint",
    )
    print(f"训练集: {result.root}")
    print(f"纳入 {result.included}，排除 {result.excluded}")
    print(f"指纹: {result.fingerprint}")
    return 0


def _latest_export(workspace: Workspace) -> Path:
    base = workspace.paths.training / "exports"
    candidates = {
        path.resolve() for path in base.glob("dataset-*") if (path / "metadata.json").is_file()
    }
    for row in workspace.read_jsonl(workspace.paths.state / "exports.jsonl"):
        path = Path(row.get("root", "")).expanduser()
        if path.is_dir() and (path / "metadata.json").is_file():
            candidates.add(path.resolve())
    ordered = sorted(candidates, key=lambda path: (path / "metadata.json").stat().st_mtime)
    if not ordered:
        raise ValueError("没有可用导出；请先执行 export")
    return ordered[-1]


def _verify_current_export(
    workspace: Workspace,
    config: PipelineConfig,
    dataset: Path,
) -> None:
    metadata_path = dataset / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"导出元数据不存在: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"导出元数据无效: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"导出元数据必须是 JSON object: {metadata_path}")
    verify_export_content(dataset, metadata)

    config_fingerprint = _dataset_config_fingerprint(config)
    if metadata.get("pipeline_config_sha256") != config_fingerprint:
        raise ValueError("导出训练集与当前预处理配置不一致；请重新执行 export")
    try:
        current = current_training_fingerprint(
            workspace,
            speaker=str(metadata["speaker"]),
            language=str(metadata["language"]),
            allow_unreviewed=bool(metadata.get("allow_unreviewed", False)),
            config_fingerprint=config_fingerprint,
            quality_enabled=config.quality.enabled,
            quality_profile_sha256=_quality_thresholds(config).fingerprint,
            require_asr=config.quality.require_asr,
            asr_profile=_asr_profile(config),
        )
    except KeyError as exc:
        raise ValueError(f"导出元数据缺少字段: {exc.args[0]}") from exc
    if metadata.get("fingerprint") != current:
        raise ValueError("导出训练集已因片段或复核结果变化而过期；请重新执行 export")


def _train(args: argparse.Namespace) -> int:
    from .training import GPTSoVITSAdapter, RVCAdapter

    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    dataset = args.dataset.resolve() if args.dataset else _latest_export(workspace)
    _verify_current_export(workspace, config, dataset)
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"训练清单不存在: {manifest}")
    if args.trainer == "gpt-sovits":
        trainer_config = config.training.gpt_sovits
        adapter = GPTSoVITSAdapter(config, workspace.root)
    else:
        trainer_config = config.training.rvc
        adapter = RVCAdapter(config, workspace.root)
    plan = adapter.plan(
        manifest,
        experiment=trainer_config.experiment_name or None,
    )
    print(json.dumps(plan.serialise(), ensure_ascii=False, indent=2))
    if args.execute:
        if not config.training.enabled or not trainer_config.enabled:
            raise ValueError("执行训练前必须同时启用 training.enabled 和对应训练器的 enabled")
        result = plan.execute()
        print("训练命令执行完成；已通过产物门禁。")
        if args.register_as:
            if args.trainer != "gpt-sovits":
                raise ValueError("--register-as 当前只支持 GPT-SoVITS")
            from .registry import VoiceModelRecord

            artifacts = {row["kind"]: row for row in result["artifacts"]}
            by_kind = {kind: Path(row["path"]) for kind, row in artifacts.items()}
            missing = sorted({"gpt", "sovits"} - by_kind.keys())
            if missing:
                raise ValueError(f"训练产物缺少可登记权重: {missing}")
            repository = trainer_config.repository
            if repository is None:
                raise ValueError("training.gpt_sovits.repository 未配置")
            provider_python = trainer_config.python
            if provider_python is None:
                raise ValueError("training.gpt_sovits.python 未配置")
            record = VoiceModelRecord(
                name=args.register_as,
                persona=trainer_config.speaker,
                version=trainer_config.model_version,
                repository=repository.resolve(),
                python=provider_python.resolve(),
                gpt_weights=by_kind["gpt"].resolve(),
                sovits_weights=by_kind["sovits"].resolve(),
                gpt_weights_sha256=str(artifacts["gpt"].get("sha256", "")),
                sovits_weights_sha256=str(artifacts["sovits"].get("sha256", "")),
                reference_manifest=manifest.resolve(),
                dataset_fingerprint=plan.metadata.get("dataset", {}).get("selected_sha256", ""),
                provider_commit=plan.metadata.get("provider", {}).get("git_head", ""),
                provider_dirty_sha256=plan.metadata.get("provider", {}).get(
                    "git_tracked_diff_sha256", ""
                ),
                provider_assets_sha256={
                    name: str(plan.metadata.get("provider", {}).get("hashes", {}).get(name, ""))
                    for name in ("bert", "hubert", "g2pw", "language_detector", "sv")
                    if plan.metadata.get("provider", {}).get("hashes", {}).get(name)
                },
            )
            _registry_for(args.workspace, config).register(record, activate=args.activate)
            print(f"已登记模型: {record.name}")
    else:
        print("仅生成计划；传入 --execute 才会启动外部训练。")
    return 0


def _registry_for(workspace: Path, config: PipelineConfig):
    from .registry import ModelRegistry

    configured = config.registry.path.expanduser()
    path = configured if configured.is_absolute() else workspace.expanduser().resolve() / configured
    return ModelRegistry(path)


def _emotion_analyzer(
    config: PipelineConfig,
    workspace: Path,
    secrets_path: Path | None,
):
    from .emotion import OpenAICompatibleEmotionAnalyzer, RuleBasedEmotionAnalyzer

    if config.emotion.provider == "rules":
        return RuleBasedEmotionAnalyzer()
    secrets = _load_secrets_for(workspace, secrets_path)
    token = secrets.get(config.emotion.api_key_env)
    if not token:
        raise ValueError(
            f"情绪服务密钥为空；请在 secrets/credentials.toml 设置 {config.emotion.api_key_env}"
        )
    return OpenAICompatibleEmotionAnalyzer(
        base_url=config.emotion.base_url,
        model=config.emotion.model,
        api_key=token,
        timeout_seconds=config.emotion.timeout_seconds,
    )


def _emotion(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    plan = _emotion_analyzer(config, args.workspace, args.secrets).analyze(args.text)
    print(plan.model_dump_json(indent=2))
    return 0


def _model(args: argparse.Namespace) -> int:
    from .registry import VoiceModelRecord

    config = _load_for(args.workspace, args.config)
    registry = _registry_for(args.workspace, config)
    if args.model_action == "list":
        print(
            json.dumps(
                [row.model_dump(mode="json") for row in registry.list()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.model_action == "activate":
        record = registry.activate(args.name)
        print(f"已激活模型: {record.name}")
        return 0
    workspace = Workspace.open(args.workspace, create=False)
    manifest = (
        args.manifest.resolve() if args.manifest else _latest_export(workspace) / "manifest.jsonl"
    )
    record = VoiceModelRecord(
        name=args.name,
        persona=args.persona,
        version=args.version,
        repository=args.repository.resolve(),
        python=args.python.resolve(),
        gpt_weights=args.gpt.resolve(),
        sovits_weights=args.sovits.resolve(),
        reference_manifest=manifest.resolve(),
        dataset_fingerprint=args.dataset_fingerprint,
        provider_commit=args.provider_commit,
        vc_backend="rvc" if args.rvc_model else "none",
        vc_repository=args.rvc_repository.resolve() if args.rvc_repository else None,
        vc_python=args.rvc_python.resolve() if args.rvc_python else None,
        vc_model=args.rvc_model.resolve() if args.rvc_model else None,
        vc_index=args.rvc_index.resolve() if args.rvc_index else None,
    )
    registry.register(record, activate=args.activate)
    print(f"已登记模型: {record.name}")
    print(f"注册表: {registry.path}")
    return 0


def _synthesize(args: argparse.Namespace) -> int:
    from .synthesis import SynthesisService

    config = _load_for(args.workspace, args.config)
    name = args.model or config.inference.default_model or None
    model = _registry_for(args.workspace, config).get(name)
    service = SynthesisService(
        model,
        _emotion_analyzer(config, args.workspace, args.secrets),
        device=config.inference.device,
        half=config.inference.half,
        preferred_reference_min=config.reference.preferred_min_seconds,
        preferred_reference_max=config.reference.preferred_max_seconds,
    )
    raw_output = args.output
    if args.postprocess == "rvc":
        raw_output = args.output.with_name(f"{args.output.stem}.sovits{args.output.suffix}")
    result = service.synthesize(
        args.text,
        raw_output,
        reference_audio=args.reference,
        reference_text=args.reference_text,
        emotion=args.emotion,
        intensity=args.intensity,
        language=args.language or config.inference.language,
        seed=args.seed if args.seed is not None else config.inference.seed,
    )
    print(f"输出: {result.output}")
    print(f"采样率: {result.sample_rate}")
    print(f"情绪: {result.emotion.emotion} ({result.emotion.intensity:.2f})")
    print(f"参考: {result.reference.audio_path}")
    if args.postprocess == "rvc":
        from .postprocess import RVCOptions, RVCPostprocessor

        if model.vc_backend != "rvc":
            raise ValueError("当前模型未登记 RVC 后处理权重")
        processor = RVCPostprocessor(
            repository=model.vc_repository,
            python=model.vc_python,
            model=model.vc_model,
            index=model.vc_index,
            model_sha256=model.vc_model_sha256,
            index_sha256=model.vc_index_sha256,
            provider_commit=model.vc_provider_commit,
            provider_dirty_sha256=model.vc_provider_dirty_sha256,
            provider_code_sha256=model.vc_provider_code_sha256,
            provider_assets_sha256=model.vc_provider_assets_sha256,
            options=RVCOptions(
                f0_method=config.postprocess.f0_method,
                transpose=config.postprocess.transpose,
                index_rate=config.postprocess.index_rate,
                rms_mix_rate=config.postprocess.rms_mix_rate,
                protect=config.postprocess.protect,
            ),
        )
        final = processor.convert(result.output, args.output)
        print(f"RVC 输出: {final}")
        print(f"SoVITS 原始输出已保留: {result.output}")
    return 0


def _status(args: argparse.Namespace) -> int:
    workspace = Workspace.open(args.workspace, create=False)
    counts: dict[str, int] = {}
    for name, path, model in (
        ("sources", workspace.paths.sources_jsonl, SourceRecord),
        ("segments", workspace.paths.segments_jsonl, Segment),
        ("clips", workspace.paths.clips_jsonl, ClipRecord),
        ("labels", workspace.paths.labels_jsonl, LabelRecord),
        ("quality", workspace.paths.quality_jsonl, None),
        ("asr", workspace.paths.asr_jsonl, None),
    ):
        counts[name] = len(workspace.read_jsonl(path, model))
    state = workspace.load_review()
    counts["reviewed"] = sum(1 for decision in state.decisions.values() if decision.confirmed)
    counts["draft_decisions"] = sum(
        1 for decision in state.decisions.values() if not decision.confirmed
    )
    counts["review_cursor"] = state.cursor
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


_HANDLERS: dict[str, Any] = {
    "init": _init,
    "ingest": _ingest,
    "split": _split,
    "preprocess": _preprocess,
    "quality": _quality,
    "transcribe": _transcribe,
    "label": _label,
    "review": _review,
    "export": _export,
    "train": _train,
    "emotion": _emotion,
    "model": _model,
    "synthesize": _synthesize,
    "status": _status,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(_HANDLERS[args.command](args))
    except KeyboardInterrupt:
        print("\n[ERROR] 已中断", file=sys.stderr)
        return 130
    except (VoiceDatasetError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
