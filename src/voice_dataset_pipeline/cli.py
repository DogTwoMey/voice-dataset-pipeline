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
from .config import PipelineConfig, load_config, write_default_config
from .errors import VoiceDatasetError
from .exporting import current_training_fingerprint, export_training_dataset
from .gemini import GeminiInteractions
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
        help="TOML 配置；默认读取 <workspace>/pipeline.toml",
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

    ingest = commands.add_parser("ingest", help="递归导入并规范化音视频")
    _add_common(ingest)
    ingest.add_argument("inputs", nargs="+", type=Path)
    ingest.add_argument("--mode", choices=[item.value for item in InputMode], default="auto")

    split = commands.add_parser("split", help="仅拆分语音片段")
    _add_common(split)
    split.add_argument("--backend", choices=[item.value for item in SplitBackend])
    split.add_argument("--modality", choices=[item.value for item in InputMode])
    split.add_argument("--limit", type=int)
    split.add_argument(
        "--replace",
        action="store_true",
        help="显式替换已有边界；保留不可变 WAV，但重置受影响复核状态",
    )

    label = commands.add_parser("label", help="调用 Gemini 转写、情绪判断和聚类")
    _add_common(label)
    label.add_argument("--provider", choices=["gemini"], default="gemini")
    label.add_argument("--language", default="auto")
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

    status = commands.add_parser("status", help="只读显示工作区阶段计数")
    status.add_argument("workspace", type=Path, help="工作区目录")
    return parser


def _load_for(workspace: Path, explicit: Path | None) -> PipelineConfig:
    if explicit is not None:
        return load_config(explicit)
    local = workspace.expanduser().resolve() / "pipeline.toml"
    return load_config(local if local.is_file() else None)


def _init(args: argparse.Namespace) -> int:
    workspace = Workspace.create(args.workspace)
    destination = workspace.root / "pipeline.toml"
    if destination.exists() and not args.overwrite_config:
        print(f"工作区已存在，保留配置: {destination}")
        return 0
    if args.config is not None:
        source = args.config.expanduser().resolve()
        load_config(source)
        if source != destination:
            shutil.copy2(source, destination)
        print(f"已复制配置模板: {source}")
    else:
        write_default_config(destination, overwrite=args.overwrite_config)
    print(f"工作区: {workspace.root}")
    print(f"配置: {destination}")
    return 0


def _ingest(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace)
    ingestor = MediaIngestor(workspace, config.media)
    records = ingestor.ingest(args.inputs, input_mode=InputMode(args.mode))
    print(f"已导入/复用 {len(records)} 个唯一媒体文件")
    print(f"来源清单: {workspace.paths.sources_jsonl}")
    return 0


def _gemini(config: PipelineConfig) -> GeminiInteractions:
    return GeminiInteractions(
        model=config.gemini.model,
        api_key_env=config.gemini.api_key_env,
        timeout_seconds=config.gemini.timeout_seconds,
        max_retries=config.gemini.max_retries,
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
    remote = _gemini(config) if backend is SplitBackend.GEMINI else None
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
            media_path = source.original_path if modality == "video" else source.normalized_path
            proposed = remote.split(
                path=media_path,
                modality=modality,
                source_id=source.source_id,
                duration_seconds=source.duration_seconds,
                min_segment_seconds=config.splitting.min_segment_seconds,
                max_segment_seconds=config.splitting.max_segment_seconds,
            )
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


def _label(args: argparse.Namespace) -> int:
    config = _load_for(args.workspace, args.config)
    workspace = Workspace.open(args.workspace, create=False)
    result = label_clips(
        workspace,
        _gemini(config),
        emotions=config.review.emotions,
        clusters=config.review.clusters,
        language_hint=args.language,
        force=args.force,
        limit=args.limit,
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

    payload = config.model_dump(mode="json", exclude={"training"})
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
        plan.execute()
        print("训练命令执行完成；已通过产物门禁。")
    else:
        print("仅生成计划；传入 --execute 才会启动外部训练。")
    return 0


def _status(args: argparse.Namespace) -> int:
    workspace = Workspace.open(args.workspace, create=False)
    counts: dict[str, int] = {}
    for name, path, model in (
        ("sources", workspace.paths.sources_jsonl, SourceRecord),
        ("segments", workspace.paths.segments_jsonl, Segment),
        ("clips", workspace.paths.clips_jsonl, ClipRecord),
        ("labels", workspace.paths.labels_jsonl, LabelRecord),
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
    "label": _label,
    "review": _review,
    "export": _export,
    "train": _train,
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
