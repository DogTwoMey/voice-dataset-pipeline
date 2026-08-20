"""Crash-resumable keyboard review for model-generated labels."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .models import (
    ASRRecord,
    ClipRecord,
    LabelRecord,
    QualityRecord,
    ReviewDecision,
    ReviewMergeReceipt,
    ReviewState,
    Segment,
    SourceRecord,
)
from .splitting import materialize_clips
from .workspace import Workspace

KeyReader = Callable[[], str]


def read_key() -> str:
    """Read one key without requiring Enter on Windows or POSIX terminals."""

    if os.name == "nt":  # pragma: no cover - exercised manually on Windows
        import msvcrt

        key = msvcrt.getwch()
        if key in {"\x00", "\xe0"}:
            msvcrt.getwch()
            return ""
        return key

    import termios  # pragma: no cover - platform dependent
    import tty

    descriptor = sys.stdin.fileno()
    old = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, old)


def play_audio(path: Path, play_command: str = "") -> None:
    """Open a clip without moving or modifying it."""

    target = path.resolve()
    if play_command:
        argv = [
            part.format(file=str(target))
            for part in shlex.split(play_command, posix=os.name != "nt")
        ]
        subprocess.Popen(argv, close_fds=True)  # noqa: S603
    elif os.name == "nt":  # pragma: no cover - exercised manually on Windows
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":  # pragma: no cover - platform dependent
        subprocess.Popen(["open", str(target)], close_fds=True)  # noqa: S603
    else:  # pragma: no cover - platform dependent
        opener = shutil.which("xdg-open")
        if not opener:
            raise RuntimeError("找不到默认播放器；请在 review.play_command 中配置命令")
        subprocess.Popen([opener, str(target)], close_fds=True)  # noqa: S603


def _fallback_decision(
    clip: ClipRecord,
    label: LabelRecord | None,
    current: ReviewDecision | None,
) -> ReviewDecision:
    if current is not None:
        return current.model_copy(deep=True)
    return ReviewDecision(
        clip_id=clip.clip_id,
        emotion=label.emotion if label else clip.emotion,
        cluster=label.cluster if label else clip.cluster,
        transcript=label.transcript if label else clip.text,
    )


def _remember(state: ReviewState, decision: ReviewDecision | None) -> None:
    payload = {
        "cursor": state.cursor,
        "clip_id": decision.clip_id if decision else state.order[state.cursor],
        "previous": decision.model_dump(mode="json") if decision else None,
    }
    state.history.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    if len(state.history) > 2_000:
        del state.history[: len(state.history) - 2_000]


def _undo(state: ReviewState) -> None:
    if not state.history:
        return
    payload = json.loads(state.history.pop())
    state.cursor = int(payload["cursor"])
    clip_id = str(payload["clip_id"])
    previous = payload.get("previous")
    if previous is None:
        state.decisions.pop(clip_id, None)
    else:
        state.decisions[clip_id] = ReviewDecision.model_validate(previous)


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")  # noqa: S605


def _joined_text(left: str, right: str) -> str:
    first = left.strip()
    second = right.strip()
    if not first:
        return second
    if not second:
        return first
    separator = " " if first[-1].isascii() and second[0].isascii() else ""
    return first + separator + second


def _active_timeline_pair(
    clips: Sequence[ClipRecord], left_id: str, right_id: str
) -> tuple[ClipRecord, ClipRecord]:
    """Resolve and validate one merge pair from the canonical active manifest."""

    if left_id == right_id:
        raise ValueError("a clip cannot be merged with itself")
    active_by_id: dict[str, ClipRecord] = {}
    for clip in clips:
        if clip.clip_id in active_by_id:
            raise ValueError(f"active clip manifest contains duplicate clip_id: {clip.clip_id}")
        active_by_id[clip.clip_id] = clip
    if left_id not in active_by_id or right_id not in active_by_id:
        raise ValueError("both clips must be present in the canonical active manifest")

    left = active_by_id[left_id]
    right = active_by_id[right_id]
    if left.source_id != right.source_id:
        raise ValueError("only clips from the same source can be merged")

    timeline = sorted(
        (clip for clip in clips if clip.source_id == left.source_id),
        key=lambda clip: (clip.start_ms, clip.end_ms, clip.clip_id),
    )
    left_index = next(index for index, clip in enumerate(timeline) if clip.clip_id == left.clip_id)
    if left_index + 1 >= len(timeline) or timeline[left_index + 1].clip_id != right.clip_id:
        raise ValueError("clips must be direct timeline neighbors in the canonical active manifest")
    if right.start_ms < left.end_ms:
        raise ValueError("overlapping clips cannot be merged")

    for clip in timeline:
        if clip.clip_id in {left.clip_id, right.clip_id}:
            continue
        if clip.start_ms < right.end_ms and clip.end_ms > left.start_ms:
            raise ValueError("merge span overlaps another active clip")
    return left, right


def merge_adjacent_clips(
    workspace: Workspace,
    left: ClipRecord,
    right: ClipRecord,
    *,
    transcript: str,
    emotion: str,
    cluster: str,
) -> tuple[ClipRecord, LabelRecord]:
    """Create a new immutable clip spanning an adjacent pair; old WAVs remain."""

    sources = workspace.read_jsonl(workspace.paths.sources_jsonl, SourceRecord)
    segments = workspace.read_jsonl(workspace.paths.segments_jsonl, Segment)
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    assert isinstance(sources, list)
    assert isinstance(segments, list)
    assert isinstance(clips, list)
    left, right = _active_timeline_pair(clips, left.clip_id, right.clip_id)
    review_state = workspace.load_review().model_copy(deep=True)
    pair_index: int | None = None
    if review_state.order:
        pair_index = next(
            (
                index
                for index in range(len(review_state.order) - 1)
                if review_state.order[index : index + 2] == [left.clip_id, right.clip_id]
            ),
            None,
        )
        if pair_index is None:
            raise ValueError("review state does not contain the adjacent clips to merge")
    source = next((row for row in sources if row.source_id == left.source_id), None)
    if source is None:
        raise ValueError(f"source is missing for merge: {left.source_id}")
    merged_segment = Segment(
        source_id=left.source_id,
        start_seconds=left.start_seconds,
        end_seconds=right.end_seconds,
        text_hint=transcript,
        provenance={
            "strategy": "manual_merge",
            "left_clip_id": left.clip_id,
            "right_clip_id": right.clip_id,
        },
    )
    merged = materialize_clips(
        source.normalized_path,
        [merged_segment],
        workspace.paths.clips,
        source_id=left.source_id,
    )[0].model_copy(update={"text": transcript, "emotion": emotion, "cluster": cluster})
    replacement: list[ClipRecord] = []
    inserted = False
    for clip in clips:
        if clip.clip_id in {left.clip_id, right.clip_id}:
            if not inserted:
                replacement.append(merged)
                inserted = True
            continue
        replacement.append(clip)
    if not inserted:  # guarded by _active_timeline_pair; retain a defensive assertion
        raise AssertionError("validated merge pair disappeared from the active manifest")
    remaining_segments = [
        row
        for row in segments
        if not (
            row.source_id == left.source_id
            and round(row.start_seconds * 1_000) in {left.start_ms, right.start_ms}
            and round(row.end_seconds * 1_000) in {left.end_ms, right.end_ms}
        )
    ]
    projections: dict[Path, list[QualityRecord] | list[ASRRecord]] = {}
    for path, model in (
        (workspace.paths.quality_jsonl, QualityRecord),
        (workspace.paths.asr_jsonl, ASRRecord),
    ):
        records = workspace.read_jsonl(path, model)
        assert isinstance(records, list)
        projections[path] = [
            row for row in records if row.clip_id not in {left.clip_id, right.clip_id}
        ]
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert isinstance(labels, list)
    label = LabelRecord(
        clip_id=merged.clip_id,
        transcript=transcript,
        emotion=emotion,
        cluster=cluster,
        rationale="manual merge; rerun quality and SenseVoice before export",
        model="manual-merge",
    )
    replacement_labels = [
        row for row in labels if row.clip_id not in {left.clip_id, right.clip_id}
    ] + [label]
    if review_state.order:
        assert pair_index is not None
        review_state.order[pair_index : pair_index + 2] = [merged.clip_id]
    else:
        review_state.order = [row.clip_id for row in replacement]
    review_state.decisions.pop(left.clip_id, None)
    review_state.decisions.pop(right.clip_id, None)
    review_state.decisions[merged.clip_id] = ReviewDecision(
        clip_id=merged.clip_id,
        emotion=label.emotion,
        cluster=label.cluster,
        transcript=label.transcript,
    )
    review_state.cursor = min(review_state.cursor, len(review_state.order))
    review_state.history.clear()
    workspace.commit_review_merge(
        ReviewMergeReceipt(
            left_clip_id=left.clip_id,
            right_clip_id=right.clip_id,
            merged_clip_id=merged.clip_id,
            clips=replacement,
            segments=[*remaining_segments, merged_segment],
            labels=replacement_labels,
            quality=list(projections[workspace.paths.quality_jsonl]),
            asr=list(projections[workspace.paths.asr_jsonl]),
            review_state=review_state,
        )
    )
    return merged, label


def _draw(
    *,
    state: ReviewState,
    clip: ClipRecord,
    label: LabelRecord | None,
    decision: ReviewDecision,
    emotions: Sequence[str],
    previous_text: str = "",
    next_text: str = "",
) -> None:
    width = min(120, max(80, shutil.get_terminal_size((100, 24)).columns))
    bar = "=" * width
    print(bar)
    print("Voice Dataset Review")
    remaining = len(state.order) - state.cursor - 1
    print(f"Progress: {state.cursor + 1}/{len(state.order)}  Remaining: {remaining}")
    print(f"Clip: {clip.clip_id}")
    print(f"Audio: {clip.audio_path}")
    print(f"Emotion: {decision.emotion}  Cluster: {decision.cluster}")
    print("-" * width)
    print(decision.transcript or "(no transcript)")
    if previous_text:
        print(f"[Previous] {previous_text}")
    if next_text:
        print(f"[Next] {next_text}")
    print("-" * width)
    options = "  ".join(f"[{index}] {emotion}" for index, emotion in enumerate(emotions, 1))
    print(options)
    print("[0] Play  [M] Merge next  [X] Exclude  [E] Edit text  [K] Edit cluster  [R] Refresh")
    print("[S] Skip to end  [B] Undo last  [Q] Save and quit")
    if label and label.rationale:
        print(f"Hint: {label.rationale}")
    print(bar)
    print("Key: ", end="", flush=True)


def review_workspace(
    workspace: Workspace,
    *,
    emotions: Sequence[str],
    play_command: str = "",
    key_reader: KeyReader = read_key,
    clear_screen: bool = True,
) -> ReviewState:
    """Review clips; every mutation is atomically saved before advancing."""

    choices = [item.strip() for item in emotions if item.strip()]
    if not choices or len(choices) > 9:
        raise ValueError("review emotions must contain between 1 and 9 entries")
    workspace.recover_review_merge()
    clips = workspace.read_jsonl(workspace.paths.clips_jsonl, ClipRecord)
    labels = workspace.read_jsonl(workspace.paths.labels_jsonl, LabelRecord)
    assert isinstance(clips, list)
    assert isinstance(labels, list)
    clip_by_id = {row.clip_id: row for row in clips}
    label_by_id = {row.clip_id: row for row in labels}

    state = workspace.load_review()
    if not state.order:
        state.order = [row.clip_id for row in clips]
        workspace.save_review(state)
    else:
        known = set(state.order)
        additions = [row.clip_id for row in clips if row.clip_id not in known]
        if additions:
            state.order.extend(additions)
            workspace.save_review(state)
    missing = [clip_id for clip_id in state.order if clip_id not in clip_by_id]
    if missing:
        raise ValueError(f"review state refers to missing clips: {missing[:3]}")

    try:
        while state.cursor < len(state.order):
            clip_id = state.order[state.cursor]
            clip = clip_by_id[clip_id]
            label = label_by_id.get(clip_id)
            current = state.decisions.get(clip_id)
            shown = _fallback_decision(clip, label, current)
            context: list[str] = []
            for offset in (-1, 1):
                index = state.cursor + offset
                if not 0 <= index < len(state.order):
                    context.append("")
                    continue
                neighbour_id = state.order[index]
                neighbour_clip = clip_by_id[neighbour_id]
                neighbour_label = label_by_id.get(neighbour_id)
                neighbour_decision = state.decisions.get(neighbour_id)
                context.append(
                    _fallback_decision(
                        neighbour_clip,
                        neighbour_label,
                        neighbour_decision,
                    ).transcript
                )
            if clear_screen:
                _clear()
            _draw(
                state=state,
                clip=clip,
                label=label,
                decision=shown,
                emotions=choices,
                previous_text=context[0],
                next_text=context[1],
            )
            key = key_reader()
            print(key)
            lowered = key.lower()
            if key == "0":
                play_audio(clip.audio_path, play_command)
            elif key.isdigit() and 1 <= int(key) <= len(choices):
                _remember(state, current)
                shown.emotion = choices[int(key) - 1]
                shown.excluded = False
                shown.confirmed = True
                if not shown.cluster or shown.cluster == "unknown":
                    shown.cluster = shown.emotion
                state.decisions[clip_id] = shown
                state.cursor += 1
                workspace.save_review(state)
            elif lowered == "x":
                _remember(state, current)
                shown.excluded = True
                shown.confirmed = True
                state.decisions[clip_id] = shown
                state.cursor += 1
                workspace.save_review(state)
            elif lowered == "e":
                _remember(state, current)
                shown.transcript = input("Text: ").strip()
                state.decisions[clip_id] = shown
                workspace.save_review(state)
            elif lowered == "k":
                _remember(state, current)
                shown.cluster = input("Cluster: ").strip() or "unknown"
                state.decisions[clip_id] = shown
                workspace.save_review(state)
            elif lowered == "s":
                skipped = state.order.pop(state.cursor)
                state.order.append(skipped)
                workspace.save_review(state)
            elif lowered == "m":
                if state.cursor + 1 >= len(state.order):
                    continue
                right_id = state.order[state.cursor + 1]
                right_clip = clip_by_id[right_id]
                right_label = label_by_id.get(right_id)
                right_decision = state.decisions.get(right_id)
                right_shown = _fallback_decision(right_clip, right_label, right_decision)
                merged, merged_label = merge_adjacent_clips(
                    workspace,
                    clip,
                    right_clip,
                    transcript=_joined_text(shown.transcript, right_shown.transcript),
                    emotion=shown.emotion,
                    cluster=shown.cluster,
                )
                clip_by_id.pop(clip_id, None)
                clip_by_id.pop(right_id, None)
                clip_by_id[merged.clip_id] = merged
                label_by_id.pop(clip_id, None)
                label_by_id.pop(right_id, None)
                label_by_id[merged.clip_id] = merged_label
                state = workspace.load_review()
            elif lowered == "b":
                _undo(state)
                workspace.save_review(state)
            elif lowered == "q":
                workspace.save_review(state)
                break
            elif lowered == "r":
                continue
    except (KeyboardInterrupt, EOFError):
        workspace.save_review(state)
    return state
