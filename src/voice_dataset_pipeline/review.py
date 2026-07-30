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

from .models import ClipRecord, LabelRecord, ReviewDecision, ReviewState
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


def _draw(
    *,
    state: ReviewState,
    clip: ClipRecord,
    label: LabelRecord | None,
    decision: ReviewDecision,
    emotions: Sequence[str],
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
    print("-" * width)
    options = "  ".join(f"[{index}] {emotion}" for index, emotion in enumerate(emotions, 1))
    print(options)
    print("[0] Play  [X] Exclude  [E] Edit text  [K] Edit cluster  [R] Refresh")
    print("[S] Skip to end  [B] Undo last  [Q] Save and quit")
    if label and label.rationale:
        print(f"Gemini: {label.rationale}")
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
            if clear_screen:
                _clear()
            _draw(
                state=state,
                clip=clip,
                label=label,
                decision=shown,
                emotions=choices,
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
