"""Terminal environment checks, surfaced at startup. tmux only: its defaults
break Esc-to-interrupt and clipboard copy, and neither symptom is obvious.
"""
from __future__ import annotations

import os
import subprocess

# tmux waits this long after Esc to decide whether it starts an escape
# sequence. The 500ms default delays Esc-to-interrupt.
ESCAPE_TIME_CEILING_MS = 50


def _tmux_get(option: str) -> str | None:
    try:
        out = subprocess.run(
            ["tmux", "show", "-gv", option],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def tmux_notes(env: dict | None = None, get=_tmux_get) -> list[str]:
    """Operator-facing warnings about the current tmux config.

    Empty when not running under tmux, or when tmux is already configured well.
    """
    env = os.environ if env is None else env
    if not env.get("TMUX"):
        return []

    notes: list[str] = []

    escape_time = get("escape-time")
    if escape_time is not None:
        try:
            ms = int(escape_time)
        except ValueError:
            ms = None
        if ms is not None and ms > ESCAPE_TIME_CEILING_MS:
            notes.append(
                f"tmux escape-time is {ms}ms, so Esc-to-interrupt will lag. "
                f"Fix: tmux set -sg escape-time 10"
            )

    clipboard = get("set-clipboard")
    if clipboard is not None and clipboard != "on":
        notes.append(
            f"tmux set-clipboard is '{clipboard}', so ctrl+c copy will not reach "
            f"the system clipboard. Fix: tmux set -g set-clipboard on "
            f"(or use /save, which never needs the clipboard)"
        )

    return notes


def clipboard_blocker(env: dict | None = None, get=_tmux_get) -> str | None:
    """Name the likely blocker for an OSC 52 copy, or None if none is known."""
    env = os.environ if env is None else env
    if env.get("TMUX"):
        clipboard = get("set-clipboard")
        if clipboard is not None and clipboard != "on":
            return (
                f"tmux set-clipboard is '{clipboard}', which drops it. "
                f"Fix: tmux set -g set-clipboard on, or use /save"
            )
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return (
            "over SSH, so it depends on your local terminal allowing OSC 52; "
            "/save always works"
        )
    return None
