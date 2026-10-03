"""Input history that persists across sessions.

Stored as one JSON string per line, since entries can be multi-line.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("djed.history")

DEFAULT_LIMIT = 1000


def default_path() -> Path:
    """Where history lives: the override (set by the container launcher), else XDG."""
    explicit = os.environ.get("DJED_HISTORY_FILE", "").strip()
    if explicit:
        return Path(explicit)
    state = os.environ.get("XDG_STATE_HOME", "").strip() or "~/.local/state"
    return Path(state).expanduser() / "djed" / "history"


class History:
    """A capped, de-duplicated command history backed by a file. Every operation is
    best-effort: failures are logged, not raised.
    """

    def __init__(self, path: Path | None = None, limit: int = DEFAULT_LIMIT) -> None:
        self.path = Path(path) if path is not None else default_path()
        self.limit = limit

    def load(self) -> list[str]:
        """Previous entries, oldest first. Missing or damaged lines are skipped."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except OSError as e:
            log.warning("could not read history at %s: %s", self.path, e)
            return []

        entries: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A truncated or hand-edited line costs only that line.
                continue
            if isinstance(value, str) and value:
                entries.append(value)
        return entries[-self.limit :]

    def append(self, line: str) -> None:
        """Record one entry, unless it repeats the previous one."""
        line = line.strip()
        if not line:
            return
        existing = self.load()
        if existing and existing[-1] == line:
            return
        entries = (existing + [line])[-self.limit :]
        self._write(entries)

    def _write(self, entries: list[str]) -> None:
        """Replace the file atomically, readable by its owner only."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
            body = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
            tmp.write_text(body, encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("could not write history to %s: %s", self.path, e)
            try:
                tmp.unlink(missing_ok=True)
            except (OSError, UnboundLocalError):
                pass
