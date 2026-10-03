"""A ceiling on what one MCP tool call may put into the transcript.

`run_command` caps its own output; MCP results do not. base-mcp's default
`FS_MAX_FILE_BYTES` (200,000 bytes, about 40,500 tokens) exceeds a 32,768-token
context window, and a result stays in the transcript and is re-sent with every
later request. An oversized read of source comes back as a line-numbered outline
of the file's definitions, from which the model can request a range.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.toolsets import WrapperToolset

# 8 KiB is about 1,650 tokens, so 12 capped results fit a 32k context window.
# Smaller than run_command's 16 KiB because file reads are more frequent.
DEFAULT_MAX_MCP_OUTPUT = 8 * 1024


def max_mcp_output() -> int:
    """Read at call time, so a late `--env-file` is seen. 0 disables capping."""
    raw = os.environ.get("DJED_MAX_MCP_OUTPUT")
    return DEFAULT_MAX_MCP_OUTPUT if raw is None or not raw.strip() else int(raw)

# Tools whose result is one file's contents, and so can be outlined.
# `fs_read_many` takes `paths`, not `path`, and is head/tail clipped instead.
READ_TOOLS = frozenset({"fs_read"})

# Below this many entries, head-clip instead of outlining.
MIN_OUTLINE_ENTRIES = 3
MAX_OUTLINE_ENTRIES = 200

# Definition patterns per language. Regex, not a parser: it only has to point
# at a line number.
SOURCE_PATTERNS: dict[str, str] = {
    ".py": r"^\s*(?:async\s+)?(?:def|class)\s+\w+|^[A-Z_][A-Z0-9_]{2,}\s*=",
    ".pyi": r"^\s*(?:async\s+)?(?:def|class)\s+\w+",
    ".js": r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function|class)\s+\w+"
           r"|^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=",
    ".mjs": r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+",
    ".ts": r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:async\s+)?"
           r"(?:function|class|interface|type|enum)\s+\w+"
           r"|^\s*(?:export\s+)?(?:const|let|var)\s+\w+\s*=",
    ".go": r"^\s*(?:func|type|const|var)\s+\w+",
    ".rs": r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
           r"(?:fn|struct|enum|impl|trait|mod|const|static)\s+\w+",
    ".java": r"^\s*(?:public|private|protected|static|final|\s)*"
             r"(?:class|interface|enum|record)\s+\w+",
    ".rb": r"^\s*(?:def|class|module)\s+\w+",
    ".sh": r"^\s*(?:function\s+)?[\w.-]+\s*\(\)\s*\{|^\s*[A-Z_][A-Z0-9_]*=",
    ".bash": r"^\s*(?:function\s+)?[\w.-]+\s*\(\)\s*\{",
    ".md": r"^#{1,6}\s+\S",
    ".yaml": r"^[A-Za-z_][\w.-]*:",
    ".yml": r"^[A-Za-z_][\w.-]*:",
    ".sql": r"^\s*(?:CREATE|ALTER|DROP|INSERT|UPDATE)\s+",
    ".tf": r"^\s*(?:resource|module|variable|output|provider|data)\s+",
}
SOURCE_PATTERNS[".jsx"] = SOURCE_PATTERNS[".js"]
SOURCE_PATTERNS[".tsx"] = SOURCE_PATTERNS[".ts"]

# How to read the rest, per tool.
CONTINUATIONS: dict[str, str] = {
    "fs_read": "Read what you need with fs_read(path, start_line=..., end_line=...).",
    "fs_read_many": "Read one file at a time with fs_read(path, start_line=...).",
    "fs_grep": "Narrow the pattern, or pass include= to scope the search.",
    "fs_list": "Pass depth=1, or list one subdirectory at a time.",
    "fs_glob": "Use a narrower pattern.",
    # Command output cannot be re-read in slices; name where it was kept.
    "run_task": "Read the full output with task_output(run_id=..., start_line=...).",
    "task_output": "Read a narrower range, or search it with pattern=.",
}
DEFAULT_CONTINUATION = "Request a narrower slice."


@dataclass(frozen=True)
class Outline:
    lines: tuple[tuple[int, str], ...]
    total_lines: int
    truncated: bool


def outline(text: str, suffix: str) -> Outline | None:
    """A line-numbered map of a file's definitions, or None if not outlinable."""
    pattern = SOURCE_PATTERNS.get(suffix.lower())
    if not pattern:
        return None
    rx = re.compile(pattern)
    hits: list[tuple[int, str]] = []
    all_lines = text.splitlines()
    for n, line in enumerate(all_lines, 1):
        if rx.match(line):
            hits.append((n, line.rstrip()[:120]))
            if len(hits) >= MAX_OUTLINE_ENTRIES:
                return Outline(tuple(hits), len(all_lines), True)
    if len(hits) < MIN_OUTLINE_ENTRIES:
        return None
    return Outline(tuple(hits), len(all_lines), False)


# A tier that could not return everything ends with a continuation line
# starting with "..." (base-mcp's names a `start_byte`). It is kept whatever
# happens to the body.
TRAILER_PREFIX = "..."


def _carried_trailer(text: str) -> str:
    """The source's own continuation line, if the result ends with one."""
    stripped = text.rstrip()
    if not stripped:
        return ""
    last = stripped.rsplit("\n", 1)[-1]
    return last.strip() if last.lstrip().startswith(TRAILER_PREFIX) else ""


def _split_ends(text: str, budget: int) -> tuple[str, str]:
    """Keep both ends on whole-line boundaries and name the gap, as exec-mcp's `_RingCapture` does.

    Command output's first error and its final verdict are at opposite ends.
    """
    half = max(budget // 2, 1)
    head = text[:half]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    tail = text[len(text) - half:]
    cut = tail.find("\n")
    if cut >= 0:
        tail = tail[cut + 1:]
    # Overlapping ends mean the whole text fits, so no gap marker.
    if len(head) + len(tail) >= len(text):
        return text, ""
    return head, tail


def _as_int(value: Any) -> int:
    """Coerce a tool argument to int. A bad value must not break the cap."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _requested_range(args: dict[str, Any]) -> str:
    """How the caller described what it wanted, for the footer."""
    if _as_int(args.get("start_byte")):
        return f"bytes from {_as_int(args['start_byte'])}"
    start, end = _as_int(args.get("start_line")), _as_int(args.get("end_line"))
    if start or end:
        return f"lines {start or 1}-{end or 'end'}"
    return ""


def _render_outline(
    mapped: Outline,
    path: str,
    nbytes: int,
    how: str,
    carried: str,
    limit: int,
    offset: int,
) -> str | None:
    """A structure map inside the budget, or None if even a short one will not fit.

    Entries are dropped from the end; a file's head holds its imports and entry
    points.
    """
    entries = list(mapped.lines)
    while entries:
        clipped = mapped.truncated or len(entries) < len(mapped.lines)
        more = " (outline truncated)" if clipped else ""
        shown = "\n".join(f"{n + offset:>6}: {line}" for n, line in entries)
        if carried:
            # The tier did not reach the end of the file, so these counts describe what
            # was read, not the whole file.
            header = (
                f"[the first {nbytes} bytes of {path} ({mapped.total_lines} "
                f"lines) - the file is longer. Structure of what was read{more}:]"
            )
        else:
            header = (
                f"[{path} is {mapped.total_lines} lines, {nbytes} bytes - too "
                f"large to return whole. Its structure{more}:]"
            )
        parts = [header, "", shown, ""]
        if carried:
            parts += [f"[{carried}]", ""]
        parts.append(f"[{how}]")
        out = "\n".join(parts)
        if len(out) <= limit:
            return out
        entries = entries[: len(entries) * 3 // 4]
    return None


def _render_clip(
    text: str, limit: int, how: str, carried: str, requested: str
) -> str:
    """Both ends of the body, the gap named, the trailer and the hint kept."""
    total = len(text.splitlines())
    # Reserve room for the footer, gap marker and carried trailer, so the whole
    # result fits the budget.
    reserve = 200 + len(how) + len(carried) + len(requested)
    head, tail = _split_ends(text, max(limit - reserve, 200))
    kept_bytes = len(head) + len(tail)
    # Line counting matches the outline path and `wc -l`: a trailing newline does
    # not start a new line. Both ends can fall inside one long line, so the count
    # is clamped.
    kept = min(len(head.splitlines()) + len(tail.splitlines()), total)
    body = head if not tail else (
        f"{head}\n\n[... {len(text) - kept_bytes} bytes omitted ...]\n\n{tail}"
    )
    note = f"showed {kept} of {total} lines ({kept_bytes} of {len(text)} bytes)"
    if requested:
        note += f"; you asked for {requested}"
    parts = [body, ""]
    # Skip the carried trailer if the tail clip already includes it.
    if carried and not body.rstrip().endswith(carried):
        parts += [f"[{carried}]", ""]
    # A carried trailer already says how to continue, so no generic hint follows it.
    tail_hint = "" if carried and how is DEFAULT_CONTINUATION else f" {how}"
    parts.append(f"[... truncated: {note}.{tail_hint}]")
    capped = "\n".join(parts)
    # The footer is about 130 bytes; cap only when that makes the result smaller.
    return capped if len(capped) < len(text) else text


def cap_text(text: str, tool_name: str, tool_args: dict[str, Any] | None) -> str:
    """Cap one tool result, preferring an outline over a clipped body.

    The tier's continuation line, if any, and a hint for reading the rest are always
    kept.
    """
    limit = max_mcp_output()
    if limit <= 0 or len(text) <= limit:
        return text

    bare = tool_name.rsplit("__", 1)[-1]
    args = tool_args or {}
    path = str(args.get("path") or "")
    how = CONTINUATIONS.get(bare, DEFAULT_CONTINUATION)
    carried = _carried_trailer(text)
    requested = _requested_range(args)

    # A range request gets its range, not an outline: an outline of a slice
    # would be numbered from the slice.
    if bare in READ_TOOLS and path and not requested:
        mapped = outline(text, Path(path).suffix)
        if mapped is not None:
            # Zero here, since a range never reaches this path; applied so the numbers
            # stay absolute.
            offset = max(_as_int(args.get("start_line")) - 1, 0)
            rendered = _render_outline(
                mapped, path, len(text), how, carried, limit, offset
            )
            if rendered is not None and len(rendered) < len(text):
                return rendered

    return _render_clip(text, limit, how, carried, requested)


# --- Structured results ----------------------------------------------------
#
# pydantic-ai returns a tool's structured content instead of its text, and
# `cap_result` passes non-text through. A tier marks a bounded result with an
# envelope, which is rendered to text here. An unmarked dict passes through
# whole.
ENVELOPE_KEY = "base_mcp_envelope"
ENVELOPE_VERSION = 1


@dataclass(frozen=True)
class Envelope:
    """A tier's own account of a result it could not return whole."""

    content: str
    complete: bool
    total_bytes: int
    total_lines: int
    continue_with: str
    label: str
    # Definitions with their extents, from the tier's own parse.
    outline: tuple[dict[str, Any], ...] = ()

    @classmethod
    def parse(cls, result: Any) -> "Envelope | None":
        """An envelope, or None for anything else - rows included."""
        if not isinstance(result, dict):
            return None
        if result.get(ENVELOPE_KEY) != ENVELOPE_VERSION:
            return None
        content = result.get("content")
        if not isinstance(content, str):
            return None
        return cls(
            content=content,
            complete=bool(result.get("complete", True)),
            # The tier's numbers describe the whole file; the body's describe
            # what fitted.
            total_bytes=_as_int(result.get("total_bytes")) or len(content),
            # 0 means unknown: a tier may leave out the line count, and the footer then
            # reports bytes.
            total_lines=_as_int(result.get("total_lines")) or (
                len(content.splitlines()) if result.get("complete", True) else 0
            ),
            continue_with=str(result.get("continue_with") or ""),
            label=str(result.get("label") or ""),
            outline=tuple(
                e for e in (result.get("outline") or []) if isinstance(e, dict)
            ),
        )


def _render_tier_outline(
    env: Envelope, carried: str, limit: int, asked: str = ""
) -> str | None:
    """The tier's parsed structure, inside the budget, or None if it will not fit.

    Entries are dropped from the end: a file's head is its imports and entry
    points, which locate the rest.
    """
    entries = list(env.outline)
    lines = f"{env.total_lines} lines, " if env.total_lines else ""
    while entries:
        cut = " (outline truncated)" if len(entries) < len(env.outline) else ""
        rows = []
        for e in entries:
            start = _as_int(e.get("start_line"))
            end = _as_int(e.get("end_line")) or start
            span = f"{start}-{end}" if end != start else f"{start}"
            indent = "  " * min(_as_int(e.get("depth")), 6)
            rows.append(
                f"{span:>13}  {indent}{e.get('kind', '')} {e.get('name', '')}".rstrip()
            )
        if asked:
            # The caller asked for the map; nothing was too large.
            header = f"[{env.label}, structure of {asked}{cut}, as line ranges:]"
        else:
            header = (
                f"[{env.label} is {lines}{env.total_bytes} bytes - too large to "
                f"return whole. Its structure{cut}, as line ranges:]"
            )
        out = header + "\n\n" + "\n".join(rows) + f"\n\n[{carried}]"
        if len(out) <= limit:
            return out
        entries = entries[: len(entries) * 3 // 4]
    return None


def _render_envelope(
    env: Envelope,
    how: str,
    limit: int,
    requested: str = "",
    wants_outline: bool = False,
    requested_label: str = "",
) -> str:
    """Text for the model, counted from the tier's totals rather than the body."""
    carried = env.continue_with or how

    # An explicit request for structure gets structure, even with an empty body.
    if wants_outline and env.outline:
        asked = requested_label or f"all {env.total_lines} lines" if env.total_lines else "the file"
        rendered = _render_tier_outline(env, carried, limit, asked)
        if rendered is not None:
            return rendered

    if env.complete and len(env.content) <= limit:
        return env.content

    # Outline instead of bytes when the bytes will not fit, but only for an
    # unqualified read: a range request gets its range, as in `cap_text`.
    if env.outline and not requested:
        rendered = _render_tier_outline(env, carried, limit)
        if rendered is not None:
            return rendered

    reserve = 200 + len(carried) + len(env.label)
    head, tail = _split_ends(env.content, max(limit - reserve, 200))
    kept_bytes = len(head) + len(tail)
    body = head if not tail else (
        f"{head}\n\n[... {len(env.content) - kept_bytes} bytes omitted ...]\n\n{tail}"
    )
    counts = f"{kept_bytes} of {env.total_bytes} bytes"
    if env.total_lines:
        kept_lines = min(
            len(head.splitlines()) + len(tail.splitlines()), env.total_lines
        )
        counts = f"{kept_lines} of {env.total_lines} lines ({counts})"
    if requested:
        # Name what was asked for, as the bare-text path does.
        counts += f"; you asked for {requested}"
    what = f"{env.label}: " if env.label else ""
    return f"{body}\n\n[... {what}showed {counts}. {carried}]"


def _cap_structure(result: Any, how: str, limit: int) -> str | None:
    """Bound an unmarked structured result, or None to leave it as it is.

    JSON is never clipped. Wide row-shaped results are rendered as a bounded number
    of whole rows as text, with the total count.
    """
    try:
        blob = json.dumps(result, default=str)
    except (TypeError, ValueError):
        return None  # not serialisable - not ours to reshape
    if len(blob) <= limit:
        return None

    items: list[Any] | None = None
    noun = "items"
    if isinstance(result, list):
        items = result
    elif isinstance(result, dict):
        # The widest list-valued field is the row payload; the rest is metadata.
        lists = [(k, v) for k, v in result.items() if isinstance(v, list)]
        if lists:
            noun, items = max(lists, key=lambda kv: len(kv[1]))

    if items is None:
        keys = ", ".join(sorted(str(k) for k in result)) if isinstance(result, dict) else ""
        detail = f" Top-level keys: {keys}." if keys else ""
        return (
            f"[a structured result of {len(blob)} bytes - too large to return "
            f"whole.{detail} {how}]"
        )

    kept: list[str] = []
    size = 0
    for item in items:
        row = json.dumps(item, default=str)
        if size + len(row) + 1 > limit - 220:
            break
        kept.append(row)
        size += len(row) + 1
    if not kept:
        # One row is wider than the whole budget; say so.
        return (
            f"[{len(items)} {noun}, {len(blob)} bytes - the first one alone is "
            f"over the budget, so none is shown whole. {how}]"
        )
    body = "\n".join(kept)
    return (
        f"{body}\n\n[... showed {len(kept)} of {len(items)} {noun} "
        f"({size} of {len(blob)} bytes). {how}]"
    )


def cap_result(result: Any, tool_name: str, tool_args: dict[str, Any] | None) -> Any:
    """Cap a tool result, leaving anything that is not text or an envelope alone.

    Unmarked structure is not truncated, since clipped JSON is a parse error. A
    marked envelope is rendered to text.
    """
    env = Envelope.parse(result)
    if env is not None:
        limit = max_mcp_output()
        bare = tool_name.rsplit("__", 1)[-1]
        how = CONTINUATIONS.get(bare, DEFAULT_CONTINUATION)
        if limit <= 0:
            return env.content
        args = tool_args or {}
        # An explicit `outline=true` is a request for structure, so the range rule
        # does not apply.
        wants_outline = bool(args.get("outline"))
        return _render_envelope(
            env,
            how,
            limit,
            "" if wants_outline else _requested_range(args),
            wants_outline,
            _requested_range(args),
        )
    if isinstance(result, str):
        return cap_text(result, tool_name, tool_args)
    if isinstance(result, list) and all(
        isinstance(item, (str, int, float, bool, type(None))) for item in result
    ):
        capped = [
            cap_text(item, tool_name, tool_args) if isinstance(item, str) else item
            for item in result
        ]
        # A list of content blocks is transport; a list of rows is payload, which can
        # exceed the budget.
        if sum(len(str(x)) for x in capped) <= max_mcp_output() or max_mcp_output() <= 0:
            return capped
        result = capped
    limit = max_mcp_output()
    if limit > 0 and isinstance(result, (dict, list)):
        bare = tool_name.rsplit("__", 1)[-1]
        rendered = _cap_structure(
            result, CONTINUATIONS.get(bare, DEFAULT_CONTINUATION), limit
        )
        if rendered is not None:
            return rendered
    return result


@dataclass
class CappedToolset(WrapperToolset):
    """Bound what one call from this toolset may add to the transcript."""

    async def call_tool(self, name, tool_args, ctx, tool):
        result = await super().call_tool(name, tool_args, ctx, tool)
        return cap_result(result, name, tool_args)
