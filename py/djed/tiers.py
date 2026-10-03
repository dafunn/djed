"""Tool-loading strategy: which MCP tiers the model sees, and how.

Definitions are re-sent every request, so a large fleet costs context; see
`djed/tokens.py`. `EagerTiers` sends the selected tiers in full; `SearchedTiers`
hides them behind tool search for one extra round trip. Both reduce to
`(toolsets, capabilities)`, so swapping is a flag.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai import AgentCapability, ApprovalRequiredToolset, DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.toolsets import AbstractToolset

from .cap import CappedToolset


@dataclass(frozen=True)
class Tier:
    """One MCP server and the tools it exposes."""

    name: str
    toolset: AbstractToolset
    tool_names: tuple[str, ...]
    # Estimated input tokens this tier's definitions add to every request.
    # See djed/tokens.py for what "estimated" means here.
    token_cost: int = 0

    @property
    def tool_count(self) -> int:
        return len(self.tool_names)


class ToolStrategy(Protocol):
    """Turns the loaded tiers into what the Agent is actually constructed with."""

    label: str
    instructions: str

    def apply(
        self, tiers: list[Tier]
    ) -> tuple[list[AbstractToolset], list[AgentCapability]]: ...


# MCP tools that change the filesystem. An MCP tool cannot gate itself, so these
# go through the approval gate. exec-mcp's tools are not listed; its own log is
# their audit trail.
WRITE_TOOLS = frozenset(
    {
        "fs_write",
        "fs_edit",
        "fs_write_base64",
        "fs_copy",
        "fs_mkdir",
        "fs_move",
        "fs_delete",
    }
)


def gate_writes_default() -> bool:
    """Gate write tools unless `DJED_GATE_WRITES` turns it off. On by default."""
    raw = (os.environ.get("DJED_GATE_WRITES") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _is_write_tool(tool_def) -> bool:
    # A name collision renames `fs_write` to `<server>__fs_write`, so match the
    # bare name rather than the exposed one.
    return getattr(tool_def, "name", "").rsplit("__", 1)[-1] in WRITE_TOOLS


def toolset_id(toolset: AbstractToolset) -> str | None:
    """The tier name behind any number of wrappers. `WrapperToolset.id` is `None`
    upstream, so the name is read from the wrapped toolset.
    """
    seen: set[int] = set()
    while getattr(toolset, "id", None) is None and hasattr(toolset, "wrapped"):
        if id(toolset) in seen:  # pragma: no cover - a cycle should be impossible
            return None
        seen.add(id(toolset))
        toolset = toolset.wrapped
    return getattr(toolset, "id", None)


def unwrap_to(toolset: AbstractToolset, cls: type) -> AbstractToolset | None:
    """Find the wrapper of type `cls` in a toolset stack, or None. A tier is wrapped
    more than once: capped, gated, and deferred under `search`.
    """
    seen: set[int] = set()
    while toolset is not None and id(toolset) not in seen:
        if isinstance(toolset, cls):
            return toolset
        seen.add(id(toolset))
        toolset = getattr(toolset, "wrapped", None)
    return None


def capped(toolset: AbstractToolset) -> AbstractToolset:
    """Bound what one call from this toolset may add to the transcript.

    Outermost of the wrappers, so it sees the result the tier finally returns.
    Applied to every tier, not just `base`: a wide `pg_query` has the same shape
    as a large file read.
    """
    return CappedToolset(toolset)


def gated(toolset: AbstractToolset) -> AbstractToolset:
    """Wrap a toolset so its write tools raise `ApprovalRequired`."""

    def needs_approval(ctx, tool_def, tool_args) -> bool:
        return _is_write_tool(tool_def)

    return ApprovalRequiredToolset(toolset, needs_approval)


# A local model will not call search_tools unless told the deferred tools
# exist; it reaches for `run_command` instead.
SEARCH_INSTRUCTIONS = """
Most of your tools are hidden from the list above. You have tools for the
cluster, for databases, and for logs, but they stay hidden until you look them
up. Before you answer that you lack a capability, and before you fall back to
run_command, call search_tools with a short description of what you need
(e.g. "list running pods", "query the database"), then call the tool it
returns."""


@dataclass
class EagerTiers:
    """Send the enabled tiers' definitions on every request.

    `enabled=None` means all tiers.
    """

    enabled: frozenset[str] | None = None
    label: str = "eager"
    instructions: str = ""
    gate_writes: bool = field(default_factory=gate_writes_default)

    def apply(
        self, tiers: list[Tier]
    ) -> tuple[list[AbstractToolset], list[AgentCapability]]:
        chosen = [t for t in tiers if self.enabled is None or t.name in self.enabled]
        gate = gated if self.gate_writes else (lambda ts: ts)
        return [capped(gate(t.toolset)) for t in chosen], []


@dataclass
class SearchedTiers:
    """Hide the enabled tiers behind tool search until the model asks.

    Without provider-side search, Pydantic AI falls back to a local
    `search_tools` function: one extra round trip, in exchange for keeping the
    definitions out of every request.
    """

    enabled: frozenset[str] | None = None
    max_results: int = 10
    label: str = "search"
    instructions: str = SEARCH_INSTRUCTIONS
    gate_writes: bool = field(default_factory=gate_writes_default)

    def apply(
        self, tiers: list[Tier]
    ) -> tuple[list[AbstractToolset], list[AgentCapability]]:
        chosen = [t for t in tiers if self.enabled is None or t.name in self.enabled]
        # Gate innermost: deferral controls whether a definition is advertised,
        # approval controls whether a call runs. Wrapping the other way round
        # would let a discovered tool execute ungated.
        gate = gated if self.gate_writes else (lambda ts: ts)
        toolsets = [DeferredLoadingToolset(capped(gate(t.toolset))) for t in chosen]
        return toolsets, [ToolSearch(max_results=self.max_results)]


def resolve_selection(
    spec: str | None,
    available: list[str],
    current: set[str] | None = None,
) -> tuple[set[str], str | None]:
    """Interpret a tier selection spec. Returns `(selection, error)`.

    Accepts `all`, `none`, a comma list (`cluster,db`) which sets the selection
    outright, or `+name` / `-name` deltas applied to `current`. Deltas and
    absolute names cannot be mixed - that is almost always a typo.
    """
    avail = set(available)
    current = set(current if current is not None else avail)

    if spec is None or not spec.strip():
        return set(avail), None

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return set(avail), None
    if len(parts) == 1 and parts[0] == "all":
        return set(avail), None
    if len(parts) == 1 and parts[0] == "none":
        return set(), None

    deltas = [p for p in parts if p[0] in "+-"]
    if deltas and len(deltas) != len(parts):
        return current, "mix of +/- deltas and absolute names; use one or the other"

    unknown = sorted({p.lstrip("+-") for p in parts} - avail)
    if unknown:
        have = ", ".join(sorted(avail))
        return current, f"unknown tier(s): {', '.join(unknown)} (have: {have})"

    if deltas:
        selection = set(current)
        for p in parts:
            if p[0] == "+":
                selection.add(p[1:])
            else:
                selection.discard(p[1:])
        return selection, None
    return set(parts), None


def build_strategy(
    kind: str | None = None,
    tiers_spec: str | None = None,
    available: list[str] | None = None,
) -> tuple[ToolStrategy, str | None]:
    """Build the configured strategy. Returns `(strategy, error)`.

    Falls back to the environment (`DJED_TOOL_STRATEGY`, `DJED_MCP_TIERS`) so the
    default can be set once in `~/.djed.env` rather than passed every run.
    """
    kind = kind or os.environ.get("DJED_TOOL_STRATEGY") or "eager"
    err: str | None = None
    if kind not in ("eager", "search"):
        kind, err = "eager", f"unknown tool strategy {kind!r}; using eager"

    enabled: frozenset[str] | None = None
    spec = tiers_spec if tiers_spec is not None else os.environ.get("DJED_MCP_TIERS")
    if spec and available is not None:
        selection, sel_err = resolve_selection(spec, available)
        enabled = frozenset(selection)
        err = err or sel_err

    if kind == "search":
        return SearchedTiers(enabled=enabled), err
    return EagerTiers(enabled=enabled), err
