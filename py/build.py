#!/usr/bin/env python
"""Headless build driver: a spec file in, edits and an outcome out.

    ./.venv/bin/python build.py task.md --workspace /workspace
    ./.venv/bin/python build.py task.md --fake --no-mcp     # offline smoke test

The agent reaches the workspace through base-mcp's file tools, and a project's
build and test commands through exec-mcp's `run_task`. There is no shell in this
process.

A run ends when the model calls `task_complete` or `task_blocked`, when a budget
limit trips, or after two consecutive inconclusive `run_task` results (a timeout
or a kill). Every ending writes HANDOFF.md and sets an exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic_ai import UsageLimits
from djed.agent import build_agent, build_kwim_client
from djed.exec_prefixes import TASK_DONE_PREFIX, TASK_KILLED_PREFIX, TASK_TIMEOUT_PREFIX
from djed.mcp import load_tiers
from djed.persona import BUILDER, default_persona
from djed.spec import SpecError, parse_spec
from djed.tiers import WRITE_TOOLS, build_strategy
from djed.session import Session, TextChunk, ToolFinished, ToolStarted, TurnEvent
from djed.handoff import HANDOFF, render_handoff, resume_preamble
from djed.tools import Deps, Outcome, kwim_recall, task_blocked, task_complete
from djed.trace import format_trace, totals
from headless import load_env_file

DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
OFF = "\x1b[0m"

# Exit codes are distinct per ending: 2 and 3 are resumable.
EXIT = {"complete": 0, "error": 1, "blocked": 2, "exhausted": 3, "review": 4}

# The tiers a build agent uses, named rather than left to `resolve_selection`'s
# default. The cluster and database tiers cost about 2.7k input tokens per
# request.
BUILD_TIERS = frozenset({"base", "exec"})


def default_tiers_spec(connected: set[str]) -> str:
    """`base` always, `exec` only when it connected.

    `base` is always named, so a disconnected base-mcp fails with
    `resolve_selection`'s "unknown tier" error. A missing `exec` runs without it;
    `--no-exec` removes "exec" from `connected` before this is called.
    """
    return ",".join(t for t in sorted(BUILD_TIERS) if t == "base" or t in connected)


def withhold_exec(tiers: list, no_exec: bool) -> list:
    """`--no-exec` removes `exec` entirely: `--tiers exec` with `--no-exec` fails as an
    unknown tier.
    """
    if not no_exec:
        return tiers
    return [t for t in tiers if t.name != "exec"]


# Sent when a turn is cut off at its request ceiling. It does not mention the
# ceiling, so the model keeps running its checks.
RESUME = (
    "Your previous turn was interrupted before you finished.\n"
    "Review what you have already done, then continue the task."
)


def _c(color: str, s: str) -> str:
    return s if os.environ.get("NO_COLOR") else f"{color}{s}{OFF}"


@dataclass
class RunState:
    """What the driver accumulates across turns, and reports at the end."""

    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Defaulted from the clock: `time.monotonic()` counts from an arbitrary origin,
    # so a zero start would exhaust the wall-clock budget at once.
    started: float = field(default_factory=time.monotonic)
    requests: int = 0
    trace: list = field(default_factory=list)
    last_text: str = ""
    error: str | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _parse_task_header(text: str) -> tuple[str, str] | None:
    """Parse `run_task`'s header line, `"{PREFIX} {name}: {short}"` (see
    `djed/exec_prefixes.py`). Returns `(name, short)`, or `None` if the text is not
    a `run_task` result.
    """
    first = text.splitlines()[0] if text else ""
    for prefix in (TASK_DONE_PREFIX, TASK_TIMEOUT_PREFIX, TASK_KILLED_PREFIX):
        if first.startswith(prefix):
            name, _, short = first[len(prefix):].strip().partition(":")
            return name.strip(), short.strip()
    return None


class _OutcomeSink:
    """Turns exec-mcp's result text into a driver decision.

    `run_task` runs in exec-mcp and returns only text, so this closes over `deps`
    and tracks whether the previous `run_task` result was inconclusive.

    Two consecutive inconclusive results (a timeout or a kill, in either order)
    force `task_blocked`. Any completed command, exit 0 or a failure, resets the
    count.
    """

    def __init__(self, deps: Deps):
        self._deps = deps
        self._pending: tuple[str, str] | None = None  # (name, short) of the last inconclusive result

    def __call__(self, event: TurnEvent) -> None:
        if isinstance(event, TextChunk):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolStarted):
            print(f"\n{_c(CYAN, f'[tool] {event.name}')} {_c(DIM, event.args)}", flush=True)
        elif isinstance(event, ToolFinished):
            style = YELLOW if event.is_error else DIM
            print(_c(style, f"  {event.text}"), flush=True)
            self._observe(event.full)

    def _observe(self, text: str) -> None:
        if self._deps.outcome is not None:
            return  # the run already ended this turn; nothing left to force
        header = _parse_task_header(text)
        if header is None:
            return
        name, short = header
        if text.startswith(TASK_DONE_PREFIX):
            self._pending = None
            return
        reason = "command_timeout" if text.startswith(TASK_TIMEOUT_PREFIX) else "command_killed"
        if self._pending is None:
            self._pending = (name, short)
            return
        prev_name, prev_short = self._pending
        self._pending = None
        self._deps.outcome = Outcome(
            kind="blocked",
            summary=(
                "Two consecutive inconclusive run_task results; the harness is not "
                "converging on its own."
            ),
            reason=reason,
            # Names both inconclusive results, not just the reason.
            detail=f"{name} ({short}) after {prev_name} ({prev_short})",
        )


async def _approve(tool_name: str, detail: str) -> bool:
    """The approval gate for an unattended run.

    Only the MCP write tools reach here, and they are approved: base-mcp enforces
    the allowed roots itself, and approving here records an audit line. exec-mcp's
    tools are not in `WRITE_TOOLS`; its own invocation log is their audit trail.

    Anything else is denied.
    """
    bare = tool_name.rsplit("__", 1)[-1]
    if bare in WRITE_TOOLS:
        print(_c(DIM, f"  approved: {tool_name} {detail}"), flush=True)
        return True
    print(_c(RED, f"  denied (not an approved tool): {tool_name} {detail}"), file=sys.stderr)
    return False


def _budget_stop(state: RunState, budget) -> str | None:
    """The budget limit, checked at the start of each turn.

    The only ending the driver imposes; every other ending comes from the model.
    """
    if state.iterations >= budget.max_iterations:
        return f"iteration limit ({budget.max_iterations})"
    if budget.max_tokens and state.total_tokens >= budget.max_tokens:
        return f"token limit ({budget.max_tokens})"
    if state.elapsed >= budget.wall_clock_s:
        return f"wall-clock limit ({budget.wall_clock_s:.0f}s)"
    return None


# An undeclared ending has no summary, so the turn's text stands in. Keep its
# tail, which says where the run got to; `resume_preamble` reads from the top.
SUMMARY_TAIL_CHARS = 1200


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) <= SUMMARY_TAIL_CHARS:
        return text
    clipped = text[-SUMMARY_TAIL_CHARS:]
    # Start on a sentence boundary when one is near the start.
    for sep in ("\n\n", ". ", "\n"):
        i = clipped.find(sep)
        if 0 <= i < 200:
            clipped = clipped[i + len(sep) :]
            break
    return "[...] " + clipped


def write_handoff(path: Path, task_id: str, kind: str, state: RunState, outcome) -> None:
    """Persist the run's account of how it ended. See `djed/handoff.py`."""
    blocked = outcome is not None and outcome.kind == "blocked"
    text = render_handoff(
        task_id=task_id,
        kind=kind,
        iterations=state.iterations,
        requests=state.requests,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        elapsed=state.elapsed,
        summary=(outcome.summary if outcome is not None else _tail(state.last_text)),
        reason=outcome.reason if blocked else "",
        detail=outcome.detail if blocked else "",
        error=state.error or "",
    )
    try:
        path.write_text(text, encoding="utf-8")
    except OSError as e:
        print(_c(RED, f"could not write {path}: {e}"), file=sys.stderr)


def _seed_prompt(workspace_md: str, prose: str, preamble: str = "") -> str:
    """Turn one. On a resumed run, the resume preamble comes first."""
    parts = [
        "Read this workspace description, then carry out the task specification "
        "that follows it.",
        f"## WORKSPACE.md\n\n{workspace_md}",
    ]
    if preamble:
        parts.append(preamble)
    parts.append(f"## Task specification\n\n{prose}")
    return "\n\n".join(parts)


async def main() -> int:
    ap = argparse.ArgumentParser(description="Headless djed build driver")
    ap.add_argument("spec", help="path to the spec file (TOML frontmatter + prose)")
    ap.add_argument(
        "--workspace",
        default=os.environ.get("DJED_WORKSPACE") or "/workspace",
        help="the workspace root, as this process sees it (default /workspace)",
    )
    ap.add_argument("--env-file", default=None, help="load a KEY=VALUE env file literally")
    ap.add_argument("--model", default=None, help="override DJED_MODEL")
    ap.add_argument("--agent", default=None, help="persona tag (default: builder)")
    ap.add_argument(
        "--tiers", default=None,
        help="MCP tiers to load (default: base, plus exec if it connected)",
    )
    ap.add_argument("--tool-strategy", choices=["eager", "search"], default=None)
    ap.add_argument("--no-mcp", action="store_true", help="skip the MCP fleet bridge")
    ap.add_argument("--fake", action="store_true", help="scripted stub model, offline")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="print per-request tokens, finish reason and response shape",
    )
    ap.add_argument(
        "--no-exec",
        action="store_true",
        help="spawn no processes at all; edits only, zero exec risk",
    )
    ap.add_argument(
        "--max-iterations", type=int, default=None, help="override the spec's budget"
    )
    args = ap.parse_args()

    if args.env_file:
        n = load_env_file(args.env_file)
        print(_c(DIM, f"loaded {n} vars from {args.env_file}"), file=sys.stderr)

    try:
        manifest, prose = parse_spec(Path(args.spec).read_text(encoding="utf-8"))
    except SpecError as e:
        print(_c(RED, f"spec: {e}"), file=sys.stderr)
        return EXIT["error"]
    except OSError as e:
        print(_c(RED, f"spec: {e}"), file=sys.stderr)
        return EXIT["error"]

    ws = Path(args.workspace)
    workspace_md_path = ws / "WORKSPACE.md"
    try:
        workspace_md = workspace_md_path.read_text(encoding="utf-8")
    except OSError as e:
        print(_c(RED, f"workspace: {e}"), file=sys.stderr)
        print(
            _c(DIM, "provisioning writes WORKSPACE.md; hand-make one for a step-1 run"),
            file=sys.stderr,
        )
        return EXIT["error"]

    # Set before anything resolves identity: provider.py (spend tags) and kwim.py
    # (agent_id) both read `default_persona()`.
    os.environ["DJED_AGENT"] = args.agent or BUILDER.tag
    persona = default_persona()
    session_id = os.environ.get("DJED_SESSION") or f"build-{manifest.task_id}"

    budget = manifest.budget
    if args.max_iterations:
        budget = type(budget)(
            max_iterations=args.max_iterations,
            max_tokens=budget.max_tokens,
            wall_clock_s=budget.wall_clock_s,
        )

    model = None
    if args.fake:
        from fake_model import scripted_builder

        model = scripted_builder()
    elif args.model:
        from djed.provider import build_model

        model = build_model(args.model)

    state = RunState(started=time.monotonic())
    outcome = None
    kind = "error"

    async with httpx.AsyncClient(timeout=30.0) as http:
        kwim = build_kwim_client(http)

        toolsets: list = []
        capabilities: list = []
        extra_instructions = ""
        if not args.no_mcp and not args.fake:
            tiers, mcp_errors, mcp_warnings = await load_tiers()
            for name, err in mcp_errors:
                print(_c(RED, f"MCP {name}: {err}"), file=sys.stderr)
            for w in mcp_warnings:
                print(_c(DIM, w), file=sys.stderr)
            tiers = withhold_exec(tiers, args.no_exec)
            available = [t.name for t in tiers]
            strategy, sel_err = build_strategy(
                args.tool_strategy, args.tiers or default_tiers_spec(set(available)), available
            )
            if sel_err:
                print(_c(RED, f"tier selection: {sel_err}"), file=sys.stderr)
                return EXIT["error"]
            toolsets, capabilities = strategy.apply(tiers)
            extra_instructions = strategy.instructions
            print(
                _c(DIM, f"MCP: {len(toolsets)}/{len(tiers)} tiers, strategy={strategy.label}"),
                file=sys.stderr,
            )

        agent = build_agent(
            kwim,
            session_id,
            model=model,
            toolsets=toolsets,
            capabilities=capabilities,
            extra_instructions=extra_instructions,
            persona=persona,
            # No shell tool: task_complete/task_blocked plus memory recall. Execution goes
            # through exec-mcp's run_task.
            tools=[kwim_recall, task_complete, task_blocked],
        )
        deps = Deps(kwim=kwim, session_id=session_id)
        session = Session(
            agent=agent,
            deps=deps,
            kwim=kwim,
            session_id=session_id,
            # Per-turn limits; the driver's budget check runs only between turns.
            # `total_tokens_limit` is not used: it raises mid-turn, before a handoff can be
            # written. The token budget is enforced between turns by `_budget_stop`.
            usage_limits=UsageLimits(
                request_limit=budget.max_requests_per_turn,
                tool_calls_limit=budget.max_requests_per_turn,
            ),
        )

        # A run is a resume when the workspace already holds a handoff.
        preamble = resume_preamble(ws)
        if preamble:
            print(_c(DIM, f"resuming: {len(preamble)} chars of prior state"))
        print(_c(DIM, f"task={manifest.task_id} persona={persona.tag} session={session_id}"))
        prompt = _seed_prompt(workspace_md, prose, preamble)
        # One instance for the whole run: the inconclusive count spans turns.
        render = _OutcomeSink(deps)

        try:
            async with agent:
                while True:
                    # Checked at the top of every iteration; skipped on entry so a run always gets
                    # one turn.
                    stop = state.iterations and _budget_stop(state, budget)
                    if stop:
                        state.error = f"budget exhausted: {stop}"
                        kind = "exhausted"
                        break
                    state.iterations += 1
                    print(_c(DIM, f"\n--- iteration {state.iterations} ---"))
                    result = await session.run_turn(prompt, render, _approve)
                    # From the trace, with `usage` as the fallback: the trace records cost per
                    # request, which is what a failed turn is diagnosed from.
                    reqs, tin, tout = totals(result.trace)
                    state.requests += reqs
                    state.input_tokens += tin or result.usage.input_tokens
                    state.output_tokens += tout or result.usage.output_tokens
                    state.trace = result.trace
                    if args.trace:
                        print(_c(DIM, format_trace(result.trace)), file=sys.stderr)
                    if result.text.strip():
                        state.last_text = result.text
                    if result.error:
                        if "UsageLimit" not in result.error:
                            state.error = result.error
                            kind = "error"
                            break
                        # A tripped usage limit ends the turn, not the run: the work stands and
                        # `drop_dangling_tool_calls` has left the transcript resumable, so start a new
                        # turn.
                        prompt = RESUME
                        continue
                    if deps.outcome is not None:
                        outcome = deps.outcome
                        kind = outcome.kind
                        break
                    # Text with no tool call and no declared outcome: hand the run to a human.
                    kind = "review"
                    break
        except KeyboardInterrupt:
            session.interrupt()
            state.error = "interrupted by the operator"
            kind = "error"

    # An undeclared ending prints the tail of the trace without --trace.
    if kind in ("error", "exhausted") and state.trace and not args.trace:
        print(_c(DIM, "\nlast requests before the ending:"), file=sys.stderr)
        print(_c(DIM, format_trace(state.trace, limit=5)), file=sys.stderr)

    write_handoff(ws / "HANDOFF.md", manifest.task_id, kind, state, outcome)

    colour = GREEN if kind == "complete" else YELLOW if kind in ("blocked", "review") else RED
    print(_c(colour, f"\n[{kind}]"), end=" ")
    if outcome is not None:
        if outcome.kind == "blocked":
            print(_c(DIM, f"{outcome.reason}: {outcome.detail}"))
        print(outcome.summary.strip())
    elif kind == "review":
        # The prose is what the model stopped to say.
        print(state.last_text.strip() or "(ended a turn with no text and no outcome)")
    elif state.error:
        print(_c(RED, state.error))
    print(
        _c(
            DIM,
            f"[{state.iterations} iteration(s), {state.requests} request(s), "
            f"{state.input_tokens} in / "
            f"{state.output_tokens} out, {state.elapsed:.0f}s]"
            f"  handoff -> {ws / 'HANDOFF.md'}",
        )
    )
    return EXIT[kind]


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
