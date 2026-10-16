#!/usr/bin/env python
"""Headless driver: streams text, shows tool calls, runs the approval gate on
the console.

    ./.venv/bin/python headless.py "what pods are running in monitoring?"
    ./.venv/bin/python headless.py --fake "smoke test"   # no cluster needed
    ./.venv/bin/python headless.py --list-models
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from djed.agent import build_agent, build_kwim_client
from djed.mcp import load_tiers
from djed.persona import default_persona
from djed.tiers import WRITE_TOOLS, build_strategy
from djed.provider import list_models
from djed.session import Session, TextChunk, ToolFinished, ToolStarted, TurnEvent
from djed.skills import discover, oversize_warning
from djed.tools import Deps
from djed.version import release_version

DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
OFF = "\x1b[0m"


def _c(color: str, s: str) -> str:
    return s if os.environ.get("NO_COLOR") else f"{color}{s}{OFF}"


def load_env_file(path: str) -> int:
    """Load a KEY=VALUE env file literally, as `podman --env-file` does.

    Never `source` it: shell quote removal turns `[{"name":"db"}]` into
    `[{name:db}]`, and the MCP bridge then loads zero servers.
    """
    resolved = os.path.expanduser(path)
    if not os.path.isfile(resolved):
        raise SystemExit(f"env file not found: {resolved}")
    loaded = 0
    with open(resolved) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # An explicitly exported variable wins over the file (dotenv semantics),
            # so a one-off override on the command line is not silently clobbered.
            os.environ.setdefault(key.strip(), value.strip())
            loaded += 1
    return loaded


def _render_event(event: TurnEvent) -> None:
    """Render one Session event to the console."""
    if isinstance(event, TextChunk):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolStarted):
        print(f"\n{_c(CYAN, f'[tool] {event.name}')} {_c(DIM, event.args)}", flush=True)
    elif isinstance(event, ToolFinished):
        marker = "" if event.is_error else "[ok] "
        style = YELLOW if event.is_error else DIM
        print(_c(style, f"  {marker}{event.text}"), flush=True)


async def _ask_approval(tool_name: str, detail: str, mode: str) -> bool:
    """Operator approval gate - the console stand-in for the TUI's y/n prompt."""
    if mode == "auto":
        print(_c(DIM, f"  auto-approved: {tool_name} {detail}"), flush=True)
        return True
    if mode == "deny":
        print(_c(RED, f"  auto-denied: {tool_name} {detail}"), flush=True)
        return False
    answer = await asyncio.to_thread(
        input, f"\n{_c(YELLOW, '[approve (y/n)]')} {tool_name}: {detail} "
    )
    return answer.strip().lower().startswith("y")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Headless Djed")
    ap.add_argument("--version", action="version", version=f"djed {release_version()}")
    ap.add_argument("prompt", nargs="*", help="prompt (reads stdin if omitted)")
    # Resolved after --env-file loads (see tui.py).
    ap.add_argument("--session", default=None)
    ap.add_argument("--model", default=None, help="override DJED_MODEL")
    ap.add_argument(
        "--approve",
        choices=["ask", "auto", "deny"],
        default="ask",
        help=(
            "how to answer the approval gate, which holds run_command and base-mcp's "
            f"file writes ({', '.join(sorted(WRITE_TOOLS))}) unless DJED_GATE_WRITES "
            "is off (default: ask)"
        ),
    )
    ap.add_argument("--no-mcp", action="store_true", help="skip the MCP fleet bridge")
    ap.add_argument(
        "--tiers",
        default=None,
        help="which MCP tiers to load: 'all', 'none', 'cluster,db', or deltas '+cluster,-db'",
    )
    ap.add_argument(
        "--tool-strategy",
        choices=["eager", "search"],
        default=None,
        help="eager = send tier definitions every request; search = hide behind tool search",
    )
    ap.add_argument(
        "--env-file",
        default=None,
        help="load a KEY=VALUE env file literally (use this, not shell `source`)",
    )
    ap.add_argument("--list-models", action="store_true", help="list gateway models and exit")
    ap.add_argument(
        "--fake",
        action="store_true",
        help="drive a scripted stub model instead of the gateway (offline smoke test)",
    )
    args = ap.parse_args()

    if args.env_file:
        n = load_env_file(args.env_file)
        print(_c(DIM, f"loaded {n} vars from {args.env_file}"), file=sys.stderr)

    persona = default_persona()
    args.session = args.session or os.environ.get("DJED_SESSION") or f"{persona.tag}-cli"

    if args.list_models:
        models = await list_models()
        print("\n".join(models) if models else "(could not list gateway models)")
        return 0 if models else 1

    prompt = " ".join(args.prompt).strip() or sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt given (pass it as an argument or on stdin)")

    model = None
    if args.fake:
        from fake_model import scripted_model

        model = scripted_model()
    elif args.model:
        # A bare string would be read as a Pydantic AI model spec ("openai:gpt-5"),
        # so resolve the id through our own gateway provider instead.
        from djed.provider import build_model

        model = build_model(args.model)

    async with httpx.AsyncClient(timeout=30.0) as http:
        kwim = build_kwim_client(http)

        toolsets: list = []
        capabilities: list = []
        extra_instructions = ""
        tiers: list = []
        if not args.no_mcp and not args.fake:
            tiers, mcp_errors, mcp_warnings = await load_tiers()
            for name, err in mcp_errors:
                print(_c(RED, f"MCP {name}: {err}"), file=sys.stderr)
            for w in mcp_warnings:
                print(_c(DIM, w), file=sys.stderr)

            strategy, sel_err = build_strategy(
                args.tool_strategy, args.tiers, [t.name for t in tiers]
            )
            if sel_err:
                print(_c(RED, f"tier selection: {sel_err}"), file=sys.stderr)
            toolsets, capabilities = strategy.apply(tiers)
            extra_instructions = strategy.instructions
            loaded = sum(t.tool_count for t in tiers)
            active = "all" if len(toolsets) == len(tiers) else f"{len(toolsets)}/{len(tiers)}"
            print(
                _c(DIM, f"MCP: {active} tiers, {loaded} tools, strategy={strategy.label}"),
                file=sys.stderr,
            )

        found = discover(persona.tag)
        for note in found.problems + [oversize_warning(s) for s in found.oversized]:
            print(_c(YELLOW, note), file=sys.stderr)

        agent = build_agent(
            kwim,
            args.session,
            model=model,
            toolsets=toolsets,
            capabilities=capabilities,
            extra_instructions=extra_instructions,
            persona=persona,
            skills=found.skills,
        )

        session = Session(
            agent=agent,
            deps=Deps(kwim=kwim, session_id=args.session, skills=found.skills),
            kwim=kwim,
            session_id=args.session,
        )
        print(_c(DIM, f"you> {prompt}\n"))

        async def approve(tool_name: str, detail: str) -> bool:
            return await _ask_approval(tool_name, detail, args.approve)

        async with agent:
            result = await session.run_turn(prompt, _render_event, approve)

        print()
        if result.error:
            print(_c(RED, f"[error] {result.error}"), file=sys.stderr)
        u = result.usage
        print(
            _c(DIM, f"\n[{u.input_tokens} in / {u.output_tokens} out, "
                    f"{u.requests} request(s)]")
        )
    return 1 if result.error else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
