#!/usr/bin/env python
"""Launch the Djed TUI.

    ./.venv/bin/python tui.py --env-file ~/.djed.env
    ./.venv/bin/python tui.py --fake                  # no cluster needed
    ./.venv/bin/python tui.py --tiers cluster,db --tool-strategy eager

Bring the MCP port-forwards up first (`./run_forwards.sh`) if you want the fleet.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack

import httpx

from headless import load_env_file
from djed.agent import build_agent, build_kwim_client
from djed.mcp import load_tiers
from djed.persona import default_persona
from djed.provider import model_id
from djed.session import Session
from djed.terminal import tmux_notes
from djed.tiers import build_strategy
from djed.tools import Deps
from djed.tui import DjedApp


async def main() -> int:
    ap = argparse.ArgumentParser(description="Djed terminal")
    # Resolved after --env-file loads, not here: an env file that sets
    # DJED_SESSION or DJED_AGENT has not been read yet at parser-build time.
    ap.add_argument("--session", default=None)
    ap.add_argument("--model", default=None, help="override DJED_MODEL")
    ap.add_argument("--env-file", default=None, help="load a KEY=VALUE env file literally")
    ap.add_argument("--no-mcp", action="store_true", help="skip the MCP fleet bridge")
    ap.add_argument(
        "--tiers", default=None, help="'all', 'none', 'cluster,db', or '+cluster,-db'"
    )
    ap.add_argument("--tool-strategy", choices=["eager", "search"], default=None)
    ap.add_argument("--fake", action="store_true", help="scripted stub model, no gateway")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    persona = default_persona()
    args.session = args.session or os.environ.get("DJED_SESSION") or f"{persona.tag}-cli"

    model = None
    if args.fake:
        from fake_model import scripted_model

        model = scripted_model()
    elif args.model:
        from djed.provider import build_model

        model = build_model(args.model)

    notes: list[str] = list(tmux_notes())
    async with httpx.AsyncClient(timeout=30.0) as http:
        kwim = build_kwim_client(http)

        tiers = []
        if not args.no_mcp and not args.fake:
            tiers, errors, warnings = await load_tiers()
            notes += [f"MCP {n}: {e}" for n, e in errors] + list(warnings)

        strategy, sel_err = build_strategy(
            args.tool_strategy, args.tiers, [t.name for t in tiers]
        )
        if sel_err:
            notes.append(f"tier selection: {sel_err}")
        toolsets, capabilities = strategy.apply(tiers)

        def agent_factory(toolsets, capabilities, extra_instructions, override=None):
            return build_agent(
                kwim,
                args.session,
                model=override if override is not None else model,
                toolsets=toolsets,
                capabilities=capabilities,
                extra_instructions=extra_instructions,
                persona=persona,
            )

        session = Session(
            agent=agent_factory(toolsets, capabilities, strategy.instructions),
            deps=Deps(kwim=kwim, session_id=args.session),
            kwim=kwim,
            session_id=args.session,
            tiers=tiers,
            strategy=strategy,
            agent_factory=agent_factory,
        )

        loaded = sum(t.tool_count for t in tiers)
        fleet = (
            f" | {len(tiers)} MCP tiers, {loaded} tools, strategy={strategy.label}"
            if tiers
            else ""
        )
        banner = (
            f"djed - {'stub model' if args.fake else model_id()}"
            f"{fleet}"
            "\n/help for commands."
        )
        if notes:
            banner += "\n" + "\n".join(notes)

        # Hold the MCP connections open for the whole session, tied to the
        # toolsets rather than to the agent - `/tier` and `/strategy` rebuild the
        # agent, and the connections must survive that.
        async with AsyncExitStack() as stack:
            for tier in tiers:
                await stack.enter_async_context(tier.toolset)
            await DjedApp(session, banner, persona=persona).run_async()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
