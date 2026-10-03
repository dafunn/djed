#!/usr/bin/env python
"""Standalone MCP bridge probe, usable as a preflight.

Connects to every server in `DJED_MCP_SERVERS`, prints each tier's tools and
token cost, and exits non-zero if any server failed or none were configured.
Use --env-file, not shell `source` (see `load_env_file` in headless.py).

    ./.venv/bin/python mcp_probe.py --env-file ~/.djed.env
    ./.venv/bin/python mcp_probe.py --quiet && ./.venv/bin/python tui.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from headless import load_env_file
from djed.mcp import load_tiers, parse_mcp_config


async def main() -> int:
    ap = argparse.ArgumentParser(description="Probe the DJED MCP fleet")
    ap.add_argument("--env-file", default=None, help="load a KEY=VALUE env file literally")
    ap.add_argument("--quiet", action="store_true", help="only report problems")
    ap.add_argument("--tools", action="store_true", help="list every tool name per tier")
    args = ap.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    if not parse_mcp_config():
        print(
            "No MCP servers configured. Set DJED_MCP_SERVERS to a JSON array.\n"
            "If already set, check it was not shell-`source`d - quote removal makes "
            "it invalid JSON.",
            file=sys.stderr,
        )
        return 1

    tiers, errors, warnings = await load_tiers()

    if not args.quiet:
        width = max((len(t.name) for t in tiers), default=4)
        for t in tiers:
            print(f"{t.name:<{width}}  {t.tool_count:>3} tools  >={t.token_cost:>5} tok")
            if args.tools:
                for name in t.tool_names:
                    print(f"{'':<{width}}      {name}")

    for w in warnings:
        print(w, file=sys.stderr)
    for name, err in errors:
        print(f"MCP {name}: {err}", file=sys.stderr)

    if errors:
        print(f"\nMCP FAILED: {len(errors)} server(s) unreachable", file=sys.stderr)
        return 1

    total_tools = sum(t.tool_count for t in tiers)
    total_tokens = sum(t.token_cost for t in tiers)
    if not args.quiet:
        print(
            f"\nMCP OK: {total_tools} tool(s) from {len(tiers)} server(s), "
            f">={total_tokens} input tokens per request"
        )
        window = int(os.environ.get("DJED_CONTEXT_WINDOW") or 40000)
        pct = 100 * total_tokens / window if window else 0
        # The block is re-sent on every request in a turn, so a multi-call turn
        # pays this several times over.
        print(
            f"        >={pct:.0f}% of a {window} token window, per request, and "
            f"re-sent on every\n        tool call in a turn. Use --tiers to load "
            f"fewer. Treat the number as a floor."
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
