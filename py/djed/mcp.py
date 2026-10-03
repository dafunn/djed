"""MCP fleet bridge: config translation, plus the one thing FastMCP does not
do - collision-only renaming.

A tool keeps its bare name unless the same name appears on more than one
server; then every instance becomes `<server>__<tool>`. Prefixing
unconditionally would yield names like `cluster_cluster_list`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from fastmcp.client import Client
from fastmcp.client.transports import SSETransport, StdioTransport, StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import RenamedToolset

from .tiers import Tier
from .tokens import estimate_tool_tokens

log = logging.getLogger("djed.mcp")


def mcp_timeout_s() -> float:
    return float(os.environ.get("DJED_MCP_TIMEOUT_MS") or 10000) / 1000.0


@dataclass
class McpServer:
    """One configured server, before its tools are known."""

    name: str
    client: Client
    toolset: MCPToolset


def parse_mcp_config(raw: str | None = None) -> list[dict[str, Any]]:
    """Parse `DJED_MCP_SERVERS`. Fail-soft: a malformed value yields no servers."""
    raw = raw if raw is not None else os.environ.get("DJED_MCP_SERVERS", "")
    if not raw or not raw.strip():
        return []
    try:
        cfgs = json.loads(raw)
    except json.JSONDecodeError as e:
        # Most common cause: the env file was `source`d by a shell. See
        # `load_env_file` in headless.py.
        log.warning("DJED_MCP_SERVERS is not valid JSON, ignoring: %s", e)
        return []
    if not isinstance(cfgs, list):
        log.warning("DJED_MCP_SERVERS must be a JSON array, ignoring")
        return []
    out = []
    for c in cfgs:
        if not isinstance(c, dict) or not c.get("name"):
            log.warning("skipping MCP entry without a name: %r", c)
            continue
        out.append(c)
    return out


def _transport(cfg: dict[str, Any]):
    kind = (cfg.get("transport") or "sse").lower()
    if kind == "stdio":
        if not cfg.get("command"):
            raise ValueError(f"MCP server {cfg['name']}: stdio transport needs a `command`")
        return StdioTransport(
            command=cfg["command"],
            args=cfg.get("args") or [],
            env=cfg.get("env") or None,
        )
    if not cfg.get("url"):
        raise ValueError(f"MCP server {cfg['name']}: {kind} transport needs a `url`")
    if kind == "sse":
        return SSETransport(url=cfg["url"], headers=cfg.get("headers") or None)
    if kind in ("http", "streamable-http", "streamable_http"):
        return StreamableHttpTransport(url=cfg["url"], headers=cfg.get("headers") or None)
    raise ValueError(f"MCP server {cfg['name']}: unknown transport {kind!r}")


def _client_timeout_s(cfg: dict[str, Any]) -> float:
    """A per-entry `timeout` overrides the global default: `base` keeps the short
    default, and `run_task` gets one sized to the run's budget.
    """
    raw = cfg.get("timeout")
    return float(raw) if raw is not None else mcp_timeout_s()


def build_mcp_servers(
    cfgs: list[dict[str, Any]] | None = None,
) -> tuple[list[McpServer], list[tuple[str, str]]]:
    """Build one server per config entry. Returns `(servers, errors)`.

    Config errors are caught per entry - one bad server must not take down the
    fleet. Construction is lazy; a server being down surfaces at connect time.
    """
    cfgs = parse_mcp_config() if cfgs is None else cfgs
    servers: list[McpServer] = []
    errors: list[tuple[str, str]] = []
    for cfg in cfgs:
        name = cfg["name"]
        try:
            client = Client(_transport(cfg), timeout=_client_timeout_s(cfg))
            servers.append(McpServer(name, client, MCPToolset(client, id=name)))
        except Exception as e:
            errors.append((name, str(e)))
    return servers, errors


async def _tool_defs(server: McpServer) -> list[Any]:
    """Return full tool definitions, not just names.

    `estimate_tool_tokens` needs each tool's description and schema. Reuses an
    already-open client so probing adds no connect/disconnect cycle.
    """
    if server.client.is_connected():
        return list(await server.client.list_tools())
    async with server.client:
        return list(await server.client.list_tools())


def resolve_collisions(
    tools_by_server: dict[str, list[str]],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Work out which tools need renaming.

    Returns `(name_maps, warnings)`; `name_maps[server]` maps new -> original,
    ready for `RenamedToolset`. Colliding names are renamed on every server, so
    an ambiguous name is never left to guesswork.
    """
    counts: dict[str, int] = {}
    for names in tools_by_server.values():
        for n in names:
            counts[n] = counts.get(n, 0) + 1
    colliding = {n for n, c in counts.items() if c > 1}

    name_maps: dict[str, dict[str, str]] = {}
    for server, names in tools_by_server.items():
        mapping = {f"{server}__{n}": n for n in names if n in colliding}
        if mapping:
            name_maps[server] = mapping

    warnings = []
    for name in sorted(colliding):
        affected = sorted(s for s, names in tools_by_server.items() if name in names)
        renamed = ", ".join(f"{s}__{name}" for s in affected)
        warnings.append(
            f'MCP name collision: "{name}" on {", ".join(affected)}; renamed to {renamed}'
        )
    return name_maps, warnings


async def load_tiers(
    cfgs: list[dict[str, Any]] | None = None,
) -> tuple[list[Tier], list[tuple[str, str]], list[str]]:
    """Connect to every configured server and return one `Tier` each.

    Returns `(tiers, errors, warnings)`. Probed concurrently and fail-soft: an
    unreachable server is reported and skipped. Which tiers reach the model is
    `djed/tiers.py`'s call.
    """
    servers, errors = build_mcp_servers(cfgs)
    if not servers:
        return [], errors, []

    probes = await asyncio.gather(
        *(_tool_defs(s) for s in servers), return_exceptions=True
    )

    live: list[McpServer] = []
    tools_by_server: dict[str, list[str]] = {}
    defs_by_server: dict[str, list[Any]] = {}
    for server, result in zip(servers, probes):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            errors.append((server.name, str(result) or type(result).__name__))
            continue
        live.append(server)
        defs_by_server[server.name] = result
        tools_by_server[server.name] = [t.name for t in result]

    name_maps, warnings = resolve_collisions(tools_by_server)

    tiers: list[Tier] = []
    for server in live:
        mapping = name_maps.get(server.name)
        toolset = RenamedToolset(server.toolset, mapping) if mapping else server.toolset
        # Report post-rename names - that is what the operator and model see.
        # `mapping` only covers colliding tools; the rest keep their bare names.
        to_new = {orig: new for new, orig in (mapping or {}).items()}
        names = tuple(to_new.get(n, n) for n in tools_by_server[server.name])
        cost = estimate_tool_tokens(defs_by_server[server.name])
        tiers.append(Tier(server.name, toolset, names, token_cost=cost))
    return tiers, errors, warnings
