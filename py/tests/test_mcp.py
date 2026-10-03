"""Config translation and collision-only renaming for the MCP fleet bridge."""
from __future__ import annotations

import pytest

from djed.mcp import (
    _client_timeout_s,
    build_mcp_servers,
    mcp_timeout_s,
    parse_mcp_config,
    resolve_collisions,
)

# --- config parsing -------------------------------------------------------


def test_empty_config_yields_no_servers():
    assert parse_mcp_config("") == []
    assert parse_mcp_config("   ") == []


def test_malformed_json_is_fail_soft():
    assert parse_mcp_config("{not json") == []


def test_shell_sourced_config_is_rejected_not_half_parsed():
    """A shell-sourced config must be refused outright, not half-parsed."""
    mangled = "[{name:db,transport:sse,url:http://localhost:3001/sse}]"
    assert parse_mcp_config(mangled) == []


def test_non_array_is_rejected():
    assert parse_mcp_config('{"name": "logs"}') == []


def test_entries_without_a_name_are_skipped():
    cfg = '[{"transport":"sse","url":"http://x/sse"},{"name":"ok","url":"http://y/sse"}]'
    assert [c["name"] for c in parse_mcp_config(cfg)] == ["ok"]


# --- per-server timeout ---------------------------------------------------


def test_an_entry_without_a_timeout_key_gets_the_global_default():
    assert _client_timeout_s({"name": "base"}) == mcp_timeout_s()


def test_an_entry_with_a_timeout_key_overrides_the_global_default():
    """`base` keeps the short default timeout; `run_task` gets its own."""
    assert _client_timeout_s({"name": "exec", "timeout": 3600}) == 3600.0


# --- server construction --------------------------------------------------


def test_sse_and_stdio_both_build():
    cfgs = [
        {"name": "logs", "transport": "sse", "url": "http://localhost:3100/sse"},
        {"name": "notes", "transport": "stdio", "command": "python3", "args": ["server.py"]},
    ]
    servers, errors = build_mcp_servers(cfgs)
    assert errors == []
    assert [s.name for s in servers] == ["logs", "notes"]


@pytest.mark.parametrize(
    "cfg",
    [
        {"name": "logs", "transport": "sse"},  # no url
        {"name": "notes", "transport": "stdio"},  # no command
        {"name": "weird", "transport": "carrier-pigeon", "url": "http://x"},
    ],
)
def test_bad_entries_are_reported_not_raised(cfg):
    """One misconfigured server must not take down the whole fleet."""
    servers, errors = build_mcp_servers([cfg])
    assert servers == []
    assert [e[0] for e in errors] == [cfg["name"]]


def test_one_bad_server_does_not_block_the_others():
    cfgs = [
        {"name": "broken", "transport": "sse"},
        {"name": "good", "transport": "sse", "url": "http://localhost:3100/sse"},
    ]
    servers, errors = build_mcp_servers(cfgs)
    assert [s.name for s in servers] == ["good"]
    assert [e[0] for e in errors] == ["broken"]


# --- collision-only renaming ----------------------------------------------


def test_unique_names_are_left_alone():
    """No `cluster_cluster_list` when nothing collides."""
    name_maps, warnings = resolve_collisions(
        {"cluster": ["cluster_list", "cluster_logs"], "db": ["db_query"]}
    )
    assert name_maps == {}
    assert warnings == []


def test_colliding_names_are_renamed_on_every_server():
    name_maps, warnings = resolve_collisions(
        {"db": ["query", "tables"], "logs": ["query", "labels"]}
    )
    assert name_maps["db"] == {"db__query": "query"}
    assert name_maps["logs"] == {"logs__query": "query"}
    assert len(warnings) == 1
    assert '"query"' in warnings[0]
    assert "db__query" in warnings[0] and "logs__query" in warnings[0]


def test_non_colliding_tools_on_a_colliding_server_keep_bare_names():
    name_maps, _ = resolve_collisions({"db": ["query", "tables"], "logs": ["query"]})
    assert "db__tables" not in name_maps["db"]
    assert name_maps["db"] == {"db__query": "query"}


def test_three_way_collision_renames_all_three():
    name_maps, warnings = resolve_collisions(
        {"a": ["status"], "b": ["status"], "c": ["status"]}
    )
    assert set(name_maps) == {"a", "b", "c"}
    assert len(warnings) == 1
    assert all(f"{s}__status" in warnings[0] for s in "abc")


def test_no_servers_yields_nothing():
    assert resolve_collisions({}) == ({}, [])
