"""The release version: from DJED_VERSION, else the checkout's djed-v* tag, else dev."""
from __future__ import annotations

import subprocess

import pytest

from djed import mcp as mcp_mod
from djed import version


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("DJED_VERSION", raising=False)
    version.release_version.cache_clear()
    yield
    version.release_version.cache_clear()


def _git_says(monkeypatch, *, stdout="", returncode=0, raises=None):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(version.subprocess, "run", fake_run)
    return calls


def test_env_var_wins_over_the_checkout(monkeypatch):
    calls = _git_says(monkeypatch, stdout="djed-v0.1.0\n")
    monkeypatch.setenv("DJED_VERSION", "0.9.0")
    assert version.release_version() == "0.9.0"
    assert calls == []


def test_blank_env_var_falls_through_to_the_tag(monkeypatch):
    _git_says(monkeypatch, stdout="djed-v0.2.0\n")
    monkeypatch.setenv("DJED_VERSION", "  ")
    assert version.release_version() == "0.2.0"


def test_tag_prefix_is_stripped_and_the_suffix_kept(monkeypatch):
    calls = _git_says(monkeypatch, stdout="djed-v0.2.0-3-gabc1234\n")
    assert version.release_version() == "0.2.0-3-gabc1234"
    assert calls[0][-2:] == ["--match", "djed-v*"]


def test_no_matching_tag_is_dev(monkeypatch):
    _git_says(monkeypatch, returncode=128)
    assert version.release_version() == "dev"


def test_no_git_is_dev(monkeypatch):
    _git_says(monkeypatch, raises=FileNotFoundError("git"))
    assert version.release_version() == "dev"


def test_git_timeout_is_dev(monkeypatch):
    _git_says(monkeypatch, raises=subprocess.TimeoutExpired("git", 5))
    assert version.release_version() == "dev"


def test_resolved_once_per_process(monkeypatch):
    calls = _git_says(monkeypatch, stdout="djed-v0.2.0\n")
    version.release_version()
    version.release_version()
    assert len(calls) == 1


def test_mcp_clients_send_djed_and_the_release_as_client_info(monkeypatch):
    monkeypatch.setenv("DJED_VERSION", "0.2.0")
    seen = []

    class RecordingClient:
        def __init__(self, transport, **kw):
            seen.append(kw.get("client_info"))

    monkeypatch.setattr(mcp_mod, "Client", RecordingClient)
    monkeypatch.setattr(mcp_mod, "MCPToolset", lambda client, id: None)
    servers, errors = mcp_mod.build_mcp_servers(
        [{"name": "db", "transport": "sse", "url": "http://pg-mcp:3000/sse"}]
    )
    assert errors == []
    assert len(servers) == 1
    assert seen[0].name == "djed"
    assert seen[0].version == "0.2.0"
