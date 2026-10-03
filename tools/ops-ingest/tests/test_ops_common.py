"""Unit tests for ops.common utilities."""
from pathlib import Path

import httpx
import pytest

import common


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


class _FakeHttpxClient:
    def __init__(self, responses=None):
        self.requests: list[dict] = []
        self.responses = responses or []
        self._index = 0

    async def get(self, url, *, params=None, headers=None):
        self.requests.append(
            {"method": "GET", "url": url, "params": params, "headers": headers}
        )
        resp = self.responses[self._index]
        self._index += 1
        return resp

    async def post(self, url, *, headers=None, json=None):
        self.requests.append({"method": "POST", "url": url, "headers": headers, "json": json})
        resp = self.responses[self._index]
        self._index += 1
        return resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_fact_identity_host_scoped_types_use_about_zero():
    for fact_type in ("host_inventory", "access_grant", "host_role"):
        assert common.fact_identity(fact_type, ["host-a", "group-a", "site-a"]) == "host-a"


def test_fact_identity_host_alias_uses_host_and_alias_pair():
    assert common.fact_identity("host_alias", ["host-a", "dashboards"]) == (
        "host-a",
        "dashboards",
    )


def test_fact_identity_network_topology_uses_about_one():
    assert common.fact_identity("network_topology", ["network", "nfs", "site-a"]) == "nfs"


def test_desired_fact_identity_set_automatically():
    fact = common.DesiredFact(
        fact_type="host_inventory", about=["host-a", "group-a", "site-a"], statement="s",
    )
    assert fact.identity == "host-a"


def test_parse_ansible_inventory():
    text = """
[group-c]
host-g ansible_host=10.0.0.13 ansible_user=root
host-e ansible_host=10.0.0.12 ansible_user=ubuntu

[group-a]
host-a ansible_host=10.0.0.11 ansible_user=root
host-b ansible_host=10.0.0.17 ansible_user=root

[group-b:children]
group-c
"""
    hosts, groups = common.parse_ansible_inventory(text)
    assert hosts["host-g"]["ansible_host"] == "10.0.0.13"
    assert hosts["host-g"]["ansible_user"] == "root"
    assert hosts["host-a"]["ansible_host"] == "10.0.0.11"
    assert groups["group-c"] == ["host-g", "host-e"]
    assert groups["group-a"] == ["host-a", "host-b"]
    assert "group-b" not in groups  # :children meta-group is skipped


def test_parse_ansible_inventory_duplicate_host_last_wins():
    text = """
[group-c]
host-k ansible_host=10.0.0.19
host-k ansible_host=10.0.0.15
"""
    hosts, groups = common.parse_ansible_inventory(text)
    assert hosts["host-k"]["ansible_host"] == "10.0.0.15"
    assert groups["group-c"] == ["host-k", "host-k"]


def test_parse_hosts_file():
    text = """
10.0.0.11 host-a
10.0.0.13 host-g dashboards
10.0.0.12 host-e app.example.org
# 10.0.0.18 deb-docker
"""
    ip_map = common.parse_hosts_file(text)
    assert ip_map["10.0.0.11"] == ["host-a"]
    assert ip_map["10.0.0.13"] == ["host-g", "dashboards"]
    assert ip_map["10.0.0.12"] == ["host-e", "app.example.org"]
    assert "10.0.0.18" not in ip_map


def test_load_host_roles(tmp_path: Path):
    path = tmp_path / "roles.yaml"
    path.write_text(
        "host-e:\n  - AppServer\n  - Database\nhost-g: Dashboards\n",
        encoding="utf-8",
    )
    roles = common.load_host_roles(path)
    assert roles["host-e"] == ["AppServer", "Database"]
    assert roles["host-g"] == ["Dashboards"]


def test_load_host_vars_filters_allow_list(tmp_path: Path):
    path = tmp_path / "host-f.yml"
    path.write_text(
        "runner_capacity: 2\nansible_user: debian\n",
        encoding="utf-8",
    )
    overrides = common.load_host_vars(path)
    assert overrides == {"ansible_user": "debian"}


def test_key_type_from_priv_files(tmp_path: Path):
    group_dir = tmp_path / "group-c"
    group_dir.mkdir()
    (group_dir / "key-b.priv").write_text("ENCRYPTED", encoding="utf-8")
    assert common.key_type_from_priv_files(group_dir) == "key-b"


def test_key_type_from_priv_files_zero(tmp_path: Path):
    group_dir = tmp_path / "group-a"
    group_dir.mkdir()
    (group_dir / "vault").write_text("ENCRYPTED", encoding="utf-8")
    assert common.key_type_from_priv_files(group_dir) is None


def test_key_type_from_priv_files_multiple(tmp_path: Path):
    group_dir = tmp_path / "net"
    group_dir.mkdir()
    (group_dir / "key-net.priv").write_text("A", encoding="utf-8")
    (group_dir / "root.example-key.priv").write_text("B", encoding="utf-8")
    assert common.key_type_from_priv_files(group_dir) is None


def test_resolve_key_type_per_host_file():
    params = {"ansible_ssh_private_key_file": "~/.ssh/key-group-e"}
    assert (
        common.resolve_key_type("host-i", params, ["group-e"], Path("/nonexistent"))
        == "key-group-e"
    )


def test_resolve_key_type_group_single_priv(tmp_path: Path):
    group_dir = tmp_path / "group-c"
    group_dir.mkdir()
    (group_dir / "key-b.priv").write_text("A", encoding="utf-8")
    params = {}
    assert common.resolve_key_type("host-g", params, ["group-c"], tmp_path) == "key-b"


def test_resolve_key_type_group_multiple_priv_is_unknown(tmp_path: Path):
    net_dir = tmp_path / "net"
    net_dir.mkdir()
    (net_dir / "key-net.priv").write_text("A", encoding="utf-8")
    (net_dir / "root.example-key.priv").write_text("B", encoding="utf-8")
    params = {}
    assert common.resolve_key_type("switch", params, ["net"], tmp_path) == "unknown"


def test_resolve_key_type_no_source_is_unknown():
    params = {}
    assert (
        common.resolve_key_type("foo", params, ["unknowngroup"], Path("/nonexistent"))
        == "unknown"
    )


def test_scan_for_secrets_catches_password_and_private_key():
    text = "The password is swordfish.\n-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"
    hits = common.scan_for_secrets(text)
    assert any("password" in h for h in hits)
    assert any("BEGIN" in h for h in hits)


def test_scan_for_secrets_catches_mysql_pflag():
    text = "docker exec -it database mysql -u appuser -psword appdb -e 'SELECT 1'"
    hits = common.scan_for_secrets(text)
    assert any("-p" in h for h in hits)


def test_scan_for_secrets_catches_long_base64():
    text = "token " + "A" * 44 + "=="
    hits = common.scan_for_secrets(text)
    assert hits


def test_scan_for_secrets_clean_text():
    assert common.scan_for_secrets(
        "This is a normal runbook section about restarting host-j."
    ) == []


def test_chunk_markdown_splits_by_headings():
    text = "# Intro\nintro line.\n## Setup\nsetup line.\n## Setup\nsecond setup.\n"
    chunks = common.chunk_markdown(text)
    assert len(chunks) == 3
    headings = [h for h, _ in chunks]
    assert headings == ["Intro", "Setup", "Setup-2"]


def test_chunk_markdown_no_heading():
    chunks = common.chunk_markdown("just an intro")
    assert chunks == [("intro", "just an intro")]


def test_section_id_stable():
    a = common.section_id("runbook.md", "Intro")
    b = common.section_id("runbook.md", "Intro")
    c = common.section_id("runbook.md", "Setup")
    assert a == b
    assert a != c
    assert len(a) == 64  # hex sha256


async def test_kwim_client_query_facts_client_side_filter():
    client = _FakeHttpxClient([
        _FakeResponse(200, [
            {"id": "f1", "about": ["host-a", "group-a", "site-a"], "statement": "s1"},
            {"id": "f2", "about": ["host-a", "group-c", "site-a"], "statement": "s2"},
        ]),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    facts = await kwim.query_facts(["host-a", "group-a"], "host_inventory")
    assert len(facts) == 1
    assert facts[0]["id"] == "f1"


async def test_kwim_client_query_owned_scopes_by_source_kind_not_env():
    client = _FakeHttpxClient([
        _FakeResponse(
            200,
            [{"id": "f1", "about": ["host-a", "group-a", "access"], "statement": "s1"}],
        ),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    facts = await kwim.query_owned("access_grant")
    assert len(facts) == 1
    params = dict(client.requests[0]["params"])
    assert params == {
        "fact_type": "access_grant",
        "status_": "current",
        "source_kind": "repo_sync",
        "limit": "10000",
    }


async def test_kwim_client_query_owned_respects_limit_param():
    client = _FakeHttpxClient([_FakeResponse(200, [])])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    await kwim.query_owned("host_inventory", limit=3)
    params = dict(client.requests[0]["params"])
    assert params["limit"] == "3"


async def test_kwim_client_proposes_when_no_current_fact():
    client = _FakeHttpxClient([
        _FakeResponse(200, []),  # exact full-about miss
        _FakeResponse(200, []),  # identity fallback miss too - genuinely new
        _FakeResponse(201, {"id": "new"}),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    fact = common.DesiredFact(
        fact_type="host_inventory",
        about=["host-a", "group-a", "site-a"],
        statement="host-a is a group-a host.",
    )
    result = await kwim.sync_fact(fact)
    assert result["id"] == "new"
    assert client.requests[2]["json"]["statement"] == fact.statement
    assert "supersedes" not in client.requests[2]["json"]


async def test_kwim_client_supersedes_via_identity_when_group_renamed():
    """A host's group changes: full-about miss, identity fallback finds the
    stale fact by host id alone and supersedes it - one fact, not a duplicate."""
    client = _FakeHttpxClient([
        _FakeResponse(200, []),  # exact match on the new about misses
        _FakeResponse(200, [
            {
                "id": "old",
                "about": ["host-a", "group-a", "site-a"],
                "statement": "host-a is a group-a host.",
            },
        ]),
        _FakeResponse(201, {"id": "new"}),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    fact = common.DesiredFact(
        fact_type="host_inventory",
        about=["host-a", "group-b", "site-a"],
        statement="host-a is a group-b host.",
    )
    result = await kwim.sync_fact(fact)
    assert result["id"] == "new"
    assert client.requests[1]["params"] == [
        ("fact_type", "host_inventory"), ("status_", "current"), ("about", "host-a"),
    ]
    assert client.requests[2]["json"]["supersedes"] == "old"


async def test_kwim_client_supersedes_when_statement_changed():
    client = _FakeHttpxClient([
        _FakeResponse(
            200,
            [{
                "id": "old",
                "about": ["host-a", "group-a", "site-a"],
                "statement": "old statement",
            }],
        ),
        _FakeResponse(201, {"id": "new"}),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    fact = common.DesiredFact(
        fact_type="host_inventory",
        about=["host-a", "group-a", "site-a"],
        statement="host-a is a group-a host at 10.0.0.11.",
    )
    result = await kwim.sync_fact(fact)
    assert result["id"] == "new"
    assert client.requests[1]["json"]["supersedes"] == "old"


async def test_kwim_client_skips_when_statement_unchanged():
    statement = "host-a is a group-a host."
    client = _FakeHttpxClient([
        _FakeResponse(
            200,
            [{
                "id": "same",
                "about": ["host-a", "group-a", "site-a"],
                "statement": statement,
            }],
        ),
        _FakeResponse(204, None),
    ])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    fact = common.DesiredFact(
        fact_type="host_inventory",
        about=["host-a", "group-a", "site-a"],
        statement=statement,
    )
    result = await kwim.sync_fact(fact)
    assert result["skipped"] is True
    assert result["reason"] == "unchanged"
    assert len(client.requests) == 2
    assert client.requests[1]["method"] == "POST"
    assert client.requests[1]["url"] == "http://kwim.test/v1/knowledge/facts/same/reaffirm"


async def test_kwim_client_reaffirm_204_returns_true():
    client = _FakeHttpxClient([_FakeResponse(204, None)])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    ok = await kwim.reaffirm("fact-1")
    assert ok is True
    assert client.requests[0]["url"] == "http://kwim.test/v1/knowledge/facts/fact-1/reaffirm"


async def test_kwim_client_reaffirm_404_returns_false_no_raise():
    client = _FakeHttpxClient([_FakeResponse(404, {"detail": "not found"})])
    kwim = common.KwimClient(client, "http://kwim.test", "key")
    ok = await kwim.reaffirm("gone")
    assert ok is False
