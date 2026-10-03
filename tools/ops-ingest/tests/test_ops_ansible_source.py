"""Unit tests for ansible_source."""
from pathlib import Path

import pytest

import ansible_source
import common


class _FakeKwimClient:
    """A fake KwimClient that persists facts between runs.

    Mirrors the real server's status_/source_kind semantics closely enough to
    catch cross-run bugs: query_facts/query_owned only ever see status_="current"
    facts, matching the server's default status_=current filter.
    """

    def __init__(self, facts=None):
        self.facts = facts or []
        self.proposals: list[dict] = []
        self.reaffirmed: list[str] = []
        self._next_id = 1

    async def query_facts(self, about, fact_type):
        return [
            f for f in self.facts
            if f["fact_type"] == fact_type
            and f.get("status_") == "current"
            and all(tag in f.get("about", []) for tag in about)
        ]

    async def query_owned(self, fact_type, limit=10000):
        owned = [
            f for f in self.facts
            if f["fact_type"] == fact_type
            and f.get("status_") == "current"
            and f.get("source_kind") == "repo_sync"
        ]
        return owned[:limit]

    async def propose_fact(
        self, statement, fact_type, about, decay_class="slow", supersedes=None
    ):
        fact = {
            "id": f"fact-{self._next_id}",
            "statement": statement,
            "fact_type": fact_type,
            "about": about,
            "decay_class": decay_class,
            "status_": "current",
            "source_kind": "repo_sync",
        }
        if supersedes:
            fact["supersedes"] = supersedes
            # Mark the old fact as superseded for introspection.
            for old in self.facts:
                if old["id"] == supersedes:
                    old["status_"] = "superseded"
        self._next_id += 1
        self.facts.append(fact)
        self.proposals.append(fact)
        return fact

    async def reaffirm(self, fact_id):
        found = any(f["id"] == fact_id and f.get("status_") == "current" for f in self.facts)
        if found:
            self.reaffirmed.append(fact_id)
        return found

    async def sync_fact(self, fact: common.DesiredFact):
        return await common.KwimClient.sync_fact(self, fact)


@pytest.fixture
def inv_fixture(tmp_path: Path):
    """Build a minimal site-a checkout with allow-listed files only."""
    root = tmp_path / "site-a"
    root.mkdir()

    inventory = root / "inventory"
    inventory.write_text(
        "[group-a]\n"
        "host-a ansible_host=10.0.0.11 ansible_user=root\n"
        "host-c ansible_host=10.0.0.16 ansible_user=root\n"
        "\n"
        "[group-c]\n"
        "host-e ansible_host=10.0.0.12 ansible_user=ubuntu\n"
        "host-g ansible_host=10.0.0.13 ansible_user=root\n"
        "\n"
        "[group-b]\n"
        "host-d ansible_host=10.0.0.14\n"
        "\n"
        "[site-a:children]\n"
        "group-b\n"
        "group-c\n",
        encoding="utf-8",
    )

    hosts = root / "hosts"
    hosts.write_text(
        "10.0.0.11 host-a\n"
        "10.0.0.13 host-g dashboards\n"
        "10.0.0.12 host-e app.example.org\n"
        "10.0.0.14 host-d host-d.example.org\n",
        encoding="utf-8",
    )

    group_vars = root / "playbooks" / "group_vars"
    groupc_dir = group_vars / "group-c"
    groupc_dir.mkdir(parents=True)
    (groupc_dir / "key-b.priv").write_text("ANSIBLE_VAULT", encoding="utf-8")
    groupb_dir = group_vars / "group-b"
    groupb_dir.mkdir(parents=True)
    (groupb_dir / "key-a.priv").write_text("ANSIBLE_VAULT", encoding="utf-8")
    all_dir = group_vars / "all"
    all_dir.mkdir(parents=True)
    (all_dir / "network.yml").write_text(
        "nfs_server: 10.0.0.14\n"
        "nfs_server_hostname: host-d\n"
        "nfs_media_path: /export/media\n"
        "networks:\n"
        "  management:\n"
        "    cidr: 10.0.0.0/24\n"
        "    gateway: 10.0.0.10\n"
        "service_ports:\n"
        "  app_http: 3000\n",
        encoding="utf-8",
    )

    host_vars = root / "host_vars"
    host_vars.mkdir()
    (host_vars / "host-e.yml").write_text(
        "ansible_user: ubuntu\nrunner_capacity: 2\n",
        encoding="utf-8",
    )

    roles = root / "ops-host-roles.yaml"
    roles.write_text(
        "host-e:\n  - AppServer\n  - Database\nhost-g:\n  - Dashboards\n",
        encoding="utf-8",
    )

    return root


@pytest.fixture
def roles_path(inv_fixture: Path) -> Path:
    return inv_fixture / "ops-host-roles.yaml"


def _write_flat_inventory(root: Path, host_ids: list[str], group: str = "group-a") -> Path:
    """Single-group inventory + hosts + roles for N hosts, one role and one
    alias each. No group_vars/network.yml, so key type is always "unknown" and
    no network_topology facts are produced."""
    inv_lines = [f"[{group}]"]
    hosts_lines = []
    roles_lines = []
    for i, host_id in enumerate(host_ids, start=1):
        ip = f"10.0.1.{i}"
        inv_lines.append(f"{host_id} ansible_host={ip} ansible_user=root")
        hosts_lines.append(f"{ip} {host_id} {host_id}-alias")
        roles_lines.append(f"{host_id}:\n  - ExampleService")
    (root / "inventory").write_text("\n".join(inv_lines) + "\n", encoding="utf-8")
    (root / "hosts").write_text("\n".join(hosts_lines) + "\n", encoding="utf-8")
    roles_path = root / "ops-host-roles.yaml"
    roles_path.write_text("\n".join(roles_lines) + "\n", encoding="utf-8")
    return roles_path


def _drop_host(root: Path, roles_path: Path, host_id: str, ip: str) -> None:
    """Remove one host from every source file written by _write_flat_inventory."""
    inventory = root / "inventory"
    inventory.write_text(
        inventory.read_text(encoding="utf-8").replace(
            f"{host_id} ansible_host={ip} ansible_user=root\n", "",
        ),
        encoding="utf-8",
    )
    hosts = root / "hosts"
    hosts.write_text(
        hosts.read_text(encoding="utf-8").replace(f"{ip} {host_id} {host_id}-alias\n", ""),
        encoding="utf-8",
    )
    roles_path.write_text(
        roles_path.read_text(encoding="utf-8").replace(
            f"{host_id}:\n  - ExampleService\n", ""
        ),
        encoding="utf-8",
    )


def test_collect_desired_facts_shape(inv_fixture: Path, roles_path: Path):
    facts, summary = ansible_source.collect_desired_facts(inv_fixture, "site-a", roles_path)

    assert summary["hosts"] == 5
    assert summary["host_inventory_facts"] == 5
    assert summary["access_grant_facts"] == 5
    assert summary["host_role_facts"] == 2

    inventory_facts = [f for f in facts if f.fact_type == "host_inventory"]
    access_facts = [f for f in facts if f.fact_type == "access_grant"]
    alias_facts = [f for f in facts if f.fact_type == "host_alias"]
    network_facts = [f for f in facts if f.fact_type == "network_topology"]

    kube_inventory = next(f for f in inventory_facts if f.about[0] == "host-a")
    assert (
        kube_inventory.statement
        == "host-a is a group-a host at 10.0.0.11 (ansible_user root)."
    )
    assert kube_inventory.about == ["host-a", "group-a", "site-a"]

    infra_access = next(f for f in access_facts if f.about[0] == "host-e")
    assert "key-b" in infra_access.statement
    assert infra_access.about == ["host-e", "group-c", "access"]

    hostg_alias = next(f for f in alias_facts if "dashboards" in f.about)
    assert hostg_alias.statement == "host-g is also known as dashboards."
    assert hostg_alias.about == ["host-g", "dashboards"]

    assert any("app.example.org" in f.statement for f in alias_facts)
    assert any("nfs server" in f.statement.lower() for f in network_facts)
    assert any("management network" in f.statement.lower() for f in network_facts)


async def test_ansible_source_update_on_change(inv_fixture: Path, roles_path: Path):
    """A changed host fact must supersede the old one."""
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")
    first_proposals = len(client.proposals)
    assert first_proposals > 0

    # Mutate the inventory IP for host-a. This changes both the host_inventory
    # and access_grant statements for host-a:group-a, so two facts supersede.
    inventory = inv_fixture / "inventory"
    text = inventory.read_text(encoding="utf-8")
    text = text.replace(
        "host-a ansible_host=10.0.0.11 ansible_user=root",
        "host-a ansible_host=10.0.0.20 ansible_user=root",
    )
    inventory.write_text(text, encoding="utf-8")

    # Second run against the same persisted fake.
    client.proposals.clear()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    superseding = [p for p in client.proposals if p.get("supersedes")]
    assert len(superseding) == 2, f"expected two supersedes, got {len(superseding)}"
    for prop in superseding:
        assert "10.0.0.20" in prop["statement"]
        assert prop["about"][0] == "host-a"


async def test_ansible_source_update_role_supersedes_one_fact(
    inv_fixture: Path, roles_path: Path
):
    """Changing only the curated role YAML supersedes exactly one host_role fact."""
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    # Mutate only the host-roles file.
    roles_path.write_text(
        "host-e:\n  - AppServer\n  - Database\nhost-g:\n  - Dashboards\n  - Logs\n",
        encoding="utf-8",
    )

    client.proposals.clear()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    superseding = [p for p in client.proposals if p.get("supersedes")]
    assert len(superseding) == 1, f"expected exactly one supersede, got {len(superseding)}"
    assert superseding[0]["fact_type"] == "host_role"
    assert superseding[0]["about"][0] == "host-g"


async def test_ansible_source_no_duplicate_on_unchanged_run(
    inv_fixture: Path, roles_path: Path
):
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")
    first_count = len(client.proposals)

    client.proposals.clear()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    assert len(client.proposals) == 0
    # Sanity: first run actually proposed something.
    assert first_count > 0


async def test_ansible_source_unknown_key_type_for_group_without_priv(
    inv_fixture: Path, roles_path: Path
):
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    hostc_access = next(
        f for f in client.facts
        if f["fact_type"] == "access_grant" and f["about"][0] == "host-c"
    )
    assert "unknown" in hostc_access["statement"]


async def test_ansible_source_never_opens_denylisted_files(
    inv_fixture: Path, roles_path: Path, monkeypatch
):
    opened_paths: list[Path] = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        opened_paths.append(Path(path).resolve())
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    denylist_names = {"*.priv", "vault", "*access*"}
    for p in opened_paths:
        name = p.name
        for pattern in denylist_names:
            if pattern.startswith("*") and name.endswith(pattern[1:]):
                raise AssertionError(f"opened denylisted file: {p}")
            if name == pattern:
                raise AssertionError(f"opened denylisted file: {p}")


async def test_reaffirm_called_for_every_unchanged_fact(inv_fixture: Path, roles_path: Path):
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")
    first_ids = {f["id"] for f in client.facts}

    client.reaffirmed.clear()
    client.proposals.clear()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    assert client.proposals == []  # nothing superseded or created
    assert set(client.reaffirmed) == first_ids  # every synced fact got a freshness touch


async def test_identity_matching_group_change_supersedes_in_place(
    inv_fixture: Path, roles_path: Path
):
    """A host's group changes: sync_fact's identity fallback finds and supersedes
    the old fact in place - one fact, not a duplicate+orphan - and a subsequent
    reconcile agrees (zero orphans)."""
    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    original = next(
        f for f in client.facts
        if f["fact_type"] == "host_inventory" and f["about"] == ["host-a", "group-a", "site-a"]
    )

    inventory = inv_fixture / "inventory"
    text = inventory.read_text(encoding="utf-8")
    original_block = (
        "[group-a]\n"
        "host-a ansible_host=10.0.0.11 ansible_user=root\n"
        "host-c ansible_host=10.0.0.16 ansible_user=root\n"
    )
    moved_block = (
        "[group-a]\n"
        "host-c ansible_host=10.0.0.16 ansible_user=root\n"
        "\n"
        "[group-z]\n"
        "host-a ansible_host=10.0.0.11 ansible_user=root\n"
    )
    assert original_block in text
    inventory.write_text(text.replace(original_block, moved_block), encoding="utf-8")

    client.proposals.clear()
    await ansible_source.run_ansible_source(client, inv_fixture, roles_path, env="site-a")

    host_inventory_supersedes = [
        p for p in client.proposals
        if p["fact_type"] == "host_inventory" and p.get("supersedes")
    ]
    assert len(host_inventory_supersedes) == 1, (
        "expected exactly one supersede, not duplicate+orphan"
    )
    assert host_inventory_supersedes[0]["supersedes"] == original["id"]
    assert host_inventory_supersedes[0]["about"] == ["host-a", "group-z", "site-a"]
    assert original["status_"] == "superseded"

    # A subsequent reconcile must agree: host-a is fully desired (just
    # relabeled), not an orphan - the two subsystems must not disagree.
    result = await ansible_source.run_reconcile(client, inv_fixture, roles_path, env="site-a")
    host_inventory_orphans = [
        t for t in result["tombstones"] if t["fact_type"] == "host_inventory"
    ]
    assert host_inventory_orphans == []


async def test_reconcile_tombstones_orphan(tmp_path: Path):
    root = tmp_path / "site-b"
    root.mkdir()
    # 8 hosts: 1/8 orphaned stays under the 15% cap
    host_ids = [f"host-{i}" for i in range(1, 9)]
    roles_path = _write_flat_inventory(root, host_ids)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-b")

    # A decoy fact of a different source_kind must never be touched by reconcile.
    client.facts.append({
        "id": "decoy-1",
        "fact_type": "host_inventory",
        "about": ["host-x", "group-a", "site-b"],
        "statement": "hand-seeded fact", "status_": "current", "source_kind": "agent_proposal",
    })

    _drop_host(root, roles_path, "host-1", "10.0.1.1")

    result = await ansible_source.run_reconcile(client, root, roles_path, env="site-b")

    tombstoned_types = {t["fact_type"] for t in result["tombstones"]}
    assert tombstoned_types == {"host_inventory", "access_grant", "host_role", "host_alias"}
    assert len(result["tombstones"]) == 4
    for t in result["tombstones"]:
        assert t["about"][0] == "host-1"
        assert "decommissioned as of" in t["statement"]
        assert not result["per_type"][t["fact_type"]]["aborted"]
        original = next(f for f in client.facts if f["id"] == t["supersedes"])
        assert original["status_"] == "superseded"

    assert result["per_type"]["network_topology"]["owned"] == 0

    # The decoy fact (different source_kind) was never touched.
    decoy = next(f for f in client.facts if f["id"] == "decoy-1")
    assert decoy["status_"] == "current"
    assert "host-x" not in {p["about"][0] for p in client.proposals}


async def test_reconcile_empty_desired_aborts_only_that_type(tmp_path: Path):
    root = tmp_path / "site-c"
    root.mkdir()
    host_ids = [f"host-{i}" for i in range(1, 5)]
    roles_path = _write_flat_inventory(root, host_ids)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-c")

    # Simulate a parser regression: the roles file goes empty, nothing else changes.
    roles_path.write_text("", encoding="utf-8")

    result = await ansible_source.run_reconcile(client, root, roles_path, env="site-c")

    host_role = result["per_type"]["host_role"]
    assert host_role["aborted"] is True
    assert host_role["abort_reason"] == "empty_desired"
    assert host_role["tombstoned"] == 0
    assert not any(t["fact_type"] == "host_role" for t in result["tombstones"])

    # Siblings are untouched by the sparse type's abort - the inventory didn't
    # change, so they see zero orphans and proceed normally.
    for fact_type in ("host_inventory", "access_grant", "host_alias"):
        record = result["per_type"][fact_type]
        assert record["aborted"] is False
        assert record["orphans"] == 0


async def test_reconcile_over_cap_aborts_then_override_proceeds(tmp_path: Path):
    root = tmp_path / "site-d"
    root.mkdir()
    host_ids = [f"host-{i}" for i in range(1, 5)]  # 4 hosts: 1/4 = 25% breaches the 15% cap
    roles_path = _write_flat_inventory(root, host_ids)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-d")

    _drop_host(root, roles_path, "host-1", "10.0.1.1")

    client.proposals.clear()
    aborted = await ansible_source.run_reconcile(client, root, roles_path, env="site-d")
    for fact_type in ("host_inventory", "access_grant", "host_role", "host_alias"):
        record = aborted["per_type"][fact_type]
        assert record["aborted"] is True
        assert record["abort_reason"] == "fraction_cap"
    assert aborted["tombstones"] == []
    assert client.proposals == []  # nothing superseded

    overridden = await ansible_source.run_reconcile(
        client, root, roles_path, env="site-d", allow_override=True,
    )
    for fact_type in ("host_inventory", "access_grant", "host_role", "host_alias"):
        assert overridden["per_type"][fact_type]["aborted"] is False
    assert len(overridden["tombstones"]) == 4


async def test_reconcile_truncation_aborts(tmp_path: Path):
    root = tmp_path / "site-e"
    root.mkdir()
    host_ids = [f"host-{i}" for i in range(1, 4)]  # 3 hosts, nothing removed
    roles_path = _write_flat_inventory(root, host_ids)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-e")

    result = await ansible_source.run_reconcile(
        client, root, roles_path, env="site-e", owned_query_limit=2,
    )

    for fact_type in ("host_inventory", "access_grant", "host_role", "host_alias"):
        record = result["per_type"][fact_type]
        assert record["aborted"] is True
        assert record["abort_reason"] == "truncation"
    assert result["tombstones"] == []


async def test_reconcile_dry_run_emits_no_writes(tmp_path: Path):
    root = tmp_path / "site-f"
    root.mkdir()
    host_ids = [f"host-{i}" for i in range(1, 9)]
    roles_path = _write_flat_inventory(root, host_ids)

    client = _FakeKwimClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-f")

    _drop_host(root, roles_path, "host-1", "10.0.1.1")

    proposals_before = len(client.proposals)
    result = await ansible_source.run_reconcile(
        client, root, roles_path, env="site-f", dry_run=True
    )

    assert len(client.proposals) == proposals_before  # no writes at all
    assert len(result["tombstones"]) == 4  # would-be tombstones still reported
    assert all("id" not in t for t in result["tombstones"])  # never actually created
