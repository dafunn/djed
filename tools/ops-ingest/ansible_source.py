"""Ops ansible adapter - emit host/access/topology facts into KWIM.

Reads only allow-listed files from an inventory checkout, syncing by
read-diff-supersede. Dry run by default: it prints the facts it would sync and
writes nothing until given --commit. Writing needs KWIM_BASE_URL and
DJED_KWIM_API_KEY. After writing it waits (up to --wait seconds) for KWIM's gate
to decide each proposal and prints what it did (committed, held for review,
rejected).

    python -m ansible_source --team <team> --inventory <path> --conf <path> --env <env> [--commit]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from common import (
    DesiredFact,
    KwimClient,
    add_common_args,
    await_outcomes,
    format_outcomes,
    outcome_counts,
    fact_identity,
    kwim_key_from_env,
    kwim_url_from_env,
    load_host_roles,
    load_host_vars,
    load_network_config,
    ops_conf_dir_from_env,
    parse_ansible_inventory,
    parse_hosts_file,
    resolve_key_type,
)

log = logging.getLogger("ops.ansible_source")

# fact_types the reconcile phase owns and sweeps for orphans. Disjoint from the
# fact_types any other repo_sync producer writes, so source_kind=repo_sync +
# fact_type isolates ops facts without needing env.
MANAGED_FACT_TYPES = (
    "host_inventory", "access_grant", "host_role", "host_alias", "network_topology",
)

# Orphans > this fraction of a fact_type's owned set abort that type's reconcile
# (override with --reconcile-override-cap) - protects against a parser regression
# reaping the graph.
FRACTION_CAP = 0.15


def _fmt_network_fact(name: str, data: dict[str, Any]) -> str | None:
    """Build a human-readable statement for a network_topology entry."""
    if name == "nfs":
        server = data.get("server", "")
        hostname = data.get("server_hostname", "")
        media_path = data.get("media_path", "")
        media_mount = data.get("media_mount", "")
        subnet = data.get("subnet", "")
        parts: list[str] = []
        if server and hostname:
            parts.append(f"NFS server is {hostname} ({server})")
        elif server:
            parts.append(f"NFS server is {server}")
        elif hostname:
            parts.append(f"NFS server is {hostname}")
        if subnet:
            parts.append(f"NFS subnet {subnet}")
        if media_path:
            parts.append(f"media path {media_path}")
        if media_mount:
            parts.append(f"mount {media_mount}")
        if not parts:
            return None
        return "; ".join(parts) + "."

    if name == "domains":
        primary = data.get("primary", "")
        alt1 = data.get("alt1", "")
        alt2 = data.get("alt2", "")
        parts = []
        if primary:
            parts.append(f"primary domain {primary}")
        if alt1:
            parts.append(f"alt1 {alt1}")
        if alt2:
            parts.append(f"alt2 {alt2}")
        if not parts:
            return None
        return "Domains: " + "; ".join(parts) + "."

    if "cidr" in data:
        cidr = data.get("cidr", "")
        gateway = data.get("gateway", "")
        vlan = data.get("vlan", "")
        description = data.get("description", "")
        parts = [f"{name} network is {cidr}"]
        if gateway:
            parts.append(f"gateway {gateway}")
        if vlan:
            parts.append(f"VLAN {vlan}")
        if description:
            parts.append(f"({description})")
        return " ".join(parts) + "."

    if "port" in data:
        port = data.get("port")
        return f"{name} service uses port {port}."

    return None


def desired_host_facts(
    env: str,
    host_id: str,
    host_params: dict[str, str],
    groups: list[str],
    roles: dict[str, list[str]],
    group_vars_root: Path,
) -> list[DesiredFact]:
    """Return desired host_inventory, access_grant, and host_role facts."""
    ip = host_params.get("ansible_host", "")
    user = host_params.get("ansible_user", "")
    primary_group = groups[0] if groups else ""
    facts: list[DesiredFact] = []

    # host_inventory: one fact per group membership.
    for group in groups:
        statement = f"{host_id} is a {group} host"
        if ip:
            statement += f" at {ip}"
        if user:
            statement += f" (ansible_user {user})"
        statement += "."
        facts.append(
            DesiredFact(
                fact_type="host_inventory",
                about=[host_id, group, env],
                statement=statement,
            )
        )

    # access_grant: one fact per group membership.
    key_type = resolve_key_type(host_id, host_params, groups, group_vars_root)
    for group in groups:
        statement = f"Reach {host_id} as {user}" if user else f"Reach {host_id}"
        if ip:
            statement += f" at {ip}"
        statement += f" using the {key_type} key type."
        facts.append(
            DesiredFact(
                fact_type="access_grant",
                about=[host_id, group, "access"],
                statement=statement,
            )
        )

    # host_role: one fact per host.
    services = roles.get(host_id, [])
    if services:
        primary_service = services[0]
        services_text = ", ".join(services)
        statement = f"{host_id} runs {services_text}"
        if primary_group:
            statement += f" ({primary_group})"
        statement += "."
        about = (
            [host_id, primary_group, primary_service, env]
            if primary_group
            else [host_id, primary_service, env]
        )
        facts.append(
            DesiredFact(
                fact_type="host_role",
                about=about,
                statement=statement,
            )
        )

    return facts


def desired_alias_facts(
    canonical_hosts: dict[str, dict[str, str]],
    hosts_map: dict[str, list[str]],
) -> list[DesiredFact]:
    """Return desired host_alias facts for IPs/FQDNs/service names from the hosts file."""
    canonical_ids = set(canonical_hosts.keys())
    emitted: set[tuple[str, str]] = set()
    facts: list[DesiredFact] = []

    for host_id, params in canonical_hosts.items():
        ip = params.get("ansible_host", "")
        if not ip:
            continue
        aliases = hosts_map.get(ip, [])
        for alias in aliases:
            alias = alias.lower()
            if alias == host_id:
                continue
            if alias in canonical_ids:
                # Another canonical host's primary name, not an alias.
                continue
            if (host_id, alias) in emitted:
                continue
            emitted.add((host_id, alias))
            facts.append(
                DesiredFact(
                    fact_type="host_alias",
                    about=[host_id, alias],
                    statement=f"{host_id} is also known as {alias}.",
                )
            )

    return facts


def desired_network_facts(
    env: str,
    network: dict[str, Any],
) -> list[DesiredFact]:
    """Return desired network_topology facts from group_vars/all/network.yml."""
    facts: list[DesiredFact] = []

    # NFS block.
    nfs_data: dict[str, Any] = {}
    if network.get("nfs_server"):
        nfs_data["server"] = network.get("nfs_server")
    if network.get("nfs_server_hostname"):
        nfs_data["server_hostname"] = network.get("nfs_server_hostname")
    if network.get("nfs_media_path"):
        nfs_data["media_path"] = network.get("nfs_media_path")
    if network.get("nfs_media_mount"):
        nfs_data["media_mount"] = network.get("nfs_media_mount")
    # Infer NFS subnet from the secondary network if present.
    secondary = (network.get("networks") or {}).get("secondary", {})
    if secondary.get("cidr"):
        nfs_data["subnet"] = secondary.get("cidr")
    if nfs_data:
        statement = _fmt_network_fact("nfs", nfs_data)
        if statement:
            about = ["network", "nfs"]
            if env:
                about.append(env)
            facts.append(
                DesiredFact(
                    fact_type="network_topology",
                    about=about,
                    statement=statement,
                )
            )

    # Networks (subnets/CIDRs/VLANs/gateways).
    networks = network.get("networks") or {}
    for net_name, net_data in networks.items():
        if isinstance(net_data, dict):
            statement = _fmt_network_fact(net_name, net_data)
            if statement:
                about = ["network", net_name]
                if env:
                    about.append(env)
                facts.append(
                    DesiredFact(
                        fact_type="network_topology",
                        about=about,
                        statement=statement,
                    )
                )

    # Domains block.
    domains = network.get("domains") or {}
    if isinstance(domains, dict) and domains:
        statement = _fmt_network_fact("domains", domains)
        if statement:
            about = ["network", "domains"]
            if env:
                about.append(env)
            facts.append(
                DesiredFact(
                    fact_type="network_topology",
                    about=about,
                    statement=statement,
                )
            )

    # Service ports.
    service_ports = network.get("service_ports") or {}
    for svc_name, port in service_ports.items():
        if isinstance(port, int):
            statement = _fmt_network_fact(svc_name, {"port": port})
            if statement:
                about = ["network", svc_name]
                if env:
                    about.append(env)
                facts.append(
                    DesiredFact(
                        fact_type="network_topology",
                        about=about,
                        statement=statement,
                    )
                )

    return facts


def collect_desired_facts(
    path: Path,
    env: str,
    roles_path: Path,
) -> tuple[list[DesiredFact], dict[str, int]]:
    """Read the ansible inventory tree and return all desired facts plus a summary."""
    inventory_path = path / "inventory"
    hosts_path = path / "hosts"
    host_vars_dir = path / "host_vars"
    group_vars_root = path / "playbooks" / "group_vars"
    network_path = group_vars_root / "all" / "network.yml"

    inventory_text = inventory_path.read_text(encoding="utf-8")
    hosts_text = hosts_path.read_text(encoding="utf-8")

    hosts, groups = parse_ansible_inventory(inventory_text)
    hosts_map = parse_hosts_file(hosts_text)
    roles = load_host_roles(roles_path)
    network = load_network_config(network_path) if network_path.exists() else {}

    # Apply host_vars overrides for allow-listed keys.
    for host_id in hosts:
        hv_path = host_vars_dir / f"{host_id}.yml"
        if hv_path.exists():
            overrides = load_host_vars(hv_path)
            hosts[host_id].update(overrides)

    # Build host -> groups mapping, deduped.
    host_groups: dict[str, list[str]] = {hid: [] for hid in hosts}
    for group, members in groups.items():
        for hid in members:
            if hid in hosts and group not in host_groups[hid]:
                host_groups[hid].append(group)

    summary = {
        "hosts": len(hosts),
        "host_inventory_facts": 0,
        "host_role_facts": 0,
        "access_grant_facts": 0,
        "host_alias_facts": 0,
        "network_topology_facts": 0,
    }

    facts: list[DesiredFact] = []

    for host_id, params in hosts.items():
        grps = host_groups[host_id]
        host_facts = desired_host_facts(env, host_id, params, grps, roles, group_vars_root)
        facts.extend(host_facts)
        summary["host_inventory_facts"] += len(
            [f for f in host_facts if f.fact_type == "host_inventory"]
        )
        summary["access_grant_facts"] += len(
            [f for f in host_facts if f.fact_type == "access_grant"]
        )
        summary["host_role_facts"] += len(
            [f for f in host_facts if f.fact_type == "host_role"]
        )

    alias_facts = desired_alias_facts(hosts, hosts_map)
    facts.extend(alias_facts)
    summary["host_alias_facts"] = len(alias_facts)

    net_facts = desired_network_facts(env, network)
    facts.extend(net_facts)
    summary["network_topology_facts"] = len(net_facts)

    return facts, summary


async def run_ansible_source(
    client: KwimClient | None,
    path: Path,
    roles_path: Path,
    env: str,
    dry_run: bool = False,
    wait_s: float = 30.0,
) -> dict[str, Any]:
    """Read the ansible inventory tree and sync ops facts into KWIM.

    dry_run=True prints proposals and makes no network calls; client may be None.
    Otherwise, after proposing, waits up to `wait_s` for the gate to decide each
    proposal and prints what it did (committed, held for review, rejected), with
    the reason for each that did not commit. `wait_s=0` skips the wait.
    """
    facts, summary = collect_desired_facts(path, env, roles_path)

    if dry_run:
        print(f"# Dry run - {len(facts)} facts would be synced (no writes)\n")
        for fact in sorted(facts, key=lambda f: (f.fact_type, f.about[0] if f.about else "")):
            print(f"[{fact.fact_type}] about={fact.about}")
            print(f"    {fact.statement}")
        print(f"\n# summary: {summary}")
        return summary

    proposed: list[tuple[str, str]] = []
    skipped: dict[str, int] = {}
    for fact in facts:
        result = await client.sync_fact(fact)
        if result.get("skipped"):
            reason = result.get("reason", "skipped")
            skipped[reason] = skipped.get(reason, 0) + 1
        elif result.get("proposal_id"):
            proposed.append((result["proposal_id"], fact.statement))

    if proposed and wait_s > 0:
        outcomes = await await_outcomes(client, proposed, timeout_s=wait_s)
        print(format_outcomes(outcomes, skipped))
        summary["outcomes"] = outcome_counts(outcomes)
    else:
        print(f"# proposed {len(proposed)}, skipped {sum(skipped.values())} {skipped or ''}".rstrip())
    summary["skipped"] = skipped

    log.info("ansible_source done: %s", summary)
    return summary


def compute_orphans(
    fact_type: str, owned: list[dict[str, Any]], desired_of_type: list[DesiredFact],
) -> list[dict[str, Any]]:
    """Owned facts of `fact_type` whose stable identity no longer appears in the
    desired set - i.e. the entity disappeared from the source of truth."""
    desired_identities = {f.identity for f in desired_of_type}
    return [
        o for o in owned
        if fact_identity(fact_type, o.get("about", [])) not in desired_identities
    ]


async def run_reconcile(
    client: KwimClient,
    path: Path,
    roles_path: Path,
    env: str,
    dry_run: bool = False,
    allow_override: bool = False,
    owned_query_limit: int = 10000,
    wait_s: float = 30.0,
) -> dict[str, Any]:
    """Tombstone owned facts whose entity left the desired set.

    Guard-railed per fact_type: an over-cap type blocks only itself. Tombstones
    auto-commit with no human review gate, so these guards are the only
    protection.
    """
    facts, _ = collect_desired_facts(path, env, roles_path)
    desired_by_type: dict[str, list[DesiredFact]] = {t: [] for t in MANAGED_FACT_TYPES}
    for fact in facts:
        if fact.fact_type in desired_by_type:
            desired_by_type[fact.fact_type].append(fact)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    per_type: dict[str, dict[str, Any]] = {}
    tombstones: list[dict[str, Any]] = []
    tombstone_ids: list[tuple[str, str]] = []   # (proposal_id, statement), for the report

    for fact_type in MANAGED_FACT_TYPES:
        owned = await client.query_owned(fact_type, limit=owned_query_limit)
        desired_of_type = desired_by_type[fact_type]
        record: dict[str, Any] = {
            "owned": len(owned), "desired": len(desired_of_type), "orphans": 0,
            "aborted": False, "abort_reason": None, "tombstoned": 0,
        }

        if len(owned) >= owned_query_limit:
            record["aborted"] = True
            record["abort_reason"] = "truncation"
            log.error(
                "reconcile abort %s: owned set hit the query limit (%d), possibly truncated; "
                "not reconciling",
                fact_type, owned_query_limit,
            )
            per_type[fact_type] = record
            continue

        orphans = compute_orphans(fact_type, owned, desired_of_type)
        record["orphans"] = len(orphans)

        if not desired_of_type and owned:
            record["aborted"] = True
            record["abort_reason"] = "empty_desired"
            log.error(
                "reconcile abort %s: desired set is empty but %d fact(s) are "
                "currently owned; treated as a parse error, no tombstones "
                "emitted",
                fact_type, len(owned),
            )
            per_type[fact_type] = record
            continue

        if owned and (len(orphans) / len(owned)) > FRACTION_CAP and not allow_override:
            record["aborted"] = True
            record["abort_reason"] = "fraction_cap"
            log.error(
                "reconcile abort %s: %d/%d owned facts (%.0f%%) orphaned, over "
                "the %.0f%% cap - pass --reconcile-override-cap to force",
                fact_type, len(orphans), len(owned),
                100 * len(orphans) / len(owned), 100 * FRACTION_CAP,
            )
            per_type[fact_type] = record
            continue

        for orphan in orphans:
            entity = fact_identity(fact_type, orphan.get("about", []))
            if isinstance(entity, tuple):
                entity = entity[0]
            was = orphan.get("statement", "")
            statement = f"{entity} decommissioned as of {today} (was: {was})"
            about = orphan.get("about", [])
            if dry_run:
                tombstones.append({
                    "statement": statement, "fact_type": fact_type,
                    "about": about, "supersedes": orphan.get("id"),
                })
            else:
                proposed = await client.propose_fact(
                    statement, fact_type, about, supersedes=orphan.get("id"),
                )
                tombstones.append(proposed)
                if proposed.get("proposal_id"):
                    tombstone_ids.append((proposed["proposal_id"], statement))
            record["tombstoned"] += 1

        per_type[fact_type] = record

    if dry_run:
        print("# Reconcile dry run - no tombstones written\n")
        for fact_type, record in per_type.items():
            reason = f" ({record['abort_reason']})" if record["abort_reason"] else ""
            print(f"[{fact_type}] owned={record['owned']} desired={record['desired']} "
                  f"orphans={record['orphans']} aborted={record['aborted']}{reason}")
        for t in tombstones:
            print(
                f"    would propose: [{t['fact_type']}] about={t['about']} "
                f"supersedes={t['supersedes']}"
            )
            print(f"        {t['statement']}")
        print(f"\n# per_type: {per_type}")

    elif tombstone_ids and wait_s > 0:
        print(format_outcomes(await await_outcomes(client, tombstone_ids, timeout_s=wait_s)))

    log.info("reconcile done: %s", per_type)
    return {"per_type": per_type, "tombstones": tombstones}


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    teams = [t.strip() for t in args.team if t.strip()]
    if not teams:
        log.error("--team is required")
        return 2

    inventory = Path(args.inventory)
    if not inventory.exists():
        log.error("inventory path does not exist: %s", inventory)
        return 2

    if not args.conf:
        log.error("--conf or OPS_CONF_DIR is required")
        return 2
    conf_dir = Path(args.conf)
    if not conf_dir.exists():
        log.error("conf dir does not exist: %s", conf_dir)
        return 2
    roles_path = conf_dir / "ops-host-roles.yaml"

    if args.commit and args.dry_run:
        log.error("--commit and --dry-run contradict each other; pass one")
        return 2
    dry_run = not args.commit

    if dry_run and not args.reconcile:
        # No network, no key needed - just print what would be synced.
        await run_ansible_source(None, inventory, roles_path, env=args.env, dry_run=True)
        return 0

    # --reconcile needs read access even in a dry run (it queries owned
    # facts from KWIM to compute the orphan set) - only the writes are skipped.
    api_key = args.key or kwim_key_from_env()
    if not api_key:
        log.error("--key or DJED_KWIM_API_KEY is required")
        return 2

    async with httpx.AsyncClient() as client:
        kwim = KwimClient(client, args.kwim_url, api_key)
        for team in teams:
            if not dry_run:
                await run_ansible_source(kwim, inventory, roles_path, env=args.env,
                                         wait_s=args.wait)
            else:
                await run_ansible_source(
                    None, inventory, roles_path, env=args.env, dry_run=True
                )
            if args.reconcile:
                await run_reconcile(
                    kwim, inventory, roles_path, env=args.env,
                    dry_run=dry_run, allow_override=args.reconcile_override_cap,
                    wait_s=args.wait,
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ansible_source", description="KWIM ops ansible adapter")
    add_common_args(ap)
    ap.add_argument(
        "--inventory", required=True, help="path to the ansible inventory checkout"
    )
    ap.add_argument("--conf", default=ops_conf_dir_from_env(),
                    help="path to the ops conf dir (or set OPS_CONF_DIR)")
    ap.add_argument("--env", required=True, help="coarse environment label")
    ap.add_argument("--kwim-url", default=kwim_url_from_env(), help="KWIM service base URL")
    ap.add_argument("--key", default=kwim_key_from_env(), help="DJED KWIM API key")
    ap.add_argument("--commit", action="store_true",
                    help="write the facts to KWIM; without it this is a dry run that prints "
                         "what would be synced (no network, no key required unless combined "
                         "with --reconcile)")
    ap.add_argument("--wait", type=float, default=30.0,
                    help="with --commit, seconds to wait for the gate's decision on each "
                         "proposal before reporting it (default 30; 0 reports without waiting)")
    ap.add_argument("--dry-run", action="store_true",
                    help="the default; accepted so existing invocations still run without writing")
    ap.add_argument("--reconcile", action="store_true",
                    help="also run the decommission reconcile phase "
                         "(tombstones orphaned facts); off by default")
    ap.add_argument("--reconcile-override-cap", action="store_true",
                    help="allow a reconcile fraction-cap guard trip to proceed anyway "
                         "(does not override the empty-desired or truncation guards)")
    args = ap.parse_args(argv)
    if args.conf:
        args.conf = os.path.expanduser(args.conf)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
