"""Shared ops-adapter utilities: source parsing, and a KWIM HTTP client that
implements read-diff-supersede.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

log = logging.getLogger("ops.common")

# Ansible inventory/host_vars fields we are allowed to read.
_ALLOWED_INVENTORY_KEYS = {
    "ansible_host",
    "ansible_user",
    "ansible_ssh_private_key_file",
}
_ALLOWED_HOST_VARS_KEYS = _ALLOWED_INVENTORY_KEYS

# Never open these, matched case-insensitively against the filename. Extend for
# your own conventions with OPS_DENYLIST_EXTRA (comma-separated globs).
_DENYLIST_GLOBS = {
    "*.priv", "*.pub", "*.key", "*.age", "*.pem",
    "vault", "*vault*",
    "*secret*", "*credential*", "*password*", "*access*",
}

# Regex for a host line key=value token, including simple quoted values.
_HOST_KV_RE = re.compile(r'([a-zA-Z0-9_]+)\s*=\s*("(?:\\.|[^"])*"|\S+)')
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Leak-scan patterns for the docs adapter (fail-closed).
_LEAK_PATTERNS = [
    re.compile(r"password|secret|token|api[_-]?key", re.IGNORECASE),
    re.compile(r"BEGIN .*PRIVATE KEY"),
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),
    # mysql -p<password>, psql -W<password>, etc.
    re.compile(r"\s-[pW][^\s]+"),
    # ansible-vault ciphertext
    re.compile(r"\$ANSIBLE_VAULT"),
]

# The URL comes from KWIM_BASE_URL / --kwim-url.
_DEFAULT_KWIM_URL = "http://localhost:8000"


def fact_identity(fact_type: str, about: list[str]) -> str | tuple[str, str]:
    """Stable per-fact-type identity from `about` - not the full `about`, which
    holds mutable refs. Keys both sides of an orphan-detection diff.
    """
    if fact_type == "host_alias":
        return (about[0], about[1])  # about[0] alone collides across a host's aliases
    if fact_type == "network_topology":
        return about[1]  # about[0] is always the literal "network"
    return about[0]  # host_inventory, access_grant, host_role: the host id


@dataclass
class DesiredFact:
    """A fact the adapter wants KWIM to hold."""

    fact_type: str
    about: list[str]
    statement: str
    decay_class: str = "slow"
    extra: dict[str, Any] = field(default_factory=dict)
    identity: str | tuple[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.identity = fact_identity(self.fact_type, self.about)


class KwimClient:
    """Public KWIM HTTP API client for ops facts. Read-diff-supersede: query
    by (about, fact_type), then propose new, propose with supersedes, or skip.
    """

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def query_facts(self, about: list[str], fact_type: str) -> list[dict[str, Any]]:
        """Return current facts matching all `about` tags and `fact_type`.

        The server semantics for repeated `about` params are not guaranteed,
        so we always filter client-side to require every tag in `about`.
        """
        url = f"{self.base_url}/v1/knowledge/query"
        params: list[tuple[str, str]] = [("fact_type", fact_type), ("status_", "current")]
        for tag in about:
            params.append(("about", tag))
        r = await self.client.get(url, params=params, headers=self.headers)
        r.raise_for_status()
        facts = r.json()
        if not isinstance(facts, list):
            return []
        return [f for f in facts if all(tag in f.get("about", []) for tag in about)]

    async def query_owned(self, fact_type: str, limit: int = 10000) -> list[dict[str, Any]]:
        """Return all current, pipeline-owned facts of `fact_type`.

        Scoped by source_kind + fact_type, not env: env is absent from `about`
        for three of the managed types, so an env filter would return zero.
        Never touches hand-seeded or agent-proposed facts.
        """
        url = f"{self.base_url}/v1/knowledge/query"
        params: list[tuple[str, str]] = [
            ("fact_type", fact_type),
            ("status_", "current"),
            ("source_kind", "repo_sync"),
            ("limit", str(limit)),
        ]
        r = await self.client.get(url, params=params, headers=self.headers)
        r.raise_for_status()
        facts = r.json()
        return facts if isinstance(facts, list) else []

    async def propose_fact(
        self,
        statement: str,
        fact_type: str,
        about: list[str],
        decay_class: str = "slow",
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "statement": statement,
            "fact_type": fact_type,
            "evidence": [],
            "about": about,
            "source_kind": "repo_sync",
            "decay_class": decay_class,
        }
        if supersedes:
            body["supersedes"] = supersedes
        r = await self.client.post(
            f"{self.base_url}/v1/knowledge/propose",
            headers=self.headers,
            json=body,
        )
        r.raise_for_status()
        return r.json()

    async def reaffirm(self, fact_id: str) -> bool:
        """Non-governance freshness touch: stamp last_verified_at = now.

        204 -> True. 404 -> the fact is gone; log and return False, do not raise.
        """
        r = await self.client.post(
            f"{self.base_url}/v1/knowledge/facts/{fact_id}/reaffirm",
            headers=self.headers,
        )
        if r.status_code == 404:
            log.warning("reaffirm: fact %s not found", fact_id)
            return False
        r.raise_for_status()
        return True

    async def sync_fact(self, fact: DesiredFact) -> dict[str, Any]:
        """Read-diff-supersede for a single desired fact.

        Exact full-`about` match first, then an identity-derived read that
        tolerates drift in the other tags. Two groups renaming in one sync can
        still land >1 identity match and skip as multiple_current.
        """
        exact = await self.query_facts(fact.about, fact.fact_type)
        if len(exact) == 1:
            current = exact
        else:
            identity_tags = (
                list(fact.identity) if isinstance(fact.identity, tuple) else [fact.identity]
            )
            current = await self.query_facts(identity_tags, fact.fact_type)
        if not current:
            return await self.propose_fact(
                fact.statement,
                fact.fact_type,
                fact.about,
                fact.decay_class,
            )
        if len(current) > 1:
            ids = [c.get("id") for c in current]
            log.warning(
                "multiple current facts for %s %s: %s; skipping",
                fact.fact_type,
                fact.about,
                ids,
            )
            return {"skipped": True, "reason": "multiple_current", "ids": ids}

        existing = current[0]
        if existing.get("statement") == fact.statement:
            await self.reaffirm(existing.get("id"))
            return {"skipped": True, "reason": "unchanged", "id": existing.get("id")}

        return await self.propose_fact(
            fact.statement,
            fact.fact_type,
            fact.about,
            fact.decay_class,
            supersedes=existing.get("id"),
        )


def kwim_url_from_env() -> str:
    return os.environ.get("KWIM_BASE_URL", _DEFAULT_KWIM_URL)


def kwim_key_from_env() -> str:
    return os.environ.get("DJED_KWIM_API_KEY", "")


def ops_conf_dir_from_env() -> str:
    return os.environ.get("OPS_CONF_DIR", "")


def denylist_globs() -> set[str]:
    """The never-open set, plus anything in OPS_DENYLIST_EXTRA."""
    extra = os.environ.get("OPS_DENYLIST_EXTRA", "")
    return _DENYLIST_GLOBS | {g.strip() for g in extra.split(",") if g.strip()}


def is_denylisted(path: Path) -> bool:
    """Return True if the filename matches a never-open pattern."""
    name = path.name.lower()
    return any(fnmatch.fnmatch(name, g.lower()) for g in denylist_globs())


def safe_list_dir(dir_path: Path) -> list[str]:
    """List filenames in a directory without opening any file."""
    if not dir_path.exists():
        return []
    return [p.name for p in dir_path.iterdir() if p.is_file()]


def key_type_from_priv_files(group_dir: Path) -> str | None:
    """Return the key-type basename if the group dir contains exactly one *.priv file.

    Never opens the file. Returns None if zero or multiple .priv files.
    """
    priv_files = [n for n in safe_list_dir(group_dir) if n.endswith(".priv")]
    if len(priv_files) != 1:
        return None
    return priv_files[0].removesuffix(".priv")


def parse_ansible_inventory(
    inventory_text: str,
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """Parse a simple Ansible INI inventory.

    Returns:
        hosts: dict[host_id, allowed_params]
        groups: dict[group_name, list[host_id]] (preserves order, may contain duplicates)
    """
    hosts: dict[str, dict[str, str]] = {}
    groups: dict[str, list[str]] = {}
    current_group: str | None = None

    section_re = re.compile(r"^\[(.+?)(?::(children))?\]$")

    for raw_line in inventory_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue

        section_match = section_re.match(line)
        if section_match:
            group_name = section_match.group(1)
            is_children = section_match.group(2) == "children"
            if is_children:
                current_group = None
            else:
                current_group = group_name
                groups.setdefault(current_group, [])
            continue

        if current_group is None:
            continue

        parts = line.split()
        if not parts:
            continue

        host_id = parts[0].strip().lower()
        params: dict[str, str] = {}
        for token in parts[1:]:
            m = _HOST_KV_RE.match(token)
            if not m:
                continue
            key = m.group(1)
            val = m.group(2)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if key in _ALLOWED_INVENTORY_KEYS:
                params[key] = val

        # Last entry wins for duplicate host definitions (Ansible semantics).
        hosts[host_id] = params
        groups[current_group].append(host_id)

    return hosts, groups


def parse_hosts_file(hosts_text: str) -> dict[str, list[str]]:
    """Parse /etc/hosts-style file: IP name [alias...].

    Returns dict[ip, list[names]] with names lowercased.
    """
    ip_map: dict[str, list[str]] = {}
    for raw_line in hosts_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
        ip = tokens[0]
        names = [t.lower() for t in tokens[1:]]
        ip_map.setdefault(ip, []).extend(names)
    return ip_map


def load_yaml_safe(path: Path) -> Any:
    """Load a YAML file from an allow-listed path."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_host_roles(path: Path) -> dict[str, list[str]]:
    """Load the curated host-roles YAML.

    Expected shape:
        host_id:
          - Service A
          - Service B
    """
    data = load_yaml_safe(path)
    if not isinstance(data, dict):
        return {}
    roles: dict[str, list[str]] = {}
    for host_id, services in data.items():
        host_id = str(host_id).lower()
        if isinstance(services, list):
            roles[host_id] = [str(s).strip() for s in services if s]
        elif isinstance(services, str):
            roles[host_id] = [str(services).strip()]
        else:
            roles[host_id] = []
    return roles


def load_host_vars(path: Path) -> dict[str, str]:
    """Load a host_vars file and return only allow-listed override keys."""
    data = load_yaml_safe(path)
    if not isinstance(data, dict):
        return {}
    return {k: str(v) for k, v in data.items() if k in _ALLOWED_HOST_VARS_KEYS}


def load_network_config(path: Path) -> dict[str, Any]:
    return load_yaml_safe(path) or {}


def resolve_key_type(
    host_id: str,
    host_params: dict[str, str],
    group_membership: list[str],
    group_vars_root: Path,
) -> str:
    """Resolve a host's SSH key type name. Never opens files, never reads
    group_vars/all.

    1. per-host ansible_ssh_private_key_file basename
    2. the group's group_vars/<group>/*.priv filename, if exactly one
    3. "unknown"
    """
    _ = host_id  # reserved for future logging/disambiguation
    priv_file = host_params.get("ansible_ssh_private_key_file", "")
    if priv_file:
        base = os.path.basename(priv_file)
        if base.endswith(".priv"):
            base = base[:-5]
        return base

    for group in group_membership:
        if group == "all":
            continue
        kt = key_type_from_priv_files(group_vars_root / group)
        if kt:
            return kt

    return "unknown"


def scan_for_secrets(text: str) -> list[str]:
    """Return matched secret-pattern snippets. Empty list means clean."""
    hits: list[str] = []
    for pat in _LEAK_PATTERNS:
        for m in pat.finditer(text):
            snippet = m.group(0)
            if len(snippet) > 60:
                snippet = snippet[:60] + "..."
            hits.append(snippet)
    return hits


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (section_heading, chunk) tuples.

    The first chunk before any heading uses section_heading "intro". If the
    same heading text appears more than once, an occurrence counter is appended
    to keep ids stable but unique per chunk.
    """
    lines = text.splitlines()
    chunks: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    heading_counts: dict[str, int] = {}

    def flush() -> None:
        if current_lines:
            heading = current_heading or "intro"
            heading_counts[heading] = heading_counts.get(heading, 0) + 1
            unique_heading = (
                f"{heading}-{heading_counts[heading]}"
                if heading_counts[heading] > 1
                else heading
            )
            chunks.append((unique_heading, "\n".join(current_lines).strip()))

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()
    return chunks


def section_id(source_doc: str, section: str) -> str:
    """Stable id for a semantic doc chunk."""
    return hashlib.sha256(f"{source_doc}:{section}".encode()).hexdigest()


def parse_teams(args: argparse.Namespace) -> list[str]:
    """Repeatable --team. `add_common_args` marks it required."""
    return list(getattr(args, "team", []) or [])


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--team", action="append", required=True, help="target team (repeatable)")
