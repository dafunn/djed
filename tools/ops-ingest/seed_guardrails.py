"""Post curated :Rule constraints to /v1/wisdom/seed. Idempotent on rule id.

Dry-run by default; --commit writes. Needs KWIM_BASE_URL and DJED_KWIM_API_KEY.
The rules go to the team the key belongs to; there is no --team.

    python -m seed_guardrails --conf <path> [--commit]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import yaml

from common import (
    is_denylisted,
    kwim_key_from_env,
    kwim_url_from_env,
    ops_conf_dir_from_env,
)

log = logging.getLogger("ops.seed_guardrails")


def load_guardrails(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"guardrail YAML root must be a mapping: {path}")
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError(f"guardrail YAML 'rules' must be a list: {path}")
    return rules


async def seed_rules(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    rules: list[dict[str, Any]],
    *,
    commit: bool,
) -> dict[str, int]:
    """Post each rule. The tenant comes from the key, not from an argument."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    posted = 0
    skipped = 0
    for rule in rules:
        rule_id = rule.get("id", "<missing-id>")
        if not rule.get("id"):
            log.warning("skipping rule without id")
            skipped += 1
            continue
        payload = {
            "id": rule["id"],
            "rule_type": rule.get("rule_type", "constraint"),
            "situation": rule.get("situation"),
            "approach": rule.get("approach"),
            "evidence_count": rule.get("evidence_count", 0),
            "action_pattern": rule.get("action_pattern"),
            "verdict": rule.get("verdict"),
            "authority": rule.get("authority"),
            "severity": rule.get("severity"),
            "check_tier": rule.get("check_tier"),
        }
        # Remove None optional fields to keep payload minimal.
        payload = {k: v for k, v in payload.items() if v is not None}
        url = f"{base_url}/v1/wisdom/seed"
        if commit:
            r = await client.post(url, headers=headers, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError:
                log.error(
                    "failed to seed rule %s: %s - %s", rule_id, r.status_code, r.text
                )
                raise
            log.info("seeded rule %s (seq=%s)", rule_id, r.json().get("seq"))
            posted += 1
        else:
            log.info("[dry-run] would POST %s: %s", url, payload)
            posted += 1
    return {"posted": posted, "skipped": skipped}


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if not args.conf:
        log.error("--conf or OPS_CONF_DIR is required")
        return 2
    rules_path = Path(args.conf) / "ops-guardrails.yaml"
    # Refuse a denylisted filename before opening it, as seed_runbook does.
    if is_denylisted(rules_path):
        log.error("refusing to read denylisted file: %s", rules_path)
        return 2
    rules = load_guardrails(rules_path)
    if not rules:
        log.warning("no rules found in %s", rules_path)
        return 0

    api_key = args.key or kwim_key_from_env()
    if not api_key:
        log.error("--key or DJED_KWIM_API_KEY is required")
        return 2

    if not args.commit:
        log.warning("Dry run - no rules seeded. Review the YAML and pass --commit to seed.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        summary = await seed_rules(
            client,
            base_url=args.kwim_url,
            key=api_key,
            rules=rules,
            commit=args.commit,
        )
        log.info("summary: %s", summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="seed_guardrails", description="KWIM ops guardrail seed"
    )
    ap.add_argument("--kwim-url", default=kwim_url_from_env(), help="KWIM service base URL")
    ap.add_argument("--key", default=kwim_key_from_env(), help="DJED KWIM API key")
    ap.add_argument("--conf", default=ops_conf_dir_from_env(),
                    help="path to the ops conf dir (or set OPS_CONF_DIR)")
    ap.add_argument(
        "--commit", action="store_true", help="actually POST rules (dry-run default)"
    )
    args = ap.parse_args(argv)
    if args.conf:
        args.conf = os.path.expanduser(args.conf)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
