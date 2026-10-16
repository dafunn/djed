"""Unit tests for seed_guardrails."""
import os
import re
from pathlib import Path

import pytest

import seed_guardrails
from common import ops_conf_dir_from_env


@pytest.fixture
def rules_path(tmp_path: Path):
    path = tmp_path / "ops-guardrails.yaml"
    path.write_text(
        "rules:\n"
        "  - id: ops-test\n"
        "    rule_type: constraint\n"
        "    action_pattern: '(?i)dangerous'\n"
        "    verdict: deny\n"
        "    authority: operator\n"
        "    severity: high\n"
        "    check_tier: deterministic\n",
        encoding="utf-8",
    )
    return path


class _FakeHttpxClient:
    def __init__(self):
        self.posts: list[tuple] = []

    async def post(self, url, *, headers=None, json=None):
        self.posts.append((url, headers, json))
        return _FakeResponse(201)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = "ok"

    def raise_for_status(self):
        pass

    def json(self):
        return {"status": "seeded", "rule_id": "ops-test", "seq": 1}


async def test_load_guardrails(rules_path: Path):
    rules = seed_guardrails.load_guardrails(rules_path)
    assert len(rules) == 1
    assert rules[0]["id"] == "ops-test"
    assert rules[0]["verdict"] == "deny"


async def test_seed_rules_dry_run_does_not_post(rules_path: Path):
    rules = seed_guardrails.load_guardrails(rules_path)
    client = _FakeHttpxClient()
    summary = await seed_guardrails.seed_rules(
        client, "http://kwim.test", "key", rules, commit=False
    )
    assert summary["posted"] == 1
    assert summary["skipped"] == 0
    assert client.posts == []


async def test_seed_rules_commit_posts_each_rule(rules_path: Path, monkeypatch):
    rules = seed_guardrails.load_guardrails(rules_path)
    client = _FakeHttpxClient()
    summary = await seed_guardrails.seed_rules(
        client, "http://kwim.test", "key", rules, commit=True
    )
    assert summary["posted"] == 1
    assert summary["skipped"] == 0
    assert len(client.posts) == 1
    url, headers, payload = client.posts[0]
    assert url == "http://kwim.test/v1/wisdom/seed"
    assert headers["Authorization"] == "Bearer key"
    assert payload["id"] == "ops-test"
    assert payload["rule_type"] == "constraint"
    assert payload["verdict"] == "deny"
    assert payload["action_pattern"] == "(?i)dangerous"
    assert "situation" not in payload
    assert "approach" not in payload


async def test_seed_rules_skips_rule_without_id():
    client = _FakeHttpxClient()
    summary = await seed_guardrails.seed_rules(
        client, "http://kwim.test", "key", [{"rule_type": "constraint"}], commit=True
    )
    assert summary["posted"] == 0
    assert summary["skipped"] == 1
    assert client.posts == []


def _resolve(rules, action):
    """Mimic /wisdom/check resolution: highest-severity match wins."""
    matches = []
    for rule in rules:
        pat = rule.get("action_pattern", "")
        if pat and re.search(pat, action):
            matches.append(rule)
    if not matches:
        return "allow"
    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    best = max(matches, key=lambda r: severity_rank.get(r.get("severity", "low"), 0))
    if best.get("severity") == "critical":
        return "escalate"
    return best.get("verdict")




# Asserted against the shipped ops-guardrails.yaml.
PINNED_RULE_IDS = ("ops-no-force-delete", "ops-pod-delete-escalate")

# One action corpus, shared by the fixture-driven tests and the drift guard, so
# the shipped YAML is held to exactly the behaviour the fixture asserts.
FLUX_FORCE_ACTIONS = [
    "flux reconcile kustomization djed --force",
    "flux reconcile source git platform-conf --force",
    "flux reconcile helmrelease frontend --force",
    "flux reconcile --force",
    "FLUX RECONCILE KUSTOMIZATION DJED --FORCE",
]
FLUX_SAFE_ACTIONS = [
    "flux reconcile kustomization djed",
    "flux reconcile source git platform-conf",
    "flux get kustomizations",
    "flux suspend kustomization djed",
]
FORCE_DELETE_ACTION = "kubectl delete pod nginx-xyz --force --grace-period=0"
PLAIN_DELETE_ACTION = "kubectl delete pod nginx-xyz"

EXPECTED_VERDICTS = (
    [(a, "deny") for a in FLUX_FORCE_ACTIONS]
    + [(a, "allow") for a in FLUX_SAFE_ACTIONS]
    + [(FORCE_DELETE_ACTION, "deny"), (PLAIN_DELETE_ACTION, "escalate")]
)


@pytest.fixture
def force_delete_rules():
    """The pinned pair from ops-guardrails.yaml: ops-no-force-delete (deny/high)
    plus ops-pod-delete-escalate (escalate/medium).

    This is a hand-copy of rules that ship from the ops conf repo. Keep it
    byte-identical to that YAML - test_fixture_matches_shipped_guardrail_yaml
    diffs the two whenever OPS_CONF_DIR is resolvable.
    """
    return [
        {
            "id": "ops-no-force-delete",
            "action_pattern": r"(?i)(kubectl\s+delete\s+pod.*(?:--force|--grace-period[= ]0|--now)|flux\s+reconcile\s+.*--force)",
            "verdict": "deny",
            "severity": "high",
        },
        {
            "id": "ops-pod-delete-escalate",
            "action_pattern": r"(?i)kubectl\s+delete\s+pod",
            "verdict": "escalate",
            "severity": "medium",
        },
    ]


def test_force_delete_matches_both_rules_and_denies(force_delete_rules):
    assert _resolve(force_delete_rules, FORCE_DELETE_ACTION) == "deny"


def test_plain_delete_escalates(force_delete_rules):
    assert _resolve(force_delete_rules, PLAIN_DELETE_ACTION) == "escalate"


@pytest.mark.parametrize("action", FLUX_FORCE_ACTIONS)
def test_flux_force_reconcile_denies(force_delete_rules, action):
    """The flux arm of ops-no-force-delete. It is a separate alternation from the
    kubectl arm, so a force reconcile matches only the deny rule - no escalate to
    outrank, and none needed.
    """
    assert _resolve(force_delete_rules, action) == "deny"


@pytest.mark.parametrize("action", FLUX_SAFE_ACTIONS)
def test_flux_without_force_is_allowed(force_delete_rules, action):
    """The flux arm needs --force: a routine reconcile is allowed."""
    assert _resolve(force_delete_rules, action) == "allow"


def _shipped_rules() -> dict[str, dict]:
    """The curated guardrails the operator actually seeds, keyed by rule id.

    The YAML lives in a separate conf repo, so this resolves through OPS_CONF_DIR
    and skips where that checkout is absent - CI included.
    """
    conf = ops_conf_dir_from_env()
    if not conf:
        pytest.skip("OPS_CONF_DIR unset - no shipped ops-guardrails.yaml to diff against")
    path = Path(os.path.expanduser(conf)) / "ops-guardrails.yaml"
    if not path.is_file():
        pytest.skip(f"no ops-guardrails.yaml under {path.parent}")
    shipped = {r["id"]: r for r in seed_guardrails.load_guardrails(path) if r.get("id")}
    missing = [rid for rid in PINNED_RULE_IDS if rid not in shipped]
    if missing:
        pytest.fail(f"{path} no longer defines pinned rule(s): {', '.join(missing)}")
    return shipped


def test_fixture_matches_shipped_guardrail_yaml(force_delete_rules):
    """Drift guard: this file's copy of the pinned rules matches the YAML that
    seed_guardrails.py posts.
    """
    shipped = _shipped_rules()
    drifted = []
    for rule in force_delete_rules:
        for field in ("action_pattern", "verdict", "severity"):
            mine, theirs = rule[field], shipped[rule["id"]].get(field)
            if mine != theirs:
                drifted.append(
                    f"  {rule['id']}.{field}\n"
                    f"    fixture: {mine!r}\n"
                    f"    shipped: {theirs!r}"
                )
    assert not drifted, (
        "the pinned rules in this file no longer match ops-guardrails.yaml:\n"
        + "\n".join(drifted)
    )


@pytest.mark.parametrize("action,expected", EXPECTED_VERDICTS)
def test_shipped_guardrail_yaml_resolves_as_specified(action, expected):
    """The corpus gets the expected verdicts from the patterns actually seeded."""
    shipped = _shipped_rules()
    assert _resolve([shipped[rid] for rid in PINNED_RULE_IDS], action) == expected


def test_the_documented_usage_matches_the_parser():
    """The tenant comes from the key: the usage line must not offer --team, and
    the parser rejects it."""
    usage = [ln for ln in seed_guardrails.__doc__.splitlines() if "python -m seed_guardrails" in ln]
    assert usage and "--team" not in usage[0]
    with pytest.raises(SystemExit) as exc:
        seed_guardrails.main(["--team", "team-a", "--conf", "/nonexistent"])
    assert exc.value.code == 2
