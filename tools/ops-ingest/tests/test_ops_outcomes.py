"""After --commit, the gate's decision on each proposal is waited for and
reported."""
from pathlib import Path

import common
import ansible_source
from tests.test_ops_ansible_source import _FakeKwimClient, _write_flat_inventory


class _GateClient(_FakeKwimClient):
    """The fake client, answering like the real API: propose returns a
    proposal_id, and the gate decides each one after `polls_until_decided`
    status reads. Statements containing a word in `held` go to review."""

    def __init__(self, held=(), polls_until_decided=1):
        super().__init__()
        self.held = held
        self.polls_until_decided = polls_until_decided
        self.polls: dict[str, int] = {}
        self.statements: dict[str, str] = {}

    async def propose_fact(self, statement, fact_type, about, decay_class="slow", supersedes=None):
        await super().propose_fact(statement, fact_type, about, decay_class, supersedes)
        pid = f"p-{len(self.statements) + 1}"
        self.statements[pid] = statement
        return {"proposal_id": pid, "status": "accepted"}

    async def proposal_status(self, pid):
        self.polls[pid] = self.polls.get(pid, 0) + 1
        if self.polls[pid] < self.polls_until_decided:
            return {"id": pid, "status": "accepted"}
        if any(w in self.statements[pid] for w in self.held):
            return {"id": pid, "status": "pending_review",
                    "detail": "similar to semantic item runbook:x (d=0.185) - restates existing memory"}
        return {"id": pid, "status": "committed", "detail": "object_id=f seq=1"}


def _inventory(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "site-c"
    root.mkdir()
    return root, _write_flat_inventory(root, ["host-1", "host-2"])


async def test_commit_reports_a_held_fact_with_its_reason(tmp_path: Path, capsys):
    root, roles_path = _inventory(tmp_path)
    client = _GateClient(held=("host-2 runs",), polls_until_decided=2)
    summary = await ansible_source.run_ansible_source(
        client, root, roles_path, env="site-c", wait_s=5)
    out = capsys.readouterr().out
    held = [s for s in client.statements.values() if "host-2 runs" in s]
    assert held, "the fixture should produce a fact the fake holds"
    assert summary["outcomes"]["pending_review"] == len(held)
    assert summary["outcomes"]["committed"] == len(client.statements) - len(held)
    assert f"{len(held)} held for review" in out
    assert f"[held for review] {held[0]}" in out
    assert "restates existing memory" in out
    assert "GET /v1/review/pending" in out


async def test_a_rerun_reports_unchanged_facts_as_skipped(tmp_path: Path, capsys):
    root, roles_path = _inventory(tmp_path)
    client = _GateClient()
    await ansible_source.run_ansible_source(client, root, roles_path, env="site-c", wait_s=5)
    first = len(client.statements)
    capsys.readouterr()
    summary = await ansible_source.run_ansible_source(
        client, root, roles_path, env="site-c", wait_s=5)
    assert len(client.statements) == first          # nothing new proposed
    assert summary["skipped"] == {"unchanged": first}
    assert f"skipped {first}" in capsys.readouterr().out


async def test_await_outcomes_stops_at_the_deadline():
    class _NeverDecides:
        async def proposal_status(self, pid):
            return {"id": pid, "status": "accepted"}

    out = await common.await_outcomes(
        _NeverDecides(), [("p-1", "a fact")], timeout_s=0.05, interval_s=0.01)
    assert out == [{"proposal_id": "p-1", "statement": "a fact", "status": "accepted", "detail": ""}]
    report = common.format_outcomes(out)
    assert "1 not yet decided" in report
    assert "[not yet decided (accepted)] a fact" in report
    assert "GET /v1/proposals/p-1" in report


def test_format_outcomes_lists_only_what_did_not_commit():
    report = common.format_outcomes([
        {"proposal_id": "p-1", "statement": "kept", "status": "committed", "detail": ""},
        {"proposal_id": "p-2", "statement": "dup", "status": "rejected", "detail": "duplicate of fact f"},
    ], skipped={"unchanged": 3})
    assert report.splitlines()[0] == (
        "# gate outcomes: 1 committed, 0 held for review, 1 rejected, 0 already committed, "
        "3 skipped (unchanged)")
    assert "[rejected] dup" in report and "duplicate of fact f" in report
    assert "kept" not in report
