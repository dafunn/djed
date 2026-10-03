"""Consumer side of the wire contract: the driver parses and classifies every
header exec-mcp's `wire_contract.toml` declares, grammar included.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

import build
from djed.tools import Deps
from tests.test_build import StubKwim

# The contract is read from the exec-mcp image, extracted by
# `tools/check-wire-contract.sh`. Without it these tests skip.
_CONTRACT_ENV = os.environ.get("DJED_WIRE_CONTRACT")
CONTRACT = Path(_CONTRACT_ENV) if _CONTRACT_ENV else None

pytestmark = pytest.mark.skipif(
    CONTRACT is None or not CONTRACT.is_file(),
    reason="run ./tools/check-wire-contract.sh - it reads the contract from the exec image",
)

# Must match the producer side's VALUES; both sides share the templates.
VALUES = {"returncode": 3, "timeout_s": 45, "signal_name": "SIGKILL",
          "output": "systemd-analyze: No such file or directory"}


def _cases():
    if CONTRACT is None or not CONTRACT.is_file():
        return []  # collected but skipped; parametrize still needs a list
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))["case"]


def _line(case, name="demo-task"):
    return f"{case['prefix']} {name}: {case['short'].format(**VALUES)}"


def _sink():
    return build._OutcomeSink(Deps(kwim=StubKwim(), session_id="s"))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["status"])
def test_every_declared_header_parses(case):
    name, short = build._parse_task_header(_line(case) + "\nrest of the output")
    assert name == "demo-task"
    assert short == case["short"].format(**VALUES)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["status"])
def test_the_driver_classifies_each_header_as_the_contract_says(case):
    """Any completion, a failure included, resets the inconclusive count."""
    sink = _sink()
    sink._observe(_line(case))
    assert (sink._pending is not None) is case["inconclusive"]


def test_two_inconclusive_in_a_row_force_a_block():
    inconclusive = [c for c in _cases() if c["inconclusive"]]
    assert len(inconclusive) >= 2, "the two-strikes rule needs two shapes"
    sink = _sink()
    sink._observe(_line(inconclusive[0], name="first"))
    assert sink._deps.outcome is None, "one is not yet a block"
    sink._observe(_line(inconclusive[1], name="second"))
    outcome = sink._deps.outcome
    assert outcome is not None and outcome.kind == "blocked"
    # Both are named: with two different shapes the reason alone misleads.
    assert "second" in outcome.detail and "first" in outcome.detail


def test_a_completion_between_two_inconclusive_resets_the_count():
    inconclusive = [c for c in _cases() if c["inconclusive"]]
    completion = next(c for c in _cases() if not c["inconclusive"])
    sink = _sink()
    sink._observe(_line(inconclusive[0]))
    sink._observe(_line(completion))
    sink._observe(_line(inconclusive[1]))
    assert sink._deps.outcome is None, "a completion is evidence the machinery works"
