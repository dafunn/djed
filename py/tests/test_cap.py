"""The MCP result cap: what an oversized result is replaced with, and that the
replacement says what was withheld.
"""
from __future__ import annotations

import json

import pytest

from djed.cap import (
    DEFAULT_MAX_MCP_OUTPUT,
    cap_result,
    cap_text,
    max_mcp_output,
    outline,
)

PY_SRC = "\n".join(
    ['"""Module."""', "import os", ""]
    + [f"def function_{i}(a, b):\n    return a + b + {i}\n" for i in range(400)]
)


def test_a_small_result_passes_through_untouched():
    assert cap_text("short", "fs_read", {"path": "/w/a.py"}) == "short"


def test_an_oversized_source_read_comes_back_as_structure():
    out = cap_text(PY_SRC, "fs_read", {"path": "/w/big.py"})
    assert len(out) < len(PY_SRC)
    assert "too large to return whole" in out
    # The outline is bounded, so a 400-definition file is reported as truncated.
    assert "def function_0" in out and "outline truncated" in out
    # Each entry carries its line number, for a ranged re-read.
    assert ": def function_0" in out
    assert "fs_read(path, start_line=" in out


def test_the_outline_reports_the_real_line_count():
    out = cap_text(PY_SRC, "fs_read", {"path": "/w/big.py"})
    assert f"is {len(PY_SRC.splitlines())} lines" in out


@pytest.mark.parametrize(
    "suffix, line",
    [
        (".py", "def handler(x):"),
        (".ts", "export function handler(x) {"),
        (".go", "func Handler(x int) {"),
        (".rs", "pub fn handler(x: u32) {"),
        (".rb", "def handler(x)"),
        (".md", "## A heading"),
    ],
)
def test_definitions_are_recognised_per_language(suffix, line):
    text = "\n".join([line.replace("handler", f"handler{i}") for i in range(5)])
    found = outline(text, suffix)
    assert found is not None
    assert len(found.lines) >= 3


def test_an_unoutlinable_file_is_head_clipped_with_its_size():
    """A capped result names what it withheld and how to get it."""
    blob = "\n".join(f"row {i},value,{i * 7}" for i in range(4000))
    out = cap_text(blob, "fs_read", {"path": "/w/data.csv"})
    assert "truncated" in out
    assert "of 4000 lines" in out
    assert out.startswith("row 0,")
    # Whole lines only: half a line is noise.
    assert "\n\n[... truncated" in out


def test_a_file_with_too_few_definitions_falls_back_to_the_head():
    """A one-entry outline is not used; the result is head-clipped instead."""
    text = "def only_one():\n" + "\n".join("    x = 1" for _ in range(4000))
    out = cap_text(text, "fs_read", {"path": "/w/thin.py"})
    assert "truncated" in out and "structure" not in out


def test_each_tool_is_told_how_to_ask_again():
    # Comfortably over the limit, so the guard above does not return it whole.
    big = "x\n" * DEFAULT_MAX_MCP_OUTPUT
    assert "Narrow the pattern" in cap_text(big, "fs_grep", {"path": "/w"})
    assert "depth=1" in cap_text(big, "fs_list", {"path": "/w"})
    assert "narrower pattern" in cap_text(big, "fs_glob", {"path": "/w"})
    assert "narrower slice" in cap_text(big, "pg_query", {})


def test_a_renamed_tool_is_still_recognised():
    """A name collision renames the tool on every server that has it."""
    out = cap_text(PY_SRC, "base__fs_read", {"path": "/w/big.py"})
    assert "structure" in out


def test_structure_that_fits_is_left_exactly_as_it_is():
    """Reshaping a result that would have fitted is pure loss."""
    payload = {"rows": [{"id": 1}, {"id": 2}]}
    assert cap_result(payload, "pg_query", {}) is payload
    assert cap_result(12345, "pg_query", {}) == 12345


def test_json_is_never_clipped_mid_document():
    """Wide row-shaped results are rendered as whole rows of text, never clipped JSON."""
    payload = {"rows": [{"id": i, "note": "y" * 200} for i in range(200)]}
    out = cap_result(payload, "pg_query", {})
    assert isinstance(out, str)
    assert len(out) <= DEFAULT_MAX_MCP_OUTPUT
    assert "of 200 rows" in out
    # Every row that was shown is whole - nothing half-parsed is produced.
    for line in out.splitlines():
        if line.startswith("{"):
            json.loads(line)


def test_a_single_row_over_the_budget_says_so_rather_than_showing_nothing():
    payload = {"rows": ["x" * (DEFAULT_MAX_MCP_OUTPUT * 2)]}
    out = cap_result(payload, "pg_query", {})
    assert "the first one alone is over the budget" in out


def test_a_list_result_has_its_text_items_capped():
    items = ["short", "y\n" * DEFAULT_MAX_MCP_OUTPUT, 7]
    out = cap_result(items, "fs_read", {"path": "/w/a.txt"})
    assert out[0] == "short"
    assert len(out[1]) < len(items[1])
    assert out[2] == 7


def test_capping_never_makes_a_result_larger():
    """A result only slightly over the limit is returned whole: the footer is about
    130 bytes.
    """
    for over in (1, 10, 100, 200):
        blob = "z" * (DEFAULT_MAX_MCP_OUTPUT + over)
        assert len(cap_text(blob, "fs_read", {"path": "/w/a.bin"})) <= len(blob)


def test_the_limit_is_read_at_call_time(monkeypatch):
    """The limit is read at call time, so a value from `--env-file` applies."""
    monkeypatch.delenv("DJED_MAX_MCP_OUTPUT", raising=False)
    assert max_mcp_output() == DEFAULT_MAX_MCP_OUTPUT
    monkeypatch.setenv("DJED_MAX_MCP_OUTPUT", "128")
    assert max_mcp_output() == 128
    assert len(cap_text(PY_SRC, "fs_grep", {})) < 400


def test_zero_disables_capping(monkeypatch):
    monkeypatch.setenv("DJED_MAX_MCP_OUTPUT", "0")
    assert cap_text(PY_SRC, "fs_read", {"path": "/w/big.py"}) == PY_SRC


def test_command_output_keeps_its_verdict():
    """Head and tail are both kept, as exec-mcp's `_RingCapture` keeps them."""
    noise = "\n".join(f"  compiling module_{i:04d}.py ... ok" for i in range(600))
    verdict = "FAILED tests/test_auth.py::test_token\n\n=== 1 failed, 212 passed ==="
    out = cap_text(f"{noise}\n{verdict}", "run_command", {"command": "pytest"})
    assert "FAILED tests/test_auth.py" in out
    assert "1 failed, 212 passed" in out
    assert "bytes omitted" in out


def test_a_tiers_own_continuation_line_survives_the_outline():
    """base-mcp's continuation line, naming a `start_byte`, survives outlining."""
    body = "\n".join(f"def function_{i}(a, b):" for i in range(400))
    trailer = (
        "... truncated at 200000 bytes (FS_MAX_FILE_BYTES) of 900000; "
        "continue with start_byte=200000"
    )
    out = cap_text(f"{body}\n{trailer}", "fs_read", {"path": "/w/big.py"})
    assert "start_byte=200000" in out
    assert "FS_MAX_FILE_BYTES" in out
    # And the numbers describe what was read, not the file, which is longer.
    assert "the file is longer" in out


def test_a_range_request_is_clipped_rather_than_outlined():
    """A range request gets its range, not an outline."""
    lines = [
        f"def function_{i:04d}(a, b):" if i % 7 == 0 else f"    body {i:04d} padding"
        for i in range(2000)
    ]
    sliced = "\n".join(lines[52:891])
    out = cap_text(
        sliced, "fs_read", {"path": "/w/server.py", "start_line": 53, "end_line": 891}
    )
    assert "structure" not in out
    assert "you asked for lines 53-891" in out
    # Both ends of what was asked for, not just the first half.
    assert out.startswith(lines[52])
    assert lines[890] in out


def test_the_outline_stays_inside_the_budget():
    """The outline itself is bounded: 200 entries at 120 characters would be about
    26 KB.
    """
    big = "\n".join(
        f"def function_{i:04d}(argument_one, argument_two):  # {'pad' * 10}"
        for i in range(2000)
    )
    out = cap_text(big, "fs_read", {"path": "/w/big.py"})
    assert len(out) <= DEFAULT_MAX_MCP_OUTPUT
    assert "outline truncated" in out


def test_a_batch_read_is_not_outlined():
    """`fs_read_many` is clipped, not outlined."""
    batch = "\n\n".join(
        f"===== f{i}.py =====\n"
        + "\n".join(f"def g{i}_{j}(argument_one, argument_two): pass" for j in range(80))
        for i in range(12)
    )
    out = cap_text(batch, "fs_read_many", {"paths": ["f0.py", "f1.py"]})
    assert "structure" not in out
    assert "Read one file at a time" in out


def _envelope(**kw):
    base = {"base_mcp_envelope": 1, "content": "hello", "complete": True}
    return base | kw


def test_an_envelope_is_unwrapped_to_text():
    """An envelope is rendered to its body; the model never sees the dict."""
    assert cap_result(_envelope(content="short body"), "fs_read", {}) == "short body"


def test_an_envelope_reports_the_tiers_totals_not_the_bodys():
    """Line counts come from the tier's totals, not from the clipped body."""
    body = "\n".join(f"line {i}" for i in range(4000))
    out = cap_result(
        _envelope(
            content=body,
            complete=False,
            total_bytes=900_000,
            total_lines=20_000,
            label="/w/big.py",
            continue_with="continue with fs_read(path, start_byte=200000).",
        ),
        "fs_read",
        {},
    )
    assert "of 20000 lines" in out
    assert "of 900000 bytes" in out
    assert "start_byte=200000" in out
    assert "/w/big.py" in out
    assert len(out) <= DEFAULT_MAX_MCP_OUTPUT


def test_an_envelope_keeps_both_ends_of_its_body():
    body = "\n".join(f"line {i:05d}" for i in range(4000))
    out = cap_result(_envelope(content=body, complete=False), "run_command", {})
    assert "line 00000" in out
    assert "line 03999" in out
    assert "bytes omitted" in out


def test_a_near_miss_on_the_marker_is_not_an_envelope():
    """It is bounded like any other structure, not rendered as one."""
    almost = {"base_mcp_envelope": 99, "content": "y" * DEFAULT_MAX_MCP_OUTPUT}
    out = cap_result(almost, "fs_read", {})
    assert "as line ranges" not in out and "too large" in out


def test_zero_disables_envelope_capping_too(monkeypatch):
    monkeypatch.setenv("DJED_MAX_MCP_OUTPUT", "0")
    body = "z" * (DEFAULT_MAX_MCP_OUTPUT * 2)
    assert cap_result(_envelope(content=body, complete=False), "fs_read", {}) == body


def _outline_env(n=400, **kw):
    entries = [
        {"name": f"function_{i}", "kind": "def", "start_line": i * 5 + 1,
         "end_line": i * 5 + 4, "depth": 0}
        for i in range(n)
    ]
    base = {
        "base_mcp_envelope": 1,
        "content": "\n".join(f"line {i}" for i in range(4000)),
        "complete": True,
        "total_bytes": 900_000,
        "total_lines": 20_000,
        "label": "/w/big.py",
        "continue_with": "read a definition with fs_read(path, start_line=..., end_line=...).",
        "outline": entries,
    }
    return base | kw


def test_a_tier_outline_is_preferred_over_clipping_the_bytes():
    """A tier outline is used when the bytes do not fit."""
    out = cap_result(_outline_env(n=20), "fs_read", {})
    assert "as line ranges" in out
    assert "1-4  def function_0" in out
    assert "line 0" not in out  # the body was not what came back
    assert "start_line=" in out


def test_a_tier_outline_reports_the_files_own_counts():
    out = cap_result(_outline_env(n=20), "fs_read", {})
    assert "20000 lines" in out and "900000 bytes" in out


def test_a_tier_outline_is_bounded_and_says_when_it_was_cut():
    out = cap_result(_outline_env(n=400), "fs_read", {})
    assert len(out) <= DEFAULT_MAX_MCP_OUTPUT
    assert "outline truncated" in out
    # The head survives: a file's imports and entry points locate the rest.
    assert "def function_0" in out


def test_nesting_shows_as_indentation():
    """A method under its class, which a flat list of start lines could not
    express."""
    env = _outline_env(n=0)
    env["outline"] = [
        {"name": "Widget", "kind": "class", "start_line": 1, "end_line": 9, "depth": 0},
        {"name": "render", "kind": "def", "start_line": 2, "end_line": 3, "depth": 1},
    ]
    out = cap_result(env, "fs_read", {})
    parent = next(ln for ln in out.splitlines() if "Widget" in ln)
    child = next(ln for ln in out.splitlines() if "render" in ln)
    assert child.index("def") > parent.index("class")


def test_an_envelope_without_an_outline_still_clips_its_body():
    """Fallback order: the tier's outline, then the harness's regex, then a clip."""
    out = cap_result(_outline_env(n=0, complete=False), "fs_read", {})
    assert "as line ranges" not in out
    assert "line 0" in out and "bytes omitted" in out


def test_a_range_request_is_never_answered_with_a_map():
    """A range request gets its range even when a tier outline is available."""
    env = _outline_env(n=20, complete=False)
    out = cap_result(env, "fs_read", {"path": "/w/big.py", "start_line": 100, "end_line": 900})
    assert "as line ranges" not in out
    assert "line 0" in out  # the bytes it asked for
    assert "you asked for lines 100-900" in out  # the footer names the request


def test_an_unqualified_read_still_prefers_the_map():
    """The other half of the range rule: an unqualified read still gets the map."""
    out = cap_result(_outline_env(n=20, complete=False), "fs_read", {"path": "/w/big.py"})
    assert "as line ranges" in out


def test_an_explicit_outline_request_is_answered_with_structure():
    """`outline=true` gets the outline, whatever the range rule would choose."""
    env = _outline_env(n=20, complete=True, content="")
    out = cap_result(
        env, "fs_read", {"path": "/w/big.py", "start_line": 100, "end_line": 900, "outline": True}
    )
    assert "as line ranges" in out
    assert "1-4  def function_0" in out


def test_an_outline_request_survives_an_empty_body():
    """An outline-only request with an empty body still returns the outline."""
    env = _outline_env(n=6, complete=True, content="")
    out = cap_result(env, "fs_read", {"path": "/w/big.py", "outline": True})
    assert out.strip()
    assert "as line ranges" in out


def test_a_range_without_the_flag_still_gets_bytes():
    """Without `outline`, inference says bytes."""
    out = cap_result(
        _outline_env(n=20, complete=False), "fs_read",
        {"path": "/w/big.py", "start_line": 100, "end_line": 900},
    )
    assert "as line ranges" not in out
