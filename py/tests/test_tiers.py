"""Tool-loading strategy: tier selection and the eager/search swap."""
from __future__ import annotations

import pytest
from pydantic_ai import DeferredLoadingToolset
from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset

from djed.tiers import (
    EagerTiers,
    SearchedTiers,
    Tier,
    build_strategy,
    resolve_selection,
    toolset_id,
    unwrap_to,
)


def tier(name: str, *tools: str) -> Tier:
    return Tier(name, FunctionToolset(id=name), tuple(tools))


FLEET = [
    tier("db", "db_query", "db_tables"),
    tier("cluster", "cluster_list", "cluster_logs"),
    tier("logs", "logs_query"),
]
NAMES = [t.name for t in FLEET]


# --- selection parsing ----------------------------------------------------


@pytest.mark.parametrize("spec", [None, "", "   ", "all"])
def test_default_and_all_select_everything(spec):
    selection, err = resolve_selection(spec, NAMES)
    assert selection == set(NAMES)
    assert err is None


def test_none_selects_nothing():
    assert resolve_selection("none", NAMES) == (set(), None)


def test_explicit_list_sets_the_selection():
    selection, err = resolve_selection("cluster,db", NAMES)
    assert selection == {"cluster", "db"}
    assert err is None


def test_whitespace_is_tolerated():
    assert resolve_selection(" cluster , db ", NAMES)[0] == {"cluster", "db"}


def test_plus_delta_adds_to_current():
    selection, err = resolve_selection("+logs", NAMES, current={"db"})
    assert selection == {"db", "logs"}
    assert err is None


def test_minus_delta_removes_from_current():
    selection, err = resolve_selection("-db", NAMES, current={"db", "cluster"})
    assert selection == {"cluster"}
    assert err is None


def test_removing_an_absent_tier_is_not_an_error():
    assert resolve_selection("-logs", NAMES, current={"db"})[0] == {"db"}


def test_unknown_tier_is_rejected_and_selection_unchanged():
    selection, err = resolve_selection("cluster,nope", NAMES, current={"db"})
    assert selection == {"db"}
    assert "nope" in err


def test_mixing_deltas_and_absolute_names_is_rejected():
    """Deltas and absolute names cannot be mixed."""
    selection, err = resolve_selection("+cluster,db", NAMES, current={"logs"})
    assert selection == {"logs"}
    assert "delta" in err


# --- eager strategy -------------------------------------------------------


def test_eager_default_sends_every_tier():
    toolsets, caps = EagerTiers().apply(FLEET)
    assert len(toolsets) == 3
    assert caps == []


def test_eager_selection_sends_only_the_chosen_tiers():
    toolsets, caps = EagerTiers(enabled=frozenset({"cluster"})).apply(FLEET)
    # Through `toolset_id`, not `.id`: the write gate wraps each tier, and a
    # WrapperToolset reports no id of its own.
    assert [toolset_id(t) for t in toolsets] == ["cluster"]
    assert caps == []


def test_eager_none_sends_nothing():
    toolsets, _ = EagerTiers(enabled=frozenset()).apply(FLEET)
    assert toolsets == []


# --- search strategy ------------------------------------------------------


def test_search_defers_every_tier_and_adds_the_capability():
    toolsets, caps = SearchedTiers().apply(FLEET)
    assert len(toolsets) == 3
    assert all(isinstance(t, DeferredLoadingToolset) for t in toolsets)
    # tool_names=None means "defer everything in this toolset"
    assert all(t.tool_names is None for t in toolsets)
    assert len(caps) == 1


def test_the_two_strategies_are_interchangeable():
    """Both strategies reduce to (toolsets, capabilities)."""
    for strategy in (EagerTiers(), SearchedTiers()):
        toolsets, caps = strategy.apply(FLEET)
        assert isinstance(toolsets, list)
        assert isinstance(caps, list)


# --- strategy construction from config ------------------------------------


def test_build_strategy_defaults_to_eager_everything(monkeypatch):
    monkeypatch.delenv("DJED_TOOL_STRATEGY", raising=False)
    monkeypatch.delenv("DJED_MCP_TIERS", raising=False)
    strategy, err = build_strategy(available=NAMES)
    assert isinstance(strategy, EagerTiers) and strategy.enabled is None
    assert err is None


def test_build_strategy_honours_the_tier_spec():
    strategy, err = build_strategy("eager", "cluster,db", NAMES)
    assert strategy.enabled == frozenset({"cluster", "db"})
    assert err is None


def test_build_strategy_reads_the_environment(monkeypatch):
    monkeypatch.setenv("DJED_TOOL_STRATEGY", "search")
    strategy, _ = build_strategy(available=NAMES)
    assert isinstance(strategy, SearchedTiers)


def test_build_strategy_reports_a_bad_spec_without_crashing():
    strategy, err = build_strategy("eager", "bogus", NAMES)
    assert err and "bogus" in err
    assert isinstance(strategy, EagerTiers)


def test_unknown_strategy_falls_back_to_eager():
    strategy, err = build_strategy("telepathy", None, NAMES)
    assert isinstance(strategy, EagerTiers)
    assert err and "telepathy" in err


# --- token cost estimation ------------------------------------------------


class _FakeTool:
    def __init__(self, name, description="", schema=None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {}}


def test_no_tools_costs_nothing():
    from djed.tokens import estimate_tool_tokens

    assert estimate_tool_tokens([]) == 0


def test_cost_grows_with_tool_count():
    from djed.tokens import estimate_tool_tokens

    one = estimate_tool_tokens([_FakeTool("a", "does a thing")])
    two = estimate_tool_tokens([_FakeTool("a", "does a thing"), _FakeTool("b", "does b")])
    assert 0 < one < two


def test_cost_counts_description_and_schema():
    from djed.tokens import estimate_tool_tokens

    bare = estimate_tool_tokens([_FakeTool("a")])
    rich = estimate_tool_tokens([
        _FakeTool("a", "a long description that certainly costs tokens",
                  {"type": "object", "properties": {"q": {"type": "string",
                   "description": "the query to run"}}})
    ])
    assert rich > bare


def test_cost_includes_the_function_envelope():
    """The estimate includes the envelope keys and JSON punctuation."""
    import json

    import tiktoken

    from djed.tokens import estimate_tool_tokens

    tool = _FakeTool("cluster_list", "List kubernetes objects")
    enc = tiktoken.get_encoding("cl100k_base")
    naive = len(enc.encode(tool.description)) + len(enc.encode(json.dumps(tool.inputSchema)))
    assert estimate_tool_tokens([tool]) > naive


def test_tier_carries_its_cost():
    t = Tier("cluster", None, ("cluster_list",), token_cost=843)
    assert t.token_cost == 843


def test_tier_cost_defaults_to_zero():
    """Tiers built without a probe (tests, --no-mcp) have a cost of zero."""
    assert Tier("db", None, ("db_query",)).token_cost == 0


def test_tier_spec_is_honoured_under_search():
    """A tier spec applies to either strategy, search included."""
    strategy, err = build_strategy("search", "cluster,db", NAMES)
    assert err is None
    assert strategy.label == "search"
    assert strategy.enabled == frozenset({"cluster", "db"})


def test_search_applies_only_the_enabled_tiers():
    toolsets, _ = SearchedTiers(enabled=frozenset({"db"})).apply(FLEET)
    assert len(toolsets) == 1


# --- the write gate ------------------------------------------------------
#
# An MCP tool cannot raise ApprovalRequired itself, so the wrapper gates
# `fs_write` and `fs_delete`.


def test_write_tools_are_recognised_bare_and_renamed():
    from djed.tiers import _is_write_tool

    class Def:
        def __init__(self, name):
            self.name = name

    assert _is_write_tool(Def("fs_write"))
    assert _is_write_tool(Def("fs_delete"))
    # A name collision renames the tool on every server that has it.
    assert _is_write_tool(Def("base__fs_edit"))
    assert not _is_write_tool(Def("fs_read"))
    assert not _is_write_tool(Def("fs_grep"))
    assert not _is_write_tool(Def("k8s_list"))


def test_eager_gates_writes_by_default():
    toolsets, _ = EagerTiers().apply(FLEET)
    assert all(unwrap_to(t, ApprovalRequiredToolset) is not None for t in toolsets)
    # The tier is still identifiable through the wrapper.
    assert set(toolset_id(t) for t in toolsets) == {t.name for t in FLEET}


def test_the_gate_can_be_turned_off_but_is_not_off_by_default():
    toolsets, _ = EagerTiers(gate_writes=False).apply(FLEET)
    assert not any(unwrap_to(t, ApprovalRequiredToolset) for t in toolsets)


def test_search_gates_inside_the_deferral_not_outside():
    """The approval gate is innermost, so a discovered tool is still gated."""
    toolsets, _ = SearchedTiers().apply(FLEET)
    assert all(unwrap_to(t, ApprovalRequiredToolset) is not None for t in toolsets)
    # Deferral stays outermost: it decides whether a definition is advertised.
    assert all(isinstance(t, DeferredLoadingToolset) for t in toolsets)


@pytest.mark.parametrize(
    "value, gated_on",
    [(None, True), ("1", True), ("true", True), ("0", False), ("no", False),
     ("off", False), ("", True)],
)
def test_gate_default_follows_the_environment(monkeypatch, value, gated_on):
    from djed.tiers import gate_writes_default

    if value is None:
        monkeypatch.delenv("DJED_GATE_WRITES", raising=False)
    else:
        monkeypatch.setenv("DJED_GATE_WRITES", value)
    assert gate_writes_default() is gated_on


def test_the_gate_predicate_only_fires_for_writes():
    """The gate prompts only for write tools."""
    toolsets, _ = EagerTiers().apply(FLEET)
    predicate = unwrap_to(toolsets[0], ApprovalRequiredToolset).approval_required_func

    class Def:
        def __init__(self, name):
            self.name = name

    assert predicate(None, Def("fs_write"), {}) is True
    assert predicate(None, Def("fs_read"), {}) is False


def test_the_gate_predicate_has_no_side_effect_and_needs_no_deps():
    """`gated()` is a pure name check: it needs no `ctx` and reads no state."""
    toolsets, _ = EagerTiers().apply(FLEET)
    predicate = unwrap_to(toolsets[0], ApprovalRequiredToolset).approval_required_func

    class Def:
        def __init__(self, name):
            self.name = name

    assert predicate(None, Def("fs_write"), {"path": "/workspace/pytest"}) is True
    assert predicate(None, Def("fs_read"), {"path": "/workspace/src/demo/greeting.py"}) is False
