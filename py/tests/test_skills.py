"""Skills: discovery, the index in the instructions, the two tools, elision, and
the TUI's `/skill` command and size warning.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from djed.agent import build_agent
from djed.decay import elide_tool_results
from djed.skills import (
    LOAD_SKILL_TOOL,
    SkillError,
    discover,
    index_instructions,
    parse_skill_md,
    read_skill_path,
)
from djed.tools import Deps, load_skill
from tests.test_kwim import StubKwim


def write_skill(root: Path, folder: str, frontmatter: str, body: str = "Do the thing.") -> Path:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return d


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    monkeypatch.delenv("DJED_MAX_SKILL_CHARS", raising=False)
    root = tmp_path / "skills"
    write_skill(root, "rotate-certs", "name: rotate-certs\ndescription: Rotate TLS certs.")
    ref = write_skill(
        root, "drain-node", "name: drain-node\ndescription: Drain a node.\nagents: [maat]"
    )
    (ref / "reference.md").write_text("the long reference", encoding="utf-8")
    write_skill(root, "build-only", "name: build-only\ndescription: x\nagents: builder")
    return root


# --- parsing and discovery -------------------------------------------------


def test_frontmatter_continuation_and_block_indicator():
    meta, body = parse_skill_md(
        "---\nname: a\ndescription: >\n  first line\n  second line\n---\nbody\n"
    )
    assert meta == {"name": "a", "description": "first line second line"}
    assert body == "body"


def test_frontmatter_must_be_fenced():
    with pytest.raises(SkillError):
        parse_skill_md("name: a\n")
    with pytest.raises(SkillError):
        parse_skill_md("---\nname: a\n")


def test_discover_filters_by_persona(skills_root):
    assert set(discover("maat", [skills_root]).skills) == {"rotate-certs", "drain-node"}
    assert set(discover("builder", [skills_root]).skills) == {"rotate-certs", "build-only"}


def test_missing_directory_is_not_an_error(tmp_path):
    found = discover("maat", [tmp_path / "absent"])
    assert found.skills == {} and found.problems == []


def test_unusable_skill_is_reported_and_skipped(skills_root):
    write_skill(skills_root, "bad", "name: Bad_Name\ndescription: x")
    write_skill(skills_root, "nodesc", "name: nodesc")
    found = discover("maat", [skills_root])
    assert "bad" not in found.skills and "nodesc" not in found.skills
    assert len(found.problems) == 2


def test_first_directory_wins_a_name_clash(skills_root, tmp_path):
    other = tmp_path / "other"
    write_skill(other, "rotate-certs", "name: rotate-certs\ndescription: shadowed")
    found = discover("maat", [skills_root, other])
    assert found.skills["rotate-certs"].description == "Rotate TLS certs."
    assert any("ignored" in p for p in found.problems)


def test_oversized_is_flagged_against_the_limit(skills_root, monkeypatch):
    monkeypatch.setenv("DJED_MAX_SKILL_CHARS", "100")
    write_skill(skills_root, "huge", "name: huge\ndescription: big", body="x" * 200)
    found = discover("maat", [skills_root])
    assert [s.name for s in found.oversized] == ["huge"]
    monkeypatch.setenv("DJED_MAX_SKILL_CHARS", "0")
    assert discover("maat", [skills_root]).oversized == []


def test_index_lists_names_and_descriptions_only(skills_root):
    text = index_instructions(discover("maat", [skills_root]).skills)
    assert "- drain-node: Drain a node." in text
    assert "Do the thing." not in text
    assert index_instructions({}) == ""


def test_read_skill_path_stays_inside_the_folder(skills_root):
    skill = discover("maat", [skills_root]).skills["drain-node"]
    assert read_skill_path(skill, "reference.md").read_text() == "the long reference"
    with pytest.raises(SkillError):
        read_skill_path(skill, "../rotate-certs/SKILL.md")


# --- the agent ---------------------------------------------------------------


def test_no_skills_adds_no_tools():
    agent = build_agent(StubKwim(), "s", model=FunctionModel(lambda m, i: None))
    assert LOAD_SKILL_TOOL not in agent._function_toolset.tools


@pytest.mark.asyncio
async def test_agent_loads_a_skill_and_reads_its_reference(skills_root):
    skills = discover("maat", [skills_root]).skills
    seen: dict[str, object] = {}

    def model(messages, info: AgentInfo):
        seen["instructions"] = info.instructions
        returns = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts
                   if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(LOAD_SKILL_TOOL, {"name": "drain-node"})])
        if len(returns) == 1:
            return ModelResponse(parts=[ToolCallPart(
                "read_skill_file", {"name": "drain-node", "path": "reference.md"})])
        seen["returns"] = [str(p.content) for p in returns]
        return ModelResponse(parts=[TextPart("done")])

    kwim = StubKwim()
    agent = build_agent(kwim, "s", model=FunctionModel(model), skills=skills)
    result = await agent.run("drain node 3", deps=Deps(kwim=kwim, session_id="s", skills=skills))
    assert result.output == "done"
    assert "- drain-node: Drain a node." in seen["instructions"]
    loaded, ref = seen["returns"]
    assert "Do the thing." in loaded and "- reference.md" in loaded
    assert ref == "the long reference"


@pytest.mark.asyncio
async def test_unknown_skill_names_the_available_ones(skills_root):
    from types import SimpleNamespace

    skills = discover("maat", [skills_root]).skills
    ctx = SimpleNamespace(deps=Deps(kwim=StubKwim(), session_id="s", skills=skills))
    out = await load_skill(ctx, "nope")
    assert "rotate-certs" in out and "drain-node" in out


# --- elision -------------------------------------------------------------------


def test_loaded_skill_is_never_elided_and_costs_no_budget():
    def pair(name, content, cid):
        return [
            ModelResponse(parts=[ToolCallPart(name, {"name": "x"}, tool_call_id=cid)]),
            ModelRequest(parts=[ToolReturnPart(name, content, tool_call_id=cid)]),
        ]

    msgs = [
        ModelRequest(parts=[UserPromptPart("go")]),
        *pair(LOAD_SKILL_TOOL, "S" * 500, "c1"),
        *pair("fs_read", "a" * 100, "c2"),
        *pair("fs_read", "b" * 100, "c3"),
    ]
    out = elide_tool_results(msgs, budget=100)
    contents = [p.content for m in out if isinstance(m, ModelRequest) for p in m.parts
                if isinstance(p, ToolReturnPart)]
    assert contents[0] == "S" * 500
    assert contents[1].startswith("[elided")
    assert contents[2] == "b" * 100
    # A zero budget still keeps the skill.
    zero = elide_tool_results(msgs, budget=0)
    assert any(p.content == "S" * 500 for m in zero if isinstance(m, ModelRequest)
               for p in m.parts if isinstance(p, ToolReturnPart))


# --- the TUI -------------------------------------------------------------------


def make_app(skills, prompts: list[str]):
    from djed.session import Session
    from djed.tiers import EagerTiers
    from djed.tui import DjedApp

    async def stream(messages, info):
        prompts.append(
            [p.content for p in messages[-1].parts if isinstance(p, UserPromptPart)][-1]
        )
        yield "ok"

    kwim = StubKwim()
    agent = build_agent(kwim, "t", model=FunctionModel(stream_function=stream), skills=skills)
    session = Session(
        agent=agent,
        deps=Deps(kwim=kwim, session_id="t", skills=skills),
        kwim=kwim,
        session_id="t",
        strategy=EagerTiers(),
    )
    return DjedApp(session)


@pytest.fixture
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setenv("DJED_HISTORY_FILE", str(tmp_path / "history"))


async def _submit(pilot, text):
    from djed.tui import PromptArea

    pilot.app.query_one(PromptArea).text = text
    await pilot.press("enter")
    await pilot.pause()
    if pilot.app._turn is not None:
        await pilot.app._turn
    await pilot.pause()


@pytest.mark.asyncio
async def test_skill_command_lists_and_marks_oversized(skills_root, monkeypatch, isolated_history):
    monkeypatch.setenv("DJED_MAX_SKILL_CHARS", "100")
    write_skill(skills_root, "huge", "name: huge\ndescription: big", body="x" * 200)
    app = make_app(discover("maat", [skills_root]).skills, [])
    async with app.run_test() as pilot:
        await _submit(pilot, "/skill")
        text = "\n".join(app.transcript_log)
        assert "`drain-node`" in text and "**too large**" in text


@pytest.mark.asyncio
async def test_skill_with_request_sends_the_body(skills_root, isolated_history):
    prompts: list[str] = []
    app = make_app(discover("maat", [skills_root]).skills, prompts)
    async with app.run_test() as pilot:
        await _submit(pilot, '/skill drain-node drain "node 3"')
    assert len(prompts) == 1
    assert "Do the thing." in prompts[0] and prompts[0].endswith('drain "node 3"')


@pytest.mark.asyncio
async def test_skill_alone_rides_on_the_next_prompt(skills_root, isolated_history):
    prompts: list[str] = []
    app = make_app(discover("maat", [skills_root]).skills, prompts)
    async with app.run_test() as pilot:
        await _submit(pilot, "/skill rotate-certs")
        assert prompts == []
        await _submit(pilot, "rotate them")
        await _submit(pilot, "and again")
    assert "Do the thing." in prompts[0] and prompts[0].endswith("rotate them")
    assert prompts[1] == "and again"


@pytest.mark.asyncio
async def test_loading_an_oversized_skill_warns(skills_root, monkeypatch, isolated_history):
    monkeypatch.setenv("DJED_MAX_SKILL_CHARS", "100")
    write_skill(skills_root, "huge", "name: huge\ndescription: big", body="x" * 200)
    app = make_app(discover("maat", [skills_root]).skills, [])
    async with app.run_test() as pilot:
        await _submit(pilot, "/skill huge")
        text = "\n".join(app.transcript_log)
        assert "over the 100 limit" in text and "separate reference files" in text
