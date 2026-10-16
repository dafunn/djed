"""TUI tests driven through Textual's Pilot - real widgets, simulated keys.

Covers ctrl+c interrupt, double-ctrl+d exit, the approval gate, prompt history,
and multiline continuation.
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from djed.tui import PromptArea

from fake_model import scripted_model
from djed.agent import build_agent
from djed.session import Session
from djed.tiers import EagerTiers, Tier
from djed.tools import Deps
from djed.persona import MAAT, Persona
from djed.tui import DjedApp
from tests.test_kwim import StubKwim


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Point the app's history at a temp file, not the operator's real one."""
    monkeypatch.setenv("DJED_HISTORY_FILE", str(tmp_path / "history"))


def make_session(model=None, tiers=None):
    kwim = StubKwim()
    model = model or scripted_model()

    def factory(toolsets, capabilities, extra_instructions, override=None):
        return build_agent(
            kwim,
            "tui-test",
            model=override if override is not None else model,
            toolsets=toolsets,
            capabilities=capabilities,
            extra_instructions=extra_instructions,
        )

    return Session(
        agent=factory([], [], ""),
        deps=Deps(kwim=kwim, session_id="tui-test"),
        kwim=kwim,
        session_id="tui-test",
        tiers=tiers or [],
        strategy=EagerTiers(),
        agent_factory=factory,
    )


def transcript_text(app: DjedApp) -> str:
    return "\n".join(app.transcript_log)


async def submit(pilot, text: str) -> None:
    pilot.app.query_one(PromptArea).text = text
    await pilot.press("enter")
    await pilot.pause()


# --- basics ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_banner_and_prompt_render():
    banner = "test banner"
    app = DjedApp(make_session(), banner=banner)
    async with app.run_test() as pilot:
        assert banner in transcript_text(app)
        assert app.query_one(PromptArea).has_focus


@pytest.mark.asyncio
async def test_prompt_placeholder_names_the_persona():
    persona = Persona("Stub", "You are a stub.", "stub-persona")
    app = DjedApp(make_session(), persona=persona)
    async with app.run_test() as pilot:
        assert app.query_one(PromptArea).placeholder == f"ask {persona.name}..."


@pytest.mark.asyncio
async def test_prompt_placeholder_defaults_to_the_default_persona():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        assert app.query_one(PromptArea).placeholder == f"ask {MAAT.name}..."


@pytest.mark.asyncio
async def test_help_command_lists_commands():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "/help")
        assert "/tier" in transcript_text(app)


@pytest.mark.asyncio
async def test_unknown_command_is_reported():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "/nonsense")
        assert "unknown command" in transcript_text(app)


@pytest.mark.asyncio
async def test_shell_escape_runs_a_command():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "!echo djed-shell-ok")
        await pilot.pause()
        for _ in range(40):
            if "djed-shell-ok" in transcript_text(app):
                break
            await asyncio.sleep(0.05)
        assert "djed-shell-ok" in transcript_text(app)


# --- key handling ---------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_c_on_empty_prompt_does_not_exit():
    """ctrl+c must never quit."""
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        assert "ctrl+c interrupts" in transcript_text(app)


@pytest.mark.asyncio
async def test_ctrl_c_clears_a_typed_line():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.query_one(PromptArea).text = "half a thought"
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.query_one(PromptArea).text == ""
        assert app.is_running


@pytest.mark.asyncio
async def test_single_ctrl_d_warns_and_does_not_exit():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert app.is_running
        assert "press ctrl+d again" in transcript_text(app)


@pytest.mark.asyncio
async def test_double_ctrl_d_exits():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+d")
        await pilot.pause()
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_ctrl_d_with_text_present_does_not_exit():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.query_one(PromptArea).text = "typing"
        await pilot.press("ctrl+d")
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert app.is_running


# --- prompt history -------------------------------------------------------


@pytest.mark.asyncio
async def test_up_arrow_recalls_previous_prompts():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "/help")
        await submit(pilot, "/clear")
        await pilot.press("up")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "/clear"
        await pilot.press("up")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "/help"


@pytest.mark.asyncio
async def test_down_arrow_returns_to_an_empty_line():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "/help")
        await pilot.press("up")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.query_one(PromptArea).text == ""


# --- multiline ------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiline_text_is_submitted_as_one_prompt():
    """Multiline parts are submitted as one prompt."""
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the disk on\nthe monitoring node")
        echoed = transcript_text(app)
        assert "check the disk on" in echoed
        assert "the monitoring node" in echoed


@pytest.mark.asyncio
async def test_paste_keeps_every_line():
    """A paste keeps every line."""
    from textual import events
    from textual.widgets import Input, TextArea

    # Input._on_paste keeps splitlines()[0] and stops the event.
    assert issubclass(PromptArea, TextArea) and not issubclass(PromptArea, Input)

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        prompt.focus()
        pasted = "line one\nline two\nline three"
        await prompt._on_paste(events.Paste(pasted))
        await pilot.pause()
        assert prompt.text == pasted


@pytest.mark.asyncio
async def test_enter_submits_and_alt_enter_adds_a_line():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        prompt = app.query_one(PromptArea)
        prompt.focus()
        prompt.text = "first"
        prompt.move_cursor(prompt.document.end)
        await pilot.press("alt+enter")
        await pilot.pause()
        assert prompt.text == "first\n"      # a line was added, nothing submitted
        assert "you>" not in transcript_text(app)

        prompt.insert("second")
        await pilot.press("enter")
        await pilot.pause()
        assert prompt.text == ""             # submitted and cleared
        assert "second" in transcript_text(app)


@pytest.mark.asyncio
async def test_up_moves_the_cursor_before_it_recalls_history():
    """Inside multi-line text the arrows move the cursor; only the top line recalls
    history.
    """
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        # Seeded directly: a real turn disables the prompt while it runs.
        app.history = ["an earlier question"]
        prompt = app.query_one(PromptArea)
        prompt.focus()
        await pilot.pause()          # focus lands on the next pump
        prompt.text = "line one\nline two"
        prompt.move_cursor(prompt.document.end)
        await pilot.press("up")
        await pilot.pause()
        assert prompt.text == "line one\nline two"   # moved, did not recall
        assert prompt.cursor_location[0] == 0

        await pilot.press("up")                      # now at the top edge
        await pilot.pause()
        assert prompt.text == "an earlier question"


# --- tiers and strategy ---------------------------------------------------


FLEET = [
    Tier("db", None, ("db_query", "db_tables")),
    Tier("cluster", None, ("cluster_list", "cluster_logs", "cluster_events")),
]


@pytest.mark.asyncio
async def test_tier_command_lists_tiers():
    app = DjedApp(make_session(tiers=FLEET))
    async with app.run_test() as pilot:
        await submit(pilot, "/tier")
        text = transcript_text(app)
        assert "db" in text and "cluster" in text


@pytest.mark.asyncio
async def test_tier_command_narrows_the_selection():
    session = make_session(tiers=FLEET)
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "/tier cluster")
        assert session.enabled_tiers == {"cluster"}
        assert session.active_tool_count == 3


@pytest.mark.asyncio
async def test_tier_deltas_work():
    session = make_session(tiers=FLEET)
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "/tier cluster")
        await submit(pilot, "/tier +db")
        assert session.enabled_tiers == {"cluster", "db"}


@pytest.mark.asyncio
async def test_bad_tier_spec_is_reported_and_changes_nothing():
    session = make_session(tiers=FLEET)
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "/tier bogus")
        assert "unknown tier" in transcript_text(app)
        assert session.enabled_tiers == {"db", "cluster"}


@pytest.mark.asyncio
async def test_strategy_switch_rebuilds_the_agent():
    session = make_session(tiers=FLEET)
    before = session.agent
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "/strategy search")
        assert session.strategy_label == "search"
        assert session.agent is not before


@pytest.mark.asyncio
async def test_clear_resets_conversation_history():
    session = make_session()
    session.history = ["not empty"]  # type: ignore[list-item]
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "/clear")
        assert session.history == []


# --- approval gate --------------------------------------------------------


async def wait_for(app, needle: str, tries: int = 60) -> bool:
    for _ in range(tries):
        if needle in transcript_text(app):
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_approval_prompt_appears_and_y_approves():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the cluster")
        assert await wait_for(app, "[approve (y/n)]")
        await pilot.press("y")
        assert await wait_for(app, "djed-roundtrip-ok"), "approved command should run"


@pytest.mark.asyncio
async def test_n_denies_and_the_command_never_runs():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the cluster")
        assert await wait_for(app, "[approve (y/n)]")
        await pilot.press("n")
        assert await wait_for(app, "denied")
        await pilot.pause()
        text = transcript_text(app)
        # The command string appears, since it was shown in the approval prompt; its
        # output does not.
        assert "[ok] Operator denied this command." in text
        assert "[ok] djed-roundtrip-ok" not in text


@pytest.mark.asyncio
async def test_escape_denies():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the cluster")
        assert await wait_for(app, "[approve (y/n)]")
        await pilot.press("escape")
        assert await wait_for(app, "denied")


@pytest.mark.asyncio
async def test_approval_swallows_other_keys():
    """Typing while the approval gate is open does not reach the editor."""
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the cluster")
        assert await wait_for(app, "[approve (y/n)]")
        await pilot.press("z", "q", "1")
        await pilot.pause()
        assert app.query_one(PromptArea).text == ""
        assert app._approval is not None, "gate should still be waiting"
        await pilot.press("n")


@pytest.mark.asyncio
async def test_ctrl_c_during_approval_denies_rather_than_interrupting():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await submit(pilot, "check the cluster")
        assert await wait_for(app, "[approve (y/n)]")
        await pilot.press("ctrl+c")
        assert await wait_for(app, "denied")
        assert app.is_running


@pytest.mark.asyncio
async def test_bare_model_reports_the_live_model_not_the_env_var(monkeypatch):
    """`/model` reports the current model after a switch, not DJED_MODEL."""
    monkeypatch.setenv("DJED_MODEL", "startup-model:1b")
    session = make_session()
    app = DjedApp(session)
    async with app.run_test() as pilot:
        session.agent.model = FunctionModel(
            lambda m, i: ModelResponse(parts=[TextPart(content="x")]),
            model_name="switched-model:9b",
        )
        await submit(pilot, "/model")
        for _ in range(40):
            if "switched-model:9b" in transcript_text(app):
                break
            await asyncio.sleep(0.05)
        text = transcript_text(app)
        assert "switched-model:9b" in text
        assert "startup-model:1b" not in text


# --- ctrl+o: toggle tool output detail ----------------------------------
#
# Tool results are collapsed to 120 characters; ctrl+o swaps in the full text
# in place.


LONG_ERROR = (
    "Error: (401) Reason: Unauthorized HTTP response headers: "
    "HTTPHeaderDict({'Audit-Id': '4ed17d79-09ac-4c75-82a7-37a416b7bd0c', "
    "'Cache-Control': 'no-cache, private', 'Content-Type': 'application/json'}) "
    "HTTP response body: the part that actually says what went wrong"
)
TAIL = "the part that actually says what went wrong"


def rendered(app) -> str:
    """What the transcript widgets currently show, read from the widgets rather than
    the transcript mirror.
    """
    from textual.widgets import Static
    return "\n".join(str(w.render()) for w in app.query(Static))


@pytest.mark.asyncio
async def test_ctrl_o_expands_then_collapses():
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text=LONG_ERROR[:120] + "...", full=LONG_ERROR))
        await pilot.pause()
        assert TAIL not in rendered(app)

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert TAIL in rendered(app)

        await pilot.press("ctrl+o")
        await pilot.pause()
        assert TAIL not in rendered(app)


@pytest.mark.asyncio
async def test_the_toggle_is_global_not_just_the_last_result():
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text="first", full="the FIRST result"))
        app.on_event_from_session(ToolFinished(text="second", full="the SECOND result"))
        await pilot.pause()
        await pilot.press("ctrl+o")
        await pilot.pause()
        body = rendered(app)
        assert "the FIRST result" in body and "the SECOND result" in body


@pytest.mark.asyncio
async def test_new_output_while_expanded_arrives_expanded():
    """Otherwise the view is half summarised and half not."""
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text="first", full="the FIRST result"))
        await pilot.pause()
        await pilot.press("ctrl+o")
        await pilot.pause()
        app.on_event_from_session(ToolFinished(text="later", full="the LATER result"))
        await pilot.pause()
        assert "the LATER result" in rendered(app)


@pytest.mark.asyncio
async def test_toggling_rewrites_rather_than_reprints():
    """Toggling updates results in place rather than adding copies."""
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text="x", full="unique-marker-text"))
        await pilot.pause()
        before = len(app.query("Static"))
        await pilot.press("ctrl+o")
        await pilot.pause()
        # One status line is added; the result itself is updated in place.
        assert rendered(app).count("unique-marker-text") == 1
        assert len(app.query("Static")) == before + 1


@pytest.mark.asyncio
async def test_ctrl_o_says_so_when_there_is_nothing_yet():
    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        await pilot.press("ctrl+o")
        await pilot.pause()
        assert "no tool output yet" in transcript_text(app)


@pytest.mark.asyncio
async def test_the_saved_transcript_is_unaffected_by_the_toggle():
    """/save carries full output; the display mode must not change it."""
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text="x", full="full text here"))
        await pilot.pause()
        before = list(app.full_log)
        await pilot.press("ctrl+o")
        await pilot.pause()
        assert [l for l in app.full_log if "full text here" in l] == \
               [l for l in before if "full text here" in l]


@pytest.mark.asyncio
async def test_ctrl_o_does_not_disturb_the_prompt():
    """It is a display command; anything half-typed must survive it."""
    from djed.session import ToolFinished

    app = DjedApp(make_session())
    async with app.run_test() as pilot:
        app.on_event_from_session(ToolFinished(text="x", full="full text"))
        app.query_one(PromptArea).text = "half-typed question"
        await pilot.press("ctrl+o")
        await pilot.pause()
        assert app.query_one(PromptArea).text == "half-typed question"


# --- memory failures ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_kwim_failure_is_shown_once_after_the_turn():
    async def answer(messages, info: AgentInfo):
        yield "ok"

    session = make_session(FunctionModel(stream_function=answer))

    async def failing_recall(query, k=4):
        session.kwim._failed("recall", "http-401", "HTTP 401 (check DJED_KWIM_API_KEY)")
        return None

    session.kwim.semantic_recall = failing_recall  # type: ignore[method-assign]
    app = DjedApp(session)
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        assert await wait_for(app, "[memory] KWIM recall failed: HTTP 401"), transcript_text(app)
        await submit(pilot, "hello again")
        for _ in range(60):
            if transcript_text(app).count("request(s)]") == 2:
                break
            await asyncio.sleep(0.05)
        assert transcript_text(app).count("request(s)]") == 2, transcript_text(app)
    assert transcript_text(app).count("[memory]") == 1
