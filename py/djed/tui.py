"""Djed - the Textual terminal: presentation and key handling only.

Agent behaviour lives in `djed/session.py`. Key bindings are in `HELP` below.
"""
from __future__ import annotations

import asyncio
import shlex
import time
from datetime import datetime
from pathlib import Path

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static, TextArea

from .persona import MAAT, Persona
from .history import History
from .provider import list_models, model_id
from .session import Session, TextChunk, ToolFinished, ToolStarted
from .tiers import resolve_selection
from .tools import MAX_OUTPUT, SHELL_TIMEOUT_S


def _clipboard_caveat() -> str:
    """OSC 52 gives no acknowledgement, so name the likely blocker rather than report
    success.
    """
    from .terminal import clipboard_blocker

    blocker = clipboard_blocker()
    return f" - {blocker}" if blocker else ". If nothing pasted, use /save"


def _plain(renderable) -> str:
    """Best-effort plain text for the transcript mirror."""
    return getattr(renderable, "plain", None) or str(renderable)


CTRL_D_WINDOW_S = 2.0
STREAM_REFRESH_S = 0.08  # re-render the streaming block at most this often

HELP = """\
commands  `/model [id]`  `/tier [spec]`  `/strategy [eager|search]`  `/save [path]`  `/copy`  `/clear`  `/help`  `/exit`
shell  `!<cmd>`   interrupt  `Esc` or `ctrl+c`   exit  `ctrl+d` twice
prompt  `enter` sends, `alt+enter` (or `ctrl+j`) adds a line; paste keeps its line breaks
tool output  `ctrl+o` toggles between the 120-character summary and the full text
copying  drag to select then `ctrl+c`, or `shift`+drag for your terminal's own selection.
`/save` writes the transcript (tool output untruncated) to a file - the reliable way to share an error.
`/tier` takes `all`, `none`, `cluster,db`, or deltas like `+cluster,-db`."""


class Transcript(VerticalScroll):
    """The scrolling conversation area."""


class PromptArea(TextArea):
    """The prompt: a TextArea, so pasted text keeps its line breaks.

    enter submits; alt+enter (or ctrl+j, for terminals that intercept alt) inserts
    a newline. Up and down move the cursor, and recall history only from the first
    or last line; the app checks that edge, since it sees keys first.
    """

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    async def _on_key(self, event) -> None:
        key = event.key
        if key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if key in ("alt+enter", "ctrl+j", "shift+enter"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return


class DjedApp(App):
    """The agent's terminal."""

    CSS = """
    Screen { layout: vertical; }
    Transcript { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    Transcript > Static { margin-bottom: 1; }
    PromptArea {
        dock: bottom; border: round $primary-darken-2;
        height: auto; max-height: 12; min-height: 3;
    }
    PromptArea:focus { border: round $primary; }
    .you { color: $text-muted; }
    .tool { color: $accent; }
    .muted { color: $text-muted; }
    .warn { color: $warning; }
    .err { color: $error; }
    """
    # ctrl+c is bound to a non-priority system quit; on_key intercepts it first.
    BINDINGS = []
    ENABLE_COMMAND_PALETTE = False

    def __init__(
        self,
        session: Session,
        banner: str = "",
        persona: Persona = MAAT,
        history: History | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session
        self.banner = banner
        self.persona = persona
        self._placeholder = f"ask {persona.name}..."
        # Seeded from disk so the up arrow reaches past this session. Appends
        # are written through.
        self._history_store = history or History()
        self.history: list[str] = self._history_store.load()
        self._hist_index: int | None = None
        self._last_ctrl_d = 0.0
        self._approval: asyncio.Future[bool] | None = None
        self._stream: Static | None = None
        self._stream_buf = ""
        self._stream_due = 0.0
        self._turn: asyncio.Task | None = None
        # Plain-text mirror of everything rendered. `_append` is the single
        # output funnel, so this stays faithful and tests can assert on it.
        self.transcript_log: list[str] = []
        # Same lines, but with tool output untruncated - what /save and /copy
        # emit, so a pasted error is the whole error.
        self.full_log: list[str] = []
        # ctrl+o expands every tool result, so each one is kept with both renderings
        # and its widget.
        self._verbose_tools = False
        self._tool_widgets: list[tuple[Static, str, str]] = []

    # --- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Transcript(id="transcript")
        yield PromptArea(placeholder=self._placeholder, id="prompt", soft_wrap=True)

    def on_mount(self) -> None:
        if self.banner:
            self.say(self.banner, "muted")
        self.query_one(PromptArea).focus()

    def on_unmount(self) -> None:
        # A turn still streaming when the app closes would otherwise keep
        # touching widgets that no longer exist.
        self.session.interrupt()

    @property
    def _alive(self) -> bool:
        """False once the app is tearing down and the widgets are gone."""
        return self.is_running and bool(self.query(Transcript))

    # --- output -----------------------------------------------------------

    def _append(
        self, renderable, plain: str | None = None, full: str | None = None
    ) -> Static | None:
        text = plain if plain is not None else _plain(renderable)
        self.transcript_log.append(text)
        self.full_log.append(full if full is not None else text)
        if not self._alive:
            return None
        widget = Static(renderable)
        self.query_one(Transcript).mount(widget)
        self.query_one(Transcript).scroll_end(animate=False)
        return widget

    def say(self, text: str, style: str = "") -> Static | None:
        return self._append(Text(text, style=style) if style else Text(text), text)

    def say_markdown(self, text: str) -> Static | None:
        return self._append(Markdown(text), text)

    # --- streaming --------------------------------------------------------

    def _stream_text(self, chunk: str) -> None:
        if not self._alive:
            return
        if self._stream is None:
            self._stream = self._append(Text(""))
            self._stream_buf = ""
            self._stream_due = 0.0
        self._stream_buf += chunk
        # Rendering markdown on every delta is far too slow; buffer and repaint
        # on a timer instead. The final repaint happens in _end_stream.
        now = time.monotonic()
        if now >= self._stream_due:
            self._stream_due = now + STREAM_REFRESH_S
            self._stream.update(Text(self._stream_buf))
            self.query_one(Transcript).scroll_end(animate=False)

    def _end_stream(self) -> None:
        """Close the current text block, re-rendering it as markdown."""
        if not self._alive:
            self._stream = None
            self._stream_buf = ""
            return
        if self._stream is not None and self._stream_buf.strip():
            self._stream.update(Markdown(self._stream_buf))
            self.transcript_log[-1] = self._stream_buf
            self.full_log[-1] = self._stream_buf
        elif self._stream is not None:
            self._stream.remove()
        self._stream = None
        self._stream_buf = ""
        self.query_one(Transcript).scroll_end(animate=False)

    def on_event_from_session(self, event) -> None:
        if isinstance(event, TextChunk):
            self._stream_text(event.text)
        elif isinstance(event, ToolStarted):
            self._end_stream()
            line = Text.assemble(
                (f"[tool] {event.name} ", "bold"), (event.args, "dim")
            )
            self._append(line, f"[tool] {event.name} {event.args}").add_class("tool")
        elif isinstance(event, ToolFinished):
            marker = "[exit] " if event.is_error else "[ok] "
            # Collapsed to one line by default, with the full text kept. New results follow
            # the current mode.
            short = f"  {marker}{event.text}"
            full = f"  {marker}{(event.full or event.text).strip()}"
            style = "warn" if event.is_error else "muted"
            widget = self._append(
                Text(full if self._verbose_tools else short, style=style),
                plain=short,
                full=full,
            )
            if widget is not None:
                self._tool_widgets.append((widget, short, full))

    # --- approval gate ----------------------------------------------------

    async def ask_approval(self, tool_name: str, detail: str) -> bool:
        self._end_stream()
        self.say(f"[approve (y/n)] {tool_name}: {detail}", "warn")
        self._approval = asyncio.get_running_loop().create_future()
        # Disable the editor while approval is pending, so `y`/`n` reach the app
        # instead of the focused prompt.
        if self._alive:
            self.query_one(PromptArea).disabled = True
        try:
            return await self._approval
        finally:
            self._approval = None
            if self._alive:
                prompt = self.query_one(PromptArea)
                prompt.disabled = False
                prompt.focus()

    def _answer_approval(self, ok: bool) -> None:
        if self._approval and not self._approval.done():
            self.say("  approved" if ok else "  denied", "muted" if ok else "err")
            self._approval.set_result(ok)

    def _copy_selection(self) -> bool:
        """Copy the current selection, if any. Returns True if something went
        to the clipboard."""
        try:
            selected = self.screen.get_selected_text()
        except Exception:
            return False
        if not selected:
            return False
        self.copy_to_clipboard(selected)
        self.clear_selection()
        self.say(f"sent {len(selected)} chars via OSC 52{_clipboard_caveat()}", "muted")
        return True

    # --- key handling -----------------------------------------------------

    async def on_key(self, event) -> None:
        key = event.key

        # Approval mode swallows everything except the answer.
        if self._approval is not None:
            if key == "ctrl+c" and self._copy_selection():
                event.stop()
                event.prevent_default()
                return
            if key in ("y", "Y"):
                self._answer_approval(True)
            elif key in ("n", "N", "escape", "ctrl+c"):
                self._answer_approval(False)
            event.stop()
            event.prevent_default()
            return

        if key == "ctrl+c":
            # Textual binds ctrl+c to screen.copy_text. A selection only exists
            # if the operator dragged one, so treat it as copy intent.
            if self._copy_selection():
                event.stop()
                event.prevent_default()
                return
            event.stop()
            event.prevent_default()
            if self.session.busy:
                self.session.interrupt()
            elif self.query_one(PromptArea).text:
                self.query_one(PromptArea).text = ""
            else:
                self.say("ctrl+c interrupts; press ctrl+d twice (or /exit) to exit", "muted")
            return

        if key == "escape" and self.session.busy:
            event.stop()
            event.prevent_default()
            self.session.interrupt()
            return

        if key == "ctrl+o":
            event.stop()
            event.prevent_default()
            self._toggle_tool_detail()
            return

        if key == "ctrl+d":
            prompt = self.query_one(PromptArea)
            if prompt.text.strip():
                return  # with text present ctrl+d stays delete-forward
            event.stop()
            event.prevent_default()
            now = time.monotonic()
            if now - self._last_ctrl_d <= CTRL_D_WINDOW_S:
                self.exit()
            else:
                self._last_ctrl_d = now
                self.say("press ctrl+d again to exit", "muted")
            return

        if key in ("up", "down") and self.history:
            # Inside multi-line text the arrows move the cursor; only at the first or last
            # line do they recall history. Checked here because the app sees keys before
            # the prompt.
            prompt = self.query_one(PromptArea)
            at_edge = prompt.cursor_at_first_line if key == "up" else prompt.cursor_at_last_line
            if at_edge:
                event.stop()
                event.prevent_default()
                self._recall_history(key)

    def _toggle_tool_detail(self) -> None:
        """Expand or collapse every tool result in the transcript, in place."""
        if not self._tool_widgets:
            self.say("no tool output yet in this session", "muted")
            return
        self._verbose_tools = not self._verbose_tools
        for widget, short, full in self._tool_widgets:
            text = full if self._verbose_tools else short
            style = "warn" if "[exit] " in short else "muted"
            try:
                widget.update(Text(text, style=style))
            except Exception:  # noqa: BLE001 - a removed widget must not break the toggle
                continue
        state = "expanded" if self._verbose_tools else "collapsed"
        self.say(f"tool output {state} ({len(self._tool_widgets)} results)", "muted")
        # Expanding shifts everything below, so scroll to the newest output.
        self.query_one(Transcript).scroll_end(animate=False)

    def _remember(self, line: str) -> None:
        """Record an entry in memory and on disk, skipping consecutive repeats."""
        if not self.history or self.history[-1] != line:
            self.history.append(line)
        self._history_store.append(line)

    def _recall_history(self, key: str) -> None:
        prompt = self.query_one(PromptArea)
        if key == "up":
            self._hist_index = (
                len(self.history) - 1
                if self._hist_index is None
                else max(0, self._hist_index - 1)
            )
        else:
            if self._hist_index is None:
                return
            self._hist_index += 1
            if self._hist_index >= len(self.history):
                self._hist_index = None
                prompt.text = ""
                return
        prompt.text = self.history[self._hist_index]
        # Cursor at the end of the recalled text, on its last line.
        prompt.move_cursor(prompt.document.end)

    # --- submit -----------------------------------------------------------

    async def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        raw = event.text
        self.query_one(PromptArea).text = ""
        self._hist_index = None
        line = raw.strip()
        if not line:
            return

        if self.session.busy:
            # Keep the text recallable so it can be resent after interrupting.
            self._remember(line)
            self.say("[busy] press Esc or ctrl+c to interrupt the current turn", "muted")
            return

        self._remember(line)

        if line.startswith("/"):
            await self.run_command(line)
            return

        self.say(f"you> {line}", "you")

        if line.startswith("!"):
            await self.run_shell(line[1:])
            return

        self._turn = asyncio.create_task(self.run_turn(line))

    # --- turn -------------------------------------------------------------

    async def run_turn(self, line: str) -> None:
        if self._alive:
            self.query_one(PromptArea).placeholder = "thinking... (Esc interrupts)"
        try:
            result = await self.session.run_turn(
                line, self.on_event_from_session, self.ask_approval
            )
            self._end_stream()
            if result.interrupted:
                self.say("[interrupted]", "warn")
            elif result.error:
                self.say(f"[error] {result.error}", "err")
            else:
                u = result.usage
                self.say(
                    f"[{u.input_tokens} in / {u.output_tokens} out, {u.requests} request(s)]",
                    "muted",
                )
        finally:
            if self._alive:
                self.query_one(PromptArea).placeholder = self._placeholder
            self._turn = None

    async def run_shell(self, cmd: str) -> None:
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=SHELL_TIMEOUT_S)
            truncated = len(out) > MAX_OUTPUT
            text = out[:MAX_OUTPUT].decode("utf-8", errors="replace").rstrip()
            if truncated:
                text += f"\n[... truncated at {MAX_OUTPUT} bytes]"
            self.say(text or "(no output)", "" if proc.returncode == 0 else "warn")
        except asyncio.TimeoutError:
            self.say(f"(command timed out after {SHELL_TIMEOUT_S:.0f}s)", "err")
        except Exception as e:
            self.say(str(e), "err")

    # --- commands ---------------------------------------------------------

    async def run_command(self, line: str) -> None:
        parts = shlex.split(line[1:]) if len(line) > 1 else []
        cmd = parts[0] if parts else ""
        arg = " ".join(parts[1:]).strip()

        if cmd in ("exit", "quit"):
            self.exit()
        elif cmd == "help":
            self.say_markdown(HELP)
        elif cmd == "clear":
            self.query_one(Transcript).remove_children()
            self.transcript_log.clear()
            self.full_log.clear()
            self.session.clear()
            self.say("conversation cleared (KWIM memory is untouched)", "muted")
        elif cmd == "model":
            await self.cmd_model(arg)
        elif cmd == "save":
            self.cmd_save(arg)
        elif cmd == "copy":
            self.cmd_copy()
        elif cmd == "tier":
            self.cmd_tier(arg)
        elif cmd == "strategy":
            self.cmd_strategy(arg)
        else:
            self.say(f"unknown command: /{cmd}  (try /help)", "err")

    async def cmd_model(self, arg: str) -> None:
        if arg:
            self.session.set_model(self.rebuild_model(arg))
            self.say(f"model -> {arg}", "muted")
            return
        models = await list_models()
        # Read the live agent: DJED_MODEL still holds the startup model after /model.
        current = getattr(self.session.agent.model, "model_name", None) or model_id()
        if not models:
            self.say(f"current: {current}\n(could not list gateway models)", "muted")
            return
        self.say_markdown(
            f"**current:** `{current}`\n\n" + "\n".join(f"- `{m}`" for m in models)
        )

    def rebuild_model(self, model_name: str):
        from .provider import build_model

        return build_model(model_name)

    def transcript_dump(self) -> str:
        """The whole conversation, with tool output untruncated."""
        return "\n".join(self.full_log)

    def cmd_save(self, arg: str) -> None:
        path = Path(arg).expanduser() if arg else (
            Path.home() / f"djed-transcript-{datetime.now():%Y%m%d-%H%M%S}.txt"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.transcript_dump(), encoding="utf-8")
        except OSError as e:
            self.say(f"could not write {path}: {e}", "err")
            return
        self.say(f"transcript -> {path}", "muted")

    def cmd_copy(self) -> None:
        """Copy the whole transcript. Terminal clipboard support varies; /save writes a
        file instead.
        """
        dump = self.transcript_dump()
        if not dump.strip():
            self.say("nothing to copy yet", "muted")
            return
        self.copy_to_clipboard(dump)
        self.say(f"sent {len(dump)} chars via OSC 52{_clipboard_caveat()}", "muted")

    def cmd_tier(self, arg: str) -> None:
        tiers = getattr(self.session, "tiers", None)
        if not tiers:
            self.say("no MCP tiers loaded", "muted")
            return
        available = [t.name for t in tiers]
        if not arg:
            enabled = self.session.enabled_tiers
            rows = "\n".join(
                f"- `{t.name}` ({t.tool_count} tools) "
                f"{'**on**' if t.name in enabled else 'off'}"
                for t in tiers
            )
            self.say_markdown(
                f"**tiers** (strategy: `{self.session.strategy_label}`)\n\n{rows}"
            )
            return
        selection, err = resolve_selection(arg, available, self.session.enabled_tiers)
        if err:
            self.say(err, "err")
            return
        self.session.set_tiers(selection)
        self.say(
            f"tiers -> {', '.join(sorted(selection)) or 'none'} "
            f"({self.session.active_tool_count} tools)",
            "muted",
        )

    def cmd_strategy(self, arg: str) -> None:
        if arg not in ("eager", "search"):
            self.say(
                f"strategy is `{self.session.strategy_label}`; use eager or search",
                "muted",
            )
            return
        self.session.set_strategy(arg)
        self.say(
            f"strategy -> {arg} ({self.session.active_tool_count} tools exposed)",
            "muted",
        )
