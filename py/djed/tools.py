"""The agent's native tools.

Where the history processor provides an always-on retrieval baseline
(warm-start + top-k relevant every turn), `kwim_recall` lets the agent decide
to go deeper when it knows it needs a specific piece of knowledge it lacks.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from pydantic_ai import ApprovalRequired, RunContext

from .kwim import KwimClient
from .skills import Skill, SkillError, read_skill_path, render_loaded

# Prefix on a non-zero-exit tool result. Front ends match on it to style the
# line, so it is defined once here and imported rather than retyped.
EXIT_PREFIX = "[exit "
SHELL_TIMEOUT_S = float(os.environ.get("DJED_SHELL_TIMEOUT_S") or 60)
# The result is sent to the model, so it has to fit a context window: 16 KiB is
# roughly 4k tokens. Raise it with DJED_MAX_TOOL_OUTPUT if you need more.
MAX_OUTPUT = int(os.environ.get("DJED_MAX_TOOL_OUTPUT") or 16 * 1024)
# Which machine an approved command runs on; it selects the scope paragraph in
# `run_command`'s description. `run_djed.sh` sets it per mode. Read at import,
# so it must be a process variable: an --env-file key is loaded after this
# module.
SHELL_SCOPE = (os.environ.get("DJED_SHELL_SCOPE") or "sandbox").strip().lower()


# How a build run may end, other than a budget limit or a crash. A closed set:
# each reason calls for a different operator action.
BLOCKED_REASONS = (
    "missing_toolchain",   # not in the exec image -> rebuild it
    "missing_dependency",  # provisioning's venv step fell short -> re-provision
    "needs_network",       # cannot proceed offline -> a decision, not a fix
    "spec_unclear",        # the spec is ambiguous or contradicts the tree
    "command_timeout",     # run_task's own budget statement -> raise timeout_s, or find out why it got slow
    "command_killed",      # an environment statement, almost always OOM -> more memory, or a smaller -j
)


@dataclass
class Outcome:
    """How a run ended. Written by the stop tools, read by the driver."""

    kind: str            # "complete" | "blocked"
    summary: str
    reason: str = ""     # blocked only; one of BLOCKED_REASONS
    detail: str = ""     # blocked only; the specific tool, command or ambiguity


@dataclass
class Deps:
    """Per-run dependencies handed to every tool."""

    kwim: KwimClient
    session_id: str
    # Set by task_complete / task_blocked, and by the build driver after two
    # consecutive inconclusive run_task results. The driver reads it between turns.
    outcome: Outcome | None = None
    # name -> Skill, from `skills.discover`. Empty means no skill tools.
    skills: dict[str, Skill] = field(default_factory=dict)


async def kwim_recall(ctx: RunContext[Deps], query: str, k: int = 4) -> str:
    """Retrieve relevant knowledge from durable memory (the ops runbook, past
    decisions, infra facts) by semantic similarity.

    Call this first, before answering, when a task touches infrastructure, tooling,
    hosts, ports, or procedures - anything you don't already have in context.
    Do not guess about the platform; recall it.

    Args:
        query: What you need to know, phrased as a search query
            (e.g. 'how to silence a firing alert').
        k: How many chunks to retrieve; keep small (3-5).
    """
    hits = await ctx.deps.kwim.semantic_recall(query, k)
    if hits is None:
        return f"(KWIM is unavailable: {ctx.deps.kwim.last_failure})"
    return "\n\n---\n\n".join(hits) if hits else "(no matching knowledge in KWIM)"


def _unknown_skill(ctx: RunContext[Deps], name: str) -> str:
    known = ", ".join(ctx.deps.skills) or "(none)"
    return f"No skill named {name!r}. Available: {known}."


async def load_skill(ctx: RunContext[Deps], name: str) -> str:
    """Load a skill's instructions. Call this before starting a task that one of
    the skills listed in your instructions covers, then follow what it says.

    The result stays in the conversation, so load each skill once.

    Args:
        name: The skill's name, exactly as listed.
    """
    skill = ctx.deps.skills.get(name)
    if skill is None:
        return _unknown_skill(ctx, name)
    try:
        return render_loaded(skill)
    except (OSError, UnicodeDecodeError) as e:
        return f"Could not read skill {name!r}: {e}"


async def read_skill_file(ctx: RunContext[Deps], name: str, path: str) -> str:
    """Read one of a loaded skill's other files, such as a reference document or
    a template. Read one only when the skill's instructions send you to it.

    Args:
        name: The skill's name.
        path: The file's path inside the skill, as load_skill listed it.
    """
    skill = ctx.deps.skills.get(name)
    if skill is None:
        return _unknown_skill(ctx, name)
    try:
        data = read_skill_path(skill, path).read_bytes()
    except SkillError as e:
        return str(e)
    except OSError as e:
        return f"Could not read {path!r}: {e}"
    text = data[:MAX_OUTPUT].decode("utf-8", errors="replace")
    if len(data) > MAX_OUTPUT:
        text += f"\n[... truncated at {MAX_OUTPUT} bytes]"
    return text


SKILL_TOOLS = [load_skill, read_skill_file]


# The paragraph of the description that depends on the machine.
_SANDBOX_SCOPE = """\
    You are in a container, and it is a narrow one. Its kubectl is
    unauthenticated and it has no route to the internet or to any git host, so a
    cluster command cannot authenticate however it is spelled and a clone cannot
    succeed whichever URL it is given. If a repository seems to be missing, list
    what is cached and refresh it; do not go looking elsewhere, and do not guess
    at a hostname."""

_HOST_SCOPE = """\
    You are on the operator's workstation, not in a container. Commands run as
    their user, with their credentials, their network and their files, and there
    is no boundary underneath you: the approval prompt is the only thing between
    a command and the machine. Nothing here is scoped to you, so a stray path or
    a wrong flag damages real work rather than a disposable sandbox. Keep
    commands narrow, prefer the reversible one, and read before you write."""


async def run_command(ctx: RunContext[Deps], command: str) -> str:
    # Description assembled below - it depends on DJED_SHELL_SCOPE.
    # The approval gate: raising suspends the run and surfaces the call to the
    # operator, who resumes it with an approve/deny decision.
    if not ctx.tool_call_approved:
        raise ApprovalRequired(metadata={"command": command})

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    return await collect_output(proc)


# Assigned rather than written as a literal because Pydantic AI reads `__doc__`
# for the tool description, and one of its paragraphs is chosen at import.
run_command.__doc__ = """Run a shell command and return its stdout/stderr.

    The last resort, not the first. Anything an MCP tool covers, that tool does
    better: it is already scoped to what you may read, it needs no approval, and
    it returns a shaped answer rather than text you then have to parse. Reach for
    a shell only when nothing else can do the job.

    Two categories in particular are already covered, and a shell is the worse
    way through both:

    - The cluster. Listing, reading, describing, logs, events, node and pod usage
      all have tools, and they run against a credential scoped to exactly what
      you may see.
    - Repositories. Reading files, listing trees, globbing, grepping, history and
      diffs all have tools, served from local mirrors that are already cloned and
      kept fresh. Do not clone a repository a tool already serves.

{scope}

    Use it for what the tools genuinely do not cover, and for the ops actions you
    are asked to perform. The operator sees and approves every command, so
    propose the exact one you want; do not ask permission in prose.

    Args:
        command: The exact shell command to run, e.g. 'ls -la'.
    """.format(scope=_HOST_SCOPE if SHELL_SCOPE == "host" else _SANDBOX_SCOPE)


async def collect_output(proc) -> str:
    """Wait for a spawned process, capping its time and its output.

    The ops shell's own supervision. Two known weaknesses: it discards output
    on timeout, and it kills only the direct child rather than the whole
    process group.
    """
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=SHELL_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return f"(command timed out after {SHELL_TIMEOUT_S:.0f}s)"

    truncated = len(out) > MAX_OUTPUT
    text = out[:MAX_OUTPUT].decode("utf-8", errors="replace").strip()
    if truncated:
        text += f"\n[... truncated at {MAX_OUTPUT} bytes]"
    # A non-zero exit is returned as content, not raised: the model needs to read
    # stderr ("kubectl: not found", "no such pod") to reason about what to do next.
    if proc.returncode != 0:
        return f"{EXIT_PREFIX}{proc.returncode}]\n{text or '(no output)'}"
    return text or "(no output)"


def _already(outcome: Outcome) -> str:
    """Answer a second stop call within the same turn. The first outcome stands; the
    driver only checks between turns.
    """
    return (
        f"Already recorded as {outcome.kind!r}; the first call stands. "
        "Stop now - the run ends when this turn does."
    )


async def task_complete(ctx: RunContext[Deps], summary: str) -> str:
    """Declare the task finished. Call this once the spec is satisfied and the
    checks named in WORKSPACE.md pass.

    Args:
        summary: What you changed, in two or three sentences.
    """
    if ctx.deps.outcome is not None:
        return _already(ctx.deps.outcome)
    ctx.deps.outcome = Outcome(kind="complete", summary=summary)
    return "Recorded. The run ends after this turn."


async def task_blocked(
    ctx: RunContext[Deps], reason: str, detail: str, summary: str
) -> str:
    """Declare that you cannot finish, and why. Use this rather than guessing or
    stopping silently - the run is resumed later from your commits.

    Args:
        reason: One of missing_toolchain, missing_dependency, needs_network,
            spec_unclear, command_timeout, command_killed.
        detail: The specific thing - a tool name, a command, or the ambiguity.
        summary: What you did complete, and what remains.
    """
    # Returned as content rather than raised: a wrong reason is a mistake the
    # model can correct on the next call, exactly as a non-zero exit is.
    if reason not in BLOCKED_REASONS:
        return (
            f"Unknown reason {reason!r}. Use one of: {', '.join(BLOCKED_REASONS)}."
        )
    if ctx.deps.outcome is not None:
        return _already(ctx.deps.outcome)
    ctx.deps.outcome = Outcome(
        kind="blocked", summary=summary, reason=reason, detail=detail
    )
    return "Recorded. The run ends after this turn."
