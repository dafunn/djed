"""Spec parsing: unqualified repo names, colliding checkout basenames and unsafe
task_ids are rejected.
"""
from __future__ import annotations

import tomllib

import pytest

from djed.spec import (
    Budget,
    SpecError,
    parse_spec,
    render_commands_toml,
    render_spec,
    slug,
    split_frontmatter,
)

GOOD = """+++
task_id = "demo-task"

[[repo]]
repo = "acme/widget"
ref = "main"
mode = "write"

[[repo]]
repo = "acme/shared-lib"
mode = "read"

[budget]
max_iterations = 12
+++

# Task

Do the thing.
"""


def test_parses_repos_prose_and_budget():
    m, prose = parse_spec(GOOD)
    assert m.task_id == "demo-task"
    assert [r.qualified for r in m.write_repos] == ["acme/widget"]
    assert [r.qualified for r in m.read_repos] == ["acme/shared-lib"]
    assert prose.startswith("# Task")
    assert m.budget.max_iterations == 12
    # Unset budget keys keep their defaults rather than becoming zero.
    assert m.budget.max_tokens == Budget().max_tokens


def test_ref_defaults_to_main_but_is_honoured_when_given():
    m, _ = parse_spec(GOOD)
    assert {r.name: r.ref for r in m.repos} == {
        "widget": "main",
        "shared-lib": "main",
    }


def test_only_the_prose_is_returned():
    """The frontmatter is provisioning's input and must not reach the model."""
    _, prose = parse_spec(GOOD)
    assert "task_id" not in prose
    assert "widget" not in prose


@pytest.mark.parametrize(
    "body, expected",
    [
        ("no fence at all\n", "frontmatter fence"),
        ('+++\ntask_id = "x"\n', "unterminated"),
        ('+++\ntask_id = "x"\n+++\n\n# T\n', "at least one"),
        ('+++\n[[repo]]\nrepo = "a/b"\nmode = "write"\n+++\n\n# T\n', "needs a task_id"),
        ('+++\ntask_id = "x" = broken\n+++\n\n# T\n', "not valid TOML"),
    ],
)
def test_structural_errors_name_the_problem(body, expected):
    with pytest.raises(SpecError, match=expected):
        parse_spec(body)


def _spec(*repo_blocks: str, task_id: str = "t") -> str:
    return f'+++\ntask_id = "{task_id}"\n' + "".join(repo_blocks) + "+++\n\n# T\n\nDo it.\n"


def _repo(name: str, mode: str = "write") -> str:
    return f'\n[[repo]]\nrepo = "{name}"\nmode = "{mode}"\n'


def test_repo_must_be_owner_qualified():
    """More than one owner is cached, so a bare name is ambiguous."""
    with pytest.raises(SpecError, match="owner-qualified"):
        parse_spec(_spec(_repo("widget")))


def test_over_qualified_repo_is_rejected():
    with pytest.raises(SpecError, match="owner-qualified"):
        parse_spec(_spec(_repo("forge/acme/widget")))


def test_unknown_mode_is_rejected():
    with pytest.raises(SpecError, match="mode"):
        parse_spec(_spec(_repo("a/b", mode="readwrite")))


def test_the_same_repo_twice_is_rejected():
    with pytest.raises(SpecError, match="listed twice"):
        parse_spec(_spec(_repo("a/b"), _repo("a/b", mode="read")))


def test_basename_collision_across_owners_is_rejected():
    """Checkout directories are named by basename, so these would collide."""
    with pytest.raises(SpecError, match="basename"):
        parse_spec(_spec(_repo("acme/shared"), _repo("globex/shared", mode="read")))


def test_a_read_only_workspace_is_rejected():
    """A run with nothing to change cannot produce a result."""
    with pytest.raises(SpecError, match='mode = "write"'):
        parse_spec(_spec(_repo("a/b", mode="read")))


def test_spec_with_no_prose_is_rejected():
    with pytest.raises(SpecError, match="no task prose"):
        parse_spec('+++\ntask_id = "t"\n[[repo]]\nrepo = "a/b"\nmode = "write"\n+++\n')


@pytest.mark.parametrize(
    "task_id",
    ["../escape", "has space", "UPPER", "", "/abs", "x" * 65, ".leading-dot"],
)
def test_unsafe_task_ids_are_rejected(task_id):
    """task_id becomes a directory, a branch segment and a container name."""
    with pytest.raises(SpecError):
        parse_spec(_spec(_repo("a/b"), task_id=task_id))


def test_budget_rejects_nonsense_rather_than_running_forever():
    base = '+++\ntask_id = "t"\n[[repo]]\nrepo = "a/b"\nmode = "write"\n'
    with pytest.raises(SpecError, match="positive"):
        parse_spec(base + "[budget]\nmax_iterations = 0\n+++\n\n# T\n\nGo.\n")
    with pytest.raises(SpecError, match="unknown keys"):
        parse_spec(base + "[budget]\nmax_iteration = 5\n+++\n\n# T\n\nGo.\n")


def test_split_frontmatter_tolerates_leading_blank_lines():
    front, prose = split_frontmatter('\n\n+++\na = 1\n+++\nbody\n')
    assert front.strip() == "a = 1"
    assert prose == "body"


def test_the_per_turn_request_cap_is_part_of_the_budget():
    """`max_requests_per_turn` bounds one turn; `max_iterations` counts turns."""
    base = '+++\ntask_id = "t"\n[[repo]]\nrepo = "a/b"\nmode = "write"\n'
    m, _ = parse_spec(base + "[budget]\nmax_requests_per_turn = 7\n+++\n\n# T\n\nGo.\n")
    assert m.budget.max_requests_per_turn == 7
    assert Budget().max_requests_per_turn > 0
    with pytest.raises(SpecError, match="positive"):
        parse_spec(base + "[budget]\nmax_requests_per_turn = 0\n+++\n\n# T\n\nGo.\n")


# --- init: prose in, builder spec out ------------------------------------


def test_render_round_trips_through_the_parser():
    """A generated spec is validated with `parse_spec`."""
    text = render_spec(
        "# Task\n\nDo the thing.",
        "demo-task",
        [("acme/widget", "main", "write"), ("acme/lib", "v2", "read")],
        {"max_iterations": 12},
    )
    m, prose = parse_spec(text)
    assert m.task_id == "demo-task"
    assert [r.qualified for r in m.write_repos] == ["acme/widget"]
    assert [(r.qualified, r.ref) for r in m.read_repos] == [("acme/lib", "v2")]
    assert m.budget.max_iterations == 12
    assert prose == "# Task\n\nDo the thing."


def test_a_default_ref_is_left_out_and_an_explicit_one_is_written():
    """A generated spec writes only what differs from the defaults."""
    text = render_spec("Go.", "t", [("a/b", "main", "write")])
    assert "ref =" not in text
    assert 'ref = "v2"' in render_spec("Go.", "t", [("a/b", "v2", "write")])


def test_render_refuses_what_the_build_path_would_reject():
    """Generation rejects what parsing would reject."""
    with pytest.raises(SpecError, match="basename"):
        render_spec("Go.", "t", [("a/shared", "main", "write"),
                                 ("b/shared", "main", "read")])
    with pytest.raises(SpecError, match="task_id"):
        render_spec("Go.", "Not A Task Id", [("a/b", "main", "write")])
    with pytest.raises(SpecError, match='mode = "write"'):
        render_spec("Go.", "t", [("a/b", "main", "read")])
    with pytest.raises(SpecError, match="empty"):
        render_spec("   \n", "t", [("a/b", "main", "write")])


def test_an_unset_budget_emits_no_table():
    """A budget of defaults and no budget parse the same."""
    text = render_spec("Go.", "t", [("a/b", "main", "write")], {"max_tokens": None})
    assert "[budget]" not in text
    assert parse_spec(text)[0].budget == Budget()


# --- [commands]: declared, always wins over provisioning's own detection ---


def _spec_with_commands(*repo_blocks: str, commands_block: str = "") -> str:
    """Like `_spec`, but the `[commands...]` text lands inside the
    frontmatter fence rather than after it."""
    return (
        f'+++\ntask_id = "t"\n' + "".join(repo_blocks) + commands_block
        + "+++\n\n# T\n\nDo it.\n"
    )


def test_commands_are_parsed_keyed_by_repo_then_task():
    text = _spec_with_commands(
        _repo("acme/widget"),
        commands_block=(
            '\n[commands.widget]\n'
            'build = { argv = ["cmake", "--build", "build"], timeout_s = 900 }\n'
            'configure = { argv = ["cmake", "-S", ".", "-B", "build"], phase = "provision" }\n'
        ),
    )
    m, _ = parse_spec(text)
    assert set(m.commands["widget"]) == {"build", "configure"}
    assert m.commands["widget"]["build"].argv == ("cmake", "--build", "build")
    assert m.commands["widget"]["build"].timeout_s == 900
    assert m.commands["widget"]["configure"].phase == "provision"


def test_commands_default_to_no_phase_and_the_default_timeout():
    text = _spec_with_commands(
        _repo("acme/widget"),
        commands_block='\n[commands.widget]\ntest = { argv = ["pytest"] }\n',
    )
    m, _ = parse_spec(text)
    cmd = m.commands["widget"]["test"]
    assert cmd.phase == ""
    assert cmd.timeout_s == 600


def test_a_spec_with_no_commands_block_has_an_empty_commands_dict():
    m, _ = parse_spec(GOOD)
    assert m.commands == {}


def test_commands_naming_a_repo_not_listed_as_write_are_rejected():
    """A read-mode repo cannot have commands."""
    text = _spec_with_commands(
        _repo("acme/widget"), _repo("acme/lib", mode="read"),
        commands_block='\n[commands.lib]\ntest = { argv = ["pytest"] }\n',
    )
    with pytest.raises(SpecError, match="not listed with"):
        parse_spec(text)


def test_commands_naming_an_unknown_repo_are_rejected():
    text = _spec_with_commands(
        _repo("acme/widget"),
        commands_block='\n[commands.nope]\ntest = { argv = ["x"] }\n',
    )
    with pytest.raises(SpecError, match="not listed with"):
        parse_spec(text)


@pytest.mark.parametrize(
    "block, expected",
    [
        ('[commands.widget]\nbuild = { argv = [] }\n', "non-empty argv"),
        ('[commands.widget]\nbuild = { argv = "x" }\n', "non-empty argv"),
        ('[commands.widget]\nbuild = { argv = ["x"], phase = "bogus" }\n', "phase"),
        ('[commands.widget]\nbuild = { argv = ["x"], timeout_s = 0 }\n', "positive"),
        ('[commands.widget]\nbuild = { argv = ["x"], timeout_s = "soon" }\n', "non-numeric"),
        ('[commands.widget]\n', "at least one command"),
    ],
)
def test_malformed_commands_are_rejected(block, expected):
    text = _spec_with_commands(_repo("acme/widget"), commands_block="\n" + block)
    with pytest.raises(SpecError, match=expected):
        parse_spec(text)


def test_render_commands_toml_omits_defaults_and_round_trips():
    text = _spec_with_commands(
        _repo("acme/widget"),
        commands_block=(
            '\n[commands.widget]\n'
            'build = { argv = ["cmake", "--build", "build"] }\n'
            'test  = { argv = ["ctest"], timeout_s = 300 }\n'
        ),
    )
    m, _ = parse_spec(text)
    rendered = render_commands_toml(m.commands)
    # The default timeout is not written out - only what differs from it.
    assert "timeout_s = 600" not in rendered
    assert "timeout_s = 300" in rendered
    reparsed = tomllib.loads(rendered)
    assert reparsed["widget"]["build"]["argv"] == ["cmake", "--build", "build"]
    assert reparsed["widget"]["test"]["timeout_s"] == 300


def test_render_commands_toml_of_nothing_declared_is_empty():
    assert render_commands_toml({}) == ""


def test_slug_produces_something_the_task_id_pattern_accepts():
    assert slug("Add Thumbnails (v2)") == "add-thumbnails-v2"
    assert slug("  Fix THE Bug  ") == "fix-the-bug"
    # A stem with nothing usable reduces to empty, and the caller must name the
    # task.
    assert slug("___") == ""
    assert len(slug("x" * 200)) == 64
