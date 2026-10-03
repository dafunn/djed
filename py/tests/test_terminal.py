"""tmux environment checks."""
from __future__ import annotations

from djed.terminal import tmux_notes


def fake_get(**values):
    return lambda option: values.get(option)


def test_no_notes_outside_tmux():
    assert tmux_notes(env={}, get=fake_get(**{"escape-time": "500"})) == []


def test_slow_escape_time_is_flagged():
    notes = tmux_notes(
        env={"TMUX": "/tmp/tmux-1000/default,1,0"},
        get=fake_get(**{"escape-time": "500", "set-clipboard": "on"}),
    )
    assert len(notes) == 1
    assert "escape-time" in notes[0] and "500ms" in notes[0]


def test_fast_escape_time_is_fine():
    notes = tmux_notes(
        env={"TMUX": "x"},
        get=fake_get(**{"escape-time": "10", "set-clipboard": "on"}),
    )
    assert notes == []


def test_external_clipboard_is_flagged():
    """tmux 3.x defaults to 'external', which silently drops OSC 52 from apps."""
    notes = tmux_notes(
        env={"TMUX": "x"},
        get=fake_get(**{"escape-time": "10", "set-clipboard": "external"}),
    )
    assert len(notes) == 1
    assert "set-clipboard" in notes[0]
    assert "/save" in notes[0], "should point at the fix that needs no config"


def test_both_problems_are_reported():
    notes = tmux_notes(
        env={"TMUX": "x"},
        get=fake_get(**{"escape-time": "500", "set-clipboard": "external"}),
    )
    assert len(notes) == 2


def test_unavailable_tmux_options_are_ignored():
    assert tmux_notes(env={"TMUX": "x"}, get=lambda o: None) == []


def test_unparseable_escape_time_is_ignored():
    notes = tmux_notes(env={"TMUX": "x"}, get=fake_get(**{"escape-time": "wat"}))
    assert notes == []


# --- clipboard blockers ---------------------------------------------------


from djed.terminal import clipboard_blocker


def test_tmux_external_clipboard_is_named_as_the_blocker():
    msg = clipboard_blocker(env={"TMUX": "x"}, get=lambda o: "external")
    assert msg and "set-clipboard" in msg and "/save" in msg


def test_tmux_with_clipboard_on_is_not_flagged():
    assert clipboard_blocker(env={"TMUX": "x"}, get=lambda o: "on") is None


def test_ssh_is_named_as_a_likely_blocker():
    assert "SSH" in (
        clipboard_blocker(env={"SSH_TTY": "/dev/pts/1"}, get=lambda o: None) or ""
    )


def test_plain_local_terminal_has_no_known_blocker():
    assert clipboard_blocker(env={}, get=lambda o: None) is None
