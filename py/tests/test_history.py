"""Tests for input history that outlives the session."""
from __future__ import annotations

import json
import os

import pytest

from djed.history import History, default_path


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert History(tmp_path / "nope").load() == []


def test_round_trip(tmp_path):
    h = History(tmp_path / "history")
    h.append("what pods are failing?")
    h.append("why did nginx restart?")
    assert History(tmp_path / "history").load() == [
        "what pods are failing?",
        "why did nginx restart?",
    ]


def test_multiline_entries_survive(tmp_path):
    """The prompt joins backslash-continued lines, so entries contain newlines."""
    h = History(tmp_path / "history")
    h.append("check this\nand that")
    assert History(tmp_path / "history").load() == ["check this\nand that"]


def test_consecutive_duplicates_are_dropped(tmp_path):
    h = History(tmp_path / "history")
    h.append("same")
    h.append("same")
    h.append("different")
    h.append("same")   # not consecutive any more, so it is kept
    assert h.load() == ["same", "different", "same"]


def test_blank_entries_are_ignored(tmp_path):
    h = History(tmp_path / "history")
    h.append("   ")
    h.append("")
    assert h.load() == []


def test_entries_are_capped_keeping_the_newest(tmp_path):
    h = History(tmp_path / "history", limit=3)
    for i in range(6):
        h.append(f"line {i}")
    assert h.load() == ["line 3", "line 4", "line 5"]


def test_a_damaged_line_costs_only_that_line(tmp_path):
    p = tmp_path / "history"
    p.write_text('"good one"\nnot json at all\n"good two"\n', encoding="utf-8")
    assert History(p).load() == ["good one", "good two"]


def test_file_is_owner_only(tmp_path):
    """The history file is readable by its owner only."""
    p = tmp_path / "history"
    History(p).append("something sensitive")
    assert oct(p.stat().st_mode & 0o777) == "0o600"


def test_write_failure_does_not_raise(tmp_path):
    """A session that cannot persist history should still run."""
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    os.chmod(unwritable, 0o500)
    try:
        History(unwritable / "history").append("still fine")
    finally:
        os.chmod(unwritable, 0o700)


def test_no_temp_files_left_behind(tmp_path):
    h = History(tmp_path / "history")
    h.append("one")
    h.append("two")
    assert [f.name for f in tmp_path.iterdir() if ".tmp-" in f.name] == []


def test_default_path_prefers_the_explicit_override(monkeypatch):
    monkeypatch.setenv("DJED_HISTORY_FILE", "/state/history")
    assert str(default_path()) == "/state/history"


def test_default_path_falls_back_to_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("DJED_HISTORY_FILE", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert default_path() == tmp_path / "djed" / "history"
