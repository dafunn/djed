"""Unit tests for the runbook seeder: leaky chunks are dropped, and every other
chunk is written.
"""
from pathlib import Path

import httpx
import pytest

import seed_runbook
from seed_runbook import SemanticSeeder, chunk_runbook, seed_file, slug


# --- chunking -------------------------------------------------------------


def test_splits_on_h2_and_h3():
    md = "# Title\nintro\n\n## Alpha\nbody a\n\n### Beta\nbody b\n"
    chunks = chunk_runbook(md)
    assert [t for t, _ in chunks] == ["title", "alpha", "beta"]


def test_content_before_the_first_header_is_kept():
    """A preamble is still knowledge; dropping it loses the doc's framing."""
    md = "# Runbook\nthe cluster lives at the gateway address\n\n## Alpha\nbody\n"
    chunks = chunk_runbook(md)
    assert "the cluster lives at the gateway address" in chunks[0][1]


def test_does_not_split_on_h4_or_deeper():
    """h4+ are sub-points; splitting there yields chunks too small to stand
    alone as a retrieval result."""
    md = "## Alpha\nbody\n\n#### Deep\nsub point\n"
    chunks = chunk_runbook(md)
    assert len(chunks) == 1
    assert "sub point" in chunks[0][1]


def test_repeated_headings_get_distinct_topics():
    """Two sections with the same heading must get distinct topics, or one
    chunk is lost."""
    md = "## Troubleshooting\nfirst\n\n## Other\nx\n\n## Troubleshooting\nsecond\n"
    topics = [t for t, _ in chunk_runbook(md)]
    assert len(topics) == len(set(topics)), "chunk topics must be unique"
    assert "troubleshooting" in topics and "troubleshooting-2" in topics


def test_repeated_headings_keep_both_bodies():
    md = "## Dup\nfirst body\n\n## Dup\nsecond body\n"
    bodies = {t: b for t, b in chunk_runbook(md)}
    assert "first body" in bodies["dup"]
    assert "second body" in bodies["dup-2"]


def test_header_kept_with_its_body():
    md = "## Alerts\nsilence with the alert CLI\n"
    _, body = chunk_runbook(md)[0]
    assert body.startswith("## Alerts")


def test_empty_and_whitespace_input():
    assert chunk_runbook("") == []
    assert chunk_runbook("\n\n   \n") == []


def test_slug_is_bounded_and_clean():
    assert slug("Alert Manager: Silences!") == "alert-manager-silences"
    assert len(slug("x" * 100)) <= 48
    assert slug("###") == ""


# --- fake transport -------------------------------------------------------


class _Recorder:
    def __init__(self, existing: set[str] | None = None, fail: bool = False):
        self.existing = existing or set()
        self.fail = fail
        self.written: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            cid = dict(request.url.params).get("meta.chunk_id")
            return httpx.Response(200, json=[{"id": cid}] if cid in self.existing else [])
        if self.fail:
            return httpx.Response(500, text="boom")
        import json as _json

        self.written.append(_json.loads(request.content))
        return httpx.Response(200, json={"id": "x"})


async def _run(tmp_path: Path, md: str, commit: bool, rec: _Recorder, source="rb"):
    p = tmp_path / "curated.md"
    p.write_text(md, encoding="utf-8")
    async with httpx.AsyncClient(transport=httpx.MockTransport(rec.handler)) as c:
        seeder = SemanticSeeder(c, "http://kwim", "key")
        return await seed_file(seeder, p, source, commit)


# --- leak scan (fail-closed) ----------------------------------------------


@pytest.mark.parametrize("leaky", [
    "## Creds\npassword = hunter2\n",
    "## Key\n-----BEGIN RSA PRIVATE KEY-----\n",
    "## Vault\n$ANSIBLE_VAULT;1.1;AES256\n",
    "## Token\napi_key: abc\n",
])
async def test_leaky_chunks_are_dropped_and_never_written(tmp_path, leaky):
    rec = _Recorder()
    counts = await _run(tmp_path, leaky, commit=True, rec=rec)
    assert counts["dropped"] == 1
    assert rec.written == [], "a leaky chunk must never reach KWIM"


async def test_a_clean_chunk_survives_alongside_a_leaky_one(tmp_path):
    md = "## Safe\nmetrics scrapes every 30s\n\n## Unsafe\npassword = hunter2\n"
    rec = _Recorder()
    counts = await _run(tmp_path, md, commit=True, rec=rec)
    assert counts["dropped"] == 1
    assert len(rec.written) == 1
    assert rec.written[0]["metadata"]["topic"] == "safe"


def test_denylisted_source_is_refused_unopened(tmp_path, monkeypatch):
    """The file must be refused by name - never opened, so a secret in it is
    never read into memory."""
    p = tmp_path / "team-access.md"
    p.write_text("password = hunter2\n", encoding="utf-8")

    def explode(*a, **kw):
        raise AssertionError("denylisted file was opened")

    monkeypatch.setattr(Path, "read_text", explode)
    assert seed_runbook.main([str(p), "--key", "k"]) == 2


# --- writes ---------------------------------------------------------------


async def test_dry_run_writes_nothing(tmp_path):
    rec = _Recorder()
    counts = await _run(tmp_path, "## Alpha\nbody\n", commit=False, rec=rec)
    assert rec.written == []
    assert counts["new"] == 1


async def test_commit_writes_expected_payload(tmp_path):
    rec = _Recorder()
    await _run(tmp_path, "## Alerts\nsilence with the alert CLI\n", commit=True, rec=rec)
    assert len(rec.written) == 1
    w = rec.written[0]
    # The caller-supplied id is what makes the write an upsert.
    assert w["id"] == "rb:alerts"
    assert w["metadata"] == {
        "doc_type": "runbook", "topic": "alerts",
        "source": "rb", "chunk_id": "rb:alerts",
    }
    assert "silence with the alert CLI" in w["content"]


async def test_existing_chunk_is_updated_not_skipped(tmp_path):
    """An existing chunk_id must be replaced, not skipped, so a corrected
    section can be re-seeded."""
    rec = _Recorder(existing={"rb:alpha"})
    counts = await _run(tmp_path, "## Alpha\ncorrected body\n", commit=True, rec=rec)
    assert counts["updated"] == 1
    assert counts["new"] == 0
    assert len(rec.written) == 1, "an existing chunk must still be written"
    assert "corrected body" in rec.written[0]["content"]


async def test_write_failures_are_counted_not_raised(tmp_path):
    rec = _Recorder(fail=True)
    counts = await _run(tmp_path, "## A\nx\n\n## B\ny\n", commit=True, rec=rec)
    assert counts["failed"] == 2
    assert counts["new"] == 0


async def test_ids_are_unique_across_repeated_headings(tmp_path):
    """Two same-named sections must produce two distinct records, not one
    overwriting the other."""
    rec = _Recorder()
    md = "## Dup\nfirst\n\n## Dup\nsecond\n"
    await _run(tmp_path, md, commit=True, rec=rec)
    ids = [w["id"] for w in rec.written]
    assert sorted(ids) == ["rb:dup", "rb:dup-2"]


# --- CLI ------------------------------------------------------------------


def test_missing_file_exits_2(tmp_path):
    assert seed_runbook.main([str(tmp_path / "nope.md"), "--key", "k"]) == 2


def test_missing_key_exits_2(tmp_path, monkeypatch):
    monkeypatch.delenv("DJED_KWIM_API_KEY", raising=False)
    p = tmp_path / "x.md"
    p.write_text("## A\nb\n", encoding="utf-8")
    assert seed_runbook.main([str(p), "--key", ""]) == 2


def test_source_defaults_to_the_file_stem(tmp_path):
    """The source tag defaults to the file stem: a shared constant would make
    two untagged files collide on any heading they have in common."""
    rec = _Recorder()
    p = tmp_path / "monitoring.md"
    p.write_text("## Alpha\nbody\n", encoding="utf-8")

    async def fake_seed(seeder, path, source, commit):
        rec.written.append({"source": source})
        return {"total": 0, "new": 0, "updated": 0, "dropped": 0, "failed": 0}

    seed_runbook.seed_file = fake_seed
    try:
        seed_runbook.main([str(p), "--key", "k"])
    finally:
        seed_runbook.seed_file = seed_file
    assert rec.written[0]["source"] == "monitoring"
