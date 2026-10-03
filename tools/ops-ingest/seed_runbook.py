"""Seed a curated runbook file into KWIM semantic memory.

Chunks on ## / ### headers into SemanticItems. Fail-closed: a chunk matching a
credential pattern is dropped, and a denylisted filename is never opened.

    python seed_runbook.py knowledge/monitoring.md            # dry run
    python seed_runbook.py knowledge/monitoring.md --commit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

from common import (
    is_denylisted,
    kwim_key_from_env,
    kwim_url_from_env,
    scan_for_secrets,
)

log = logging.getLogger("ops.seed_runbook")

# Split before a ## or ### header so each header keeps its body. Deliberately
# not common.chunk_markdown, which splits on every level #..######: h4+ in a
# runbook are sub-points, and breaking on them yields chunks too small to stand
# alone as a retrieval result.
_SPLIT_RE = re.compile(r"\n(?=#{2,3} )")
_HEADER_RE = re.compile(r"^#+\s*")


def slug(s: str) -> str:
    """Lowercase, hyphenate, trim to 48 chars - stable and greppable."""
    s = re.sub(r"[^a-z0-9]+", "-", s.lower())
    return s.strip("-")[:48]


def chunk_runbook(md: str) -> list[tuple[str, str]]:
    """Split into (topic, body) on ## / ### headers.

    A repeated heading gets an occurrence suffix, so two sections with the same
    title get distinct `chunk_id`s.
    """
    out: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for block in _SPLIT_RE.split(md):
        body = block.strip()
        if not body:
            continue
        header = _HEADER_RE.sub("", body.split("\n", 1)[0]).strip()
        topic = slug(header) or "section"
        seen[topic] = seen.get(topic, 0) + 1
        if seen[topic] > 1:
            topic = f"{topic}-{seen[topic]}"
        out.append((topic, body))
    return out


class SemanticSeeder:
    """Writes runbook chunks to KWIM's semantic store. Team comes from the key."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    async def exists(self, chunk_id: str) -> bool:
        """Metadata-only lookup (q omitted). Fail-soft: unknown reads as absent,
        which only affects whether a write is reported as new or updated."""
        try:
            r = await self.client.get(
                f"{self.base_url}/v1/memory/semantic",
                params={"meta.chunk_id": chunk_id, "limit": 1},
                headers=self.headers,
            )
            if r.status_code != 200:
                return False
            return bool(r.json())
        except Exception as e:
            log.debug("exists() failed for %s (fail-soft): %s", chunk_id, e)
            return False

    async def write(self, chunk_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Upsert the chunk.

        The caller-supplied `id` makes this an update-in-place: KWIM's
        materialize_semantic MERGEs on id, so a corrected section can be
        re-seeded without retiring its source tag first.
        """
        r = await self.client.post(
            f"{self.base_url}/v1/memory/semantic",
            json={"id": chunk_id, "content": content, "metadata": metadata},
            headers=self.headers,
        )
        r.raise_for_status()


async def seed_file(
    seeder: SemanticSeeder | None, path: Path, source: str, commit: bool,
) -> dict[str, int]:
    """Chunk, scan, and (if committing) write. Returns a counts summary."""
    md = path.read_text(encoding="utf-8")
    counts = {"total": 0, "new": 0, "updated": 0, "dropped": 0, "failed": 0}

    for topic, body in chunk_runbook(md):
        counts["total"] += 1
        chunk_id = f"{source}:{topic}"

        leaks = scan_for_secrets(body)
        if leaks:
            counts["dropped"] += 1
            log.error("  drop   %s (leak scan: %s)", chunk_id, ", ".join(leaks))
            continue

        # Dry run makes no network calls, so it cannot know new vs update.
        already = await seeder.exists(chunk_id) if (seeder and commit) else False
        verb = "update" if already else "new"

        if not commit or seeder is None:
            log.info("  %-6s %s  (~%d tok)", verb, chunk_id, round(len(body) / 4))
            counts["updated" if already else "new"] += 1
            continue

        try:
            await seeder.write(
                chunk_id,
                body,
                {"doc_type": "runbook", "topic": topic, "source": source,
                 "chunk_id": chunk_id},
            )
            counts["updated" if already else "new"] += 1
            log.info("  ok     %s  (~%d tok, %s)", chunk_id, round(len(body) / 4), verb)
        except Exception as e:
            counts["failed"] += 1
            log.error("  fail   %s  %s", chunk_id, e)

    return counts


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # httpx logs every request at INFO; that is noise in an operator plan.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    path = Path(os.path.expanduser(args.file))

    # Denylist first, before the file is ever opened.
    if is_denylisted(path):
        log.error("refusing to read denylisted source file: %s", path)
        return 2
    if not path.is_file():
        log.error("no such file: %s", path)
        return 2

    # Default the source tag to the file stem, not a shared constant
    # "runbook" - two files seeded without a tag would otherwise share a source
    # and collide on any heading they have in common.
    source = args.source or slug(path.stem) or "runbook"

    api_key = args.key or kwim_key_from_env()
    if not api_key:
        log.error("--key or DJED_KWIM_API_KEY is required")
        return 2

    if not args.commit:
        log.warning("Dry run - nothing written. Review the plan and pass --commit to seed.")

    if args.commit:
        async with httpx.AsyncClient(timeout=30.0) as client:
            seeder = SemanticSeeder(client, args.kwim_url, api_key)
            counts = await seed_file(seeder, path, source, True)
    else:
        counts = await seed_file(None, path, source, False)

    log.info(
        "\n%s: %d new, %d updated, %d dropped, %d failed of %d chunks (source=%s)",
        "seeded" if args.commit else "would seed",
        counts["new"], counts["updated"], counts["dropped"], counts["failed"],
        counts["total"], source,
    )
    return 1 if counts["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="seed_runbook",
        description="Seed a curated runbook slice into KWIM semantic memory",
    )
    ap.add_argument("file", help="curated markdown file to seed")
    ap.add_argument("--source", default=None,
                    help="source tag for chunk ids (default: the file stem)")
    ap.add_argument("--kwim-url", default=kwim_url_from_env(), help="KWIM service base URL")
    ap.add_argument("--key", default=kwim_key_from_env(), help="DJED KWIM API key")
    ap.add_argument("--commit", action="store_true",
                    help="actually write chunks (default is dry-run)")
    return asyncio.run(_amain(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
