"""The djed release this checkout is.

djed runs from its checkout - the sandbox image carries only the dependencies
and run_djed.sh bind-mounts the checkout over them - so the version comes from
the checkout at startup, not from an image build. In order:

- `DJED_VERSION`, for a copy with no git history to read. A process variable,
  not an env-file key: `--version` resolves it before --env-file is read;
- `git describe --tags --match 'djed-v*'` on the checkout: the nearest release
  tag, with a `-<n>-g<sha>` suffix for commits after it;
- `dev`, when neither gives one.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

TAG_PREFIX = "djed-v"
_CHECKOUT = Path(__file__).resolve().parent


def _describe(cwd: Path = _CHECKOUT) -> str:
    """The checkout's nearest release tag, prefix stripped, or "" if it has none."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "describe", "--tags", "--match", f"{TAG_PREFIX}*"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip().removeprefix(TAG_PREFIX)


@functools.cache
def release_version() -> str:
    return os.environ.get("DJED_VERSION", "").strip() or _describe() or "dev"
