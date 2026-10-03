"""Wire contract between exec-mcp and the build driver.

Stdlib-only, so exec-mcp can carry the same constants without pydantic_ai; they
are duplicated in exec-mcp's `prefixes.py`. The authoritative list is
exec-mcp's `wire_contract.toml`, which both sides' tests read
(`py/tests/test_wire_contract.py` here, via `tools/check-wire-contract.sh`).

`run_task`'s result opens with one header line, `"{PREFIX} {task_name}: {short}"`:

- TASK_DONE_PREFIX: the process ran to completion, exit 0 or a failure. Also
  used when the binary is missing; `short` then carries the spawn error.
- TASK_TIMEOUT_PREFIX: exec-mcp's timeout fired.
- TASK_KILLED_PREFIX: the process was killed by something else, usually the
  OOM killer.

`short` is "ok", "exit {code}", "timed out at {N}s", "killed by {SIGNAL}", or the
spawn error. Two consecutive timeout or killed results force `task_blocked`; a
done result resets the count.
"""
from __future__ import annotations

TASK_DONE_PREFIX = "[task done]"
TASK_TIMEOUT_PREFIX = "[task timeout]"
TASK_KILLED_PREFIX = "[task killed]"
