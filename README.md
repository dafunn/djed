# djed

A terminal agent for platform operations, on the
[KWIM](https://github.com/dafunn/kwim) substrate and built on
[Pydantic AI](https://ai.pydantic.dev). It remembers your infrastructure across
sessions, and reaches your cluster only through sandboxed, least-privilege,
read-only MCP servers. The default agent identity is Maat.

## What it is

- **Durable memory.** Every turn is enriched by KWIM: KNN recall over curated
  knowledge, warm-start from prior sessions, and wisdom rules injected into
  the system prompt.
- **MCP bridge.** Tools come from read-only MCP servers, one per backend -
  databases, the cluster, logs. Each has its capability surface pinned by env
  flags and its data scope by allowlists matching the credential's grants.
  Tiers load selectively, or hide behind tool search, to keep the context
  window clear.
- **Sandboxed cluster access.** Cluster reads go through the MCP tiers (RBAC
  `view` only: no mutation, no Secrets), never a mounted kubeconfig. The agent
  runs in a minimal container, and every shell command is operator-approved at
  the TUI before it runs. Against the local fleet that container joins only an
  internal network, so everything it sends or receives passes through one of the
  tools; `podman/compose.example.yml` is the worked topology.
- **An unattended mode.** Hand it a spec file and it provisions a workspace,
  edits, tests and commits on a branch, then stops and reports how it ended -
  no terminal, no approval prompts, a deny-by-default shell and a branch left
  for you to review. See `BUILD.md`.
- **Local-model first.** Drives any OpenAI-compatible endpoint (designed
  against a local LiteLLM gateway), with per-agent spend attribution.

## Layout

- `py/djed/` - the agent harness. Textual TUI, a UI-agnostic session loop (turn
  handling, history, approval gate, interrupt), KWIM memory client, MCP fleet
  bridge, and the tool-loading strategy that decides what reaches the model.
  `persona.py` holds the per-agent parts: a name, an instructions block, and
  the durable tag that LiteLLM bills and KWIM files memory under.
- `tools/ops-ingest/` - sets up djed's memory in KWIM. Python tooling that
  injects operations documents (runbooks, guardrails, inventory) into the djed
  team's KWIM memory as curated knowledge and `:Rule` constraints, for djed's
  agent to recall. It writes through KWIM's HTTP API with djed's team key; it is
  not part of KWIM. Dry-run by default, and safe to re-run
- `py/` - entry points: `tui.py` (terminal), `headless.py` (one-shot),
  `mcp_probe.py` (fleet preflight), plus `py/tests/`
- `sandbox/` - the sandbox image: `python:3.14-slim` + a pinned,
  checksum-verified `kubectl`
- `run_djed.sh`, `py/run_forwards.sh` - dev launchers; see the note below

## Running

Requirements: Python 3.14+, podman, and a reachable KWIM service +
OpenAI-compatible LLM endpoint (see the KWIM repo for the substrate).

`run_djed.sh` reads `~/.djed.env`:

```bash
cp .env.example ~/.djed.env   # fill in endpoints, model name, keys

./run_djed.sh         # sandboxed TUI
./run_djed.sh --fake  # stub model, no backend needed
```

Or drive it directly, passing the env file explicitly:

```bash
cd py
uv sync --frozen --extra dev

.venv/bin/python mcp_probe.py --env-file ~/.djed.env   # check DJED_MCP_SERVERS
.venv/bin/python tui.py --env-file ~/.djed.env         # the terminal
.venv/bin/python headless.py --env-file ~/.djed.env "what pods are in monitoring?"
```

`mcp_probe.py` exits non-zero if a tier is unreachable or the config is
unusable, so it works as a preflight in your own launcher.

`.env.example` documents every knob: endpoints, model, session id, context
window, recall depth, and the MCP server list (SSE or stdio transports). Load it
with `--env-file` rather than shell `source` - shell assignment strips the quotes
inside the JSON server list.

`run_djed.sh` is a development launcher: it port-forwards the in-cluster MCP
services, preflights them, then starts the sandboxed TUI. Adapt it to your
environment.

