# Running a headless build

Djed's headless build mode: given a spec file, it edits, tests and commits on a
branch unattended, then stops and reports how it ended. There is no terminal and
no approval prompt.

## Prerequisites

- The MCP fleet up, which provides the gateways network and the repository cache
  volume the build path clones from.
- The sandbox image built. `./run_djed.sh` builds it on first run, or:
  `podman build -t djed-sandbox -f sandbox/Dockerfile.python .`
- The exec-mcp and base-mcp images, built from their own repositories. The
  defaults are `localhost/exec-mcp:latest` and `localhost/base-mcp:latest`,
  which must already be on disk: build and tag each from its repository. To use
  a registry image instead, set `DJED_EXEC_IMAGE` / `DJED_BASE_MCP_IMAGE`;
  `run_build.sh` then pulls it every run.
- An env file at `~/.djed.env` with a key and a reachable model gateway. The
  agent runs in a container on an internal network, so its endpoints must be the
  ones reachable from there - see `.env.example`.
- The venv at `py/.venv`, which `run_build.sh` runs the driver from:
  `cd py && uv sync --frozen --extra dev`

## Run it

```bash
./run_build.sh /path/to/spec.md                      # provision, then build
./run_build.sh /path/to/spec.md -- --model MODEL     # pick the model
./run_build.sh /path/to/spec.md --resume             # keep the tree and its commits
./run_build.sh /path/to/spec.md --fresh              # discard the tree, start over
./run_build.sh gc <task-id>                          # remove one finished workspace
./run_build.sh gc --older-than 7                     # remove workspaces older than N days
```

`--resume` and `--fresh` are provisioning's, and go ahead of the `--`.
Everything following `--` is passed to `build.py`. The two combine:

```bash
./run_build.sh /path/to/spec.md --resume -- --model MODEL --trace
```

Model selection is `--model`, or `DJED_MODEL` in the env file. The flag is
per-run and leaves the interactive terminal's default alone.
`headless.py --list-models` prints what your gateway serves; a name the gateway
does not know fails at the first request, not at startup.

## The spec file

TOML frontmatter fenced by `+++`, then prose. **Only the prose reaches the
model** - the frontmatter is provisioning's input and is stripped before the
agent sees anything.

```toml
+++
task_id = "widget-thumbnails"

[[repo]]
repo = "acme/widget"
ref = "main"
mode = "write"

[[repo]]
repo = "acme/shared-lib"
mode = "read"

[budget]
max_iterations = 25
max_tokens = 400000
wall_clock_s = 3600
max_requests_per_turn = 30
+++

# Task

What to do, written for someone who has the tree in front of them and no other
context. Name the checks that have to pass.
```

Rules the parser enforces:

| key | requirement |
| --- | --- |
| `task_id` | lowercase alphanumeric start, then `a-z0-9._-`, 64 chars max. It becomes a directory name, a branch segment and a container name. |
| `[[repo]]` | at least one, and at least one with `mode = "write"`. |
| `repo` | owner-qualified `owner/name`. More than one owner can be cached, so a bare name is ambiguous. |
| `mode` | `write` (cloned to `/workspace/src/<name>`) or `read` (cloned to `/refs/<name>`, mounted read-only). |
| `ref` | optional, defaults to `main`. |
| basenames | must not collide across owners - the checkout directory is the basename. |
| `[budget]` | optional; every key optional; all values positive. Unknown keys are rejected. |
| `[commands.<repo>]` | optional, covers write repos only; see below. |

Validate without running anything:

```bash
py/.venv/bin/python -m djed.spec manifest /path/to/spec.md   # TSV: task_id, budget, one row per repo
py/.venv/bin/python -m djed.spec prose    /path/to/spec.md   # exactly what the model is given
py/.venv/bin/python -m djed.spec commands /path/to/spec.md   # declared [commands], as runtime manifest TOML
```

### Declaring the command set

Provisioning detects `CMakeLists.txt` and Python (`pyproject.toml` or
`requirements.txt`) on its own. For anything else - or to override detection -
declare it:

```toml
[commands.widget]
build = { argv = ["cmake", "--build", "build"], timeout_s = 900 }
test  = { argv = ["ctest"], timeout_s = 300 }
```

`argv` is required and never a shell string - no pipes, redirection or
chaining, one argv per command. `phase = "provision"` marks a command that
needs the network and runs during provisioning (dependency resolution,
`cmake` configure); everything else runs offline, later, in exec-mcp.
`timeout_s` defaults to 600. A declared `[commands.<repo>]` table always wins
over detection for that repo; a repo matching neither CMake nor Python - or
matching both, which is treated the same as neither - gets no commands and a
warning, and the run proceeds edit-only.

### Starting from prose you already have

`init` adds the frontmatter to a prose file you already have:

```bash
py/.venv/bin/python -m djed.spec init notes.md \
    --write owner/name \
    --read owner/other@v2 \
    --max-iterations 25 \
    -o task.md
```

- `--write` is repeatable and at least one is required; `--read` is repeatable
  and optional. Both take `owner/name` or `owner/name@ref`.
- `task_id` defaults to the prose file's name reduced to a legal id
  (`Add Thumbnails.prose.md` becomes `add-thumbnails`); `--task-id` overrides it.
- Budget flags mirror the `[budget]` keys and are all optional. Keys you do not
  set are left out rather than written as defaults.
- Without `-o` it writes to stdout; with it, it refuses to overwrite unless you
  pass `--force`.

Repos cannot be inferred from prose, so they are the only required input.

The generated file is checked with `parse_spec` before it is written, so the
same rejections apply: a basename collision between two owners, a `task_id`
that is not a safe path segment, a manifest with nothing writable. `init`
refuses a file that already has frontmatter.

## Where repos come from

Not from your working copy. Provisioning clones from the repository cache the
fleet maintains, mounted read-only, which mirrors the owners it is configured
for. A repo that exists only on your disk cannot be built until it is pushed and
mirrored.

The cache refreshes on its own loop. At clone time, provisioning uses the
cache's remote-tracking refs, since a cache checkout's working branch can lag
what it has fetched.

## What the agent gets

- **Files** through the `base` MCP tier: `fs_read`, `fs_read_many`, `fs_glob`,
  `fs_grep`, `fs_list`, `fs_stat`, `fs_write`, `fs_edit`, `fs_move`,
  `fs_delete`, `fs_mkdir`, `fs_copy`, `fs_roots`, `net_fetch`. Reads are size-
  and match-capped. `net_fetch`'s reach depends on the server's allowed-hosts
  setting, which the build path does not set.
- **Execution and git** through exec-mcp, a separate server holding no
  credentials and no route off the task network:
  - `list_tasks(repo)` names what is available and what each runs;
    `run_task(repo, name)` runs one - the whole suite or nothing, no filter
    or argument passthrough. The argv comes from the manifest, not from the
    model, so there is no command string to compose or escape.
  - `git_status`, `git_diff`, `git_add`, `git_commit`, `git_log` - a fixed,
    hand-written set, language-agnostic. `git_add`'s `paths` is the one
    model-supplied argument in the whole surface, and it is confined to the
    repo root.
  - Every call is logged to the task's `exec.log` - resolved argv, exit code
    or signal, duration - whether it succeeds, fails, or is refused.
  - Two consecutive inconclusive `run_task` results (a timeout or a kill, in
    either order) end the run as `task_blocked`.
- **Memory** through `kwim_recall`.
- **Two stop tools**, `task_complete` and `task_blocked`. A run ends when one is
  called, a budget trips, or exec-mcp forces it as above.

`.git` is hidden from the file tools by policy, so git is reachable only
through exec-mcp's git tools. Git identity and the branch are configured
during provisioning, so the first commit cannot fail on either.

**Only Python dependencies are installed.** Provisioning builds a `.venv` and
installs `pytest` when it finds `pyproject.toml` or `requirements.txt`. There is
no `npm ci` or `cargo fetch` step, and the agent container has no network to add
one later - a Node or Rust project will report `missing_dependency` or
`needs_network` and stop.

### What a build run can reach

`run_task` runs the project's own code, which the agent can modify, so exec-mcp
is treated as untrusted even though it holds no credentials. Isolation is by
network membership, per container:

- **The agent** joins a per-task network (this run's base-mcp and exec-mcp,
  nothing else) and the gateways network (the model and memory forwards,
  nothing else). Its environment carries the gateway and memory keys, and
  both gateways are reachable from it by design.
- **exec-mcp** joins the per-task network only: no gateways network, no
  credentials, no route off the host.
- **base-mcp** joins the per-task network only, same as exec-mcp.

None of the three joins the fleet's main tools network, which carries the tiers
holding live database and cluster credentials. Every network here is
`internal`; the only routes off the host are the two forwards on the gateways
network, which only the agent joins.

## What the agent reads

Provisioning writes `WORKSPACE.md` into the workspace and the driver puts it in
the prompt ahead of the task prose. It names each writable checkout with its
branch, HEAD and commit count, the exec-mcp task names available for it (or
says plainly that none were declared or detected, and the repo can be edited
but not verified), each read-only reference, and the standing rules. The
agent is never told how the tree arrived.

On a resumed run it also gets `HANDOFF.md` - its own account, from the previous
run, of why that run stopped.

## Where the work lands

```
<build-root>/<task_id>/ws/src/<name>     the checkout, on branch djed/<task_id>
<build-root>/<task_id>/ws/WORKSPACE.md   the handoff the agent reads
<build-root>/<task_id>/ws/SPEC.md        the prose it was given
<build-root>/<task_id>/refs/<name>       read-mode repos, read-only
<build-root>/<task_id>/snapshots/        git bundles, taken before and after
<build-root>/<task_id>/manifest.toml     the command manifest, mounted read-only into exec-mcp
<build-root>/<task_id>/exec.log          every run_task/git call, one JSON object per line
```

The build root defaults to `$XDG_STATE_HOME/djed/builds`; override it with
`DJED_BUILD_ROOT`.

**Nothing is pushed and nothing is deleted on success.** Review in `ws/src`,
push yourself, then `gc`.

The snapshots are bundles of each write repo's branch, taken before and after
the run and mounted into neither container. The "before" bundle preserves
commits a resumed run inherited.

## How a run ends

| exit | meaning | next |
| --- | --- | --- |
| 0 | `task_complete` | review and push from `ws/src` |
| 1 | error - bad spec, missing workspace, MCP failure | fix and re-run |
| 2 | `task_blocked` | fix the cause, then `--resume` |
| 3 | budget exhausted | `--resume`, or discard with `gc` |

`task_blocked` carries one of six reasons, each mapping to a different fix.
The last two can be forced by exec-mcp itself, after two consecutive
inconclusive results, without the model choosing to report them:

| reason | the fix |
| --- | --- |
| `missing_toolchain` | not in the exec image - rebuild it |
| `missing_dependency` | provisioning's venv step fell short - re-provision |
| `needs_network` | cannot proceed offline - a decision, not a fix |
| `spec_unclear` | the spec is ambiguous or contradicts the tree |
| `command_timeout` | a budget statement - raise `timeout_s` in `[commands]`, or find out why it got slow |
| `command_killed` | an environment statement, almost always the OOM killer - more memory, or a smaller `-j` |

## Budgets

Defaults: 40 iterations, 400k tokens, 3600s wall clock, 30 requests per turn.
Set them per-spec in `[budget]`, or override iterations for a probe run with
`--max-iterations`.

`max_requests_per_turn` bounds a single turn: `agent.run()` loops tool calls
until the model stops making them, which `max_iterations` does not see.

## Flags

`run_build.sh`'s own flags (not passed through `--`):

```
--resume             keep the tree and its commits; re-describe only, touches no git
--fresh              delete the workspace and start over
```

`build.py` (after `--`):

```
--model NAME        override DJED_MODEL
--max-iterations N  override the spec's budget
--trace             per-request tokens, finish reason and response shape
--no-exec           load no exec tier at all: edits only, zero exec risk
--tiers NAME        MCP tiers to load (default: base, plus exec if it connected)
--tool-strategy     eager | search
--no-mcp            skip the MCP bridge
--fake              scripted stub model, fully offline - smoke-tests the plumbing
--agent TAG         persona tag (default: builder)
```

`--trace` shows each request's tokens and finish reason, including responses
that are only thinking and were truncated at `DJED_MAX_TOKENS`, which otherwise
render as nothing.

## When it does not start

| message | cause |
| --- | --- |
| `run_build: network ... is missing` | the fleet is down |
| `run_build: no env file at ...` | no `~/.djed.env`, or `DJED_ENV_FILE` points elsewhere |
| `run_build: no venv at ...` | no `py/.venv` |
| `provision: no repo cache volume` | the fleet has never been up |
| `provision: <owner>/<name> is not in the cache` | the repo is not mirrored, or the owner is not configured |
| `provision: <ws> exists; use --resume or --fresh` | a workspace for this `task_id` is already there |
| `spec: ...` | the frontmatter is malformed; the message names the key |
| `workspace: ... WORKSPACE.md` | running `build.py` directly without provisioning first |
