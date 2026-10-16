#!/usr/bin/env bash
# Run a headless build: provision a workspace, start the tools it needs, drive
# the agent, tear the tools down.
#
#   ./run_build.sh task.md                 # provision, then build
#   ./run_build.sh task.md --resume        # keep the tree and its commits
#   ./run_build.sh task.md --fresh         # discard the tree and start over
#   ./run_build.sh task.md -- --trace      # extra args pass through to build.py
#   ./run_build.sh gc <task-id>            # remove one finished workspace
#   ./run_build.sh gc --older-than 7       # remove workspaces older than N days
#
# Three phases with decreasing privilege: provision.sh holds a route to the
# network and clones, branches and installs; this script then starts a
# task-scoped base-mcp and an agent container that have neither.
#
# The servers are per-task: the fleet's shared base-mcp serves the TUI's
# workspace, and per-task containers let two builds run at once.
#
# Workspaces are kept on success for the operator to review and push; `gc`
# removes them.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${DJED_SANDBOX_IMAGE:-localhost/djed-sandbox}"
# The shared provisioning/exec toolchain - not the agent's image above.
EXEC_IMAGE="${DJED_EXEC_IMAGE:-localhost/exec-mcp:latest}"
# The two socat forwards (model and memory) and nothing else - see the
# `gateways` network in podman/compose.example.yml.
# The fleet's compose project name; its networks and volumes are prefixed by it.
PROJECT="${DJED_PROJECT:-djed}"
GATEWAYS="${DJED_GATEWAYS_NETWORK:-${PROJECT}_gateways}"
BASE_IMAGE="${DJED_BASE_MCP_IMAGE:-localhost/base-mcp:latest}"
BUILD_ROOT="${DJED_BUILD_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/djed/builds}"
ENV_FILE="${DJED_ENV_FILE:-$HOME/.djed.env}"
PY="$HERE/py/.venv/bin/python"

# --- gc -------------------------------------------------------------------

if [ "${1:-}" = gc ]; then
  shift
  if [ "${1:-}" = --older-than ]; then
    days="${2:?usage: run_build.sh gc --older-than <days>}"
    find "$BUILD_ROOT" -mindepth 1 -maxdepth 1 -type d -mtime "+$days" \
      -print -exec rm -rf {} + 2>/dev/null || true
    exit 0
  fi
  task="${1:?usage: run_build.sh gc <task-id>}"
  case "$task" in */*|..|.) echo "gc: refusing unsafe task id: $task" >&2; exit 2 ;; esac
  rm -rf "${BUILD_ROOT:?}/$task"
  echo "gc: removed $BUILD_ROOT/$task" >&2
  exit 0
fi

SPEC=""
PROVISION_ARGS=()
BUILD_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    # Provisioning's flags; build.py does not take them.
    --resume|--fresh) PROVISION_ARGS+=("$1"); shift ;;
    --) shift; BUILD_ARGS+=("$@"); break ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) SPEC="$1"; shift ;;
  esac
done

[ -n "$SPEC" ] || { echo "usage: run_build.sh <spec-file> [--resume|--fresh] [-- args]" >&2; exit 2; }
[ -f "$SPEC" ] || { echo "run_build: no such spec file: $SPEC" >&2; exit 2; }
[ -x "$PY" ] || { echo "run_build: no venv at $PY (see py/README.md)" >&2; exit 2; }
[ -f "$ENV_FILE" ] || { echo "run_build: no env file at $ENV_FILE" >&2; exit 2; }
SPEC="$(cd "$(dirname "$SPEC")" && pwd)/$(basename "$SPEC")"

if ! podman network exists "$GATEWAYS"; then
  echo "run_build: network '$GATEWAYS' is missing. Re-create the fleet:" >&2
  echo "    podman-compose -p $PROJECT -f $HERE/podman/compose.example.yml up -d" >&2
  exit 1
fi

# wall_clock_s becomes the exec MCP client's timeout: a run_task can run for an
# hour.
SPEC_MANIFEST="$("$PY" -m djed.spec manifest "$SPEC")"
TASK_ID="$(awk -F'\t' '$1=="task_id"{print $2}' <<<"$SPEC_MANIFEST")"
[ -n "$TASK_ID" ] || { echo "run_build: spec has no task_id" >&2; exit 1; }
WALL_CLOCK_S="$(awk -F'\t' '$1=="budget" && $2=="wall_clock_s"{print $3}' <<<"$SPEC_MANIFEST")"

# A `localhost/` tag must already be on disk: build and tag it from its
# repository. Any other image is pulled every run, so a rebuilt `:latest` is
# picked up. Checked before provisioning, which runs on the exec image.
ensure_image() {
  local image="$1" var="$2"
  if [ "${image#localhost/}" != "$image" ]; then
    # podman never resolves `localhost/` to a registry, so require it on disk
    # rather than attempt a pull.
    if ! podman image exists "$image"; then
      echo "run_build: $image is a local tag and is not on disk." >&2
      echo "run_build: build and tag it, or set $var to something pullable." >&2
      exit 1
    fi
  elif ! podman pull -q "$image" >/dev/null 2>&1; then
    # On a failed pull, use an image already on disk, and say so.
    if podman image exists "$image"; then
      echo "run_build: could not pull $image; using the copy on disk" >&2
    else
      echo "run_build: could not pull $image and no local copy exists." >&2
      echo "run_build: check the registry login, or set $var." >&2
      exit 1
    fi
  fi
}
ensure_image "$EXEC_IMAGE" DJED_EXEC_IMAGE
ensure_image "$BASE_IMAGE" DJED_BASE_MCP_IMAGE

TASK_DIR="$(DJED_EXEC_IMAGE="$EXEC_IMAGE" "$HERE/podman/provision.sh" "$SPEC" "${PROVISION_ARGS[@]}" | tail -1)"
[ -d "$TASK_DIR" ] || { echo "run_build: provisioning produced no workspace" >&2; exit 1; }

BASE_NAME="djed-build-${TASK_ID}-base"
EXEC_NAME="djed-build-${TASK_ID}-exec"
# One internal network per task: this run's containers can address each other
# and nothing else, and base-mcp has no route off it, so its `net_fetch` is
# inert.
TASK_NETWORK="djed-build-${TASK_ID}"
SNAPSHOTS="$TASK_DIR/snapshots"
# A single file, not $TASK_DIR: mounting the directory would hand exec-mcp
# `snapshots/`. Created here so the log outlives `podman rm -f`.
EXEC_LOG="$TASK_DIR/exec.log"
touch "$EXEC_LOG"

# Where exec-mcp spills a command's full output. Mounted rather than created by
# the container, which runs unprivileged under a root-owned `/`. Outside every
# checkout so it cannot dirty a git status.
RUN_LOGS_DIR="$TASK_DIR/exec-runs"
mkdir -p "$RUN_LOGS_DIR"

# A bundle of each write repo's branch, mounted into neither container so the
# agent cannot reach its own backups. Taken before and after the agent runs;
# the "before" one protects commits a resumed run inherits.
snapshot_repos() {
  local when="$1" stamp name branch count out
  stamp="$(date +%Y%m%d-%H%M%S)"
  for d in "$TASK_DIR"/ws/src/*/; do
    [ -d "$d/.git" ] || continue
    name="$(basename "$d")"
    branch="djed/$TASK_ID"
    git -C "$d" rev-parse --verify --quiet "$branch" >/dev/null 2>&1 || continue
    # Commits the agent made, i.e. those the clone source does not have.
    count="$(git -C "$d" rev-list --count "$branch" --not --remotes 2>/dev/null || echo 0)"
    [ "$count" -gt 0 ] 2>/dev/null || continue
    out="$SNAPSHOTS/${name}-${stamp}-${when}.bundle"
    if git -C "$d" bundle create "$out" "$branch" >/dev/null 2>&1; then
      echo "run_build: snapshot $name ($count commit(s)) -> snapshots/$(basename "$out")" >&2
    fi
  done
}

# Containers first: a network with an attached container refuses to go.
cleanup() {
  podman rm -f "$BASE_NAME" "$EXEC_NAME" >/dev/null 2>&1 || true
  podman network rm "$TASK_NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM
cleanup

podman network create --internal "$TASK_NETWORK" >/dev/null

# base-mcp, scoped to this task. `/refs` is mounted read-only, and
# FS_READONLY_ROOTS says the same in the server so `fs_roots` advertises it.
# NET_ALLOW_HOSTS is set empty explicitly rather than left to the image default.
podman run -d --rm --name "$BASE_NAME" \
  --userns=keep-id --user "$(id -u):$(id -g)" --network "$TASK_NETWORK" \
  -v "$TASK_DIR/ws:/workspace:z" \
  -v "$TASK_DIR/refs:/refs:ro,z" \
  -e MCP_TRANSPORT=sse -e MCP_HOST=0.0.0.0 -e MCP_PORT=3000 \
  -e FS_ALLOW_ROOTS=/workspace,/refs \
  -e FS_READONLY_ROOTS=/refs \
  -e EXPOSE_WRITE=true -e AUDIT_LOG=true \
  -e NET_ALLOW_HOSTS= \
  "$BASE_IMAGE" >/dev/null

for _ in $(seq 1 40); do
  podman exec "$BASE_NAME" python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/healthz')" \
    >/dev/null 2>&1 && break
  sleep 0.25
done

# exec-mcp, scoped to this task. No --env-file: it holds no credentials. No
# source mount: the image determines what runs. ws/ read-write, the manifest
# and log as single files, no /app and no /refs.
podman run -d --rm --name "$EXEC_NAME" \
  --userns=keep-id --user "$(id -u):$(id -g)" --network "$TASK_NETWORK" \
  -v "$TASK_DIR/ws:/workspace:z" \
  -v "$TASK_DIR/manifest.toml:/manifest.toml:ro,z" \
  -v "$EXEC_LOG:/exec.log:z" \
  -v "$RUN_LOGS_DIR:/exec-runs:z" \
  -e MCP_TRANSPORT=sse -e MCP_HOST=0.0.0.0 -e MCP_PORT=3000 \
  "$EXEC_IMAGE" >/dev/null

for _ in $(seq 1 40); do
  podman exec "$EXEC_NAME" python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/healthz')" \
    >/dev/null 2>&1 && break
  sleep 0.25
done

snapshot_repos before

# The agent. No /state, no kubeconfig, /app read-only. Two networks: the task
# network for this run's servers, `gateways` for the model and memory forwards.
# Not `djed_tools`: a project's tests run its code, which could open a socket
# to pg-mcp or k8s-mcp whether or not the model was offered them.
set +e
podman run --rm --userns=keep-id \
  --network "$TASK_NETWORK" --network "$GATEWAYS" \
  -v "$HERE:/app:ro" \
  -v "$TASK_DIR/ws:/workspace:z" \
  -v "$TASK_DIR/refs:/refs:ro,z" \
  -v "$SPEC:/spec.md:ro,z" \
  --env-file "$ENV_FILE" \
  -e DJED_MCP_SERVERS="[{\"name\":\"base\",\"transport\":\"sse\",\"url\":\"http://$BASE_NAME:3000/sse\"},{\"name\":\"exec\",\"transport\":\"sse\",\"url\":\"http://$EXEC_NAME:3000/sse\",\"timeout\":$WALL_CLOCK_S}]" \
  -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp \
  -w /app/py "$IMAGE" \
  python build.py /spec.md --workspace /workspace "${BUILD_ARGS[@]}"
STATUS=$?
set -e

snapshot_repos after

case "$STATUS" in
  0) echo "run_build: complete. Review and push from $TASK_DIR/ws/src" >&2 ;;
  2) echo "run_build: blocked. Fix, then: ./run_build.sh $SPEC --resume" >&2 ;;
  3) echo "run_build: budget exhausted. Resume with --resume, or discard with gc" >&2 ;;
  *) echo "run_build: failed. Workspace kept at $TASK_DIR" >&2 ;;
esac
# Named, not tailed: each result already printed as it happened.
echo "run_build: exec-mcp's invocation log -> $EXEC_LOG" >&2
exit "$STATUS"
