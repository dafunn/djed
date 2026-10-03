#!/usr/bin/env bash
# Run the Djed TUI with port-forwards to the MCP tiers. Forwards are torn down
# by the EXIT trap.
#
#   ./run_djed.sh                     # sandboxed TUI against the local podman fleet
#   ./run_djed.sh --forwards          # ...against the in-cluster fleet, port-forwarded
#   ./run_djed.sh --up                # bring the local fleet up first, then the TUI
#   ./run_djed.sh --host              # workstation, no container
#   ./run_djed.sh --fake              # stub model, no gateway or cluster
#   ./run_djed.sh -- --tiers cluster  # extra args pass through to tui.py
#   ./run_djed.sh --host --env-file ~/.djed.host.env   # host TUI, host addresses
set -eo pipefail

IMAGE=localhost/djed-sandbox
# Host-side state for the sandboxed TUI (input history); the container
# filesystem is discarded on exit.
STATE_DIR="${DJED_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/djed}"
# Created by the podman fleet's compose project (`podman-compose -p djed`).
NETWORK="${DJED_TOOLS_NETWORK:-djed_tools}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=sandbox
# `local`: the podman fleet on the `tools` network; `forwards`: the in-cluster
# services, port-forwarded to loopback.
FLEET=local
# Opt-in, and nothing here takes the fleet down: its volumes hold the clone
# cache and the workspace.
BRING_UP=0
FAKE=0
NO_MCP=0
# Read by the preflight and the TUI alike. The default names endpoints on the
# fleet's internal network; --host needs one whose addresses resolve on the
# workstation.
ENV_FILE="${DJED_ENV_FILE:-$HOME/.djed.env}"
ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)   MODE=host; shift ;;
    --forwards) FLEET=forwards; shift ;;
    --up)     BRING_UP=1; shift ;;
    --fake)   FAKE=1; ARGS+=(--fake); shift ;;
    --no-mcp)   NO_MCP=1; ARGS+=(--no-mcp); shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --env-file=*) ENV_FILE="${1#*=}"; shift ;;
    --)       shift; ARGS+=("$@"); break ;;
    *)        ARGS+=("$1"); shift ;;
  esac
done

# Checked here rather than left to podman, whose --env-file failure names the
# flag and not the mode that chose the path.
if [ ! -r "$ENV_FILE" ]; then
  echo "error: env file '$ENV_FILE' does not exist or is not readable." >&2
  echo "see .env.example for the keys it needs." >&2
  exit 1
fi

pids=()
cleanup() {
  [ "${#pids[@]}" -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The local fleet is reached by container name over the `tools` network; check
# it is up before starting a TUI that would otherwise have no tools. Sandbox
# mode only: --host cannot reach it either way.
if [ "$FAKE" -eq 0 ] && [ "$NO_MCP" -eq 0 ] && [ "$FLEET" = local ] &&
   [ "$MODE" = sandbox ]; then
  if [ "$BRING_UP" -eq 1 ]; then
    "$REPO/podman/setup-fleet.sh"
  fi
  if ! podman network exists "$NETWORK"; then
    echo "error: podman network '$NETWORK' does not exist." >&2
    echo "start the fleet first:  ./podman/setup-fleet.sh   (or re-run with --up)" >&2
    echo "(or pass --forwards to use the in-cluster fleet instead)" >&2
    exit 1
  fi
  # A torn-down project leaves the network behind, so check for containers too.
  running=$(podman ps --filter "label=io.podman.compose.project=djed" --format '{{.Names}}' | wc -l)
  if [ "$running" -eq 0 ]; then
    echo "error: the '$NETWORK' network exists but no fleet containers are running." >&2
    echo "start them with:  ./podman/setup-fleet.sh   (or re-run with --up)" >&2
    exit 1
  fi
  # Off host networking, localhost is the container itself.
  if grep -q "DJED_MCP_SERVERS=.*\(localhost\|127\.0\.0\.1\)" "$ENV_FILE" 2>/dev/null; then
    echo "warning: DJED_MCP_SERVERS still points at localhost; the local fleet is reached" >&2
    echo "         by container name, e.g. http://pg-mcp:3000/sse - see .env.example" >&2
  fi
fi

# --host: the fleet's container names do not resolve on the workstation and the
# rootless fleet publishes no ports, so refuse an env file whose hosts do not
# resolve here.
url_host() {
  printf '%s\n' "$1" | sed -E 's|^[a-z]+://||; s|/.*$||; s|:[0-9]+$||; s|^\[||; s|\]$||'
}

# Only names are looked up: `getent hosts` on an address literal is a PTR
# lookup, which many addresses have no record for.
resolves() {
  case "$1" in
    *:*)       return 0 ;;                          # IPv6 literal
    *[!0-9.]*) getent hosts "$1" >/dev/null 2>&1 ;; # a name
    *)         return 0 ;;                          # IPv4 literal
  esac
}

if [ "$MODE" = host ] && [ "$FAKE" -eq 0 ]; then
  for var in KWIM_LLM_BASE_URL KWIM_BASE_URL; do
    url=$(sed -n "s/^${var}=//p" "$ENV_FILE" | tail -1)
    [ -n "$url" ] || continue
    if ! resolves "$(url_host "$url")"; then
      echo "error: $var names a host that does not resolve on this workstation." >&2
      echo "       '$ENV_FILE' is written for the sandbox, whose endpoints live on" >&2
      echo "       the fleet's internal network. --host needs an env file whose" >&2
      echo "       addresses resolve here:" >&2
      echo "         ./run_djed.sh --host --env-file ~/.djed.host.env" >&2
      exit 1
    fi
  done
  # The MCP servers publish no ports, so the test is whether the file names
  # something local.
  if [ "$NO_MCP" -eq 0 ] &&
     grep -q '^DJED_MCP_SERVERS=[^[:space:]]' "$ENV_FILE" 2>/dev/null &&
     ! grep -q '^DJED_MCP_SERVERS=.*\(localhost\|127\.0\.0\.1\)' "$ENV_FILE" 2>/dev/null; then
    echo "error: DJED_MCP_SERVERS names no host-reachable address, and the fleet" >&2
    echo "       publishes no ports - its servers are reachable only from inside" >&2
    echo "       its own network. Pass --no-mcp to run the shell without them, or" >&2
    echo "       publish the fleet to loopback and point this at those ports." >&2
    exit 1
  fi
fi

# --fake needs neither the cluster nor the gateway, so skip the forwards.
if [ "$FAKE" -eq 0 ] && [ "$NO_MCP" -eq 0 ] && [ "$FLEET" = forwards ]; then
  echo "starting port-forwards to the djed MCP tiers..." >&2
  kubectl port-forward -n djed svc/pg-mcp    3001:3000 >/tmp/djed-pf-db.log      2>&1 & pids+=($!)
  kubectl port-forward -n djed svc/k8s-mcp   3002:3000 >/tmp/djed-pf-cluster.log 2>&1 & pids+=($!)
  kubectl port-forward -n djed svc/loki-mcp  3003:3000 >/tmp/djed-pf-logs.log    2>&1 & pids+=($!)
  kubectl port-forward -n djed svc/forgejo-mcp 3004:3000 >/tmp/djed-pf-forge.log 2>&1 & pids+=($!)
  kubectl port-forward -n djed svc/repo-mcp  3005:3000 >/tmp/djed-pf-repos.log   2>&1 & pids+=($!)
  # The LLM gateway is reached directly; no forward needed.

  # Wait (up to ~8s each) for the forwards to accept connections. Logs are in
  # /tmp/djed-pf-*.log.
  for port in 3001 3002 3003 3004 3005; do
    for _ in $(seq 1 32); do
      if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then break; fi
      sleep 0.25
    done
  done
fi

# Preflight the MCP servers from wherever the TUI will run: inside the image on
# the tools network for the local fleet, on the workstation for --host and
# --forwards.
if [ "$FAKE" -eq 0 ] && [ "$NO_MCP" -eq 0 ]; then
  if [ "$MODE" = sandbox ] && [ "$FLEET" = local ]; then
    if podman image exists "$IMAGE"; then
      if ! podman run --rm --userns=keep-id --network "$NETWORK" \
          -v "$REPO:/app:ro,z" -v "$ENV_FILE:/env:ro,z" \
          -w /app/py "$IMAGE" python mcp_probe.py --env-file /env >&2; then
        echo "MCP preflight failed - fix the above, or pass --no-mcp to start anyway." >&2
        exit 1
      fi
    fi
  elif [ -x "$REPO/py/.venv/bin/python" ]; then
    if ! "$REPO/py/.venv/bin/python" "$REPO/py/mcp_probe.py" \
        --env-file "$ENV_FILE" >&2; then
      echo "MCP preflight failed - fix the above, or pass --no-mcp to start anyway." >&2
      exit 1
    fi
  fi
fi

if [ "$MODE" = host ]; then
  echo "WARNING: --host runs run_command unsandboxed, as your user." >&2
  # Selects run_command's description. A process variable, not an env-file
  # key: the description is built at import, before --env-file is parsed.
  export DJED_SHELL_SCOPE=host
  exec "$REPO/py/.venv/bin/python" "$REPO/py/tui.py" \
    --env-file "$ENV_FILE" "${ARGS[@]}"
fi

if ! podman image exists "$IMAGE"; then
  echo "building $IMAGE (first run only)..." >&2
  podman build -t "$IMAGE" -f "$REPO/sandbox/Dockerfile.python" "$REPO"
fi

# No kubeconfig mount; kubectl here is unauthed, for run_command only. On the
# local fleet the container joins only the `tools` network - the MCP servers by
# name, and nothing else. `--forwards` keeps host networking because the
# port-forwards bind to loopback. --env-file is podman's parser, not a shell,
# so DJED_MCP_SERVERS stays quoted.
#
# --fake loads no tiers and drives a stub model, so it needs no network at all:
# joining `tools` would demand a fleet that need not exist. The KWIM client
# fails soft, so the session runs with no recall. --no-mcp still reaches the
# model gateway, which on the local fleet is a container on `tools`.
if [ "$FAKE" -eq 1 ]; then
  NET_ARGS=(--network none)
elif [ "$FLEET" = local ]; then
  NET_ARGS=(--network "$NETWORK")
else
  NET_ARGS=(--network host)
fi

mkdir -p "$STATE_DIR"

# The checkout is mounted read-only: an approved `run_command` must not reach
# it. History and HOME go to /state; PYTHONDONTWRITEBYTECODE stops __pycache__
# writes.
exec podman run --rm -it --userns=keep-id "${NET_ARGS[@]}" \
  -v "$REPO:/app:ro,z" \
  -v "$STATE_DIR:/state:z" \
  -e DJED_HISTORY_FILE=/state/history \
  -e HOME=/state \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e DJED_SHELL_SCOPE=sandbox \
  -w /app/py --env-file "$ENV_FILE" \
  "$IMAGE" python tui.py "${ARGS[@]}"
