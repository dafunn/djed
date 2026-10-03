#!/usr/bin/env bash
# Port-forwards to the MCP tiers, without the podman/TUI launch. Torn down on
# exit. The LLM gateway needs no forward; KWIM_LLM_BASE_URL reaches it directly.
#
#   ./run_forwards.sh                      # hold open until ctrl+c
#   ./run_forwards.sh <command> [args...]  # run a command with them up
set -eo pipefail

pids=()
cleanup() {
  [ "${#pids[@]}" -gt 0 ] && kill "${pids[@]}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting port-forwards to the djed MCP tiers..." >&2
kubectl port-forward -n djed svc/pg-mcp    3001:3000 >/tmp/djed-pf-db.log      2>&1 & pids+=($!)
kubectl port-forward -n djed svc/k8s-mcp   3002:3000 >/tmp/djed-pf-cluster.log 2>&1 & pids+=($!)
kubectl port-forward -n djed svc/loki-mcp  3003:3000 >/tmp/djed-pf-logs.log    2>&1 & pids+=($!)
kubectl port-forward -n djed svc/forgejo-mcp 3004:3000 >/tmp/djed-pf-forge.log 2>&1 & pids+=($!)
kubectl port-forward -n djed svc/repo-mcp  3005:3000 >/tmp/djed-pf-repos.log   2>&1 & pids+=($!)

for port in 3001 3002 3003 3004 3005; do
  for _ in $(seq 1 32); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then break; fi
    sleep 0.25
  done
done
echo "forwards up" >&2

if [ "$#" -gt 0 ]; then
  "$@"
else
  echo "ctrl+c to tear down." >&2
  wait
fi
