#!/usr/bin/env bash
# Build a workspace for a headless build run.
#
# The first of three stages, and the only one holding a credential or a route
# to the network: it clones, branches, and installs each repo's dependencies,
# writes WORKSPACE.md for the agent, then exits.
#
#   ./provision.sh task.md              # clone, branch, install, describe
#   ./provision.sh task.md --resume     # re-describe only; touches no git
#   ./provision.sh task.md --fresh      # delete the workspace and start over
#
# Repos come from repo-mcp's clone cache, mounted read-only.
#
# `--no-hardlinks`: the cache is a volume mount and the task directory a bind
# mount, so hardlinks fail with "Invalid cross-device link". `--shared` would
# break the moment the build container runs without /cache.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# The exec-mcp toolchain image, not the agent's djed-sandbox.
IMAGE="${DJED_EXEC_IMAGE:-localhost/djed-exec}"
# The fleet's compose project name prefixes the volume it creates.
CACHE_VOLUME="${DJED_REPO_CACHE:-${DJED_PROJECT:-djed}_repo-cache}"
BUILD_ROOT="${DJED_BUILD_ROOT:-${XDG_STATE_HOME:-$HOME/.local/state}/djed/builds}"
PY="$REPO/py/.venv/bin/python"

SPEC=""
MODE=new

while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) MODE=resume; shift ;;
    --fresh)  MODE=fresh; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) SPEC="$1"; shift ;;
  esac
done

[ -n "$SPEC" ] || { echo "usage: provision.sh <spec-file> [--resume|--fresh]" >&2; exit 2; }
[ -f "$SPEC" ] || { echo "provision: no such spec file: $SPEC" >&2; exit 2; }
[ -x "$PY" ] || { echo "provision: no venv at $PY (see py/README.md)" >&2; exit 2; }
SPEC="$(cd "$(dirname "$SPEC")" && pwd)/$(basename "$SPEC")"

# The manifest is parsed by the module that validates it; bash has no TOML.
MANIFEST="$("$PY" -m djed.spec manifest "$SPEC")"
TASK_ID="$(printf '%s\n' "$MANIFEST" | awk -F'\t' '$1=="task_id"{print $2}')"
[ -n "$TASK_ID" ] || { echo "provision: spec has no task_id" >&2; exit 1; }

TASK_DIR="$BUILD_ROOT/$TASK_ID"
WS="$TASK_DIR/ws"
REFS="$TASK_DIR/refs"
# Mounted into neither ws/ nor refs/, so the agent cannot reach its own backups.
SNAPSHOTS="$TASK_DIR/snapshots"

if [ "$MODE" = fresh ]; then
  echo "provision: removing $TASK_DIR" >&2
  rm -rf "$TASK_DIR"
  MODE=new
fi

if [ "$MODE" = resume ]; then
  [ -d "$WS" ] || { echo "provision: nothing to resume at $WS" >&2; exit 1; }
else
  [ ! -d "$WS" ] || { echo "provision: $WS exists; use --resume or --fresh" >&2; exit 1; }
fi

mkdir -p "$WS/src" "$REFS" "$SNAPSHOTS" "$TASK_DIR/.cache"

# Runs as the host user so everything it writes is the operator's to review.
# The mount paths match the build run's: a virtualenv bakes absolute paths into
# its scripts and pyvenv.cfg, so a venv built under one path fails under another.
in_sandbox() {
  podman run --rm --userns=keep-id --user "$(id -u):$(id -g)" \
    -v "$CACHE_VOLUME:/cache:ro" \
    -v "$WS:/workspace:z" \
    -v "$REFS:/refs:z" \
    -v "$TASK_DIR/.cache:/pipcache:z" \
    -e PIP_CACHE_DIR=/pipcache \
    -w / "$IMAGE" "$@"
}

RUN_NUMBER=1
if [ -f "$TASK_DIR/.run" ]; then RUN_NUMBER=$(( $(cat "$TASK_DIR/.run") + 1 )); fi

if [ "$MODE" = new ]; then
  podman volume exists "$CACHE_VOLUME" \
    || { echo "provision: no repo cache volume '$CACHE_VOLUME'; start the fleet first" >&2; exit 1; }

  # Process substitution, not a pipe: `exit 1` inside a piped `while` only ends
  # the subshell.
  while IFS=$'\t' read -r _ OWNER NAME REF MODE_R; do
    # DEST is the container path; HOST_DEST is the same tree from the host.
    if [ "$MODE_R" = write ]; then
      DEST="/workspace/src/$NAME"; HOST_DEST="$WS/src/$NAME"
    else
      DEST="/refs/$NAME"; HOST_DEST="$REFS/$NAME"
    fi
    echo "provision: $OWNER/$NAME ($REF) -> $DEST" >&2
    if ! in_sandbox test -d "/cache/$OWNER/$NAME"; then
      echo "provision: $OWNER/$NAME is not in the cache; refresh repo-mcp" >&2
      exit 1
    fi
    in_sandbox git clone --quiet --no-hardlinks "/cache/$OWNER/$NAME" "$DEST" \
      || { echo "provision: could not clone $OWNER/$NAME" >&2; exit 1; }

    # The cache holds non-bare checkouts whose working branch can sit behind
    # its remote-tracking ref, and `git clone` builds origin/* from the
    # source's local branches. Fetch the source's remote-tracking refs and
    # prefer them.
    in_sandbox git -C "$DEST" fetch --quiet origin \
      "+refs/remotes/origin/*:refs/remotes/upstream/*" 2>/dev/null || true
    if in_sandbox git -C "$DEST" rev-parse --verify --quiet \
         "refs/remotes/upstream/$REF" >/dev/null 2>&1; then
      in_sandbox git -C "$DEST" checkout --quiet -B "$REF" "refs/remotes/upstream/$REF"
    else
      in_sandbox git -C "$DEST" checkout --quiet "$REF" 2>/dev/null || true
    fi

    if [ "$MODE_R" = write ]; then
      # The branch and the identity are set here so the agent's first commit
      # cannot fail on either.
      in_sandbox git -C "$DEST" checkout --quiet -b "djed/$TASK_ID"
      in_sandbox git -C "$DEST" config user.name "djed builder"
      in_sandbox git -C "$DEST" config user.email "djed@localhost"

      # Dependencies are installed here, while there is still a network; the
      # build container has none.
      if [ -f "$HOST_DEST/pyproject.toml" ] || [ -f "$HOST_DEST/requirements.txt" ]; then
        echo "provision: building $DEST/.venv" >&2
        in_sandbox python -m venv "$DEST/.venv"
        in_sandbox "$DEST/.venv/bin/pip" install --quiet --upgrade pip
        if [ -f "$HOST_DEST/pyproject.toml" ]; then
          in_sandbox "$DEST/.venv/bin/pip" install --quiet -e "$DEST[dev]" \
            || in_sandbox "$DEST/.venv/bin/pip" install --quiet -e "$DEST" \
            || echo "provision: WARNING could not install $DEST" >&2
        else
          in_sandbox "$DEST/.venv/bin/pip" install --quiet -r "$DEST/requirements.txt" \
            || echo "provision: WARNING could not install $DEST requirements" >&2
        fi
        in_sandbox "$DEST/.venv/bin/pip" install --quiet pytest \
          || echo "provision: WARNING could not install pytest into $DEST" >&2
      fi
    fi
  done < <(printf '%s\n' "$MANIFEST" | awk -F'\t' '$1=="repo"')
fi

# --- command manifest -----------------------------------------------------
#
# A [commands.<repo>] table declared in the spec wins; otherwise CMake and
# Python are detected, and a repo matching both or neither gets no commands
# rather than a guess. Regenerated every run, so a resume re-detects.
DECLARED_COMMANDS="$("$PY" -m djed.spec commands "$SPEC")"
MANIFEST_TOML=""
declare -A REPO_TASKS

# A declared repo's own [name] table out of $DECLARED_COMMANDS, or empty.
_declared_block() {
  awk -v name="[$1]" '
    $0 == name { found = 1; print; next }
    found && /^\[/ { exit }
    found { print }
  ' <<<"$DECLARED_COMMANDS"
}

# Task names in a manifest block, minus phase="provision" ones.
_task_names() {
  awk '
    /^[^[:space:]]+[[:space:]]*=[[:space:]]*\{/ {
      name = $1
      if ($0 !~ /phase = "provision"/) print name
    }
  ' <<<"$1"
}

for d in "$WS"/src/*/; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  block="$(_declared_block "$name")"

  if [ -n "$block" ]; then
    echo "provision: $name: commands declared in the spec" >&2
  elif [ -f "$d/CMakeLists.txt" ] && { [ -f "$d/pyproject.toml" ] || [ -f "$d/requirements.txt" ]; }; then
    echo "provision: $name: matches both CMake and Python - ambiguous;" >&2
    echo "  declare [commands.$name] in the spec to resolve it." >&2
  elif [ -f "$d/CMakeLists.txt" ]; then
    echo "provision: $name: CMakeLists.txt detected - configuring" >&2
    in_sandbox cmake -S "/workspace/src/$name" -B "/workspace/src/$name/build" \
      || echo "provision: WARNING cmake configure failed for $name" >&2
    block="[$name]
configure = { argv = [\"cmake\", \"-S\", \".\", \"-B\", \"build\"], phase = \"provision\" }
build     = { argv = [\"cmake\", \"--build\", \"build\", \"-j4\"], timeout_s = 600 }
test      = { argv = [\"ctest\", \"--test-dir\", \"build\", \"--output-on-failure\"], timeout_s = 300 }"
  elif [ -f "$d/pyproject.toml" ] || [ -f "$d/requirements.txt" ]; then
    # Either file gets the same pytest entry; the venv step installs pytest
    # for both.
    if [ -x "$d/.venv/bin/pytest" ]; then
      block="[$name]
test = { argv = [\".venv/bin/pytest\", \"tests/\", \"-q\"] }"
    else
      echo "provision: $name: pytest was not installed; no test command declared" >&2
    fi
  else
    echo "provision: $name: no recognised build system - no commands declared" >&2
  fi

  if [ -n "$block" ]; then
    MANIFEST_TOML+="$block"$'\n\n'
    REPO_TASKS["$name"]="$(_task_names "$block" | tr '\n' ' ')"
    REPO_TASKS["$name"]="${REPO_TASKS[$name]% }"
  fi
done

printf '%s' "$MANIFEST_TOML" > "$TASK_DIR/manifest.toml"

# --- WORKSPACE.md ---------------------------------------------------------
"$PY" -m djed.spec prose "$SPEC" > "$WS/SPEC.md"
echo "$RUN_NUMBER" > "$TASK_DIR/.run"

{
  echo "# Workspace"
  echo
  echo "Task: $TASK_ID"
  if [ "$RUN_NUMBER" -gt 1 ]; then
    echo "Run: $RUN_NUMBER (resumed)"
  else
    echo "Run: 1"
  fi
  echo
  echo "## Writable (edit these)"
  for d in "$WS"/src/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    head="$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    branch="$(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    ahead="$(git -C "$d" rev-list --count "$branch" --not --remotes 2>/dev/null || echo 0)"
    echo "- /workspace/src/$name - git repo, branch \`$branch\` at $head, $ahead commit(s) ahead."
    if [ -n "${REPO_TASKS[$name]:-}" ]; then
      echo "  Commands (via exec-mcp, list_tasks/run_task): ${REPO_TASKS[$name]}"
    else
      echo "  No commands were declared or detected for this repo - you can edit"
      echo "  it, but not verify your changes here."
    fi
  done
  if compgen -G "$REFS/*/" > /dev/null; then
    echo
    echo "## Reference (read-only, writes will fail)"
    for d in "$REFS"/*/; do
      name="$(basename "$d")"
      head="$(git -C "$d" rev-parse --short HEAD 2>/dev/null || echo unknown)"
      echo "- /refs/$name - git repo at $head"
    done
  fi
  echo
  echo "## Notes"
  echo "- These are git checkouts, but \`.git\` is hidden from your file tools by"
  echo "  policy. Use exec-mcp's git_status, git_diff, git_add and git_commit."
  echo "- git identity is already configured. Do not run \`git push\`."
  echo "- Read and write files with the fs_ tools, not through a command."
  echo "  Building and testing go through exec-mcp's list_tasks/run_task."
  if [ "$RUN_NUMBER" -gt 1 ] && [ -f "$WS/HANDOFF.md" ]; then
    echo "- Read HANDOFF.md first: it is your own account of why run"
    echo "  $(( RUN_NUMBER - 1 )) stopped."
  fi
} > "$WS/WORKSPACE.md"

echo "provision: $TASK_DIR ready (run $RUN_NUMBER)" >&2
echo "$TASK_DIR"
