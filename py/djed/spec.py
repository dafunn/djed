"""Parse a build spec: TOML frontmatter, then the task prose.

Only the prose reaches the model; the frontmatter is provisioning's input. TOML
via stdlib `tomllib`, so parsing adds no dependency. The checks reject
unqualified repo names, colliding checkout basenames and unsafe task_ids.
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field

FENCE = "+++"
MODES = ("write", "read")
DEFAULT_TIMEOUT_S = 600
# A task_id becomes a directory name, a git branch segment and a container name.
# Restricted to what all three accept unquoted.
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class SpecError(ValueError):
    """The spec file is malformed. The message is shown to the operator."""


@dataclass(frozen=True)
class RepoSpec:
    owner: str
    name: str
    ref: str
    mode: str

    @property
    def qualified(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class Budget:
    """Backstops for an unattended run. None of these is a target."""

    max_iterations: int = 10
    # 0 means no ceiling (the default). Set a limit when the model behind the
    # gateway is metered.
    max_tokens: int = 0
    # A safety limit, not the work budget (max_iterations x max_requests_per_turn).
    # run_build.sh also passes it to the exec MCP client as its request timeout.
    wall_clock_s: float = 14400.0
    # Bounds one turn. Tripping it starts a new turn; max_iterations is how many
    # turns a run gets, so the two multiply.
    max_requests_per_turn: int = 30


@dataclass(frozen=True)
class Command:
    """One named task exec-mcp (or provisioning, for `phase = "provision"`)
    may run."""

    argv: tuple[str, ...]
    phase: str = ""  # "" (exec, offline) | "provision"
    timeout_s: int = DEFAULT_TIMEOUT_S


@dataclass(frozen=True)
class Manifest:
    task_id: str
    repos: tuple[RepoSpec, ...]
    budget: Budget
    # From the [commands] table in spec frontmatter; empty means provisioning
    # detects instead. Keyed by write-repo basename, then task name.
    commands: dict[str, dict[str, Command]] = field(default_factory=dict)

    @property
    def write_repos(self) -> tuple[RepoSpec, ...]:
        return tuple(r for r in self.repos if r.mode == "write")

    @property
    def read_repos(self) -> tuple[RepoSpec, ...]:
        return tuple(r for r in self.repos if r.mode == "read")


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split `+++`-fenced frontmatter from the prose body."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != FENCE:
        raise SpecError(f"spec must open with a {FENCE} frontmatter fence")
    start = i + 1
    for j in range(start, len(lines)):
        if lines[j].strip() == FENCE:
            return "\n".join(lines[start:j]), "\n".join(lines[j + 1 :]).strip()
    raise SpecError(f"unterminated frontmatter: no closing {FENCE}")


def _repo(entry: object, index: int) -> RepoSpec:
    if not isinstance(entry, dict):
        raise SpecError(f"[[repo]] #{index} is not a table")
    raw = str(entry.get("repo", "")).strip()
    parts = raw.split("/")
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise SpecError(
            f"repo {raw!r} must be owner-qualified as 'owner/name' "
            "(e.g. acme/widget) - more than one owner is cached, "
            "so a bare name is ambiguous"
        )
    mode = str(entry.get("mode", "")).strip()
    if mode not in MODES:
        raise SpecError(
            f"repo {raw!r} has mode {mode!r}; expected one of {', '.join(MODES)}"
        )
    return RepoSpec(
        owner=parts[0].strip(),
        name=parts[1].strip(),
        # Defaulted rather than required; an explicit ref is still honoured.
        ref=str(entry.get("ref", "main")).strip() or "main",
        mode=mode,
    )


def _budget(data: object) -> Budget:
    if data is None:
        return Budget()
    if not isinstance(data, dict):
        raise SpecError("[budget] is not a table")
    defaults = Budget()
    unknown = set(data) - {
        "max_iterations",
        "max_tokens",
        "wall_clock_s",
        "max_requests_per_turn",
    }
    if unknown:
        raise SpecError(f"[budget] has unknown keys: {', '.join(sorted(unknown))}")
    try:
        budget = Budget(
            max_iterations=int(data.get("max_iterations", defaults.max_iterations)),
            max_tokens=int(data.get("max_tokens", defaults.max_tokens)),
            wall_clock_s=float(data.get("wall_clock_s", defaults.wall_clock_s)),
            max_requests_per_turn=int(
                data.get("max_requests_per_turn", defaults.max_requests_per_turn)
            ),
        )
    except (TypeError, ValueError) as e:
        raise SpecError(f"[budget] has a non-numeric value: {e}") from e
    if min(
        budget.max_iterations,
        budget.wall_clock_s,
        budget.max_requests_per_turn,
    ) <= 0:
        raise SpecError("[budget] values must be positive")
    # `max_tokens` may be zero, which turns it off; negative values are rejected.
    if budget.max_tokens < 0:
        raise SpecError("[budget] max_tokens cannot be negative (0 means no ceiling)")
    return budget


def _command(repo_name: str, task_name: str, entry: object) -> Command:
    where = f"[commands.{repo_name}].{task_name}"
    if not isinstance(entry, dict):
        raise SpecError(f"{where} is not a table")
    argv = entry.get("argv")
    if not isinstance(argv, list) or not argv or not all(
        isinstance(a, str) and a for a in argv
    ):
        raise SpecError(f"{where} needs a non-empty argv of strings")
    phase = str(entry.get("phase", "")).strip()
    if phase not in ("", "provision"):
        raise SpecError(f'{where} has phase {phase!r}; expected "provision" or unset')
    timeout_s = entry.get("timeout_s", DEFAULT_TIMEOUT_S)
    try:
        timeout_s = int(timeout_s)
    except (TypeError, ValueError):
        raise SpecError(f"{where} has a non-numeric timeout_s") from None
    if timeout_s <= 0:
        raise SpecError(f"{where} needs a positive timeout_s")
    return Command(argv=tuple(argv), phase=phase, timeout_s=timeout_s)


def _commands(data: object, write_names: set[str]) -> dict[str, dict[str, Command]]:
    """A `[commands]` block overrides provisioning's detection, and covers write repos
    only.
    """
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SpecError("[commands] is not a table")
    out: dict[str, dict[str, Command]] = {}
    for repo_name, tasks in data.items():
        if repo_name not in write_names:
            raise SpecError(
                f"[commands.{repo_name}] names a repo that is not listed with "
                'mode = "write" in this spec - commands cover write repos only'
            )
        if not isinstance(tasks, dict) or not tasks:
            raise SpecError(f"[commands.{repo_name}] needs at least one command")
        out[repo_name] = {
            task_name: _command(repo_name, task_name, entry)
            for task_name, entry in tasks.items()
        }
    return out


def render_commands_toml(commands: dict[str, dict[str, Command]]) -> str:
    """The runtime manifest's TOML text for the repos with declared commands -
    what provisioning writes to `$TASK_DIR/manifest.toml` when `[commands]`
    suppressed detection. Empty input means nothing was declared."""
    blocks = []
    for repo_name, tasks in commands.items():
        lines = [f"[{repo_name}]"]
        for task_name, c in tasks.items():
            argv_str = "[" + ", ".join(json.dumps(a) for a in c.argv) + "]"
            fields = [f"argv = {argv_str}"]
            if c.phase:
                fields.append(f'phase = "{c.phase}"')
            if c.timeout_s != DEFAULT_TIMEOUT_S:
                fields.append(f"timeout_s = {c.timeout_s}")
            lines.append(f"{task_name} = {{ {', '.join(fields)} }}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_spec(text: str) -> tuple[Manifest, str]:
    """Parse a spec file into `(manifest, prose)`.

    Raises `SpecError` with an operator-readable message on any problem.
    """
    front, prose = split_frontmatter(text)
    try:
        data = tomllib.loads(front)
    except tomllib.TOMLDecodeError as e:
        raise SpecError(f"frontmatter is not valid TOML: {e}") from e

    task_id = str(data.get("task_id", "")).strip()
    if not task_id:
        raise SpecError("frontmatter needs a task_id")
    if not TASK_ID_RE.match(task_id):
        raise SpecError(
            f"task_id {task_id!r} is not a safe path segment: lowercase letters, "
            "digits, dot, dash and underscore only, starting alphanumeric, "
            "64 characters at most"
        )

    entries = data.get("repo") or []
    if not entries:
        raise SpecError("frontmatter needs at least one [[repo]]")

    repos: list[RepoSpec] = []
    by_qualified: dict[str, RepoSpec] = {}
    by_basename: dict[str, RepoSpec] = {}
    for i, entry in enumerate(entries, 1):
        r = _repo(entry, i)
        if r.qualified in by_qualified:
            raise SpecError(f"repo {r.qualified} is listed twice")
        # Checkouts are named by basename, so two owners with the same
        # repository name would land on the same directory.
        clash = by_basename.get(r.name)
        if clash is not None:
            raise SpecError(
                f"{r.qualified} and {clash.qualified} share the basename "
                f"{r.name!r}, which is the checkout directory name"
            )
        by_qualified[r.qualified] = r
        by_basename[r.name] = r
        repos.append(r)

    if not any(r.mode == "write" for r in repos):
        raise SpecError("frontmatter needs at least one repo with mode = \"write\"")
    if not prose:
        raise SpecError("spec has no task prose after the frontmatter")

    write_names = {r.name for r in repos if r.mode == "write"}
    manifest = Manifest(
        task_id=task_id,
        repos=tuple(repos),
        budget=_budget(data.get("budget")),
        commands=_commands(data.get("commands"), write_names),
    )
    return manifest, prose


# --- building a spec from prose -------------------------------------------
#
# A task is written as prose; the frontmatter is generated here and validated
# by `parse_spec`.


def slug(text: str) -> str:
    """Reduce arbitrary text, usually a filename stem, to a legal `task_id`.

    Anything `TASK_ID_RE` rejects becomes a dash, runs of dashes collapse, and the
    result is trimmed to the 64 characters the pattern allows. If nothing remains,
    the caller must name the task explicitly.
    """
    out = re.sub(r"[^a-z0-9._-]+", "-", text.strip().lower())
    out = re.sub(r"-{2,}", "-", out)
    out = re.sub(r"^[^a-z0-9]+", "", out)
    return out[:64].rstrip("-._")


def _repo_arg(arg: str, mode: str) -> tuple[str, str, str]:
    """Parse `owner/name` or `owner/name@ref` into its three parts.

    Shape only - `parse_spec` is still the judge of whether the result is a
    usable manifest, so this rejects nothing it can pass along.
    """
    target, _, ref = arg.partition("@")
    if target.count("/") != 1 or not all(p.strip() for p in target.split("/")):
        raise SpecError(
            f"repo {arg!r} must be owner-qualified as 'owner/name', "
            "optionally with '@ref'"
        )
    return target.strip(), (ref.strip() or "main"), mode


def render_spec(
    prose: str,
    task_id: str,
    repos: list[tuple[str, str, str]],
    budget: dict[str, object] | None = None,
) -> str:
    """Render `(prose, task_id, repos)` into a builder spec, and validate it.

    `repos` is a list of `(qualified, ref, mode)`. The returned text always parses.
    """
    if not prose.strip():
        raise SpecError("the prose file is empty; there is no task to render")
    lines = [FENCE, f'task_id = "{task_id}"']
    for qualified, ref, mode in repos:
        lines += ["", "[[repo]]", f'repo = "{qualified}"']
        # Written only when it differs from the default.
        if ref != "main":
            lines.append(f'ref = "{ref}"')
        lines.append(f'mode = "{mode}"')
    # Only the keys given, and no [budget] table when none were.
    set_keys = [
        (k, budget[k])
        for k in ("max_iterations", "max_tokens", "wall_clock_s",
                  "max_requests_per_turn")
        if budget and budget.get(k) is not None
    ]
    if set_keys:
        lines += ["", "[budget]"]
        lines += [f"{k} = {v}" for k, v in set_keys]
    lines += [FENCE, "", prose.strip(), ""]
    text = "\n".join(lines)
    parse_spec(text)      # raises SpecError rather than emitting a bad file
    return text


def _main(argv: list[str] | None = None) -> int:
    """Read and write specs from the shell; `provision.sh` uses this rather than
    parsing TOML itself.

        python -m djed.spec manifest task.md   # TSV: task_id, budget, one row/repo
        python -m djed.spec prose    task.md   # the body the model is given
        python -m djed.spec commands task.md   # declared [commands], as runtime TOML
        python -m djed.spec init notes.md --write owner/name

    `init` adds frontmatter to prose you already wrote. Repos must be named; every
    other field has a default.
    """
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m djed.spec")
    sub = ap.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("manifest", "TSV: task_id, budget, then one row per repo"),
        ("prose", "the task body, as the model receives it"),
        ("commands", "declared [commands], rendered as runtime manifest TOML"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("spec", help="path to a builder spec file")

    p = sub.add_parser("init", help="wrap a prose file in builder frontmatter")
    p.add_argument("prose_file", help="a plain prose task file, no frontmatter")
    p.add_argument(
        "--write", action="append", default=[], metavar="OWNER/NAME[@REF]",
        help="a repo the run may change; repeatable, at least one required",
    )
    p.add_argument(
        "--read", action="append", default=[], metavar="OWNER/NAME[@REF]",
        help="a reference repo, mounted read-only; repeatable",
    )
    p.add_argument(
        "--task-id",
        help="default: the prose file's name, reduced to a legal task_id",
    )
    p.add_argument("-o", "--output", help="write here instead of stdout")
    p.add_argument("--force", action="store_true", help="overwrite --output")
    p.add_argument("--max-iterations", type=int)
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--wall-clock-s", type=float)
    p.add_argument("--max-requests-per-turn", type=int)

    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "init":
        return _cmd_init(args)

    try:
        manifest, prose = parse_spec(
            open(args.spec, encoding="utf-8").read()
        )
    except (SpecError, OSError) as e:
        print(f"spec: {e}", file=sys.stderr)
        return 1

    if args.command == "prose":
        print(prose)
        return 0

    if args.command == "commands":
        # Empty output means nothing was declared; provisioning falls back to
        # detection.
        text = render_commands_toml(manifest.commands)
        if text:
            print(text)
        return 0

    print(f"task_id\t{manifest.task_id}")
    print(f"budget\twall_clock_s\t{manifest.budget.wall_clock_s}")
    for r in manifest.repos:
        print(f"repo\t{r.owner}\t{r.name}\t{r.ref}\t{r.mode}")
    return 0


def _cmd_init(args) -> int:
    import sys
    from pathlib import Path

    src = Path(args.prose_file)
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as e:
        print(f"spec: {e}", file=sys.stderr)
        return 1

    # Refuse a file that already has frontmatter: a nested one parses, with the
    # real manifest demoted to prose.
    try:
        split_frontmatter(text)
    except SpecError:
        pass
    else:
        print(
            f"spec: {src} already opens with a {FENCE} fence - it is a builder "
            "spec already, not prose to wrap.",
            file=sys.stderr,
        )
        return 1

    # The first dot-segment, so `task.prose.md` and `task.md` agree.
    task_id = args.task_id or slug(src.name.split(".")[0])
    if not task_id:
        print(
            f"spec: cannot derive a task_id from {src.name!r}; pass --task-id",
            file=sys.stderr,
        )
        return 1

    budget = {
        "max_iterations": args.max_iterations,
        "max_tokens": args.max_tokens,
        "wall_clock_s": args.wall_clock_s,
        "max_requests_per_turn": args.max_requests_per_turn,
    }
    try:
        repos = [_repo_arg(a, "write") for a in args.write]
        repos += [_repo_arg(a, "read") for a in args.read]
        if not repos:
            raise SpecError(
                "name at least one repo with --write; a run with nothing to "
                "change cannot produce a result"
            )
        rendered = render_spec(text, task_id, repos, budget)
    except SpecError as e:
        print(f"spec: {e}", file=sys.stderr)
        return 1

    if not args.output:
        sys.stdout.write(rendered)
        return 0

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"spec: {out} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out.write_text(rendered, encoding="utf-8")
    print(f"spec: wrote {out} (task_id {task_id})", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the shell
    raise SystemExit(_main())
