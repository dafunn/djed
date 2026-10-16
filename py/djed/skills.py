"""Skills: task instructions the agent loads by name when a task calls for one.

A skill is a folder holding a `SKILL.md`, whose `---`-fenced frontmatter gives
its `name` and `description`, and optionally `agents`, the persona tags it is
offered to. The body is the instructions. Other files in the folder are read on
demand with `read_skill_file`.

Only the index - one line per skill - goes into the instructions, which are
re-sent with every request. A body enters the conversation when `load_skill`
returns it, and `decay.py` never elides that result.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SKILL_FILE = "SKILL.md"
FENCE = "---"
# The tool whose results stay whole for the rest of the conversation.
LOAD_SKILL_TOOL = "load_skill"
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
# A loaded body is never elided, so it is re-sent with every later request. 8 KiB
# matches `cap.py`'s per-result limit, about 1,650 tokens.
DEFAULT_MAX_SKILL_CHARS = 8 * 1024
# `skills/` at the repository root: /app/skills in the sandbox and build containers.
DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


def max_skill_chars() -> int:
    """SKILL.md size above which a skill is reported as too large. 0 disables it."""
    raw = (os.environ.get("DJED_MAX_SKILL_CHARS") or "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_SKILL_CHARS
    except ValueError:
        return DEFAULT_MAX_SKILL_CHARS


def skills_dirs() -> list[Path]:
    """`DJED_SKILLS_DIR` (colon-separated) if set, else the repository's `skills/`."""
    raw = (os.environ.get("DJED_SKILLS_DIR") or "").strip()
    if not raw:
        return [DEFAULT_SKILLS_DIR]
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    folder: Path
    size: int  # characters in SKILL.md
    agents: tuple[str, ...] = ()

    @property
    def oversized(self) -> bool:
        limit = max_skill_chars()
        return bool(limit) and self.size > limit

    def body(self) -> str:
        """The instructions, read from disk so an edit is seen without a restart."""
        text = (self.folder / SKILL_FILE).read_text(encoding="utf-8")
        return parse_skill_md(text)[1]

    def other_files(self) -> list[str]:
        """Paths of the folder's other files, relative to it."""
        return sorted(
            p.relative_to(self.folder).as_posix()
            for p in self.folder.rglob("*")
            if p.is_file() and p.name != SKILL_FILE
            and not any(part.startswith(".") for part in p.relative_to(self.folder).parts)
        )


class SkillError(ValueError):
    """A SKILL.md that cannot be used."""


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _list(value: str) -> tuple[str, ...]:
    """`[a, b]` or a bare `a` - the two forms the `agents` field takes."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return tuple(_scalar(v) for v in value.split(",") if v.strip())


def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """Split frontmatter from body. Frontmatter is flat `key: value` lines; a
    value continues onto following indented lines.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != FENCE:
        raise SkillError(f"must open with a {FENCE} frontmatter fence")
    end = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == FENCE), None)
    if end is None:
        raise SkillError(f"unterminated frontmatter: no closing {FENCE}")

    meta: dict[str, str] = {}
    key = None
    for line in lines[i + 1 : end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0] in " \t" and key is not None:
            meta[key] = f"{meta[key]} {line.strip()}".strip()
            continue
        if ":" not in line:
            raise SkillError(f"frontmatter line is not `key: value`: {line.strip()!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        # A YAML block indicator; the text itself is on the indented lines.
        meta[key] = "" if value.strip() in (">", "|", ">-", "|-") else value.strip()
    return meta, "\n".join(lines[end + 1 :]).strip()


def load_skill_folder(folder: Path) -> Skill:
    text = (folder / SKILL_FILE).read_text(encoding="utf-8")
    meta, body = parse_skill_md(text)
    name = _scalar(meta.get("name", ""))
    description = _scalar(meta.get("description", ""))
    if not name:
        raise SkillError("frontmatter has no name")
    if len(name) > MAX_NAME or not NAME_RE.match(name):
        raise SkillError(f"name {name!r} must be lowercase letters, digits and hyphens")
    if not description:
        raise SkillError("frontmatter has no description")
    if len(description) > MAX_DESCRIPTION:
        raise SkillError(f"description is over {MAX_DESCRIPTION} characters")
    if not body:
        raise SkillError("has no instructions after the frontmatter")
    return Skill(
        name=name,
        description=description,
        folder=folder,
        size=len(text),
        agents=_list(meta.get("agents", "")),
    )


@dataclass
class Discovery:
    skills: dict[str, Skill] = field(default_factory=dict)
    # Skills that could not be used, and duplicates: shown, not fatal.
    problems: list[str] = field(default_factory=list)

    @property
    def oversized(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.oversized]


def oversize_warning(skill: Skill) -> str:
    return (
        f"skill {skill.name}: SKILL.md is {skill.size} characters, over the "
        f"{max_skill_chars()} limit. A loaded skill stays in context for the rest of "
        f"the conversation; move the bulk into separate reference files in "
        f"{skill.folder} and point to them from SKILL.md."
    )


def discover(persona_tag: str, dirs: list[Path] | None = None) -> Discovery:
    """The skills offered to `persona_tag`. A missing directory is not an error;
    an unusable skill is reported and skipped. On a name clash the first
    directory wins.
    """
    found = Discovery()
    for root in skills_dirs() if dirs is None else dirs:
        if not root.is_dir():
            continue
        for skill_md in sorted(root.glob(f"*/{SKILL_FILE}")):
            folder = skill_md.parent
            try:
                skill = load_skill_folder(folder)
            except (OSError, UnicodeDecodeError, SkillError) as e:
                found.problems.append(f"skill {folder}: {e}")
                continue
            if skill.agents and persona_tag not in skill.agents:
                continue
            if skill.name in found.skills:
                found.problems.append(
                    f"skill {skill.name}: {folder} ignored, already loaded from "
                    f"{found.skills[skill.name].folder}"
                )
                continue
            found.skills[skill.name] = skill
    return found


def index_instructions(skills: dict[str, Skill]) -> str:
    """The paragraph appended to the instructions. Empty when there are no skills."""
    if not skills:
        return ""
    rows = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
    return (
        "\n\n## Skills\n\n"
        "Each skill below holds instructions for one kind of task. When a task "
        f"matches one, call {LOAD_SKILL_TOOL} with its name before you start, and "
        "follow what it says. A loaded skill stays in this conversation, so load "
        "each one once.\n\n" + rows
    )


def render_loaded(skill: Skill) -> str:
    """What `load_skill` returns, and what `/skill` puts in front of a prompt."""
    out = [f"# Skill: {skill.name}", "", skill.body()]
    files = skill.other_files()
    if files:
        out += [
            "",
            f"Files in this skill (folder {skill.folder}), read with read_skill_file:",
            "",
            *(f"- {f}" for f in files),
        ]
    return "\n".join(out)


def read_skill_path(skill: Skill, path: str) -> Path:
    """Resolve `path` inside the skill's folder, refusing anything outside it."""
    root = skill.folder.resolve()
    target = (root / path).resolve()
    if target != root and not target.is_relative_to(root):
        raise SkillError(f"{path!r} is outside the skill's folder")
    if not target.is_file():
        raise SkillError(f"{path!r} is not a file in skill {skill.name}")
    return target
