"""devdir — THE ONE RESOLVER for devdir slots. This file is the whole devdir file-finding system.

    RESOLVE(launch_dir, cwd, slot) -> [DevdirFile{path, frontmatter, content, source}]

Give it where the app launched (`launch_dir`), where the agent is standing (`cwd` — the dir of
its most recent read/bash), and a slot name (`rules` / `skills` / `hooks` / `agents` / `morphs`
/ anything); it returns every file for that slot from every source, precedence-ordered
(least-specific first, nearest level LAST — the nearest refines), each stamped with its
provenance. ADDING A SLOT IS ADDING AN ENTRY TO `SLOT_GRAMMARS` — an unknown slot already
resolves via the DEFAULT grammar (every file one and two levels under `<devdir>/<slot>/`).

The sources, in order (mirrors Claude Code's markdownConfigLoader source ladder):
  1. USER    — the home level (`~`): `~/.claude/<slot>` + `~/.heaven/<slot>`. ALWAYS included
               (`~/.heaven` IS the registry — the global root place), independent of any walk.
  2. LAUNCH  — the ancestor walk of `launch_dir` (ENTER: climb to the nearest devdir-bearing
               ancestor; CLIMB: keep going while the PARENT also has one; a gap ends the chain).
  3. ACTIVE  — the same walk of `cwd` (the dir the agent last read/bashed into; None = skip).
At each level both devdirs are read, `.claude` first then `.heaven` (parity order). Files are
de-duplicated by resolved path, then by content hash (identical bodies included once).

WHAT THIS FILE DELIBERATELY DOES NOT DO (the consumers own it): char caps + skip notices
(`baseheavenagent.resolve_devdirs` — the 400k cap is a SEMANTIC ALARM and stays there), hook
REGISTRATION (executing a hook file), skill-block rendering, agent construction. This file
FINDS files; consumers decide what loading them means.

Consumers: `heaven_base.baseheavenagent.resolve_devdirs` (rules + skills + hooks) ·
`heaven_base.hooks.skill_autoload` (the <AVAILABLE_SKILLS> block) · onionmorph's runtime/hooks
loaders. No heaven module is imported here — this file is import-terminal by design.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

DEVDIR_NAMES = (".claude", ".heaven")
MAX_DEVDIR_WALKUP = 12


class DevdirSource(BaseModel):
    """Provenance of a resolved file: which root kind found it, at which level, in which devdir."""
    root: str      # "user" | "launch" | "active" | "extra"
    level: str     # the dir whose devdir this file came from (absolute path)
    devdir: str    # ".claude" | ".heaven"


class DevdirFile(BaseModel):
    """One resolved slot file, provenance-stamped."""
    path: str
    frontmatter: Dict[str, str]
    content: str
    source: DevdirSource


def dir_has_devdir(d: Path) -> bool:
    """True if `d` carries any devdir marker: a `.claude/` or `.heaven/` dir, or a root
    CLAUDE.md/AGENTS.md. This is the walk boundary — climb while the PARENT has one."""
    try:
        return ((d / ".claude").is_dir() or (d / ".heaven").is_dir()
                or (d / "CLAUDE.md").is_file() or (d / "AGENTS.md").is_file())
    except Exception:
        return False


def devdir_levels(start: Union[str, Path]) -> List[Path]:
    """The devdir-bearing ancestor dirs that apply to `start`, root-first (nearest last).
    Two phases (NOT bounded by `.git`):
      1. ENTER: climb to the NEAREST ancestor that has a devdir (skipping a leading gap, so an
         agent deep in a repo whose .claude lives only at the root still reaches the root).
      2. CLIMB: keep going up while the PARENT also has a devdir; STOP at the first parent
         with none (a gap ABOVE the entry ends the chain — never jump over it).
    `MAX_DEVDIR_WALKUP` caps each phase. No devdir anywhere up the chain -> just `start`."""
    current = Path(start).expanduser()
    try:
        current = current.resolve()
    except Exception:
        current = current.absolute()
    if current.is_file():
        current = current.parent
    entry = None
    probe = current
    for _ in range(MAX_DEVDIR_WALKUP):
        if dir_has_devdir(probe):
            entry = probe
            break
        if probe.parent == probe:
            break
        probe = probe.parent
    if entry is None:
        return [current]
    chain = [entry]
    cur = entry
    for _ in range(MAX_DEVDIR_WALKUP):
        parent = cur.parent
        if parent == cur or not dir_has_devdir(parent):
            break
        chain.append(parent)
        cur = parent
    return list(reversed(chain))


# ── The slot grammars — where a slot's files live inside a devdir ──────────────────────────
# A grammar = a function (level, devdir_name) -> ordered file paths. Named slots carry the
# real on-disk grammar each slot has today; ANY OTHER slot name resolves via _default_slot.

def _rules_files(level: Path, devdir_name: str) -> List[Path]:
    """The instruction surface: level-root CLAUDE.md/AGENTS.md + the devdir's own instruction
    file(s) + `<devdir>/rules/*.md`. (.claude reads CLAUDE.md; .heaven reads AGENTS.md too.)"""
    if devdir_name == ".claude":
        files = [level / "CLAUDE.md", level / ".claude" / "CLAUDE.md"]
    else:
        files = [level / "AGENTS.md", level / "CLAUDE.md",
                 level / ".heaven" / "AGENTS.md", level / ".heaven" / "CLAUDE.md"]
    rules_dir = level / devdir_name / "rules"
    if rules_dir.is_dir():
        files.extend(sorted(rules_dir.glob("*.md")))
    return [p for p in files if p.is_file()]


def _skills_files(level: Path, devdir_name: str) -> List[Path]:
    """`<devdir>/skills/*/SKILL.md` (dir form) + `<devdir>/skills/*.md` (flat form)."""
    d = level / devdir_name / "skills"
    if not d.is_dir():
        return []
    return sorted(d.glob("*/SKILL.md")) + sorted(d.glob("*.md"))


def _hooks_files(level: Path, devdir_name: str) -> List[Path]:
    """`<devdir>/hooks/*.py` — presence = active; `<name>.py.inactive` is structurally
    excluded by the glob (the ONION toggle). `_`/`.`-prefixed names are private."""
    d = level / devdir_name / "hooks"
    if not d.is_dir():
        return []
    return [p for p in sorted(d.glob("*.py")) if not p.name.startswith(("_", "."))]


def _agents_files(level: Path, devdir_name: str) -> List[Path]:
    """`<devdir>/agents/{name}/*.py` — an agent config is a PYTHON FILE loading its json
    (`{name}/{name}_config.py`, or a Replicant class). Flat `agents/*.py` kept for the legacy
    flat convention. Non-.py files (e.g. a stray .md) do NOT resolve — an agent is code."""
    d = level / devdir_name / "agents"
    if not d.is_dir():
        return []
    out = [p for p in sorted(d.glob("*/*.py")) if not p.name.startswith(("_", "."))]
    out += [p for p in sorted(d.glob("*.py")) if not p.name.startswith(("_", "."))]
    return out


def _morphs_files(level: Path, devdir_name: str) -> List[Path]:
    """`<devdir>/morphs/*.json` — toggle-set loadouts."""
    d = level / devdir_name / "morphs"
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def _default_slot(slot: str):
    """The grammar for a slot this file has never seen: every non-private regular file one and
    two levels under `<devdir>/<slot>/`. Adding a named grammar later only NARROWS this."""
    def _files(level: Path, devdir_name: str) -> List[Path]:
        d = level / devdir_name / slot
        if not d.is_dir():
            return []
        found = [p for p in sorted(d.glob("*")) if p.is_file()]
        found += [p for p in sorted(d.glob("*/*")) if p.is_file()]
        return [p for p in found if not p.name.startswith(("_", "."))]
    return _files


SLOT_GRAMMARS = {
    "rules": _rules_files,
    "skills": _skills_files,
    "hooks": _hooks_files,
    "agents": _agents_files,
    "morphs": _morphs_files,
}


# ── Frontmatter ────────────────────────────────────────────────────────────────────────────

_FM_LINE = re.compile(r"^([A-Za-z0-9_\-]+):\s*(.*)$")


def parse_frontmatter(content: str) -> Dict[str, str]:
    """Simple `key: value` frontmatter. A leading `---` block is parsed line-by-line; without
    one, top-level `name:`/`description:` lines anywhere are picked up (today's contract for
    skills/rules). Dependency-free by design — no YAML lists/nesting (devdir files don't use
    them; a file that needs more carries its own structure in its body)."""
    out: Dict[str, str] = {}
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for ln in lines[1:]:
            if ln.strip() == "---":
                break
            m = _FM_LINE.match(ln)
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out
    for key in ("name", "description"):
        m = re.search(rf"^{key}:\s*(.+)$", content, re.M)
        if m:
            out[key] = m.group(1).strip()
    return out


# ── THE RESOLVER ───────────────────────────────────────────────────────────────────────────

def resolve(launch_dir: Union[str, Path],
            cwd: Optional[Union[str, Path]] = None,
            slot: str = "rules",
            extra_roots: tuple = ()) -> List[DevdirFile]:
    """THE ONE FUNCTION. Every devdir slot loads through this.

    launch_dir  — where the app launched (the agent's configured cwd; always-on).
    cwd         — where the agent is standing (its last read/bash dir; None = launch only).
    slot        — the slot name ("rules"/"skills"/"hooks"/"agents"/"morphs"/anything).
    extra_roots — additional walk roots a caller must include (e.g. an explicit `start`).

    Returns provenance-stamped files, precedence-ordered: USER level first, then the launch
    walk root-first, then the active walk (nearest level LAST — the nearest refines). Files
    deduped by resolved path, then by content hash. Unreadable files are skipped silently
    (finding, not judging); empty-vs-missing is the consumer's distinction to make.
    """
    grammar = SLOT_GRAMMARS.get(slot) or _default_slot(slot)

    # (root_kind, level) pairs, in precedence order, level-deduped.
    ordered: List[tuple] = [("user", Path.home())]
    for kind, root in (("launch", launch_dir), ("active", cwd)) + tuple(
            ("extra", r) for r in extra_roots):
        if root is None:
            continue
        for level in devdir_levels(root):
            ordered.append((kind, level))
    seen_levels: set = set()
    seen_paths: set = set()
    seen_hashes: set = set()
    out: List[DevdirFile] = []
    for kind, level in ordered:
        lk = str(level)
        if lk in seen_levels:
            continue
        seen_levels.add(lk)
        for devdir_name in DEVDIR_NAMES:
            for path in grammar(level, devdir_name):
                try:
                    rp = str(path.resolve())
                except Exception:
                    rp = str(path)
                if rp in seen_paths:
                    continue
                seen_paths.add(rp)
                try:
                    content = path.read_text()
                except Exception:
                    continue
                digest = hashlib.md5(content.encode("utf-8", "ignore")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                out.append(DevdirFile(
                    path=rp,
                    frontmatter=parse_frontmatter(content),
                    content=content,
                    source=DevdirSource(root=kind, level=lk, devdir=devdir_name),
                ))
    return out


RESOLVE = resolve   # the name the design speaks in

__all__ = ["RESOLVE", "resolve", "DevdirFile", "DevdirSource", "DEVDIR_NAMES",
           "SLOT_GRAMMARS", "devdir_levels", "dir_has_devdir", "parse_frontmatter",
           "MAX_DEVDIR_WALKUP"]
