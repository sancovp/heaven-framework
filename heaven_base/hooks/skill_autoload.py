"""Skill auto-loader — dep-free heaven-skills parity (NO skill_manager, NO carton).

Injects the AVAILABLE-SKILLS catalog (name + description + path) into the SYSTEM PROMPT so EVERY agent
knows what skills it has, with zero dependencies. Progressive disclosure: only name/description/path are
injected; the agent READS the full SKILL.md on demand (so the catalog stays cheap even with many skills).

FILE-FINDING = `heaven_base.devdir` (THE ONE RESOLVER): `devdir.resolve(cwd, None, "skills")` — the
user root (`~/.claude` + `~/.heaven`) always-on plus the cwd's ancestor walk — plus a
`devdir.scan_slot_dir` over `$HEAVEN_DATA_DIR/skills` (the registry-bind store, not a devdir level).
Name collisions: NEAREST WINS (the D1 project-wins ruling) — the resolver list is iterated reversed.

One hook: a BEFORE_SYSTEM_PROMPT renderer, idempotent (strip the prior block, re-append) and hot (re-scan
each turn). This is the dep-free alternative to heaven's native `make_skill_description_hook`, which uses
`skill_manager.equip_skillset`/`list_equipped` and therefore pulls in `carton_mcp` — we don't, because we
aren't using equip/list; we only need the skills to auto-load.
"""
import os
import re
import logging
from pathlib import Path
from typing import Optional

from .. import devdir
from ..baseheavenagent import HookPoint, HookContext, HookRegistry

logger = logging.getLogger(__name__)

PER_SKILL_DESC_CHARS = 600       # cap each description so the catalog stays cheap
MAX_SKILLS = 300                 # backstop on a runaway skills dir


def skill_summary(f: "devdir.DevdirFile") -> dict:
    """{name, description, path} from a resolved skill file.
    name: frontmatter > parent dir (SKILL.md) > filename stem (flat .md).
    description: frontmatter > first non-blank, non-heading, non-fence line."""
    p = Path(f.path)
    name = f.frontmatter.get("name") or (p.parent.name if p.name == "SKILL.md" else p.stem)
    desc = f.frontmatter.get("description")
    if not desc:
        desc = next((ln.strip() for ln in f.content[:4000].splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#") and not ln.strip().startswith("---")), "")
    return {"name": name, "description": desc[:PER_SKILL_DESC_CHARS], "path": f.path}


def collect_skill_files() -> list:
    """Every skill file, priority-ordered for name dedup (nearest devdir level first, then the
    registry store), through THE ONE RESOLVER."""
    files = list(reversed(devdir.resolve(os.getcwd(), None, "skills")))   # nearest wins (D1)
    hdd = os.environ.get("HEAVEN_DATA_DIR")
    if hdd:
        files += devdir.scan_slot_dir(Path(hdd) / "skills", "skills")
    return files


def collect_skills() -> Optional[str]:
    """Assemble the AVAILABLE_SKILLS block (deduped by name), or None if there are no skills."""
    seen, lines = set(), []
    for f in collect_skill_files():
        s = skill_summary(f)
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        lines.append(f"- **{s['name']}** — {s['description']}  (full skill: read {s['path']})")
        if len(lines) >= MAX_SKILLS:
            break
    if not lines:
        return None
    return ("\n\n<AVAILABLE_SKILLS>\n"
            "Skills available to you — when one fits the task, READ its SKILL.md path for the full "
            "instructions, then follow it:\n" + "\n".join(lines) + "\n</AVAILABLE_SKILLS>")


def make_skill_autoload_hook():
    """Return a BEFORE_SYSTEM_PROMPT renderer that injects the skills catalog idempotently + hot."""
    def renderer(ctx: HookContext):
        try:
            current = ctx.data.get("system_prompt", ctx.prompt or "")
            current = re.sub(r'\n*<AVAILABLE_SKILLS>.*?</AVAILABLE_SKILLS>', '', current, flags=re.DOTALL)  # idempotent
            block = collect_skills()                                    # hot: re-scan each turn
            ctx.data["system_prompt"] = current + block if block else current
        except Exception:
            pass  # never block the agent

    return renderer


def register_skill_autoload(registry: HookRegistry):
    """Wire the dep-free skill auto-loader: a BEFORE_SYSTEM_PROMPT renderer (intended for EVERY agent)."""
    registry.register(HookPoint.BEFORE_SYSTEM_PROMPT, make_skill_autoload_hook())
