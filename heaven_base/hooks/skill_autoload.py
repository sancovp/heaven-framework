"""Skill auto-loader — dep-free heaven-skills parity (NO skill_manager, NO carton).

Injects the AVAILABLE-SKILLS catalog (name + description + path) into the SYSTEM PROMPT so EVERY agent
knows what skills it has, with zero dependencies. Progressive disclosure: only name/description/path are
injected; the agent READS the full SKILL.md on demand (so the catalog stays cheap even with many skills).

Scans both skill layouts, deduped by name:
  - `<dir>/*/SKILL.md`   (the Claude-Code-plugin / heaven dir form)
  - `<dir>/*.md`          (the flat form the ONION morph system scaffolds)
across these roots:
  - `$HEAVEN_DATA_DIR/skills`     (global heaven skills)
  - `<cwd>/.heaven/skills`        (dir-relative — what ONION writes)
  - `~/.heaven/skills`            (user-global)

One hook: a BEFORE_SYSTEM_PROMPT renderer, idempotent (strip the prior block, re-append) and hot (re-scan
each turn). This is the dep-free alternative to heaven's native `make_skill_description_hook`, which uses
`skill_manager.equip_skillset`/`list_equipped` and therefore pulls in `carton_mcp` — we don't, because we
aren't using equip/list; we only need the skills to auto-load.
"""
import os
import re
import glob
import logging
from pathlib import Path
from typing import Optional

from ..baseheavenagent import HookPoint, HookContext, HookRegistry

logger = logging.getLogger(__name__)

PER_SKILL_DESC_CHARS = 600       # cap each description so the catalog stays cheap
MAX_SKILLS = 300                 # backstop on a runaway skills dir


def _skill_roots() -> list:
    roots = []
    hdd = os.environ.get("HEAVEN_DATA_DIR")
    if hdd:
        roots.append(os.path.join(hdd, "skills"))
    roots.append(os.path.join(os.getcwd(), ".heaven", "skills"))
    roots.append(str(Path.home() / ".heaven" / "skills"))
    return roots


def collect_skills() -> Optional[str]:
    """Assemble the AVAILABLE_SKILLS block (deduped by name, both layouts), or None if there are no skills."""
    seen, lines = set(), []
    for root in _skill_roots():
        paths = sorted(glob.glob(os.path.join(root, "*", "SKILL.md")))   # dir form (CC plugin / heaven)
        paths += sorted(glob.glob(os.path.join(root, "*.md")))           # flat form (ONION morphs)
        for sk in paths:
            try:
                text = Path(sk).read_text()[:4000]
            except Exception:
                continue
            nm = re.search(r"^name:\s*(.+)$", text, re.M)
            ds = re.search(r"^description:\s*(.+)$", text, re.M)
            # name: frontmatter > parent dir (SKILL.md) > filename stem (flat .md)
            if nm:
                name = nm.group(1).strip()
            elif os.path.basename(sk) == "SKILL.md":
                name = os.path.basename(os.path.dirname(sk))
            else:
                name = os.path.splitext(os.path.basename(sk))[0]
            if name in seen:
                continue
            seen.add(name)
            # description: frontmatter > first non-blank, non-heading, non-fence line
            if ds:
                desc = ds.group(1).strip()
            else:
                desc = next((ln.strip() for ln in text.splitlines()
                             if ln.strip() and not ln.lstrip().startswith("#") and not ln.strip().startswith("---")), "")
            lines.append(f"- **{name}** — {desc[:PER_SKILL_DESC_CHARS]}  (full skill: read {sk})")
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
