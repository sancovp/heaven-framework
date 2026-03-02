"""Skill execution utility — check equipped skills and return content."""

from typing import Optional

# Module-level agent context — set by BaseHeavenAgent at init time
# When set, SkillManager uses agent-scoped mode (no Claude Code sync)
_current_agent_id: Optional[str] = None


def set_agent_context(agent_id: Optional[str]):
    """Set the current agent context for skill operations."""
    global _current_agent_id
    _current_agent_id = agent_id


def _get_skill_manager():
    """Lazy import SkillManager to avoid import-time dependency."""
    from skill_manager.core import SkillManager
    return SkillManager(agent_id=_current_agent_id)


def execute_skill(skill_name: str) -> str:
    """Get SKILL.md content and path for an equipped skill by name.

    Returns skill content + path if equipped, error message if not.
    This is the dynamic enum: only equipped skills are valid inputs.
    """
    manager = _get_skill_manager()
    equipped_list = manager.list_equipped()  # list[dict] with 'name' key
    equipped_names = [s["name"] for s in equipped_list]

    if skill_name not in equipped_names:
        available = ", ".join(equipped_names) if equipped_names else "(none equipped)"
        return f"ERROR: Skill '{skill_name}' is not equipped. Equipped skills: {available}"

    skill_md_path = manager._skill_md_path(skill_name)
    if not skill_md_path.exists():
        return f"ERROR: Skill '{skill_name}' SKILL.md not found at {skill_md_path}"

    content = skill_md_path.read_text()
    skill_dir = str(skill_md_path.parent)

    return f"Skill path: {skill_dir}\nSKILL.md path: {skill_md_path}\n\n{content}"


def get_equipped_skill_names() -> list[str]:
    """Return list of currently equipped skill names."""
    try:
        manager = _get_skill_manager()
        equipped_list = manager.list_equipped()
        return [s["name"] for s in equipped_list]
    except Exception:
        return []
