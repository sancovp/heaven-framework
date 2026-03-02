"""SkillTool — gives any Heaven agent access to the skill system.

Actions: list, equip, get, get_equipped, search, list_personas, equip_persona
"""
from typing import Any, Dict, Optional, Type

from langchain.tools import Tool, BaseTool

from ..baseheaventool import BaseHeavenTool, ToolResult, ToolError, ToolArgsSchema


class SkillToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'action': {
            'name': 'action',
            'type': 'str',
            'description': (
                'The skill action to perform. One of: '
                'list, equip, get, get_equipped, search, '
                'list_personas, equip_persona, list_skillsets, equip_skillset'
            ),
            'required': True,
        },
        'name': {
            'name': 'name',
            'type': 'str',
            'description': 'Skill/persona/skillset name (required for equip, get, equip_persona, equip_skillset)',
            'required': False,
        },
        'query': {
            'name': 'query',
            'type': 'str',
            'description': 'Search query (required for search action)',
            'required': False,
        },
    }


async def skill_tool_func(
    action: str,
    name: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """Execute a skill manager action."""
    try:
        from skill_manager.treeshell_functions import (
            list_skills,
            get_skill,
            equip,
            get_equipped_content,
            search_skills,
            list_skillsets,
            list_personas,
            equip_persona,
        )
    except ImportError:
        raise ToolError("skill_manager package not installed")

    action = action.strip().lower()

    if action == "list":
        return list_skills()

    elif action == "equip":
        if not name:
            raise ToolError("'name' is required for equip action")
        return equip(name)

    elif action == "get":
        if not name:
            raise ToolError("'name' is required for get action")
        return get_skill(name)

    elif action == "get_equipped":
        return get_equipped_content()

    elif action == "search":
        if not query:
            raise ToolError("'query' is required for search action")
        return search_skills(query)

    elif action == "list_personas":
        return list_personas()

    elif action == "equip_persona":
        if not name:
            raise ToolError("'name' is required for equip_persona action")
        return equip_persona(name)

    elif action == "list_skillsets":
        return list_skillsets()

    elif action == "equip_skillset":
        if not name:
            raise ToolError("'name' is required for equip_skillset action")
        from skill_manager.treeshell_functions import create_skillset
        # equip_skillset not exposed directly, use equip on the skillset
        return equip(name)

    else:
        raise ToolError(
            f"Unknown action '{action}'. Valid: list, equip, get, get_equipped, "
            f"search, list_personas, equip_persona, list_skillsets, equip_skillset"
        )


class SkillTool(BaseHeavenTool):
    name = "SkillTool"
    description = (
        "Manage skills — list, equip, get, search skills/personas/skillsets. "
        "Skills provide domain knowledge and workflow templates."
    )
    func = skill_tool_func
    args_schema = SkillToolArgsSchema
    is_async = True
