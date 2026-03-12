"""SkillTool — gives any Heaven agent access to the skill system.

Actions: list, equip, get, get_equipped, search, list_personas, equip_persona
"""
from typing import Any, Dict, Optional, Type

from langchain.tools import Tool, BaseTool

from ..baseheaventool import BaseHeavenTool, ToolResult, ToolError, ToolArgsSchema


class SkillToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'skill_name': {
            'name': 'skill_name',
            'type': 'str',
            'description': 'Name of the skill to execute (get its content)',
            'required': True,
        },
    }


async def skill_tool_func(
    skill_name: str,
) -> str:
    """Execute a skill - forwards directly to skill manager, no checking."""
    try:
        from skill_manager.treeshell_functions import get_skill
    except ImportError:
        raise ToolError("skill_manager package not installed")
    
    # Just forward - no checking, no validation
    return get_skill(skill_name)


class SkillTool(BaseHeavenTool):
    name = "SkillTool"
    description = (
        "Execute a skill by name - gets skill content from skill manager."
    )
    func = skill_tool_func
    args_schema = SkillToolArgsSchema
    is_async = True
