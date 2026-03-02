"""Default hooks for BaseHeavenAgent.

skill_description_injection_hook: BEFORE_SYSTEM_PROMPT — appends equipped skill
    descriptions to the system prompt so the agent knows what skills it has.

skill_identity_injection_hook: BEFORE_TOOL_CALL — when agent calls sancrev
    treeshell skill-related actions, injects agent name as identity context.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..baseheavenagent import HookContext


def skill_description_injection_hook(ctx: "HookContext") -> None:
    """Inject equipped skill descriptions into the system prompt.

    Reads the agent's skillset from config. If set, gets equipped skill names
    and appends a summary block to the system prompt via ctx.data["system_prompt"].
    """
    agent = ctx.agent
    if not hasattr(agent, 'config') or agent.config is None:
        return
    if not getattr(agent.config, 'skillset', None):
        return

    try:
        from .skill_utils import get_equipped_skill_names
        names = get_equipped_skill_names()
        if not names:
            return

        # Build skill block to append
        skill_block = "\n\n<EQUIPPED_SKILLS>\n"
        skill_block += f"Skillset: {agent.config.skillset}\n"
        skill_block += f"Available skills: {', '.join(names)}\n"
        skill_block += "Use the Skill tool to execute any equipped skill by name.\n"
        skill_block += "</EQUIPPED_SKILLS>"

        # Modify system prompt through ctx.data
        current_prompt = ctx.prompt or getattr(agent.config, 'system_prompt', '')
        ctx.data["system_prompt"] = current_prompt + skill_block
    except Exception:
        pass  # Fail silently — skill injection is optional enhancement


def skill_identity_injection_hook(ctx: "HookContext") -> None:
    """Inject agent identity into sancrev treeshell skill calls.

    When the agent calls sancrev_treeshell with skill-related commands
    (equip, unequip, list_equipped, etc.), this hook adds the agent name
    as context so skillmanager can scope per-agent.
    """
    if not ctx.tool_name:
        return

    # Only intercept sancrev treeshell calls related to skills
    tool_lower = ctx.tool_name.lower()
    if 'treeshell' not in tool_lower and 'sancrev' not in tool_lower:
        return

    # Check if the command is skill-related
    command = ctx.tool_args.get('command', '')
    skill_commands = ['equip', 'unequip', 'list_equipped', 'get_equipped_content',
                      'list_skills', 'search_skills', 'equip_persona']
    if not any(cmd in command for cmd in skill_commands):
        return

    # Inject agent identity into the command context
    agent = ctx.agent
    agent_name = getattr(agent, 'name', 'unnamed_agent')
    ctx.data["agent_identity"] = agent_name
