"""Default hooks for Heaven agents — skill injection via hook system."""
from ..baseheavenagent import HookPoint, HookContext, HookRegistry


def make_skill_description_hook(skillset_name: str, agent_id: str = None):
    """BEFORE_SYSTEM_PROMPT hook: injects agent-scoped skill descriptions into system prompt.

    Uses SkillManager with agent_id for per-agent equipped state.
    Equips the persona's skillset on first call, then reads from agent-scoped equipped list.
    """
    _initialized = [False]

    def _inject_skill_descriptions(ctx: HookContext):
        try:
            import re
            from skill_manager.core import SkillManager
            manager = SkillManager(agent_id=agent_id)

            # First call: equip the persona's skillset into agent-scoped state
            if not _initialized[0] and skillset_name:
                manager.equip_skillset(skillset_name)
                _initialized[0] = True

            equipped = manager.list_equipped()
            if not equipped:
                return

            skill_block = "\n\n<EQUIPPED_SKILLS>\n"
            for s in equipped:
                desc = s.get("description", "")
                if desc:
                    skill_block += f"- {s['name']}: {desc}\n"
            skill_block += "</EQUIPPED_SKILLS>"

            # Strip any previous injection, then append
            current = ctx.data.get("system_prompt", ctx.prompt or "")
            current = re.sub(r'\n*<EQUIPPED_SKILLS>.*?</EQUIPPED_SKILLS>', '', current, flags=re.DOTALL)
            ctx.data["system_prompt"] = current + skill_block
        except Exception:
            pass  # Skill injection is best-effort, never block agent
    return _inject_skill_descriptions


def make_skill_identity_hook(agent_name: str):
    """BEFORE_TOOL_CALL hook: injects agent identity into sancrev treeshell skill calls.

    When agent calls sancrev treeshell with skill-related actions (equip, list_equipped, etc.),
    this hook injects the agent_name so skillmanager tracks per-agent state.
    """
    SKILL_ACTIONS = {"equip", "unequip", "list_equipped", "get_equipped_content", "unequip_all"}

    def _inject_identity(ctx: HookContext):
        tool_name = ctx.tool_name or ""
        tool_args = ctx.tool_args or {}
        # Only intercept sancrev treeshell calls
        if "treeshell" not in tool_name.lower() and "sancrev" not in tool_name.lower():
            return
        command = tool_args.get("command", "")
        # Check if this is a skill-related action
        for action in SKILL_ACTIONS:
            if action in command:
                # Inject agent identity if not already present
                if "agent_id" not in command:
                    ctx.data["injected_identity"] = agent_name
                break
    return _inject_identity


def register_skill_hooks(registry: HookRegistry, agent_name: str, skillset_name: str = None):
    """Register both default skill hooks on a HookRegistry.

    Args:
        registry: The HookRegistry to register hooks on
        agent_name: Agent name for identity injection and agent-scoped skill state
        skillset_name: Optional skillset name for description injection
    """
    if skillset_name:
        registry.register(HookPoint.BEFORE_SYSTEM_PROMPT, make_skill_description_hook(skillset_name, agent_id=agent_name))
    registry.register(HookPoint.BEFORE_TOOL_CALL, make_skill_identity_hook(agent_name))
