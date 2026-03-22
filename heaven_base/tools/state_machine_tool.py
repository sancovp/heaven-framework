"""StateMachineTool — a Heaven tool wrapping KeywordBasedStateMachine.

The agent calls this tool to transition between states. The tool validates
the transition, executes it internally, saves state to disk, and returns
the new phase prompt as the tool result — immediately, in the same turn.

Follows the same pattern as NetworkEditTool, BashTool, etc:
- Class with static name/description/args_schema
- create() classmethod sets cls.func and calls super().create()
- Runtime data (the SM instance) passed to create() as a parameter
"""

from typing import Dict, Any, Optional
from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema
from ..state_machine import KeywordBasedStateMachine


class SMToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        "transition_to": {
            "name": "transition_to",
            "type": "str",
            "description": (
                "The state to transition to. Check the STATE_MACHINE "
                "section in your system prompt for valid targets."
            ),
            "required": True,
        },
        "reason": {
            "name": "reason",
            "type": "str",
            "description": "Brief explanation of why you are transitioning.",
            "required": True,
        },
    }


class StateMachineTool(BaseHeavenTool):
    name = "StateMachineTool"
    description = "Transition the state machine to a new state. Call this when you have completed your current phase and are ready to move to the next one."
    args_schema = SMToolArgsSchema
    is_async = True

    @classmethod
    def create(cls, adk: bool = False, state_machine: Optional[KeywordBasedStateMachine] = None, agent_name: str = "agent"):
        """Create a StateMachineTool instance bound to a specific state machine.

        Args:
            adk: Whether to use ADK mode.
            state_machine: The SM instance to wrap.
            agent_name: Agent name for persistence path.
        """
        sm = state_machine
        name = agent_name

        async def sm_transition_func(**kwargs):
            transition_to = kwargs.get("transition_to", "")
            reason = kwargs.get("reason", "")

            if not transition_to:
                return CLIResult(
                    output=f"❌ transition_to is required. "
                    f"Valid targets: {sm.valid_transitions}"
                )

            consumed = sm.process_kvs({transition_to: reason})

            if consumed:
                sm.save_state(name)

                new_config = sm.config
                lines = [
                    f"✅ Transitioned: {sm._history[-1]['from']} → {sm.current_state}",
                    f"Reason: {reason}",
                    f"Cycles completed: {sm.cycles_completed}",
                    "",
                ]

                if sm.is_terminal:
                    lines.append(
                        "🏁 TERMINAL STATE REACHED. This SM cycle is complete. "
                        "You may now call GOAL ACCOMPLISHED."
                    )
                else:
                    lines.append(f"📋 NEW PHASE: {sm.current_state}")
                    if new_config.goal:
                        lines.append(f"Goal: {new_config.goal}")
                    if new_config.prompt_suffix:
                        lines.append(f"\n{new_config.prompt_suffix}")
                    lines.append(f"\nValid next transitions: {sm.valid_transitions}")

                return CLIResult(output="\n".join(lines))
            else:
                return CLIResult(
                    output=f"❌ Invalid transition: {sm.current_state} → {transition_to}. "
                    f"Valid from '{sm.current_state}': {sm.valid_transitions}"
                )

        cls.func = sm_transition_func
        return super().create(adk=adk)


def create_sm_tool(state_machine: KeywordBasedStateMachine, agent_name: str):
    """Factory: creates a StateMachineTool for a specific SM + agent.

    Args:
        state_machine: The SM instance to wrap.
        agent_name: Agent name for persistence path.

    Returns:
        A configured StateMachineTool instance.
    """
    return StateMachineTool.create(state_machine=state_machine, agent_name=agent_name)
