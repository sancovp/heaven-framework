"""StateMachineTool — a Heaven tool wrapping KeywordBasedStateMachine.

The agent calls this tool to transition between states. The tool validates
the transition, executes it internally, saves state to disk, and returns
the new phase prompt as the tool result — immediately, in the same turn.

Factory: create_sm_tool(state_machine, agent_name)
Called from BaseHeavenAgent.__init__ when config.state_machine is set.
"""

from typing import Dict, Any
from ..state_machine import KeywordBasedStateMachine


def create_sm_tool(state_machine: KeywordBasedStateMachine, agent_name: str):
    """Factory: creates a StateMachineTool for a specific SM + agent.

    The closure captures both `sm` and `agent_name` so:
    - Transitions are validated against the SM's transition map
    - State is persisted to disk after every transition
    - The agent gets phase instructions back immediately

    Args:
        state_machine: The SM instance to wrap.
        agent_name: Agent name for persistence path.

    Returns:
        A configured BaseHeavenTool instance.
    """
    # Lazy imports to avoid circular dependency
    from ..baseheaventool import (
        BaseHeavenTool, ToolArgsSchema, CLIResult, ToolError,
    )
    from langchain_core.tools import Tool

    sm = state_machine
    name = agent_name

    # Schema
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

    # The closure
    async def sm_transition_func(**kwargs):
        transition_to = kwargs.get("transition_to", "")
        reason = kwargs.get("reason", "")

        if not transition_to:
            return CLIResult(
                output=f"❌ transition_to is required. "
                f"Valid targets: {sm.valid_transitions}"
            )

        # Try the transition
        consumed = sm.process_kvs({transition_to: reason})

        if consumed:
            # Persist to disk immediately
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

    # Build tool description with current SM info
    description = (
        f"Transition the '{sm.name}' state machine to a new state. "
        f"Call this when you have completed your current phase and are "
        f"ready to move to the next one. Returns your new phase instructions "
        f"immediately.\n\n"
        f"States: {sm.state_keywords}\n"
        f"Terminal states: {sorted(sm.terminal_states)}"
    )

    # Create the LangChain Tool directly (simpler than subclassing)
    schema_instance = SMToolArgsSchema()
    pydantic_schema = SMToolArgsSchema.to_pydantic_schema(schema_instance.arguments)

    def sync_stub(**kwargs):
        raise NotImplementedError("StateMachineTool is async only")

    base_tool = Tool(
        name="StateMachineTool",
        description=description,
        func=sync_stub,
        coroutine=sm_transition_func,
        args_schema=pydantic_schema,
    )

    return BaseHeavenTool(
        base_tool=base_tool,
        args_schema=SMToolArgsSchema,
        is_async=True,
    )
