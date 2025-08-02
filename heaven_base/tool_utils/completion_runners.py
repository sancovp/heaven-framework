"""
Completion-style execution runners for HEAVEN agents.

This module provides unified agent execution functions that support both:
- Full agent mode (with goal/iterations formatting)
- Completion style (direct prompt → response)

Automatically routes between local and Docker execution based on container differences.
"""

from typing import Optional, Union, List, Type
from ..baseheavenagent import HeavenAgentConfig, BaseHeavenAgentReplicant
from .hermes_utils import (
    exec_agent_run_locally_without_docker,
    exec_agent_run_via_docker
)


async def exec_agent_run(
    prompt: str,
    agent: Optional[Union[str, HeavenAgentConfig, Type[BaseHeavenAgentReplicant]]] = None,
    target_container: Optional[str] = None,
    history_id: Optional[str] = None,
    remove_agents_config_tools: bool = False,
    orchestration_preprocess: bool = False,
    system_prompt_suffix: Optional[str] = None
):
    """
    Completion-style agent execution that routes to local or Docker based on containers.
    
    Always uses completion style (direct prompt → response) with iterations=1.
    
    Args:
        prompt: The prompt to send directly to the model
        agent: Agent name, config, or replicant class
        target_container: Container to execute in
        history_id: Optional history to continue
        
    Returns:
        Direct model response without agent formatting
        
    Routing Logic:
        - If agent is HeavenAgentConfig: use local execution
        - If agent is string/None and target_container differs from default: use Docker execution
        - Otherwise: use local execution
    """
    
    # Default container
    source_container = "mind_of_god"
    if target_container is None:
        target_container = "mind_of_god"
    
    # Determine execution method
    should_execute_locally = (
        source_container == target_container or 
        isinstance(agent, HeavenAgentConfig)  # HeavenAgentConfig can't be JSON serialized for Docker
    )
    
    if should_execute_locally:
        return await exec_agent_run_locally_without_docker(
            target_container=target_container,
            goal=prompt,
            iterations=1,  # Always 1 for completion style
            agent=agent,
            source_container=source_container,
            history_id=history_id,
            return_summary=False,
            ai_messages_only=False,
            remove_agents_config_tools=remove_agents_config_tools,
            orchestration_preprocess=orchestration_preprocess,
            continuation=None,
            additional_tools=None,
            system_prompt_suffix=system_prompt_suffix,
            agent_mode=False  # Always completion style
        )
    else:
        # Note: Docker execution requires agent to be string or None (not HeavenAgentConfig)
        if isinstance(agent, HeavenAgentConfig):
            raise ValueError("Docker execution doesn't support HeavenAgentConfig objects. Use agent name string instead.")
            
        return exec_agent_run_via_docker(
            target_container=target_container,
            goal=prompt,
            iterations=1,  # Always 1 for completion style
            agent=agent,
            source_container=source_container,
            history_id=history_id,
            return_summary=False,
            ai_messages_only=False,
            remove_agents_config_tools=remove_agents_config_tools,
            orchestration_preprocess=orchestration_preprocess,
            continuation=None,
            additional_tools=None,
            system_prompt_suffix=system_prompt_suffix,
            agent_mode=False  # Always completion style
        )


async def exec_completion_style(
    prompt: str,
    agent: Optional[Union[str, HeavenAgentConfig, Type[BaseHeavenAgentReplicant]]] = None,
    target_container: Optional[str] = None,
    source_container: Optional[str] = None,
    history_id: Optional[str] = None,
    **kwargs
):
    """
    Convenience function for completion-style execution (agent_mode=False).
    
    Args:
        prompt: Direct prompt to send to model
        agent: Agent to use for execution
        target_container: Container to execute in
        source_container: Container executing from
        history_id: Optional history to continue
        **kwargs: Additional arguments passed to exec_agent_run
        
    Returns:
        Direct model response without agent formatting
    """
    return await exec_agent_run(
        prompt=prompt,
        agent=agent,
        target_container=target_container,
        history_id=history_id,
        **kwargs
    )


# Test the completion runners
if __name__ == "__main__":
    import asyncio
    from computer_use_demo.tools.base.baseheavenagent import HeavenAgentConfig
    from computer_use_demo.tools.base.unified_chat import ProviderEnum
    
    async def test_completion_runners():
        print("=== Testing Completion Runners ===\n")
        
        # Test config
        test_agent_config = HeavenAgentConfig(
            name="CompletionTestAgent",
            system_prompt="You are a helpful assistant. Give direct, concise answers.",
            tools=[],
            provider=ProviderEnum.ANTHROPIC,
            model="claude-3-5-sonnet-latest", 
            temperature=0.3
        )
        
        # Test 1: Local execution with HeavenAgentConfig
        print("=== TEST 1: Local execution (HeavenAgentConfig) ===")
        try:
            result = await exec_agent_run(
                prompt="What is 2+2?",
                agent=test_agent_config
            )
            print("✅ Local execution succeeded")
            messages = result.get('messages', [])
            for msg in messages[-2:]:
                if hasattr(msg, 'content'):
                    print(f"{msg.__class__.__name__}: {msg.content}")
        except Exception as e:
            print(f"❌ Local execution failed: {e}")
        
        print("\n" + "="*50 + "\n")
        
        # Test 2: Docker execution with string agent
        print("=== TEST 2: Docker execution (string agent, different container) ===")
        try:
            result = await exec_agent_run(
                prompt="What is 3+3?",
                agent="basic_heaven_agent",
                target_container="mind_of_god"  # Different from default
            )
            print("✅ Docker execution succeeded")
            messages = result.get('messages', [])
            for msg in messages[-2:]:
                if hasattr(msg, 'content'):
                    print(f"{msg.__class__.__name__}: {msg.content}")
        except Exception as e:
            print(f"❌ Docker execution failed: {e}")
        
        print("\n" + "="*50 + "\n")
        
        # Test 3: Convenience function
        print("=== TEST 3: exec_completion_style convenience function ===")
        try:
            result = await exec_completion_style(
                prompt="What is 4+4?",
                agent=test_agent_config
            )
            print("✅ Convenience function succeeded")
            messages = result.get('messages', [])
            for msg in messages[-2:]:
                if hasattr(msg, 'content'):
                    print(f"{msg.__class__.__name__}: {msg.content}")
        except Exception as e:
            print(f"❌ Convenience function failed: {e}")
    
    asyncio.run(test_completion_runners())


