#!/usr/bin/env python3
"""
Agent Config Test Utility - Run agents with dynamic JSON configs.

Only supported provider: MiniMax (via Anthropic-compatible API).
Tools must be passed as BaseHeavenTool classes, not strings.
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from ..unified_chat import UnifiedChat, ProviderEnum
from ..memory.history import History
from ..baseheaventool import ToolResult, ToolError
from .find_tool_use_in_history import (
    find_tool_use_in_history,
    count_tool_calls_in_history
)

async def agent_config_test(
    test_prompt: str,
    system_prompt: str,
    iterations: int = 1,
    agent_mode: bool = True,
    name: str = "DynamicTestAgent",
    tools: Optional[List] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    thinking_budget: Optional[int] = None,
    additional_kws: Optional[List[str]] = None,
    additional_kw_instructions: str = "",
    known_config_paths: Optional[List[str]] = None,
    prompt_suffix_blocks: Optional[List[str]] = None,
    max_tool_calls: int = 10,
    orchestrator: bool = False,
    history_id: Optional[str] = None,
    system_prompt_suffix: Optional[str] = None,
    adk: bool = False,
    duo_enabled: bool = False,
    run_on_langchain: bool = False,
    assert_tool_used: Optional[str] = None,
    assert_no_errors: bool = False,
    assert_goal_accomplished: bool = False,
    assert_extracted_keys: Optional[List[str]] = None,
    assert_extracted_contains: Optional[Dict[str, Any]] = None,
    assert_min_tool_calls: Optional[int] = None,
    assert_output_contains: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test an agent configuration by running it with a live LLM call.

    Args:
        test_prompt: The prompt to test with
        system_prompt: System prompt for the agent
        tools: List of BaseHeavenTool CLASSES (not strings)
        provider: "anthropic" (MiniMax uses this path)
        model: Model name (e.g. "MiniMax-M2.5")
        Other params: see HeavenAgentConfig and BaseHeavenAgent

    Returns:
        Dict with success, tool_calls, message_count, assertions, etc.
    """
    try:
        from ..baseheavenagent import BaseHeavenAgent, HeavenAgentConfig

        tool_classes = tools if tools else []

        if isinstance(provider, str):
            provider_map = {
                'anthropic': ProviderEnum.ANTHROPIC,
                'openai': ProviderEnum.OPENAI,
                'google': ProviderEnum.GOOGLE
            }
            provider_enum = provider_map.get(provider.lower(), ProviderEnum.ANTHROPIC)
        else:
            provider_enum = provider

        agent_config = HeavenAgentConfig(
            name=name,
            system_prompt=system_prompt,
            tools=tool_classes,
            provider=provider_enum,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            additional_kws=additional_kws or [],
            additional_kw_instructions=additional_kw_instructions,
            known_config_paths=known_config_paths,
            prompt_suffix_blocks=prompt_suffix_blocks
        )

        agent = BaseHeavenAgent(
            config=agent_config,
            unified_chat=UnifiedChat(),
            max_tool_calls=max_tool_calls,
            orchestrator=orchestrator,
            history=History(messages=[]) if not history_id else None,
            history_id=history_id,
            system_prompt_suffix=system_prompt_suffix,
            adk=adk,
            duo_enabled=duo_enabled,
            run_on_langchain=run_on_langchain
        )

        if agent_mode:
            formatted_prompt = f"agent goal={test_prompt}, iterations={iterations}"
        else:
            formatted_prompt = test_prompt

        result = await agent.run(prompt=formatted_prompt)

        message_count = 0
        tool_calls = 0

        if isinstance(result, dict) and "history" in result:
            history = result["history"]
            message_count = len(history.messages)
            tool_calls = count_tool_calls_in_history(history)

        extracted_content = {}
        if (isinstance(result, dict) and
            result.get('agent_status') and
            result['agent_status'].extracted_content):
            extracted_content = result['agent_status'].extracted_content

        assertion_results = {}
        all_assertions_passed = True

        if assert_tool_used:
            tool_was_used = False
            if isinstance(result, dict) and "history" in result:
                tool_was_used = find_tool_use_in_history(result["history"], assert_tool_used)
            assertion_results["tool_used"] = {
                "passed": tool_was_used,
                "expected": assert_tool_used,
                "message": f"Tool '{assert_tool_used}' {'was' if tool_was_used else 'was NOT'} used"
            }
            if not tool_was_used:
                all_assertions_passed = False

        if assert_min_tool_calls is not None:
            enough_calls = tool_calls >= assert_min_tool_calls
            assertion_results["min_tool_calls"] = {
                "passed": enough_calls,
                "expected": f">= {assert_min_tool_calls}",
                "actual": tool_calls,
                "message": f"Made {tool_calls} tool calls, {'enough' if enough_calls else 'not enough'}"
            }
            if not enough_calls:
                all_assertions_passed = False

        return {
            "success": True,
            "final_output": history.messages,
            "message_count": message_count,
            "tool_calls": tool_calls,
            "history_id": result.get("history_id") if isinstance(result, dict) else None,
            "agent_status": result.get("agent_status") if isinstance(result, dict) else None,
            "extracted_content": extracted_content,
            "config_used": {
                "name": agent_config.name,
                "provider": agent_config.provider.value,
                "model": agent_config.model,
                "temperature": agent_config.temperature,
                "max_tokens": agent_config.max_tokens,
                "tools": [t.__name__ if hasattr(t, '__name__') else str(t) for t in tool_classes],
                "max_tool_calls": max_tool_calls,
                "orchestrator": orchestrator,
                "adk": adk,
                "duo_enabled": duo_enabled,
                "run_on_langchain": run_on_langchain
            },
            "assertions": {
                "all_passed": all_assertions_passed,
                "results": assertion_results,
                "count": len(assertion_results)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "final_output": "",
            "message_count": 0,
            "tool_calls": 0,
            "extracted_content": {},
            "assertions": {
                "all_passed": False,
                "results": {},
                "count": 0
            }
        }
