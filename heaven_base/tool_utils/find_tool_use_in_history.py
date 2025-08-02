#!/usr/bin/env python3
"""
Utility function to find tool usage in HEAVEN history objects using unified event system
"""
from typing import List, Dict, Any, Optional
from ..memory.history import History
from ..memory.heaven_event import HeavenEvent, convert_langchain_to_heaven


def find_tool_use_in_history(history: History, tool_name: str) -> bool:
    """
    Check if a specific tool was used in the conversation history.
    
    Uses HEAVEN event system to unify detection across all frameworks:
    - Anthropic-style tool_use blocks in message content
    - LangChain-style tool_calls in AIMessage objects (OpenAI/Google)
    - ADK events
    
    Args:
        history: The History object to search
        tool_name: Name of the tool to look for (e.g., "SafeCodeReaderTool")
        
    Returns:
        bool: True if the tool was used, False otherwise
    """
    # Convert all messages to HEAVEN events for unified processing
    heaven_events = convert_langchain_to_heaven(history.messages)
    print("DEBUG: find_tool_use_in_history: printing heaven_events")
    print(heaven_events)
    # Look for TOOL_USE events with matching tool name
    for event_dict in heaven_events:
        if (event_dict.get('event_type') == 'TOOL_USE' and 
            event_dict.get('data', {}).get('name') == tool_name):
            return True
    
    return False


def get_all_tool_usage_in_history(history: History) -> List[Dict[str, Any]]:
    """
    Get all tool usage from the conversation history using HEAVEN events.
    
    Returns:
        List of dicts with tool usage info: [{"name": str, "args": dict, "call_id": str}, ...]
    """
    tool_usage = []
    
    # Convert all messages to HEAVEN events for unified processing
    heaven_events = convert_langchain_to_heaven(history.messages)
    
    # Extract all TOOL_USE events
    for event_dict in heaven_events:
        if event_dict.get('event_type') == 'TOOL_USE':
            data = event_dict.get('data', {})
            tool_usage.append({
                "name": data.get('name'),
                "args": data.get('input', {}),
                "call_id": data.get('id'),
                "provider": data.get('provider', 'unknown')
            })
    
    return tool_usage


def count_tool_calls_in_history(history: History) -> int:
    """
    Count the total number of tool calls in the history using HEAVEN events.
    
    Returns:
        int: Total number of tool calls
    """
    return len(get_all_tool_usage_in_history(history))





