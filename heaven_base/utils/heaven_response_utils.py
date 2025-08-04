#!/usr/bin/env python3
"""
Utility functions for extracting responses from HEAVEN completion results.
Handles the iteration-based history system properly.
"""

from typing import Optional, Any, Dict, List
import os
import json
import glob


def extract_last_ai_message_from_iteration(iteration_messages: List[Any]) -> Optional[str]:
    """
    Extract the last AI message from a single iteration's messages.
    
    Args:
        iteration_messages: List of messages from one iteration
        
    Returns:
        String content of the last AIMessage, or None if not found
    """
    for msg in reversed(iteration_messages):
        if hasattr(msg, 'content') and hasattr(msg, '__class__'):
            if msg.__class__.__name__ == 'AIMessage' and msg.content:
                return msg.content
    return None


def extract_response_from_history_iterations(history_obj: Any) -> Optional[str]:
    """
    Extract AI response from a HEAVEN History object using iterations.
    
    Args:
        history_obj: HEAVEN History object with iterations property
        
    Returns:
        String content of the last AI response, or None if not found
    """
    if not hasattr(history_obj, 'iterations'):
        return None
        
    iterations = history_obj.iterations
    if not iterations:
        return None
    
    # Get the last iteration (highest numbered iteration_N)
    last_iteration_key = max(iterations.keys())
    last_iteration_messages = iterations[last_iteration_key]
    
    return extract_last_ai_message_from_iteration(last_iteration_messages)


def extract_response_from_legacy_messages(raw_result: Dict[str, Any]) -> Optional[str]:
    """
    Fallback method for extracting response from legacy message format.
    
    Args:
        raw_result: Dictionary with "messages" key containing message list
        
    Returns:
        String content of the last AIMessage, or None if not found
    """
    if not isinstance(raw_result, dict) or "messages" not in raw_result:
        return None
        
    messages = raw_result["messages"]
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get('type') == 'AIMessage' and msg.get('content'):
            return msg['content']
    return None


def load_history_by_id(history_id: str) -> Optional[Any]:
    """
    Load a HEAVEN History object by its ID from HEAVEN_DATA_DIR.
    
    Args:
        history_id: The history ID to load
        
    Returns:
        History object if found and loaded successfully, None otherwise
    """
    try:
        from heaven_base.memory.history import History
        
        # Use the existing History._load_history_file method
        history = History._load_history_file(history_id)
        return history
        
    except Exception as e:
        print(f"Error loading history {history_id}: {e}")
        return None


def extract_history_id_from_result(completion_result: Dict[str, Any]) -> Optional[str]:
    """
    Extract history_id from HEAVEN runner result (works for both completion_runner and hermes_runner).
    
    Args:
        completion_result: Result dictionary from completion_runner or hermes_runner
        
    Returns:
        History ID string if found, None otherwise
    """
    if not isinstance(completion_result, dict) or "results" not in completion_result:
        return None
    
    results_list = completion_result["results"]
    if not results_list:
        return None
    
    last_result = results_list[-1]
    if "raw_result" not in last_result:
        return None
    
    raw_result = last_result["raw_result"]
    
    # For completion_runner: history_id is directly in raw_result
    if isinstance(raw_result, dict) and "history_id" in raw_result:
        return raw_result["history_id"]
    
    # For hermes_runner: history_id is in nested raw_result
    if isinstance(raw_result, dict) and "raw_result" in raw_result:
        inner_raw_result = raw_result["raw_result"]
        if isinstance(inner_raw_result, dict) and "history_id" in inner_raw_result:
            return inner_raw_result["history_id"]
    
    return None


def extract_heaven_response(completion_result: Dict[str, Any]) -> str:
    """
    Main utility function to extract AI response from HEAVEN completion_runner result.
    Loads History object by ID and computes iterations to extract response.
    
    Args:
        completion_result: Result dictionary from completion_runner or hermes_runner
        
    Returns:
        AI response string, or "No response" if extraction fails
    """
    # Step 1: Extract history_id from result
    history_id = extract_history_id_from_result(completion_result)
    if not history_id:
        # Fallback to old method for legacy results
        return extract_heaven_response_legacy(completion_result)
    
    # Step 2: Load History object by ID
    history = load_history_by_id(history_id)
    if not history:
        return "No response - could not load history"
    
    # Step 3: Extract response from History iterations
    response = extract_response_from_history_iterations(history)
    if response:
        return response
    
    return "No response"


def extract_heaven_response_legacy(completion_result: Dict[str, Any]) -> str:
    """
    Legacy method for extracting response directly from result dict (fallback).
    
    Args:
        completion_result: Result dictionary from completion_runner
        
    Returns:
        AI response string, or "No response" if extraction fails
    """
    if not isinstance(completion_result, dict) or "results" not in completion_result:
        return "No response"
    
    results_list = completion_result["results"]
    if not results_list:
        return "No response"
    
    last_result = results_list[-1]
    if "raw_result" not in last_result:
        return "No response"
    
    raw_result = last_result["raw_result"]
    
    # Try iteration-based extraction (preferred method)
    response = extract_response_from_history_iterations(raw_result)
    if response:
        return response
    
    # Fallback to legacy message extraction
    response = extract_response_from_legacy_messages(raw_result)
    if response:
        return response
    
    return "No response"


def format_iteration_to_markdown(iteration_key: str, iteration_messages: List[Any]) -> str:
    """
    Format a single iteration's messages into readable markdown.
    
    Args:
        iteration_key: The iteration identifier (e.g., "iteration_0")
        iteration_messages: List of messages/events in this iteration
        
    Returns:
        Formatted markdown string for this iteration
    """
    md_parts = [f"## {iteration_key.replace('_', ' ').title()}"]
    
    for i, msg in enumerate(iteration_messages):
        # Handle LangChain BaseMessage types
        if hasattr(msg, '__class__'):
            msg_type = msg.__class__.__name__
            
            if msg_type == 'SystemMessage':
                md_parts.append(f"**🔧 System**: {msg.content}")
                
            elif msg_type == 'HumanMessage':
                md_parts.append(f"**👤 User**: {msg.content}")
                
            elif msg_type == 'AIMessage':
                md_parts.append(f"**🤖 AI**: {msg.content}")
                
                # Handle tool calls in AI messages
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('function', {}).get('name', 'unknown_tool')
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        md_parts.append(f"  **🔧 Tool Call**: `{tool_name}({tool_args})`")
                
                # Handle additional_kwargs tool calls
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('tool_calls'):
                    for tool_call in msg.additional_kwargs['tool_calls']:
                        tool_name = tool_call.get('function', {}).get('name', 'unknown_tool')
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        md_parts.append(f"  **🔧 Tool Call**: `{tool_name}({tool_args})`")
                        
            elif msg_type == 'ToolMessage':
                tool_id = getattr(msg, 'tool_call_id', 'unknown')
                md_parts.append(f"**🛠️ Tool Result** (id: `{tool_id}`):")
                md_parts.append(f"```\n{msg.content}\n```")
        
        # Handle ADK Events
        elif hasattr(msg, 'author') and hasattr(msg, 'content'):
            author = getattr(msg, 'author', 'unknown')
            timestamp = getattr(msg, 'timestamp', None)
            event_id = getattr(msg, 'id', 'unknown')
            
            time_str = f" (at {timestamp})" if timestamp else ""
            md_parts.append(f"**📡 ADK Event** - {author}{time_str} (id: `{event_id}`):")
            
            # Handle ADK content parts
            content = getattr(msg, 'content', None)
            if hasattr(content, 'parts'):
                for part in content.parts:
                    if hasattr(part, 'function_call'):
                        fc = part.function_call
                        md_parts.append(f"  **🔧 Function Call**: `{fc.name}({fc.args})`")
                    elif hasattr(part, 'function_response'):
                        fr = part.function_response
                        md_parts.append(f"  **🛠️ Function Response**:")
                        md_parts.append(f"  ```\n  {fr.response}\n  ```")
                    elif hasattr(part, 'text'):
                        md_parts.append(f"  **💬 Text**: {part.text}")
            elif content:
                md_parts.append(f"  {content}")
        
        # Handle dictionary format messages (fallback)
        elif isinstance(msg, dict):
            msg_type = msg.get('type', 'unknown')
            if msg_type == 'SystemMessage':
                md_parts.append(f"**🔧 System**: {msg.get('content', '')}")
            elif msg_type == 'HumanMessage':
                md_parts.append(f"**👤 User**: {msg.get('content', '')}")
            elif msg_type == 'AIMessage':
                md_parts.append(f"**🤖 AI**: {msg.get('content', '')}")
            elif msg_type == 'ToolMessage':
                tool_id = msg.get('tool_call_id', 'unknown')
                md_parts.append(f"**🛠️ Tool Result** (id: `{tool_id}`):")
                md_parts.append(f"```\n{msg.get('content', '')}\n```")
            else:
                md_parts.append(f"**❓ Unknown Message Type** ({msg_type}): {str(msg)[:100]}...")
        
        # Unknown message format
        else:
            md_parts.append(f"**❓ Unknown Message Format**: {str(type(msg))} - {str(msg)[:100]}...")
        
        # Add separator between messages (except for last one)
        if i < len(iteration_messages) - 1:
            md_parts.append("")
    
    return "\n".join(md_parts)


def format_all_iterations_to_markdown(history_obj: Any) -> str:
    """
    Format all iterations from a HEAVEN History object into markdown.
    
    Args:
        history_obj: HEAVEN History object with iterations property
        
    Returns:
        Complete markdown string with all iterations formatted
    """
    if not hasattr(history_obj, 'iterations'):
        return "❌ No iterations found in history object"
    
    iterations = history_obj.iterations
    if not iterations:
        return "📭 No iterations in history"
    
    md_parts = [
        "# 📚 HEAVEN History Iterations",
        f"Total iterations: {len(iterations)}",
        ""
    ]
    
    # Sort iterations by key to ensure proper order
    sorted_iterations = sorted(iterations.items(), key=lambda x: int(x[0].split('_')[1]))
    
    for iteration_key, iteration_messages in sorted_iterations:
        md_parts.append(format_iteration_to_markdown(iteration_key, iteration_messages))
        md_parts.append("")  # Separator between iterations
    
    return "\n".join(md_parts)


def extract_last_iteration_complete(completion_result: Dict[str, Any]) -> Optional[List[Any]]:
    """
    Extract the complete last iteration from HEAVEN completion result.
    Returns ALL messages/events in the final iteration, not just AI response.
    
    Args:
        completion_result: Result dictionary from completion_runner or hermes_runner
        
    Returns:
        List of all messages/events in the last iteration, or None if not found
    """
    # Step 1: Extract history_id from result
    history_id = extract_history_id_from_result(completion_result)
    if not history_id:
        return None
    
    # Step 2: Load History object by ID
    history = load_history_by_id(history_id)
    if not history:
        return None
    
    # Step 3: Get iterations and return the last one completely
    if not hasattr(history, 'iterations'):
        return None
        
    iterations = history.iterations
    if not iterations:
        return None
    
    # Get the last iteration (highest numbered iteration_N)
    last_iteration_key = max(iterations.keys())
    last_iteration_messages = iterations[last_iteration_key]
    
    return last_iteration_messages


def extract_truncated_actions_with_response(completion_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract final AI response with truncated action summaries from the last iteration.
    Shows what actions were taken without full content details.
    
    Args:
        completion_result: Result dictionary from completion_runner or hermes_runner
        
    Returns:
        Dictionary with:
        {
            "action_summary": ["ToolUse: WriteBlockReportTool was used", ...],
            "final_response": "Based on my analysis...",
            "has_actions": True/False
        }
    """
    # Step 1: Get the complete last iteration
    last_iteration = extract_last_iteration_complete(completion_result)
    if not last_iteration:
        return {
            "action_summary": [],
            "final_response": "No response",
            "has_actions": False
        }
    
    action_summary = []
    final_response = "No response"
    
    # Step 2: Process each message in the iteration
    for msg in last_iteration:
        if hasattr(msg, '__class__'):
            msg_type = msg.__class__.__name__
            
            if msg_type == 'SystemMessage':
                action_summary.append("System: Provided context")
                
            elif msg_type == 'HumanMessage':
                content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                action_summary.append(f"User: {content_preview}")
                
            elif msg_type == 'AIMessage':
                # Check for tool calls first
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown_tool')
                        action_summary.append(f"ToolUse: {tool_name} was used")
                elif hasattr(msg, 'additional_kwargs') and msg.additional_kwargs.get('tool_calls'):
                    for tool_call in msg.additional_kwargs['tool_calls']:
                        tool_name = tool_call.get('function', {}).get('name', 'unknown_tool')
                        action_summary.append(f"ToolUse: {tool_name} was used")
                
                # Capture the final AI response (last AIMessage with content)
                if msg.content and msg.content.strip():
                    final_response = msg.content
                    
            elif msg_type == 'ToolMessage':
                tool_id = getattr(msg, 'tool_call_id', 'unknown')
                content_length = len(str(msg.content)) if msg.content else 0
                action_summary.append(f"ToolResult: Response received ({content_length} chars)")
        
        # Handle ADK Events
        elif hasattr(msg, 'author') and hasattr(msg, 'content'):
            author = getattr(msg, 'author', 'unknown')
            if author == 'user':
                content_preview = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
                action_summary.append(f"User: {content_preview}")
            else:
                action_summary.append(f"Agent: Responded via ADK")
    
    return {
        "action_summary": action_summary,
        "final_response": final_response,
        "has_actions": len(action_summary) > 0
    }


def format_truncated_actions_display(truncated_data: Dict[str, Any]) -> str:
    """
    Format truncated actions data into readable display format.
    
    Args:
        truncated_data: Result from extract_truncated_actions_with_response()
        
    Returns:
        Formatted string for display
    """
    parts = []
    
    # Add action summary if there are actions
    if truncated_data["has_actions"] and truncated_data["action_summary"]:
        parts.append("## 📋 Action Summary")
        for i, action in enumerate(truncated_data["action_summary"], 1):
            parts.append(f"{i}. **{action}**")
        parts.append("")  # Blank line
    
    # Add final response
    parts.append("## 🤖 Final Response")
    parts.append(truncated_data["final_response"])
    
    return "\n".join(parts)


def extract_and_format_iterations(completion_result: Dict[str, Any]) -> str:
    """
    Extract iterations from HEAVEN completion result and format as markdown.
    Loads History object by ID and computes iterations.
    
    Args:
        completion_result: Result dictionary from completion_runner or hermes_runner
        
    Returns:
        Formatted markdown string of all iterations, or message if none found
    """
    # Step 1: Extract history_id from result
    history_id = extract_history_id_from_result(completion_result)
    if not history_id:
        return "❌ No history_id found in result"
    
    # Step 2: Load History object by ID
    history = load_history_by_id(history_id)
    if not history:
        return "❌ Could not load history object"
    
    # Step 3: Format iterations from History object
    return format_all_iterations_to_markdown(history)


def debug_heaven_result_structure(completion_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Debug utility to inspect the structure of a HEAVEN completion result.
    
    Args:
        completion_result: Result dictionary from completion_runner
        
    Returns:
        Dictionary with debug information about the result structure
    """
    debug_info = {
        "result_type": type(completion_result).__name__,
        "has_results_key": "results" in completion_result if isinstance(completion_result, dict) else False,
        "results_count": 0,
        "raw_result_type": None,
        "has_iterations": False,
        "iteration_count": 0,
        "has_messages": False,
        "message_count": 0
    }
    
    if isinstance(completion_result, dict) and "results" in completion_result:
        results_list = completion_result["results"]
        debug_info["results_count"] = len(results_list)
        
        if results_list:
            last_result = results_list[-1]
            if "raw_result" in last_result:
                raw_result = last_result["raw_result"]
                debug_info["raw_result_type"] = type(raw_result).__name__
                
                # Check for iterations
                if hasattr(raw_result, 'iterations'):
                    debug_info["has_iterations"] = True
                    iterations = raw_result.iterations
                    debug_info["iteration_count"] = len(iterations)
                    debug_info["iteration_keys"] = list(iterations.keys())
                
                # Check for legacy messages
                if isinstance(raw_result, dict) and "messages" in raw_result:
                    debug_info["has_messages"] = True
                    debug_info["message_count"] = len(raw_result["messages"])
    
    return debug_info