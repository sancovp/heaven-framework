#!/usr/bin/env python3
"""
Universal utility for reading agent results in a standardized way.

This handles the common pattern of extracting readable content from 
BaseHeavenAgent.run() results without manually reconstructing LangChain functionality.
"""

from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage


def read_agent_result(result: Any) -> Dict[str, Any]:
    """
    Universal reader for agent results that prints messages and returns structured data.
    
    Args:
        result: Result from BaseHeavenAgent.run()
        
    Returns:
        Dict with 'messages', 'last_message_content', 'success' keys
        
    Raises:
        ValueError: If result format is unexpected
    """
    if isinstance(result, dict) and "history" in result:
        messages = []
        last_content = ""
        
        for msg in result["history"].messages:
            message_info = {
                "type": msg.__class__.__name__,
                "content": msg.content
            }
            messages.append(message_info)
            print(f"{msg.__class__.__name__}: {msg.content}")
            
            # Keep track of last message content for easy access
            last_content = str(msg.content)
        
        return {
            "messages": messages,
            "last_message_content": last_content,
            "success": True,
            "raw_result": result
        }
    else:
        print(f"Unexpected result format: {result}")
        raise ValueError("Agent result did not contain expected history format")


def extract_last_ai_message(result: Any) -> str:
    """
    Extract just the last AI message content from agent result.
    
    Args:
        result: Result from BaseHeavenAgent.run()
        
    Returns:
        Content of the last AI message as string
    """
    if isinstance(result, dict) and "history" in result:
        # Find the last AI message
        for msg in reversed(result["history"].messages):
            if msg.__class__.__name__ == "AIMessage":
                return str(msg.content)
        return ""
    else:
        raise ValueError("Agent result did not contain expected history format")


def extract_boolean_from_agent_result(result: Any) -> bool:
    """
    Extract boolean value from agent result for validation tasks.
    
    Useful for .check() interfaces that need True/False responses.
    
    Args:
        result: Result from BaseHeavenAgent.run()
        
    Returns:
        Boolean interpretation of the agent's response
    """
    last_content = extract_last_ai_message(result).lower()
    return any(word in last_content for word in ["true", "yes", "valid", "sufficient", "ready"])


def print_agent_conversation(result: Any) -> None:
    """
    Print a readable conversation from agent result.
    
    Just prints the conversation without returning structured data.
    
    Args:
        result: Result from BaseHeavenAgent.run()
    """
    if isinstance(result, dict) and "history" in result:
        print("=== Agent Conversation ===")
        for msg in result["history"].messages:
            print(f"{msg.__class__.__name__}: {msg.content}")
        print("=== End Conversation ===")
    else:
        print(f"Unexpected result format: {result}")