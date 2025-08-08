#!/usr/bin/env python3
"""
Debug tool capture in IterationSummarizerAgent.
"""

import asyncio
import sys
sys.path.insert(0, '/home/GOD/heaven-base')

from heaven_base.utils.auto_summarize import IterationSummarizerAgent
from langchain_core.messages import AIMessage, ToolMessage


async def debug_tool_capture():
    """Debug what's happening with tool capture."""
    print("=== Debugging Tool Capture ===")
    
    agent = IterationSummarizerAgent()
    
    goal = """Summarize iteration 1 using the IterationSummarizerTool:
    - Actions taken: Created token counter utility
    - Outcomes: Working token counter
    - Challenges: Model type handling  
    - Tools used: tiktoken library"""
    
    prompt = f'agent goal="{goal}", iterations=3'
    
    print("Running IterationSummarizerAgent...")
    result = await agent.run(prompt)
    
    print(f"\nAgent completed. History has {len(agent.history.messages)} messages")
    
    # Debug message structure
    for i, msg in enumerate(agent.history.messages):
        print(f"\nMessage {i}: {type(msg).__name__}")
        if isinstance(msg, AIMessage):
            print(f"  Content type: {type(msg.content)}")
            if isinstance(msg.content, list):
                print(f"  Content items: {len(msg.content)}")
                for j, item in enumerate(msg.content):
                    print(f"    Item {j}: {type(item)} - {item}")
            else:
                print(f"  Content preview: {str(msg.content)[:100]}...")
        elif isinstance(msg, ToolMessage):
            print(f"  Tool content: {msg.content[:100]}...")
    
    # Try manual tool capture
    print(f"\n=== Manual Tool Capture ===")
    found_tools = []
    
    for i, msg in enumerate(agent.history.messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        print(f"Found tool_use: {item.get('name')}")
                        found_tools.append((i, item))
                        
                        # Check next message for result
                        if i + 1 < len(agent.history.messages):
                            next_msg = agent.history.messages[i + 1]
                            print(f"  Next message: {type(next_msg).__name__}")
                            if isinstance(next_msg, ToolMessage):
                                print(f"  Tool result: {next_msg.content[:200]}...")
    
    print(f"\nFound {len(found_tools)} tool uses")
    
    # Run the original capture method
    print(f"\n=== Original Capture Method ===")
    agent.look_for_particular_tool_calls()
    print(f"agent.last_summary: {agent.last_summary}")
    
    return agent


if __name__ == "__main__":
    asyncio.run(debug_tool_capture())