#!/usr/bin/env python3
"""
HEAVEN Agent with Tools Example
Shows how to create an agent with access to tools like NetworkEditTool and BashTool
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum
)
from heaven_base.memory.history import History
from heaven_base.tools import NetworkEditTool, BashTool

async def main():
    # Create agent configuration with tools
    config = HeavenAgentConfig(
        name="DeveloperAgent",
        system_prompt="""You are a helpful developer assistant with file and system access.
You can read, write, and edit files using NetworkEditTool.
You can execute bash commands using BashTool.
Always be careful with file operations and confirm your actions.""",
        tools=[NetworkEditTool, BashTool],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Run the agent with a task that requires tools
    prompt = """Create a simple Python script at /tmp/hello_heaven.py that prints:
- 'Hello from HEAVEN framework!'
- The current date and time
Then execute the script and show me the output."""
    
    print(f"User: {prompt}\n")
    
    result = await agent.run(prompt=prompt)
    
    # Display the response
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"Assistant: {msg.content}")
    
    # Show the history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())