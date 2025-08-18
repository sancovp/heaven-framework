#!/usr/bin/env python3
"""
HEAVEN Agent with Tools Example
Demonstrates creating an agent with tools like NetworkEditTool and BashTool
"""

import asyncio
from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig, 
    UnifiedChat,
    ProviderEnum,
    History
)
from heaven_base.tools import NetworkEditTool, BashTool, CodeLocalizerTool

async def main():
    config = HeavenAgentConfig(
        name="DeveloperAgent",
        system_prompt="""You are a developer assistant with file and system access.
You can read/write files and execute bash commands.
Always check file existence before editing.""",
        tools=[NetworkEditTool, BashTool, CodeLocalizerTool],
        provider=ProviderEnum.ANTHROPIC,
        model="claude-3-haiku-20240307",
        temperature=0.3
    )
    
    chat = UnifiedChat()
    history = History(messages=[])
    
    agent = BaseHeavenAgent(config, chat, history=history)
    
    prompt = """Create a simple Python script at /tmp/hello_heaven.py that:
1. Prints 'Hello from HEAVEN!'
2. Shows the current date
Then run it and show the output."""
    
    result = await agent.run(prompt=prompt)
    
    print("Agent completed task:")
    history_id = result.get("history_id")
    print(f"History ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())