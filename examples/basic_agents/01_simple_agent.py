#!/usr/bin/env python3
"""
Basic HEAVEN Agent Example
Demonstrates creating and running a simple agent
"""

import asyncio
from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum,
    History
)

async def main():
    config = HeavenAgentConfig(
        name="SimpleAgent",
        system_prompt="You are a helpful assistant. Be concise and clear.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    chat = UnifiedChat()
    history = History(messages=[])
    
    agent = BaseHeavenAgent(config, chat, history=history)
    
    result = await agent.run(prompt="What is HEAVEN framework?")
    
    print("Agent Response:")
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content'):
                print(f"{msg.__class__.__name__}: {msg.content}")
    
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())