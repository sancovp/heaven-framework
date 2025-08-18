#!/usr/bin/env python3
"""
Basic HEAVEN Agent Example
Shows how to create and run a simple agent
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

async def main():
    # Create agent configuration
    config = HeavenAgentConfig(
        name="BasicAssistant",
        system_prompt="You are a helpful AI assistant. Be concise and clear in your responses.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Initialize components
    
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Run the agent with a simple prompt
    prompt = "What are the key components of the HEAVEN framework?"
    print(f"User: {prompt}\n")
    
    result = await agent.run(prompt=prompt)
    
    # Display the response
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"Assistant: {msg.content}")
    
    # Show the history ID for continuity
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")
    print("\nYou can use this history_id to continue the conversation later.")

if __name__ == "__main__":
    asyncio.run(main())