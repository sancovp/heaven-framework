#!/usr/bin/env python3
"""
HEAVEN Hermes Runner Example
Shows how to use the hermes utility functions from the framework
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    HeavenAgentConfig,
    ProviderEnum
)
from heaven_base.tool_utils.hermes_utils import use_hermes_dict

async def main():
    # Create an agent config 
    config = HeavenAgentConfig(
        name="HermesTestAgent",
        system_prompt="You are a helpful assistant. Provide structured responses.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Use hermes with local execution
    goal = "Describe the key benefits of modular software architecture"
    print(f"Goal: {goal}\n")
    
    try:
        # Use hermes_dict for structured result
        result = await use_hermes_dict(
            goal=goal,
            iterations=2,
            agent=config,
            target_container="mind_of_god",
            source_container="mind_of_god",
            return_summary=False,
            ai_messages_only=True
        )
        
        print("=== HERMES RUNNER RESULT ===")
        print(f"Result type: {type(result)}")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"History ID: {result.get('history_id', 'No history ID')}")
        
        if 'messages' in result:
            print(f"Messages count: {len(result['messages'])}")
            # Show the final AI response
            for msg in result['messages']:
                if msg.get('type') == 'AIMessage':
                    content = msg.get('content', '')[:200]  # First 200 chars
                    print(f"AI Response preview: {content}...")
    
    except Exception as e:
        print(f"Error using hermes: {e}")
        print("Note: This may require Docker container setup for cross-container execution")

if __name__ == "__main__":
    asyncio.run(main())