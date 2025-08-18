#!/usr/bin/env python3
"""
HEAVEN Completion Runner Example
Shows how to use the completion style runner from the framework
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    HeavenAgentConfig,
    ProviderEnum
)
from heaven_base.tool_utils.completion_runners import exec_completion_style

async def main():
    # Create a simple agent config for the completion runner
    config = HeavenAgentConfig(
        name="CompletionAgent",
        system_prompt="You are a helpful assistant that provides clear, concise responses.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Use the completion style runner
    prompt = "Explain what HEAVEN framework is in 2-3 sentences."
    print(f"Prompt: {prompt}\n")
    
    # Run completion using the completion style runner
    result = await exec_completion_style(
        prompt=prompt,
        agent=config
    )
    
    print("=== COMPLETION STYLE RESULT ===")
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())