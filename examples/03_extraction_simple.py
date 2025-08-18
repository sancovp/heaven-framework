#!/usr/bin/env python3
"""
HEAVEN Extraction Patterns Example - Simple Version
Shows how to extract structured data from agent responses using keywords
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
    # Create agent configuration with simple system prompt
    config = HeavenAgentConfig(
        name="ExtractionAgent",
        system_prompt="You are a helpful analyst that provides structured insights.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3,
        additional_kws=["summary", "key_points", "recommendations", "confidence"],
        additional_kw_instructions="""summary: Brief summary
key_points: Key points identified
recommendations: Actionable recommendations  
confidence: Numerical confidence rating"""
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Use agent mode with proper goal formatting
    goal = "Analyze remote work trends and provide summary, key_points, recommendations, and confidence in fenced code blocks"
    prompt = f'agent goal="{goal}", iterations=3'
    print(f"User: {prompt}\n")
    
    result = await agent.run(prompt=prompt)
    
    # Extract the structured data
    if agent.history and agent.history.agent_status:
        extracts = agent.history.agent_status.extracted_content or {}
        
        print("=== EXTRACTED CONTENT ===")
        print(f"\nSummary:\n{extracts.get('summary', 'Not extracted')}")
        print(f"\nKey Points:\n{extracts.get('key_points', 'Not extracted')}")  
        print(f"\nRecommendations:\n{extracts.get('recommendations', 'Not extracted')}")
        print(f"\nConfidence:\n{extracts.get('confidence', 'Not extracted')}")
    else:
        print("No agent_status found")
    
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())