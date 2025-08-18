#!/usr/bin/env python3
"""
HEAVEN Extraction Patterns Example - Working Version
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
    # Create agent configuration with extraction keywords
    config = HeavenAgentConfig(
        name="ExtractionAgent",
        system_prompt="""You are an agent that provides structured outputs.

IMPORTANT: Always end your response with the exact fenced sections below:

```summary
<Your brief summary>
```

```key_points  
<Your 3-5 key points>
```

```recommendations
<Your 2-3 recommendations>
```

```confidence
<A number from 1-10>
```

Use these exact section names. This is mandatory.""",
        tools=[],  # No tools to keep it simple
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3,
        # These keywords tell HEAVEN what to extract from responses
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
    
    # Use a simpler, more direct prompt
    prompt = "Analyze remote work trends. Provide summary, key points, recommendations, and confidence score."
    print(f"User: {prompt}\n")
    
    # Use regular mode, not agent mode for simpler execution
    result = await agent.run(prompt=prompt)
    
    # Check what the agent actually said
    print("=== AGENT RESPONSE ===")
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"Assistant: {msg.content}")
                print("---")
    
    # Extract the structured data that was parsed automatically
    if agent.history and agent.history.agent_status:
        extracts = agent.history.agent_status.extracted_content or {}
        
        print("\n=== EXTRACTED CONTENT ===")
        print(f"\nSummary:\n{extracts.get('summary', 'Not extracted')}")
        print(f"\nKey Points:\n{extracts.get('key_points', 'Not extracted')}")  
        print(f"\nRecommendations:\n{extracts.get('recommendations', 'Not extracted')}")
        print(f"\nConfidence:\n{extracts.get('confidence', 'Not extracted')}")
    else:
        print("No agent_status found")
    
    # Show the history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())