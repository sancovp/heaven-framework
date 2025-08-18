#!/usr/bin/env python3
"""
HEAVEN Extraction Patterns Example
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
from heaven_base.tools import ThinkTool

async def main():
    # Create agent configuration with extraction keywords
    config = HeavenAgentConfig(
        name="AnalysisAgent",
        system_prompt="""You are an analysis agent that provides structured insights.

When analyzing a topic, provide your output in these exact fenced sections:

```summary
<Brief summary of the topic>
```

```key_points
<List of 3-5 key points>
```

```recommendations
<3-4 actionable recommendations>
```

```confidence_score
<A number from 1-10 indicating your confidence in this analysis>
```

Always use these exact section names with triple backticks.""",
        tools=[ThinkTool],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7,
        # These keywords tell HEAVEN what to extract from responses
        additional_kws=["summary", "key_points", "recommendations", "confidence_score"],
        additional_kw_instructions="""summary: Brief summary of the analysis
key_points: List of key points identified
recommendations: Actionable recommendations
confidence_score: Numerical confidence rating"""
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Run the agent with a topic to analyze
    prompt = "Analyze the benefits and challenges of remote work in 2025"
    print(f"User: {prompt}\n")
    
    # Use agent mode for iterative analysis
    result = await agent.run(prompt=f"agent goal={prompt}, iterations=2")
    
    # Extract the structured data that was parsed automatically
    if agent.history and agent.history.agent_status:
        extracts = agent.history.agent_status.extracted_content or {}
        
        print("=== EXTRACTED CONTENT ===")
        print(f"\nSummary:\n{extracts.get('summary', 'Not extracted')}")
        print(f"\nKey Points:\n{extracts.get('key_points', 'Not extracted')}")
        print(f"\nRecommendations:\n{extracts.get('recommendations', 'Not extracted')}")
        print(f"\nConfidence Score:\n{extracts.get('confidence_score', 'Not extracted')}")
    else:
        print("No extracted content found")
    
    # Show the history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())