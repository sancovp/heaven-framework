#!/usr/bin/env python3
"""
HEAVEN Agent with Extraction Example
Demonstrates keyword extraction from agent responses
"""

import asyncio
from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum,
    History
)
from heaven_base.tools import ThinkTool

async def main():
    config = HeavenAgentConfig(
        name="IdeaExtractorAgent",
        system_prompt="""You are an idea generation agent. 
When given a topic, generate creative ideas and insights.

Output your results in these exact fenced sections:

```main_idea
<Your primary creative idea>
```

```insights
<Key insights and observations>
```

```next_steps
<Recommended next steps>
```""",
        tools=[ThinkTool],
        provider=ProviderEnum.OPENAI,
        model="gpt-4-turbo-preview",
        temperature=0.8,
        additional_kws=["main_idea", "insights", "next_steps"],
        additional_kw_instructions="""main_idea: The primary creative idea
insights: Key insights and observations
next_steps: Recommended next steps"""
    )
    
    chat = UnifiedChat()
    history = History(messages=[])
    
    agent = BaseHeavenAgent(config, chat, history=history)
    
    prompt = "Generate ideas for sustainable urban transportation"
    
    result = await agent.run(prompt=f"agent goal={prompt}, iterations=2")
    
    if agent.history and agent.history.agent_status:
        extracts = agent.history.agent_status.extracted_content or {}
        
        print("=== Extracted Content ===")
        print(f"\nMain Idea:\n{extracts.get('main_idea', 'Not extracted')}")
        print(f"\nInsights:\n{extracts.get('insights', 'Not extracted')}")
        print(f"\nNext Steps:\n{extracts.get('next_steps', 'Not extracted')}")
    
    print(f"\nHistory ID: {result.get('history_id')}")

if __name__ == "__main__":
    asyncio.run(main())