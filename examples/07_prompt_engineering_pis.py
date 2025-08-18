#!/usr/bin/env python3
"""
HEAVEN Prompt Engineering with PIS Example
Shows how to use the Prompt Injection System (PIS) for advanced prompt engineering
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
from heaven_base.tool_utils.prompt_injection_system_vX1 import (
    PromptInjectionSystemVX1,
    PromptInjectionSystemConfigVX1,
    PromptStepDefinitionVX1,
    PromptBlockDefinitionVX1,
    BlockTypeVX1
)

async def main():
    # Create a base agent config for PIS reference resolution
    base_config = HeavenAgentConfig(
        name="PISAgent",
        system_prompt="You are a helpful assistant.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Create PIS configuration with enhanced prompts
    pis_config = PromptInjectionSystemConfigVX1(
        steps=[
            PromptStepDefinitionVX1(
                name="enhanced_system_prompt",
                blocks=[
                    PromptBlockDefinitionVX1(
                        type=BlockTypeVX1.FREESTYLE,
                        content="You are a software architecture expert with {experience_level} years of experience. "
                               "Your specialty is {domain}. Always provide {response_style} responses."
                    ),
                    PromptBlockDefinitionVX1(
                        type=BlockTypeVX1.FREESTYLE,
                        content="\n\nGuidelines:\n- Focus on scalability and maintainability\n"
                               "- Provide concrete examples\n- Structure responses clearly"
                    )
                ]
            ),
            PromptStepDefinitionVX1(
                name="user_prompt_enhancement",
                blocks=[
                    PromptBlockDefinitionVX1(
                        type=BlockTypeVX1.FREESTYLE,
                        content="Context: You are helping with {task_type}. "
                               "Please provide detailed guidance on: {user_request}"
                    )
                ]
            )
        ],
        template_vars={
            "experience_level": "10+",
            "domain": "microservices architecture",
            "response_style": "detailed and practical",
            "task_type": "e-commerce system design",
            "user_request": "designing a scalable microservices architecture"
        },
        agent_config=base_config
    )
    
    # Initialize PIS
    pis = PromptInjectionSystemVX1(pis_config)
    
    print("=== HEAVEN PIS Example ===\n")
    
    # Generate enhanced prompts using PIS
    enhanced_system_prompt = pis.get_next_prompt()
    enhanced_user_prompt = pis.get_next_prompt()
    
    print("=== Enhanced System Prompt ===")
    print(f"{enhanced_system_prompt}\n")
    
    print("=== Enhanced User Prompt ===")
    print(f"{enhanced_user_prompt}\n")
    
    # Create agent with enhanced system prompt
    enhanced_config = HeavenAgentConfig(
        name="EnhancedArchitectAgent",
        system_prompt=enhanced_system_prompt,
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the enhanced agent
    agent = BaseHeavenAgent(enhanced_config, UnifiedChat, history=history)
    
    # Test the PIS-enhanced agent
    result = await agent.run(prompt=enhanced_user_prompt)
    
    # Display the response
    print("=== PIS-Enhanced Agent Response ===")
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"Assistant: {msg.content}")
    
    # Show the history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")
    
    # Demonstrate different PIS configurations
    print("\n=== Demonstrating Different PIS Template Variables ===")
    
    # Reset PIS and try different variables
    pis.reset_sequence()
    pis.config.template_vars.update({
        "experience_level": "5+",
        "domain": "frontend development",
        "response_style": "concise and actionable"
    })
    
    different_system_prompt = pis.get_next_prompt()
    print(f"Different enhancement: {different_system_prompt[:150]}...")

if __name__ == "__main__":
    asyncio.run(main())