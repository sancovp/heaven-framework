#!/usr/bin/env python3
"""
Simple Hermes Agent - Using only heaven-base built-in functionality
"""

import asyncio
import sys
sys.path.insert(0, '/home/GOD/heaven-base')

from heaven_base import HeavenAgentConfig, ProviderEnum
from heaven_base.langgraph import HeavenState, hermes_runner

# Create a simple agent configuration using only built-in tools
def create_simple_agent() -> HeavenAgentConfig:
    return HeavenAgentConfig(
        name="SimpleHermesAgent",
        system_prompt="""You are a helpful assistant. Answer questions clearly and concisely.""",
        model="gpt-4o-mini",
        temperature=0.7
    )

async def test_simple_hermes_agent():
    """Test the simple hermes agent"""
    
    print("🤖 Testing Simple Hermes Agent")
    print("=" * 50)
    
    try:
        # Create the agent
        simple_agent = create_simple_agent()
        print("✅ Created simple agent configuration")
        
        # Test questions
        test_questions = [
            "What is the capital of France?",
            "Explain what gravity is in simple terms",
            "Write a haiku about programming",
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Test {i}: {question}")
            print("-" * 40)
            
            try:
                # Set up initial state
                initial_state = {
                    "results": [],
                    "context": {},
                    "agents": {}
                }
                
                # Use hermes_runner directly
                result = await hermes_runner(
                    initial_state,
                    agent=simple_agent,
                    goal=question,
                    iterations=1
                )
                
                print(f"✅ Hermes execution completed")
                
                # Show results
                if result.get("results"):
                    last_result = result["results"][-1]
                    print(f"📊 Agent: {last_result.get('agent_name', 'Unknown')}")
                    print(f"🎯 Goal: {question}")
                    
                    # Extract the response
                    raw_result = last_result.get("raw_result", {})
                    if isinstance(raw_result, dict):
                        if "result" in raw_result:
                            print(f"💬 Response: {raw_result['result']}")
                        elif "messages" in raw_result:
                            # Look for the final AI message
                            messages = raw_result["messages"]
                            for msg in reversed(messages):
                                if hasattr(msg, 'content') and msg.content:
                                    print(f"💬 Response: {msg.content}")
                                    break
                        else:
                            print(f"💬 Raw result keys: {list(raw_result.keys())}")
                    else:
                        print(f"💬 Raw result: {raw_result}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the simple hermes agent test"""
    print("🧪 Testing Simple Hermes Agent with heaven-base")
    print()
    
    success = await test_simple_hermes_agent()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Simple Hermes Agent is working!")
    else:
        print("❌ FAILED: Simple Hermes Agent has issues")

if __name__ == "__main__":
    asyncio.run(main())