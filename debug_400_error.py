#!/usr/bin/env python3
"""
Debug the 400 error by capturing the exact request that fails.
"""

import sys
sys.path.insert(0, '/home/GOD/heaven-base')
import asyncio
import json
from heaven_base.utils.auto_summarize import AggregationSummarizerAgent

# Monkey patch the invoke_uni_api to capture failing requests
original_invoke_uni_api = None

def debug_invoke_uni_api(self, model, uni_messages, uni_api_url=None, **kwargs):
    global original_invoke_uni_api
    
    print(f"\n=== DEBUG: Request #{getattr(debug_invoke_uni_api, 'request_count', 0) + 1} ===")
    debug_invoke_uni_api.request_count = getattr(debug_invoke_uni_api, 'request_count', 0) + 1
    
    print(f"Model: {model}")
    print(f"Messages count: {len(uni_messages)}")
    print(f"Kwargs: {kwargs}")
    
    # Show message structure
    for i, msg in enumerate(uni_messages):
        print(f"Message {i}: {msg.get('role', 'unknown')} - {len(str(msg.get('content', '')))} chars")
        if 'tool_calls' in msg:
            print(f"  Has tool_calls: {len(msg['tool_calls'])}")
        if msg.get('role') == 'tool':
            print(f"  Tool content preview: {str(msg.get('content', ''))[:100]}...")
    
    # Check for problematic patterns
    total_content_length = sum(len(str(msg.get('content', ''))) for msg in uni_messages)
    print(f"Total content length: {total_content_length}")
    
    if total_content_length > 100000:
        print("⚠️ LARGE PAYLOAD - might be causing 400")
    
    # Check for malformed messages
    for i, msg in enumerate(uni_messages):
        if not isinstance(msg.get('content'), (str, list)):
            print(f"⚠️ MALFORMED MESSAGE {i}: content is {type(msg.get('content'))}")
        if msg.get('role') not in ['system', 'user', 'assistant', 'tool']:
            print(f"⚠️ INVALID ROLE {i}: {msg.get('role')}")
    
    try:
        # Call original function
        result = original_invoke_uni_api(model, uni_messages, uni_api_url, **kwargs)
        print(f"✅ Request #{debug_invoke_uni_api.request_count} SUCCESS")
        return result
    except Exception as e:
        print(f"❌ Request #{debug_invoke_uni_api.request_count} FAILED: {e}")
        
        # Save the failing payload for analysis
        failing_payload = {
            "model": model,
            "messages": uni_messages,
            "kwargs": kwargs
        }
        
        with open(f'/home/GOD/heaven-base/failing_request_{debug_invoke_uni_api.request_count}.json', 'w') as f:
            json.dump(failing_payload, f, indent=2, default=str)
        
        print(f"💾 Saved failing payload to failing_request_{debug_invoke_uni_api.request_count}.json")
        raise

async def debug_400_error():
    print("=== Debugging 400 Error ===")
    
    # Monkey patch the invoke_uni_api method
    global original_invoke_uni_api
    from heaven_base.unified_chat import UnifiedChat
    original_invoke_uni_api = UnifiedChat.invoke_uni_api
    UnifiedChat.invoke_uni_api = debug_invoke_uni_api
    
    try:
        agent = AggregationSummarizerAgent()
        
        # Use the exact same data that was failing
        combined_summaries = '''## Iteration 1 Summary

**Actions Taken:**
Analyzed the content of the conversation to identify key actions, outcomes, and challenges.

**Outcomes:**
Extracted relevant information for summarization.

**Challenges:**
Ensuring accurate extraction of details from the conversation.

**Tools Used:**
IterationSummarizerTool

## Iteration 2 Summary

**Actions Taken:**
Identified key points for summarization and assigned tasks to team members.

**Outcomes:**
Team members are clear on their responsibilities and deadlines.

**Challenges:**
Addressing concerns about tight deadlines.

**Tools Used:**
IterationSummarizerTool, project management software

## Iteration 3 Summary

**Actions Taken:**
Conducted a meeting to discuss project expectations and deliverables.

**Outcomes:**
Positive feedback on the project and a shared document for tracking progress.

**Challenges:**
Need for clarity on deliverables and timelines.

**Tools Used:**
Communication tools, document sharing platforms'''
        
        goal = f"Create an aggregated summary using the AggregationSummaryTool from these iteration summaries: {combined_summaries}"
        prompt = f'agent goal="{goal}", iterations=5'
        
        print("Running AggregationSummarizerAgent with debug monitoring...")
        result = await agent.run(prompt)
        
        print(f"✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Restore original function
        if original_invoke_uni_api:
            UnifiedChat.invoke_uni_api = original_invoke_uni_api

if __name__ == "__main__":
    asyncio.run(debug_400_error())