#!/usr/bin/env python3
"""
Debug the ACTUAL failing request being sent to uni-api.
"""

import sys
sys.path.insert(0, '/home/GOD/heaven-base')
import asyncio
import json
from heaven_base.utils.auto_summarize import AggregationSummarizerAgent

# Monkey patch the invoke_uni_api to capture the EXACT failing request
original_invoke_uni_api = None

def debug_invoke_uni_api(self, model, uni_messages, uni_api_url=None, **kwargs):
    global original_invoke_uni_api
    
    request_count = getattr(debug_invoke_uni_api, 'request_count', 0) + 1
    debug_invoke_uni_api.request_count = request_count
    
    print(f"\n=== LIVE REQUEST #{request_count} ===")
    print(f"Model: {model}")
    print(f"Messages count: {len(uni_messages)}")
    
    # Show exact conversation structure
    for i, msg in enumerate(uni_messages):
        role = msg.get('role', 'unknown')
        content_len = len(str(msg.get('content', ''))) if msg.get('content') else 0
        
        if role == 'assistant' and msg.get('tool_calls'):
            tool_call_ids = [tc['id'] for tc in msg['tool_calls']]
            print(f"  Message {i}: {role} - {content_len} chars, tool_calls: {tool_call_ids}")
        elif role == 'tool':
            print(f"  Message {i}: {role} - {content_len} chars, tool_call_id: {msg.get('tool_call_id')}")
        else:
            print(f"  Message {i}: {role} - {content_len} chars")
    
    # Look for the problematic pattern
    assistant_with_tools = None
    tool_responses = []
    
    for i, msg in enumerate(uni_messages):
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            assistant_with_tools = msg
            expected_ids = [tc['id'] for tc in msg['tool_calls']]
            print(f"  🔍 Found assistant with tool_calls at index {i}, expecting responses for: {expected_ids}")
            
            # Look for corresponding tool responses
            for j in range(i+1, len(uni_messages)):
                next_msg = uni_messages[j]
                if next_msg.get('role') == 'tool':
                    tool_responses.append(next_msg['tool_call_id'])
                elif next_msg.get('role') == 'assistant':
                    break  # Stop at next assistant message
            
            print(f"  🔍 Found tool responses: {tool_responses}")
            
            missing = set(expected_ids) - set(tool_responses)
            if missing:
                print(f"  ❌ MISSING TOOL RESPONSES: {missing}")
            else:
                print(f"  ✅ All tool responses present")
            break
    
    try:
        # Call original function
        result = original_invoke_uni_api(model, uni_messages, uni_api_url, **kwargs)
        print(f"  ✅ Request #{request_count} SUCCESS")
        return result
    except Exception as e:
        print(f"  ❌ Request #{request_count} FAILED: {e}")
        
        # Save the ACTUAL failing request
        failing_payload = {
            "model": model,
            "messages": uni_messages,
            "kwargs": kwargs
        }
        
        filename = f'/home/GOD/heaven-base/actual_failing_request_{request_count}.json'
        with open(filename, 'w') as f:
            json.dump(failing_payload, f, indent=2, default=str)
        
        print(f"  💾 Saved ACTUAL failing payload to {filename}")
        raise

async def debug_live_400_error():
    print("=== Debugging LIVE 400 Error ===")
    
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
        
        print("Running AggregationSummarizerAgent with LIVE monitoring...")
        result = await agent.run(prompt)
        
        print(f"✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        # Restore original function
        if original_invoke_uni_api:
            UnifiedChat.invoke_uni_api = original_invoke_uni_api

if __name__ == "__main__":
    asyncio.run(debug_live_400_error())