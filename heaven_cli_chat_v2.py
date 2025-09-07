#!/usr/bin/env python3
"""
Minimal HEAVEN CLI Chat Interface V2
Using proper conversation flow from heaven-tree-repl
"""

import asyncio
import os
import sys
import logging
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

# Set up logging (only if this is run as main script)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Set data directory
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum,
    completion_runner
)
from heaven_base.langgraph.foundation import HeavenState
from heaven_base.memory.history import History
from heaven_base.memory.heaven_event import HeavenEvent
from heaven_base.utils.heaven_response_utils import extract_heaven_response, extract_history_id_from_result

# ============================================================
# CONFIGURATION - Edit these to customize your agent
# ============================================================

AGENT_CONFIG = HeavenAgentConfig(
    name="CLIAssistant",
    system_prompt="""You are a helpful AI assistant in a CLI interface. 
Be concise and clear in your responses.""",
    tools=[],  # Add your tools here
    provider=ProviderEnum.OPENAI,
    model="gpt-5-mini",
    temperature=0.7,
    max_tokens=1000
)

# Set to True to show raw events, False for clean rendering only
SHOW_RAW_EVENTS = False

# ============================================================
# EVENT RENDERING (reused from v1)
# ============================================================

def render_user_message(data: Dict[str, Any]) -> str:
    """Render user message event."""
    return f"\n[USER]: {data.get('content', '')}"

def render_agent_message(data: Dict[str, Any]) -> str:
    """Render agent message event."""
    return f"\n[ASSISTANT]: {data.get('content', '')}"

def render_tool_use(data: Dict[str, Any]) -> str:
    """Render tool use event."""
    tool_name = data.get('name', 'unknown')
    tool_input = data.get('input', {})
    return f"\n[TOOL CALL]: {tool_name}\n  Input: {tool_input}"

def render_tool_result(data: Dict[str, Any]) -> str:
    """Render tool result event."""
    output = data.get('output', '')
    # Truncate long outputs
    if len(output) > 500:
        output = output[:500] + "... (truncated)"
    return f"\n[TOOL RESULT]: {output}"

def render_event(event: HeavenEvent) -> str:
    """Render a single event to a clean string."""
    event_type = event.event_type
    data = event.data
    
    renderers = {
        "USER_MESSAGE": render_user_message,
        "AGENT_MESSAGE": render_agent_message,
        "THINKING": lambda d: f"\n[THINKING]: {d.get('content', '')}",
        "TOOL_USE": render_tool_use,
        "TOOL_RESULT": render_tool_result,
        "SYSTEM_MESSAGE": lambda d: f"\n[SYSTEM]: {d.get('content', '')}"
    }
    
    renderer = renderers.get(event_type)
    if renderer:
        return renderer(data)
    return f"\n[{event_type}]: {data}"

def render_heaven_result(result: Any) -> str:
    """Render the result from completion_runner."""
    try:
        # Extract the response text
        response_text = extract_heaven_response(result)
        
        if response_text:
            return f"\n[ASSISTANT]: {response_text}"
        else:
            logger.warning("No response text extracted from result")
            return "\n[ASSISTANT]: (no response)"
    except Exception as e:
        logger.error(f"Error rendering result: {e}", exc_info=True)
        return f"\n[ERROR rendering response]: {e}"

# ============================================================
# MAIN CLI LOOP - Using proper conversation flow
# ============================================================

async def run_cli():
    """Main CLI chat loop with proper conversation handling."""
    print("=" * 60)
    print("HEAVEN CLI Chat Interface V2")
    print("=" * 60)
    print("Type your message and press Enter to send")
    print("Press Ctrl+C to interrupt/exit")
    print("Edit this file to change agent configuration")
    print("=" * 60)
    
    # Check if API key is set
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        logger.info(f"OPENAI_API_KEY is set (length: {len(api_key)})")
    else:
        logger.warning("OPENAI_API_KEY is not set!")
    
    # Track current conversation
    current_history_id: Optional[str] = None
    
    # Main loop
    while True:
        try:
            # Get user input
            print("\n> ", end="", flush=True)
            user_input = input()
            
            if not user_input.strip():
                continue
            
            # Process with agent
            print("\n[Processing...]", flush=True)
            logger.info(f"Sending prompt: {user_input[:100]}...")
            
            # Create HEAVEN state
            state = HeavenState({
                "results": [],
                "context": {},
                "agents": {}
            })
            
            # Call completion_runner with or without history_id
            try:
                if current_history_id:
                    logger.info(f"Continuing conversation with history_id: {current_history_id}")
                    result = await completion_runner(
                        state,
                        prompt=user_input,
                        agent=AGENT_CONFIG,
                        history_id=current_history_id  # Pass previous history to continue
                    )
                else:
                    logger.info("Starting new conversation")
                    result = await completion_runner(
                        state,
                        prompt=user_input,
                        agent=AGENT_CONFIG
                        # No history_id for first message
                    )
                
                logger.info(f"Got result type: {type(result)}")
                
                # Extract and save the new history_id for next turn
                new_history_id = extract_history_id_from_result(result)
                if new_history_id:
                    logger.info(f"Extracted history_id: {new_history_id}")
                    current_history_id = new_history_id
                else:
                    logger.warning("Failed to extract history_id from result")
                
                # Render the response
                rendered_response = render_heaven_result(result)
                print(rendered_response)
                
            except Exception as e:
                logger.error(f"completion_runner failed: {e}", exc_info=True)
                print(f"\n[ERROR]: {e}")
                print(f"[TRACEBACK]: {traceback.format_exc()}")
                
        except KeyboardInterrupt:
            print("\n\n[Interrupted by user]")
            print("\nExiting... Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            print(f"\n[ERROR]: {e}")
            print(f"[TRACEBACK]: {traceback.format_exc()}")
            print("Continuing...")

def main():
    """Entry point."""
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        logger.info("User interrupted program", exc_info=True)
        print("\nGoodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()