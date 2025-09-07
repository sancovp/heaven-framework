#!/usr/bin/env python3
"""
Minimal HEAVEN CLI Chat Interface
- Just chat and interrupt (Ctrl+C)
- Edit this file directly to change agent config, tools, etc.
"""

import asyncio
import os
import sys
import logging
import traceback
from typing import List, Dict, Any
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
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum
)
from heaven_base.memory.history import History
from heaven_base.memory.heaven_event import HeavenEvent

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
# EVENT RENDERING
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

def render_events(messages: List[Any]) -> List[str]:
    """Convert messages to rendered event strings."""
    rendered = []
    
    for msg in messages:
        # Convert to HEAVEN events
        events = HeavenEvent.from_langchain_message(msg)
        
        for event in events:
            if SHOW_RAW_EVENTS:
                # Show raw event data for debugging
                rendered.append(f"\n[RAW EVENT]: {event.to_dict()}")
            
            # Always show clean rendered version
            rendered_str = render_event(event)
            if rendered_str.strip():  # Only add non-empty renders
                rendered.append(rendered_str)
    
    return rendered

# ============================================================
# MAIN CLI LOOP
# ============================================================

async def run_cli():
    """Main CLI chat loop."""
    print("=" * 60)
    print("HEAVEN CLI Chat Interface")
    print("=" * 60)
    print("Type your message and press Enter to send")
    print("Press Ctrl+C to interrupt/exit")
    print("Edit this file to change agent configuration")
    print("=" * 60)
    
    # Initialize agent
    logger.info("Initializing agent...")
    history = History(messages=[], history_id=None)
    
    # Check if API key is set
    import os
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if api_key:
        logger.info(f"OPENAI_API_KEY is set (length: {len(api_key)})")
    else:
        logger.warning("OPENAI_API_KEY is not set!")
    
    # Pass UnifiedChat class, not instance
    agent = BaseHeavenAgent(AGENT_CONFIG, UnifiedChat, history=history, adk=False)
    logger.info("Agent initialized successfully")
    
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
            logger.info(f"Sending prompt to agent: {user_input[:100]}...")
            
            try:
                result = await agent.run(prompt=user_input)
                logger.info(f"Got result type: {type(result)}")
                logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            except Exception as e:
                logger.error(f"Agent.run failed: {e}", exc_info=True)
                raise
            
            # Extract and render events from history
            if isinstance(result, dict) and "history" in result:
                # Get the new messages from this turn
                all_messages = result["history"].messages
                
                # Find where the last user message starts
                last_user_idx = -1
                for i in range(len(all_messages) - 1, -1, -1):
                    if hasattr(all_messages[i], '__class__') and \
                       all_messages[i].__class__.__name__ == "HumanMessage":
                        last_user_idx = i
                        break
                
                # Render messages from this turn
                if last_user_idx >= 0:
                    turn_messages = all_messages[last_user_idx:]
                    rendered = render_events(turn_messages)
                    
                    # Print rendered events
                    for rendered_event in rendered:
                        print(rendered_event)
                
                # Update history for next turn
                history = result["history"]
                agent.history = history
            
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