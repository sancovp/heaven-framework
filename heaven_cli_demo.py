#!/usr/bin/env python3
"""
🌟 HEAVEN CLI Chat Demo
A simple, clean CLI interface for chatting with HEAVEN agents.

TO CUSTOMIZE:
1. Edit AGENT_CONFIG below to change the agent's behavior
2. Add your own tools to the tools=[] list
3. Change the model, provider, or system prompt as needed
"""

import asyncio
import os
import sys
import logging
import warnings
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from datetime import datetime

# ============================================================
# LOGGING SETUP - Write to file, not console
# ============================================================

# Create logs directory if it doesn't exist
log_dir = "/tmp/heaven_cli_logs"
os.makedirs(log_dir, exist_ok=True)

# Configure file logging
log_file = os.path.join(log_dir, f"heaven_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

# Set data directory
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

# Load API key from .claude.json if not in environment
if not os.environ.get('OPENAI_API_KEY'):
    try:
        import json
        with open('/home/GOD/.claude.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('OPENAI_API_KEY', '')
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
    except:
        pass  # Silent fail, let the agent handle missing key

from heaven_base import (
    HeavenAgentConfig,
    ProviderEnum,
    completion_runner
)
from heaven_base.langgraph.foundation import HeavenState
from heaven_base.utils.heaven_response_utils import extract_heaven_response, extract_history_id_from_result
from heaven_base.tools import WorkflowRelayTool

# ============================================================
# 🎨 AGENT CONFIGURATION - CUSTOMIZE THIS!
# ============================================================

AGENT_CONFIG = HeavenAgentConfig(
    name="HeavenAssistant",
    system_prompt="""You are a helpful AI assistant with workflow relay capabilities.
Be friendly, concise, and clear in your responses.
Use emojis occasionally to be more engaging.
You have access to WorkflowRelayTool which can help with complex workflows.""",
    tools=[WorkflowRelayTool],  # Example tool to demonstrate functionality
    provider=ProviderEnum.OPENAI,
    model="gpt-5-mini",  # Change to your preferred model
    temperature=0.7,
    max_tokens=1000
)

# ============================================================
# 🎭 DISPLAY FUNCTIONS
# ============================================================

def show_welcome_screen():
    """Display a nice welcome screen."""
    print("\n" + "="*60)
    print("🌟 " + " HEAVEN CLI Chat Interface ".center(56) + "🌟")
    print("="*60)
    print("\n📝 Instructions:")
    print("  • Type your message and press Enter to send")
    print("  • Press Ctrl+C to exit anytime")
    print("  • Your conversation is automatically saved")
    print("\n🤖 Agent: " + AGENT_CONFIG.name)
    print("🧠 Model: " + AGENT_CONFIG.model)
    print("🌡️  Temperature: " + str(AGENT_CONFIG.temperature))
    
    if AGENT_CONFIG.tools:
        print("🛠️  Tools: " + ", ".join([str(t) for t in AGENT_CONFIG.tools]))
    else:
        print("🛠️  Tools: None configured")
    
    print("\n" + "="*60)
    print("💬 Let's start chatting!\n")

def render_user_message(message: str):
    """Display user message with nice formatting."""
    print("\n" + "─"*50)
    print("👤 You:")
    print("   " + message)
    print("─"*50)

def render_assistant_message(message: str):
    """Display assistant message with nice formatting."""
    print("\n🤖 Assistant:")
    # Indent multi-line responses
    lines = message.split('\n')
    for line in lines:
        if line.strip():
            print("   " + line)
    print("\n" + "─"*50)

def render_thinking():
    """Show thinking indicator."""
    print("\n⚡ Thinking...", flush=True)

def render_error(error_msg: str):
    """Display error with formatting."""
    print("\n" + "❌"*20)
    print("❌ ERROR:")
    print("   " + error_msg)
    print("❌"*20 + "\n")

# ============================================================
# 💬 MAIN CHAT LOOP
# ============================================================

async def run_chat():
    """Main chat loop."""
    show_welcome_screen()
    
    # Track conversation state
    current_history_id: Optional[str] = None
    
    while True:
        try:
            # Get user input with nice prompt
            print("\n💭 Your message: ", end="", flush=True)
            user_input = input()
            
            # Skip empty input
            if not user_input.strip():
                continue
            
            # Log the input (to file only)
            logger.info(f"User input: {user_input}")
            
            # Show processing indicator (not real thinking events)
            print("\n⚡ Processing...", flush=True)
            
            # Create HEAVEN state
            state = HeavenState({
                "results": [],
                "context": {},
                "agents": {}
            })
            
            # Process the message (suppress debug output)
            try:
                # Suppress warnings and debug prints from framework
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Capture stdout/stderr to prevent framework debug output
                    captured_stdout = StringIO()
                    captured_stderr = StringIO()
                    
                    with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                        if current_history_id:
                            # Continue existing conversation
                            logger.info(f"Continuing with history_id: {current_history_id}")
                            result = await completion_runner(
                                state,
                                prompt=user_input,
                                agent=AGENT_CONFIG,
                                history_id=current_history_id
                            )
                        else:
                            # Start new conversation
                            logger.info("Starting new conversation")
                            result = await completion_runner(
                                state,
                                prompt=user_input,
                                agent=AGENT_CONFIG
                            )
                    
                    # Log captured output to file only
                    stdout_content = captured_stdout.getvalue()
                    stderr_content = captured_stderr.getvalue()
                    if stdout_content:
                        logger.debug(f"Framework stdout: {stdout_content}")
                    if stderr_content:
                        logger.debug(f"Framework stderr: {stderr_content}")
                
                # Extract response and update history
                response_text = extract_heaven_response(result)
                new_history_id = extract_history_id_from_result(result)
                
                if new_history_id:
                    current_history_id = new_history_id
                    logger.info(f"Updated history_id: {new_history_id}")
                
                # Display the response
                if response_text:
                    render_assistant_message(response_text)
                else:
                    render_assistant_message("(No response generated)")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                render_error(str(e))
                
        except KeyboardInterrupt:
            # Clean exit
            print("\n\n" + "="*60)
            print("👋 Thanks for chatting! Goodbye!")
            print("📁 Logs saved to: " + log_file)
            print("="*60 + "\n")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            render_error(f"Unexpected error: {str(e)}")

def main():
    """Entry point."""
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass  # Clean exit handled in run_chat
    except Exception as e:
        import traceback
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print(f"📁 Check logs at: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()