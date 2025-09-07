#!/usr/bin/env python3
"""
HEAVEN CLI Demo - HTTP Client Example

This shows how to create a CLI for your agent using the HEAVEN HTTP server.

Usage:
1. Start the HEAVEN HTTP server: python /home/GOD/core/image/http_server.py
2. Run this script: python cli_demo.py
"""

# Make sure to add the framework to path if needed
import sys
import os
from pathlib import Path

# Add heaven-framework-repo to path
framework_path = Path(__file__).parent.parent
if str(framework_path) not in sys.path:
    sys.path.insert(0, str(framework_path))

from heaven_base.cli import make_cli
from heaven_base import HeavenAgentConfig, ProviderEnum
from heaven_base.tools import WorkflowRelayTool

# Configure your agent
AGENT_CONFIG = HeavenAgentConfig(
    name="DemoAssistant",
    system_prompt="""You are a helpful AI assistant with workflow capabilities.
Be friendly, concise, and clear in your responses.
You have access to WorkflowRelayTool which can help with complex workflows.""",
    tools=[WorkflowRelayTool],  # Add your tools here
    provider=ProviderEnum.OPENAI,
    model="gpt-5-mini",
    temperature=0.7,
    max_tokens=1000
)

def main():
    # Create CLI instance
    cli = make_cli(
        agent_config=AGENT_CONFIG,
        server_url="http://localhost:8080"  # Default HEAVEN server URL
    )
    
    # Run the CLI
    cli.run_sync()

if __name__ == "__main__":
    main()