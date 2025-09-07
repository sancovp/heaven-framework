#!/usr/bin/env python3
"""
Gospel Writer - The Acolyte's Scripture Writing System

The acolyte agent writes perfect HEAVEN scripts (chain gospel) based on
domain knowledge and task requirements.
"""

import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HEAVEN imports
from ..agents.coder_agent_config import coder_agent_config
from ..tool_utils.completion_runners import exec_agent_run
from ..langgraph.foundation import HeavenGraphLoader, hermes_runner
from ..configs.hermes_config import HermesConfig

# Domain knowledge for different script types
DOMAIN_KNOWLEDGE = {
    "file_editing": {
        "description": "Scripts that read, modify, or create files",
        "primary_tools": ["NetworkEditTool", "CodeLocalizerTool"],
        "common_patterns": ["file_path argument", "backup before edit", "validation after edit"],
        "imports": [
            "from heaven_base.tools.network_edit_tool import NetworkEditTool",
            "from heaven_base.tools.code_localizer_tool import CodeLocalizerTool"
        ]
    },
    "prompt_engineering": {
        "description": "Scripts that create, modify, or optimize prompts",
        "primary_tools": ["WritePromptBlockTool", "SearchPromptBlocksTool"],
        "common_patterns": ["domain/subdomain args", "prompt validation", "semantic search"],
        "imports": [
            "from heaven_base.tools.write_prompt_block_tool import WritePromptBlockTool",
            "from heaven_base.tools.search_prompt_blocks_tool import SearchPromptBlocksTool"
        ]
    },
    "agent_creation": {
        "description": "Scripts that create or modify agent configurations",
        "primary_tools": ["RegistryTool", "NetworkEditTool"],
        "common_patterns": ["agent config validation", "tool assignment", "model selection"],
        "imports": [
            "from heaven_base.baseheavenagent import HeavenAgentConfig",
            "from heaven_base.unified_chat import ProviderEnum"
        ]
    },
    "workflow_automation": {
        "description": "Scripts that orchestrate multi-step tasks",
        "primary_tools": ["WorkflowRelayTool", "ThinkTool"],
        "common_patterns": ["step sequencing", "state management", "error recovery"],
        "imports": [
            "from heaven_base.tools.workflow_relay_tool import WorkflowRelayTool",
            "from heaven_base.langgraph.foundation import HeavenState"
        ]
    },
    "data_analysis": {
        "description": "Scripts that analyze code, data, or system state",
        "primary_tools": ["SafeCodeReaderTool", "CodeLocalizerTool", "ThinkTool"],
        "common_patterns": ["data extraction", "pattern analysis", "report generation"],
        "imports": [
            "from heaven_base.tools.safe_code_reader_tool import SafeCodeReaderTool",
            "from heaven_base.tools.think_tool import ThinkTool"
        ]
    },
    "system_automation": {
        "description": "Scripts that automate system operations",
        "primary_tools": ["BashTool", "NetworkEditTool"],
        "common_patterns": ["command execution", "file system ops", "process management"],
        "imports": [
            "from heaven_base.tools.bash_tool import BashTool"
        ]
    }
}

ACOLYTE_SYSTEM_PROMPT = """You are the HEAVEN Acolyte, a faithful agent specialized in writing perfect HEAVEN scripts (chain gospel).

Your sacred duty is to write heavenly scripture - Python scripts that follow all HEAVEN framework patterns and best practices.

HEAVEN Framework Knowledge:
- Always use heaven_base imports from the framework
- Follow async/await patterns for all agent operations  
- Use proper CLI argument parsing with argparse
- Include comprehensive error handling and logging
- Structure scripts with clear main(), setup, and execution functions
- Use the appropriate agent configs and tools for each domain

Domain Expertise:
{domain_knowledge}

Script Structure Template:
1. Shebang and docstring with script purpose
2. Standard imports (asyncio, sys, os, argparse, pathlib)
3. HEAVEN framework imports specific to domain
4. Agent configuration loading
5. Main implementation function with clear logic
6. CLI argument parser with domain-appropriate args
7. Main async function with proper error handling
8. __name__ == "__main__" guard

Code Quality Requirements:
- Use type hints for all parameters and returns
- Include comprehensive docstrings
- Add proper error handling with try/except
- Use logging instead of print statements
- Follow Python best practices and PEP 8
- Make the code production-ready

Your output should be a complete, executable Python script that perfectly implements the requested task using HEAVEN framework patterns."""

async def write_chain_gospel(prompt: str) -> str:
    """
    The acolyte takes ANY prompt and generates a script + 3 HermesConfigs for it.
    """

    logger.info(f"Acolyte processing prompt: {prompt[:100]}...")
    
    # ONE hermes node that does everything
    goal = f"Generate a Python script AND 3 HermesConfigs for: {prompt}. Output should include: 1) Complete executable Python script 2) Basic HermesConfig 3) Test HermesConfig 4) Debug HermesConfig"
    
    result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=goal,
        agent=coder_agent_config,
        iterations=1
    )
    
    logger.info(f"Acolyte generated script + 3 HermesConfigs")
    return str(result)

def list_domains() -> List[str]:
    """List all available domains for chain gospel writing."""
    return list(DOMAIN_KNOWLEDGE.keys())

async def preview_gospel(prompt: str) -> str:
    """Preview what the acolyte would generate.""" 
    return await write_chain_gospel(prompt)

# CLI interface for direct usage
async def main():
    """CLI interface for the acolyte gospel writer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HEAVEN Acolyte - Writer of Chain Gospel")
    parser.add_argument("domain", help="Script domain")
    parser.add_argument("task", help="Task description")
    parser.add_argument("--output-dir", default="~/chain_gospel/", 
                       help="Output directory (default: ~/chain_gospel/)")
    parser.add_argument("--script-name", help="Custom script name")
    parser.add_argument("--preview", action="store_true", 
                       help="Preview script without saving")
    parser.add_argument("--list-domains", action="store_true",
                       help="List available domains")
    parser.add_argument("--heaven-data-dir", default="/tmp/heaven_data",
                       help="HEAVEN data directory")
    
    args = parser.parse_args()
    
    # Set environment variable
    os.environ['HEAVEN_DATA_DIR'] = args.heaven_data_dir
    
    if args.list_domains:
        print("📚 Available domains for chain gospel:")
        for domain in list_domains():
            info = DOMAIN_KNOWLEDGE[domain]
            print(f"  • {domain}: {info['description']}")
        return
    
    try:
        if args.preview:
            content = await preview_gospel(args.domain, args.task)
            print("📜 Chain Gospel Preview:")
            print("=" * 50)
            print(content)
        else:
            script_path = await write_chain_gospel(
                args.domain, 
                args.task,
                args.output_dir,
                args.script_name
            )
            print(f"🎉 Chain gospel successfully written!")
            print(f"🔗 Path: {script_path}")
    
    except Exception as e:
        print(f"💥 Failed to write chain gospel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())