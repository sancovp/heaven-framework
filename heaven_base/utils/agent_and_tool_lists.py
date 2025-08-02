from typing import List, Dict
import os
import importlib

def get_tool_modules() -> str:
    """Get formatted string of available tool classes from tools.__all__"""
    try:
        # Try to import from heaven_base tools
        tools_module = importlib.import_module("heaven_base.tools")
        available_tools = [tool for tool in getattr(tools_module, '__all__', []) if tool != "HermesTool"]
        return ", ".join(sorted(available_tools))
    except (ImportError, AttributeError):
        # Fallback to basic tools we know exist
        return "StraightforwardSummarizerTool, WriteBlockReportTool"

def get_agent_modules() -> str:
    """Get formatted string of available agent directories"""
    # Get the path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    agents_path = os.path.join(os.path.dirname(current_dir), "agents")
    
    modules = []
    if os.path.exists(agents_path):
        for item in os.listdir(agents_path):
            full_path = os.path.join(agents_path, item)
            if (os.path.isdir(full_path) and 
                not item.startswith('__') and 
                item != '__pycache__'):
                modules.append(item)
    
    # Fallback if no agents directory exists
    if not modules:
        modules = ["summary_agent"]
    
    return ", ".join(sorted(modules))