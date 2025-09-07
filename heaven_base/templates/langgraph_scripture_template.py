#!/usr/bin/env python3
"""
{{SCRIPT_NAME}} - {{SCRIPT_DESCRIPTION}}

Usage:
    python {{SCRIPT_NAME}}.py "{{EXAMPLE_PROMPT}}"

This is a HEAVEN LangGraph Scripture - a standardized pattern for building 
composable agent execution systems using proper hermes_node integration.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, '/home/GOD/heaven-framework-repo')

# === LANGGRAPH IMPORTS ===
# These are the core LangGraph components needed for any scripture
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# === HEAVEN IMPORTS ===
# Import the hermes_node (the standardized HEAVEN execution node)
# and HermesState (the state schema that works with hermes_node)
from heaven_base.langgraph.hermes_legos import hermes_node, HermesState
# from {{AGENT_CONFIG_MODULE}} import {{AGENT_CONFIG_NAME}}
# NOTE: Replace the above line with actual import when generating scripture

def build_scripture_graph():
    """
    Build a proper LangGraph with hermes_node.
    
    This is the standard pattern for HEAVEN scriptures:
    1. Create StateGraph with HermesState schema
    2. Add hermes_node (which includes complete hermes pipeline)  
    3. Connect START -> hermes_node -> END
    4. Compile and return
    
    The hermes_node handles:
    - hermes_step execution
    - Block report processing via handle_hermes_response
    - Complete HEAVEN agent pipeline
    """
    graph = StateGraph(HermesState)
    
    # Add the hermes node - this is the ONLY node most scriptures need
    # hermes_node contains the complete HEAVEN execution pipeline
    graph.add_node("scripture_execute", hermes_node)
    
    # Standard edges: START -> execution -> END  
    graph.add_edge(START, "scripture_execute")
    graph.add_edge("scripture_execute", END)
    
    return graph.compile()

async def run_scripture(prompt: str):
    """
    Execute the scripture with proper LangGraph.
    
    Standard execution pattern:
    1. Build the graph
    2. Create initial state with ALL required HermesState fields
    3. Execute with LangGraph
    4. Extract results from hermes_result field
    """
    
    # Build the graph using our standard pattern
    scripture_graph = build_scripture_graph()
    
    # Create initial state - MUST include ALL HermesState fields
    # These are the minimum required fields for hermes_node:
    initial_state = {
        "messages": [],                    # LangGraph message history (usually empty for simple scripts)
        "heaven_events": [],               # HEAVEN event tracking (hermes_node will populate this)
        "current_goal": prompt,            # The goal/prompt for the agent (REQUIRED)
        "agent_config": None,              # The agent configuration to execute (REQUIRED - replace with actual config)
        "extracted_content": {},           # Content extraction results (hermes_node may populate)
        "iteration_count": 0,             # Iteration tracking (hermes_node uses this)
        "max_iterations": 1,              # Maximum iterations (usually 1 for simple scripts)
        "hermes_result": None             # Will contain the complete hermes execution result
    }
    
    # Execute the graph with standard LangGraph pattern
    memory = MemorySaver()  # Memory management for LangGraph
    config = {"configurable": {"thread_id": "scripture_session"}}  # Session config
    result = await scripture_graph.ainvoke(initial_state, config)
    
    # Extract and display results from hermes_result
    # The hermes_result contains the complete processed output from HEAVEN
    if result.get('hermes_result'):
        if isinstance(result['hermes_result'], dict) and 'prepared_message' in result['hermes_result']:
            print("=== {{SCRIPT_NAME}} OUTPUT ===")
            print(result['hermes_result']['prepared_message'])
            print("=== END OUTPUT ===")
        else:
            print("=== {{SCRIPT_NAME}} OUTPUT ===")
            print(result['hermes_result'])
            print("=== END OUTPUT ===")
    else:
        print("No hermes result found")
    
    return result

async def main():
    """
    Standard main function for HEAVEN scriptures.
    
    Pattern:
    1. Validate command line arguments
    2. Set required environment variables
    3. Execute the scripture function
    """
    if len(sys.argv) != 2:
        print("Usage: python {{SCRIPT_NAME}}.py 'your prompt here'")
        sys.exit(1)
    
    # Set environment variable (required for HEAVEN)
    os.environ['HEAVEN_DATA_DIR'] = os.environ.get('HEAVEN_DATA_DIR', '/tmp/heaven_data')
    
    prompt = sys.argv[1]
    result = await run_scripture(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

# === TEMPLATE VARIABLES TO REPLACE ===
"""
When generating a new scripture from this template, replace:

{{SCRIPT_NAME}} - The name of the script file (e.g., "run_coder", "run_analyst") 
{{SCRIPT_DESCRIPTION}} - What this scripture does (e.g., "Executes code analysis agent")
{{EXAMPLE_PROMPT}} - Example prompt for usage (e.g., "analyze this Python file")
{{AGENT_CONFIG_MODULE}} - Module path for agent config (e.g., "heaven_base.agents.coder_agent_config")
{{AGENT_CONFIG_NAME}} - Agent config variable name (e.g., "coder_agent_config")
{{SCRIPT_FUNCTION_NAME}} - Function name (e.g., "coder", "analyst") 
{{NODE_NAME}} - LangGraph node name (e.g., "coder_execute", "analyst_execute")

CRITICAL PRINCIPLES:

1. **Always use hermes_node** - Don't create custom nodes, use the standard hermes_node
2. **Complete HermesState** - Include ALL required fields in initial_state
3. **hermes_result extraction** - Results come from hermes_result.prepared_message
4. **Standard patterns** - Follow the exact LangGraph setup pattern shown
5. **Composability** - Each scripture is a complete LangGraph that can be used as subgraph

This template creates scriptures that:
- Execute any HEAVEN agent through proper hermes pipeline
- Are composable into larger LangGraph workflows  
- Follow consistent patterns for maintainability
- Handle all HEAVEN execution requirements automatically
"""