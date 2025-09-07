#!/usr/bin/env python3
"""
Run the Acolyte Agent - Generates scripts and HermesConfigs

Usage:
    python run_acolyte.py "create a script that analyzes log files"
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, '/home/GOD/heaven-framework-repo')

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from heaven_base.langgraph.hermes_legos import hermes_node, HermesState
from heaven_base.acolyte_v2.acolyte_agent_config import acolyte_agent_config

def build_acolyte_graph():
    """Build a proper LangGraph with hermes_node"""
    graph = StateGraph(HermesState)
    
    # Add the hermes node
    graph.add_node("acolyte_execute", hermes_node)
    
    # Add edges
    graph.add_edge(START, "acolyte_execute")
    graph.add_edge("acolyte_execute", END)
    
    return graph.compile()

async def run_acolyte(prompt: str):
    """Run the acolyte agent with proper LangGraph"""
    
    # Build the graph
    acolyte_graph = build_acolyte_graph()
    
    # Create initial state - all required fields for HermesState
    initial_state = {
        "messages": [],
        "heaven_events": [],
        "current_goal": prompt,
        "agent_config": acolyte_agent_config,
        "extracted_content": {},
        "iteration_count": 0,
        "max_iterations": 1,
        "hermes_result": None
    }
    
    # Run the graph
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "acolyte_session"}}
    result = await acolyte_graph.ainvoke(initial_state, config)
    
    # Debug: print entire result structure
    print("=== FULL RESULT DEBUG ===")
    for key, value in result.items():
        print(f"{key}: {type(value)}")
        if key == 'hermes_result' and value:
            print(f"hermes_result keys: {value.keys() if isinstance(value, dict) else 'not dict'}")
            if isinstance(value, dict) and 'prepared_message' in value:
                print("=== ACOLYTE OUTPUT ===")
                print(value['prepared_message'])
                print("=== END OUTPUT ===")
    
    return result

async def main():
    if len(sys.argv) != 2:
        print("Usage: python run_acolyte.py 'your prompt here'")
        sys.exit(1)
    
    # Set environment variable
    os.environ['HEAVEN_DATA_DIR'] = os.environ.get('HEAVEN_DATA_DIR', '/tmp/heaven_data')
    
    prompt = sys.argv[1]
    result = await run_acolyte(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())