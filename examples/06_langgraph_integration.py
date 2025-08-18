#!/usr/bin/env python3
"""
HEAVEN LangGraph Integration Example
Shows how to use HEAVEN components with LangGraph workflows
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    HeavenAgentConfig,
    ProviderEnum
)
from heaven_base.langgraph import (
    HeavenState,
    HeavenNodeType,
    completion_runner,
    hermes_runner
)
from langgraph.graph import StateGraph, END, START

async def main():
    # Create agent configuration
    config = HeavenAgentConfig(
        name="LangGraphAgent",
        system_prompt="You are a helpful assistant integrated with LangGraph workflows.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.7
    )
    
    # Define a simple workflow node using HEAVEN completion runner
    async def analysis_node(state: HeavenState) -> dict:
        """Node that analyzes the input using HEAVEN completion runner"""
        prompt = f"Analyze this request: {state.get('user_input', 'No input provided')}"
        
        # Use HEAVEN's completion runner within the LangGraph node
        result = await completion_runner(
            state=state,
            prompt=prompt,
            agent=config
        )
        
        # Update state with analysis results
        return {
            "results": state.get("results", []) + [f"Analysis: {result.get('response', 'No response')}"],
            "context": {**state.get("context", {}), "analysis_complete": True},
            "agents": {**state.get("agents", {}), "analyzer": config}
        }
    
    # Define a summary node
    async def summary_node(state: HeavenState) -> dict:
        """Node that summarizes all results"""
        results = state.get("results", [])
        summary = f"Workflow completed with {len(results)} results: " + "; ".join(results)
        
        return {
            "results": results + [f"Summary: {summary}"],
            "context": {**state.get("context", {}), "workflow_complete": True}
        }
    
    # Build the LangGraph workflow
    print("=== Building LangGraph Workflow ===")
    workflow = StateGraph(HeavenState)
    
    # Add nodes
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("summarize", summary_node)
    
    # Add edges
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "summarize")
    workflow.add_edge("summarize", END)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    print("✓ LangGraph workflow compiled successfully")
    
    # Create initial state
    initial_state = HeavenState(
        results=[],
        context={"project": "langgraph_integration"},
        agents={},
        user_input="What are the benefits of using LangGraph with HEAVEN framework?"
    )
    
    # Execute the workflow
    print("\n=== Executing LangGraph Workflow ===")
    print(f"Input: {initial_state['user_input']}")
    
    try:
        final_state = await compiled_workflow.ainvoke(initial_state)
        
        print("\n=== Workflow Results ===")
        for i, result in enumerate(final_state.get("results", []), 1):
            print(f"{i}. {result}")
        
        print(f"\nContext: {final_state.get('context', {})}")
        print(f"Agents used: {list(final_state.get('agents', {}).keys())}")
        
    except Exception as e:
        print(f"Workflow execution error: {e}")
        print("Note: This demonstrates the LangGraph integration structure")

if __name__ == "__main__":
    asyncio.run(main())