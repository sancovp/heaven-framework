#!/usr/bin/env python3
"""
Working example of LangGraph LEGOs in HEAVEN framework
This demonstrates actual usage of the components
"""

import asyncio
import sys
from pathlib import Path

# Add heaven-framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heaven_base.langgraph.foundation import (
    HeavenState,
    HeavenNodeType,
    completion_runner,
    hermes_runner
)
from heaven_base.langgraph.hermes_legos import (
    HermesState,
    completion_node,
    hermes_node,
    build_simple_hermes_graph,
    build_iterative_hermes_graph,
    should_continue_iterations
)
from heaven_base.baseheavenagent import HeavenAgentConfig
from heaven_base.unified_chat import ProviderEnum
from langgraph.graph import StateGraph, END, START


# === Example 1: Simple completion runner ===

async def example_completion_runner():
    """Example using the completion_runner from foundation.py"""
    print("\n=== Example 1: Completion Runner ===")
    
    # Create state
    state = HeavenState(
        results=[],
        context={"project": "test"},
        agents={}
    )
    
    # Create test agent
    agent = HeavenAgentConfig(
        name="SimpleAgent",
        system_prompt="You are a helpful assistant. Be concise.",
        tools=[],
        provider=ProviderEnum.OPENAI,
        model="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Store agent in state
    state["agents"]["simple"] = agent
    
    # Run completion (mock - would need actual API key to run)
    print("Would execute: completion_runner with prompt 'Hello, HEAVEN!'")
    print(f"State has {len(state['agents'])} agent(s) registered")
    
    # In real usage:
    # result = await completion_runner(
    #     state,
    #     prompt="Hello, HEAVEN!",
    #     agent=agent
    # )
    # print(f"Results stored: {len(result['results'])}")


# === Example 2: Building custom graph with LEGOs ===

async def example_custom_graph():
    """Example building a custom graph using HEAVEN LEGOs"""
    print("\n=== Example 2: Custom Graph with LEGOs ===")
    
    # Define custom node that uses HEAVEN components
    async def analysis_node(state: HermesState) -> dict:
        """Custom node that analyzes the current state"""
        print(f"  Analyzing state with {len(state['messages'])} messages")
        
        # Add analysis to extracted content
        analysis = {
            "message_count": len(state["messages"]),
            "has_goal": bool(state["current_goal"]),
            "iteration": state["iteration_count"]
        }
        
        return {
            "extracted_content": {
                **state["extracted_content"],
                "analysis": analysis
            },
            "iteration_count": state["iteration_count"] + 1
        }
    
    # Build graph
    graph = StateGraph(HermesState)
    
    # Add nodes
    graph.add_node("analyze", analysis_node)
    graph.add_node("complete", completion_node)
    
    # Add flow
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "complete")
    graph.add_edge("complete", END)
    
    # Compile
    compiled_graph = graph.compile()
    print("✓ Built custom graph with analysis → completion flow")
    
    return compiled_graph


# === Example 3: Iterative processing with conditions ===

async def example_iterative_processing():
    """Example of iterative processing with conditional edges"""
    print("\n=== Example 3: Iterative Processing ===")
    
    # Custom condition function
    def check_quality(state: HermesState) -> str:
        """Check if we have enough quality results"""
        # Check extracted content for a quality signal
        quality_score = state["extracted_content"].get("quality", 0)
        
        if quality_score >= 0.8:
            return "good"
        elif state["iteration_count"] >= state["max_iterations"]:
            return "max_reached"
        else:
            return "continue"
    
    # Build graph with conditional routing
    graph = StateGraph(HermesState)
    
    # Mock quality assessment node
    async def assess_quality(state: HermesState) -> dict:
        # Simulate quality improving with iterations
        quality = min(0.3 + (state["iteration_count"] * 0.25), 1.0)
        print(f"  Iteration {state['iteration_count']}: Quality = {quality:.2f}")
        
        return {
            "extracted_content": {
                **state["extracted_content"],
                "quality": quality
            },
            "iteration_count": state["iteration_count"] + 1
        }
    
    # Add nodes
    graph.add_node("assess", assess_quality)
    
    # Add conditional routing
    graph.add_edge(START, "assess")
    graph.add_conditional_edges(
        "assess",
        check_quality,
        {
            "good": END,
            "max_reached": END,
            "continue": "assess"
        }
    )
    
    compiled = graph.compile()
    print("✓ Built iterative graph with quality-based conditions")
    
    # Test run (mock)
    initial_state = HermesState(
        messages=[],
        heaven_events=[],
        current_goal="Improve quality",
        agent_config=None,
        extracted_content={},
        iteration_count=0,
        max_iterations=5
    )
    
    print("\nSimulating iterative quality improvement:")
    result = await compiled.ainvoke(initial_state)
    
    final_quality = result["extracted_content"].get("quality", 0)
    print(f"\n✓ Completed after {result['iteration_count']} iterations")
    print(f"  Final quality: {final_quality:.2f}")


# === Example 4: Node type usage ===

def example_node_types():
    """Example showing all available node types"""
    print("\n=== Example 4: Available Node Types ===")
    
    node_categories = {
        "Execution": [
            HeavenNodeType.COMPLETION,
            HeavenNodeType.HERMES,
            HeavenNodeType.HERMES_CONFIG
        ],
        "Context Management": [
            HeavenNodeType.CONTEXT_WEAVE,
            HeavenNodeType.CONTEXT_INJECT,
            HeavenNodeType.PIS_INJECTION
        ],
        "Tool Operations": [
            HeavenNodeType.OMNITOOL_LIST,
            HeavenNodeType.OMNITOOL_GET_INFO,
            HeavenNodeType.OMNITOOL_CALL
        ],
        "Data Processing": [
            HeavenNodeType.RESULT_EXTRACTOR,
            HeavenNodeType.DYNAMIC_FUNCTION,
            HeavenNodeType.CHAIN_PATTERN
        ],
        "Advanced": [
            HeavenNodeType.SUBGRAPH,
            HeavenNodeType.BRAIN_AGENT
        ]
    }
    
    for category, types in node_categories.items():
        print(f"\n{category}:")
        for node_type in types:
            print(f"  • {node_type}")
    
    print("\n✓ These node types can be used to build complex workflows")


# === Example 5: Practical workflow pattern ===

async def example_practical_workflow():
    """Example of a practical workflow using LEGOs"""
    print("\n=== Example 5: Practical Workflow Pattern ===")
    
    # This shows how you might structure a real workflow
    workflow_spec = {
        "name": "DocumentAnalysisWorkflow",
        "nodes": [
            {
                "id": "extract",
                "type": HeavenNodeType.COMPLETION,
                "config": {
                    "prompt": "Extract key points from: {document}",
                    "agent": "analyzer"
                }
            },
            {
                "id": "enhance", 
                "type": HeavenNodeType.HERMES,
                "config": {
                    "goal": "Enhance and structure the extracted points",
                    "iterations": 2
                }
            },
            {
                "id": "format",
                "type": HeavenNodeType.RESULT_EXTRACTOR,
                "config": {
                    "extract_keys": ["summary", "key_points", "recommendations"]
                }
            }
        ],
        "edges": [
            {"from": "START", "to": "extract"},
            {"from": "extract", "to": "enhance"},
            {"from": "enhance", "to": "format"},
            {"from": "format", "to": "END"}
        ]
    }
    
    print(f"Workflow: {workflow_spec['name']}")
    print(f"Pipeline: {' → '.join([n['id'] for n in workflow_spec['nodes']])}")
    print("\n✓ This pattern can be used to build document processing pipelines")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("HEAVEN LangGraph LEGOs - Working Examples")
    print("=" * 60)
    
    # Run examples
    await example_completion_runner()
    await example_custom_graph()
    await example_iterative_processing()
    example_node_types()
    await example_practical_workflow()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("• LEGOs are composable building blocks for workflows")
    print("• States manage data flow between nodes")
    print("• Nodes can be simple functions or complex operations")
    print("• Conditional edges enable dynamic flow control")
    print("• Node types provide semantic meaning to operations")


if __name__ == "__main__":
    asyncio.run(main())