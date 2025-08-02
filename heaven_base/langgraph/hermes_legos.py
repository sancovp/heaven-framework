"""
LangGraph LEGO blocks for HEAVEN Hermes System

This file demonstrates how to create reusable LangGraph components that integrate
with our HermesConfig and execution methods.
"""

import asyncio
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union
from functools import partial

# LangGraph imports
from langgraph.graph import StateGraph, END, START, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# HEAVEN imports
from ..baseheavenagent import HeavenAgentConfig
from ..unified_chat import ProviderEnum
from ..tool_utils.completion_runners import exec_completion_style
from ..tool_utils.hermes_utils import hermes_step
from ..configs.hermes_config import HermesConfig
from ..memory.heaven_event import HeavenEvent
from ..memory.heaven_history import HeavenHistory


# === STATE DEFINITIONS ===

class HermesState(TypedDict):
    """Standard state for Hermes workflows"""
    messages: Annotated[List[BaseMessage], add_messages]
    heaven_events: List[Dict[str, Any]]  # HEAVEN event history
    current_goal: str
    agent_config: Optional[HeavenAgentConfig]
    extracted_content: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    
class ChainState(TypedDict):
    """State for agent chains"""
    messages: Annotated[List[BaseMessage], add_messages]
    heaven_events: List[Dict[str, Any]]
    agents: List[HeavenAgentConfig]  # List of agents in the chain
    current_agent_index: int
    chain_memory: Dict[str, Any]  # Shared memory across chain


# === LEGO NODES ===

async def completion_node(state: HermesState) -> Dict[str, Any]:
    """
    Node that executes an agent using completion style.
    Simple prompt → response execution.
    """
    # Get the last user message as prompt
    prompt = state["current_goal"]
    if not prompt and state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            prompt = last_msg.content
    
    # Execute using completion style
    result = await exec_completion_style(
        prompt=prompt,
        agent=state.get("agent_config")
    )
    
    # Extract messages and convert to HEAVEN events
    new_events = []
    if "messages" in result:
        for msg in result["messages"]:
            if hasattr(msg, "content"):
                event = HeavenEvent(
                    event_type="AGENT_MESSAGE",
                    data={"content": msg.content}
                )
                new_events.append(event.to_dict())
    
    # Update state
    return {
        "heaven_events": state["heaven_events"] + new_events,
        "messages": result.get("messages", []),
        "iteration_count": state["iteration_count"] + 1
    }


async def hermes_node(state: HermesState) -> Dict[str, Any]:
    """
    Node that executes an agent using hermes_step.
    Orchestrated execution with history tracking.
    """
    # Execute using hermes_step
    result = await hermes_step(
        target_container="mind_of_god",
        source_container="mind_of_god",
        goal=state["current_goal"],
        iterations=1,
        agent=state.get("agent_config")
    )
    
    # Extract HEAVEN events from result
    new_events = []
    if isinstance(result, dict):
        # Process the raw_result if available
        raw_result = result.get("raw_result", {})
        messages = raw_result.get("messages", [])
        
        for msg in messages:
            if hasattr(msg, "content"):
                event_type = "USER_MESSAGE" if isinstance(msg, HumanMessage) else "AGENT_MESSAGE"
                event = HeavenEvent(
                    event_type=event_type,
                    data={"content": msg.content}
                )
                new_events.append(event.to_dict())
    
    # Update state
    return {
        "heaven_events": state["heaven_events"] + new_events,
        "extracted_content": {
            **state["extracted_content"],
            **result.get("extracted_content", {})
        },
        "iteration_count": state["iteration_count"] + 1
    }


async def hermes_config_node(state: HermesState, config: HermesConfig, variable_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that executes using HermesConfig with template substitution.
    """
    # Convert config to command data with variable substitution
    command_data = config.to_command_data(variable_inputs)
    
    # Execute using hermes_step
    result = await hermes_step(
        target_container="mind_of_god",
        source_container="mind_of_god",
        **command_data
    )
    
    # Extract and process results
    new_events = []
    if isinstance(result, dict):
        raw_result = result.get("raw_result", {})
        messages = raw_result.get("messages", [])
        
        for msg in messages:
            if hasattr(msg, "content"):
                event_type = "USER_MESSAGE" if isinstance(msg, HumanMessage) else "AGENT_MESSAGE"
                event = HeavenEvent(
                    event_type=event_type,
                    data={"content": msg.content}
                )
                new_events.append(event.to_dict())
    
    return {
        "heaven_events": state["heaven_events"] + new_events,
        "extracted_content": {
            **state["extracted_content"],
            **result.get("extracted_content", {})
        },
        "iteration_count": state["iteration_count"] + 1
    }


async def chain_controller_node(state: ChainState) -> Dict[str, Any]:
    """
    Node that manages agent chains - decides which agent to run next.
    """
    current_index = state["current_agent_index"]
    agents = state["agents"]
    
    # Check if we've completed the chain
    if current_index >= len(agents):
        return {"current_agent_index": -1}  # Signal completion
    
    # Get current agent
    current_agent = agents[current_index]
    
    # Execute the agent with chain context
    context = f"Chain step {current_index + 1} of {len(agents)}. "
    if state["messages"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content"):
            context += last_msg.content
    
    result = await exec_completion_style(
        prompt=context,
        agent=current_agent
    )
    
    # Update chain memory
    chain_memory = state["chain_memory"]
    chain_memory[f"agent_{current_index}_result"] = result
    
    # Create HEAVEN events
    new_events = []
    event = HeavenEvent(
        event_type="AGENT_MESSAGE", 
        data={
            "content": result.get("output", ""),
            "agent_name": current_agent.name,
            "chain_position": current_index
        }
    )
    new_events.append(event.to_dict())
    
    return {
        "messages": state["messages"] + result.get("messages", []),
        "heaven_events": state["heaven_events"] + new_events,
        "current_agent_index": current_index + 1,
        "chain_memory": chain_memory
    }


# === LEGO EDGES ===

def should_continue_iterations(state: HermesState) -> str:
    """Edge function to decide if we should continue iterations"""
    if state["iteration_count"] >= state["max_iterations"]:
        return "end"
    return "continue"


def should_continue_chain(state: ChainState) -> str:
    """Edge function to decide if chain should continue"""
    if state["current_agent_index"] == -1 or state["current_agent_index"] >= len(state["agents"]):
        return "end"
    return "continue"


# === GRAPH BUILDERS ===

def build_simple_hermes_graph(agent_config: HeavenAgentConfig) -> StateGraph:
    """
    Build a simple graph that executes an agent once.
    """
    graph = StateGraph(HermesState)
    
    # Add nodes
    graph.add_node("execute", completion_node)
    
    # Add edges
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    
    return graph.compile()


def build_iterative_hermes_graph(agent_config: HeavenAgentConfig, max_iterations: int = 3) -> StateGraph:
    """
    Build a graph that can iterate multiple times.
    """
    graph = StateGraph(HermesState)
    
    # Add nodes
    graph.add_node("hermes_execute", hermes_node)
    
    # Add conditional edges
    graph.add_edge(START, "hermes_execute")
    graph.add_conditional_edges(
        "hermes_execute",
        should_continue_iterations,
        {
            "continue": "hermes_execute",
            "end": END
        }
    )
    
    return graph.compile()


def build_agent_chain_graph(agents: List[HeavenAgentConfig]) -> StateGraph:
    """
    Build a graph that chains multiple agents together.
    """
    graph = StateGraph(ChainState)
    
    # Add controller node
    graph.add_node("chain_controller", chain_controller_node)
    
    # Add edges
    graph.add_edge(START, "chain_controller")
    graph.add_conditional_edges(
        "chain_controller",
        should_continue_chain,
        {
            "continue": "chain_controller",
            "end": END
        }
    )
    
    return graph.compile()


def build_config_template_graph(config: HermesConfig, variable_inputs: Dict[str, Any]) -> StateGraph:
    """
    Build a graph that uses HermesConfig templates.
    """
    graph = StateGraph(HermesState)
    
    # Create node with config bound
    config_node = partial(hermes_config_node, config=config, variable_inputs=variable_inputs)
    
    # Add nodes
    graph.add_node("template_execute", config_node)
    
    # Add edges
    graph.add_edge(START, "template_execute")
    graph.add_edge("template_execute", END)
    
    return graph.compile()


# === UTILITY FUNCTIONS ===

async def run_graph_with_history(
    graph: StateGraph,
    initial_goal: str,
    agent_config: Optional[HeavenAgentConfig] = None,
    heaven_history: Optional[HeavenHistory] = None
) -> HeavenHistory:
    """
    Run a compiled graph and return updated HeavenHistory.
    """
    # Initialize history if not provided
    if heaven_history is None:
        heaven_history = HeavenHistory(messages=[])
    
    # Create initial state
    initial_state = {
        "messages": heaven_history.messages,
        "heaven_events": heaven_history.heaven_events,
        "current_goal": initial_goal,
        "agent_config": agent_config,
        "extracted_content": {},
        "iteration_count": 0,
        "max_iterations": 3
    }
    
    # Run the graph
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "hermes_graph_session"}}
    result = await graph.ainvoke(initial_state, config)
    
    # Update heaven history with results
    heaven_history.heaven_events = result.get("heaven_events", [])
    heaven_history.messages = result.get("messages", [])
    
    return heaven_history


# === EXAMPLE USAGE ===

async def demo_langgraph_legos():
    """Demonstrate using the LangGraph LEGO blocks"""
    
    print("=== LangGraph LEGO Blocks Demo ===\n")
    
    # Define test agents
    pattern_agent = HeavenAgentConfig(
        name="PatternDetector",
        system_prompt="You are a pattern detection expert. Identify patterns concisely.",
        tools=[],
        provider=ProviderEnum.ANTHROPIC,
        model="claude-3-5-sonnet-latest",
        temperature=0.3
    )
    
    math_agent = HeavenAgentConfig(
        name="MathExpert",
        system_prompt="You are a mathematics expert. Solve problems step by step.",
        tools=[],
        provider=ProviderEnum.ANTHROPIC,
        model="claude-3-5-sonnet-latest",
        temperature=0.1
    )
    
    # Example 1: Simple execution
    print("1. Simple Graph Execution:")
    simple_graph = build_simple_hermes_graph(pattern_agent)
    history = await run_graph_with_history(
        simple_graph,
        "What pattern do you see in: 2, 4, 8, 16, ?",
        pattern_agent
    )
    print(f"Events generated: {len(history.heaven_events)}\n")
    
    # Example 2: Iterative execution
    print("2. Iterative Graph Execution:")
    iter_graph = build_iterative_hermes_graph(pattern_agent, max_iterations=2)
    history = await run_graph_with_history(
        iter_graph,
        "Analyze patterns in: red, blue, red, blue, ?",
        pattern_agent
    )
    print(f"Events generated: {len(history.heaven_events)}\n")
    
    # Example 3: Agent chain
    print("3. Agent Chain Graph:")
    chain_graph = build_agent_chain_graph([pattern_agent, math_agent])
    chain_state = {
        "messages": [HumanMessage(content="First identify the pattern in 1,4,9,16,?, then calculate the next 3 terms")],
        "heaven_events": [],
        "agents": [pattern_agent, math_agent],
        "current_agent_index": 0,
        "chain_memory": {}
    }
    memory = MemorySaver()
    result = await chain_graph.ainvoke(chain_state, {"configurable": {"thread_id": "chain_demo"}})
    print(f"Chain completed with {len(result['heaven_events'])} events\n")
    
    # Example 4: Config template
    print("4. Config Template Graph:")
    template_config = HermesConfig(
        func_name="use_hermes",
        args_template={
            "goal": "Find the pattern in: {sequence}",
            "iterations": 1,
            "agent": pattern_agent,
            "additional_tools": None
        }
    )
    config_graph = build_config_template_graph(
        template_config,
        {"goal": {"sequence": "triangle, square, pentagon, ?"}}
    )
    history = await run_graph_with_history(config_graph, "", pattern_agent)
    print(f"Template execution generated {len(history.heaven_events)} events\n")
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(demo_langgraph_legos())