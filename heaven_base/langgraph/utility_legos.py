"""
LangGraph Utility LEGO blocks for HEAVEN

Additional nodes and edges for common patterns:
- Tool execution nodes
- Memory/context nodes  
- Routing/decision nodes
- Error handling nodes
- Aggregation nodes
"""

from typing import TypedDict, Dict, Any, List, Optional, Union, Callable
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

from ..baseheavenagent import HeavenAgentConfig
from ..baseheaventool import BaseHeavenTool
from ..memory.heaven_event import HeavenEvent
from ..tools.network_edit_tool import NetworkEditTool
from ..tools.bash_tool import BashTool


# === EXTENDED STATE TYPES ===

class ToolState(TypedDict):
    """State for tool execution workflows"""
    messages: List[BaseMessage]
    heaven_events: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]  # Pending tool calls
    tool_results: List[Dict[str, Any]]  # Tool execution results
    error_count: int
    max_retries: int


class RouterState(TypedDict):
    """State for routing/decision workflows"""
    messages: List[BaseMessage]
    heaven_events: List[Dict[str, Any]]
    route_context: Dict[str, Any]
    selected_route: Optional[str]
    route_history: List[str]


class MemoryState(TypedDict):
    """State for memory-enhanced workflows"""
    messages: List[BaseMessage]
    heaven_events: List[Dict[str, Any]]
    short_term_memory: Dict[str, Any]
    long_term_memory: Dict[str, Any]
    working_memory: Dict[str, Any]
    memory_queries: List[str]


# === UTILITY NODES ===

async def tool_execution_node(state: ToolState, tools: List[BaseHeavenTool]) -> Dict[str, Any]:
    """
    Execute pending tool calls and collect results.
    """
    tool_results = []
    new_events = []
    
    # Create tool map
    tool_map = {tool.name: tool for tool in tools}
    
    for tool_call in state["tool_calls"]:
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("tool_call_id", "")
        
        if tool_name not in tool_map:
            # Tool not found - create error event
            error_event = HeavenEvent(
                event_type="TOOL_ERROR",
                data={
                    "tool_name": tool_name,
                    "error": f"Tool '{tool_name}' not found",
                    "tool_call_id": tool_call_id
                }
            )
            new_events.append(error_event.to_dict())
            continue
        
        try:
            # Execute the tool
            tool = tool_map[tool_name]
            result = await tool.execute(**arguments)
            
            # Create success event
            result_event = HeavenEvent(
                event_type="TOOL_RESULT",
                data={
                    "tool_name": tool_name,
                    "output": result,
                    "tool_call_id": tool_call_id
                }
            )
            new_events.append(result_event.to_dict())
            tool_results.append({
                "tool_name": tool_name,
                "result": result,
                "tool_call_id": tool_call_id
            })
            
        except Exception as e:
            # Create error event
            error_event = HeavenEvent(
                event_type="TOOL_ERROR",
                data={
                    "tool_name": tool_name,
                    "error": str(e),
                    "tool_call_id": tool_call_id
                }
            )
            new_events.append(error_event.to_dict())
            state["error_count"] += 1
    
    # Add tool messages to state
    messages = []
    for result in tool_results:
        messages.append(ToolMessage(
            content=str(result["result"]),
            tool_call_id=result["tool_call_id"]
        ))
    
    return {
        "messages": state["messages"] + messages,
        "heaven_events": state["heaven_events"] + new_events,
        "tool_results": state["tool_results"] + tool_results,
        "tool_calls": []  # Clear pending calls
    }


async def smart_router_node(state: RouterState, route_logic: Callable) -> Dict[str, Any]:
    """
    Intelligent routing based on custom logic.
    """
    # Get routing context
    context = {
        "messages": state["messages"],
        "route_context": state["route_context"],
        "route_history": state["route_history"]
    }
    
    # Apply routing logic
    selected_route = route_logic(context)
    
    # Create routing event
    route_event = HeavenEvent(
        event_type="ROUTE_DECISION",
        data={
            "selected_route": selected_route,
            "context": state["route_context"],
            "history": state["route_history"]
        }
    )
    
    return {
        "selected_route": selected_route,
        "route_history": state["route_history"] + [selected_route],
        "heaven_events": state["heaven_events"] + [route_event.to_dict()]
    }


async def memory_retrieval_node(state: MemoryState) -> Dict[str, Any]:
    """
    Retrieve relevant information from different memory stores.
    """
    retrieved_info = {}
    new_events = []
    
    for query in state["memory_queries"]:
        # Search in different memory stores
        results = {
            "short_term": search_memory(state["short_term_memory"], query),
            "long_term": search_memory(state["long_term_memory"], query),
            "working": search_memory(state["working_memory"], query)
        }
        
        retrieved_info[query] = results
        
        # Create retrieval event
        event = HeavenEvent(
            event_type="MEMORY_RETRIEVAL",
            data={
                "query": query,
                "results": results
            }
        )
        new_events.append(event.to_dict())
    
    # Update working memory with retrieved info
    state["working_memory"]["retrieved"] = retrieved_info
    
    return {
        "working_memory": state["working_memory"],
        "heaven_events": state["heaven_events"] + new_events
    }


async def memory_storage_node(state: MemoryState, storage_policy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store information in appropriate memory stores based on policy.
    """
    # Extract information from recent messages
    recent_info = extract_information(state["messages"][-3:])
    
    # Apply storage policy
    if storage_policy.get("store_short_term", True):
        state["short_term_memory"].update(recent_info)
    
    if storage_policy.get("store_long_term", False):
        # Only store important info in long-term
        important_info = {k: v for k, v in recent_info.items() 
                         if is_important(k, v, storage_policy)}
        state["long_term_memory"].update(important_info)
    
    # Create storage event
    event = HeavenEvent(
        event_type="MEMORY_STORAGE",
        data={
            "stored_info": recent_info,
            "storage_locations": list(storage_policy.keys())
        }
    )
    
    return {
        "short_term_memory": state["short_term_memory"],
        "long_term_memory": state["long_term_memory"],
        "heaven_events": state["heaven_events"] + [event.to_dict()]
    }


async def error_handler_node(state: Dict[str, Any], error_strategy: str = "retry") -> Dict[str, Any]:
    """
    Handle errors with different strategies.
    """
    error_count = state.get("error_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if error_strategy == "retry" and error_count < max_retries:
        # Reset for retry
        return {
            "error_count": error_count,
            "should_retry": True
        }
    elif error_strategy == "fallback":
        # Switch to fallback behavior
        return {
            "use_fallback": True,
            "error_count": error_count
        }
    else:
        # Fail gracefully
        error_event = HeavenEvent(
            event_type="ERROR_LIMIT_REACHED",
            data={
                "error_count": error_count,
                "strategy": error_strategy
            }
        )
        return {
            "heaven_events": state.get("heaven_events", []) + [error_event.to_dict()],
            "should_terminate": True
        }


async def aggregation_node(state: Dict[str, Any], aggregation_key: str = "results") -> Dict[str, Any]:
    """
    Aggregate results from multiple sources.
    """
    results_to_aggregate = state.get(aggregation_key, [])
    
    # Different aggregation strategies
    aggregated = {
        "count": len(results_to_aggregate),
        "all_results": results_to_aggregate,
        "summary": summarize_results(results_to_aggregate),
        "consensus": find_consensus(results_to_aggregate)
    }
    
    # Create aggregation event
    event = HeavenEvent(
        event_type="AGGREGATION_COMPLETE",
        data={
            "aggregated": aggregated,
            "source_count": len(results_to_aggregate)
        }
    )
    
    return {
        "aggregated_results": aggregated,
        "heaven_events": state.get("heaven_events", []) + [event.to_dict()]
    }


# === UTILITY EDGES ===

def route_by_content(state: RouterState) -> str:
    """Route based on message content"""
    if not state["messages"]:
        return "empty"
    
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "")
    
    if "error" in content.lower():
        return "error_handler"
    elif "help" in content.lower():
        return "help_system"
    elif any(word in content.lower() for word in ["calculate", "compute", "solve"]):
        return "math_agent"
    else:
        return "general_agent"


def should_retry(state: Dict[str, Any]) -> str:
    """Decide if we should retry after error"""
    if state.get("should_retry", False):
        return "retry"
    elif state.get("use_fallback", False):
        return "fallback"
    else:
        return "end"


def memory_routing(state: MemoryState) -> str:
    """Route based on memory needs"""
    if state.get("memory_queries"):
        return "retrieve"
    elif needs_storage(state["messages"]):
        return "store"
    else:
        return "continue"


# === COMPOSED GRAPHS ===

def build_tool_enabled_graph(agent_config: HeavenAgentConfig, tools: List[BaseHeavenTool]) -> StateGraph:
    """
    Build a graph with tool execution capabilities.
    """
    from functools import partial
    # Import completion_node from the same directory
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from langgraph_hermes_legos import completion_node
    
    graph = StateGraph(ToolState)
    
    # Add nodes
    graph.add_node("agent", completion_node)
    graph.add_node("tools", partial(tool_execution_node, tools=tools))
    graph.add_node("error_handler", error_handler_node)
    
    # Add routing
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda x: "tools" if x.get("tool_calls") else "end",
        {
            "tools": "tools",
            "end": END
        }
    )
    graph.add_conditional_edges(
        "tools",
        lambda x: "error_handler" if x["error_count"] > 0 else "agent",
        {
            "error_handler": "error_handler",
            "agent": "agent"
        }
    )
    graph.add_conditional_edges(
        "error_handler",
        should_retry,
        {
            "retry": "agent",
            "fallback": "agent",
            "end": END
        }
    )
    
    return graph.compile()


def build_memory_enhanced_graph() -> StateGraph:
    """
    Build a graph with memory capabilities.
    """
    graph = StateGraph(MemoryState)
    
    # Add nodes
    graph.add_node("retrieve", memory_retrieval_node)
    graph.add_node("process", lambda x: x)  # Processing node
    graph.add_node("store", partial(memory_storage_node, storage_policy={"store_short_term": True}))
    
    # Add routing
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "process")
    graph.add_conditional_edges(
        "process",
        memory_routing,
        {
            "retrieve": "retrieve",
            "store": "store",
            "continue": END
        }
    )
    graph.add_edge("store", END)
    
    return graph.compile()


# === HELPER FUNCTIONS ===

def search_memory(memory: Dict[str, Any], query: str) -> List[Any]:
    """Simple memory search implementation"""
    results = []
    query_lower = query.lower()
    for key, value in memory.items():
        if query_lower in str(key).lower() or query_lower in str(value).lower():
            results.append({key: value})
    return results


def extract_information(messages: List[BaseMessage]) -> Dict[str, Any]:
    """Extract key information from messages"""
    info = {}
    for i, msg in enumerate(messages):
        if hasattr(msg, "content"):
            info[f"msg_{i}"] = {
                "type": type(msg).__name__,
                "content": msg.content[:100]  # First 100 chars
            }
    return info


def is_important(key: str, value: Any, policy: Dict[str, Any]) -> bool:
    """Determine if information is important enough for long-term storage"""
    importance_keywords = policy.get("importance_keywords", ["decision", "result", "conclusion"])
    return any(keyword in str(value).lower() for keyword in importance_keywords)


def needs_storage(messages: List[BaseMessage]) -> bool:
    """Determine if recent messages contain info worth storing"""
    if not messages:
        return False
    # Store if we have results or conclusions
    recent_content = " ".join([m.content for m in messages[-3:] if hasattr(m, "content")])
    return any(word in recent_content.lower() for word in ["result", "conclusion", "found", "discovered"])


def summarize_results(results: List[Any]) -> str:
    """Create a summary of results"""
    if not results:
        return "No results to summarize"
    return f"Aggregated {len(results)} results: {', '.join(str(r)[:50] for r in results[:3])}..."


def find_consensus(results: List[Any]) -> Any:
    """Find consensus among results"""
    if not results:
        return None
    # Simple implementation - return most common result
    from collections import Counter
    counter = Counter(str(r) for r in results)
    return counter.most_common(1)[0][0] if counter else None


# === USAGE EXAMPLE ===

async def demo_utility_legos():
    """Demonstrate utility LEGO blocks"""
    print("=== LangGraph Utility LEGOs Demo ===\n")
    
    # Just show that we can build the utility components
    print("1. Utility Nodes Available:")
    print("   - tool_execution_node")
    print("   - smart_router_node") 
    print("   - memory_retrieval_node")
    print("   - memory_storage_node")
    print("   - error_handler_node")
    print("   - aggregation_node")
    
    print("\n2. Utility Edges Available:")
    print("   - route_by_content")
    print("   - should_retry")
    print("   - memory_routing")
    
    print("\n3. State Types Available:")
    print("   - ToolState")
    print("   - RouterState")
    print("   - MemoryState")
    
    print("\n4. Graph Builders Available:")
    print("   - build_tool_enabled_graph")
    print("   - build_memory_enhanced_graph")
    
    print("\n=== Demo Complete ===")
    print("All utility LEGO blocks are ready for use!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_utility_legos())