# HEAVEN LangGraph LEGOs

HEAVEN provides composable LangGraph building blocks (LEGOs) for creating agent-driven workflows. These components integrate seamlessly with HEAVEN agents, Hermes orchestration, and the broader HEAVEN ecosystem.

## Overview

LangGraph LEGOs are pre-built, tested components that you can compose into complex workflows:

- **Nodes**: Execute agents, process data, manage context
- **Edges**: Control flow between nodes with conditions
- **States**: Manage data flow through the workflow
- **Runners**: Execute different types of agent operations

## Quick Start

```python
from heaven_base.langgraph.foundation import HeavenState, completion_runner
from heaven_base.langgraph.hermes_legos import build_simple_hermes_graph
from heaven_base.baseheavenagent import HeavenAgentConfig
from heaven_base.unified_chat import ProviderEnum

# Create an agent
agent = HeavenAgentConfig(
    name="MyAgent",
    system_prompt="You are a helpful assistant",
    provider=ProviderEnum.OPENAI,
    model="gpt-4"
)

# Build a simple workflow
graph = build_simple_hermes_graph(agent)

# Execute
state = {"messages": [], "current_goal": "Hello!", "agent_config": agent}
result = await graph.ainvoke(state)
```

## Core Components

### 1. Foundation (`foundation.py`)
Core state management and runners for universal workflows.

### 2. Hermes LEGOs (`hermes_legos.py`)
Specialized components for Hermes orchestration workflows.

### 3. Utility LEGOs (`utility_legos.py`)
General-purpose utilities for workflow construction.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                LangGraph Workflow                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │  Node   │───▶│  Node   │───▶│  Node   │             │
│  │ (LEGO)  │    │ (LEGO)  │    │ (LEGO)  │             │
│  └─────────┘    └─────────┘    └─────────┘             │
│       │              │              │                   │
│       ▼              ▼              ▼                   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              State Management                       │ │
│  │        (HeavenState / HermesState)                  │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              HEAVEN Integration                      │ │
│  │    • BaseHeavenAgent execution                      │ │
│  │    • Hermes orchestration                           │ │
│  │    • Event system                                   │ │
│  │    • Registry access                                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Component Documentation

### [Foundation Components](foundation.md)
- HeavenState
- HeavenNodeType
- completion_runner
- hermes_runner
- Core state management

### [Hermes LEGOs](hermes-legos.md)
- HermesState
- completion_node
- hermes_node
- chain_controller_node
- Graph builders

### [Utility LEGOs](utility-legos.md)
- Context management
- Result extraction
- Dynamic functions
- OmniTool integration

## Common Patterns

### Simple Agent Execution
```python
from heaven_base.langgraph.hermes_legos import build_simple_hermes_graph

graph = build_simple_hermes_graph(agent)
result = await graph.ainvoke(initial_state)
```

### Iterative Processing
```python
from heaven_base.langgraph.hermes_legos import build_iterative_hermes_graph

graph = build_iterative_hermes_graph(agent, max_iterations=5)
result = await graph.ainvoke(initial_state)
```

### Agent Chains
```python
from heaven_base.langgraph.hermes_legos import build_agent_chain_graph

graph = build_agent_chain_graph([agent1, agent2, agent3])
result = await graph.ainvoke(chain_state)
```

### Custom Workflows
```python
from langgraph.graph import StateGraph, START, END
from heaven_base.langgraph.foundation import completion_runner

graph = StateGraph(HeavenState)
graph.add_node("process", lambda state: completion_runner(state, prompt="Process this"))
graph.add_edge(START, "process")
graph.add_edge("process", END)

compiled = graph.compile()
result = await compiled.ainvoke(state)
```

## Best Practices

### 1. State Design
- Use appropriate state types (HeavenState vs HermesState)
- Include all necessary context in state
- Design for data flow between nodes

### 2. Node Composition
- Keep nodes focused on single responsibilities
- Use existing LEGOs before creating custom nodes
- Handle errors gracefully within nodes

### 3. Edge Logic
- Use conditional edges for dynamic flow control
- Implement clear termination conditions
- Avoid infinite loops

### 4. Agent Integration
- Store agents in state for reuse
- Use appropriate execution methods (completion vs hermes)
- Leverage HEAVEN events for communication

## Examples

See the `/examples` directory for complete working examples:

- `langgraph_demo.py` - Comprehensive examples of all patterns
- `simple_workflow.py` - Basic workflow construction
- `iterative_processing.py` - Advanced conditional flows
- `agent_chains.py` - Multi-agent orchestration

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure heaven-base is properly installed
2. **State Mismatches**: Use correct state types for your LEGOs
3. **Agent Errors**: Verify agent configurations are valid
4. **Flow Control**: Check conditional edge logic for infinite loops

### Debug Tips

- Use print statements in custom nodes to trace execution
- Inspect state at each step
- Test individual LEGOs before composing complex workflows
- Use the examples as reference implementations

## Integration with HEAVEN

LangGraph LEGOs integrate seamlessly with:

- **BaseHeavenAgent**: Execute agents within workflows
- **Hermes System**: Orchestrate cross-container operations
- **Registry System**: Access shared data and configurations
- **Event System**: Communicate between workflow components
- **Tool System**: Use OmniTool for dynamic tool execution

This makes it possible to build sophisticated, self-modifying agent workflows that leverage the full power of the HEAVEN ecosystem.