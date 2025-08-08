# Quick Start Guide

This guide will get you up and running with HEAVEN in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heaven-framework-repo.git
cd heaven-framework-repo

# Install dependencies
pip install -r requirements.txt

# Set up the package
pip install -e .
```

## Basic Concepts

### Agent Hierarchy

HEAVEN agents are organized in a hierarchy:

- **SubdomainWorker**: Lords over a specific process with tools to complete it
- **SubdomainManager**: Calls subdomain workers within its subdomain
- **DomainOrchestrator**: Calls subdomain managers within its domain
- **SuperOrchestrator**: Can call any orchestrator across domains
- **Progenitor**: Creates system prompts for new agents
- **Deity**: Creates new system prompt species (templates)

### Running Agents

There are two main ways to run agents in HEAVEN:

1. **Direct Execution** (for dynamically created agents):
```python
agent = BaseHeavenAgent(config)
result = await agent.run("Your prompt here")
```

2. **Hermes Runners** (for registered agents):
```python
from heaven_base.tool_utils.hermes_utils import use_hermes

result = await use_hermes(
    goal="Your goal here",
    agent="registered_agent_name",
    iterations=3
)
```

## Your First Agent

Let's create a simple HEAVEN agent:

```python
import asyncio
from heaven_base.baseheavenagent import BaseHeavenAgent, HeavenAgentConfig
from heaven_base.unified_chat import ProviderEnum, UnifiedChat
from heaven_base.memory.history import History

# Configure the agent
config = HeavenAgentConfig(
    name="MyFirstAgent",
    system_prompt="You are a helpful AI assistant running in HEAVEN.",
    tools=[],  # No tools for now
    provider=ProviderEnum.OPENAI,
    model="gpt-4",
    temperature=0.7
)

async def main():
    # Create the agent
    agent = BaseHeavenAgent(
        config, 
        UnifiedChat(), 
        history=History(messages=[])
    )
    
    # Run the agent
    result = await agent.run("What is HEAVEN?")
    
    # Print the result
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            print(f"{msg.__class__.__name__}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Your First Tool

Tools extend agent capabilities. Here's how to create one:

```python
from heaven_base.baseheaventool import BaseHeavenTool, ToolArgsSchema

class CalculatorArgsSchema(ToolArgsSchema):
    arguments = {
        'expression': {
            'type': 'str',
            'description': 'Mathematical expression to evaluate',
            'required': True
        }
    }

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # In production, use a proper expression parser
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

class CalculatorTool(BaseHeavenTool):
    name = "CalculatorTool"
    description = "Evaluates mathematical expressions"
    args_schema = CalculatorArgsSchema
    func = calculate  # The underlying function
    is_async = False  # Set to True if func is async
```

## Using Tools with Agents

```python
# Add the tool to your agent config
config = HeavenAgentConfig(
    name="MathAgent",
    system_prompt="You are a math assistant. Use the calculator for computations.",
    tools=[CalculatorTool],  # Add the tool here
    provider=ProviderEnum.OPENAI,
    model="gpt-4"
)

# The agent can now use the tool
agent = BaseHeavenAgent(config, UnifiedChat(), history=History(messages=[]))
result = await agent.run("What is 15 * 23 + 42?")
```

## Working with History

HEAVEN provides powerful history management:

```python
# Option 1: Start fresh
agent = BaseHeavenAgent(config, UnifiedChat(), history=History(messages=[]))
result = await agent.run("First message")
history_id = result.get("history_id")

# Option 2: Continue from existing history
agent2 = BaseHeavenAgent(config, UnifiedChat(), history_id=history_id)
result2 = await agent2.run("Follow-up message")

# Option 3: Use continue_iterations for complex workflows
result3 = await agent2.continue_iterations(
    history_id=history_id,
    continuation_iterations=3,
    continuation_prompt="Continue working on the problem"
)
```

## Agent Modes

Agents can run in different modes:

```python
# Regular mode - single response
result = await agent.run("Hello")

# Agent mode - multiple iterations
result = await agent.run(
    prompt="agent goal=Build a calculator app, iterations=5"
)
```

## Important Notes

### Tool Calls vs Function Calls

From the cheatsheets: **"Code calls functions, not tools... EXCEPT OmniTool"**

- When writing Python code, call the underlying function directly
- Tools are wrappers for agents to use
- Only agents should use tool objects
- In scripts, import and call the function, not the tool

```python
# WRONG in Python script:
from heaven_base.tools.calculator_tool import CalculatorTool
result = CalculatorTool._run(expression="2+2")  # Don't do this!

# RIGHT in Python script:
from my_module import calculate
result = calculate("2+2")  # Call the function directly

# RIGHT for agents:
config = HeavenAgentConfig(tools=[CalculatorTool])  # Agent uses the tool
```

### BaseHeavenAgent vs BaseHeavenAgentReplicant

- **BaseHeavenAgent**: Takes a HeavenAgentConfig at initialization
- **BaseHeavenAgentReplicant**: Has its config built-in, can be initialized without arguments

```python
# BaseHeavenAgent - needs config
agent = BaseHeavenAgent(config)

# BaseHeavenAgentReplicant - config is internal
from heaven_base.agents.my_replicant import MyReplicant
replicant = MyReplicant()  # No config needed
replicant.run("Hello")
```

## Next Steps

1. Explore the [Agent Development Guide](agents/README.md)
2. Learn about [Tool Creation](tools/README.md)
3. Understand the [Hermes System](hermes/README.md)
4. Dive into [Advanced Topics](advanced/README.md)

## Common Patterns

### Using Prompt Suffix Blocks

Add context to your agent dynamically:

```python
config = HeavenAgentConfig(
    name="ContextAwareAgent",
    system_prompt="Base prompt",
    prompt_suffix_blocks=[
        # Read from file
        "path=/home/GOD/context.md",
        
        # Get from registry
        "registry_heaven_variable={'registry_name':'cheatsheets','key':'agent_hierarchy'}",
        
        # Call a function
        "dynamic_call={\"path\":\"my_module\",\"func\":\"get_context\"}"
    ]
)
```

### Error Handling

Always wrap agent operations in try-catch:

```python
try:
    result = await agent.run("Do something complex")
except Exception as e:
    print(f"Agent failed: {e}")
    # Agents can return block reports when stuck
    if "BLOCKED" in str(e):
        print("Agent needs help!")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the right directory and have installed dependencies
2. **API Keys**: Set environment variables for your LLM providers
3. **Tool Errors**: Check that your tool's args_schema matches the function signature
4. **History Not Found**: Ensure the history_id exists and is accessible

For more help, see the full documentation or check the cheatsheets registry.