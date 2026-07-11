# Heaven Framework Patterns

## Hook System Architecture

BaseHeavenAgent provides 9 lifecycle hook points. Each hook receives a HookContext with:
- `agent` - The agent instance
- `iteration` - Current iteration number
- `prompt` - The prompt being executed
- `response` - The response received
- `tool_name` - Name of tool being called
- `tool_args` - Arguments to the tool
- `tool_result` - Result from tool execution
- `error` - Exception if one occurred
- `data` - Shared dict for state between hooks

### Hook Execution Order

Single iteration:
```
BEFORE_RUN
  BEFORE_SYSTEM_PROMPT
  BEFORE_ITERATION
    BEFORE_TOOL_CALL (for each tool)
    AFTER_TOOL_CALL (for each tool)
    ON_BLOCK_REPORT (if WriteBlockReportTool used)
  AFTER_ITERATION
AFTER_RUN
ON_ERROR (if error occurs anywhere)
```

### Hook Registration

```python
from heaven_base import HookPoint, HookContext, HookRegistry

def my_hook(ctx: HookContext):
    # Access agent state
    agent = ctx.agent

    # Share data between hooks
    ctx.data['my_state'] = 'value'

    # Access tool information
    if ctx.tool_name == 'specific_tool':
        # Do something

# Register hook
agent.hooks.register(HookPoint.AFTER_TOOL_CALL, my_hook)
```

## Execution Pattern: Completion vs Hermes

### Completion (Single-Shot)

Used for: One-off questions, simple tasks, no iteration needed

```python
from heaven_base import BaseHeavenAgent

agent = MyAgent()
result = await agent.run("What is 2+2?")
```

### Hermes (Iterative Goal-Driven)

Used for: Complex tasks, multi-step reasoning, autonomous work

```python
from heaven_base.tool_utils.hermes_utils import hermes_step
from heaven_base.configs.hermes_config import HermesConfig

config = HermesConfig(
    args_template={
        'goal': 'Analyze the codebase and find bugs',
        'iterations': 10,
        'agent': my_agent,
        'history_id': 'session_123'
    }
)

result = hermes_step(hermes_config=config)
```

Key differences:
- Completion: Direct agent.run()
- Hermes: hermes_step() with iterations, block report handling, task tracking

## UnifiedChat Provider Patterns

### Provider Selection

```python
from heaven_base import UnifiedChat, ProviderEnum

# Anthropic (Claude)
chat = UnifiedChat.create(
    provider=ProviderEnum.ANTHROPIC,
    model="claude-3-5-sonnet-20241022"
)

# OpenAI (GPT-4, o-series)
chat = UnifiedChat.create(
    provider=ProviderEnum.OPENAI,
    model="gpt-4o"
)

# Google (Gemini)
chat = UnifiedChat.create(
    provider=ProviderEnum.GOOGLE,
    model="gemini-2.0-flash-exp"
)
```

### MiniMax via Anthropic API

MiniMax models use Anthropic-compatible API with different endpoint:

```python
chat = UnifiedChat.create(
    provider=ProviderEnum.ANTHROPIC,
    model="MiniMax-M2.5"
    # Automatically sets:
    # - api_key from MINIMAX_API_KEY
    # - anthropic_api_url = "https://api.minimax.io/anthropic"
)
```

### Thinking/Reasoning Support

Claude 3.7+:
```python
chat = UnifiedChat.create(
    provider=ProviderEnum.ANTHROPIC,
    model="claude-3-7-sonnet-20250219",
    thinking_budget=2000  # Budget in tokens
)
```

OpenAI o-series:
```python
chat = UnifiedChat.create(
    provider=ProviderEnum.OPENAI,
    model="o3-mini",
    thinking_budget=4096  # Maps to low/medium/high effort
)
```

Gemini 2.5+:
```python
chat = UnifiedChat.create(
    provider=ProviderEnum.GOOGLE,
    model="gemini-2.5-flash",
    thinking_budget=1000,
    include_thoughts=True
)
```

## ToolResult Patterns

ToolResult is an immutable dataclass. Use `replace()` for modifications:

```python
from heaven_base import ToolResult

# Create result
result = ToolResult(output="Success")

# Modify (creates new instance)
result2 = result.replace(error="Warning: something")

# Combine results (uses + operator)
combined = result1 + result2
# output, error, system concatenate
# base64_image: only one allowed (error if both present)
```

### Result Types

- `ToolResult` - Base result type
- `CLIResult` - Extends ToolResult for CLI rendering
- `ToolFailure` - Indicates failure (still a ToolResult)
- `ToolError` - Exception type (raise this in tools)

## MCP Tool Loading Pattern

BaseHeavenAgent loads MCP tools automatically if mcp_servers config provided:

```python
from heaven_base import HeavenAgentConfig

config = HeavenAgentConfig(
    name="MyAgent",
    mcp_servers={
        "server_name": {
            "command": "python",
            "args": ["-m", "my_mcp_server"],
            "env": {"API_KEY": "..."}
        }
    }
)
```

The agent will:
1. Connect to MCP server via MultiServerMCPClient
2. Get available tools via get_tools()
3. Convert each tool to BaseHeavenTool via create_heaven_tool_from_mcp_tool()
4. Add to agent's tool list

## History Persistence Pattern

All agent executions auto-persist to ~/.heaven/:

```python
from heaven_base.memory.conversations import start_chat, continue_chat

# Start new conversation
history = start_chat(agent=my_agent, user_message="Hello")

# Continue existing
history = continue_chat(history_id="abc123", user_message="More questions")

# Load for inspection
history = load_chat(history_id="abc123")
```

History includes:
- All messages (user, agent, tool)
- AgentStatus (goal, tasks, completed flag, extracted_content)
- Metadata (timestamps, agent name, etc)

## Schema Utilities Pattern

When creating tools with complex schemas, use the utilities:

```python
from heaven_base.baseheaventool import (
    fix_ref_paths,
    flatten_array_anyof,
    recursive_flatten,
    fix_empty_object_properties
)

# Fix $ref paths (#/$defs/ → #/defs/)
schema = fix_ref_paths(raw_schema)

# Handle nullable arrays
schema = flatten_array_anyof(schema)

# Deep normalization
schema = recursive_flatten(schema)

# Fix empty object properties
schema = fix_empty_object_properties(schema)
```

## LangGraph Integration Pattern

Use HeavenState + runners for workflows:

```python
from heaven_base.langgraph import HeavenState, completion_runner, hermes_runner
from langgraph.graph import StateGraph

# Define workflow
workflow = StateGraph(HeavenState)

# Add nodes using runners
workflow.add_node("step1", completion_runner)
workflow.add_node("step2", hermes_runner)

# Connect
workflow.add_edge("step1", "step2")
workflow.set_entry_point("step1")

# Execute
result = workflow.compile().invoke({
    "results": [],
    "context": {},
    "agents": {"my_agent": config}
})
```

## DovetailModel Chaining Pattern

Chain multiple Hermes executions:

```python
from heaven_base.configs.hermes_config import HermesConfig, DovetailModel, HermesConfigInput

# Define expected outputs and input mapping
dovetail = DovetailModel(
    expected_outputs=["analysis", "recommendations"],
    input_map={
        "goal": HermesConfigInput(
            source_key="analysis",
            transform=lambda x: f"Based on: {x}",
            required=True
        )
    }
)

# First execution
result1 = hermes_step(config1)
extracts = result1['agent_status']['extracted_content']

# Prepare next config
next_inputs = dovetail.prepare_next_config(extracts)
config2.args_template.update(next_inputs)

# Second execution
result2 = hermes_step(config2)
```

## Testing Patterns

Always test with multiple providers:

```python
import pytest
from heaven_base import UnifiedChat, ProviderEnum

@pytest.mark.parametrize("provider,model", [
    (ProviderEnum.ANTHROPIC, "claude-3-5-sonnet-20241022"),
    (ProviderEnum.OPENAI, "gpt-4o"),
    (ProviderEnum.GOOGLE, "gemini-2.0-flash-exp"),
])
def test_agent_with_provider(provider, model):
    agent = MyAgent.from_config(
        HeavenAgentConfig(provider=provider, model=model)
    )
    result = await agent.run("test")
    assert result is not None
```

## Critical Don'ts

1. **Don't modify hook signatures** - Breaking changes affect entire ecosystem
2. **Don't add provider-specific code to BaseHeavenAgent** - Use UnifiedChat
3. **Don't mutate ToolResult directly** - It's immutable, use replace()
4. **Don't skip history persistence** - It's automatic for good reason
5. **Don't bypass schema utilities** - They fix known compatibility issues
6. **Don't add fallback defaults in execution paths** - Fail loudly, fix properly
