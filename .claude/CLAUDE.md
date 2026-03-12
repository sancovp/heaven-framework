# HEAVEN Framework - STARSYSTEM

This is the **heaven-framework-repo** - the foundational agent infrastructure that all GNOSYS systems depend on.

## What This Is

HEAVEN Base Framework provides the core classes and systems for building autonomous AI agents:

- **BaseHeavenAgent** - The foundation class all agents inherit from (220KB core)
- **BaseHeavenTool** - The foundation class all tools inherit from (52KB core)
- **UnifiedChat** - Multi-provider LLM interface (Anthropic, OpenAI, Google, Groq, DeepSeek)
- **History System** - Conversation tracking with ADK session support
- **HermesConfig** - SDNA AriadneChain execution configuration
- **LangGraph Foundation** - Workflow orchestration primitives
- **MCP Integration** - 3-part system (client, tool converter, agent orchestrator)
- **Registry System** - Structured data storage
- **Prompt Engineering** - Reusable prompt blocks and composition
- **Tool Utils** - Support functions (hermes_step, exec_completion_style, context management)
- **CLI** - Command-line interface

## Architecture

```
heaven_base/
├── baseheavenagent.py       # Core agent class (9 hook points, tool loading, execution)
├── baseheaventool.py        # Core tool class (ToolResult, schema utilities)
├── unified_chat.py          # Multi-provider LLM abstraction
├── memory/
│   ├── history.py           # Conversation tracking (History, AgentStatus)
│   ├── heaven_history.py    # Collection manager
│   └── conversations.py     # start_chat, continue_chat, load_chat
├── configs/
│   ├── hermes_config.py     # HermesConfig (AriadneChain layer)
│   └── base_config.py       # BaseFunctionConfig
├── langgraph/
│   └── foundation.py        # HeavenState, HeavenNodeType, runners
├── tools/                   # Standard tools (WriteBlockReportTool, etc)
├── tool_utils/              # Support functions
│   ├── hermes_utils.py      # hermes_step (iterative execution)
│   ├── completion_runners.py # exec_completion_style
│   └── context_manager.py   # Weave/inject operations
├── utils/
│   ├── mcp_client.py        # MCP registry/discovery/execution
│   ├── mcp_tool_converter.py # MCP→BaseHeavenTool
│   └── mcp_agent_orchestrator.py # Server→Agent
├── registry/                # Data storage system
├── prompts/                 # Prompt blocks and variables
└── cli/                     # Command-line interface
```

## Key Concepts

### The Hook System

BaseHeavenAgent provides **9 hook points** for lifecycle interception:

1. `BEFORE_RUN` - Before agent execution starts
2. `AFTER_RUN` - After agent execution completes
3. `BEFORE_ITERATION` - Before each iteration in a loop
4. `AFTER_ITERATION` - After each iteration
5. `BEFORE_TOOL_CALL` - Before any tool is called
6. `AFTER_TOOL_CALL` - After tool execution
7. `BEFORE_SYSTEM_PROMPT` - Before system prompt is assembled
8. `ON_BLOCK_REPORT` - When WriteBlockReportTool is used (task tracking)
9. `ON_ERROR` - When an error occurs

This enables CAVE integration, OMNISANC state management, and autonomous loop control.

### Execution Patterns

**Completion Style** (single-shot):
```python
from heaven_base import BaseHeavenAgent, HeavenAgentConfig

agent = MyAgent()
result = await agent.run("What is the meaning of life?")
```

**Hermes Style** (iterative goal-driven):
```python
from heaven_base.tool_utils.hermes_utils import hermes_step
from heaven_base.configs.hermes_config import HermesConfig

config = HermesConfig(...)
result = hermes_step(hermes_config=config)
# Handles block reports, iteration limits, continuation
```

### Multi-Provider LLM Support

UnifiedChat abstracts provider differences:

- **Anthropic**: MiniMax models via `anthropic_api_url`, thinking budget for Claude 3.7+
- **OpenAI**: Reasoning effort for o-series, `use_responses_api` for extended thinking
- **Google**: `thinking_budget` and `include_thoughts` for Gemini 2.5+
- **Groq & DeepSeek**: Standard configurations

API keys resolved via `EnvConfigUtil.get_env_value()` with `DynamicString` pattern.

### MCP Integration

Three subsystems for Model Context Protocol:

1. **MCP Client** (`utils/mcp_client.py`) - Registry, discovery, execution, session management
2. **Tool Converter** (`utils/mcp_tool_converter.py`) - Converts MCP tools to BaseHeavenTool
3. **Agent Orchestrator** (`utils/mcp_agent_orchestrator.py`) - Creates agents from MCP servers

BaseHeavenAgent loads MCP tools via `_load_mcp_tools()` using `langchain_mcp_adapters`.

### History & Memory

The History system tracks conversations with:
- LangChain messages (SystemMessage, HumanMessage, AIMessage, ToolMessage)
- Google ADK events via `adk_session`
- AgentStatus for task tracking (goal, task_list, current_task, completed, extracted_content)
- Persistence to `~/.heaven/` directory
- ConversationManager for session continuity

### SDNA Integration

HermesConfig is the **AriadneChain** layer in SDNA architecture:
- Context preparation before PoimandresChain execution
- DovetailModel for chaining executions (expected_outputs, input_map)
- Variable substitution via `to_command_data()`
- Supports all hermes_step parameters (goal, iterations, agent, history_id, etc)

## Development Patterns

### Creating a Custom Agent

```python
from heaven_base import BaseHeavenAgent, HeavenAgentConfig

class MyAgent(BaseHeavenAgent):
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="MyAgent",
            system_prompt="You are a helpful assistant.",
            model="gpt-4",
            temperature=0.7
        )
```

### Creating a Custom Tool

```python
from heaven_base import BaseHeavenTool, ToolArgsSchema, ToolResult

class MyToolArgsSchema(ToolArgsSchema):
    arguments = {
        'input': {'type': 'str', 'description': 'Input data', 'required': True}
    }

class MyTool(BaseHeavenTool):
    name = "my_tool"
    description = "Does something useful"
    args_schema = MyToolArgsSchema

    def _run(self, input: str) -> ToolResult:
        # Do work
        return ToolResult(output=f"Processed: {input}")
```

### Using Hooks

```python
from heaven_base import HookPoint, HookContext

def my_hook(ctx: HookContext):
    print(f"Tool called: {ctx.tool_name}")
    ctx.data['my_state'] = 'something'

agent.hooks.register(HookPoint.AFTER_TOOL_CALL, my_hook)
```

## Testing

```bash
# Run all tests
pytest tests/

# Test specific component
pytest tests/test_heaven_components.py

# Test with specific provider
pytest tests/test_basic_agent_all_runners.py
```

## Version

Current version: **0.1.20**

Published to PyPI as `heaven-framework`

## Dependencies

This repo is the **BASE** - everything depends on it:
- sanctuary-revolution (CAVE, game harness)
- SDNA (sanctuary-dna agent framework)
- All MCP servers using BaseHeavenTool
- All agents using BaseHeavenAgent

Changes here affect the entire ecosystem. Test thoroughly.

## STARSYSTEM Workflow

Before working in this repo:

1. `orient()` - Check project status
2. Start starlog session
3. Work with tests (TDD when possible)
4. Log insights/bugs to debug diary
5. End starlog when done

Equip persona: `starship-pilot-heaven-framework-repo` (to be created)

## Key Files

- `baseheavenagent.py` - Start here for agent architecture
- `baseheaventool.py` - Start here for tool development
- `unified_chat.py` - Provider abstraction
- `tool_utils/hermes_utils.py` - Core iterative execution
- `configs/hermes_config.py` - SDNA AriadneChain layer
- `langgraph/foundation.py` - Workflow primitives

## Critical Rules

1. **Never break the hook system** - All 9 hook points must continue to work
2. **Provider abstraction must remain** - UnifiedChat is how we support multiple LLMs
3. **ToolResult is immutable** - Use `replace()` for modifications
4. **History must persist** - Conversation continuity depends on it
5. **MCP integration is sacred** - Don't break tool loading
6. **Test before committing** - This is foundational infrastructure

## CartON Knowledge

Full architecture documented in CartON collection: `Architecture_Heaven_Framework_Collection`

Activate this collection to load complete understanding of all 11 subsystems.
