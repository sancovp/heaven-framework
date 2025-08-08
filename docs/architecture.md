# HEAVEN Architecture

## Overview

HEAVEN (Hierarchical, Embodied, Autonomously Validating Evolution Network) is a metaprogrammatic agent framework designed for self-modifying, self-improving AI systems. The architecture enables agents, tools, prompts, and code to generate each other in a recursive, evolutionary manner.

## Core Design Principles

### 1. Metaprogrammatic Design
Everything in HEAVEN can generate everything else:
- Agents can create new agents
- Tools can generate new tools
- Prompts can create new prompts
- Code can write new code

### 2. Container-Based Evolution
HEAVEN uses a three-container architecture for safe evolution:

```
mind_of_god (MoG)     →  Production (frozen, stable)
    ↓
image_of_god (IoG)    →  User's custom fork
    ↓
creation_of_god (CoG) →  Development playground
```

### 3. Event-Driven Communication
All components communicate through standardized HEAVEN events:
- `SYSTEM_MESSAGE`: Framework notifications
- `USER_MESSAGE`: Human input
- `AGENT_MESSAGE`: Agent responses
- `TOOL_USE`: Tool invocations
- `TOOL_RESULT`: Tool outputs
- `ERROR`: Error conditions

## Framework Backend Selection

HEAVEN agents can run on different backends, controlled by flags on the agent:

### LangChain Backend (Default)
```python
# Default - adk=False
agent = BaseHeavenAgent(config, unified_chat, adk=False)
# Or for replicants
class MyReplicant(BaseHeavenAgentReplicant):
    adk = False  # Class attribute
```
- Uses LangChain's ChatModel and BaseMessage
- Uses LangChain's tool system
- Does NOT use LangChain's agent or agent executor

### Google ADK Backend
```python
# Enable ADK mode
agent = BaseHeavenAgent(config, unified_chat, adk=True)
# Or for replicants
class MyReplicant(BaseHeavenAgentReplicant):
    adk = True  # Class attribute
```
- Uses full ADK stack
- Passes ADKAgent into ADK Runner
- Compiles BaseHeavenTools from LangChain format to ADK format
- Stricter typing requirements (may need debugging for new tools)

### Uni API Backend
For containerized universal API endpoints (advanced usage).

## OmniTool - The Universal Adapter

OmniTool is the **ONLY** exception to the "code calls functions, not tools" rule. It's the bridge between dynamic tool selection (semantic) and static function calls (programmatic).

### What is OmniTool?

OmniTool is the universal tool executor that can:
1. Discover all available tools
2. Get tool schemas/info
3. Execute any registered tool dynamically

### OmniTool Usage

```python
from computer_use_demo.tools.base.utils.omnitool import omnitool

# List all available tools
result = omnitool(list_tools=True)
# Returns: String of dict with 'available_tools' list

# Get info about a specific tool
result = omnitool('NetworkEditTool', get_tool_info=True)
# OR snake_case works too:
result = omnitool('network_edit_tool', get_tool_info=True)

# Execute a tool with parameters
result = omnitool('NetworkEditTool', parameters={
    'command': 'view',
    'path': '/home/GOD/core/some_file.py',
    'command_arguments': {}
})

# BashTool example
result = omnitool('bash_tool', parameters={
    'command': 'ls -la /tmp'
})
```

### Why OmniTool is Special

OmniTool is allowed to break the rule because it's the **universal adapter**:
- It bridges semantic tool selection with programmatic execution
- It allows dynamic tool discovery and execution
- It's used for meta-operations on the tool ecosystem
- It's the ONLY code that should import and call tool objects directly

## Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     HEAVEN Framework                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Agents    │  │    Tools    │  │   Events    │     │
│  │             │◄─►│             │◄─►│             │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│         │                 │                 │            │
│         │                 ▼                 │            │
│         │           ┌──────────┐           │            │
│         │           │ OmniTool │           │            │
│         │           │(Universal│           │            │
│         │           │ Adapter) │           │            │
│         │           └──────────┘           │            │
│         │                 │                 │            │
│  ┌──────▼─────────────────▼─────────────────▼──────┐    │
│  │              UnifiedChat Interface               │    │
│  │  (Multi-provider LLM abstraction layer)          │    │
│  └──────────────────────┬───────────────────────────┘   │
│                         │                                │
│     Backend selection via agent flags:                   │
│     ┌─────────────────────────────────────┐            │
│     │ if agent.adk:                        │            │
│     │     → Google ADK Backend             │            │
│     │ else:                                │            │
│     │     → LangChain Backend (default)    │            │
│     └─────────────────────────────────────┘            │
│                                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Persistent Systems                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Registry │  │  Neo4j   │  │  Heaven  │     │    │
│  │  │  System  │  │  Graph   │  │  Store   │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Core Components

### BaseHeavenAgent

The foundation of all agents in HEAVEN:

```python
class BaseHeavenAgent:
    def __init__(self, config: HeavenAgentConfig, unified_chat: UnifiedChat, 
                 history: History = None, history_id: str = None, adk: bool = False):
        self.config = config
        self.unified_chat = unified_chat
        self.history = history or History.load_from_id(history_id)
        self.adk = adk  # Framework selection flag
        
    async def run(self, prompt: str) -> dict:
        # Routes based on adk flag
        if self.adk:
            return await self.run_adk(prompt)
        else:
            return await self.run_langchain(prompt)
```

### BaseHeavenTool

The foundation for all tools:

```python
class BaseHeavenTool:
    name: str              # Tool identifier
    description: str       # LLM-facing description
    args_schema: Type      # Pydantic validation schema
    func: Callable         # Underlying function
    is_async: bool        # Sync/async flag
```

**Critical Rules**:
1. **"Code calls functions, not tools... EXCEPT OmniTool"**
2. Tools are wrappers for agent use only
3. Python scripts call the underlying `func` directly
4. Never import tools in scripts (except OmniTool)
5. BaseHeavenTool._run and _arun should NEVER be overridden
6. BaseHeavenTool._arun is the UNIVERSAL RUNNER that resolves sync/async

### Tool Call vs Function Call

```python
# WRONG - Don't call tools from code
from heaven_base.tools.my_tool import MyTool
result = MyTool._run(args)  # NO!
result = MyTool.func(args)   # Also NO! (no sync/async resolution)

# RIGHT - Call the underlying function directly
from my_module import my_function
result = my_function(args)  # YES!

# RIGHT - Agents use tools
config = HeavenAgentConfig(tools=[MyTool])

# EXCEPTION - OmniTool is allowed
from computer_use_demo.tools.base.utils.omnitool import omnitool
result = omnitool('MyTool', parameters=args)  # OK! OmniTool is special
```

### UnifiedChat

Provides consistent interface across LLM providers:

```python
class UnifiedChat:
    def __init__(self, provider: ProviderEnum = None):
        self.provider = provider or self._detect_provider()
        
    async def complete(self, messages: List[dict]) -> dict:
        # Normalize to OpenAI format internally
        # Route to appropriate provider
        # Return standardized response
```

Supported providers:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude 3 family)
- Google (Gemini models)
- DeepSeek
- Groq
- Local (Ollama)

### Registry System

Persistent storage for shared data between agents:

```python
# Get registry instance
registry = RegistryFactory.get_registry("project_name")

# Store data
registry.store("key", data)

# Retrieve data
data = registry.retrieve("key")

# Cross-registry references
registry.store("ref", {
    "type": "cross_registry_reference",
    "registry": "other_registry",  
    "key": "other_key"
})
```

## The Hermes System

Hermes is HEAVEN's orchestration layer for running registered agents:

### Core Functions

```python
# use_hermes() - Returns string
result = await use_hermes(
    target_container="mind_of_god",  # Where to run
    source_container="mind_of_god",  # Where from
    goal="Your goal",
    agent="registered_agent_name",  # Or None for default
    iterations=3,
    history_id=None,
    continuation=False
)

# use_hermes_dict() - Returns dict
result = await use_hermes_dict(...)

# hermes_step() - Returns dict with block report handling
result = await hermes_step(...)
```

### How Hermes Works

1. Runs agents in target containers from source containers
2. Uses `docker exec` and socket mounts for communication
3. Internally orchestrates `run` vs `continue_iterations`
4. Handles block reports and error states

### Agent Registration

Agents must be registered to use Hermes:
```
base/agents/{agent_name}/
├── config_file.py      # Agent configuration
├── test_file.py        # Tests
└── memories/           # Auto-created
```

Tools that auto-register agents:
- `AgentMakerTool`
- `ConstructHermesConfigTool` (when `make_tool=True`)

## Agent Hierarchy

```
OverallDeity (creates new species)
    ↓
Deity (creates new prompt templates)
    ↓
Progenitor (fills templates for specific domains)
    ↓
SuperOrchestrator (calls any orchestrator)
    ↓
DomainOrchestrator (calls subdomain managers)
    ↓
SubdomainManager (calls workers)
    ↓
SubdomainWorker (does the actual work)
```

### Hierarchy Construction

Hierarchies can be built:
1. Manually by creating configs
2. Automatically via: ToolMaker → EvolutionaryIntent → AgentMaker → ConstructHermesConfig

DomainOrchestrators and SubdomainManagers are metaprogrammatic:
- DomainOrchestrators get tools to search/list/call SubdomainManagers
- SubdomainManagers get all ProduceXTools in their subdomain
- Workers are any agent with a ProduceXTool made for them

## Container Workflow

### Evolution Process

1. **MoG** instructs **IoG** to evolve
2. **IoG** uses EvolveTools on **CoG**
3. **CoG** runs experiments (can fail safely)
4. On success: CoG state replaces IoG
5. Old IoG backed up to heaven-backups

### Why Containers Instead of Git?

From the cheatsheets:
- **Hidden file generation**: Agents create files through tools without awareness
- **Hot reload requirement**: Instant updates for self-modification
- **Atomic state capture**: Entire container states vs individual files
- **Agent-friendly abstractions**: Agents don't need to understand git

## Memory and Context

### History Management

```python
# Start new conversation
result = await agent.run("Message")
history_id = result["history_id"]

# Continue from history
agent2 = BaseHeavenAgent(config, unified_chat, history_id=history_id)

# Or use continue_iterations
result = await agent.continue_iterations(
    history_id=history_id,
    continuation_iterations=3,
    continuation_prompt="Continue..."
)
```

### Prompt Suffix Blocks

Dynamic context injection types:

```python
config = HeavenAgentConfig(
    prompt_suffix_blocks=[
        # 1. File path
        "path=/path/to/file.md",
        
        # 2. Registry variable
        "registry_heaven_variable={'registry_name':'cheatsheets','key':'heaven_cs'}",
        
        # 3. Heaven variable (from Python module)
        "heaven_variable={'path':'/path/to/module.py','variable_name':'MY_VAR'}",
        
        # 4. Dynamic function call (zero-arg function)
        "dynamic_call={'path':'module','func':'get_context'}",
        
        # 5. Prompt block by name (fallback)
        "MyCustomBlock"
    ]
)
```

## The Maker System

Automated scaffolding for new components:

1. **ToolMakerTool**: Scaffolds Tool class, ArgsSchema, and test stub
2. **AgentMakerTool**: Scaffolds agent config and test harness
3. **ConstructHermesConfigTool**: Creates HermesConfig JSON and optional `Produce<Deliverable>Tool`

This creates a **program-specific functional programming library** when used with `make_tool=True`.

## Important Distinctions

### BaseHeavenAgent vs BaseHeavenAgentReplicant

- **BaseHeavenAgent**: Takes HeavenAgentConfig at init
- **BaseHeavenAgentReplicant**: Config is built-in, initialized without args

```python
# BaseHeavenAgent - needs config
agent = BaseHeavenAgent(config, unified_chat)

# BaseHeavenAgentReplicant - no config needed
from heaven_base.agents.my_replicant import MyReplicant
replicant = MyReplicant()  # Config is internal
```

### Running Agents

```python
# Dynamic agents (not registered)
agent.run(prompt)
agent.continue_iterations(...)

# Registered agents via Hermes
use_hermes(goal="...", agent="registered_name")
```

### Runners and Execution

From the cheatsheets:
- `agent.run()` - Runs on history_id attr, outputs new history_id
- `agent.continue_iterations()` - Resumes from history_id in agent mode
- `use_hermes()` - Internally orchestrates run vs continue_iterations, returns string
- `hermes_step()` - Uses use_hermes internally, handles block reports, returns dict

## The Agent-as-REPL Model

Agents are sophisticated REPLs at the semantic-programmatic interface:

1. **Read**: Semantic input from users/systems
2. **Evaluate**: Transform semantics to programmatic values
3. **Process**: Execute programs with those values  
4. **Loop**: Return semantic output and continue

This creates a bidirectional translation layer where:
- Agents predict translations between intent and values
- Execute programs with those values
- Translate results back to semantic understanding

## Performance and Safety

### Concurrency
- Multiple agents can run in parallel via asyncio.gather()
- Registry handles concurrent access
- History is thread-safe

### Error Handling
- Tools return ToolResult or ToolError
- Agents can return block reports when stuck
- Container isolation prevents cascade failures

### Cleanup
- Nightly cleanup moves old versions to heaven-maybe-trash
- Histories preserved in heaven-histories repo
- IoG snapshots saved before replacement

## Key Takeaways

1. **Framework selection via agent flags** (`adk=True/False`)
2. **OmniTool is the ONLY exception** to "code calls functions not tools"
3. **Tools are for agents only** - code calls functions directly
4. **Containers enable safe evolution** and hot reload
5. **Hermes orchestrates registered agents** across containers
6. **Everything can generate everything else** (metaprogrammatic)
7. **BaseHeavenTool._arun is universal** - handles sync/async resolution