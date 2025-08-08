# HEAVEN Framework Documentation

**HEAVEN** - **H**ierarchical, **E**mbodied, **A**utonomously **V**alidating **E**volution **N**etwork

## What is HEAVEN?

HEAVEN is a metaprogrammatic agent framework where prompts, tools, agents, and code can all generate each other. It provides a complete ecosystem for building self-modifying, self-improving AI agent systems with cross-framework compatibility.

## Table of Contents

- [Quick Start](quickstart.md)
- [Core Architecture](architecture.md)
- [Agent Development](agents/README.md)
  - [BaseHeavenAgent](agents/base-agent.md)
  - [Agent Configuration](agents/configuration.md)
  - [Replicants](agents/replicants.md)
- [Tool Development](tools/README.md)
  - [BaseHeavenTool](tools/base-tool.md)
  - [Tool Schemas](tools/schemas.md)
  - [Tool Composition](tools/composition.md)
- [Hermes System](hermes/README.md)
  - [Hermes Runners](hermes/runners.md)
  - [Hermes Configs](hermes/configs.md)
  - [Orchestration](hermes/orchestration.md)
- [Registry System](registry/README.md)
- [Advanced Topics](advanced/README.md)
  - [Container Architecture](advanced/containers.md)
  - [Evolution System](advanced/evolution.md)
  - [Metaprogramming](advanced/metaprogramming.md)

## Core Components

### The HEAVEN Container Architecture

HEAVEN operates through a sophisticated container system:

- **mind_of_god**: Frozen stable core (production)
- **image_of_god**: User's custom fork (last working version)
- **creation_of_god**: Development environment (experimental)
- **heaven_store**: Persistent storage for all code and registries
- **neo4j**: Graph database for registry metadata and computation

### Key Framework Classes

#### BaseHeavenAgent
The foundation class for all HEAVEN agents with cross-framework compatibility (LangChain and Google ADK).

#### BaseHeavenTool  
Base class for creating reusable tools with standardized interfaces.

#### UnifiedChat
Multi-provider LLM interface supporting OpenAI, Anthropic, Google, DeepSeek, Groq, and more.

#### HEAVEN Events
Standardized event format enabling seamless agent-to-agent communication.

## Philosophy

HEAVEN operates on several key principles:

1. **Metaprogrammatic Design**: Agents can generate other agents, tools, and even modify their own code
2. **Cross-Framework Compatibility**: Works seamlessly with LangChain, Google ADK, and other frameworks
3. **Event-Driven Architecture**: All communication uses standardized HEAVEN events
4. **Self-Evolution**: Agents can improve themselves through the evolution system
5. **Hierarchical Organization**: Agents are organized in hierarchies from workers to orchestrators

## Quick Examples

### Creating a Simple Agent

```python
from heaven_base import BaseHeavenAgent, HeavenAgentConfig

config = HeavenAgentConfig(
    name="MyAgent",
    system_prompt="You are a helpful assistant",
    tools=[],
    provider=ProviderEnum.OPENAI,
    model="gpt-4"
)

agent = BaseHeavenAgent(config)
result = await agent.run("Hello, HEAVEN!")
```

### Creating a Tool

```python
from heaven_base import BaseHeavenTool, ToolArgsSchema

class MyToolArgsSchema(ToolArgsSchema):
    arguments = {
        'input': {
            'type': 'str',
            'description': 'Input text',
            'required': True
        }
    }

class MyTool(BaseHeavenTool):
    name = "MyTool"
    description = "A simple tool"
    args_schema = MyToolArgsSchema
    func = my_function  # The underlying function
```

## The Agent-as-REPL Model

Agents in HEAVEN can be understood as sophisticated REPLs (Read-Evaluate-Print-Loop) that operate at the interface between semantic and programmatic domains:

1. **Read**: Takes semantic input from users or systems
2. **Evaluate**: Transforms semantics into programmatic values and operations
3. **Process**: Executes programs with those values
4. **Loop**: Returns semantic output and continues the cycle

This creates a bidirectional translation layer between human intent and computational action.

## Getting Started

See the [Quick Start Guide](quickstart.md) to begin building with HEAVEN.

## Support

For questions and issues, refer to the troubleshooting guide or contact the development team.