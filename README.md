# HEAVEN Base Framework

**H**ierarchical, **E**mbodied, **A**utonomously **V**alidating **E**volution **N**etwork

HEAVEN Base is the foundational framework for building autonomous AI agents with cross-framework compatibility, event-driven architecture, and self-modifying capabilities. This is the core library that powers the HEAVEN metaprogrammatic agent framework where prompts, tools, agents, and code can all generate each other.

## Features

- **Cross-Framework Support**: Works with both LangChain and Google ADK backends
- **Unified Event System**: Standardized HEAVEN events for consistent agent communication
- **Flexible Agent Architecture**: Build custom agents by extending `BaseHeavenAgent`
- **Tool System**: Create reusable tools by extending `BaseHeavenTool`
- **History Management**: Advanced history tracking with event extraction
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, Google, DeepSeek, Groq, and more

## Installation

```bash
pip install heaven-framework
```

# Toolbox MCP
#### Claude Code
```
"heaven-framework-toolbox": {
      "type": "stdio",
      "command": "python",
      "args": [
        "-m",
        "heaven_base.mcps.toolbox_server"
      ],
      "env": {
        "HEAVEN_DATA_DIR": "YOUR_HEAVEN_DATA_DIR", # /tmp/heaven_data recommended
        "OPENAI_API_KEY": "sk-YOUR_OPENAI_API_KEY"
      }
    }
```

# SDK
## Quick Start

### Creating a Simple Agent

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

# Run the agent
agent = MyAgent()
result = await agent.run("What is the meaning of life?")
print(result)
```

### Creating a Custom Tool

```python
from heaven_base import BaseHeavenTool, ToolArgsSchema

class CalculatorToolArgsSchema(ToolArgsSchema):
    arguments = {
        'expression': {
            'type': 'str',
            'description': 'Mathematical expression to evaluate',
            'required': True
        }
    }

class CalculatorTool(BaseHeavenTool):
    name = "calculator"
    description = "Evaluates mathematical expressions"
    args_schema = CalculatorToolArgsSchema
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
```

## Core Components

- **BaseHeavenAgent**: Base class for all HEAVEN agents
- **BaseHeavenTool**: Base class for all HEAVEN tools
- **UnifiedChat**: Multi-provider LLM interface
- **HeavenEvent**: Standardized event format
- **History**: Advanced conversation history management

## Development Roadmap

HEAVEN Base is actively evolving with a clear roadmap to become a complete agent development toolkit:

### 🚀 Core Infrastructure (Current)
- ✅ **BaseHeavenAgent & BaseHeavenTool** - Foundation classes for agents and tools
- ✅ **Cross-Framework Support** - LangChain and Google ADK compatibility
- ✅ **HEAVEN Events** - Standardized event system for agent communication
- ✅ **Registry System** - Data storage and retrieval with cross-registry references
- ✅ **History Management** - Conversation tracking with ~/.heaven/ user directory

### 🛠️ Local Execution (In Progress)
- **Hermes Local Execution** - Non-containerized agent orchestration (replaces complex Docker setup)
- **Core Default Tools** - BashTool, NetworkEditTool, SafeCodeReaderTool out of the box
- **Simple Agent Runner** - run_agent() utility for immediate agent execution

### 🧠 LLM Integration Improvements
- **LiteLLM Integration** - Replace LangChain internals with LiteLLM for better provider support
- **Unified Message Formats** - Standardize all message handling to OpenAI format
- **Enhanced Provider Support** - Better error handling and feature parity across providers

### 🎯 Prompt Engineering System
- **Prompt Injection System** - Build input prompts from reusable blocks or freestyle strings
- **Template Management** - Organize and version prompt templates
- **Dynamic Composition** - Programmatically construct complex prompts

### 🔄 Context Engineering
- **Weave Operations** - Extract and reorganize conversation history
- **Inject Operations** - Add files, context, or arbitrary content to conversations
- **Context Compression** - Manage long conversations with smart truncation

### 🏗️ Workflow Orchestration
- **HeavenWorkflow Class** - Turn all components into LangGraph nodes
- **Lego-Style Building** - Compose agents, tools, and operations as reusable blocks
- **Visual Workflow Design** - Drag-and-drop agent workflow construction
- **Hierarchical Execution** - Nested workflows with state management

### 🎯 The Vision
HEAVEN Base aims to be the complete toolkit for agent development:
- **Install** → pip install heaven-framework
- **Run** → hermes.run_agent(agent, prompt)
- **Compose** → Build prompts from blocks and inject context
- **Orchestrate** → Create complex workflows as LangGraph nodes
- **Scale** → Hierarchical agent networks with automatic state management

## Documentation

For full documentation, visit [https://heaven-base.readthedocs.io](https://heaven-base.readthedocs.io)

## License

MIT License - see LICENSE file for details.
