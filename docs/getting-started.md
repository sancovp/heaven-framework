# Getting Started with HEAVEN Framework

This guide will help you get up and running with HEAVEN in minutes.

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install HEAVEN Base
```bash
pip install heaven-base
```

### Optional Dependencies
```bash
# For Google ADK support
pip install heaven-base[google]

# For advanced LangGraph features  
pip install heaven-base[langgraph]

# For all optional features
pip install heaven-base[all]
```

## Your First Agent {#first-agent}

Let's create a simple assistant agent:

```python
from heaven_base import BaseHeavenAgent, HeavenAgentConfig

class SimpleAssistant(BaseHeavenAgent):
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="SimpleAssistant",
            system_prompt="You are a helpful assistant that provides clear, concise answers.",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

# Create and run the agent
async def main():
    agent = SimpleAssistant()
    result = await agent.run("What is the capital of France?")
    print(result.content)

# Run it
import asyncio
asyncio.run(main())
```

### What's Happening Here?

1. **Inherit from BaseHeavenAgent**: This gives you all the framework capabilities
2. **Define default_config**: Specifies your agent's behavior and LLM settings
3. **Call agent.run()**: Executes your prompt and returns a result

## Adding Tools to Your Agent {#tools}

Tools extend your agent's capabilities. Let's add a calculator tool:

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
    description = "Evaluates mathematical expressions safely"
    args_schema = CalculatorToolArgsSchema
    
    def _run(self, expression: str) -> str:
        try:
            # Safe evaluation (you'd want more robust parsing in production)
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Add tool to agent
class MathAssistant(BaseHeavenAgent):
    @classmethod 
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="MathAssistant",
            system_prompt="You are a math assistant. Use the calculator tool for computations.",
            model="gpt-4",
            tools=[CalculatorTool()]
        )

# Test it
async def test_math():
    agent = MathAssistant()
    result = await agent.run("What is 15 * 23 + 7?")
    print(result.content)

asyncio.run(test_math())
```

## Working with Multiple LLM Providers

HEAVEN supports multiple LLM providers out of the box:

```python
from heaven_base import ProviderEnum

# OpenAI (default)
openai_config = HeavenAgentConfig(
    name="OpenAIAgent",
    provider=ProviderEnum.OPENAI,
    model="gpt-4",
    api_key="your-openai-key"
)

# Anthropic
anthropic_config = HeavenAgentConfig(
    name="ClaudeAgent", 
    provider=ProviderEnum.ANTHROPIC,
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-key"
)

# Google
google_config = HeavenAgentConfig(
    name="GeminiAgent",
    provider=ProviderEnum.GOOGLE,
    model="gemini-pro",
    api_key="your-google-key"
)
```

## Event-Driven Communication

HEAVEN agents communicate through standardized events:

```python
from heaven_base.memory.heaven_event import HeavenEvent, EventType

# Create an event
event = HeavenEvent(
    event_type=EventType.USER_MESSAGE,
    content="Hello from Agent A",
    agent_id="agent_a",
    timestamp="2024-01-01T12:00:00Z"
)

# Agents can listen for and respond to events
class EventListenerAgent(BaseHeavenAgent):
    def process_event(self, event: HeavenEvent):
        if event.event_type == EventType.USER_MESSAGE:
            return f"I received: {event.content}"
        return None
```

## Using the Registry System

Store and share data between agents:

```python
from heaven_base.registry import RegistryFactory

# Get a registry instance
registry = RegistryFactory.get_registry("my_project")

# Store data
registry.store("user_preferences", {
    "language": "en",
    "theme": "dark"
})

# Retrieve data
prefs = registry.retrieve("user_preferences")

# Use in agent
class PersonalizedAgent(BaseHeavenAgent):
    def __init__(self):
        super().__init__()
        self.registry = RegistryFactory.get_registry("my_project")
        
    async def run(self, message: str):
        prefs = self.registry.retrieve("user_preferences") 
        # Customize response based on preferences
        return await super().run(f"[User prefers {prefs['language']}] {message}")
```

## Next Steps

Now that you have the basics, explore:

- **[Agent Architecture](agents/architecture.md)** - Deep dive into agent design patterns
- **[Custom Tools](tools/custom-tools.md)** - Build sophisticated tools for your agents
- **[LangGraph Integration](langgraph/workflows.md)** - Create complex multi-agent workflows
- **[Examples](examples/)** - Real-world usage patterns and code samples

## Common Patterns

### Agent with Persistent Memory
```python
class MemoryAgent(BaseHeavenAgent):
    def __init__(self):
        super().__init__()
        self.memory = []
        
    async def run(self, message: str):
        # Add to memory
        self.memory.append(f"User: {message}")
        
        # Include memory in prompt
        context = "\\n".join(self.memory[-5:])  # Last 5 exchanges
        prompt = f"Context:\\n{context}\\n\\nUser: {message}"
        
        result = await super().run(prompt)
        self.memory.append(f"Assistant: {result.content}")
        return result
```

### Tool Composition
```python
# Combine multiple tools
class PowerUserAgent(BaseHeavenAgent):
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="PowerUser",
            tools=[
                CalculatorTool(),
                WebSearchTool(),
                FileManagerTool(),
                CodeExecutorTool()
            ]
        )
```

### Error Handling
```python
class RobustAgent(BaseHeavenAgent):
    async def run(self, message: str):
        try:
            result = await super().run(message)
            return result
        except Exception as e:
            # Log error and provide fallback
            logger.error(f"Agent error: {e}")
            return AgentResult(
                content="I encountered an error. Please try rephrasing your request.",
                error=str(e)
            )
```

## Troubleshooting

### Common Issues

**Import Error**: Make sure you've installed heaven-base correctly
```bash
pip install --upgrade heaven-base
```

**API Key Error**: Set your API keys as environment variables
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**Provider Not Available**: Install provider-specific dependencies
```bash
pip install heaven-base[google]  # For Google/Gemini
```

**Tool Not Working**: Check that tool arguments match the schema exactly

For more help, see our [Troubleshooting Guide](troubleshooting.md).