# HEAVEN Framework Examples

This directory contains comprehensive, working examples demonstrating all major components of the HEAVEN framework. Each example is fully functional and tested.

## Prerequisites

```bash
# Set required environment variable
export HEAVEN_DATA_DIR=/tmp/heaven_data

# Ensure you have API keys configured for uni-api
# The examples use model "o4-mini" through uni-api
```

## Examples Overview

### 1. Basic Agent (`01_basic_agent.py`)
**Purpose**: Shows the simplest possible HEAVEN agent setup

**What it demonstrates**:
- Creating a `HeavenAgentConfig`
- Initializing `BaseHeavenAgent`
- Running a basic conversation
- Getting history IDs for conversation continuity

**Key code**:
```python
config = HeavenAgentConfig(
    name="BasicAssistant",
    system_prompt="You are a helpful AI assistant.",
    tools=[],
    provider=ProviderEnum.OPENAI,
    model="o4-mini",
    temperature=0.7
)

agent = BaseHeavenAgent(config, UnifiedChat, history=history)
result = await agent.run(prompt="What is HEAVEN framework?")
```

### 2. Agent with Tools (`02_agent_with_tools.py`)
**Purpose**: Demonstrates agents using tools for file operations and system commands

**What it demonstrates**:
- Adding tools to agent configuration (`NetworkEditTool`, `BashTool`)
- Agent autonomously using tools to complete tasks
- File creation and execution workflow

**Key features**:
- Agents can read, write, and edit files
- Agents can execute bash commands
- Tools are called automatically based on the prompt

### 3. Extraction Patterns (`03_extraction_simple.py`)
**Purpose**: Shows how to extract structured data from agent responses

**What it demonstrates**:
- Using `additional_kws` and `additional_kw_instructions` in agent config
- Agents producing structured output in fenced code blocks
- Automatic parsing and extraction of specific content sections

**Key concepts**:
- Extraction keywords: `["summary", "key_points", "recommendations", "confidence"]`
- Agent mode vs regular mode for extraction
- Simple system prompts work better than complex instructions

### 4. Completion Runner (`04_completion_runner.py`)
**Purpose**: Shows how to use HEAVEN's completion-style execution utilities

**What it demonstrates**:
- Using `exec_completion_style()` for direct prompt-response execution
- Alternative to full agent mode for simpler tasks
- Integration with HEAVEN's runner utilities

**Key usage**:
```python
from heaven_base.tool_utils.completion_runners import exec_completion_style

result = await exec_completion_style(
    prompt=prompt,
    agent=config
)
```

### 5. Hermes Runner (`05_hermes_runner.py`)
**Purpose**: Demonstrates the Hermes orchestration system

**What it demonstrates**:
- Using `use_hermes_dict()` for structured agent execution
- Agent mode with goal/iterations formatting
- Cross-container execution capabilities (when properly configured)

**Key features**:
- Structured goal-based execution
- Support for multiple iterations
- Integration with HEAVEN's orchestration system

### 6. LangGraph Integration (`06_langgraph_integration.py`)
**Purpose**: Shows how to integrate HEAVEN agents with LangGraph workflows

**What it demonstrates**:
- Using `HeavenState` for workflow state management
- Creating custom nodes that use HEAVEN completion runners
- Building complex workflows with multiple HEAVEN agents
- State management and data flow between nodes

**Key workflow pattern**:
```python
async def analysis_node(state: HeavenState) -> dict:
    result = await completion_runner(
        state=state,
        prompt=prompt,
        agent=config
    )
    return {"results": state.get("results", []) + [result]}
```

### 7. Prompt Engineering with PIS (`07_prompt_engineering_pis.py`)
**Purpose**: Advanced prompt engineering using HEAVEN's Prompt Injection System

**What it demonstrates**:
- Creating structured prompt configurations with `PromptInjectionSystemVX1`
- Template variable substitution in prompts
- Multi-step prompt generation
- Professional prompt enhancement patterns

**Key PIS concepts**:
- `PromptStepDefinitionVX1`: Define prompt generation steps
- `PromptBlockDefinitionVX1`: Individual prompt components
- `BlockTypeVX1.FREESTYLE`: Template-based prompt blocks
- Template variables for dynamic prompt customization

### 8. Dynamic Tool Creation (`08_dynamic_tool_creation.py`)
**Purpose**: Shows how to create tools dynamically from function docstrings

**What it demonstrates**:
- Using `make_heaven_tool_from_docstring()` to create tools from Python functions
- Automatic tool schema generation from function signatures and docstrings
- Type hint integration for parameter validation
- Creating utility tools for calculations and conversions

**Key pattern**:
```python
def calculate_tax(amount: float, tax_rate: float) -> dict:
    """
    Calculate tax and total amount for a purchase.
    
    Args:
        amount (float): The base amount before tax
        tax_rate (float): The tax rate as a decimal
    
    Returns:
        dict: Calculation results
    """
    # Implementation here
    
# Convert function to HEAVEN tool
TaxCalculatorTool = make_heaven_tool_from_docstring(calculate_tax)
```

**Benefits**:
- No need to manually create tool classes
- Docstrings automatically become tool descriptions
- Type hints provide automatic validation
- Rapid prototyping of new agent capabilities

### 9. OmniTool Universal Interface (`09_omnitool_universal_interface.py`)
**Purpose**: Demonstrates OmniTool - the universal gateway to all HEAVEN tools

**What it demonstrates**:
- Using `omnitool()` to discover all available tools
- Getting detailed tool information and schemas
- Calling any HEAVEN tool through the universal interface
- Agent integration with OmniTool for dynamic tool access

**Key tools accessible through OmniTool**:
- `BashTool`: System command execution
- `NetworkEditTool`: File operations (create, view, edit, delete)
- `RegistryTool`: Registry operations (list, get, set values)
- `ThinkTool`: AI reasoning and analysis
- `WebSearchTool`: Web search capabilities
- `WorkflowRelayTool`: Workflow orchestration
- And many more...

**Universal Interface Pattern**:
```python
# Discover available tools
tools = await omnitool(list_tools=True)

# Get tool information
info = await omnitool('BashTool', get_tool_info=True)

# Execute any tool
result = await omnitool('BashTool', parameters={
    'command': 'echo "Hello from OmniTool!"'
})

# File operations
result = await omnitool('NetworkEditTool', parameters={
    'command': 'create',
    'path': '/tmp/test.txt',
    'file_text': 'Content here',
    'command_arguments': {}
})
```

**Benefits**:
- Single interface to access ALL HEAVEN tools
- Dynamic tool discovery and execution
- No need to import individual tool classes
- Perfect for meta-programming and tool orchestration
- Consistent parameter handling across all tools

## Running the Examples

Each example is self-contained and can be run independently:

```bash
cd /home/GOD/heaven-framework-repo

# Basic examples
python examples/01_basic_agent.py
python examples/02_agent_with_tools.py

# Advanced examples  
python examples/03_extraction_simple.py
python examples/04_completion_runner.py
python examples/05_hermes_runner.py
python examples/06_langgraph_integration.py
python examples/07_prompt_engineering_pis.py
python examples/08_dynamic_tool_creation.py
```

## Common Patterns

### Environment Setup
All examples include:
```python
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'
```

### Agent Configuration
Standard pattern:
```python
config = HeavenAgentConfig(
    name="AgentName",
    system_prompt="Agent description",
    tools=[Tool1, Tool2],  # Optional
    provider=ProviderEnum.OPENAI,
    model="o4-mini",
    temperature=0.7
)
```

### Agent Execution
Two main patterns:
```python
# Regular mode
result = await agent.run(prompt="Direct question")

# Agent mode (for complex tasks)
result = await agent.run(prompt='agent goal="Complex task", iterations=3')
```

## Advanced Topics

### Model Selection
- Use `o4-mini` for cost-effective development
- Switch to `gpt-5` or `claude-3-5-sonnet-latest` for production
- Anthropic models generally follow instructions better but cost more

### Agent Mode vs Regular Mode
- **Regular mode**: Direct question-answer, good for simple tasks
- **Agent mode**: Goal-oriented with task planning, good for complex workflows

### Tool Integration
- Tools are automatically available to agents when added to `tools=[]`
- Agents decide when to use tools based on the prompt
- `NetworkEditTool` and `BashTool` are the most commonly used

### Error Handling
- OpenAI models sometimes get stuck in meta-planning loops
- Anthropic models are more reliable for complex agent mode tasks
- Simple system prompts work better than overly detailed instructions

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the heaven-framework-repo directory
2. **API errors**: Check that uni-api container is running (`docker ps | grep uni-api`)
3. **Model not available**: Make sure the model is configured in your uni-api setup
4. **Agent gets stuck**: Try reducing complexity of system prompt or switching to Anthropic models

### Debug Tips

1. Check the console output for tool calls and API responses
2. Look for `WriteBlockReportTool` calls - indicates agent is stuck/confused
3. Use fewer iterations when testing new patterns
4. Start with simple prompts and gradually add complexity

## Next Steps

After working through these examples:

1. **Customize for your domain**: Modify system prompts and tools for your specific use case
2. **Explore the registry system**: Learn how to persist and share agent configurations
3. **Build workflows**: Chain multiple agents together using LangGraph patterns
4. **Deploy to containers**: Set up multi-container HEAVEN deployment for production

## Contributing

When adding new examples:
1. Follow the naming convention: `XX_descriptive_name.py`
2. Include comprehensive docstrings
3. Test thoroughly before committing
4. Update this README with the new example

---

**Note**: These examples are based on the heaven-framework library version 1.2.0 and demonstrate production-ready patterns for building AI agent systems.