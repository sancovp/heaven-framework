# Test Heaven Agent

How to test BaseHeavenAgent with live LLM calls using MiniMax (the only supported provider).

## The Testing System

`heaven_base.tool_utils.agent_config_test.agent_config_test()` is a function that creates a BaseHeavenAgent, runs it with a prompt, and returns results. But for full control, create the agent directly.

## Required Setup

```python
import asyncio, os, subprocess
os.environ["HEAVEN_ALLOW_STDOUT"] = "1"
os.environ["HEAVEN_DATA_DIR"] = "/tmp/heaven_data"

from heaven_base import BaseHeavenAgent, HeavenAgentConfig, ProviderEnum, UnifiedChat
from heaven_base import BaseHeavenTool, ToolArgsSchema, ToolResult
from heaven_base.memory.history import History
from typing import Dict, Any
```

## Making a Test Tool

```python
class BashToolSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'command': {
            'name': 'command',
            'type': 'str',
            'description': 'Bash command to run',
            'required': True
        }
    }

def bash_func(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}"

class BashTool(BaseHeavenTool):
    name = "BashTool"
    description = "Runs a bash command and returns the output"
    func = bash_func
    args_schema = BashToolSchema
    is_async = False
```

## Running a Test

```python
async def test(prompt, tools=None):
    config = HeavenAgentConfig(
        name="TestAgent",
        system_prompt="You are a test agent. Use tools as instructed.",
        tools=tools or [],
        provider=ProviderEnum.ANTHROPIC,
        model="MiniMax-M2.5",
        temperature=0.1
    )
    agent = BaseHeavenAgent(
        config=config,
        unified_chat=UnifiedChat(),
        max_tool_calls=10,
        history=History(messages=[])
    )
    result = await agent.run(prompt=prompt)
    history = result.get("history")
    tool_calls = sum(1 for m in history.messages if m.__class__.__name__ == "ToolMessage")
    return result, tool_calls
```

## The Three Critical Tests

### 1. Simple completion (no tools)
```python
result, tc = await test("What is 2+2? Answer with just the number.")
assert tc == 0
```

### 2. Single tool call
```python
result, tc = await test('Use BashTool with "echo hello"', tools=[BashTool])
assert tc >= 1
```

### 3. Interleaved Text+Tool (THE critical test)

This tests the pattern where the model returns Text + Tool_use in the SAME response, then needs to continue with more tool calls. This was broken before the Mar 2 langchain inner loop fix.

```python
result, tc = await test(
    'Use BashTool with pwd and tell me the result. '
    'After you tell me the pwd, then use BashTool again with whoami and tell me the result. '
    'You must tell me the result of each bash call as soon as you get it. Do not batch them.',
    tools=[BashTool]
)
assert tc >= 2  # Must chain through Text+Tool+Text+Tool pattern
```

## Execution Path

MiniMax goes through the **langchain path** (`run_langchain` in baseheavenagent.py):
- Provider is ANTHROPIC → uses ChatAnthropic via LangChain
- MiniMax API key from `MINIMAX_API_KEY` env var
- API URL auto-set to `https://api.minimax.io/anthropic`
- Inner tool loop: simple `while current_response.tool_calls` → execute → re-invoke → re-check

## Reading Test Results

```python
history = result.get("history")
for i, msg in enumerate(history.messages):
    cn = msg.__class__.__name__  # SystemMessage, HumanMessage, AIMessage, ToolMessage
    print(f"[{i}] {cn}: {str(msg.content)[:100]}")
```

Message types in order: SystemMessage → HumanMessage → AIMessage (may contain tool_use blocks) → ToolMessage → AIMessage → ...

## What Can Go Wrong

1. **Tool calls not chaining**: Inner loop exits early. Check `baseheavenagent.py` `run_langchain` inner while loop.
2. **Tool_use blocks dropped**: Old bug where AIMessage content was destructured and tool_use blocks filtered out. Fixed Mar 2 — whole AIMessage is now appended.
3. **HEAVEN_DATA_DIR not set**: Non-fatal — history save fails but test runs. Set it to avoid warnings.
4. **thinking_budget warning**: Non-fatal langchain warning. Ignore it.
