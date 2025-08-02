from typing import Dict, Any
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tool_utils.agent_config_test import agent_config_test

class AgentConfigTestToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        # Core test parameters
        'test_prompt': {
            'name': 'test_prompt',
            'type': 'str',
            'description': 'The prompt to test the agent configuration with',
            'required': True
        },
        'iterations': {
            'name': 'iterations',
            'type': 'int',
            'description': 'Number of iterations for agent mode (default: 1)',
            'required': False
        },
        'agent_mode': {
            'name': 'agent_mode',
            'type': 'bool',
            'description': 'Whether to run in agent mode (goal/iterations format) or direct mode (default: True)',
            'required': False
        },
        
        # HeavenAgentConfig parameters
        'name': {
            'name': 'name',
            'type': 'str',
            'description': 'Name for the agent configuration',
            'required': False
        },
        'system_prompt': {
            'name': 'system_prompt',
            'type': 'str',
            'description': 'System prompt for the agent',
            'required': True
        },
        'tools': {
            'name': 'tools',
            'type': 'array',
            'description': 'List of tool names to include (e.g., ["NetworkEditTool", "BashTool"])',
            'required': False
        },
        'provider': {
            'name': 'provider',
            'type': 'str',
            'description': 'AI provider: "anthropic", "openai", or "google" (default: "openai")',
            'required': False
        },
        'model': {
            'name': 'model',
            'type': 'str',
            'description': 'Model name (e.g., "claude-3-7-sonnet-latest", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gemini-2.5-flash-preview-04-17")',
            'required': False
        },
        'temperature': {
            'name': 'temperature',
            'type': 'float',
            'description': 'Temperature for AI generation (0.0-1.0, default: 0.7)',
            'required': False
        },
        'max_tokens': {
            'name': 'max_tokens',
            'type': 'int',
            'description': 'Maximum tokens for AI response (default: 8000)',
            'required': False
        },
        'thinking_budget': {
            'name': 'thinking_budget',
            'type': 'int',
            'description': 'Thinking budget for reasoning (optional). This is a number of tokens between 1024 and 16000',
            'required': False
        },
        'additional_kws': {
            'name': 'additional_kws',
            'type': 'array',
            'description': 'Additional keywords for extracts agent is expected to produce when contextually appropriate (e.g., ["summary", "result", "path", "whatever"])',
            'required': False
        },
        'additional_kw_instructions': {
            'name': 'additional_kw_instructions',
            'type': 'str',
            'description': 'Instructions for additional keyword extraction',
            'required': False
        },
        'known_config_paths': {
            'name': 'known_config_paths',
            'type': 'array',
            'description': 'List of Hermes config paths for orchestrator mode',
            'required': False
        },
        'prompt_suffix_blocks': {
            'name': 'prompt_suffix_blocks',
            'type': 'array',
            'description': 'List of prompt suffix block names to append',
            'required': False
        },
        
        # BaseHeavenAgent initialization parameters
        'max_tool_calls': {
            'name': 'max_tool_calls',
            'type': 'int',
            'description': 'Maximum number of tool calls allowed (default: 10)',
            'required': False
        },
        'orchestrator': {
            'name': 'orchestrator',
            'type': 'bool',
            'description': 'Whether to run in orchestrator mode with Hermes Switchboard prompt block (default: False)',
            'required': False
        },
        'history_id': {
            'name': 'history_id',
            'type': 'str',
            'description': 'Existing history ID to continue from (optional)',
            'required': False
        },
        'system_prompt_suffix': {
            'name': 'system_prompt_suffix',
            'type': 'str',
            'description': 'Additional text to append to the system prompt',
            'required': False
        },
        'adk': {
            'name': 'adk',
            'type': 'bool',
            'description': 'Whether to use Google ADK instead of LangChain (default: False)',
            'required': False
        },
        'duo_enabled': {
            'name': 'duo_enabled',
            'type': 'bool',
            'description': 'Whether to enable DUO prompt injection system (default: False)',
            'required': False
        },
        'run_on_langchain': {
            'name': 'run_on_langchain',
            'type': 'bool',
            'description': 'Force LangChain usage even with Google provider (default: False)',
            'required': False
        },
        
        # Assertion parameters for validation
        'assert_tool_used': {
            'name': 'assert_tool_used',
            'type': 'str',
            'description': 'Assert that this specific tool was used (e.g., "SafeCodeReaderTool")',
            'required': False
        },
        'assert_no_errors': {
            'name': 'assert_no_errors',
            'type': 'bool',
            'description': 'Assert that no tool errors occurred during execution (default: False)',
            'required': False
        },
        'assert_goal_accomplished': {
            'name': 'assert_goal_accomplished',
            'type': 'bool',
            'description': 'Assert that the agent marked the goal as accomplished (default: False)',
            'required': False
        },
        'assert_extracted_keys': {
            'name': 'assert_extracted_keys',
            'type': 'array',
            'description': 'Assert that these keys exist in extracted_content (e.g., ["summary", "result"])',
            'required': False
        },
        'assert_extracted_contains': {
            'name': 'assert_extracted_contains',
            'type': 'object',
            'description': 'Assert that extracted_content contains specific key-value pairs (e.g., {"status": "complete"})',
            'required': False
        },
        'assert_min_tool_calls': {
            'name': 'assert_min_tool_calls',
            'type': 'int',
            'description': 'Assert minimum number of tool calls made (default: no assertion)',
            'required': False
        },
        'assert_output_contains': {
            'name': 'assert_output_contains',
            'type': 'str',
            'description': 'Assert that final output contains this substring',
            'required': False
        }
    }

class AgentConfigTestTool(BaseHeavenTool):
    name = "AgentConfigTestTool"
    description = "Test agent configurations by running them with dynamic JSON configs. Perfect for prompt engineering and rapid prototyping without creating agent files."
    func = agent_config_test
    args_schema = AgentConfigTestToolArgsSchema
    is_async = True