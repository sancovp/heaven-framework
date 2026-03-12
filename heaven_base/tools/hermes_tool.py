"""
HermesTool - Send goals to agents using Hermes cross-container execution.

This is the tool wrapper around use_hermes from hermes_utils.
Agents equipped with this tool can orchestrate other agents by sending
them goals to accomplish, optionally on different containers.
"""

from typing import Dict, Any
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tool_utils.hermes_utils import use_hermes


class HermesArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'target_container': {
            'name': 'target_container',
            'type': 'str',
            'description': 'Target container name or ID.',
            'required': True
        },
        'source_container': {
            'name': 'source_container',
            'type': 'str',
            'description': 'Source container name or ID.',
            'required': True
        },
        'goal': {
            'name': 'goal',
            'type': 'str',
            'description': 'Goal for the agent to accomplish',
            'required': False
        },
        'agent': {
            'name': 'agent',
            'type': 'str',
            'description': 'Optional agent type to use. If not provided, uses default agent.',
            'required': False
        },
        'iterations': {
            'name': 'iterations',
            'type': 'int',
            'description': 'Number of iterations to attempt. Max 20.',
            'required': False,
            'default': 1
        },
        'system_prompt_suffix': {
            'name': 'system_prompt_suffix',
            'type': 'str',
            'description': 'Optional suffix to add to the system prompt',
            'required': False
        },
        'history_id': {
            'name': 'history_id',
            'type': 'str',
            'description': 'Optional history ID to continue from',
            'required': False
        },
        'return_summary': {
            'name': 'return_summary',
            'type': 'bool',
            'description': 'Return summary instead of full conversation',
            'required': False,
            'default': False
        },
        'ai_messages_only': {
            'name': 'ai_messages_only',
            'type': 'bool',
            'description': 'When True (default), shows simplified view with injection string for HumanMessages. When False, shows full message content.',
            'required': False,
            'default': True
        },
        'continuation': {
            'name': 'continuation',
            'type': 'bool',
            'description': 'Force continuation mode if history_id provided',
            'required': False
        },
        'remove_agents_config_tools': {
            'name': 'remove_agents_config_tools',
            'type': 'bool',
            'description': 'When True, removes all tools from agent config and only uses additional_tools. When False (default), additional_tools operates as described.',
            'required': False,
            'default': False
        },
        'orchestration_preprocess': {
            'name': 'orchestration_preprocess',
            'type': 'bool',
            'description': 'When True, initializes the agent with orchestrator=True to give it knowledge of available tools and agents',
            'required': False,
            'default': False
        },
        'hermes_config': {
            'name': 'hermes_config',
            'type': 'str',
            'description': 'Optional hermes config name. If provided, the according hermes config will be loaded and its settings will override the individual parameters. All other parameters can be set with whatever they would need like empty strings or lists if a hermes config name string is provided',
            'required': False
        },
        'variable_inputs': {
            'name': 'variable_inputs',
            'type': 'dict',
            'description': """Optional Dictionary of values for templated parameters required when using hermes_config.
            Structure matches the hermes_config's variable_inputs template:
            - For string templates (e.g. goal): {"param_name": {"variable1": "value1", ...}}
            - For list templates (e.g. tools): {"param_name": ["value1", "value2"]}
            - For direct values: {"param_name": value}
            Example: {
                "goal": {"tool_name": "TestTool", "input": "test"},
                "additional_tools": ["TestTool"],
                "agent": "test_agent"
            }""",
            'required': False,
            'default': None
        },
        'return_last_response_only': {
            'name': 'return_last_response_only',
            'type': 'bool',
            'description': 'Option to return only the last AI response',
            'required': False,
            'default': False
        }
    }


class HermesTool(BaseHeavenTool):
    name = "HermesTool"
    description = "Send a goal to an agent using Hermes."
    func = use_hermes
    args_schema = HermesArgsSchema
    is_async = True
