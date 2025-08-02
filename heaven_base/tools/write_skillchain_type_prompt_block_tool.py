from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tool_utils.skillchain_utils import make_skillchain_prompt_block, SkillchainStep
from typing import Dict, Any, List

class WriteSkillchainTypePromptBlockToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'name': {
            'name': 'name',
            'type': 'str',
            'description': 'Unique name for the skillchain prompt block',
            'required': True
        },
        'domain': {
            'name': 'domain',
            'type': 'str',
            'description': 'Main domain category (e.g., "coding", "writing")',
            'required': True
        },
        'subdomain': {
            'name': 'subdomain',
            'type': 'str',
            'description': 'Specific subdomain within the main domain',
            'required': True
        },
        'assignments': {
            'name': 'assignments',
            'type': 'dict',
            'description': 'Emoji variable assignments (emoji -> entity/tool/chain name)',
            'required': True
        },
        'steps': {
            'name': 'steps',
            'type': 'list',
            'description': 'List of SkillchainStep dicts describing the workflow steps',
            'required': True
        },
        'final_return': {
            'name': 'final_return',
            'type': 'str',
            'description': 'The final output variable (must be an emoji variable assigned in the chain)',
            'required': True
        }
    }

class WriteSkillchainTypePromptBlockTool(BaseHeavenTool):
    name = "WriteSkillchainTypePromptBlockTool"
    description = """Creates a modular skillchain prompt block using emoji-based workflow definitions. The block can be used as a suffix in agent system prompts and is stored as a JSON file.

The steps for WriteSkillchainTypePromptBlockTool must be a list of SkillchainStep dictionaries, each having these required fields:

caller (str, emoji variable assigned to a tool/agent)
callee (str, emoji variable or tool/agent)
action (str, e.g., "call" or "loop")
args (optional, dict of arguments)
result (optional, emoji var for storing output)
loop_var (optional, for loops)
The SEL string, while human-readable, is not what the tool expects. It needs structured fields in each step.

Here’s How to Write a Valid Step
Suppose:

✍️ = WritePromptBlockTool
🔍 = SearchPromptBlocksTool
📝 = aggregator_agent
Steps should look like:

[
  {
    "caller": "✍️",
    "callee": "WritePromptBlockTool",
    "action": "call",
    "args": {"prompt_block_content": "prompt_block_content"},
    "result": "🆕"
  },
  {
    "caller": "🔍",
    "callee": "SearchPromptBlocksTool",
    "action": "call",
    "args": {"query": "prompt_block_content"},
    "result": "🔎"
  },
  {
    "caller": "📝",
    "callee": "aggregator_agent",
    "action": "call",
    "args": {"prompt_info": "🆕", "search_results": "🔎"},
    "result": "📦"
  }
]

Each step uses an assigned emoji as caller.
Each tool/entity is referenced by name as the callee.
action is "call".
args uses the appropriate variables.
result is a new emoji var for the output.
All emoji variables in steps are defined in assignments or as previous results.
final_return must be an emoji variable produced by a step ("📦" above).
    """
    func = make_skillchain_prompt_block
    args_schema = WriteSkillchainTypePromptBlockToolArgsSchema
    is_async = False
