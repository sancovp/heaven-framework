# write_prompt_block_tool.py

from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..prompts.prompt_blocks.prompt_block_utils import write_prompt_block as write_prompt_block_func
from typing import Dict, Any

class WritePromptBlockToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'name': {
            'name': 'name',
            'type': 'str',
            'description': 'Unique name for the prompt block',
            'required': True
        },
        'text': {
            'name': 'text',
            'type': 'str',
            'description': 'The actual prompt text content',
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
        }
    }

def write_prompt_block_tool_func(name: str, text: str, domain: str, subdomain: str) -> str:
    """
    Write a prompt block to a JSON file in the prompt_blocks directory.
    
    Args:
        name (str): Unique name for the prompt block
        text (str): The actual prompt text content
        domain (str): Main domain category (e.g., 'coding', 'writing')
        subdomain (str): Specific subdomain within the main domain
        
    Returns:
        str: Success message with details about the created prompt block
    """
    # Call the actual implementation
    write_prompt_block_func(name, text, domain, subdomain)
    
    # Return a success message
    return f"✅ Prompt block '{name}' created successfully!\n" \
           f"Domain: {domain}\n" \
           f"Subdomain: {subdomain}\n" \
           f"Length: {len(text)} characters\n\n" \
           f"To use this prompt block in a HeavenAgentConfig, add it to the prompt_suffix_blocks list:\n" \
           f"```python\n" \
           f"config = HeavenAgentConfig(\n" \
           f"    # ... other parameters ...\n" \
           f"    prompt_suffix_blocks=['{name}']\n" \
           f")\n" \
           f"```"

class WritePromptBlockTool(BaseHeavenTool):
    name = "WritePromptBlockTool"
    description = "Creates a prompt block that can be used as a suffix to system prompts in agent configurations. Prompt blocks are stored as JSON files and can be referenced by name in HeavenAgentConfig's prompt_suffix_blocks parameter."
    func = write_prompt_block_tool_func
    args_schema = WritePromptBlockToolArgsSchema
    is_async = False
