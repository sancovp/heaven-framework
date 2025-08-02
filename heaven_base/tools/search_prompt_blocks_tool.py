# search_prompt_blocks_tool.py

from typing import Dict, Any, List
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..prompts.prompt_blocks.prompt_block_registry import search_blocks_by_similarity

class SearchPromptBlocksToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'query': {
            'name': 'query',
            'type': 'str',
            'description': 'The search query to find similar prompt blocks',
            'required': True
        },
        'top_n': {
            'name': 'top_n',
            'type': 'int',
            'description': 'Maximum number of results to return (default: 5)',
            'required': False
        },
        'min_similarity': {
            'name': 'min_similarity',
            'type': 'float',
            'description': 'Minimum similarity score (0-1) for results (default: 0.1)',
            'required': False
        },
        'full_text': {
            'name': 'full_text',
            'type': 'bool',
            'description': 'If True, returns the entire block text without truncation',
            'required': False
        }
    }

def search_prompt_blocks_tool_func(query: str, top_n: int = 5, min_similarity: float = 0.01, full_text: bool = False) -> str:
    """
    Search for prompt blocks similar to the given query.

    Args:
        query (str): The search query
        top_n (int): Maximum number of results to return
        min_similarity (float): Minimum similarity score (0-1) for results
        full_text (str): Returns the full text of the prompt block
    Returns:
        str: Formatted search results and optional full text
    """
    # Search for similar blocks
    results = search_blocks_by_similarity(query, top_n, min_similarity)

    if not results:
        return "No matching prompt blocks found."

    # Format the results
    output = f"Found {len(results)} prompt blocks similar to your query:\n\n"

    for i, (block, similarity) in enumerate(results, 1):
        # Format the block text for display (truncate if too long)
        block_text = block['text']
        if not full_text and len(text) > 200:
            text = text[:197] + "..."
        

        # Replace newlines with spaces for compact display
        block_text = block_text.replace("\n", " ")
        
        output += f"{i}. **{block['name']}** (Similarity: {similarity:.2f})\n"
        output += f"   Domain: {block['domain']}, Subdomain: {block['subdomain']}\n"
        if not full_text:
            output += f"   Preview: {block_text}\n\n"
        else:
              output += f"Full Text:\n\n{block_text}\n\n"

    output += "\nTo use any of these blocks in a HeavenAgentConfig, add the block name to the prompt_suffix_blocks list:\n"
    output += "```python\n"
    output += "config = HeavenAgentConfig(\n"
    output += "    # ... other parameters ...\n"
    output += f"    prompt_suffix_blocks=['{results[0][0]['name']}']\n"
    output += ")\n"
    output += "```"

    return output

class SearchPromptBlocksTool(BaseHeavenTool):
    name = "SearchPromptBlocksTool"
    description = "Searches for prompt blocks similar to a given query using semantic similarity. Returns a list of matching blocks with their similarity scores or full text."
    func = search_prompt_blocks_tool_func
    args_schema = SearchPromptBlocksToolArgsSchema
    is_async = False