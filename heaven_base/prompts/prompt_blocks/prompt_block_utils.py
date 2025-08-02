import json
import os
from pathlib import Path

def write_prompt_block(name: str, text: str, domain: str, subdomain: str) -> None:
    """
    Write a prompt block to a JSON file in the prompt_blocks directory.
    
    Args:
        name (str): Unique name for the prompt block
        text (str): The actual prompt text content
        domain (str): Main domain category (e.g., 'coding', 'writing')
        subdomain (str): Specific subdomain within the main domain
    """
    # Get the directory of this file
    current_dir = Path(__file__).parent
    
    # Create blocks directory if it doesn't exist
    blocks_dir = current_dir / 'blocks'
    blocks_dir.mkdir(exist_ok=True)
    
    # Create the block data
    block_data = {
        'name': name,
        'text': text,
        'domain': domain,
        'subdomain': subdomain
    }
    
    # Create filename - use lowercase and replace spaces with underscores
    filename = f"{name.lower().replace(' ', '_')}.json"
    file_path = blocks_dir / filename
    
    # Write the JSON file
    with open(file_path, 'w') as f:
        json.dump(block_data, f, indent=4)

    return f"Prompt block created with text:\n```markdown\n{text}\n```\n"
