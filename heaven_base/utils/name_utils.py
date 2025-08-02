import re
import os
from pathlib import Path
from typing import Optional


def get_agent_base_dir() -> str:
    """Get the agent base directory (lazy loading to avoid import-time env var requirements)."""
    from .get_env_value import EnvConfigUtil
    return os.path.join(EnvConfigUtil.get_heaven_data_dir(), "agents")


def pascal_to_snake(pascal: str) -> str:
    """Convert PascalCase to snake_case."""
    # Add underscore before any capital letter and convert to lowercase
    snake = re.sub('(?!^)([A-Z])', r'_\1', pascal).lower()
    return snake


def to_pascal_case(text: str) -> str:
    # Assuming words are separated by spaces, underscores, or hyphens
    return ''.join(word.capitalize() for word in re.split(r'[\s_-]+', text))


def camel_to_snake(name: str) -> str:
    return ''.join(['_'+c.lower() if c.isupper() else c for c in name]).lstrip('_')

def normalize_agent_name(name: str) -> str:
    """Normalize agent names for filesystem consistency.
    
    Converts:
    - "TestAgent" -> "test_agent"
    - "AgentmakerTestAgent" -> "agentmaker_test_agent"
    - "ResearcherAgent" -> "researcher_agent"
    """
    # Handle already snake_case names
    if '_' in name:
        return name.lower()
        
    # Convert camelCase/PascalCase to snake_case
    words = re.findall('[A-Z][^A-Z]*', name)
    if not words:  # If no capital letters found
        return name.lower()
    return '_'.join(word.lower() for word in words)



def resolve_agent_name(agent_name_input: str) -> Optional[str]:
    """
    Checks if an agent directory exists based on standard naming conventions
    and returns the normalized snake_case name if found, otherwise None.

    Args:
        agent_name_input: The potential agent name (PascalCase or snake_case).

    Returns:
        The normalized snake_case agent name if its directory exists, otherwise None.
    """
    if not agent_name_input:
        return None

    # Normalize the input name to snake_case for directory lookup
    agent_name_snake = normalize_agent_name(agent_name_input)
    print(f"[resolve_agent_name] Checking for '{agent_name_input}' (normalized dir: '{agent_name_snake}')")

    # Construct the expected directory path
    agent_dir_path = Path(get_agent_base_dir()) / agent_name_snake

    # Check if the directory exists
    if agent_dir_path.is_dir():
        print(f"[resolve_agent_name] Found directory: {agent_dir_path}")
        # Return the normalized snake_case name used for the directory,
        # as this is what hermes/agent loading likely expects.
        return agent_name_snake
    else:
        print(f"[resolve_agent_name] Directory not found: {agent_dir_path}")
        return None