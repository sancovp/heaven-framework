import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter

class PromptBlockRegistry:
    def __init__(self):
        self.blocks: List[Dict] = []
        self._load_blocks()

    def _load_blocks(self) -> None:
        """Load all prompt blocks from the blocks directory."""
        blocks_dir = Path(__file__).parent / 'blocks'

        # Create directory if it doesn't exist
        blocks_dir.mkdir(exist_ok=True)

        # Load all JSON files in the blocks directory
        for json_file in blocks_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    block_data = json.load(f)
                    self.blocks.append(block_data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    def get_prompt_block(self, block_name: str) -> Optional[str]:
        """
        Retrieve a prompt block's text by its name.

        Args:
            block_name (str): Name of the prompt block to retrieve

        Returns:
            Optional[str]: The prompt block text if found, None otherwise
        """
        for block in self.blocks:
            if block['name'].lower() == block_name.lower():
                return block['text']
        return None

    def get_blocks_by_domain(self, domain: str) -> List[Dict]:
        """
        Get all blocks for a specific domain.

        Args:
            domain (str): Domain to filter by

        Returns:
            List[Dict]: List of matching blocks
        """
        return [block for block in self.blocks if block['domain'].lower() == domain.lower()]

    def get_blocks_by_subdomain(self, subdomain: str) -> List[Dict]:
        """
        Get all blocks for a specific subdomain.

        Args:
            subdomain (str): Subdomain to filter by

        Returns:
            List[Dict]: List of matching blocks
        """
        return [block for block in self.blocks if block['subdomain'].lower() == subdomain.lower()]

    def search_blocks_by_similarity(self, query: str, top_n: int = 5, min_similarity: float = 0.1) -> List[Tuple[Dict, float]]:
        """
        Search for prompt blocks similar to the given query using a simple cosine similarity.

        Args:
            query (str): The search query
            top_n (int): Maximum number of results to return
            min_similarity (float): Minimum similarity score (0-1) for results

        Returns:
            List[Tuple[Dict, float]]: List of (block, similarity_score) tuples, sorted by similarity
        """
        if not self.blocks:
            return []

        # Compute similarity between query and each block
        results = []
        for block in self.blocks:
            similarity = self._compute_similarity(query, block['text'])
            if similarity >= min_similarity:
                results.append((block, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]

    def _tokenize(self, text):
        """Simple tokenization function that splits text into words"""
        # Convert to lowercase and split on non-alphanumeric characters
        return re.findall(r'\w+', text.lower())

    def _compute_similarity(self, text1, text2):
        """
        Compute a simple similarity score between two texts
        using word overlap and term frequency
        """
        # Tokenize texts
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)

        # Count token frequencies
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)

        # Get all unique tokens
        all_tokens = set(tokens1).union(set(tokens2))
        if not all_tokens:
            return 0.0

        # Calculate dot product
        dot_product = sum(counter1.get(token, 0) * counter2.get(token, 0) for token in all_tokens)

        # Calculate magnitudes
        magnitude1 = sum(count * count for count in counter1.values()) ** 0.5
        magnitude2 = sum(count * count for count in counter2.values()) ** 0.5

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)

# Create a global instance of the registry
_registry = None

def get_registry():
    global _registry
    if _registry is None:
        _registry = PromptBlockRegistry()
    return _registry

# Convenience functions
def get_prompt_block(block_name: str) -> Optional[str]:
    """
    Convenience function to get a prompt block's text by name.

    Args:
        block_name (str): Name of the prompt block to retrieve

    Returns:
        Optional[str]: The prompt block text if found, None otherwise
    """
    return get_registry().get_prompt_block(block_name)

def search_blocks_by_similarity(query: str, top_n: int = 5, min_similarity: float = 0.1) -> List[Tuple[Dict, float]]:
    """
    Convenience function to search for prompt blocks by similarity.

    Args:
        query (str): The search query
        top_n (int): Maximum number of results to return
        min_similarity (float): Minimum similarity score (0-1) for results

    Returns:
        List[Tuple[Dict, float]]: List of (block, similarity_score) tuples, sorted by similarity
    """
    return get_registry().search_blocks_by_similarity(query, top_n, min_similarity)
