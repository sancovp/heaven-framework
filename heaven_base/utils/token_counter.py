"""
Token counting utilities for precise context management.

Provides accurate token counting for OpenAI models using tiktoken,
with fallback approximations for other models.
"""

import tiktoken
from typing import Optional, List, Union
from langchain_core.messages import BaseMessage


def count_tokens(
    text: str, 
    model: str = "gpt-4o-mini"
) -> int:
    """
    Count tokens in text using tiktoken for OpenAI models.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding selection
        
    Returns:
        int: Number of tokens in the text
    """
    try:
        # Map model names to tiktoken encodings
        if "gpt-4" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "o1" in model.lower() or "o3" in model.lower() or "o4" in model.lower():
            # Use gpt-4 encoding for o-series models
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            # Default to gpt-4 encoding for other models (Claude, Gemini, etc.)
            encoding = tiktoken.encoding_for_model("gpt-4")
        
        return len(encoding.encode(text))
        
    except Exception as e:
        # Fallback to rough approximation if tiktoken fails
        # OpenAI's rule of thumb: ~4 characters per token
        return max(1, len(text) // 4)


def count_tokens_in_messages(
    messages: List[Union[BaseMessage, dict]], 
    model: str = "gpt-4o-mini"
) -> int:
    """
    Count tokens in a list of messages.
    
    Args:
        messages: List of LangChain messages or dict messages
        model: Model name for encoding selection
        
    Returns:
        int: Total number of tokens in all messages
    """
    total_tokens = 0
    
    for message in messages:
        if isinstance(message, BaseMessage):
            # LangChain message
            text = str(message.content)
        elif isinstance(message, dict):
            # Dict message (uni-api format)
            text = message.get("content", "")
        else:
            # Fallback: convert to string
            text = str(message)
        
        total_tokens += count_tokens(text, model)
    
    return total_tokens


def estimate_tokens_for_model(text: str, model: str) -> int:
    """
    Estimate tokens for different model families with family-specific approximations.
    
    Args:
        text: Text to estimate tokens for
        model: Model name
        
    Returns:
        int: Estimated number of tokens
    """
    model_lower = model.lower()
    
    if "gpt" in model_lower or "openai" in model_lower:
        # Use tiktoken for OpenAI models
        return count_tokens(text, model)
    elif "claude" in model_lower:
        # Claude tokenization is similar to GPT-4 but slightly different
        # Rough approximation: 3.8 chars per token
        return max(1, len(text) // 4)
    elif "gemini" in model_lower:
        # Gemini has different tokenization
        # Rough approximation: 4.2 chars per token  
        return max(1, len(text) // 4)
    else:
        # Generic fallback
        return max(1, len(text) // 4)


class TokenBudget:
    """
    Utility class for managing token budgets across conversation sections.
    """
    
    def __init__(self, total_budget: int, model: str = "gpt-4o-mini"):
        self.total_budget = total_budget
        self.model = model
        self.allocations = {}
        self.used_tokens = {}
    
    def allocate(self, section: str, tokens: int) -> None:
        """Allocate tokens to a section."""
        self.allocations[section] = tokens
        if section not in self.used_tokens:
            self.used_tokens[section] = 0
    
    def use_tokens(self, section: str, text: str) -> bool:
        """
        Use tokens for a section. Returns True if within budget.
        
        Args:
            section: Section name
            text: Text to count tokens for
            
        Returns:
            bool: True if within section budget, False otherwise
        """
        tokens_needed = count_tokens(text, self.model)
        
        if section not in self.allocations:
            return False
        
        if self.used_tokens.get(section, 0) + tokens_needed <= self.allocations[section]:
            self.used_tokens[section] = self.used_tokens.get(section, 0) + tokens_needed
            return True
        
        return False
    
    def get_remaining(self, section: str) -> int:
        """Get remaining tokens for a section."""
        allocated = self.allocations.get(section, 0)
        used = self.used_tokens.get(section, 0)
        return max(0, allocated - used)
    
    def get_total_used(self) -> int:
        """Get total tokens used across all sections."""
        return sum(self.used_tokens.values())
    
    def get_total_remaining(self) -> int:
        """Get total remaining budget."""
        return max(0, self.total_budget - self.get_total_used())
    
    def reset_section(self, section: str) -> None:
        """Reset usage for a section."""
        self.used_tokens[section] = 0
    
    def get_budget_summary(self) -> dict:
        """Get a summary of budget usage."""
        return {
            "total_budget": self.total_budget,
            "total_used": self.get_total_used(),
            "total_remaining": self.get_total_remaining(),
            "sections": {
                section: {
                    "allocated": self.allocations.get(section, 0),
                    "used": self.used_tokens.get(section, 0),
                    "remaining": self.get_remaining(section)
                }
                for section in set(list(self.allocations.keys()) + list(self.used_tokens.keys()))
            }
        }


# Convenience functions for common use cases
def count_conversation_tokens(messages: List[Union[BaseMessage, dict]], model: str = "gpt-4o-mini") -> int:
    """Count tokens in a full conversation."""
    return count_tokens_in_messages(messages, model)


def check_token_limit(text: str, limit: int, model: str = "gpt-4o-mini") -> bool:
    """Check if text is within token limit."""
    return count_tokens(text, model) <= limit


def truncate_to_token_limit(text: str, limit: int, model: str = "gpt-4o-mini") -> str:
    """
    Truncate text to fit within token limit.
    
    This is a rough approximation - for precise truncation,
    you'd need to decode tokens back to text.
    """
    current_tokens = count_tokens(text, model)
    
    if current_tokens <= limit:
        return text
    
    # Rough truncation based on character ratio
    char_ratio = len(text) / current_tokens
    target_chars = int(limit * char_ratio * 0.9)  # 10% safety margin
    
    return text[:target_chars] + "..."


if __name__ == "__main__":
    # Example usage
    test_text = "Hello world! This is a test of the token counting system."
    
    print(f"Text: {test_text}")
    print(f"Tokens (GPT-4): {count_tokens(test_text, 'gpt-4o-mini')}")
    print(f"Tokens (Claude): {estimate_tokens_for_model(test_text, 'claude-3-5-sonnet')}")
    
    # Test budget system
    budget = TokenBudget(1000, "gpt-4o-mini")
    budget.allocate("summary", 300)
    budget.allocate("history", 600)
    budget.allocate("prompt", 100)
    
    print(f"\nBudget summary: {budget.get_budget_summary()}")