"""
ContextWindowConfig - Token-aware context window management for HEAVEN agents.

This module provides precise token counting and context window management
for different AI models, integrating with tiktoken for OpenAI models
and providing fallback approximations for other models.
"""

from typing import Dict, Optional, Any
from .token_counter import count_tokens, count_tokens_in_messages

# Model context window limits (in tokens)
MODEL_LIMITS = {
    # OpenAI models
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8000,
    "gpt-3.5-turbo": 16000,
    "o1-preview": 128000,
    "o1-mini": 128000,
    "o3-mini": 128000,
    
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    
    # Google Gemini models
    "gemini-1.5-pro": 2000000,
    "gemini-1.5-flash": 1000000,
    "gemini-pro": 30720,
    
    # Other models
    "llama-3.1-70b": 128000,
    "llama-3.1-8b": 128000,
    "deepseek-coder": 16000,
    "qwen2.5-coder": 32000,
    
    # Default fallback
    "default": 8000
}


class ContextWindowConfig:
    """
    Manages context window limits and token counting for AI models.
    
    Features:
    - Model-specific context window limits
    - Token counting using tiktoken (OpenAI) or approximations (others)
    - Safety margins to prevent context overflow
    - Integration with uni-api token usage data
    """
    
    def __init__(
        self, 
        model: str,
        effective_ratio: float = 0.85,  # Use 85% of total context window
        compactor_summary_ratio: float = 0.15,  # 15% for compactor summary
        recursive_summaries_ratio: float = 0.25,  # 25% for recursive stack  
        workspace_ratio: float = 0.45,  # 45% for conversation workspace
        padding_ratio: float = 0.10,  # 10% padding alert zone
        limit_ratio: float = 0.05  # 5% hard limit zone
    ):
        """
        Initialize context window configuration with engineered buffer allocations.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet")
            effective_ratio: Percentage of total context window to use (0.0-1.0)
            compactor_summary_ratio: Percentage allocated for compactor summaries
            recursive_summaries_ratio: Percentage allocated for recursive summary stack
            workspace_ratio: Percentage allocated for conversation workspace
            padding_ratio: Percentage for padding alert zone
            limit_ratio: Percentage for hard limit zone
        """
        self.model = model
        self.effective_ratio = effective_ratio
        
        # Get model limits
        self.model_max_window = self._get_model_limit(model)
        self.usable_window = int(self.model_max_window * effective_ratio)
        
        # ENGINEERED BUFFER ALLOCATIONS
        self.compactor_summary_buffer = int(self.usable_window * compactor_summary_ratio)
        self.recursive_summaries_buffer = int(self.usable_window * recursive_summaries_ratio)
        self.workspace_buffer = int(self.usable_window * workspace_ratio)
        self.padding_buffer = int(self.usable_window * padding_ratio)
        self.limit_buffer = int(self.usable_window * limit_ratio)
        
        # CURRENT USAGE TRACKING
        self.compactor_summary_tokens = 0
        self.recursive_summaries_tokens = 0
        self.workspace_tokens = 0
        
        # Legacy support
        self.current_tokens = 0
        self.last_uni_api_usage = None
        
    def _get_model_limit(self, model: str) -> int:
        """Get context window limit for a model."""
        # Direct match
        if model in MODEL_LIMITS:
            return MODEL_LIMITS[model]
            
        # Fuzzy matching for model variants
        model_lower = model.lower()
        for model_key, limit in MODEL_LIMITS.items():
            if model_key.lower() in model_lower or model_lower in model_key.lower():
                return limit
                
        # Default fallback
        return MODEL_LIMITS["default"]
    
    def update_from_uni_api(self, usage_data: Dict[str, Any]) -> None:
        """
        Update token count from uni-api response.
        This updates the workspace tokens with the total conversation usage.
        
        Args:
            usage_data: Usage data from uni-api response
                       (e.g., {"total_tokens": 1234, "prompt_tokens": 800, "completion_tokens": 434})
        """
        if isinstance(usage_data, dict):
            total_tokens = usage_data.get("total_tokens", 0)
            # Update workspace with the conversation tokens (excluding any summaries already counted)
            self.update_workspace_tokens(total_tokens)
            self.last_uni_api_usage = usage_data
        else:
            # Fallback if not dict format
            self.update_workspace_tokens(0)
    
    def update_from_messages(self, messages) -> None:
        """
        Update workspace token count using tiktoken estimation from messages.
        
        Args:
            messages: List of messages to count tokens for
        """
        if messages:
            workspace_tokens = count_tokens_in_messages(messages, self.model)
            self.update_workspace_tokens(workspace_tokens)
        else:
            self.update_workspace_tokens(0)
    
    def update_from_text(self, text: str) -> None:
        """
        Update workspace token count from raw text.
        
        Args:
            text: Text to count tokens for
        """
        workspace_tokens = count_tokens(text, self.model)
        self.update_workspace_tokens(workspace_tokens)
    
    def update_compactor_summary_tokens(self, tokens: int) -> None:
        """Update compactor summary token usage."""
        self.compactor_summary_tokens = tokens
        self.current_tokens = self.get_total_usage()  # Legacy support
    
    def update_recursive_summaries_tokens(self, tokens: int) -> None:
        """Update recursive summaries token usage."""
        self.recursive_summaries_tokens = tokens
        self.current_tokens = self.get_total_usage()  # Legacy support
    
    def update_workspace_tokens(self, tokens: int) -> None:
        """Update workspace (conversation) token usage."""
        self.workspace_tokens = tokens
        self.current_tokens = self.get_total_usage()  # Legacy support
    
    def get_total_usage(self) -> int:
        """Get total tokens used across all buffers."""
        return (
            self.compactor_summary_tokens + 
            self.recursive_summaries_tokens + 
            self.workspace_tokens
        )
    
    def should_summarize(self) -> bool:
        """
        Check if compaction should be triggered.
        
        Triggers when workspace buffer is approaching its limit.
        
        Returns:
            True if workspace usage indicates need for compaction
        """
        return self.workspace_tokens >= self.workspace_buffer
    
    def approaching_limit(self) -> bool:
        """
        Check if we're approaching the padding alert zone.
        
        Returns:
            True if total usage has entered padding buffer zone
        """
        total_used = self.get_total_usage()
        return total_used >= (self.usable_window - self.padding_buffer)
    
    def at_hard_limit(self) -> bool:
        """
        Check if we've hit the hard limit zone.
        
        Returns:
            True if total usage has entered hard limit zone
        """
        total_used = self.get_total_usage()
        return total_used >= (self.usable_window - self.limit_buffer)
    
    def get_remaining_workspace_tokens(self) -> int:
        """Get remaining tokens available in workspace buffer."""
        return max(0, self.workspace_buffer - self.workspace_tokens)
    
    def get_buffer_usage_percentage(self, buffer_name: str) -> float:
        """Get usage percentage for a specific buffer."""
        if buffer_name == "compactor_summary":
            if self.compactor_summary_buffer == 0:
                return 0.0
            return (self.compactor_summary_tokens / self.compactor_summary_buffer) * 100
        elif buffer_name == "recursive_summaries":
            if self.recursive_summaries_buffer == 0:
                return 0.0
            return (self.recursive_summaries_tokens / self.recursive_summaries_buffer) * 100
        elif buffer_name == "workspace":
            if self.workspace_buffer == 0:
                return 0.0
            return (self.workspace_tokens / self.workspace_buffer) * 100
        else:
            return 0.0
    
    def exceeds_limit(self) -> bool:
        """Legacy method - check if we've exceeded any limits."""
        return self.at_hard_limit()
    
    def get_remaining_tokens(self) -> int:
        """Legacy method - get remaining workspace tokens."""
        return self.get_remaining_workspace_tokens()
    
    def get_usage_percentage(self) -> float:
        """Legacy method - get total usage percentage."""
        if self.usable_window == 0:
            return 0.0
        return (self.get_total_usage() / self.usable_window) * 100
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of context window usage.
        
        Returns:
            Dictionary with usage statistics and limits
        """
        return {
            "model": self.model,
            "model_max_window": self.model_max_window,
            "usable_window": self.usable_window,
            
            # Buffer allocations
            "buffers": {
                "compactor_summary": {
                    "allocated": self.compactor_summary_buffer,
                    "used": self.compactor_summary_tokens,
                    "usage_percentage": round(self.get_buffer_usage_percentage("compactor_summary"), 2)
                },
                "recursive_summaries": {
                    "allocated": self.recursive_summaries_buffer,
                    "used": self.recursive_summaries_tokens,
                    "usage_percentage": round(self.get_buffer_usage_percentage("recursive_summaries"), 2)
                },
                "workspace": {
                    "allocated": self.workspace_buffer,
                    "used": self.workspace_tokens,
                    "remaining": self.get_remaining_workspace_tokens(),
                    "usage_percentage": round(self.get_buffer_usage_percentage("workspace"), 2)
                },
                "padding": self.padding_buffer,
                "limit": self.limit_buffer
            },
            
            # Overall status
            "total_used": self.get_total_usage(),
            "total_usage_percentage": round(self.get_usage_percentage(), 2),
            "should_summarize": self.should_summarize(),
            "approaching_limit": self.approaching_limit(),
            "at_hard_limit": self.at_hard_limit(),
            
            # Legacy support
            "current_tokens": self.current_tokens,
            "remaining_tokens": self.get_remaining_tokens(),
            "exceeds_limit": self.exceeds_limit(),
            "last_uni_api_usage": self.last_uni_api_usage
        }
    
    
    def __str__(self) -> str:
        """String representation of context window status."""
        status = self.get_status()
        return (
            f"ContextWindow({status['model']}: "
            f"{status['current_tokens']}/{status['usable_window']} tokens, "
            f"{status['usage_percentage']}%)"
        )
    
    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"ContextWindowConfig(model='{self.model}', "
            f"current_tokens={self.current_tokens}, "
            f"usable_window={self.usable_window})"
        )


def create_context_window_config(model: str, **kwargs) -> ContextWindowConfig:
    """
    Factory function to create ContextWindowConfig with defaults.
    
    Args:
        model: Model name
        **kwargs: Additional configuration options
        
    Returns:
        Configured ContextWindowConfig instance
    """
    return ContextWindowConfig(model, **kwargs)


if __name__ == "__main__":
    # Example usage
    config = ContextWindowConfig("gpt-4o-mini")
    print(f"Context window config: {config}")
    print(f"Status: {config.get_status()}")
    
    # Test with some usage
    config.update_from_text("Hello world! " * 1000)
    print(f"After adding text: {config}")
    print(f"Should summarize: {config.should_summarize()}")