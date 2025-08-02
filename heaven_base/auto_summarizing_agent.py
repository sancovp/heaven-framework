"""
AutoSummarizingAgent - An agent that integrates recursive summarization for context management.

This agent subclass automatically manages conversation history using the RecursiveSummarizer,
truncating old history and maintaining a hierarchical tree of summaries. It provides a foundation
for long-term memory in agents that need to handle extended conversations beyond context limits.
"""

from typing import Optional, Dict, Any, List, Union
import asyncio
import os
from datetime import datetime

from .baseheavenagent import BaseHeavenAgent, HeavenAgentConfig
from .memory.history import History
from .unified_chat import UnifiedChat
from .utils.recursive_summarize import (
    RecursiveSummarizer, 
    create_recursive_summarizer
)
from .utils.auto_summarize import auto_summarize_flag


class AutoSummarizingAgent(BaseHeavenAgent):
    """
    An agent that automatically manages conversation history using recursive summarization.
    
    This agent extends BaseHeavenAgent with:
    1. Automatic detection of when history needs summarization
    2. Integration with RecursiveSummarizer for hierarchical summary management
    3. Context injection of relevant summaries for improved agent performance
    4. Truncation of history to maintain reasonable context window usage
    
    The agent uses a project-specific RecursiveSummarizer to maintain summaries across
    multiple conversations and sessions, enabling long-term memory capabilities.
    """
    
    def __init__(
        self, 
        config: HeavenAgentConfig, 
        unified_chat: UnifiedChat, 
        history: Optional[History] = None, 
        history_id: Optional[str] = None,
        project_name: Optional[str] = None,
        project_description: Optional[str] = None,
        summarize_threshold: int = 450000,  # Character fallback when token data unavailable
        token_threshold: float = 0.8,  # Trigger at 80% of context window
        inject_summaries: bool = True,
        summarize_on_init: bool = True,
        summarize_on_complete: bool = True,
        keep_recent_iterations: int = 3,
        truncate_after_summarization: bool = True,
        on_summarization_start: Optional[callable] = None,
        on_summarization_complete: Optional[callable] = None
    ):
        """
        Initialize the AutoSummarizingAgent.
        
        Args:
            config: Agent configuration
            unified_chat: Chat provider interface
            history: Optional history object
            history_id: Optional history ID to load
            project_name: Name for the project in the RecursiveSummarizer
                          (defaults to agent name if not provided)
            project_description: Description for the project 
                                (defaults to system prompt if not provided)
            summarize_threshold: Character count threshold for triggering summarization (fallback)
            token_threshold: Percentage of context window to trigger summarization (0.0-1.0)
            inject_summaries: Whether to inject summaries back into the conversation
            summarize_on_init: Whether to check for summarization on initialization
            summarize_on_complete: Whether to check for summarization after completion
            keep_recent_iterations: Number of recent iterations to keep in full after truncation
            truncate_after_summarization: Whether to truncate history after summarization
            on_summarization_start: Callback fired when summarization begins
            on_summarization_complete: Callback fired when summarization completes
        """
        super().__init__(config, unified_chat, history, history_id)
        
        # Set default project info if not provided
        self.project_name = project_name or f"{config.name}_Project"
        self.project_description = project_description or config.system_prompt[:500]
        
        # Summarization settings
        self.summarize_threshold = summarize_threshold
        self.token_threshold = token_threshold
        self.inject_summaries = inject_summaries
        self.summarize_on_init = summarize_on_init
        self.summarize_on_complete = summarize_on_complete
        self.keep_recent_iterations = keep_recent_iterations
        self.truncate_after_summarization = truncate_after_summarization
        
        # Callbacks
        self.on_summarization_start = on_summarization_start
        self.on_summarization_complete = on_summarization_complete
        
        # Initialize summarizer
        self.summarizer = None
        self.summary_state = {
            "last_summarized_at": None,
            "has_pending_summarization": False,
            "current_summaries": {},
            "last_injected_summary": None
        }
        
        # Create summarizer if needed and check for initial summarization
        if self.summarize_on_init and self.history:
            # Use create_task to not block initialization
            asyncio.create_task(self._initialize_summarizer())
    
    async def _initialize_summarizer(self) -> None:
        """Initialize the RecursiveSummarizer and check for initial summarization."""
        # Create the RecursiveSummarizer for this agent/project
        self.summarizer = await create_recursive_summarizer(
            self.project_name, 
            self.project_description
        )
        
        # Check if we need to summarize on init
        if self.history and auto_summarize_flag(self.history, self.summarize_threshold):
            await self._perform_summarization()
    
    async def _perform_summarization(self) -> Dict[str, Any]:
        """
        Perform the summarization process on the current history.
        
        This method:
        1. Fires on_summarization_start callback
        2. Ensures the summarizer is initialized
        3. Adds the current history to the summarizer
        4. Retrieves relevant summaries 
        5. Truncates history if configured to do so
        6. Fires on_summarization_complete callback
        7. Returns the summary information
        
        Returns:
            Dict with summary information
        """
        # Fire start callback
        if self.on_summarization_start:
            current_tokens = getattr(self, 'context_window_config', None)
            token_count = current_tokens.current_tokens if current_tokens else 0
            try:
                if asyncio.iscoroutinefunction(self.on_summarization_start):
                    await self.on_summarization_start(token_count)
                else:
                    self.on_summarization_start(token_count)
            except Exception as e:
                print(f"Warning: on_summarization_start callback failed: {e}")
        
        # Ensure summarizer is initialized
        if not self.summarizer:
            self.summarizer = await create_recursive_summarizer(
                self.project_name, 
                self.project_description
            )
        
        # Save current history if needed
        if not self.history.history_id:
            self.history.save(self.config.name)
            
        # Perform summarization using RecursiveSummarizer
        await self.summarizer.add_history_summary(self.history)
        
        # Get the highest level summary
        highest_summary = self.summarizer.get_highest_level_summary()
        
        # Get all concepts for this history
        concepts = self.summarizer.get_all_concepts()
        
        # Record summarization state
        self.summary_state = {
            "last_summarized_at": datetime.now(),
            "has_pending_summarization": False,
            "current_summaries": {
                "highest_summary": highest_summary,
                "concepts": concepts
            },
            "last_injected_summary": None
        }
        
        # Truncate history if configured to do so
        if self.truncate_after_summarization:
            self._truncate_history()
        
        # Inject summaries if configured to do so
        if self.inject_summaries:
            await self._inject_summaries_as_context()
        
        # Update compactor summary buffer usage
        if hasattr(self, 'context_window_config') and self.context_window_config:
            highest_summary = self.summary_state["current_summaries"].get("highest_summary", "")
            if highest_summary:
                from .utils.token_counter import count_tokens
                summary_tokens = count_tokens(highest_summary, self.context_window_config.model)
                self.context_window_config.update_compactor_summary_tokens(summary_tokens)
        
        # Fire completion callback
        if self.on_summarization_complete:
            try:
                summary_result = self.summary_state["current_summaries"]
                if asyncio.iscoroutinefunction(self.on_summarization_complete):
                    await self.on_summarization_complete(summary_result)
                else:
                    self.on_summarization_complete(summary_result)
            except Exception as e:
                print(f"Warning: on_summarization_complete callback failed: {e}")
            
        return self.summary_state["current_summaries"]
    
    def _truncate_history(self) -> None:
        """
        Truncate the history to keep only recent messages.
        
        This method:
        1. Identifies message boundaries for iterations
        2. Keeps a configured number of recent iterations
        3. Replaces older iterations with a summary message
        """
        if not self.history or not self.history.messages:
            return
            
        # Get iterations from the history
        iterations = self.history.iterations
        total_iterations = len(iterations)
        
        # If we don't have enough iterations to truncate, just return
        if total_iterations <= self.keep_recent_iterations:
            return
            
        # Determine which iterations to keep
        iterations_to_keep = list(range(
            total_iterations - self.keep_recent_iterations, 
            total_iterations
        ))
        
        # Create a summary message with the highest level summary
        from langchain_core.messages import SystemMessage
        summary_content = "## [CONVERSATION HISTORY SUMMARY]\n\n"
        
        if self.summary_state["current_summaries"].get("highest_summary"):
            summary_content += self.summary_state["current_summaries"]["highest_summary"]
        else:
            summary_content += (
                f"This summarizes {total_iterations - self.keep_recent_iterations} "
                f"earlier iterations of this conversation."
            )
            
        summary_message = SystemMessage(content=summary_content)
        
        # Get messages from iterations to keep
        kept_messages = []
        for i in iterations_to_keep:
            iteration_key = f"iteration_{i}"
            if iteration_key in iterations:
                kept_messages.extend(iterations[iteration_key])
        
        # Reconstruct history with summary + kept messages
        new_messages = [summary_message] + kept_messages
        
        # Update history
        self.history.messages = new_messages
        
        # Save truncated history
        self.history.save(self.config.name)
        
        print(f"Truncated history from {len(self.history.messages)} to {len(new_messages)} messages")
    
    async def _inject_summaries_as_context(self) -> None:
        """
        Inject relevant summaries into the conversation as context.
        
        This adds a summary message to the history to provide context
        from previous summarized parts of the conversation.
        """
        if not self.summary_state["current_summaries"].get("highest_summary"):
            return
            
        # Create a summary system message
        from langchain_core.messages import SystemMessage
        
        # Get concepts as comma-separated list if available
        concepts_text = ""
        if self.summary_state["current_summaries"].get("concepts"):
            concepts = self.summary_state["current_summaries"]["concepts"][:5]  # Top 5 concepts
            if concepts:
                concepts_text = f"\n\nKey concepts: {', '.join(concepts)}"
        
        summary_content = (
            "## [CONVERSATION CONTEXT]\n\n"
            f"{self.summary_state['current_summaries']['highest_summary']}"
            f"{concepts_text}"
        )
        
        summary_message = SystemMessage(content=summary_content)
        
        # Add to beginning of history
        self.history.messages.insert(0, summary_message)
        
        # Record that we injected this summary
        self.summary_state["last_injected_summary"] = summary_content
        
        # Save updated history
        self.history.save(self.config.name)
    
    async def _check_and_summarize_if_needed(self) -> bool:
        """
        Check if summarization is needed and perform it if so.
        Uses workspace buffer checking when available, falls back to character-based.
        
        Returns:
            True if summarization was performed, False otherwise
        """
        # Skip if no history
        if not self.history:
            return False
        
        # Update workspace tokens from current conversation
        if hasattr(self, 'context_window_config') and self.context_window_config:
            # Count tokens in current conversation messages
            from .utils.token_counter import count_tokens_in_messages
            workspace_tokens = count_tokens_in_messages(self.history.messages, self.context_window_config.model)
            self.context_window_config.update_workspace_tokens(workspace_tokens)
            
            # Check if workspace buffer is full (primary trigger)
            if self.context_window_config.should_summarize():
                await self._perform_summarization()
                return True
        
        # Fallback to character-based check
        elif auto_summarize_flag(self.history, self.summarize_threshold):
            await self._perform_summarization()
            return True
            
        return False
    
    async def run(self, prompt: str, agent_mode: bool = False, iterations: int = 0) -> Dict[str, Any]:
        """
        Override of the run method to include summarization logic.
        
        This method:
        1. Checks if summarization is needed before running
        2. Calls the parent run method
        3. Checks if summarization is needed after running
        
        Args:
            prompt: The user prompt
            agent_mode: Whether to run in agent mode
            iterations: Number of iterations for agent mode
            
        Returns:
            The result dict from the parent run method
        """
        # Check for summarization before running
        await self._check_and_summarize_if_needed()
        
        # Call parent run method
        result = await super().run(prompt, agent_mode, iterations)
        
        # Check for summarization after running if configured
        if self.summarize_on_complete:
            summarized = await self._check_and_summarize_if_needed()
            
            # If we summarized, update result with summarization info
            if summarized and isinstance(result, dict):
                result["summarization_performed"] = True
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["summarization"] = {
                    "timestamp": self.summary_state["last_summarized_at"].isoformat(),
                    "summary_length": len(self.summary_state["current_summaries"].get("highest_summary", "")),
                    "has_been_truncated": self.truncate_after_summarization
                }
        
        return result
    
    async def get_memory_by_concept(self, concept: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve memories (summaries) related to a specific concept.
        
        This method provides semantic search across the agent's memory.
        
        Args:
            concept: The concept to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of summary dictionaries containing the found memories
        """
        # Ensure summarizer is initialized
        if not self.summarizer:
            self.summarizer = await create_recursive_summarizer(
                self.project_name, 
                self.project_description
            )
            
        # Get summaries by concept
        concept_summaries = self.summarizer.get_summary_by_concept(
            concept, 
            exact_match=False
        )
        
        # Limit results
        return concept_summaries[:max_results]
    
    async def get_memory_timeline(self, max_entries: int = 10) -> List[Dict[str, Any]]:
        """
        Get a timeline of memories (summaries) in chronological order.
        
        Args:
            max_entries: Maximum number of entries to return
            
        Returns:
            List of summary dictionaries in chronological order
        """
        # Ensure summarizer is initialized
        if not self.summarizer:
            self.summarizer = await create_recursive_summarizer(
                self.project_name, 
                self.project_description
            )
        
        # Get all summaries from level 0 (base level)
        base_summaries = self.summarizer._get_summaries_by_level(0)
        
        # Sort by timestamp
        sorted_summaries = sorted(base_summaries, key=lambda x: x.timestamp)
        
        # Format summaries
        result = []
        for entry in sorted_summaries[-max_entries:]:  # Get most recent entries
            result.append({
                "timestamp": entry.timestamp.isoformat(),
                "content": entry.content,
                "source_paths": entry.source_paths,
                "formatted_text": entry.get_formatted_content()
            })
            
        return result
    
    async def inject_memory(self, concept: Optional[str] = None) -> bool:
        """
        Inject relevant memories as context based on the current conversation.
        
        Args:
            concept: Optional concept to focus on. If not provided, 
                    will use automatic concept detection.
                    
        Returns:
            True if memories were injected, False otherwise
        """
        # Ensure summarizer is initialized
        if not self.summarizer:
            self.summarizer = await create_recursive_summarizer(
                self.project_name, 
                self.project_description
            )
        
        # If no concept provided, try to extract from recent messages
        if not concept and self.history and self.history.messages:
            # Get last 3 messages for context
            recent_messages = self.history.messages[-3:]
            recent_content = "\n".join([
                m.content if hasattr(m, "content") and isinstance(m.content, str) else str(m.content)
                for m in recent_messages
            ])
            
            # Simple concept extraction based on frequency
            from collections import Counter
            import re
            
            # Get all summaries' concepts
            all_concepts = self.summarizer.get_all_concepts()
            
            # Check which concepts appear in recent messages
            detected_concepts = []
            for c in all_concepts:
                if c.lower() in recent_content.lower():
                    detected_concepts.append(c)
            
            if detected_concepts:
                concept = detected_concepts[0]  # Use the first detected concept
        
        # If we have a concept, get related memories
        if concept:
            memories = await self.get_memory_by_concept(concept, max_results=1)
            
            if memories:
                # Create a memory injection message
                from langchain_core.messages import SystemMessage
                
                memory_content = (
                    f"## [RELEVANT MEMORY: {concept}]\n\n"
                    f"{memories[0]['formatted_text']}"
                )
                
                memory_message = SystemMessage(content=memory_content)
                
                # Add to beginning of history
                self.history.messages.insert(0, memory_message)
                
                # Save updated history
                self.history.save(self.config.name)
                
                return True
        
        return False