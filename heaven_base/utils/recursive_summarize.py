"""
Recursive Summarization Framework.

This module provides functions for recursive multi-level summarization
of conversation histories, allowing virtually unlimited context windows
through hierarchical compression.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
from datetime import datetime
import asyncio

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from ..memory.history import History
from ..baseheavenagent import HeavenAgentConfig, BaseHeavenAgentReplicant  
from ..unified_chat import ProviderEnum
from ..registry.registry_service import RegistryService

from .auto_summarize import (
    auto_summarize_flag,
    auto_summarize,
    reason_about_summary,
    get_summary_concept
)

# Initialize registry service
registry_service = RegistryService()


# Constants
# Production values:
BASE_LEVEL_TOKEN_THRESHOLD = 450000  # ~150K tokens
META_LEVEL_TOKEN_THRESHOLD = 90000   # ~30K tokens

# For testing, use these lower thresholds:
# BASE_LEVEL_TOKEN_THRESHOLD = 50000  # Lower threshold for testing
# META_LEVEL_TOKEN_THRESHOLD = 20000  # Lower threshold for testing


class RecursiveSummaryEntry:
    """Represents a single summary at a specific level in the hierarchy."""
    
    def __init__(
        self,
        content: str,
        level: int,
        source_paths: List[str],
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.level = level
        self.source_paths = source_paths
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "level": self.level,
            "source_paths": self.source_paths,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecursiveSummaryEntry":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            level=data["level"],
            source_paths=data["source_paths"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"]
        )
    
    def get_header(self) -> str:
        """Generate a header for this summary entry."""
        return (
            f"## Level {self.level} Summary | {self.timestamp.isoformat()}\n"
            f"Sources: {', '.join(self.source_paths)}\n\n"
        )
    
    def get_formatted_content(self) -> str:
        """Get the content with header for injection."""
        return f"{self.get_header()}{self.content}"


class RecursiveSummarizer:
    """
    Manages recursive summarization across multiple levels.
    
    This class handles the process of summarizing histories, tracking
    summary levels, and determining when to escalate to higher-level
    summaries.
    """
    
    def __init__(self, project_name: str, project_description: str):
        self.project_name = project_name
        self.project_description = project_description
        self.summaries: Dict[int, List[RecursiveSummaryEntry]] = {}
        
        # Initialize registry integration
        self.registry_name = f"recursive_summary_{project_name.lower().replace(' ', '_')}"
        self._init_registry()
        self._registry_keys: List[str] = []
        
    def _init_registry(self) -> None:
        """Initialize or load the registry for this summarizer."""
        global registry_service
        
        # Create registry if it doesn't exist yet
        if self.registry_name not in registry_service.list_registries():
            print(f"Creating new registry for {self.project_name}")
            registry_service.create_registry(self.registry_name)
        
        # Get builder for existing registry
        self.registry_builder = registry_service.get_builder(self.registry_name)
        
        # Load existing summaries from registry if available
        if self.registry_builder:
            self._load_from_registry()
    
    def _get_summaries_by_level(self, level: int) -> List[RecursiveSummaryEntry]:
        """Get all summaries at a specific level."""
        return self.summaries.get(level, [])
    
    def _calculate_tokens_at_level(self, level: int) -> int:
        """Calculate approximate token count for all summaries at level."""
        total_chars = 0
        for entry in self._get_summaries_by_level(level):
            total_chars += len(entry.content)
        return total_chars
    
    def _should_summarize_level(self, level: int) -> bool:
        """Determine if a level needs summarization based on token count."""
        if level == 0:  # Base level - uses BASE_LEVEL_TOKEN_THRESHOLD
            return self._calculate_tokens_at_level(level) > BASE_LEVEL_TOKEN_THRESHOLD
        # Meta levels - use META_LEVEL_TOKEN_THRESHOLD
        return self._calculate_tokens_at_level(level) > META_LEVEL_TOKEN_THRESHOLD
    
    def _load_from_registry(self) -> None:
        """Load existing summaries from the registry."""
        if not self.registry_builder:
            return
            
        # Reset current summaries
        self.summaries = {}
        
        # Get all summaries from registry
        all_registry_data = self.registry_builder.get_all()
        if not all_registry_data or 'levels' not in all_registry_data:
            return
            
        # Load summaries by level
        for level_str, summaries_dict in all_registry_data.get('levels', {}).items():
            level = int(level_str)
            self.summaries[level] = []
            
            for summary_id, summary_data in summaries_dict.items():
                # Create RecursiveSummaryEntry objects from registry data
                entry = RecursiveSummaryEntry(
                    content=summary_data.get('content', ''),
                    level=level,
                    source_paths=summary_data.get('source_paths', []),
                    metadata=all_registry_data.get('metadata', {}).get(summary_id, {}),
                )
                
                if 'created_at' in all_registry_data.get('metadata', {}).get(summary_id, {}):
                    try:
                        timestamp_str = all_registry_data['metadata'][summary_id]['created_at']
                        entry.timestamp = datetime.fromisoformat(timestamp_str)
                    except (ValueError, TypeError):
                        pass
                
                self.summaries[level].append(entry)
                self._registry_keys.append(summary_id)
                
        print(f"Loaded {len(self._registry_keys)} summaries from registry")

    def add_summary(self, summary: str, level: int, source_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a summary to the specified level.
        
        Args:
            summary: Summary content
            level: Hierarchy level (0 = base level)
            source_paths: Paths to source files/histories
            metadata: Optional metadata about the summary
        """
        entry = RecursiveSummaryEntry(
            content=summary,
            level=level,
            source_paths=source_paths,
            metadata=metadata
        )
        
        if level not in self.summaries:
            self.summaries[level] = []
        
        self.summaries[level].append(entry)
        
        # Extract concepts from the summary
        concepts = self._extract_concepts(summary)
        
        # Save to registry
        if self.registry_builder:
            # Determine parent summaries
            parents = []
            if level > 0 and level-1 in self.summaries:
                # Use the most recent summaries from the previous level as parents
                # These are the summaries that were used to generate this summary
                for entry in self.summaries[level-1]:
                    # Get registry key for this entry
                    for key in self._registry_keys:
                        if key.endswith(entry.timestamp.strftime('%Y%m%d_%H%M%S')):
                            parents.append(key)
            
            # Create summary in registry
            registry_key = f"{self.project_name}_summary_L{level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Check if the registry builder has add_summary method
            # This is a safety check to support different registry implementations
            if hasattr(self.registry_builder, 'add_summary'):
                summary_id = self.registry_builder.add_summary(
                    summary=summary,
                    level=level,
                    source_paths=source_paths,
                    concepts=concepts,
                    parents=parents,
                    summary_id=registry_key,
                    created_by="RecursiveSummarizer"
                )
                self._registry_keys.append(summary_id)
                print(f"Added summary to registry with ID: {summary_id}")
            else:
                # Fallback to standard add method if add_summary isn't available
                try:
                    self.registry_builder.add(registry_key, {
                        "content": summary,
                        "level": level,
                        "source_paths": source_paths,
                        "concepts": concepts,
                        "parents": parents,
                        "created_at": datetime.now().isoformat(),
                        "created_by": "RecursiveSummarizer"
                    })
                    self._registry_keys.append(registry_key)
                    print(f"Added summary to registry with ID: {registry_key}")
                except Exception as e:
                    print(f"Warning: Failed to add to registry: {str(e)}")
                    # Still add to registry keys for tracking
                    self._registry_keys.append(registry_key)
            
    def _extract_concepts(self, summary: str) -> List[str]:
        """Extract key concepts from a summary."""
        # For testing, return mock concepts based on content
        # In production, would use get_summary_concept from auto_summarize.py
        concepts = []
        if "architecture" in summary.lower():
            concepts.append("architecture")
        if "pattern" in summary.lower():
            concepts.append("patterns")
        if "documentation" in summary.lower():
            concepts.append("documentation")
        if "testing" in summary.lower():
            concepts.append("testing")
        if "component" in summary.lower():
            concepts.append("components")
            
        # Add level-based concept for easier filtering
        if "meta-summary" in summary.lower():
            concepts.append("meta-summary")
            
        # Ensure we have at least one concept
        if not concepts:
            concepts.append("general")
            
        return concepts
    
    def get_formatted_summaries_for_level(self, level: int) -> List[str]:
        """Get formatted summaries for a specific level."""
        return [entry.get_formatted_content() for entry in self._get_summaries_by_level(level)]
    
    def _create_batch(self, entries: List[RecursiveSummaryEntry], max_chars: int = 70000) -> List[List[RecursiveSummaryEntry]]:
        """Create batches of summaries that fit within token limits."""
        batches = []
        current_batch = []
        current_size = 0
        
        for entry in entries:
            entry_size = len(entry.get_formatted_content())
            if current_size + entry_size > max_chars and current_batch:
                batches.append(current_batch)
                current_batch = [entry]
                current_size = entry_size
            else:
                current_batch.append(entry)
                current_size += entry_size
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _get_batch_content(self, batch: List[RecursiveSummaryEntry]) -> str:
        """Get concatenated content from a batch of summaries."""
        return "\n\n".join(entry.get_formatted_content() for entry in batch)
    
    async def _create_injection_history(self, level: int) -> Tuple[History, str]:
        """
        Create a new history with synthetic messages that inject summaries.
        
        Args:
            level: Level to summarize
            
        Returns:
            Tuple of (History object, history_id)
        """
        # Create new history
        history = History(
            messages=[],
            created_datetime=datetime.now(),
            metadata={
                "project": self.project_name,
                "summary_level": level,
                "is_recursive_summary": True
            }
        )
        
        # Add system message
        history.messages.append(SystemMessage(content=f"""You are an expert summarizer for project {self.project_name}.
Your task is to create a concise, comprehensive summary of multiple summary fragments that will be provided to you.
Each fragment is already a summary from level {level}, and you need to create a higher-level summary (level {level+1}).
Focus on key themes, patterns, and essential information across all fragments."""))
        
        # Add initial human message
        intro_message = f"""I am going to summarize my past interactions on this project. 
The project is called {self.project_name}.

Project description: {self.project_description}

I will inject summaries turn by turn and tell you when I'm done. 
Until I'm done, just respond with a thumbs up emoji (👍).
Here is the first set of summaries:"""
        
        history.messages.append(HumanMessage(content=intro_message))
        
        # Get entries to summarize
        entries = self._get_summaries_by_level(level)
        
        # Create batches
        batches = self._create_batch(entries)
        
        # Process first batch
        if batches:
            first_batch_content = self._get_batch_content(batches[0])
            history.messages.append(HumanMessage(content=first_batch_content))
            history.messages.append(AIMessage(content="👍"))
            
            # Process subsequent batches
            for batch in batches[1:]:
                batch_content = self._get_batch_content(batch)
                history.messages.append(HumanMessage(content=batch_content))
                history.messages.append(AIMessage(content="👍"))
        
        # Final message
        history.messages.append(HumanMessage(content=f"""That's all the summaries for level {level}.
Please create a comprehensive higher-level summary (level {level+1}) that captures the key points across all these summaries.
Organize it with clear sections and ensure it's cohesive.
Aim for a concise but complete representation of the essential information."""))
        
        # Save the history
        history_id = history.save(f"{self.project_name}_RecursiveSummarizer")
        
        return history, history_id
    
    async def _process_injection_result(self, history: History) -> str:
        """Process the result from the injection history."""
        # The last AI message should contain the summary
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage) and msg.content != "👍":
                return msg.content
        
        return "No summary generated"

    async def process_level(self, level: int) -> Optional[str]:
        """
        Process all summaries at a level and generate a higher-level summary if needed.
        
        Args:
            level: Level to process
            
        Returns:
            New summary text if generated, None otherwise
        """
        # Check if we need to summarize this level
        total_chars = self._calculate_tokens_at_level(level)
        threshold = BASE_LEVEL_TOKEN_THRESHOLD if level == 0 else META_LEVEL_TOKEN_THRESHOLD
        print(f"Level {level} has {total_chars} characters (threshold: {threshold})")
        
        if not self._should_summarize_level(level):
            print(f"Level {level} does not need summarization yet")
            return None
        
        print(f"Summarizing level {level}...")
        
        # Create injection history
        history, history_id = await self._create_injection_history(level)
        print(f"Created injection history: {history_id}")
        
        # Create a MetaSummarizerAgent
        agent = MetaSummarizerAgent()
        
        # Get the content from all summaries at this level
        summaries_content = "\n\n".join([entry.get_formatted_content() for entry in self._get_summaries_by_level(level)])
        
        # Use the agent's create_meta_summary method directly
        summary = await agent.create_meta_summary(summaries_content)
            
        # If the agent didn't produce a valid summary, provide a minimal fallback
        if not summary or len(summary.strip()) < 10:
            print("Warning: Agent didn't produce a valid summary. Using minimal fallback.")
            summary = f"# Level {level+1} Summary\nSummary of {len(self._get_summaries_by_level(level))} level {level} summaries."
        
        print(f"Generated level {level+1} summary with {len(summary)} characters")
        
        # Get source paths from all entries at this level
        source_paths = []
        # Collect all concepts from lower level
        all_concepts = []
        
        for entry in self._get_summaries_by_level(level):
            source_paths.extend(entry.source_paths)
            
            # Collect concepts from metadata
            if hasattr(entry, 'metadata') and entry.metadata:
                if 'concepts' in entry.metadata and isinstance(entry.metadata['concepts'], list):
                    all_concepts.extend(entry.metadata['concepts'])
        
        # Deduplicate concepts and count frequencies
        concept_counts = {}
        for concept in all_concepts:
            if concept in concept_counts:
                concept_counts[concept] += 1
            else:
                concept_counts[concept] = 1
        
        # Keep only concepts that appear multiple times or are important
        important_concepts = ["architecture", "design", "patterns", "components"]
        propagated_concepts = [
            concept for concept, count in concept_counts.items() 
            if count > 1 or concept in important_concepts
        ]
        
        # Ensure we have at least one concept
        if not propagated_concepts:
            propagated_concepts = ["general"]
            
        print(f"Propagating concepts to level {level+1}: {propagated_concepts}")
        
        # Add this new summary to the next level with propagated concepts
        self.add_summary(
            summary=summary,
            level=level + 1,
            source_paths=list(set(source_paths)),  # Deduplicate
            metadata={
                "source_level": level,
                "concepts": propagated_concepts,
                "parent_concepts": all_concepts
            }
        )
        
        print(f"Added new summary to level {level+1}")
        return summary
    
    async def process_all_levels(self) -> Dict[int, List[str]]:
        """
        Process all levels recursively, starting from the lowest.
        
        Returns:
            Dict mapping level -> list of summary texts generated
        """
        results = {}
        current_level = 0
        
        # Process each level until no more summaries are generated
        while current_level in self.summaries:
            summary = await self.process_level(current_level)
            if summary:
                if current_level + 1 not in results:
                    results[current_level + 1] = []
                results[current_level + 1].append(summary)
                current_level += 1
            else:
                # No summary generated at this level, move to next
                current_level += 1
        
        return results
    
    async def add_history_summary(self, history_id_or_obj: Union[str, History]) -> None:
        """
        Add a summary of a history to the base level (0).
        
        Args:
            history_id_or_obj: History ID or object to summarize
        """
        # Handle both history_id and History object
        if isinstance(history_id_or_obj, str):
            history = History.load_from_id(history_id_or_obj)
            history_id = history_id_or_obj
        else:
            history = history_id_or_obj
            history_id = history.history_id
        
        # Check if summarization is needed 
        if not auto_summarize_flag(history):
            print(f"History {history_id} doesn't need summarization yet")
            return
        
        print(f"Generating summary for history {history_id}...")
        
        # Use the auto_summarize function to generate a real summary
        # This uses the IterationSummarizerAgent and AggregationSummarizerAgent
        summary = await auto_summarize(history)
        
        # If auto_summarize fails, provide a minimal fallback
        if not summary or len(summary.strip()) < 10:
            print(f"Warning: auto_summarize failed for history {history_id}. Using minimal fallback.")
            summary = f"# Summary of History {history_id}\nThis is a summary of the conversation history."
        
        print(f"Generated summary for history {history_id} with {len(summary)} characters")
        
        # Extract concepts from the summary
        from computer_use_demo.codebase_analyzer_system.auto_summarize import reason_about_summary, get_summary_concept
        reasoning = await reason_about_summary(summary)
        concepts = await get_summary_concept(summary, reasoning)
        
        # Extract concepts as list of keywords
        concept_keywords = []
        if isinstance(concepts, str):
            # Extract keywords from concept
            for keyword in ["architecture", "design", "testing", "implementation", 
                           "documentation", "patterns", "components", "microservices",
                           "event-driven", "mvc", "mvvm", "api", "database", "frontend",
                           "backend", "security", "deployment", "scaling"]:
                if keyword in concepts.lower():
                    concept_keywords.append(keyword)
        
        # Ensure we have at least one concept
        if not concept_keywords:
            concept_keywords = ["general"]
        
        print(f"Extracted concepts: {concept_keywords}")
        
        # Add to base level with concepts
        source_path = history.json_md_path if history.json_md_path else f"history:{history_id}"
        self.add_summary(
            summary=summary,
            level=0,
            source_paths=[source_path],
            metadata={"history_id": history_id, "concepts": concept_keywords, "reasoning": reasoning}
        )
        
        print(f"Added summary to level 0")
        
        # Process all levels
        await self.process_all_levels()
    
    def get_highest_level_summary(self) -> Optional[str]:
        """Get the latest summary from the highest level available."""
        highest_level = max(self.summaries.keys()) if self.summaries else None
        if highest_level is None:
            return None
        
        entries = self._get_summaries_by_level(highest_level)
        if not entries:
            return None
        
        # Get the latest entry
        latest = max(entries, key=lambda x: x.timestamp)
        return latest.get_formatted_content()
    
    def get_summary_by_concept(self, concept: str, level: Optional[int] = None, exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Get all summaries tagged with a specific concept.
        
        Args:
            concept: The concept to search for
            level: Optional level to filter by
            exact_match: If True, only return summaries explicitly tagged with this concept
                        If False, also look for the concept in content
            
        Returns:
            List of summary dictionaries with content, level, relevance, etc.
        """
        results = []
        
        # Manual search through summaries, prioritizing explicit tags
        for summary_level, entries in self.summaries.items():
            # Skip if level filter doesn't match
            if level is not None and summary_level != level:
                continue
                
            for entry in entries:
                relevance_score = 0
                concept_matches = []
                
                # Check if explicitly tagged with this concept in metadata
                if hasattr(entry, 'metadata') and entry.metadata:
                    if 'concepts' in entry.metadata and isinstance(entry.metadata['concepts'], list):
                        if concept.lower() in [c.lower() for c in entry.metadata['concepts']]:
                            # Directly tagged is highest relevance
                            relevance_score = 10
                            concept_matches.append(f"Tagged with '{concept}'")
                
                # If not exact match or no explicit tag found, check content
                if not exact_match or relevance_score == 0:
                    # Check if concept is mentioned in content
                    if entry.content and concept.lower() in entry.content.lower():
                        # Content match has medium relevance
                        content_relevance = 5
                        # If there's a direct match (including word boundaries), add more relevance
                        import re
                        if re.search(r'\b' + re.escape(concept.lower()) + r'\b', entry.content.lower()):
                            content_relevance += 2
                        
                        # Count occurrences of the concept
                        occurrences = entry.content.lower().count(concept.lower())
                        if occurrences > 1:
                            content_relevance += min(occurrences, 3)  # Cap at +3 for multiple occurrences
                            
                        if relevance_score < content_relevance:
                            relevance_score = content_relevance
                            concept_matches.append(f"Content mentions '{concept}' {occurrences} times")
                
                # Skip if no relevance and exact matching
                if relevance_score == 0 and exact_match:
                    continue
                
                # Skip if no relevance at all
                if relevance_score == 0:
                    continue
                    
                # Prepare the result
                entry_id = f"summary_{summary_level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}"
                result = {
                    "level": summary_level,
                    "id": entry_id,
                    "content": entry.content,
                    "source_paths": entry.source_paths,
                    "timestamp": entry.timestamp.isoformat(),
                    "relevance_score": relevance_score,
                    "match_details": ", ".join(concept_matches)
                }
                
                # Add metadata
                if hasattr(entry, 'metadata') and entry.metadata:
                    result["metadata"] = entry.metadata
                
                results.append(result)
        
        # Try registry method if available
        if (hasattr(self, 'registry_builder') and 
            self.registry_builder and 
            hasattr(self.registry_builder, 'get_summaries_by_concept')):
            try:
                registry_summaries = self.registry_builder.get_summaries_by_concept(concept)
                
                # Filter by level if specified
                if level is not None:
                    registry_summaries = [s for s in registry_summaries if s.get('level') == level]
                    
                # Format and add the summaries
                for summary in registry_summaries:
                    # Registry summaries are considered highly relevant
                    relevance_score = 9
                    result = {
                        "level": summary.get('level'),
                        "id": summary.get('id'),
                        "content": summary.get('content', ''),
                        "source_paths": summary.get('source_paths', []),
                        "relevance_score": relevance_score,
                        "match_details": "Registry match",
                        "from_registry": True
                    }
                    results.append(result)
            except (AttributeError, TypeError):
                pass
        
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: (-x["relevance_score"], x["level"]))
        
        # Format the results for return
        formatted_results = []
        for result in results:
            # Create a header with level, id, and relevance
            header = f"## Level {result['level']} Summary | {result['id']} | Relevance: {result['relevance_score']}/10\n"
            header += f"Sources: {', '.join(result['source_paths'])}\n"
            if "match_details" in result:
                header += f"Match: {result['match_details']}\n"
            
            # Format all the result info into a dictionary
            formatted_results.append({
                "formatted_text": f"{header}\n{result['content']}",
                "raw": result
            })
            
        return formatted_results
        
    def get_related_summaries(self, summary_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get summaries related to a specific summary using concept matching and content similarity.
        
        Args:
            summary_id: ID of the summary to find relations for
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries with formatted text and raw data
        """
        results = []
        
        # Find the target summary first
        target_level = None
        target_entry = None
        target_concepts = []
        
        for level, entries in self.summaries.items():
            for entry in entries:
                entry_id = f"summary_{level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}"
                if entry_id == summary_id:
                    target_level = level
                    target_entry = entry
                    # Get concepts from the target entry
                    if hasattr(entry, 'metadata') and entry.metadata:
                        if 'concepts' in entry.metadata and isinstance(entry.metadata['concepts'], list):
                            target_concepts = entry.metadata['concepts']
                    break
            if target_entry:
                break
                
        # If we found the target, use its concepts to find related summaries
        if target_entry:
            # If we have concepts from the target, use them for relevance scoring
            if target_concepts:
                print(f"Finding summaries related to concepts: {target_concepts}")
                
                # Find summaries with similar concepts
                concept_related_entries = []
                
                for level, entries in self.summaries.items():
                    for entry in entries:
                        if entry == target_entry:  # Skip self
                            continue
                            
                        entry_concepts = []
                        if hasattr(entry, 'metadata') and entry.metadata:
                            if 'concepts' in entry.metadata and isinstance(entry.metadata['concepts'], list):
                                entry_concepts = entry.metadata['concepts']
                        
                        # If entry has concepts, check for overlap
                        if entry_concepts:
                            # Count concept overlap
                            overlap = set(target_concepts) & set(entry_concepts)
                            if overlap:
                                concept_score = len(overlap) * 5  # Each shared concept is worth 5 points
                                concept_related_entries.append((
                                    entry, 
                                    level, 
                                    concept_score, 
                                    f"Shares concepts: {', '.join(overlap)}"
                                ))
                
                # Sort concept-based matches by score
                concept_related_entries.sort(key=lambda x: x[2], reverse=True)
                
                # Take top concept-based matches up to 3/4 of max_results
                top_concept_matches = concept_related_entries[:int(max_results * 0.75)]
                
                # Format and add concept-based matches
                for entry, level, score, match_reason in top_concept_matches:
                    entry_id = f"summary_{level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}"
                    result = {
                        "level": level,
                        "id": entry_id,
                        "content": entry.content,
                        "source_paths": entry.source_paths,
                        "timestamp": entry.timestamp.isoformat() if hasattr(entry, 'timestamp') else None,
                        "relevance_score": score,
                        "match_details": match_reason,
                        "match_type": "concept"
                    }
                    
                    # Add metadata
                    if hasattr(entry, 'metadata') and entry.metadata:
                        result["metadata"] = entry.metadata
                    
                    results.append(result)
            
            # Additionally, always check for content similarity
            # This handles cases where concepts weren't tagged but content is similar
            if target_entry.content:
                # Find summaries with similar content
                similar_entries = []
                
                for level, entries in self.summaries.items():
                    for entry in entries:
                        # Skip entries we already found via concept matching
                        already_found = False
                        for result in results:
                            if result.get("id", "").endswith(entry.timestamp.strftime('%Y%m%d_%H%M%S')):
                                already_found = True
                                break
                                
                        if already_found or entry == target_entry or not entry.content:
                            continue
                            
                        # Simple similarity: count word overlap
                        target_words = set(target_entry.content.lower().split())
                        entry_words = set(entry.content.lower().split())
                        intersection = len(target_words & entry_words)
                        
                        # Only consider meaningful similarity
                        if intersection > 5:  # At least 5 words in common
                            # Calculate relevance - normalize by length
                            total_unique_words = len(target_words | entry_words)
                            if total_unique_words > 0:
                                jaccard_similarity = intersection / total_unique_words
                                # Scale to 0-10 range
                                content_score = min(int(jaccard_similarity * 100), 10)
                                similar_entries.append((
                                    entry, 
                                    level, 
                                    content_score, 
                                    f"Content similarity: {intersection} common words"
                                ))
                
                # Sort by similarity score
                similar_entries.sort(key=lambda x: x[2], reverse=True)
                
                # Add remaining slots with content-based matches
                remaining_slots = max_results - len(results)
                for entry, level, score, match_reason in similar_entries[:remaining_slots]:
                    entry_id = f"summary_{level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}"
                    result = {
                        "level": level,
                        "id": entry_id,
                        "content": entry.content,
                        "source_paths": entry.source_paths,
                        "timestamp": entry.timestamp.isoformat() if hasattr(entry, 'timestamp') else None,
                        "relevance_score": score,
                        "match_details": match_reason,
                        "match_type": "content"
                    }
                    
                    # Add metadata
                    if hasattr(entry, 'metadata') and entry.metadata:
                        result["metadata"] = entry.metadata
                    
                    results.append(result)
        
        # As a fallback or additional source, try the registry method if available
        if (hasattr(self, 'registry_builder') and 
            self.registry_builder and 
            hasattr(self.registry_builder, 'find_related_summaries')):
            try:
                registry_related = self.registry_builder.find_related_summaries(summary_id, max_results)
                
                # Add registry results if we have space
                remaining_slots = max_results - len(results)
                if remaining_slots > 0 and registry_related:
                    for summary in registry_related[:remaining_slots]:
                        # Registry summaries use their own relevance scoring
                        relevance_score = summary.get('relevance_score', 5)
                        result = {
                            "level": summary.get('level'),
                            "id": summary.get('id'),
                            "content": summary.get('content', ''),
                            "source_paths": summary.get('source_paths', []),
                            "relevance_score": relevance_score,
                            "match_details": "Registry match",
                            "match_type": "registry",
                            "from_registry": True
                        }
                        results.append(result)
            except (AttributeError, TypeError):
                pass
        
        # Sort all results by relevance score
        results.sort(key=lambda x: (-x["relevance_score"], x.get("level", 0)))
        
        # Limit to max_results
        results = results[:max_results]
        
        # Format the results for return
        formatted_results = []
        for result in results:
            # Create a header with level, id, and relevance
            header = f"## Level {result['level']} Summary | {result['id']} | Relevance: {result['relevance_score']}/10\n"
            if "source_paths" in result:
                header += f"Sources: {', '.join(result['source_paths'])}\n"
            if "match_details" in result:
                header += f"Match: {result['match_details']}\n"
            if "match_type" in result:
                header += f"Match Type: {result['match_type']}\n"
            
            # Format all the result info into a dictionary
            formatted_results.append({
                "formatted_text": f"{header}\n{result['content']}",
                "raw": result
            })
        
        return formatted_results
    
    def save_state(self, file_path: str) -> None:
        """Save the current state to a file."""
        state = {
            "project_name": self.project_name,
            "project_description": self.project_description,
            "summaries": {
                str(level): [entry.to_dict() for entry in entries] 
                for level, entries in self.summaries.items()
            },
            "registry_keys": self._registry_keys
        }
        
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
    
    @classmethod
    def load_state(cls, file_path: str) -> "RecursiveSummarizer":
        """Load state from a file."""
        with open(file_path, "r") as f:
            state = json.load(f)
        
        summarizer = cls(
            project_name=state["project_name"],
            project_description=state["project_description"]
        )
        
        # Restore summaries
        for level_str, entries_data in state["summaries"].items():
            level = int(level_str)
            summarizer.summaries[level] = [
                RecursiveSummaryEntry.from_dict(entry_data)
                for entry_data in entries_data
            ]
        
        # Restore registry keys
        summarizer._registry_keys = state["registry_keys"]
        
        return summarizer
        
    def get_all_concepts(self) -> List[str]:
        """Get all concepts used in summaries."""
        concepts = []
        concept_counts = {}
        
        # Extract concepts directly from summaries
        for level, entries in self.summaries.items():
            for entry in entries:
                # Get concepts from metadata
                if hasattr(entry, 'metadata') and entry.metadata:
                    if 'concepts' in entry.metadata and isinstance(entry.metadata['concepts'], list):
                        # Track which concepts appear at which levels
                        for concept in entry.metadata['concepts']:
                            concepts.append(concept)
                            if concept in concept_counts:
                                concept_counts[concept] += 1
                            else:
                                concept_counts[concept] = 1
                
                # Also look for concept keywords in content
                if entry.content:
                    for keyword in ["architecture", "design", "testing", "implementation", 
                                   "documentation", "patterns", "components", "microservices",
                                   "event-driven", "mvc", "mvvm", "api", "database", "frontend",
                                   "backend", "security", "deployment", "scaling"]:
                        if keyword in entry.content.lower() and keyword not in concepts:
                            concepts.append(keyword)
                            if keyword in concept_counts:
                                concept_counts[keyword] += 1
                            else:
                                concept_counts[keyword] = 1
        
        # If we have registry integration
        if hasattr(self, 'registry_builder') and self.registry_builder:
            # Try to use registry's method if available
            if hasattr(self.registry_builder, 'get_all_concepts'):
                try:
                    registry_concepts = self.registry_builder.get_all_concepts()
                    # Add these to our results too
                    for concept in registry_concepts:
                        if concept not in concepts:
                            concepts.append(concept)
                            # Concepts from registry are considered as count=1
                            concept_counts[concept] = 1
                except (AttributeError, TypeError):
                    pass
        
        # Sort concepts by frequency (most common first)
        sorted_concepts = sorted(
            concepts, 
            key=lambda c: (-(concept_counts.get(c, 0)), c)  # Sort by -count, then alphabetically
        )
        
        # Return sorted concepts (no duplicates due to how we built the list)
        return sorted_concepts


class MetaSummarizerAgent(BaseHeavenAgentReplicant):
    """Agent specialized for meta-summarization of injected summaries."""
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="MetaSummarizerAgent",
            system_prompt="""You are an expert meta-summarizer.
Your task is to create higher-level summaries from existing summaries.
Focus on:
1. Identifying key themes and patterns across all summaries
2. Extracting the most important information 
3. Creating a cohesive, well-structured overview
4. Maintaining essential details while reducing redundancy

Your summary should be comprehensive yet concise, capturing the core insights
from all the input summaries.""",
            tools=[],  # No special tools needed
            provider=ProviderEnum.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            temperature=0.3
        )
    
    async def create_meta_summary(self, summaries_text: str) -> str:
        """
        Create a meta-summary from a collection of summaries.
        
        Args:
            summaries_text: Text containing all the summaries to summarize
            
        Returns:
            A consolidated meta-summary
        """
        prompt = f"""Please create a comprehensive higher-level summary 
that captures the key points across the following summaries:

{summaries_text}

Focus on:
1. Identifying key themes and patterns across all summaries
2. Extracting the most important information 
3. Creating a cohesive, well-structured overview
4. Maintaining essential details while reducing redundancy
"""
        
        result = await self.run(prompt=prompt)
        
        # Handle different result formats
        if hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict) and 'content' in result:
            return result['content']
        else:
            return str(result)


# Utility functions
async def create_recursive_summarizer(project_name: str, project_description: str) -> RecursiveSummarizer:
    """Create a new recursive summarizer for a project."""
    return RecursiveSummarizer(project_name, project_description)


async def add_history_to_summarizer(
    summarizer: RecursiveSummarizer,
    history_id_or_obj: Union[str, History]
) -> None:
    """Add a history to the summarizer and process all levels."""
    await summarizer.add_history_summary(history_id_or_obj)


async def get_highest_summary(summarizer: RecursiveSummarizer) -> Optional[str]:
    """Get the highest level summary available."""
    return summarizer.get_highest_level_summary()


async def save_summarizer(summarizer: RecursiveSummarizer, file_path: str) -> None:
    """Save the summarizer state to a file."""
    summarizer.save_state(file_path)


async def load_summarizer(file_path: str) -> RecursiveSummarizer:
    """Load a summarizer from a file."""
    return RecursiveSummarizer.load_state(file_path)


# Registry-enhanced utility functions
async def get_summaries_by_concept(
    summarizer: RecursiveSummarizer, 
    concept: str, 
    level: Optional[int] = None
) -> List[str]:
    """Get summaries by concept tag."""
    return summarizer.get_summary_by_concept(concept, level)


async def get_all_concepts(summarizer: RecursiveSummarizer) -> List[str]:
    """Get all concepts used in summaries."""
    return summarizer.get_all_concepts()


async def get_related_summaries(
    summarizer: RecursiveSummarizer,
    summary_id: str,
    max_results: int = 5
) -> List[str]:
    """Get summaries related to a specific summary."""
    return summarizer.get_related_summaries(summary_id, max_results)


async def find_by_source_path(
    summarizer: RecursiveSummarizer,
    source_path: str
) -> List[Dict[str, Any]]:
    """Find summaries that include a specific source path."""
    results = []
    
    # Manual search through summaries if registry doesn't have the method
    for level, entries in summarizer.summaries.items():
        for entry in entries:
            if source_path in entry.source_paths:
                # Create a dict representation
                results.append({
                    "level": level,
                    "id": f"summary_L{level}_{entry.timestamp.strftime('%Y%m%d_%H%M%S')}",
                    "content": entry.content,
                    "concepts": [],  # No concepts available in this fallback
                    "source_paths": entry.source_paths
                })
    
    # Try registry method if available
    if (hasattr(summarizer, 'registry_builder') and 
        summarizer.registry_builder and 
        hasattr(summarizer.registry_builder, 'get_summaries_by_source_path')):
        try:
            registry_results = summarizer.registry_builder.get_summaries_by_source_path(source_path)
            if registry_results:
                results.extend(registry_results)
        except (AttributeError, TypeError):
            pass
    
    return results