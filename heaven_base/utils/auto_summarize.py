"""
Auto-summarize functionality for handling large history contexts.
Provides automatic summarization, reasoning, and concept extraction.
"""

from typing import Dict, Any, Optional, List, Type, Union
import json
import os
from datetime import datetime

from ..baseheavenagent import HeavenAgentConfig, BaseHeavenAgentReplicant
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tools.think_tool import ThinkTool
from ..tools.registry_tool import RegistryTool
from ..tools.view_history_tool import ViewHistoryTool
from ..unified_chat import ProviderEnum
from ..memory.history import History
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
#### 
# A major problem with this file is that it uses agent.run for real tasks that can get blocked. It should use hermes instead... Technically, auto_summarize should go through the langgraph system... that was already supposed to be done...
#
####

# ITERATION SUMMARIZER TOOL
class IterationSummarizerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'iteration_number': {
            'name': 'iteration_number',
            'type': 'int',
            'description': 'The iteration number being summarized',
            'required': True
        },
        'actions_taken': {
            'name': 'actions_taken',
            'type': 'str',
            'description': 'Summary of actions taken in this iteration',
            'required': True
        },
        'outcomes': {
            'name': 'outcomes',
            'type': 'str',
            'description': 'Summary of outcomes from this iteration',
            'required': True
        },
        'challenges': {
            'name': 'challenges',
            'type': 'str',
            'description': 'Summary of challenges encountered in this iteration',
            'required': True
        },
        'tools_used': {
            'name': 'tools_used',
            'type': 'str',
            'description': 'Summary of tools used in this iteration',
            'required': True
        }
    }

def iteration_summarizer_func(iteration_number: int, actions_taken: str, outcomes: str, challenges: str, tools_used: str) -> str:
    """Summarize a single iteration."""
    summary = f"""## Iteration {iteration_number} Summary

**Actions Taken:**
{actions_taken}

**Outcomes:**
{outcomes}

**Challenges:**
{challenges}

**Tools Used:**
{tools_used}
"""
    return summary

class IterationSummarizerTool(BaseHeavenTool):
    name = "IterationSummarizerTool"
    description = "Summarizes a single iteration of conversation with actions, outcomes, challenges, and tools used"
    func = iteration_summarizer_func
    args_schema = IterationSummarizerToolArgsSchema
    is_async = False


# AGGREGATION SUMMARIZER TOOL
class AggregationSummarizerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'total_iterations': {
            'name': 'total_iterations',
            'type': 'int',
            'description': 'Total number of iterations summarized',
            'required': True
        },
        'overall_progress': {
            'name': 'overall_progress',
            'type': 'str',
            'description': 'Summary of overall progress across all iterations',
            'required': True
        },
        'key_actions': {
            'name': 'key_actions',
            'type': 'str',
            'description': 'Key actions taken across all iterations',
            'required': True
        },
        'major_outcomes': {
            'name': 'major_outcomes',
            'type': 'str',
            'description': 'Major outcomes achieved across all iterations',
            'required': True
        },
        'recurring_challenges': {
            'name': 'recurring_challenges',
            'type': 'str',
            'description': 'Recurring challenges encountered across iterations',
            'required': True
        },
        'tools_summary': {
            'name': 'tools_summary',
            'type': 'str',
            'description': 'Summary of tools used across iterations',
            'required': True
        },
        'final_state': {
            'name': 'final_state',
            'type': 'str',
            'description': 'Description of the final state after all iterations',
            'required': True
        }
    }

def aggregation_summarizer_func(total_iterations: int, overall_progress: str, key_actions: str, 
                               major_outcomes: str, recurring_challenges: str, tools_summary: str, 
                               final_state: str) -> str:
    """Aggregate multiple iteration summaries into one."""
    aggregated = f"""# Aggregated Summary

**Total Iterations:** {total_iterations}

## Overall Progress
{overall_progress}

### Key Actions Across Iterations
{key_actions}

### Major Outcomes
{major_outcomes}

### Recurring Challenges
{recurring_challenges}

### Tools and Methods Used
{tools_summary}

## Final State
{final_state}
"""
    return aggregated

class AggregationSummarizerTool(BaseHeavenTool):
    name = "AggregationSummarizerTool"
    description = "Aggregates multiple iteration summaries into one cohesive overview"
    func = aggregation_summarizer_func
    args_schema = AggregationSummarizerToolArgsSchema
    is_async = False


# CONCEPT SUMMARY TOOL
class ConceptSummaryToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'summary_essence': {
            'name': 'summary_essence',
            'type': 'str',
            'description': 'The essence of what was summarized',
            'required': True
        },
        'next_steps': {
            'name': 'next_steps',
            'type': 'str',
            'description': 'Recommended next steps from reasoning',
            'required': True
        },
        'strategic_direction': {
            'name': 'strategic_direction',
            'type': 'str',
            'description': 'Strategic direction from reasoning',
            'required': True
        },
        'unified_concept': {
            'name': 'unified_concept',
            'type': 'str',
            'description': 'The unified concept combining summary and reasoning',
            'required': True
        }
    }

def concept_summary_func(summary_essence: str, next_steps: str, strategic_direction: str, unified_concept: str) -> str:
    """Create a unified concept from summary and reasoning."""
    concept = f"""## Unified Concept

**Core Summary:** {summary_essence}

**Strategic Direction:** {strategic_direction}

**Immediate Actions:** {next_steps}

**Unified Concept:**
{unified_concept}
"""
    return concept

class ConceptSummaryTool(BaseHeavenTool):
    name = "ConceptSummaryTool"
    description = "Creates a unified concept from summary and reasoning"
    func = concept_summary_func
    args_schema = ConceptSummaryToolArgsSchema
    is_async = False


# SUMMARIZER AGENTS
class IterationSummarizerAgent(BaseHeavenAgentReplicant):
    """Agent that summarizes individual iterations."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_summary = None
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="IterationSummarizerAgent",
            system_prompt="""You are a specialized agent that systematically summarizes ALL iterations in conversation histories.

Your systematic process:
1. Use ViewHistoryTool to read the first iteration of the history id provided.
    - Check the result for the total number of iterations.
2. View the iterations one at a time.
3. For each iteration you see, create a summary using IterationSummarizerTool
4. Continue until you have processed all of them
5. You are done only when you've summarized every single iteration in the history

CRITICAL: You must process ALL iterations, not just the first batch. If a history has 51 iterations, you must summarize all 51 individually.

IMPORTANT: Only call ONE tool at a time. Never make multiple tool calls in a single response.

Process: ViewHistoryTool → IterationSummarizerTool → ViewHistoryTool → IterationSummarizerTool → ... (continue until complete).""",
            tools=[IterationSummarizerTool, ViewHistoryTool],
            provider=ProviderEnum.OPENAI,
            model="gpt-4.1",
            temperature=0.2
        )
    
    def look_for_particular_tool_calls(self) -> None:
        """Look for IterationSummarizerTool calls and save the result."""
        for i, msg in enumerate(self.history.messages):
            if isinstance(msg, AIMessage):
                # Check Anthropic format (list content)
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            if item.get('name') == "IterationSummarizerTool":
                                # Get the next message which should be the ToolMessage with the result
                                if i + 1 < len(self.history.messages):
                                    tool_result = self.history.messages[i + 1]
                                    if isinstance(tool_result, ToolMessage):
                                        self.last_summary = tool_result.content
                # Check OpenAI format (tool_calls attribute)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] == "IterationSummarizerTool":
                            # Get the next message which should be the ToolMessage with the result
                            if i + 1 < len(self.history.messages):
                                tool_result = self.history.messages[i + 1]
                                if isinstance(tool_result, ToolMessage):
                                    self.last_summary = tool_result.content
    
    def save_summary(self, summary_content: str) -> None:
        """Save the summary for later retrieval."""
        self.last_summary = summary_content


class AggregationSummarizerAgent(BaseHeavenAgentReplicant):
    """Agent that aggregates multiple summaries."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_summary = None
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="AggregationSummarizerAgent",
            system_prompt="""You are a summary aggregation specialist.
Your role is to:
1. Take multiple iteration summaries
2. Find patterns and common themes
3. Use the AggregationSummarizerTool to create a cohesive overview
4. Provide a comprehensive view of the entire conversation

IMPORTANT: Only call ONE tool at a time. Never make multiple tool calls in a single response.

You will be given iteration summaries to aggregate. Use the AggregationSummarizerTool to create your aggregated summary.""",
            tools=[AggregationSummarizerTool],
            provider=ProviderEnum.OPENAI,
            model="gpt-4.1",
            temperature=0.3
        )
    
    def look_for_particular_tool_calls(self) -> None:
        """Look for AggregationSummarizerTool calls and save the result."""
        for i, msg in enumerate(self.history.messages):
            if isinstance(msg, AIMessage):
                # Check Anthropic format (list content)
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            if item.get('name') == "AggregationSummarizerTool":
                                # Get the next message which should be the ToolMessage with the result
                                if i + 1 < len(self.history.messages):
                                    tool_result = self.history.messages[i + 1]
                                    if isinstance(tool_result, ToolMessage):
                                        self.last_summary = tool_result.content
                # Check OpenAI format (tool_calls attribute)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] == "AggregationSummarizerTool":
                            # Get the next message which should be the ToolMessage with the result
                            if i + 1 < len(self.history.messages):
                                tool_result = self.history.messages[i + 1]
                                if isinstance(tool_result, ToolMessage):
                                    self.last_summary = tool_result.content
    
    def save_summary(self, summary_content: str) -> None:
        """Save the aggregated summary."""
        self.last_summary = summary_content


class ReasoningAgent(BaseHeavenAgentReplicant):
    """Reasons about what should happen next based on a summary."""
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="ReasoningAgent",
            system_prompt="""You analyze summaries and reason about next steps.

IMPORTANT: Only call ONE tool at a time. Never make multiple tool calls in a single response.

Use ThinkTool to reason through:
1. What has been accomplished
2. What remains to be done
3. Strategic implications
4. Recommended actions""",
            tools=[ThinkTool],
            provider=ProviderEnum.OPENAI,
            model="o4-mini",
            temperature=0.5,
            additional_kws=["NextSteps", "StrategicAnalysis"],
            additional_kw_instructions="""
Extract content using this exact format:
```NextSteps
Recommended next actions go here
```

```StrategicAnalysis
Strategic considerations and analysis go here
```
"""
        )


class ConceptExtractorAgent(BaseHeavenAgentReplicant):
    """Agent that extracts unified concepts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_concept = None
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="ConceptExtractorAgent",
            system_prompt="""You extract unified concepts from summaries and reasoning.
Your role is to:
1. Analyze the provided summary and reasoning
2. Use ConceptSummaryTool to create a unified concept
3. Capture the essence of what was done and what should happen next
4. Create a concise, actionable concept

IMPORTANT: Only call ONE tool at a time. Never make multiple tool calls in a single response.

Use the ConceptSummaryTool to create your unified concept.""",
            tools=[ConceptSummaryTool, RegistryTool],
            provider=ProviderEnum.OPENAI,
            model="o4-mini",
            temperature=0.4
        )
    
    def look_for_particular_tool_calls(self) -> None:
        """Look for ConceptSummaryTool calls and save the result."""
        for i, msg in enumerate(self.history.messages):
            if isinstance(msg, AIMessage):
                # Check Anthropic format (list content)
                if isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            if item.get('name') == "ConceptSummaryTool":
                                # Get the next message which should be the ToolMessage with the result
                                if i + 1 < len(self.history.messages):
                                    tool_result = self.history.messages[i + 1]
                                    if isinstance(tool_result, ToolMessage):
                                        self.last_concept = tool_result.content
                # Check OpenAI format (tool_calls attribute)
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call['name'] == "ConceptSummaryTool":
                            # Get the next message which should be the ToolMessage with the result
                            if i + 1 < len(self.history.messages):
                                tool_result = self.history.messages[i + 1]
                                if isinstance(tool_result, ToolMessage):
                                    self.last_concept = tool_result.content
    
    def save_concept(self, concept: str) -> None:
        """Save the extracted concept."""
        self.last_concept = concept


# UTILITY FUNCTIONS
def auto_summarize_flag(history_id_or_obj: Union[str, History], summarize_at: int = 450000) -> bool:
    """
    Check if history content exceeds the token threshold for summarization.
    
    Args:
        history_id_or_obj: Either a history_id string or History object
        summarize_at: Character threshold (roughly 150k tokens at default of 450000 chars)
                      Approximating 3 characters per token
    
    Returns:
        bool: True if summarization is needed, False otherwise
    """
    # Handle both history_id and History object
    if isinstance(history_id_or_obj, str):
        history = History.load_from_id(history_id_or_obj)
    else:
        history = history_id_or_obj
    
    # Calculate total character count (rough approximation of tokens)
    total_chars = 0
    for msg in history.messages:
        if isinstance(msg, BaseMessage):
            total_chars += len(str(msg.content))
    
    return total_chars > summarize_at


async def auto_summarize(history_id_or_obj: Union[str, History]) -> str:
    """Summarize history by reading the actual conversation and creating summaries."""
    # Handle both history_id and History object
    if isinstance(history_id_or_obj, str):
        history = History.load_from_id(history_id_or_obj)
        history_id = history_id_or_obj
    else:
        history = history_id_or_obj
        history_id = history.history_id or "temp_history"
        # Save the history if it doesn't have an ID yet
        if not history.history_id:
            history.save("auto_summarize_temp")
            history_id = history.history_id
    
    # Create ONE summarizer agent that will read the actual history
    summarizer = IterationSummarizerAgent()
    
    # Create the goal - pass the actual history_id so the agent can read it
    goal = f"""Use ViewHistoryTool to get the total iterations of the history_id: {history_id}. Then, for each iteration, read it individually and then summarize it with the IterationSummarizerTool. When you finish every iteration in the history, you are done. You can use 10 tool calls per agent mode iteration. Make a task list like: 1) Use ViewHistoryTool to discover total iterations, 2) Process all iterations systematically in batches, 3) Summarize each iteration found, 4) Continue until ALL iterations are processed."""
    
    # Use agent mode with more iterations to allow reading + multiple summaries
    # Need enough iterations: ~6 for viewing batches + ~51 for summarizing = ~60 iterations
    prompt = f'agent goal="{goal}", iterations=60'
    
    # Run the agent - this creates and saves a new history automatically
    result = await summarizer.run(prompt)
    
    print(f"Iteration summarizer run complete. History ID: {result['history_id']}")
    
    # Look for tool calls to get all summaries created
    summarizer.look_for_particular_tool_calls()
    
    # Get all the summaries from the agent's history
    iteration_summaries = []
    if summarizer.history and summarizer.history.messages:
        for msg in summarizer.history.messages:
            if isinstance(msg, ToolMessage) and "## Iteration" in msg.content:
                iteration_summaries.append(msg.content)
    
    # Return ALL components for complete compaction
    if len(iteration_summaries) > 1:
        # Create aggregator to combine multiple iteration summaries
        aggregator = AggregationSummarizerAgent()
        
        combined_summaries = "\n\n".join(iteration_summaries)
        
        # Create the aggregation goal
        goal = f"Create an aggregated summary using the AggregationSummaryTool from these iteration summaries: {combined_summaries}"
        
        # Use agent mode with 5 iterations
        prompt = f'agent goal="{goal}", iterations=5'
        
        # Run the aggregator
        result = await aggregator.run(prompt)
        print(f"Aggregation summarizer run complete. History ID: {result['history_id']}")
        
        # Look for tool calls
        aggregator.look_for_particular_tool_calls()
        
        # Return ALL components: iteration summaries + aggregated summary
        return {
            "iteration_summaries": iteration_summaries,
            "aggregated_summary": aggregator.last_summary or "Failed to create aggregated summary",
            "total_summary": combined_summaries + "\n\n" + (aggregator.last_summary or "Failed to create aggregated summary")
        }
    
    elif len(iteration_summaries) == 1:
        # Single iteration case
        return {
            "iteration_summaries": iteration_summaries,
            "aggregated_summary": iteration_summaries[0],
            "total_summary": iteration_summaries[0]
        }
    
    else:
        # Fallback case
        fallback = summarizer.last_summary or "Failed to create summary"
        return {
            "iteration_summaries": [fallback],
            "aggregated_summary": fallback,
            "total_summary": fallback
        }


async def reason_about_summary(summary: str, output_dir: str = "/home/GOD/reasoning_outputs") -> Dict[str, Any]:
    """Use reasoning agent to analyze summary and determine next steps."""
    reasoner = ReasoningAgent()
    
    # Create goal for reasoning
    goal = f"Analyze this summary and determine what should happen next using ThinkTool. Extract NextSteps and StrategicAnalysis. <summary>{summary}</summary>"
    
    # Use agent mode format
    prompt = f'agent goal="{goal}", iterations=5'
    
    # Run the reasoner - this creates and saves a new history automatically
    result = await reasoner.run(prompt)
    print(f"Reasoning agent run complete. History ID: {result['history_id']}")
    
    # Extract reasoning from the agent's extractions
    reasoning_data = {
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "next_steps": "",
        "strategic_analysis": ""
    }
    
    # Get agent_status from the result
    if reasoner.history and reasoner.history.agent_status:
        extracts = reasoner.history.agent_status.extracted_content or {}
        reasoning_data["next_steps"] = extracts.get("NextSteps", "No next steps identified")
        reasoning_data["strategic_analysis"] = extracts.get("StrategicAnalysis", "No strategic analysis")
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reasoning_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(reasoning_data, f, indent=2)
    
    return reasoning_data


async def summarize_and_reason(history_id_or_obj: Union[str, History]) -> Dict[str, Any]:
    """Summarize history and reason about next steps."""
    # Generate summary
    summary = await auto_summarize(history_id_or_obj)
    
    # Reason about summary
    reasoning = await reason_about_summary(summary)
    
    return {
        "summary": summary,
        "reasoning": reasoning
    }


async def get_summary_concept(summary: str, reasoning: Dict[str, Any]) -> str:
    """Extract a single concept from summary and reasoning."""
    extractor = ConceptExtractorAgent()
    
    next_steps = reasoning.get('next_steps', 'No next steps identified')
    strategic = reasoning.get('strategic_analysis', 'No strategic analysis')
    
    # Create goal for concept extraction
    goal = f"""Create a unified concept using the ConceptSummaryTool based on this summary: '{summary}' 
    and reasoning: Next Steps: '{next_steps}', Strategic Analysis: '{strategic}'"""
    
    # Use agent mode format
    prompt = f'agent goal="{goal}", iterations=3'
    
    # Run the extractor - this creates and saves a new history automatically
    result = await extractor.run(prompt)
    print(f"Concept extractor run complete. History ID: {result['history_id']}")
    
    # Look for tool calls
    extractor.look_for_particular_tool_calls()
    
    return extractor.last_concept or "Failed to extract concept"


async def get_summary_reasoning_and_concept(history_id_or_obj: Union[str, History]) -> Dict[str, Any]:
    """Get summary, reasoning, and unified concept from a history."""
    # Get summary and reasoning
    summary_and_reasoning = await summarize_and_reason(history_id_or_obj)
    
    # Extract concept
    concept = await get_summary_concept(
        summary_and_reasoning["summary"],
        summary_and_reasoning["reasoning"]
    )
    
    return {
        "summary_and_reasoning": summary_and_reasoning,
        "concept": concept
    }


# TEST FUNCTION
async def test_auto_summarize():
    """Test the auto-summarization functionality."""
    print("Testing auto-summarization system...")
    
    # Test with a mock history_id
    test_history_id = "test_history_001"
    
    # Test flag function
    print("\n1. Testing auto_summarize_flag...")
    needs_summary = auto_summarize_flag(test_history_id)
    print(f"   Needs summary (>150k chars): {needs_summary}")
    
    # Test iteration summarizer directly
    print("\n2. Testing iteration summarizer...")
    summarizer = IterationSummarizerAgent()
    goal = "Summarize iteration 1 using IterationSummaryTool with actions: code analyzed, outcomes: patterns found, challenges: none, tools: CodeLocalizerTool"
    prompt = f'agent goal="{goal}", iterations=3'
    await summarizer.run(prompt)
    summarizer.look_for_particular_tool_calls()
    print(f"   Iteration summary created: {summarizer.last_summary is not None}")
    
    # Test aggregation summarizer  
    print("\n3. Testing aggregation summarizer...")
    aggregator = AggregationSummarizerAgent()
    goal = "Aggregate these summaries using AggregationSummaryTool: 'Iteration 1: analyzed', 'Iteration 2: refactored'. Create overview with 3 total iterations."
    prompt = f'agent goal="{goal}", iterations=3'
    await aggregator.run(prompt)
    aggregator.look_for_particular_tool_calls()
    print(f"   Aggregation created: {aggregator.last_summary is not None}")
    
    # Test auto_summarize
    print("\n4. Testing auto_summarize...")
    summary = await auto_summarize(test_history_id)
    print(f"   Summary generated: {len(summary) > 0}")
    print(f"   Summary preview: {summary[:100]}...")
    
    # Test reasoning
    print("\n5. Testing reasoning...")
    reasoning = await reason_about_summary(summary)
    print(f"   Reasoning completed: {reasoning is not None}")
    print(f"   Next steps: {reasoning.get('next_steps', 'None')[:50]}...")
    
    # Test concept extraction
    print("\n6. Testing concept extraction...")
    concept = await get_summary_concept(summary, reasoning)
    print(f"   Concept extracted: {len(concept) > 0}")
    print(f"   Concept preview: {concept[:100]}...")
    
    # Test full pipeline
    print("\n7. Testing full pipeline...")
    result = await get_summary_reasoning_and_concept(test_history_id)
    print(f"   Pipeline complete: {result is not None}")
    print(f"   Has summary: {'summary' in result['summary_and_reasoning']}")
    print(f"   Has reasoning: {'reasoning' in result['summary_and_reasoning']}")
    print(f"   Has concept: {'concept' in result}")
    
    print("\n✅ All tests completed!")
    return result


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_auto_summarize())
    print("\nFinal result structure:")
    print(json.dumps({
        "summary_and_reasoning": {
            "summary": result["summary_and_reasoning"]["summary"][:100] + "...",
            "reasoning": {
                "next_steps": result["summary_and_reasoning"]["reasoning"]["next_steps"][:50] + "...",
                "strategic_analysis": result["summary_and_reasoning"]["reasoning"]["strategic_analysis"][:50] + "..."
            }
        },
        "concept": result["concept"][:100] + "..."
    }, indent=2))