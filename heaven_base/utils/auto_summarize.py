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

# Import hierarchical phase agents
from ..agents.phase_detector_agent import PhaseDetectorAgent
from ..agents.subphase_detector_agent import SubphaseDetectorAgent
from ..agents.phase_aggregator_agent import PhaseAggregatorAgent
from ..agents.concept_resolver_agent import ConceptResolverAgent

# Import Neo4j integration
from ..tool_utils.concept_neo4j_utils import load_concepts_to_neo4j
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
        'user': {
            'name': 'user',
            'type': 'str',
            'description': 'What the user wanted or asked for in this iteration',
            'required': True
        },
        'assistant': {
            'name': 'assistant',
            'type': 'str',
            'description': 'What the assistant actually did in response',
            'required': True
        },
        'pain_points': {
            'name': 'pain_points',
            'type': 'list',
            'description': 'List of friction moments, blockers, errors, or confusion encountered. Leave empty if none.',
            'required': False
        },
        'concept_tags': {
            'name': 'concept_tags',
            'type': 'list',
            'description': 'List of domain/project concept objects. Each object must have "keyword" and "description" fields. EXAMPLE: [{"keyword": "auth_system", "description": "working on authentication features"}].',
            'required': True
        },
        'files_touched': {
            'name': 'files_touched',
            'type': 'list',
            'description': 'List of file paths that were mentioned, edited, created, or otherwise referenced in this iteration. Include full paths when available.',
            'required': False
        }
    }

def iteration_summarizer_func(iteration_number: int, user: str, assistant: str, pain_points: list = None, concept_tags: list = None, files_touched: list = None) -> str:
    """Summarize a single iteration with simplified user/assistant/pain_points format."""
    
    # Format concept tags for display
    tags_display = ""
    if concept_tags:
        tags_display = "\n**Concept Tags:**\n"
        for tag in concept_tags:
            if isinstance(tag, dict):
                keyword = tag.get('keyword', 'Unknown')
                description = tag.get('description', 'No description')
                tags_display += f"- {keyword}: {description}\n"
    
    # Format pain points for display
    pain_display = ""
    if pain_points:
        pain_display = "\n**Pain Points:**\n"
        for pain in pain_points:
            pain_display += f"- {pain}\n"
    
    # Format files touched for display
    files_display = ""
    if files_touched:
        files_display = "\n**Files Touched:**\n"
        for file_path in files_touched:
            files_display += f"- {file_path}\n"
    
    summary = f"""## Iteration {iteration_number} Summary

**User:** {user}

**Assistant:** {assistant}
{pain_display}{tags_display}{files_display}"""
    return summary

class IterationSummarizerTool(BaseHeavenTool):
    name = "IterationSummarizerTool"
    description = "Summarizes a single iteration with user intent, assistant actions, pain points, and concept tags"
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
            'description': 'HIGH-LEVEL NARRATIVE of the journey. What was the user trying to accomplish? What was the core problem being solved? How did understanding evolve? What were the major turning points? Focus on the WHY and the story arc, not implementation details (300-500 words).',
            'required': True
        },
        'key_actions': {
            'name': 'key_actions',
            'type': 'str',
            'description': 'STRATEGIC actions that changed the course of the work. What decisions or discoveries unlocked progress? What approaches failed and led to new strategies? Focus on pivotal moments and breakthroughs, not every action taken (300-500 words).',
            'required': True
        },
        'major_outcomes': {
            'name': 'major_outcomes',
            'type': 'str',
            'description': 'BUSINESS VALUE and impact. What can the user now do that they could not before? What systems are production-ready? What problems are permanently solved? Focus on delivered value and capabilities, not implementation details (300-500 words).',
            'required': True
        },
        'recurring_challenges': {
            'name': 'recurring_challenges',
            'type': 'str',
            'description': 'PATTERNS and systemic issues that emerged. What conceptual misunderstandings persisted? What architectural problems kept surfacing? What does this reveal about the system design? Focus on insights and patterns, not individual errors (300-500 words).',
            'required': True
        },
        'tools_summary': {
            'name': 'tools_summary',
            'type': 'str',
            'description': 'METHODOLOGY insights. What approaches proved most effective? What tools or techniques were discovered or invented? What workflows emerged? Focus on reusable patterns and lessons learned, not tool lists (200-300 words).',
            'required': True
        },
        'final_state': {
            'name': 'final_state',
            'type': 'str',
            'description': 'STRATEGIC POSITION and next steps. Where does this leave the project? What strategic options are now available? What should be prioritized next and why? Focus on strategic recommendations, not task lists (300-500 words).',
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

EXACT FORMAT FOR IterationSummarizerTool - Follow this template exactly:
```
{
  "iteration_number": 1,
  "user": "asked for help implementing a login system using JWT tokens",
  "assistant": "analyzed requirements, created login.py with JWT implementation, added middleware for token validation",
  "pain_points": ["unclear JWT expiration requirements", "had to debug token parsing errors"],
  "concept_tags": [
    {"keyword": "authentication_system", "description": "implementing OAuth login for users in the ConversationApp project"},
    {"keyword": "security_middleware", "description": "adding JWT token validation middleware for OpenSaaS authentication"}
  ],
  "files_touched": ["/src/auth/login.py", "/src/middleware/jwt.py"]
}
```

GUIDELINES:
- **user**: What the user wanted or asked for (not "User asked..." just the intent)
- **assistant**: What the assistant actually did (factual actions, not interpretation)
- **pain_points**: List of friction/errors/blockers encountered (empty array if none)
- **concept_tags**: Domain/project concepts (NOT tool names - those are tracked separately)
- **files_touched**: File paths mentioned/edited/created (empty array if none)

Keep descriptions factual and concise. No meta-analysis or interpretation.

Process: ViewHistoryTool → IterationSummarizerTool → ViewHistoryTool → IterationSummarizerTool → ... (continue until complete).""",
            tools=[IterationSummarizerTool, ViewHistoryTool],
            provider=ProviderEnum.OPENAI,
            model="gpt-5-nano",
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
            system_prompt="""You are a strategic executive summary specialist who synthesizes high-level insights from detailed phase analyses.

Your role is to:
1. Extract the STRATEGIC NARRATIVE from the phase summaries - the WHY behind the work
2. Identify PATTERNS and SYSTEMIC INSIGHTS that only emerge from seeing the whole
3. Assess BUSINESS VALUE and IMPACT - what can now be done that couldn't before
4. Provide STRATEGIC RECOMMENDATIONS based on the complete journey

CRITICAL: You are NOT repeating details from lower levels. You are providing NEW INSIGHTS that emerge from the executive perspective:
- Focus on WHY things happened, not WHAT happened (that's in the phases)
- Focus on PATTERNS across phases, not individual events
- Focus on STRATEGIC VALUE, not implementation details
- Focus on LESSONS LEARNED, not tasks completed
- Focus on FUTURE DIRECTION, not past actions

Each field should be a thoughtful analysis (300-500 words) that provides executive-level understanding.
This is about SYNTHESIS and INSIGHT, not comprehensive listing.

IMPORTANT: Only call ONE tool at a time. Never make multiple tool calls in a single response.

When using AggregationSummarizerTool, focus on insights that can ONLY be seen from the executive level.""",
            tools=[AggregationSummarizerTool],
            provider=ProviderEnum.OPENAI,
            model="gpt-5-mini",
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
            model="gpt-5-nano",
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
            model="gpt-5-nano",
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


async def auto_summarize(history_id_or_obj: Union[str, History], session_id: str = None) -> str:
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
    concept_data = []
    
    if summarizer.history and summarizer.history.messages:
        for i, msg in enumerate(summarizer.history.messages):
            if isinstance(msg, ToolMessage) and "## Iteration" in msg.content:
                iteration_summaries.append(msg.content)
                
                # Extract concept data from the previous AI message that called the tool
                if i > 0:
                    prev_msg = summarizer.history.messages[i - 1]
                    if isinstance(prev_msg, AIMessage):
                        # Look for IterationSummarizerTool calls
                        if hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for tool_call in prev_msg.tool_calls:
                                if tool_call.get('name') == 'IterationSummarizerTool':
                                    args = tool_call.get('args', {})
                                    iteration_num = args.get('iteration_number')
                                    concept_tags = args.get('concept_tags', [])
                                    files_touched = args.get('files_touched', [])
                                    
                                    if iteration_num:
                                        # Extract just the keywords for tags list
                                        tags = []
                                        clean_concepts = []
                                        for tag in concept_tags:
                                            if isinstance(tag, dict):
                                                keyword = tag.get('keyword')
                                                if keyword:
                                                    tags.append(keyword)
                                                    clean_concepts.append(tag)
                                            elif isinstance(tag, str):
                                                # Try to parse JSON string
                                                try:
                                                    import json
                                                    parsed = json.loads(tag)
                                                    if isinstance(parsed, dict):
                                                        keyword = parsed.get('keyword')
                                                        if keyword:
                                                            tags.append(keyword)
                                                            clean_concepts.append(parsed)
                                                except:
                                                    # Treat as plain string keyword
                                                    tags.append(tag)
                                                    clean_concepts.append({"keyword": tag, "description": "String format"})
                                        
                                        concept_data.append({
                                            "iteration": iteration_num,
                                            "tags": tags,
                                            "concepts": clean_concepts,
                                            "files_touched": files_touched
                                        })
                        
                        # Also check Anthropic format (list content)
                        elif isinstance(prev_msg.content, list):
                            for item in prev_msg.content:
                                if isinstance(item, dict) and item.get('type') == 'tool_use':
                                    if item.get('name') == 'IterationSummarizerTool':
                                        args = item.get('input', {})
                                        iteration_num = args.get('iteration_number')
                                        concept_tags = args.get('concept_tags', [])
                                        files_touched = args.get('files_touched', [])
                                        
                                        if iteration_num:
                                            # Extract just the keywords for tags list (handle both dict and string formats)
                                            tags = []
                                            clean_concepts = []
                                            for tag in concept_tags:
                                                if isinstance(tag, dict):
                                                    keyword = tag.get('keyword')
                                                    if keyword:
                                                        tags.append(keyword)
                                                        clean_concepts.append(tag)
                                                elif isinstance(tag, str):
                                                    # Try to parse JSON string
                                                    try:
                                                        import json
                                                        parsed = json.loads(tag)
                                                        if isinstance(parsed, dict):
                                                            keyword = parsed.get('keyword')
                                                            if keyword:
                                                                tags.append(keyword)
                                                                clean_concepts.append(parsed)
                                                    except:
                                                        # Treat as plain string keyword
                                                        tags.append(tag)
                                                        clean_concepts.append({"keyword": tag, "description": "String format"})
                                            
                                            concept_data.append({
                                                "iteration": iteration_num,
                                                "tags": tags,
                                                "concepts": clean_concepts,
                                                "files_touched": files_touched
                                            })
    
    # HIERARCHICAL CONCEPT AGGREGATION PIPELINE
    # Level 2: Phase Detection - Analyze conversation flow and group related work
    print(f"Starting hierarchical pipeline with {len(iteration_summaries)} iteration summaries")
    
    phase_data = []
    subphase_data = []
    phase_summaries = []
    canonical_concepts = []
    
    if len(iteration_summaries) > 1:
        # Level 2: Phase Detection
        phase_detector = PhaseDetectorAgent()
        summaries_text = "\n\n".join(iteration_summaries)
        
        phase_goal = f"Analyze these {len(iteration_summaries)} iteration summaries to identify logical phases and boundaries using PhaseDetectorTool: {summaries_text}"
        phase_prompt = f'agent goal="{phase_goal}", iterations=10'
        
        try:
            phase_result = await phase_detector.run(phase_prompt)
            print(f"Phase detection complete. History ID: {phase_result['history_id']}")
            
            # Extract phase boundaries from the result
            # TODO: Add proper phase extraction logic based on tool outputs
            phase_data = [{"phase_id": 1, "iterations": list(range(1, len(iteration_summaries)+1)), "label": "Full Conversation"}]
            
        except Exception as e:
            print(f"Phase detection failed: {e}")
            # Fallback: treat entire conversation as one phase
            phase_data = [{"phase_id": 1, "iterations": list(range(1, len(iteration_summaries)+1)), "label": "Full Conversation"}]
        
        # Level 3: Subphase Detection for each phase
        for phase in phase_data:
            try:
                subphase_detector = SubphaseDetectorAgent()
                phase_iterations = [iteration_summaries[i-1] for i in phase["iterations"] if i-1 < len(iteration_summaries)]
                phase_text = "\n\n".join(phase_iterations)
                
                subphase_goal = f"Break this phase into subphases using SubphaseDetectorTool. Phase {phase['phase_id']} ({phase['label']}): {phase_text}"
                subphase_prompt = f'agent goal="{subphase_goal}", iterations=10'
                
                subphase_result = await subphase_detector.run(subphase_prompt)
                print(f"Subphase detection for phase {phase['phase_id']} complete. History ID: {subphase_result['history_id']}")
                
                # TODO: Extract actual subphase data from tool outputs
                subphase_data.append({
                    "phase_id": phase["phase_id"],
                    "subphases": [{"subphase_id": f"{phase['phase_id']}.1", "iterations": phase["iterations"], "label": f"Subphase {phase['phase_id']}.1"}]
                })
                
            except Exception as e:
                print(f"Subphase detection failed for phase {phase['phase_id']}: {e}")
                # Fallback: single subphase for the phase
                subphase_data.append({
                    "phase_id": phase["phase_id"],
                    "subphases": [{"subphase_id": f"{phase['phase_id']}.1", "iterations": phase["iterations"], "label": f"Subphase {phase['phase_id']}.1"}]
                })
        
        # Level 4: Phase Aggregation - Enhanced per-phase summaries
        for phase in phase_data:
            try:
                phase_aggregator = PhaseAggregatorAgent()
                phase_iterations = [iteration_summaries[i-1] for i in phase["iterations"] if i-1 < len(iteration_summaries)]
                
                phase_summary = await phase_aggregator.aggregate_phase(phase_iterations, phase["label"])
                phase_summaries.append({
                    "phase_id": phase["phase_id"],
                    "label": phase["label"],
                    "summary": phase_summary
                })
                print(f"Phase aggregation for phase {phase['phase_id']} complete")
                
            except Exception as e:
                print(f"Phase aggregation failed for phase {phase['phase_id']}: {e}")
                # Fallback: basic concatenation
                phase_iterations = [iteration_summaries[i-1] for i in phase["iterations"] if i-1 < len(iteration_summaries)]
                basic_summary = "\n\n".join(phase_iterations)
                phase_summaries.append({
                    "phase_id": phase["phase_id"],
                    "label": phase["label"],
                    "summary": basic_summary
                })
        
        print(f"Phase aggregation complete: {len(phase_data)} phases, {len(subphase_data)} subphase groups")
    
    # Level 5: Summary Aggregation - Executive summary from phase summaries
    if len(iteration_summaries) > 1:
        # Create aggregator to combine phase summaries into executive summary
        aggregator = AggregationSummarizerAgent()
        
        # Build batched conversation to avoid 30k character limit
        from datetime import datetime
        
        # Split summaries into chunks of 15 to stay under character limits
        chunk_size = 15
        messages = []
        
        # Start with system message
        messages.append(SystemMessage(content=aggregator.config.system_prompt))
        
        # Build aggregator input with FULL hierarchical phase/subphase structure
        aggregator_input_data = []
        
        # Add phase summaries first (high-level overview)
        for phase_summary in phase_summaries:
            aggregator_input_data.append(f"## PHASE SUMMARY {phase_summary['phase_id']}: {phase_summary['label']}\n{phase_summary['summary']}")
        
        # Add subphase analysis for each phase
        for subphase_group in subphase_data:
            subphase_content = f"## SUBPHASE ANALYSIS for Phase {subphase_group['phase_id']}:\n"
            for subphase in subphase_group['subphases']:
                # Get iterations for this subphase
                subphase_iterations = [iteration_summaries[i-1] for i in subphase['iterations'] if i-1 < len(iteration_summaries)]
                subphase_content += f"### Subphase {subphase['subphase_id']}: {subphase['label']}\n**Iterations:** {subphase['iterations']}\n**Content:**\n" + "\n".join(subphase_iterations) + "\n\n"
            aggregator_input_data.append(subphase_content)
        
        # Initial instruction
        messages.append(HumanMessage(content=f"I have complete hierarchical analysis with {len(phase_summaries)} phase summaries and {len(subphase_data)} subphase analyses to aggregate. I'll send them in chunks."))
        
        # Process hierarchical data in chunks  
        for i in range(0, len(aggregator_input_data), chunk_size):
            chunk = aggregator_input_data[i:i+chunk_size]
            chunk_text = "\n\n".join(chunk)
            batch_num = (i // chunk_size) + 1
            
            # User message with chunk
            messages.append(HumanMessage(content=f"Hierarchical analysis batch {batch_num}: {chunk_text}"))
            
            # AI acknowledgment (except for last batch)
            if i + chunk_size < len(aggregator_input_data):
                messages.append(AIMessage(content="OK, received hierarchical batch. Ready for next batch."))
        
        # Set the conversation history on the aggregator
        aggregator.history.messages = messages
        
        # Final instruction to create aggregated summary from hierarchical data
        prompt = "That's all hierarchical analysis (phase summaries + subphase analyses). Now create a comprehensive aggregated summary using AggregationSummarizerTool."
        
        # Run the aggregator
        result = await aggregator.run(prompt)
        print(f"Aggregation summarizer run complete. History ID: {result['history_id']}")
        
        # Look for tool calls
        aggregator.look_for_particular_tool_calls()
        
        # Level 6: Concept Resolution - Canonical concepts (runs AFTER summary aggregation)
        canonical_concepts = []
        try:
            concept_resolver = ConceptResolverAgent()
            all_phase_summaries = [p["summary"] for p in phase_summaries]
            executive_summary = aggregator.last_summary or "No executive summary"
            
            conversation_context = {
                "history_id": history_id,
                "total_iterations": len(iteration_summaries),
                "phase_count": len(phase_data),
                "subphase_count": sum(len(sp["subphases"]) for sp in subphase_data),
                "executive_summary": executive_summary
            }
            
            concept_goal = f"Resolve canonical concepts and relationships using ConceptResolverTool. Executive summary: {executive_summary}. Phase summaries: {all_phase_summaries}. Context: {conversation_context}"
            concept_prompt = f'agent goal="{concept_goal}", iterations=15'
            
            concept_result = await concept_resolver.run(concept_prompt)
            print(f"Concept resolution complete. History ID: {concept_result['history_id']}")
            
            # TODO: Extract canonical concepts from tool outputs
            canonical_concepts = ["concept_placeholder_1", "concept_placeholder_2"]
            
        except Exception as e:
            print(f"Concept resolution failed: {e}")
            canonical_concepts = []
        
        print(f"Hierarchical pipeline complete: {len(phase_data)} phases, {len(subphase_data)} subphase groups, {len(canonical_concepts)} canonical concepts")
        
        # Load concepts to Neo4j if pipeline completed successfully
        try:
            if canonical_concepts and len(canonical_concepts) > 0:
                print("Loading concepts to Neo4j...")
                
                # Prepare concept data for Neo4j
                concept_resolution_output = {
                    'canonical_concepts': {},
                    'concept_relationships': []
                }
                
                # Convert concept_data to Neo4j format
                for idx, concept_item in enumerate(concept_data):
                    for concept in concept_item.get('concepts', []):
                        if isinstance(concept, dict) and concept.get('keyword'):
                            canonical_form = concept.get('keyword', f'concept_{idx}')
                            concept_resolution_output['canonical_concepts'][canonical_form] = {
                                'canonical_form': canonical_form,
                                'type': 'conversation_concept',
                                'description': concept.get('description', 'No description'),
                                'phases_mentioned': [1]  # Default to phase 1
                            }
                
                # Prepare conversation metadata
                conversation_metadata = {
                    'conversation_id': history_id,
                    'source': 'HEAVEN_auto_summarize',
                    'total_iterations': len(iteration_summaries),
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_complete': True
                }
                
                # Load to Neo4j
                neo4j_result = load_concepts_to_neo4j(concept_resolution_output, conversation_metadata)
                print(f"Neo4j loading result: {neo4j_result}")
                
        except Exception as e:
            print(f"Neo4j loading failed (continuing anyway): {e}")
        
        # Return ALL components: Enhanced hierarchical structure
        return {
            "iteration_summaries": iteration_summaries,
            "aggregated_summary": aggregator.last_summary or "Failed to create aggregated summary",
            "concept_data": concept_data,
            # Hierarchical pipeline results
            "phase_data": phase_data,
            "subphase_data": subphase_data,
            "phase_summaries": phase_summaries,
            "canonical_concepts": canonical_concepts,
            "pipeline_complete": True
        }
    
    elif len(iteration_summaries) == 1:
        # Single iteration case - no hierarchical processing needed
        return {
            "iteration_summaries": iteration_summaries,
            "aggregated_summary": iteration_summaries[0],
            "concept_data": concept_data,
            # Empty hierarchical data for single iteration
            "phase_data": [],
            "subphase_data": [],
            "phase_summaries": [],
            "canonical_concepts": [],
            "pipeline_complete": False
        }
    
    else:
        # Fallback case
        fallback = summarizer.last_summary or "Failed to create summary"
        return {
            "iteration_summaries": [fallback],
            "aggregated_summary": fallback,
            "concept_data": [],
            # Empty hierarchical data for fallback
            "phase_data": [],
            "subphase_data": [],
            "phase_summaries": [],
            "canonical_concepts": [],
            "pipeline_complete": False
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