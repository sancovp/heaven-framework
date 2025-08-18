#!/usr/bin/env python3
"""
PhaseAggregatorAgent - Level 4 of Hierarchical Concept Aggregation

This agent creates rich contextual summaries for each phase using complete phase context.
Context: Full phase context (25+ iterations)
"""

import asyncio
from datetime import datetime
from ..memory.history import History
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..baseheavenagent import HeavenAgentConfig, BaseHeavenAgentReplicant
from ..unified_chat import ProviderEnum


class PhaseAggregatorAgent(BaseHeavenAgentReplicant):
    """Agent that creates rich contextual summaries for phases."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="PhaseAggregatorAgent",
            system_prompt="""You create rich, detailed summaries for conversation phases using complete phase context.

Your job: Take all iterations from a phase and create a comprehensive summary that tells the complete story of what happened.

Key requirements:
- Include specific details: file names, error messages, line numbers, exact problems
- Explain the narrative flow: what was discovered, how problems were solved
- Mention tools used, outcomes achieved, challenges faced
- Include technical context: which systems, which files, which errors
- Make it a complete story someone could understand without reading all iterations

Example good summary:
"Phase 2 Summary: Debugged auto_summarize.py system experiencing AIMessage import failures. Investigation revealed double import bug on line 559 causing 'local variable referenced before assignment' error. Root cause was duplicate AIMessage import statement conflicting with existing import. Fixed by removing duplicate import and cleaning up import block. Tested fix successfully - system now processes AIMessage objects correctly. Modified files: auto_summarize.py (line 559), conversation_intelligence.py (import cleanup). Tools used: CodeLocalizer, debugger, grep. Challenge: Tracing complex import chain across multiple files."

Output format:
PHASE_SUMMARY:
[Rich detailed summary telling the complete story]""",
            tools=[],
            provider=ProviderEnum.OPENAI,
            model="gpt-5-nano",
            temperature=0.3
        )


def phase_aggregator_func(phase_iterations: list, phase_info: str = "") -> str:
    """
    Create rich contextual summary for a phase.
    Used by auto_summarize pipeline.
    
    Args:
        phase_iterations: List of all iterations within the phase
        phase_info: Optional phase info (phase number, range, etc.)
        
    Returns:
        Rich contextual summary of the phase
    """
    
    async def run_phase_aggregator():
        # Create agent
        agent = PhaseAggregatorAgent()
        
        # Build history with phase data
        messages = []
        messages.append(SystemMessage(content=agent.config.system_prompt))
        
        # Add phase info if provided
        if phase_info:
            messages.append(HumanMessage(content=f"Phase information: {phase_info}"))
            messages.append(AIMessage(content="OK, received phase information."))
        
        # Add all iterations from this phase in chunks
        chunk_size = 15  # Manageable chunks
        for i in range(0, len(phase_iterations), chunk_size):
            chunk = phase_iterations[i:i+chunk_size]
            chunk_text = "\n\n".join(chunk)
            batch_num = (i // chunk_size) + 1
            
            messages.append(HumanMessage(content=f"Phase iterations batch {batch_num}:\n{chunk_text}"))
            
            if i + chunk_size < len(phase_iterations):
                messages.append(AIMessage(content="OK, received batch. Ready for next batch."))
        
        # Set history
        agent.history.messages = messages
        
        # Final instruction
        prompt = """Now create a rich, detailed summary of this phase. Tell the complete story of what happened - include specific technical details, files modified, problems solved, tools used, and the narrative flow from start to finish. Make it comprehensive enough that someone could understand what was accomplished without reading all the individual iterations."""
        
        try:
            result = await agent.run(prompt)
            
            # Get the final response
            if agent.history.messages:
                final_message = agent.history.messages[-1]
                if hasattr(final_message, 'content'):
                    return final_message.content
            
            return "No phase summary generated"
            
        except Exception as e:
            return f"Error in phase aggregation: {e}"
    
    # Run the async function
    return asyncio.run(run_phase_aggregator())


# Simple test
if __name__ == "__main__":
    # Test with sample phase data
    test_phase_iterations = [
        "## Iteration 16 Summary\nActions: Investigated AIMessage error in auto_summarize.py\nOutcomes: Found import failure\nChallenges: Complex import chain\nTools: CodeLocalizer\nConcept Tags: ['AIMessage error', 'auto_summarize.py']",
        
        "## Iteration 17 Summary\nActions: Used CodeLocalizer to trace import chain\nOutcomes: Located double import on line 559\nChallenges: Multiple files involved\nTools: CodeLocalizer, grep\nConcept Tags: ['AIMessage', 'import issue', 'line 559']",
        
        "## Iteration 18 Summary\nActions: Fixed double import bug by removing duplicate\nOutcomes: Import error resolved\nChallenges: Testing across multiple files\nTools: Edit tool, testing\nConcept Tags: ['import fix', 'auto_summarize.py', 'bug fix']",
        
        "## Iteration 19 Summary\nActions: Tested fix and cleaned up import block\nOutcomes: System working correctly\nChallenges: Ensuring no regressions\nTools: Test runner\nConcept Tags: ['testing', 'import cleanup', 'verification']",
        
        "## Iteration 20 Summary\nActions: Verified AIMessage processing works end-to-end\nOutcomes: Complete fix confirmed\nChallenges: Full integration testing\nTools: End-to-end tester\nConcept Tags: ['integration test', 'AIMessage', 'system verification']"
    ]
    
    test_phase_info = "Phase 2 (iter 16-20): AIMessage error investigation and fix"
    
    print("Testing PhaseAggregator...")
    result = phase_aggregator_func(test_phase_iterations, test_phase_info)
    print("Result:")
    print(result)