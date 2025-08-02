"""Summary agent utilities."""
from typing import Optional
from .summary_agent import SummaryAgent

async def call_summary_agent(history_id: str) -> Optional[str]:
    """Call summary agent on a specific history.
    
    Args:
        history_id: ID of history to summarize
        
    Returns:
        Summary content if successful, None if failed
    """
    try:
        # Create agent with history_id
        agent = SummaryAgent(history_id=history_id)
        
        # Have it summarize
        result = await agent.run("Please analyze this conversation and create a summary using the StraightforwardSummarizerTool. Include: overall summary, completed tasks, and key observations.")
        
        # Summary will be saved by the agent
        return agent.last_summary
        
    except Exception as e:
        print(f"Error in call_summary_agent: {str(e)}")
        return None