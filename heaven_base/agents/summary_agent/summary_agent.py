# summary_agent.py
import os
from typing import Optional, List, Type
from ...baseheavenagent import BaseHeavenAgentReplicant, HeavenAgentConfig
from ...baseheaventool import BaseHeavenTool
from ...unified_chat import UnifiedChat, ProviderEnum
from ...tools.straightforwardsummarizer_tool import StraightforwardSummarizerTool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage

class SummaryAgent(BaseHeavenAgentReplicant):
    def __init__(self, 
                 history_id: Optional[str] = None, 
                 system_prompt_suffix: str = '',
                 additional_tools: Optional[List[Type[BaseHeavenTool]]] = None,
                 remove_agents_config_tools: bool = False):
      super().__init__(
        history_id=history_id, 
        system_prompt_suffix=system_prompt_suffix,
        additional_tools=additional_tools,
        remove_agents_config_tools=remove_agents_config_tools
      )
      self.last_summary = None
      
    @classmethod
    def get_default_config(cls) -> HeavenAgentConfig:
        return HeavenAgentConfig(
            name="summary_agent",
            system_prompt="You are a specialized agent that creates clear, straightforward summaries. You analyze conversations and create: overall summary, list of completed tasks, and key observations with a list of obstacles encountered by the agent (if any; a task counts as an obstacle (obstacle: have to do X) and a problem during the task is also an obstacle (obstacle: have to do X, while doing X encountered another obstacle Y...)) and the attempted overcomes (if any) and result.",
            tools=[StraightforwardSummarizerTool],
            provider=ProviderEnum.ANTHROPIC,
            model="claude-3-5-haiku-latest",
            temperature=0.2
        )

    def look_for_particular_tool_calls(self) -> None:
        for i, msg in enumerate(self.history.messages):
            if isinstance(msg, AIMessage) and isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        if item.get('name') == "StraightforwardSummarizerTool":
                            # print(f"StraightforwardSummarizerTool call detected on {self.original_history_id} with path {self.original_json_md_path}!")
                            # Get the next message which should be the ToolMessage with the result
                            if i + 1 < len(self.history.messages):
                                tool_result = self.history.messages[i + 1]
                                if isinstance(tool_result, ToolMessage):
                                    self.save_summary(tool_result.content)



    def save_summary(self, summary_content: str) -> None:
        if not self.original_history_id or not self.original_json_md_path:
            raise ValueError(f"No history_id or path found! history_id: {self.original_history_id} and path: {self.original_json_md_path}")

        # Save summary right next to the original history files
        summary_filepath = os.path.join(self.original_json_md_path, f"{self.original_history_id}_summary.md")
        with open(summary_filepath, 'w') as f:
            f.write(summary_content)
        self.last_summary = summary_content
