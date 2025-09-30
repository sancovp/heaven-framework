"""
Heaven-Base Tools
Collection of tools for the HEAVEN agent framework
"""

# Import all tools
from .write_block_report_tool import WriteBlockReportTool
from .straightforwardsummarizer_tool import StraightforwardSummarizerTool
from .retrieve_tool_info_tool import RetrieveToolInfoTool
from .bash_tool import BashTool
from .network_edit_tool import NetworkEditTool
from .safe_code_reader_tool import SafeCodeReaderTool
from .code_localizer_tool import CodeLocalizerTool
from .think_tool import ThinkTool
from .websearch_tool import WebsearchTool
from .registry_tool import RegistryTool
from .workflow_relay_tool import WorkflowRelayTool
from .view_history_tool import ViewHistoryTool

# Prompt block tools
from .write_prompt_block_tool import WritePromptBlockTool
from .search_prompt_blocks_tool import SearchPromptBlocksTool
from .write_skillchain_type_prompt_block_tool import WriteSkillchainTypePromptBlockTool

# Additional tools
from .neo4j_tool import Neo4jTool
from .redaction_tool import RedactionTool
from .agent_config_test_tool import AgentConfigTestTool

# Acolyte chain tools - excluded to avoid circular imports - available separately
# from .acolyte_chain_tools import (
#     ScriptOnlyChainTool,
#     ScriptWithConfigsChainTool, 
#     AnalysisImprovementChainTool,
#     FullSystemBuildChainTool,
#     ToolGenerationChainTool,
#     PromptEngineeringChainTool
# )

__all__ = [
    # Original tools
    "WriteBlockReportTool",
    "StraightforwardSummarizerTool", 
    "RetrieveToolInfoTool",
    "BashTool",
    "NetworkEditTool",
    "SafeCodeReaderTool",
    "CodeLocalizerTool",
    "ThinkTool",
    "WebsearchTool",
    "RegistryTool",
    "WorkflowRelayTool",
    "ViewHistoryTool",
    
    # Prompt block tools
    "WritePromptBlockTool",
    "SearchPromptBlocksTool",
    "WriteSkillchainTypePromptBlockTool",
    
    # Additional tools
    "Neo4jTool",
    "RedactionTool",
    "AgentConfigTestTool"
    
    # Acolyte chain tools excluded to avoid circular imports
]