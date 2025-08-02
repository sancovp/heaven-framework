"""
HEAVEN Base - Hierarchical, Embodied, Autonomously Validating Evolution Network
Complete core framework for building autonomous AI agents.
"""

__version__ = "1.2.0"

# Core agent classes
from .baseheavenagent import (
    BaseHeavenAgent,
    BaseHeavenAgentReplicant,
    HeavenAgentConfig,
    DuoSystemConfig,
    HookPoint,
    HookContext,
    HookRegistry
)

# Core tool classes
from .baseheaventool import (
    BaseHeavenTool,
    ToolArgsSchema,
    ToolResult,
    ToolError,
    CLIResult
)

# Unified chat interface
from .unified_chat import (

    UnifiedChat,

    ProviderEnum

)

# Decorators
from .decorators import heaven_tool, make_heaven_tool_from_function
from .make_heaven_tool_from_docstring import make_heaven_tool_from_docstring

# Memory and history
from .memory.history import (
    History,
    AgentStatus
)
from .memory.heaven_event import HeavenEvent
from .memory.heaven_history import HeavenHistory
from .memory.base_piece import BasePiece

# Registry system
from .tools.registry_tool import RegistryTool

# Prompts system
from .prompts.heaven_variable import RegistryHeavenVariable

# Utils
from .utils.name_utils import (
    normalize_agent_name,
    camel_to_snake,
    pascal_to_snake,
    to_pascal_case,
    resolve_agent_name
)

# Tools
from .tools.write_block_report_tool import WriteBlockReportTool

# LangGraph integration
from .langgraph import HeavenState, HeavenNodeType, completion_runner, hermes_runner

# Configs
from .configs.hermes_config import HermesConfig

# Agents
from .agents.summary_agent.summary_agent import SummaryAgent
from .agents.summary_agent.summary_util import call_summary_agent

__all__ = [
    # Version
    "__version__",
    
    # Agent classes
    "BaseHeavenAgent",
    "BaseHeavenAgentReplicant", 
    "HeavenAgentConfig",
    "DuoSystemConfig",
    "HookPoint",
    "HookContext",
    "HookRegistry",
    
    # Tool classes
    "BaseHeavenTool",
    "ToolArgsSchema",
    "ToolResult",
    "ToolError",
    "CLIResult",
    
    # Chat
    "UnifiedChat",
    "ProviderEnum",
    
    # Decorators
    "heaven_tool",
    "make_heaven_tool_from_function",
    "make_heaven_tool_from_docstring",
    
    # Memory
    "History",
    "AgentStatus",
    "HeavenEvent",
    "HeavenHistory",
    "BasePiece",
    
    # Registry
    "RegistryTool",
    
    # Prompts
    "RegistryHeavenVariable",
    
    # Utils
    "normalize_agent_name",
    "camel_to_snake",
    "pascal_to_snake",
    "to_pascal_case",
    "resolve_agent_name",
    
    # Tools
    "WriteBlockReportTool",
    
    # LangGraph
    "HeavenState",
    "HeavenNodeType", 
    "completion_runner",
    "hermes_runner",
    
    # Configs
    "HermesConfig",
    
    # Agents
    "SummaryAgent",
    "call_summary_agent"
]