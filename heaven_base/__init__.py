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

# Conversation management
from .memory.conversations import (
    ConversationManager,
    start_chat,
    continue_chat,
    load_chat,
    list_chats,
    search_chats,
    get_latest_history
)

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

# MCP (Model Context Protocol) Integration
from .utils.mcp_client import (
    # Registry functions
    list_servers, get_server, register_server, unregister_server,
    blacklist_server, unblacklist_server,
    # Discovery functions
    discover_servers, get_server_info, discover_and_register,
    # Execution functions
    execute_tool, agent_query, manager_query,
    # Session management
    start_session, close_session, get_session_tools, get_tool_args, execute_session_tool,
    # Testing functions
    test_server, test_server_agent, test_manager,
    # Convenience functions
    use_mcp_via_server_manager, use_mcp_via_session_manager
)

from .utils.mcp_tool_converter import (
    create_mcp_tool_info, create_heaven_tool_from_mcp_tool,
    create_all_tools_for_mcp_server, get_mcp_server_tools,
    test_mcp_tool_conversion
)

from .utils.mcp_agent_orchestrator import (
    discover_and_create_agent_for_mcp_server, create_mcp_agent_config,
    create_multi_server_mcp_agent, discover_popular_mcp_servers,
    create_smart_mcp_agent
)

from .tools.mcp_session_tools import (
    connect_mcp_session, disconnect_mcp_session, get_mcp_session_status,
    ConnectMCPSessionTool, DisconnectMCPSessionTool, GetMCPSessionStatusTool
)

# CLI
from .cli.heaven_cli import HeavenCLI, make_cli

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
    "call_summary_agent",
    
    # MCP Client functions
    "list_servers", "get_server", "register_server", "unregister_server",
    "blacklist_server", "unblacklist_server",
    "discover_servers", "get_server_info", "discover_and_register", 
    "execute_tool", "agent_query", "manager_query",
    "start_session", "close_session", "get_session_tools", "get_tool_args", "execute_session_tool",
    "test_server", "test_server_agent", "test_manager",
    "use_mcp_via_server_manager", "use_mcp_via_session_manager",
    
    # MCP Tool Converter functions
    "create_mcp_tool_info", "create_heaven_tool_from_mcp_tool",
    "create_all_tools_for_mcp_server", "get_mcp_server_tools",
    "test_mcp_tool_conversion",
    
    # MCP Agent Orchestrator functions
    "discover_and_create_agent_for_mcp_server", "create_mcp_agent_config",
    "create_multi_server_mcp_agent", "discover_popular_mcp_servers",
    "create_smart_mcp_agent",
    
    # MCP Session Management tools
    "connect_mcp_session", "disconnect_mcp_session", "get_mcp_session_status",
    "ConnectMCPSessionTool", "DisconnectMCPSessionTool", "GetMCPSessionStatusTool",
    
    # CLI
    "HeavenCLI", "make_cli"
]