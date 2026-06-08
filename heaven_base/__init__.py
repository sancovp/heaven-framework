"""
HEAVEN Base - Hierarchical, Embodied, Autonomously Validating Evolution Network
Complete core framework for building autonomous AI agents.
"""

# Redirect print() to stderr to prevent MCP protocol corruption
# Set HEAVEN_ALLOW_STDOUT=1 to disable this (e.g. for docker cross-machine scenarios)
import sys
import os
import builtins

if os.environ.get("HEAVEN_ALLOW_STDOUT") != "1":
    _original_print = builtins.print
    def _stderr_print(*args, **kwargs):
        kwargs.setdefault('file', sys.stderr)
        _original_print(*args, **kwargs)
    builtins.print = _stderr_print

__version__ = "0.1.21"

# Lazy imports — heavy deps (langchain → transformers → torch) load on first access only
_LAZY_IMPORTS = {
    "BaseHeavenAgent": ".baseheavenagent",
    "BaseHeavenAgentReplicant": ".baseheavenagent",
    "HeavenAgentConfig": ".baseheavenagent",
    "DuoSystemConfig": ".baseheavenagent",
    "HookPoint": ".baseheavenagent",
    "HookContext": ".baseheavenagent",
    "HookRegistry": ".baseheavenagent",
    "BaseHeavenTool": ".baseheaventool",
    "ToolArgsSchema": ".baseheaventool",
    "ToolResult": ".baseheaventool",
    "ToolError": ".baseheaventool",
    "CLIResult": ".baseheaventool",
    "UnifiedChat": ".unified_chat",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache for next access
        return val
    raise AttributeError(f"module 'heaven_base' has no attribute {name!r}")

# These were previously imported eagerly — now lazy via __getattr__
# from .baseheavenagent import BaseHeavenAgent, ...
# from .baseheaventool import BaseHeavenTool, ...
# from .unified_chat import UnifiedChat, ...

# ALL imports below are lazy — resolved via __getattr__ on first access.
# This prevents torch/langchain/transformers from loading for lightweight consumers
# (strata MCPs that only need registry_tool, etc.)

_LAZY_IMPORTS.update({
    "ProviderEnum": ".unified_chat",
    "heaven_tool": ".decorators", "make_heaven_tool_from_function": ".decorators",
    "make_heaven_tool_from_docstring": ".make_heaven_tool_from_docstring",
    "History": ".memory.history", "AgentStatus": ".memory.history",
    "HeavenEvent": ".memory.heaven_event",
    "HeavenHistory": ".memory.heaven_history",
    "BasePiece": ".memory.base_piece",
    "ConversationManager": ".memory.conversations",
    "start_chat": ".memory.conversations", "continue_chat": ".memory.conversations",
    "load_chat": ".memory.conversations", "list_chats": ".memory.conversations",
    "search_chats": ".memory.conversations", "get_latest_history": ".memory.conversations",
    "RegistryTool": ".tools.registry_tool",
    "RegistryHeavenVariable": ".prompts.heaven_variable",
    "normalize_agent_name": ".utils.name_utils", "camel_to_snake": ".utils.name_utils",
    "pascal_to_snake": ".utils.name_utils", "to_pascal_case": ".utils.name_utils",
    "resolve_agent_name": ".utils.name_utils",
    "WriteBlockReportTool": ".tools.write_block_report_tool",
    "HeavenState": ".langgraph", "HeavenNodeType": ".langgraph",
    "completion_runner": ".langgraph", "hermes_runner": ".langgraph",
    "HermesConfig": ".configs.hermes_config",
    "SummaryAgent": ".agents.summary_agent.summary_agent",
    "call_summary_agent": ".agents.summary_agent.summary_util",
    "HeavenCLI": ".cli.heaven_cli", "make_cli": ".cli.heaven_cli",
    "list_servers": ".utils.mcp_client", "get_server": ".utils.mcp_client",
    "register_server": ".utils.mcp_client", "unregister_server": ".utils.mcp_client",
    "blacklist_server": ".utils.mcp_client", "unblacklist_server": ".utils.mcp_client",
    "discover_servers": ".utils.mcp_client", "get_server_info": ".utils.mcp_client",
    "discover_and_register": ".utils.mcp_client",
    "execute_tool": ".utils.mcp_client", "agent_query": ".utils.mcp_client",
    "manager_query": ".utils.mcp_client",
    "start_session": ".utils.mcp_client", "close_session": ".utils.mcp_client",
    "get_session_tools": ".utils.mcp_client", "get_tool_args": ".utils.mcp_client",
    "execute_session_tool": ".utils.mcp_client",
    "test_server": ".utils.mcp_client", "test_server_agent": ".utils.mcp_client",
    "test_manager": ".utils.mcp_client",
    "use_mcp_via_server_manager": ".utils.mcp_client",
    "use_mcp_via_session_manager": ".utils.mcp_client",
    "create_mcp_tool_info": ".utils.mcp_tool_converter",
    "create_heaven_tool_from_mcp_tool": ".utils.mcp_tool_converter",
    "create_all_tools_for_mcp_server": ".utils.mcp_tool_converter",
    "get_mcp_server_tools": ".utils.mcp_tool_converter",
    "test_mcp_tool_conversion": ".utils.mcp_tool_converter",
    "discover_and_create_agent_for_mcp_server": ".utils.mcp_agent_orchestrator",
    "create_mcp_agent_config": ".utils.mcp_agent_orchestrator",
    "create_multi_server_mcp_agent": ".utils.mcp_agent_orchestrator",
    "discover_popular_mcp_servers": ".utils.mcp_agent_orchestrator",
    "create_smart_mcp_agent": ".utils.mcp_agent_orchestrator",
    "connect_mcp_session": ".tools.mcp_session_tools",
    "disconnect_mcp_session": ".tools.mcp_session_tools",
    "get_mcp_session_status": ".tools.mcp_session_tools",
    "ConnectMCPSessionTool": ".tools.mcp_session_tools",
    "DisconnectMCPSessionTool": ".tools.mcp_session_tools",
    "GetMCPSessionStatusTool": ".tools.mcp_session_tools",
})

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
    "HermesTool",
    
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