#!/usr/bin/env python3
"""
HEAVEN MCP Client

Complete MCP (Model Context Protocol) client for HEAVEN framework.
Provides wrapper functions for all MCP-Use API endpoints, including registry,
discovery, execution, testing, and session management.

The MCP system allows access to 5000+ MCP servers from the Smithery registry,
each containing multiple tools for various tasks.
"""

import requests
import json
from typing import Dict, List, Any, Optional

# Default API URL for MCP-Use sidecar
DEFAULT_API_URL = "http://host.docker.internal:9000"

#
# REGISTRY ENDPOINTS
#

def list_servers(api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """List all registered MCP servers"""
    response = requests.get(f"{api_url}/servers")
    return response.json()

def get_server(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Get configuration for a specific MCP server"""
    response = requests.get(f"{api_url}/servers/{server_name}")
    return response.json()

def register_server(server_name: str, config: Dict[str, Any], api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Register an MCP server configuration"""
    payload = {
        "server_name": server_name,
        "config": config
    }
    response = requests.post(f"{api_url}/servers", json=payload)
    return response.json()

def unregister_server(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Unregister an MCP server"""
    response = requests.delete(f"{api_url}/servers/{server_name}")
    return response.json()

def blacklist_server(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Add an MCP server to the blacklist"""
    response = requests.post(f"{api_url}/servers/blacklist/{server_name}")
    return response.json()

def unblacklist_server(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Remove an MCP server from the blacklist"""
    response = requests.post(f"{api_url}/servers/unblacklist/{server_name}")
    return response.json()

#
# DISCOVERY ENDPOINTS
#

def discover_servers(
    api_key: Optional[str] = None,
    query: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Discover available MCP servers from Smithery registry"""
    payload = {
        "api_key": api_key,
        "query": query,
        "page": page,
        "page_size": page_size
    }
    response = requests.post(f"{api_url}/discovery/servers", json=payload)
    return response.json()

def get_server_info(
    server_name: str,
    api_key: Optional[str] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Get detailed information about an MCP server from Smithery"""
    params = {}
    if api_key:
        params["api_key"] = api_key
    
    response = requests.get(f"{api_url}/discovery/servers/{server_name}", params=params)
    return response.json()

def discover_and_register(
    server_name: str,
    api_key: Optional[str] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Discover, register, and test an MCP server in one operation"""
    params = {}
    if api_key:
        params["api_key"] = api_key
    
    response = requests.post(f"{api_url}/discovery/register/{server_name}", params=params)
    return response.json()

#
# EXECUTION ENDPOINTS
#

def execute_tool(
    server_name: str,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Execute a tool on a registered MCP server"""
    payload = {
        "server_name": server_name,
        "tool_name": tool_name,
        "tool_args": tool_args or {}
    }
    response = requests.post(f"{api_url}/tools/execute", json=payload)
    return response.json()

def agent_query(
    server_name: str,
    query: str,
    model_name: str = "o4-mini",
    max_steps: int = 30,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Run a natural language query on a specific MCP server using an agent"""
    payload = {
        "server_name": server_name,
        "query": query,
        "model_name": model_name,
        "max_steps": max_steps
    }
    response = requests.post(f"{api_url}/agents/query", json=payload)
    return response.json()

def manager_query(
    query: str,
    servers: Optional[List[str]] = None,
    model_name: str = "o4-mini",
    max_steps: int = 30,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Run a query using the server manager agent (can access multiple servers)"""
    payload = {
        "query": query,
        "model_name": model_name,
        "max_steps": max_steps
    }
    if servers is not None:
        payload["servers"] = servers
        
    response = requests.post(f"{api_url}/manager/query", json=payload)
    return response.json()

#
# SESSION MANAGEMENT ENDPOINTS
#

def start_session(
    server_name: str, 
    config: Optional[Dict[str, Any]] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Start a new session with an MCP server"""
    payload = {
        "server_name": server_name
    }
    if config:
        payload["config"] = config
        
    response = requests.post(f"{api_url}/session/start", json=payload)
    return response.json()

def get_session_tools(api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """List all available tools on the current session"""
    response = requests.get(f"{api_url}/session/tools")
    return response.json()

def get_tool_args(tool_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Get parameter schema for a tool in the current session"""
    response = requests.get(f"{api_url}/session/tool/{tool_name}/args")
    return response.json()

def execute_session_tool(
    tool_name: str, 
    tool_args: Optional[Dict[str, Any]] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Execute a tool on the current session"""
    payload = {
        "tool_name": tool_name,
        "tool_args": tool_args or {}
    }
    response = requests.post(f"{api_url}/session/execute", json=payload)
    return response.json()

def close_session(api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Close the current session"""
    response = requests.post(f"{api_url}/session/close")
    return response.json()

#
# TESTING ENDPOINTS
#

def test_server(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Test connection to a registered MCP server"""
    response = requests.get(f"{api_url}/servers/test/{server_name}")
    return response.json()

def test_server_agent(server_name: str, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Test an MCP server using an agent"""
    response = requests.get(f"{api_url}/servers/test/{server_name}/agent")
    return response.json()

def test_manager(servers: Optional[List[str]] = None, api_url: str = DEFAULT_API_URL) -> Dict[str, Any]:
    """Test MCP servers using server manager agent"""
    payload = {}
    if servers is not None:
        payload["servers"] = servers
        
    response = requests.post(f"{api_url}/servers/test/manager", json=payload)
    return response.json()

#
# CONVENIENCE FUNCTIONS
#

def use_mcp_via_server_manager(
    query: str,
    servers: Optional[List[str]] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """Execute a query using the server manager (agentic approach)"""
    return manager_query(query, servers, api_url=api_url)

def use_mcp_via_session_manager(
    server_name: str,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """
    Execute a tool using the session manager (direct function approach).
    
    This function:
    1. Starts a new session with the specified server
    2. Executes the specified tool
    3. Closes the session
    
    Args:
        server_name: Name of the MCP server to use
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool
        api_url: URL of the MCP-Use API service
        
    Returns:
        Dict with tool execution result
    """
    try:
        # Start session
        start_result = start_session(server_name, api_url=api_url)
        if start_result.get("status") != "ok":
            return {
                "status": "error",
                "error": f"Failed to start session: {start_result.get('error', 'Unknown error')}"
            }
        
        # Execute tool
        execute_result = execute_session_tool(tool_name, tool_args, api_url=api_url)
        
        # Always close session, even if tool execution fails
        close_session(api_url=api_url)
        
        # Return execute result
        if execute_result.get("status") != "ok":
            return {
                "status": "error",
                "error": f"Tool execution failed: {execute_result.get('error', 'Unknown error')}"
            }
        
        return {
            "status": "ok",
            "server": server_name,
            "tool": tool_name,
            "args": tool_args,
            "result": execute_result.get("result")
        }
    except Exception as e:
        # Ensure session is closed on error
        try:
            close_session(api_url=api_url)
        except:
            pass
        
        return {
            "status": "error",
            "error": f"Error in session manager flow: {str(e)}"
        }

# Example usage and testing
if __name__ == "__main__":
    print("HEAVEN MCP Client - Example Usage")
    print("=" * 40)
    
    # Example 1: Using server manager
    print("\n1. Running a query with the server manager")
    manager_result = use_mcp_via_server_manager(
        query="What servers do you have access to? List all tools on each server."
    )
    print(f"Server Manager Result: {json.dumps(manager_result, indent=2)}")
    
    # Example 2: Discovery
    print("\n2. Discovering servers from Smithery")
    discover_result = discover_servers(page_size=5)
    print(f"Discovery Result: {json.dumps(discover_result, indent=2)}")
    
    # Example 3: Session manager (if servers available)
    if discover_result.get("status") == "ok" and discover_result.get("servers"):
        test_server = discover_result["servers"][0]
        print(f"\n3. Using session manager with server: {test_server}")
        
        tool_result = use_mcp_via_session_manager(
            server_name=test_server,
            tool_name="list_directory",
            tool_args={"path": "/tmp"}
        )
        print(f"Tool Execution Result: {json.dumps(tool_result, indent=2)}")