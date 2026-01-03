#!/usr/bin/env python3
"""
MCP Session Management Tools

These tools provide explicit MCP session management for HEAVEN agents.
Agents should use these tools to properly connect to and disconnect from
MCP servers before using MCP tools.

Usage pattern:
1. ConnectMCPSessionTool - Connect to MCP server
2. Use MCP tools
3. DisconnectMCPSessionTool - Disconnect from MCP server
"""

from typing import Dict, Any
from heaven_base.utils.mcp_client import start_session, close_session

def connect_mcp_session(server_name: str) -> Dict[str, Any]:
    """
    Connect to an MCP server session.
    
    This function opens a session with the specified MCP server.
    Agents should call this before using any MCP tools from that server.
    
    Args:
        server_name (str): Name of the MCP server to connect to (e.g., "@wonderwhy-er/desktop-commander")
        
    Returns:
        dict: Status and result information
    """
    try:
        result = start_session(server_name)
        if result.get("status") == "ok":
            return {
                "status": "success",
                "message": f"Successfully connected to MCP server: {server_name}",
                "server_name": server_name,
                "session_info": result
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to connect to MCP server: {server_name}",
                "error": result.get("error", "Unknown error"),
                "server_name": server_name
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to MCP server: {server_name}",
            "error": str(e),
            "server_name": server_name
        }

def disconnect_mcp_session() -> Dict[str, Any]:
    """
    Disconnect from the current MCP server session.
    
    This function closes the current MCP session.
    Agents should call this after they're done using MCP tools.
    
    Returns:
        dict: Status and result information
    """
    try:
        result = close_session()
        if result.get("status") == "ok":
            return {
                "status": "success", 
                "message": "Successfully disconnected from MCP server",
                "session_info": result
            }
        else:
            return {
                "status": "error",
                "message": "Failed to disconnect from MCP server",
                "error": result.get("error", "Unknown error")
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error disconnecting from MCP server",
            "error": str(e)
        }

def get_mcp_session_status() -> Dict[str, Any]:
    """
    Get the current MCP session status.
    
    This function checks if there's an active MCP session and provides
    information about available tools if connected.
    
    Returns:
        dict: Current session status and available tools
    """
    try:
        from heaven_base.utils.mcp_client import get_session_tools
        
        tools_result = get_session_tools()
        if tools_result.get("status") == "ok":
            tools = tools_result.get("tools", [])
            return {
                "status": "success",
                "message": f"Active MCP session with {len(tools)} tools available",
                "session_active": True,
                "available_tools": tools,
                "tool_count": len(tools)
            }
        else:
            return {
                "status": "info",
                "message": "No active MCP session",
                "session_active": False,
                "available_tools": [],
                "tool_count": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error checking MCP session status",
            "error": str(e),
            "session_active": False
        }

# Create HEAVEN tools from these functions
if __name__ != "__main__":
    try:
        from heaven_base import make_heaven_tool_from_docstring
        
        # Create the session management tools
        ConnectMCPSessionTool = make_heaven_tool_from_docstring(connect_mcp_session)
        DisconnectMCPSessionTool = make_heaven_tool_from_docstring(disconnect_mcp_session)
        GetMCPSessionStatusTool = make_heaven_tool_from_docstring(get_mcp_session_status)
        
    except ImportError:
        # Fallback if make_heaven_tool_from_docstring is not available
        ConnectMCPSessionTool = None
        DisconnectMCPSessionTool = None
        GetMCPSessionStatusTool = None

# Test the functions if run directly
if __name__ == "__main__":
    import json
    
    # Test server
    TEST_SERVER = "@wonderwhy-er/desktop-commander"
    
    print("=== Testing MCP Session Management ===\n")
    
    # Test session status (should be inactive)
    print("1. Checking initial session status...")
    status_result = get_mcp_session_status()
    print(f"Status: {json.dumps(status_result, indent=2)}\n")
    
    # Test connect
    print(f"2. Connecting to {TEST_SERVER}...")
    connect_result = connect_mcp_session(TEST_SERVER)
    print(f"Connect result: {json.dumps(connect_result, indent=2)}\n")
    
    if connect_result.get("status") == "success":
        # Test session status (should be active)
        print("3. Checking session status after connection...")
        status_result = get_mcp_session_status()
        print(f"Status: {json.dumps(status_result, indent=2)}\n")
        
        # Test disconnect
        print("4. Disconnecting from MCP server...")
        disconnect_result = disconnect_mcp_session()
        print(f"Disconnect result: {json.dumps(disconnect_result, indent=2)}\n")
        
        # Test final status
        print("5. Final session status check...")
        status_result = get_mcp_session_status()
        print(f"Status: {json.dumps(status_result, indent=2)}")
    else:
        print("Connection failed, skipping remaining tests.")