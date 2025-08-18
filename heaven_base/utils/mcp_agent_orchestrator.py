#!/usr/bin/env python3
"""
MCP Agent Orchestrator

High-level orchestration for creating HEAVEN agents that can use MCP servers.
This module combines MCP server discovery, tool conversion, and agent creation
into streamlined workflows.

Key workflows:
1. Discovery-based agent creation
2. Server-specific agent creation  
3. Multi-server agent orchestration
4. Dynamic MCP tool integration
"""

import json
from typing import Dict, Any, List, Optional

from .mcp_client import (
    discover_servers, discover_and_register, list_servers,
    use_mcp_via_server_manager, get_server_info
)
from .mcp_tool_converter import (
    create_all_tools_for_mcp_server, create_heaven_tool_from_mcp_tool,
    get_mcp_server_tools
)

def discover_and_create_agent_for_mcp_server(
    server_name: str,
    max_tools: Optional[int] = None,
    auto_register: bool = True
) -> Dict[str, Any]:
    """
    Discover, register, and create a HEAVEN agent for an MCP server.
    
    This is the main workflow for taking an MCP server name and creating
    a complete HEAVEN agent that can use all tools from that server.
    
    Args:
        server_name: Name of the MCP server (e.g., "@wonderwhy-er/desktop-commander")
        max_tools: Maximum number of tools to include (for testing)
        auto_register: Whether to automatically register the server if not found
        
    Returns:
        Dict with agent creation results
    """
    print(f"=== Creating HEAVEN Agent for MCP Server: {server_name} ===\n")
    
    # Step 1: Check if server is already registered
    print("Step 1: Checking server registration...")
    registered_servers = list_servers()
    
    server_registered = False
    if registered_servers.get("status") == "ok":
        server_list = registered_servers.get("servers", [])
        server_registered = server_name in server_list
        print(f"  Server {'already registered' if server_registered else 'not registered'}")
    
    # Step 2: Register server if needed
    if not server_registered and auto_register:
        print(f"\nStep 2: Auto-registering server {server_name}...")
        register_result = discover_and_register(server_name)
        if register_result.get("status") != "ok":
            return {
                "status": "error",
                "step": "registration",
                "error": f"Failed to register server: {register_result.get('error', 'Unknown error')}",
                "server_name": server_name
            }
        print(f"  ✓ Server registered successfully")
    elif not server_registered:
        return {
            "status": "error",
            "step": "registration",
            "error": "Server not registered and auto_register=False",
            "server_name": server_name
        }
    
    # Step 3: Get available tools
    print(f"\nStep 3: Discovering tools on {server_name}...")
    available_tools = get_mcp_server_tools(server_name)
    if not available_tools:
        return {
            "status": "error",
            "step": "tool_discovery",
            "error": "No tools found on server",
            "server_name": server_name
        }
    
    if max_tools and max_tools > 0:
        available_tools = available_tools[:max_tools]
        print(f"  Limiting to {max_tools} tools: {', '.join(available_tools)}")
    else:
        print(f"  Found {len(available_tools)} tools: {', '.join(available_tools)}")
    
    # Step 4: Create HEAVEN tools
    print(f"\nStep 4: Creating HEAVEN tools...")
    tools_result = create_all_tools_for_mcp_server(server_name, max_tools)
    if tools_result.get("status") != "ok":
        return {
            "status": "error",
            "step": "tool_creation",
            "error": tools_result.get("error"),
            "server_name": server_name
        }
    
    created_tools = tools_result.get("tools", [])
    failed_tools = tools_result.get("failed_tools", [])
    
    print(f"  ✓ Created {len(created_tools)} tools")
    if failed_tools:
        print(f"  ⚠ Failed to create {len(failed_tools)} tools")
    
    # Step 5: Create agent configuration
    print(f"\nStep 5: Creating agent configuration...")
    agent_config = create_mcp_agent_config(server_name, created_tools)
    
    return {
        "status": "ok",
        "server_name": server_name,
        "agent_config": agent_config,
        "created_tools": len(created_tools),
        "failed_tools": len(failed_tools),
        "tool_details": created_tools,
        "failed_tool_details": failed_tools,
        "available_tools": available_tools
    }

def create_mcp_agent_config(server_name: str, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a HEAVEN agent configuration for an MCP server.
    
    Args:
        server_name: Name of the MCP server
        tool_results: List of tool creation results
        
    Returns:
        Dict with agent configuration
    """
    # Extract tool classes from results
    tool_classes = []
    tool_names = []
    
    for tool_result in tool_results:
        if tool_result.get("status") == "ok":
            tool_classes.append(tool_result.get("tool_class"))
            tool_names.append(tool_result.get("tool_name"))
    
    # Add session management tools
    try:
        from heaven_base.tools.mcp_session_tools import (
            ConnectMCPSessionTool, DisconnectMCPSessionTool, GetMCPSessionStatusTool
        )
        if ConnectMCPSessionTool:
            tool_classes.extend([ConnectMCPSessionTool, DisconnectMCPSessionTool, GetMCPSessionStatusTool])
            tool_names.extend(["ConnectMCPSessionTool", "DisconnectMCPSessionTool", "GetMCPSessionStatusTool"])
    except ImportError:
        print("Warning: Session management tools not available")
    
    # Create clean server name for agent
    server_part = ''.join(word.capitalize() for word in server_name.replace('@', '').replace('/', ' ').replace('-', ' ').split())
    agent_name = f"MCP{server_part}Agent"
    
    # Create system prompt
    system_prompt = f"""You are an AI agent with access to tools from the MCP server "{server_name}".

MCP (Model Context Protocol) allows you to interact with external systems and services.

IMPORTANT: Before using any MCP tools from {server_name}, you must:
1. First use ConnectMCPSessionTool to connect to the server
2. Then use the available MCP tools as needed
3. Finally use DisconnectMCPSessionTool when done

Your available tools include:
{chr(10).join(f'- {name}: {name}' for name in tool_names)}

Session Management:
- ConnectMCPSessionTool: Connect to {server_name} before using its tools
- DisconnectMCPSessionTool: Disconnect when finished
- GetMCPSessionStatusTool: Check current session status

Always follow proper session management to prevent errors."""
    
    return {
        "agent_name": agent_name,
        "system_prompt": system_prompt,
        "tool_classes": tool_classes,
        "tool_names": tool_names,
        "server_name": server_name,
        "tool_count": len(tool_classes)
    }

def create_multi_server_mcp_agent(
    server_names: List[str],
    max_tools_per_server: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a HEAVEN agent that can access multiple MCP servers.
    
    Args:
        server_names: List of MCP server names
        max_tools_per_server: Maximum tools per server (for testing)
        
    Returns:
        Dict with multi-server agent configuration
    """
    print(f"=== Creating Multi-Server MCP Agent ===")
    print(f"Servers: {', '.join(server_names)}\n")
    
    all_tool_classes = []
    all_tool_names = []
    server_details = {}
    
    # Process each server
    for server_name in server_names:
        print(f"Processing server: {server_name}")
        
        # Create agent for this server
        server_result = discover_and_create_agent_for_mcp_server(
            server_name, 
            max_tools=max_tools_per_server,
            auto_register=True
        )
        
        if server_result.get("status") == "ok":
            config = server_result["agent_config"]
            all_tool_classes.extend(config["tool_classes"])
            all_tool_names.extend(config["tool_names"])
            
            server_details[server_name] = {
                "tool_count": config["tool_count"],
                "created_tools": server_result["created_tools"],
                "failed_tools": server_result["failed_tools"]
            }
            
            print(f"  ✓ Added {config['tool_count']} tools from {server_name}")
        else:
            print(f"  ✗ Failed to process {server_name}: {server_result.get('error')}")
            server_details[server_name] = {
                "error": server_result.get("error"),
                "tool_count": 0
            }
    
    # Create combined system prompt
    system_prompt = f"""You are an AI agent with access to tools from multiple MCP servers:
{chr(10).join(f'- {name}' for name in server_names)}

MCP (Model Context Protocol) allows you to interact with external systems and services.

IMPORTANT SESSION MANAGEMENT:
Before using tools from any MCP server, you must connect to that specific server:
1. Use ConnectMCPSessionTool with the server name
2. Use the MCP tools from that server
3. Use DisconnectMCPSessionTool when finished with that server

Available tools from all servers:
{chr(10).join(f'- {name}' for name in all_tool_names)}

Session Management Tools:
- ConnectMCPSessionTool: Connect to a specific MCP server
- DisconnectMCPSessionTool: Disconnect from current server
- GetMCPSessionStatusTool: Check current session status

You can switch between servers by disconnecting from one and connecting to another.
Always follow proper session management to prevent errors."""
    
    return {
        "status": "ok",
        "agent_name": "MCPMultiServerAgent",
        "system_prompt": system_prompt,
        "tool_classes": all_tool_classes,
        "tool_names": all_tool_names,
        "server_names": server_names,
        "total_tools": len(all_tool_classes),
        "server_details": server_details
    }

def discover_popular_mcp_servers(limit: int = 10) -> List[str]:
    """
    Discover popular MCP servers from the Smithery registry.
    
    Args:
        limit: Maximum number of servers to return
        
    Returns:
        List of popular server names
    """
    print(f"Discovering popular MCP servers (limit: {limit})...")
    
    try:
        # Use server manager to get server recommendations
        query = f"What are the {limit} most popular and useful MCP servers available? List just the server names."
        result = use_mcp_via_server_manager(query)
        
        if result.get("status") == "ok":
            # Try to extract server names from the response
            response_text = result.get("result", "")
            
            # Simple extraction - look for @server/name patterns
            import re
            server_pattern = r'@[\w-]+/[\w-]+'
            servers = re.findall(server_pattern, response_text)
            
            if servers:
                print(f"Found {len(servers)} servers: {', '.join(servers[:limit])}")
                return servers[:limit]
        
        # Fallback to discovery API
        print("Falling back to discovery API...")
        discover_result = discover_servers(page_size=limit)
        
        if discover_result.get("status") == "ok":
            servers = discover_result.get("servers", [])
            print(f"Found {len(servers)} servers via discovery")
            return servers[:limit]
            
    except Exception as e:
        print(f"Error discovering servers: {e}")
    
    # Return known working servers as final fallback
    fallback_servers = [
        "@wonderwhy-er/desktop-commander",
        "@modelcontextprotocol/filesystem",
        "@modelcontextprotocol/postgres"
    ]
    print(f"Using fallback servers: {', '.join(fallback_servers[:limit])}")
    return fallback_servers[:limit]

def create_smart_mcp_agent(
    task_description: str,
    max_servers: int = 3,
    max_tools_per_server: int = 5
) -> Dict[str, Any]:
    """
    Create an MCP agent automatically based on a task description.
    
    This function uses the server manager to recommend appropriate MCP servers
    for the given task, then creates an agent with those capabilities.
    
    Args:
        task_description: Description of what the agent should be able to do
        max_servers: Maximum number of servers to include
        max_tools_per_server: Maximum tools per server
        
    Returns:
        Dict with smart agent configuration
    """
    print(f"=== Creating Smart MCP Agent ===")
    print(f"Task: {task_description}\n")
    
    # Step 1: Use server manager to recommend servers
    print("Step 1: Getting server recommendations...")
    query = f"""Given this task: "{task_description}"

What are the best MCP servers that would help accomplish this task? 
Please recommend up to {max_servers} servers and explain why each would be useful.
List the server names clearly."""
    
    recommendation_result = use_mcp_via_server_manager(query)
    
    recommended_servers = []
    if recommendation_result.get("status") == "ok":
        response = recommendation_result.get("result", "")
        print(f"Server recommendations:\n{response}\n")
        
        # Extract server names from response
        import re
        server_pattern = r'@[\w-]+/[\w-]+'
        recommended_servers = re.findall(server_pattern, response)[:max_servers]
    
    # Fallback if no servers found
    if not recommended_servers:
        print("No specific recommendations found, using popular servers...")
        recommended_servers = discover_popular_mcp_servers(max_servers)
    
    print(f"Selected servers: {', '.join(recommended_servers)}")
    
    # Step 2: Create multi-server agent
    print(f"\nStep 2: Creating agent with selected servers...")
    agent_result = create_multi_server_mcp_agent(
        recommended_servers,
        max_tools_per_server=max_tools_per_server
    )
    
    if agent_result.get("status") == "ok":
        # Customize system prompt for the specific task
        original_prompt = agent_result["system_prompt"]
        task_specific_prompt = f"""TASK FOCUS: {task_description}

{original_prompt}

TASK-SPECIFIC GUIDANCE:
You have been specifically configured to help with: {task_description}

The recommended servers for this task are:
{chr(10).join(f'- {server}' for server in recommended_servers)}

Focus on using the tools that are most relevant to accomplishing the specified task."""
        
        agent_result["system_prompt"] = task_specific_prompt
        agent_result["task_description"] = task_description
        agent_result["agent_name"] = "MCPSmartAgent"
    
    return agent_result

# Example usage and testing
if __name__ == "__main__":
    print("HEAVEN MCP Agent Orchestrator - Example Usage")
    print("=" * 60)
    
    # Example 1: Single server agent
    print("\n1. Creating agent for single MCP server...")
    single_result = discover_and_create_agent_for_mcp_server(
        "@wonderwhy-er/desktop-commander",
        max_tools=3
    )
    
    if single_result.get("status") == "ok":
        print(f"✓ Created {single_result['agent_config']['agent_name']}")
        print(f"  Tools: {single_result['created_tools']}")
    else:
        print(f"✗ Error: {single_result.get('error')}")
    
    # Example 2: Multi-server agent
    print("\n2. Creating multi-server agent...")
    multi_result = create_multi_server_mcp_agent(
        ["@wonderwhy-er/desktop-commander"],
        max_tools_per_server=2
    )
    
    if multi_result.get("status") == "ok":
        print(f"✓ Created {multi_result['agent_name']}")
        print(f"  Total tools: {multi_result['total_tools']}")
    else:
        print(f"✗ Error: {multi_result.get('error')}")
    
    # Example 3: Smart agent creation
    print("\n3. Creating smart agent for specific task...")
    smart_result = create_smart_mcp_agent(
        "Analyze and manipulate files on the local system",
        max_servers=2,
        max_tools_per_server=3
    )
    
    if smart_result.get("status") == "ok":
        print(f"✓ Created {smart_result['agent_name']}")
        print(f"  Task: {smart_result['task_description']}")
        print(f"  Servers: {len(smart_result['server_names'])}")
    else:
        print(f"✗ Error: {smart_result.get('error')}")