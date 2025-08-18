#!/usr/bin/env python3
"""
HEAVEN MCP Integration Example

This example demonstrates the complete MCP (Model Context Protocol) integration
in HEAVEN framework. MCP provides access to 5000+ servers from the Smithery
registry, each containing multiple tools for various tasks.

Key MCP capabilities:
1. Server discovery and registration
2. Automatic tool conversion from MCP to HEAVEN format
3. Session management for proper MCP connections
4. Agent creation with MCP tools
5. Multi-server agent orchestration
6. Smart agent creation based on task descriptions

Prerequisites:
- MCP-Use sidecar running at http://host.docker.internal:9000
- HEAVEN_DATA_DIR environment variable set
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    # Core HEAVEN classes
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum,
    
    # MCP Client functions
    list_servers, discover_servers, discover_and_register,
    use_mcp_via_server_manager, use_mcp_via_session_manager,
    
    # MCP Tool Converter
    create_heaven_tool_from_mcp_tool, get_mcp_server_tools,
    test_mcp_tool_conversion,
    
    # MCP Agent Orchestrator
    discover_and_create_agent_for_mcp_server, create_smart_mcp_agent,
    create_multi_server_mcp_agent,
    
    # MCP Session Management
    ConnectMCPSessionTool, DisconnectMCPSessionTool, GetMCPSessionStatusTool
)
from heaven_base.memory.history import History

async def demonstrate_mcp_discovery():
    """Demonstrate MCP server discovery and registration."""
    print("=== MCP Server Discovery ===\n")
    
    # 1. List currently registered servers
    print("1. Checking registered servers...")
    registered_servers = list_servers()
    
    if registered_servers.get("status") == "ok":
        servers = registered_servers.get("servers", [])
        print(f"   Found {len(servers)} registered servers")
        for server in servers[:3]:  # Show first 3
            print(f"   - {server}")
        if len(servers) > 3:
            print(f"   ... and {len(servers) - 3} more")
    else:
        print(f"   Error: {registered_servers.get('error')}")
    
    # 2. Discover servers from Smithery
    print("\n2. Discovering servers from Smithery registry...")
    discovery_result = discover_servers(page_size=5)
    
    if discovery_result.get("status") == "ok":
        discovered = discovery_result.get("servers", [])
        print(f"   Found {len(discovered)} servers in Smithery")
        for server in discovered:
            print(f"   - {server}")
    else:
        print(f"   Error: {discovery_result.get('error')}")
    
    # 3. Auto-register a server for testing
    test_server = "@wonderwhy-er/desktop-commander"
    print(f"\n3. Auto-registering test server: {test_server}")
    
    register_result = discover_and_register(test_server)
    if register_result.get("status") == "ok":
        print(f"   ✓ Successfully registered {test_server}")
    else:
        print(f"   Note: {register_result.get('error', 'Already registered or unavailable')}")
    
    return test_server

async def demonstrate_mcp_server_manager():
    """Demonstrate using the MCP server manager (agentic approach)."""
    print("\n=== MCP Server Manager (Agentic Approach) ===\n")
    
    # Use server manager to get information about available servers
    query = "What MCP servers do you have access to? List them and describe what each one does."
    
    print(f"Query: {query}")
    print("Sending to server manager...")
    
    result = use_mcp_via_server_manager(query)
    
    if result.get("status") == "ok":
        response = result.get("result", "")
        print(f"\nServer Manager Response:")
        print(f"{response}")
    else:
        print(f"Error: {result.get('error')}")

async def demonstrate_mcp_session_manager():
    """Demonstrate using the MCP session manager (direct tool calls)."""
    print("\n=== MCP Session Manager (Direct Tool Calls) ===\n")
    
    server_name = "@wonderwhy-er/desktop-commander"
    
    # Get available tools on the server
    print(f"Getting tools for {server_name}...")
    tools = get_mcp_server_tools(server_name)
    
    if tools:
        print(f"Available tools: {', '.join(tools[:5])}")
        if len(tools) > 5:
            print(f"... and {len(tools) - 5} more")
        
        # Test a simple tool (if available)
        test_tool = "read_file" if "read_file" in tools else tools[0] if tools else None
        
        if test_tool:
            print(f"\nTesting tool: {test_tool}")
            
            # Create test arguments based on the tool
            if test_tool == "read_file":
                tool_args = {"path": "/etc/hostname"}
            elif test_tool == "list_directory":
                tool_args = {"path": "/tmp"}
            else:
                tool_args = {}
            
            result = use_mcp_via_session_manager(
                server_name=server_name,
                tool_name=test_tool,
                tool_args=tool_args
            )
            
            if result.get("status") == "ok":
                tool_result = result.get("result")
                print(f"Tool result: {str(tool_result)[:200]}...")
            else:
                print(f"Tool error: {result.get('error')}")
    else:
        print(f"No tools found for {server_name}")

async def demonstrate_mcp_tool_conversion():
    """Demonstrate converting MCP tools to HEAVEN tools."""
    print("\n=== MCP Tool Conversion ===\n")
    
    server_name = "@wonderwhy-er/desktop-commander"
    test_tool = "read_file"
    
    print(f"Converting MCP tool to HEAVEN tool...")
    print(f"Server: {server_name}")
    print(f"Tool: {test_tool}")
    
    # Test the conversion process
    conversion_result = test_mcp_tool_conversion(server_name, test_tool)
    
    if conversion_result.get("status") == "ok":
        tool_info = conversion_result["tool_info"]
        heaven_tool = conversion_result["heaven_tool"]
        
        print(f"✓ Successfully converted to: {heaven_tool['tool_name']}")
        print(f"  Description: {tool_info['tool_description']}")
        print(f"  Arguments: {len(tool_info['python_args'])}")
        
        for arg_name, arg_info in tool_info['python_args'].items():
            required = "required" if arg_info['required'] else "optional"
            print(f"    - {arg_name} ({arg_info['type']}, {required}): {arg_info['description']}")
    else:
        print(f"✗ Conversion failed: {conversion_result.get('error')}")

async def demonstrate_mcp_agent_creation():
    """Demonstrate creating HEAVEN agents with MCP tools."""
    print("\n=== MCP Agent Creation ===\n")
    
    server_name = "@wonderwhy-er/desktop-commander"
    
    print(f"Creating HEAVEN agent for MCP server: {server_name}")
    print("(Limiting to 3 tools for demo purposes)")
    
    # Create agent with automatic tool discovery and conversion
    agent_result = discover_and_create_agent_for_mcp_server(
        server_name=server_name,
        max_tools=3,  # Limit for demo
        auto_register=True
    )
    
    if agent_result.get("status") == "ok":
        config = agent_result["agent_config"]
        
        print(f"✓ Created agent: {config['agent_name']}")
        print(f"  Total tools: {config['tool_count']}")
        print(f"  MCP tools created: {agent_result['created_tools']}")
        print(f"  Failed tools: {agent_result['failed_tools']}")
        
        # Create and test the agent
        print(f"\nTesting the created agent...")
        
        heaven_config = HeavenAgentConfig(
            name=config['agent_name'],
            system_prompt=config['system_prompt'],
            tools=config['tool_classes'],
            provider=ProviderEnum.OPENAI,
            model="o4-mini",
            temperature=0.3
        )
        
        # Initialize components
        history = History(messages=[])
        
        # Create the agent
        agent = BaseHeavenAgent(heaven_config, UnifiedChat, history=history)
        
        # Test with a simple task
        test_prompt = f"""Check the current MCP session status. If not connected, connect to {server_name}, 
then list what tools are available. Finally, disconnect when done."""
        
        print(f"Test prompt: {test_prompt}")
        result = await agent.run(prompt=test_prompt)
        
        # Display the response
        if isinstance(result, dict) and "history" in result:
            for msg in result["history"].messages:
                if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                    print(f"\nAgent response: {msg.content}")
                    break
        
        return agent_result
    else:
        print(f"✗ Agent creation failed: {agent_result.get('error')}")
        return None

async def demonstrate_smart_mcp_agent():
    """Demonstrate creating a smart MCP agent based on task description."""
    print("\n=== Smart MCP Agent Creation ===\n")
    
    task_description = "Help me analyze and work with files on my computer"
    
    print(f"Task: {task_description}")
    print("Creating smart agent that will automatically select appropriate MCP servers...")
    
    smart_agent_result = create_smart_mcp_agent(
        task_description=task_description,
        max_servers=2,  # Limit for demo
        max_tools_per_server=3
    )
    
    if smart_agent_result.get("status") == "ok":
        print(f"✓ Created smart agent: {smart_agent_result['agent_name']}")
        print(f"  Selected servers: {', '.join(smart_agent_result['server_names'])}")
        print(f"  Total tools: {smart_agent_result['total_tools']}")
        
        # Show server breakdown
        for server, details in smart_agent_result['server_details'].items():
            if 'tool_count' in details:
                print(f"  - {server}: {details['tool_count']} tools")
            else:
                print(f"  - {server}: error - {details.get('error', 'unknown')}")
        
        # Create and test the smart agent
        print(f"\nTesting the smart agent...")
        
        heaven_config = HeavenAgentConfig(
            name=smart_agent_result['agent_name'],
            system_prompt=smart_agent_result['system_prompt'],
            tools=smart_agent_result['tool_classes'],
            provider=ProviderEnum.OPENAI,
            model="o4-mini",
            temperature=0.3
        )
        
        # Initialize components
        history = History(messages=[])
        
        # Create the agent
        agent = BaseHeavenAgent(heaven_config, UnifiedChat, history=history)
        
        # Test with task-specific prompt
        test_prompt = "I need to check what files are in my /tmp directory. Please help me with this."
        
        print(f"Test prompt: {test_prompt}")
        result = await agent.run(prompt=test_prompt)
        
        # Display the response
        if isinstance(result, dict) and "history" in result:
            for msg in result["history"].messages:
                if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                    print(f"\nSmart Agent response: {msg.content}")
                    break
        
        return smart_agent_result
    else:
        print(f"✗ Smart agent creation failed: {smart_agent_result.get('error')}")
        return None

async def demonstrate_manual_session_management():
    """Demonstrate manual MCP session management tools."""
    print("\n=== Manual MCP Session Management ===\n")
    
    # Create an agent with just session management tools
    config = HeavenAgentConfig(
        name="MCPSessionManager",
        system_prompt="""You are an agent that demonstrates MCP session management.
        
You have tools to:
- ConnectMCPSessionTool: Connect to an MCP server
- DisconnectMCPSessionTool: Disconnect from the current server
- GetMCPSessionStatusTool: Check current session status

Demonstrate how to properly manage MCP sessions.""",
        tools=[ConnectMCPSessionTool, DisconnectMCPSessionTool, GetMCPSessionStatusTool],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Test session management workflow
    test_prompt = """Please demonstrate MCP session management by:
1. Checking the current session status
2. Connecting to the @wonderwhy-er/desktop-commander server
3. Checking the session status again
4. Disconnecting from the server
5. Checking the final status"""
    
    print(f"Test prompt: {test_prompt}")
    result = await agent.run(prompt=test_prompt)
    
    # Display the response
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"\nSession Manager response: {msg.content}")
                break

async def main():
    """Main demonstration of HEAVEN MCP integration."""
    print("=== HEAVEN MCP Integration Demonstration ===\n")
    print("This example shows how HEAVEN integrates with the Model Context Protocol")
    print("to provide access to 5000+ MCP servers from the Smithery registry.\n")
    
    try:
        # 1. Server discovery and registration
        test_server = await demonstrate_mcp_discovery()
        
        # 2. Server manager (agentic approach)
        await demonstrate_mcp_server_manager()
        
        # 3. Session manager (direct tool calls)
        await demonstrate_mcp_session_manager()
        
        # 4. Tool conversion
        await demonstrate_mcp_tool_conversion()
        
        # 5. Agent creation with MCP tools
        agent_result = await demonstrate_mcp_agent_creation()
        
        # 6. Smart agent creation
        smart_result = await demonstrate_smart_mcp_agent()
        
        # 7. Manual session management
        await demonstrate_manual_session_management()
        
        # Summary
        print("\n" + "=" * 60)
        print("=== MCP Integration Summary ===")
        print("✓ MCP server discovery and registration")
        print("✓ Server manager (agentic) queries")
        print("✓ Session manager (direct) tool calls")
        print("✓ Automatic MCP-to-HEAVEN tool conversion")
        print("✓ Agent creation with MCP tools")
        print("✓ Smart agent creation based on task descriptions")
        print("✓ Manual session management tools")
        
        print(f"\nMCP Integration provides access to 5000+ servers with multiple tools each.")
        print(f"This opens unlimited possibilities for agent capabilities!")
        
        # Show created tools/agents summary
        if agent_result:
            print(f"\nCreated agent: {agent_result['agent_config']['agent_name']}")
        if smart_result:
            print(f"Created smart agent: {smart_result['agent_name']}")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        print("This may indicate that the MCP-Use sidecar is not running.")
        print("Please ensure the MCP-Use service is available at http://host.docker.internal:9000")

if __name__ == "__main__":
    asyncio.run(main())