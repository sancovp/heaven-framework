#!/usr/bin/env python3
"""
HEAVEN OmniTool Universal Interface Example
Shows how to use OmniTool to call any registered tool in HEAVEN
"""

import asyncio
import os
os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

from heaven_base import (
    BaseHeavenAgent,
    HeavenAgentConfig,
    UnifiedChat,
    ProviderEnum
)
from heaven_base.memory.history import History
from heaven_base.utils.omnitool import omnitool

async def main():
    print("=== HEAVEN OmniTool Universal Interface Example ===\n")
    
    # First, let's discover what tools are available
    print("=== Step 1: Discovering Available Tools ===")
    tools_result = await omnitool(list_tools=True)
    print(f"Available tools discovery result: {str(tools_result)[:100]}...")
    
    # Get detailed info about specific tools
    print("\n=== Step 2: Getting Tool Information ===")
    
    # Check BashTool
    bash_info = await omnitool('bash_tool', get_tool_info=True)
    print(f"BashTool info: {str(bash_info)[:200]}...")
    
    # Check NetworkEditTool  
    network_info = await omnitool('NetworkEditTool', get_tool_info=True)
    print(f"NetworkEditTool info: {str(network_info)[:200]}...")
    
    # Demonstrate OmniTool usage patterns
    print("\n=== Step 3: Using OmniTool for Various Operations ===")
    
    # 1. BashTool - Execute system commands
    print("\n--- Using BashTool via OmniTool ---")
    bash_result = await omnitool('bash_tool', parameters={
        'command': 'echo "Hello from OmniTool!" && date'
    })
    print(f"Bash result: {str(bash_result)[:200]}...")
    
    # 2. NetworkEditTool - File operations
    print("\n--- Using NetworkEditTool via OmniTool ---")
    
    # Create a test file
    create_result = await omnitool('NetworkEditTool', parameters={
        'command': 'create',
        'path': '/tmp/omnitool_test.txt',
        'file_text': 'This file was created using OmniTool!\nOmniTool provides universal access to all HEAVEN tools.',
        'command_arguments': {}
    })
    print(f"File creation result: {str(create_result)[:100]}...")
    
    # Read the file back
    read_result = await omnitool('NetworkEditTool', parameters={
        'command': 'view',
        'path': '/tmp/omnitool_test.txt',
        'command_arguments': {}
    })
    print(f"File read result: {str(read_result)[:150]}...")
    
    # 3. RegistryTool - Registry operations
    print("\n--- Using RegistryTool via OmniTool ---")
    
    # List available registries
    registry_list = await omnitool('RegistryTool', parameters={
        'operation': 'list_registries'
    })
    print(f"Available registries: {str(registry_list)[:100]}...")
    
    # 4. ThinkTool - AI reasoning
    print("\n--- Using ThinkTool via OmniTool ---")
    think_result = await omnitool('think_tool', parameters={
        'thoughts': 'OmniTool is a powerful abstraction that allows universal access to any HEAVEN tool',
        'conclusion': 'This makes agent development much more flexible'
    })
    print(f"Think result: {str(think_result)[:150]}...")
    
    # 5. WebSearchTool - Web search capabilities
    print("\n--- Using WebSearchTool via OmniTool ---")
    try:
        search_result = await omnitool('websearch_tool', parameters={
            'query': 'HEAVEN framework AI agents',
            'num_results': 3
        })
        print(f"Web search result: {str(search_result)[:150]}...")
    except Exception as e:
        print(f"Web search note: {e}")
    
    # Now demonstrate using OmniTool within an agent
    print("\n=== Step 4: Agent Using OmniTool ===")
    
    # Create a custom tool wrapper for agents to use omnitool
    def omnitool_wrapper(tool_name: str, parameters: dict = None, get_info: bool = False) -> str:
        """
        Wrapper to use omnitool within HEAVEN agents.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Parameters to pass to the tool
            get_info: Whether to get tool info instead of calling
            
        Returns:
            Tool execution result or tool information
        """
        if get_info:
            return omnitool(tool_name, get_tool_info=True)
        elif parameters:
            return omnitool(tool_name, parameters=parameters)
        else:
            return omnitool(list_tools=True)
    
    # Create tool from our wrapper
    from heaven_base import make_heaven_tool_from_docstring
    OmniToolWrapper = make_heaven_tool_from_docstring(omnitool_wrapper)
    
    # Create agent with OmniTool access
    config = HeavenAgentConfig(
        name="OmniToolAgent",
        system_prompt="""You are an agent with access to OmniTool, which gives you universal access to all HEAVEN tools.

OmniTool can:
- List all available tools (set get_info=false, parameters=None)
- Get information about specific tools (set get_info=true)
- Execute any tool with parameters (set get_info=false, provide parameters)

Key tools you can access via OmniTool:
- bash_tool: Execute system commands
- NetworkEditTool: File operations (create, view, edit files)
- RegistryTool: Registry operations
- think_tool: AI reasoning and analysis
- websearch_tool: Web search capabilities

Always use OmniTool when users ask for operations that require these capabilities.""",
        tools=[OmniToolWrapper],
        provider=ProviderEnum.OPENAI,
        model="o4-mini",
        temperature=0.3
    )
    
    # Initialize components
    history = History(messages=[])
    
    # Create the agent
    agent = BaseHeavenAgent(config, UnifiedChat, history=history)
    
    # Test the agent using OmniTool
    prompt = """Create a small text file at /tmp/agent_omnitool_test.txt with some information about OmniTool, 
then use a system command to check if the file was created successfully."""
    
    print(f"User: {prompt}\n")
    
    result = await agent.run(prompt=prompt)
    
    # Display the response
    if isinstance(result, dict) and "history" in result:
        for msg in result["history"].messages:
            if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                print(f"Assistant: {msg.content}")
    
    print(f"\n=== OmniTool Benefits ===")
    print("✓ Universal access to any HEAVEN tool")
    print("✓ Dynamic tool discovery and usage")
    print("✓ Consistent interface across all tools")
    print("✓ No need to import individual tool classes")
    print("✓ Perfect for meta-programming and tool orchestration")
    
    # Show the history ID
    history_id = result.get("history_id")
    print(f"\nHistory ID: {history_id}")

if __name__ == "__main__":
    asyncio.run(main())