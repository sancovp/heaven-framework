#!/usr/bin/env python3
"""
MCP-to-HEAVEN Tool Converter

This module provides functions to convert MCP server tools into HEAVEN tools.
It handles schema conversion, wrapper function generation, and integration
with HEAVEN's tool creation system.

The conversion process:
1. Connects to MCP server to get tool schemas
2. Converts MCP schemas to HEAVEN format
3. Generates wrapper functions that call MCP tools
4. Creates HEAVEN tools using make_heaven_tool_from_docstring
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from .mcp_client import (
    start_session, close_session, get_tool_args, 
    get_session_tools, use_mcp_via_session_manager
)

# Type mapping from MCP to Python types
TYPE_MAPPING = {
    "string": "str",
    "integer": "int", 
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict"
}

#
# SCHEMA CONVERSION FUNCTIONS
#

def convert_mcp_schema_to_python_args(mcp_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MCP tool argument schema to Python function arguments.
    
    Args:
        mcp_args: The MCP tool arguments schema
        
    Returns:
        Dict with Python argument information
    """
    python_args = {}
    
    for arg_name, arg_spec in mcp_args.items():
        python_arg = {
            'name': arg_name,
            'type': TYPE_MAPPING.get(arg_spec.get('type', 'string'), 'str'),
            'description': arg_spec.get('description', ''),
            'required': arg_spec.get('required', False),
            'default': arg_spec.get('default')
        }
        
        python_args[arg_name] = python_arg
    
    return python_args

def generate_function_signature(tool_name: str, python_args: Dict[str, Any]) -> str:
    """
    Generate Python function signature from argument schema.
    
    Args:
        tool_name: Name of the tool
        python_args: Python argument information
        
    Returns:
        String containing the function signature
    """
    # Create function name
    func_name = f"mcp_{tool_name.replace('-', '_')}"
    
    # Build parameter list
    params = []
    for arg_name, arg_info in python_args.items():
        param_type = arg_info['type']
        param = f"{arg_name}: {param_type}"
        
        # Add default value if not required
        if not arg_info['required'] and arg_info.get('default') is not None:
            default_val = repr(arg_info['default'])
            param += f" = {default_val}"
        elif not arg_info['required']:
            param += " = None"
            
        params.append(param)
    
    return f"def {func_name}({', '.join(params)})"

def generate_mcp_wrapper_function(
    server_name: str, 
    tool_name: str, 
    tool_description: str,
    python_args: Dict[str, Any]
) -> str:
    """
    Generate a complete Python function that wraps an MCP tool.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to wrap
        tool_description: Description of the tool
        python_args: Python argument information
        
    Returns:
        String containing the complete Python function code
    """
    # Generate function signature
    signature = generate_function_signature(tool_name, python_args)
    
    # Generate function name
    func_name = f"mcp_{tool_name.replace('-', '_')}"
    
    # Build docstring
    docstring_lines = [f'    """', f'    {tool_description}', '']
    
    if python_args:
        docstring_lines.append('    Args:')
        for arg_name, arg_info in python_args.items():
            arg_desc = arg_info['description'] or 'No description provided'
            arg_type = arg_info['type']
            required_text = '' if arg_info['required'] else ' (optional)'
            docstring_lines.append(f"        {arg_name} ({arg_type}): {arg_desc}{required_text}")
    
    docstring_lines.extend(['', '    Returns:', '        dict: Tool execution result', '    """'])
    
    # Build function body
    function_body = f'''
{signature}:
{chr(10).join(docstring_lines)}
    from heaven_base.utils.mcp_client import use_mcp_via_session_manager
    
    # Build tool arguments
    tool_args = {{}}'''
    
    for arg_name, arg_info in python_args.items():
        if arg_info['required']:
            function_body += f'''
    tool_args["{arg_name}"] = {arg_name}'''
        else:
            function_body += f'''
    if {arg_name} is not None:
        tool_args["{arg_name}"] = {arg_name}'''
    
    function_body += f'''
    
    # Execute the MCP tool
    result = use_mcp_via_session_manager(
        server_name="{server_name}",
        tool_name="{tool_name}",
        tool_args=tool_args
    )
    
    if result.get("status") != "ok":
        error = result.get("error", "Unknown error")
        raise Exception(f"Error calling {tool_name}: {{error}}")
    
    return result.get("result")'''
    
    return function_body

#
# TOOL CREATION FUNCTIONS
#

def create_mcp_tool_info(server_name: str, tool_name: str) -> Dict[str, Any]:
    """
    Create tool information for an MCP tool.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to wrap
        
    Returns:
        Dict with information about the tool for HEAVEN tool creation
    """
    # Start a session with the server
    start_result = start_session(server_name)
    if start_result.get("status") != "ok":
        return {
            "status": "error",
            "error": f"Failed to start session with {server_name}: {start_result.get('error', 'Unknown error')}"
        }
    
    try:
        # Get tool arguments schema
        args_result = get_tool_args(tool_name)
        if args_result.get("status") != "ok":
            return {
                "status": "error", 
                "error": f"Failed to get args for {tool_name}: {args_result.get('error', 'Unknown error')}"
            }
        
        # Get the tool description
        tool_description = args_result.get("description", f"MCP tool {tool_name} from server {server_name}")
        
        # Convert MCP schema to Python args
        python_args = convert_mcp_schema_to_python_args(args_result.get("args", {}))
        
        # Generate wrapper function
        wrapper_function = generate_mcp_wrapper_function(server_name, tool_name, tool_description, python_args)
        
        # Create function name
        func_name = f"mcp_{tool_name.replace('-', '_')}"
        
        return {
            "status": "ok",
            "server_name": server_name,
            "mcp_tool_name": tool_name, 
            "tool_description": tool_description,
            "python_args": python_args,
            "wrapper_function": wrapper_function,
            "function_name": func_name
        }
    finally:
        # Always close the session
        close_session()

def create_heaven_tool_from_mcp_tool(server_name: str, tool_name: str) -> Dict[str, Any]:
    """
    Create a HEAVEN tool from an MCP tool.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to wrap
        
    Returns:
        Dict with the created HEAVEN tool class
    """
    # Get tool information
    tool_info = create_mcp_tool_info(server_name, tool_name)
    if tool_info.get("status") != "ok":
        return tool_info
    
    try:
        from heaven_base import make_heaven_tool_from_docstring
        
        # Create a local namespace and execute the wrapper function
        namespace = {}
        exec(tool_info["wrapper_function"], namespace)
        
        # Get the function object
        func = namespace[tool_info["function_name"]]
        
        # Create HEAVEN tool from the function
        tool_class = make_heaven_tool_from_docstring(func)
        
        # Create a descriptive tool name
        server_part = ''.join(word.capitalize() for word in server_name.replace('@', '').replace('/', ' ').replace('-', ' ').split())
        tool_part = ''.join(word.capitalize() for word in tool_name.replace('-', '_').split('_'))
        tool_name_clean = f"MCP{server_part}{tool_part}Tool"
        
        return {
            "status": "ok",
            "tool_class": tool_class,
            "tool_name": tool_name_clean,
            "server_name": server_name,
            "mcp_tool_name": tool_name,
            "description": tool_info["tool_description"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error creating HEAVEN tool: {str(e)}",
            "tool_info": tool_info
        }

def create_all_tools_for_mcp_server(server_name: str, max_tools: Optional[int] = None) -> Dict[str, Any]:
    """
    Create HEAVEN tools for all tools on an MCP server.
    
    Args:
        server_name: Name of the MCP server
        max_tools: Maximum number of tools to create (for testing)
        
    Returns:
        Dict with information about created tools
    """
    # Start a session with the server
    start_result = start_session(server_name)
    if start_result.get("status") != "ok":
        return {
            "status": "error",
            "error": f"Failed to start session with {server_name}: {start_result.get('error', 'Unknown error')}"
        }
    
    try:
        # Get all available tools on the server
        tools_result = get_session_tools()
        if tools_result.get("status") != "ok":
            return {
                "status": "error",
                "error": f"Failed to get tools from {server_name}: {tools_result.get('error', 'Unknown error')}"
            }
        
        available_tools = tools_result.get("tools", [])
        
        # Limit the number of tools if max_tools is specified
        if max_tools is not None and max_tools > 0:
            available_tools = available_tools[:max_tools]
        
        created_tools = []
        failed_tools = []
        
        # For each tool, create a HEAVEN wrapper
        for tool_name in available_tools:
            print(f"Processing MCP tool: {tool_name}")
            
            # Close the session first (we'll reopen it in create_heaven_tool_from_mcp_tool)
            close_session()
            
            # Create the HEAVEN tool
            tool_result = create_heaven_tool_from_mcp_tool(server_name, tool_name)
            
            if tool_result.get("status") == "ok":
                print(f"✓ Created HEAVEN tool: {tool_result['tool_name']}")
                created_tools.append(tool_result)
            else:
                error = tool_result.get('error', 'Unknown error')
                print(f"✗ Failed to create tool for {tool_name}: {error}")
                failed_tools.append({"tool_name": tool_name, "error": error})
        
        return {
            "status": "ok",
            "server_name": server_name,
            "tools_created": len(created_tools),
            "tools": created_tools,
            "failed_tools": failed_tools
        }
    finally:
        # Always close the session
        try:
            close_session()
        except:
            pass

#
# UTILITY FUNCTIONS
#

def get_mcp_server_tools(server_name: str) -> List[str]:
    """
    Get a list of all tools available on an MCP server.
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        List of tool names
    """
    # Start session
    start_result = start_session(server_name)
    if start_result.get("status") != "ok":
        return []
    
    try:
        # Get tools
        tools_result = get_session_tools()
        if tools_result.get("status") == "ok":
            return tools_result.get("tools", [])
        return []
    finally:
        close_session()

def test_mcp_tool_conversion(server_name: str, tool_name: str) -> Dict[str, Any]:
    """
    Test the conversion of a single MCP tool to HEAVEN tool.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to test
        
    Returns:
        Dict with test results
    """
    print(f"Testing MCP tool conversion: {server_name} / {tool_name}")
    
    # Step 1: Get tool info
    print("  1. Getting tool information...")
    tool_info = create_mcp_tool_info(server_name, tool_name)
    if tool_info.get("status") != "ok":
        return {
            "status": "error",
            "step": "tool_info",
            "error": tool_info.get("error")
        }
    print(f"     ✓ Tool description: {tool_info['tool_description'][:100]}...")
    
    # Step 2: Create HEAVEN tool
    print("  2. Creating HEAVEN tool...")
    heaven_tool = create_heaven_tool_from_mcp_tool(server_name, tool_name)
    if heaven_tool.get("status") != "ok":
        return {
            "status": "error", 
            "step": "heaven_tool",
            "error": heaven_tool.get("error"),
            "tool_info": tool_info
        }
    print(f"     ✓ Created tool: {heaven_tool['tool_name']}")
    
    return {
        "status": "ok",
        "tool_info": tool_info,
        "heaven_tool": heaven_tool,
        "message": f"Successfully converted {tool_name} to {heaven_tool['tool_name']}"
    }

# Example usage and testing
if __name__ == "__main__":
    print("HEAVEN MCP Tool Converter - Example Usage")
    print("=" * 50)
    
    # Test with a known server and tool
    test_server = "@wonderwhy-er/desktop-commander"
    test_tool = "read_file"
    
    print(f"\nTesting conversion of {test_tool} from {test_server}")
    result = test_mcp_tool_conversion(test_server, test_tool)
    
    if result.get("status") == "ok":
        print(f"✓ Success: {result['message']}")
        
        # Show tool details
        tool_info = result["tool_info"]
        print(f"\nTool Details:")
        print(f"  Description: {tool_info['tool_description']}")
        print(f"  Arguments: {len(tool_info['python_args'])}")
        for arg_name, arg_info in tool_info['python_args'].items():
            print(f"    - {arg_name} ({arg_info['type']}): {arg_info['description']}")
    else:
        print(f"✗ Failed at step {result.get('step')}: {result.get('error')}")