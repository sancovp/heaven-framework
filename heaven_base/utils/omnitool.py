"""
OmniTool: dynamically invoke any registered tool by name.
"""
import sys
import importlib
from heaven_base.utils.agent_and_tool_lists import get_tool_modules
from heaven_base.baseheaventool import BaseHeavenTool, ToolArgsSchema
from typing import Dict, Any, Optional
import asyncio, threading

__all__ = ['omnitool']


def _ensure_loop():

    try:

        asyncio.get_running_loop()

    except RuntimeError:

        loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)

        return loop

    return None   # already had one

async def omnitool(tool_name: Optional[str] = None, list_tools: Optional[bool] = False, get_tool_info: Optional[bool] = False, **kwargs):
    """
    Dynamically find, instantiate, and invoke a tool by class name or snake_case module name.
    """
    # DEFENSIVE PATTERN 1: Detect get_tool_info='list_tools' confusion
    if 'get_tool_info' in kwargs and kwargs['get_tool_info'] == 'list_tools':
        print("OMNITOOL DEFENSE: Auto-correcting get_tool_info='list_tools' to RetrieveToolInfoTool info")
        return await omnitool('RetrieveToolInfoTool', parameters={'tool_name': 'RetrieveToolInfoTool'})
    
    # DEFENSIVE PATTERN 2: Detect get_tool_info without tool_name  
    if 'get_tool_info' in kwargs and tool_name is None:
        print("OMNITOOL DEFENSE: get_tool_info requires tool_name - providing helpful error")
        return "ERROR: get_tool_info requires a tool_name parameter. Use list_tools=True to see available tools."
    
    if list_tools:
        return await omnitool('RetrieveToolInfoTool', parameters={'list_tools': True})

    if get_tool_info and tool_name:
        return await omnitool('RetrieveToolInfoTool', parameters={'tool_name': tool_name})
    # Discover available tool class names (PascalCase)
    available_str = get_tool_modules()
    available = [tool.strip() for tool in available_str.split(',')]

    # Normalize to class name
    if tool_name not in available:
        alt = ''.join(part.capitalize() for part in tool_name.split('_'))
        if alt in available:
            tool_name = alt
        else:
            raise ImportError(f'Tool "{tool_name}" not found among: {available}')

    
    
    # Derive module name from class name
    module_name = ''.join(
        ('_' + c.lower() if c.isupper() else c) for c in tool_name
    ).lstrip('_')
    full_module = f'heaven_base.tools.{module_name}'
    mod = importlib.import_module(full_module)

    ToolClass = getattr(mod, tool_name)

    # Instantiate via LangChain wrapper and call through executor
    tool = ToolClass.create(adk=False)

    # Handle sync vs async

    import asyncio

    if asyncio.iscoroutinefunction(tool._arun):
        # Tool function is async - use await
        if 'parameters' in kwargs:
            result = await tool._arun(**kwargs['parameters'])
            print(f"OMNITOOL DEBUG: async result = {result}, type = {type(result)}") 
        else:
            result = await tool._arun(**kwargs)
            print(f"OMNITOOL DEBUG: async result = {result}, type = {type(result)}")
    else:
        # Tool function is sync - call directly (works fine inside async function)
        if 'parameters' in kwargs:
            result = tool._arun(**kwargs['parameters'])
            print(f"OMNITOOL DEBUG: sync result = {result}, type = {type(result)}") 
        else:
            result = tool._arun(**kwargs)
            print(f"OMNITOOL DEBUG: sync result = {result}, type = {type(result)}")
    # # Handle sync vs. async
    # import asyncio
    # if asyncio.iscoroutinefunction(tool._arun):
    #     result = asyncio.get_event_loop().run_until_complete(tool._arun(**kwargs))
    # else:
    #     result = tool._arun(**kwargs)

    # Unwrap ToolResult if needed
    try:
        from heaven_base.baseheaventool import ToolResult
        if isinstance(result, ToolResult):
            for attr in ('result', 'output', 'value'):
                if hasattr(result, attr):
                    return getattr(result, attr)
            return result
    except ImportError:
        pass

    return result

class OmniToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        "tool_name": {
            "name": "tool_name",
            "type": "str",
            "description": "Name of the tool to invoke (PascalCase or snake_case)",
            "required": False
        },
        "list_tools": {
            "name": "list_tools",
            "type": "bool",
            "description": "Lists all registered tools callable by OmniTool",
            "required": False
        },
        "get_tool_info": {
            "name": "get_tool_info",
            "type": "bool",
            "description": "Optionally retrieve description and args for tool_name (get_tool_info needs the tool_name arg given with it)",
            "required": False
        },
        "parameters": {
            "name": "parameters",
            "type": "dict",
            "description": "Keyword arguments to pass to the target tool. Must be given if calling a tool",
            "required": False
        },
    }

class OmniTool(BaseHeavenTool):
    name = "OmniTool"
    description = """Dynamically invoke any tool in `...base/tools/...` by name with parameters dictionary; can also get_tool_info for tool_name; can also list_tools for all tools in the current HEAVEN build
    """
    func = omnitool
    args_schema = OmniToolArgsSchema
    is_async = False
  
# Simple test when run as script
def _test():
    # Test with test_tool
    result = omnitool('test_tool', message='hello via omnitool')
    print('Result from test_tool:', result)

if __name__ == '__main__':
    _test()
