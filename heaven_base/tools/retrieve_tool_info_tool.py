
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..utils.agent_and_tool_lists import get_tool_modules
from typing import Any, Dict, Optional

def retrieve_tool_info_util(tool_name: Optional[str] = None, list_tools: Optional[bool] = False):
    
    if list_tools:
        available_tools_str = get_tool_modules()
        available_tools = [tool.strip() for tool in available_tools_str.split(",")]
        return {
            "success": True,
            "available_tools": available_tools,
            "count": len(available_tools),
            "message": f"Found {len(available_tools)} available tools that can be used by OmniTool."
        }
    elif tool_name:
        # Get actual tool information using the same pattern as omnitool
        try:
            import importlib
            
            # Get available tools (same as omnitool)
            available = get_tool_modules()
            
            # Normalize to class name (same logic as omnitool)
            if tool_name not in available:
                alt = ''.join(part.capitalize() for part in tool_name.split('_'))
                if alt in available:
                    tool_name = alt
                else:
                    return f"Tool '{tool_name}' not found among available tools: {available}"
            
            # Derive module name from class name (same as omnitool)
            module_name = ''.join(
                ('_' + c.lower() if c.isupper() else c) for c in tool_name
            ).lstrip('_')
            full_module = f'heaven_base.tools.{module_name}'
            mod = importlib.import_module(full_module)
            
            ToolClass = getattr(mod, tool_name)
            
            # Instantiate via LangChain wrapper (same as omnitool)
            tool = ToolClass.create(adk=False)
            
            # Get tool information
            description = getattr(ToolClass, 'description', 'No description available')
            is_async = getattr(ToolClass, 'is_async', False)
            
            # Get args schema from the instantiated tool
            args_schema = None
            if hasattr(tool, 'args_schema'):
                args_schema = tool.args_schema
            elif hasattr(ToolClass, 'args_schema') and ToolClass.args_schema:
                try:
                    schema_instance = ToolClass.args_schema()
                    if hasattr(schema_instance, 'arguments'):
                        args_schema = schema_instance.arguments
                except Exception as e:
                    args_schema = f"Error loading schema: {str(e)}"
            
            return {
                "tool_name": tool_name,
                "description": description,
                "args_schema": args_schema,
                "is_async": is_async,
                "usage": f"Use omnitool('{tool_name}', parameters={{...}}) to execute this tool."
            }
            
        except Exception as e:
            return f"Error retrieving tool info for '{tool_name}': {str(e)}"
    else:
        return "ERROR: Either tool_name or list_tools=True is required!"
    

class RetrieveToolInfoToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        "tool_name": {
            "name": "tool_name",
            "type": "str",
            "description": "snake_case or PascalCase Name of the tool to invoke (ie code_localizer_tool, network_edit_tool, bash_tool, etc. or BashTool, etc.)",
            "required": False
        },
        "list_tools": {
            "name": "list_tools",
            "type": "bool",
            "description": "lists all registered tools with tool info that can be retrieved by RetrieveToolInfoTool",
            "required": False
        }
    }

class RetrieveToolInfoTool(BaseHeavenTool):
    name = "RetrieveToolInfoTool"
    description = "Retrieve the description and args schema for any tool registered in `...base/tools/` or list_tools to list registered tools."
    func = retrieve_tool_info_util
    args_schema = RetrieveToolInfoToolArgsSchema
    is_async = False