
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
        # Simplified tool info - just return basic info
        return f"""=== TOOL INFO FOR {tool_name}

This is a Heaven-Base tool. Use omnitool('{tool_name}', parameters={{...}}) to execute it.

For detailed usage, consult the tool's documentation or source code."""
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