"""
MCP Tool Wrapper for HEAVEN Framework
Wraps LangChain MCP tools to work as BaseHeavenTools
"""

from typing import Dict, Any, Type, Optional
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from .baseheaventool import BaseHeavenTool, ToolResult, ToolArgsSchema


class MCPToolWrapper(BaseHeavenTool):
    """Wrapper to make MCP LangChain tools work as HEAVEN tools"""
    
    def __init__(self, mcp_tool: BaseTool):
        """Initialize wrapper with a LangChain MCP tool
        
        Args:
            mcp_tool: A StructuredTool or BaseTool instance from langchain_mcp_adapters
        """
        # Store the LangChain tool
        self.base_tool = mcp_tool
        
        # Extract basic info
        self.name = mcp_tool.name
        self.description = mcp_tool.description or ""
        self.is_async = True  # MCP tools are always async (they have coroutine)
        
        # Convert args_schema to HEAVEN format
        self.args_schema = self._convert_schema(mcp_tool)
        
        # Not a class-based tool, so no func attribute
        self.func = None
    
    def _convert_schema(self, mcp_tool: BaseTool) -> Type[ToolArgsSchema]:
        """Convert MCP tool schema to HEAVEN ToolArgsSchema format"""
        
        # Get the schema from the tool
        if hasattr(mcp_tool, 'args_schema') and mcp_tool.args_schema:
            if isinstance(mcp_tool.args_schema, dict):
                # Direct JSON schema
                json_schema = mcp_tool.args_schema
            else:
                # Pydantic model
                json_schema = mcp_tool.args_schema.model_json_schema()
        else:
            # Fallback to empty schema
            json_schema = {"properties": {}, "required": []}
        
        props = json_schema.get("properties", {})
        required = json_schema.get("required", [])
        
        # Convert to HEAVEN arguments format
        arguments = {}
        for name, prop_def in props.items():
            arg_type = prop_def.get('type', 'string')
            
            # Handle nested types
            if arg_type == 'object':
                arg_type = 'dict'
            elif arg_type == 'array':
                arg_type = 'list'
            
            arguments[name] = {
                'name': name,
                'type': arg_type,
                'description': prop_def.get('description', ''),
                'required': name in required
            }
            
            # Add default if present
            if 'default' in prop_def:
                arguments[name]['default'] = prop_def['default']
        
        # Create a dynamic ToolArgsSchema subclass
        # We need to capture arguments in a way that doesn't rely on the local variable
        from typing import ClassVar
        
        class MCPArgsSchema(ToolArgsSchema):
            pass
        
        # Set the arguments as a class attribute after class creation
        MCPArgsSchema.arguments = arguments
        
        return MCPArgsSchema
    
    @classmethod
    def create(cls, adk: bool = False):
        """This shouldn't be called for MCP tools"""
        raise NotImplementedError("MCPToolWrapper instances are created directly via __init__, not create()")
    
    def get_openai_function(self):
        """Get OpenAI function format for this tool"""
        # The StructuredTool already has the right schema
        if hasattr(self.base_tool, 'get_input_schema'):
            schema = self.base_tool.get_input_schema().model_json_schema()
        elif hasattr(self.base_tool, 'args_schema'):
            if isinstance(self.base_tool.args_schema, dict):
                schema = self.base_tool.args_schema
            else:
                schema = self.base_tool.args_schema.model_json_schema()
        else:
            schema = {"properties": {}, "required": []}
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema
            }
        }
    
    # The base _arun and _run methods from BaseHeavenTool will handle execution
    # They check for self.base_tool._arun which StructuredTool has