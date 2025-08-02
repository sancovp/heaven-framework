"""
Decorators for HEAVEN framework
"""

from typing import Callable, Any
import inspect
from functools import wraps


def heaven_tool(name: str = None, description: str = None):
    """
    Decorator that converts a function into a BaseHeavenTool.
    
    Usage:
        @heaven_tool()
        def my_function(x: int, y: int) -> int:
            '''Add two numbers'''
            return x + y
            
        # Now my_function is a BaseHeavenTool class
        tool = my_function()
        result = await tool.arun(x=5, y=3)
    """
    def decorator(func: Callable) -> type:
        from .baseheaventool import BaseHeavenTool, ToolResult
        
        tool_name = name or f"{func.__name__.title()}Tool"
        tool_description = description or func.__doc__ or f"Execute {func.__name__} function"
        
        # Create the tool class dynamically
        class GeneratedTool(BaseHeavenTool):
            __name__ = tool_name
            __qualname__ = tool_name
            
            name = tool_name.lower().replace('tool', '')
            description = tool_description
            
            def __init__(self):
                super().__init__()
                self.func = func
            
            async def _arun(self, **kwargs) -> ToolResult:
                """Execute the wrapped function."""
                try:
                    # Handle both sync and async functions
                    if inspect.iscoroutinefunction(self.func):
                        result = await self.func(**kwargs)
                    else:
                        result = self.func(**kwargs)
                    return ToolResult(output=str(result))
                except Exception as e:
                    return ToolResult(error=f"Error executing {func.__name__}: {str(e)}")
        
        # Set the class name properly
        GeneratedTool.__name__ = tool_name
        GeneratedTool.__module__ = func.__module__
        
        return GeneratedTool
    
    return decorator


def make_heaven_tool_from_function(func: Callable, name: str = None, description: str = None) -> type:
    """
    Convert a function into a BaseHeavenTool class.
    
    Args:
        func: The function to convert
        name: Optional tool name (defaults to FunctionNameTool)
        description: Optional description (defaults to function docstring)
        
    Returns:
        A BaseHeavenTool class that wraps the function
        
    Example:
        def add(x: int, y: int) -> int:
            return x + y
            
        AddTool = make_heaven_tool_from_function(add)
        tool_instance = AddTool()
        result = await tool_instance.arun(x=5, y=3)
    """
    return heaven_tool(name=name, description=description)(func)
