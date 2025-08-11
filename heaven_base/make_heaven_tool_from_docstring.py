#!/usr/bin/env python3
"""
make_heaven_tool_from_docstring v2 - Proper dict vs object handling

Handles the distinction between:
- Dict[str, Any] → 'dict' type (additionalProperties: True)
- Pydantic models → 'object' type (strict schema with nested)
- Issues warnings for ambiguous dict usage
"""

import inspect
import re
import warnings
from dataclasses import MISSING
from typing import Any, Callable, Dict, Optional, Type, Union, List, get_type_hints, get_origin, get_args
from pydantic import BaseModel, Field
from langchain_core.tools import Tool

# Import HEAVEN components
from heaven_base.baseheaventool import BaseHeavenTool, ToolArgsSchema


def extract_docstring_info(func: Callable) -> Dict[str, Any]:
    """Extract function description and parameter descriptions from docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return {
            'description': f"Auto-generated tool for {func.__name__}",
            'param_descriptions': {}
        }
    
    lines = docstring.strip().split('\n')
    
    # Extract main description (everything before Args/Parameters section)
    description_lines = []
    param_descriptions = {}
    
    in_params = False
    current_param = None
    current_desc = []
    
    for line in lines:
        line = line.strip()
        
        # Check for parameter section start
        if line.lower() in ['args:', 'arguments:', 'parameters:', 'params:']:
            in_params = True
            continue
            
        # Check for other sections that end parameters
        if in_params and line.lower().endswith(':') and line.lower() not in ['args:', 'arguments:', 'parameters:', 'params:']:
            break
            
        if not in_params:
            description_lines.append(line)
        else:
            # Parse parameter descriptions
            param_match = re.match(r'^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+)', line)
            if param_match:
                if current_param:
                    param_descriptions[current_param] = ' '.join(current_desc).strip()
                current_param = param_match.group(1)
                current_desc = [param_match.group(2)]
            elif current_param and line:
                current_desc.append(line)
    
    # Add the last parameter
    if current_param:
        param_descriptions[current_param] = ' '.join(current_desc).strip()
    
    description = ' '.join(description_lines).strip()
    if not description:
        description = f"Auto-generated tool for {func.__name__}"
    
    return {
        'description': description,
        'param_descriptions': param_descriptions
    }


def analyze_python_type(param_type: Type, param_name: str, func_name: str) -> Dict[str, Any]:
    """
    Analyze Python type hint and return ToolArgsSchema type info.
    
    Returns dict with 'type', 'items' (for arrays), 'nested' (for objects), etc.
    """
    origin = get_origin(param_type)
    args = get_args(param_type)
    
    # Handle Union types (including Optional which is Union[T, None])
    if origin is Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # Optional[T] case
            result = analyze_python_type(non_none_types[0], param_name, func_name)
            result['nullable'] = True
            return result
        else:
            # Multiple non-None types - default to flexible dict with warning
            warnings.warn(f"Complex Union type for parameter '{param_name}' in {func_name}(). "
                         f"Defaulting to flexible dict. Consider using a Pydantic model for better type safety.")
            return {'type': 'dict'}
    
    # Handle List/Array types
    if origin is list or param_type is list:
        if args:
            item_type = args[0]
            item_info = analyze_python_type(item_type, f"{param_name}_item", func_name)
            return {
                'type': 'array',
                'items': item_info
            }
        else:
            # Untyped list
            return {
                'type': 'array',
                'items': {'type': 'string'}  # Default to string items
            }
    
    # Handle Dict types - THIS IS THE KEY DISTINCTION
    if origin is dict or param_type is dict:
        if args and len(args) == 2:
            key_type, value_type = args
            if key_type is str:
                if value_type is Any:
                    # Dict[str, Any] - flexible dict
                    warnings.warn(
                        f"Parameter '{param_name}' in {func_name}() uses Dict[str, Any]. "
                        f"This creates a flexible dict that accepts any key-value pairs. "
                        f"If this parameter has a specific structure, consider using a Pydantic model "
                        f"so LARGE CHAIN can convert it to a proper JSON object with validation."
                    )
                    return {'type': 'dict'}
                else:
                    # Dict[str, SomeType] - typed dict
                    value_info = analyze_python_type(value_type, f"{param_name}_value", func_name)
                    return {
                        'type': 'dict',
                        'additionalProperties': value_info
                    }
            else:
                # Dict[NonStr, T] - unusual, default to flexible dict
                warnings.warn(f"Non-string key type in Dict for parameter '{param_name}' in {func_name}(). "
                             f"Defaulting to flexible dict.")
                return {'type': 'dict'}
        else:
            # Untyped dict
            warnings.warn(
                f"Parameter '{param_name}' in {func_name}() uses untyped dict. "
                f"This creates a flexible dict. If this parameter has a specific structure, "
                f"use Dict[str, Type] or a Pydantic model for better validation."
            )
            return {'type': 'dict'}
    
    # Handle Pydantic models - structured objects
    if isinstance(param_type, type) and issubclass(param_type, BaseModel):
        return {
            'type': 'object',
            'nested': convert_pydantic_to_nested(param_type, param_name, func_name)
        }
    
    # Handle dataclasses
    if hasattr(param_type, '__dataclass_fields__'):
        return {
            'type': 'object', 
            'nested': convert_dataclass_to_nested(param_type, param_name, func_name)
        }
    
    # Handle basic types
    basic_type_mapping = {
        str: 'string',
        int: 'integer',
        float: 'number', 
        bool: 'boolean',
        Any: 'string'  # Default Any to string
    }
    
    if param_type in basic_type_mapping:
        return {'type': basic_type_mapping[param_type]}
    
    # Unknown type - default to string with warning
    warnings.warn(f"Unknown type '{param_type}' for parameter '{param_name}' in {func_name}(). "
                 f"Defaulting to string type.")
    return {'type': 'string'}


def convert_pydantic_to_nested(model_class: Type[BaseModel], param_name: str, func_name: str) -> Dict[str, Dict]:
    """Convert Pydantic model to ToolArgsSchema nested structure."""
    nested = {}
    
    # Get model fields
    if hasattr(model_class, 'model_fields'):
        # Pydantic v2
        fields = model_class.model_fields
        for field_name, field_info in fields.items():
            field_type = field_info.annotation
            is_required = field_info.is_required() if hasattr(field_info, 'is_required') else True
            description = field_info.description or f"{field_name} field"
            
            field_analysis = analyze_python_type(field_type, f"{param_name}.{field_name}", func_name)
            
            nested[field_name] = {
                field_name: {
                    **field_analysis,
                    'description': description,
                    'required': is_required
                }
            }
    else:
        # Pydantic v1 or other
        warnings.warn(f"Could not extract fields from Pydantic model {model_class} for parameter '{param_name}' in {func_name}(). "
                     f"Using flexible object type.")
        return {}
    
    return nested


def convert_dataclass_to_nested(dataclass_type: Type, param_name: str, func_name: str) -> Dict[str, Dict]:
    """Convert dataclass to ToolArgsSchema nested structure."""
    nested = {}
    
    for field_name, field in dataclass_type.__dataclass_fields__.items():
        field_type = field.type
        is_required = field.default == MISSING and field.default_factory == MISSING
        description = f"{field_name} field"
        
        field_analysis = analyze_python_type(field_type, f"{param_name}.{field_name}", func_name)
        
        nested[field_name] = {
            field_name: {
                **field_analysis,
                'description': description,
                'required': is_required
            }
        }
    
    return nested


def make_heaven_tool_from_docstring(func: Callable, tool_name: Optional[str] = None) -> Type[BaseHeavenTool]:
    """
    Auto-generate a HEAVEN tool from a function's signature and docstring.
    
    Properly handles dict vs object types:
    - Dict[str, Any] → 'dict' type (flexible, additionalProperties: True)  
    - Pydantic models → 'object' type (strict schema with nested structure)
    - Issues warnings for ambiguous dict usage
    """
    
    # Get function metadata
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring_info = extract_docstring_info(func)
    
    # Generate tool name
    if not tool_name:
        tool_name = f"{func.__name__.title().replace('_', '')}Tool"
    
    # Build ToolArgsSchema arguments dict
    arguments = {}
    
    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
            
        # Get type from type hints or annotation
        param_type = type_hints.get(param_name, param.annotation)
        if param_type == inspect.Parameter.empty:
            param_type = str  # Default to string
            
        # Check if parameter is required (no default value)
        is_required = param.default == inspect.Parameter.empty
        
        # Get description from docstring
        description = docstring_info['param_descriptions'].get(
            param_name, 
            f"Parameter {param_name} for {func.__name__}"
        )
        
        # Analyze the type and get ToolArgsSchema format
        type_analysis = analyze_python_type(param_type, param_name, func.__name__)
        
        # Build argument definition
        arg_def = {
            'name': param_name,
            'description': description,
            'required': is_required,
            **type_analysis  # Include type, nested, items, etc.
        }
        
        # Add default value if present
        if not is_required:
            arg_def['default'] = param.default
            
        arguments[param_name] = arg_def
    
    # Create ToolArgsSchema class
    class AutoToolArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = Field(
            default_factory=lambda: arguments,
            description=f"Auto-generated arguments for {tool_name}"
        )
    
    # Create method to instantiate the tool
    def create_tool(adk: bool = False):
        """Create tool instance compatible with HEAVEN patterns"""
        # Create LangChain Tool
        schema_instance = AutoToolArgsSchema()
        pydantic_schema = AutoToolArgsSchema.to_pydantic_schema(schema_instance.arguments)
        
        langchain_tool = Tool(
            name=tool_name,
            description=docstring_info['description'],
            func=func,
            args_schema=pydantic_schema
        )
        
        # Determine if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        # Create BaseHeavenTool instance
        heaven_tool = AutoHeavenTool(
            base_tool=langchain_tool,
            args_schema=AutoToolArgsSchema,
            is_async=is_async
        )
        
        return heaven_tool
    
    # Create BaseHeavenTool subclass using type()
    AutoHeavenTool = type(
        tool_name.replace('Tool', '') + 'HeavenTool',
        (BaseHeavenTool,),
        {
            'name': tool_name,
            'description': docstring_info['description'],
            'func': func,
            'args_schema': AutoToolArgsSchema,
            'create': classmethod(lambda cls, adk=False: create_tool(adk))
        }
    )
    
    return AutoHeavenTool


# Test with complex types
def test_complex_types():
    """Test the improved type handling with warnings"""
    
    from pydantic import BaseModel
    from dataclasses import dataclass
    from typing import List
    
    class UserProfile(BaseModel):
        name: str
        email: str
        age: Optional[int] = None
    
    @dataclass
    class Settings:
        theme: str
        debug: bool = False
    
    def complex_function(
        user: UserProfile,
        metadata: Dict[str, Any],
        flags: List[str],
        config: Settings,
        extra_data: dict  # This should trigger warning
    ) -> bool:
        """
        A complex function with mixed types.
        
        Args:
            user: User profile information
            metadata: Flexible metadata dictionary (can contain any keys)
            flags: List of feature flags to enable
            config: Application configuration settings
            extra_data: Additional data (unstructured)
        """
        return True
    
    print("=== Testing complex type handling ===")
    
    # This should generate warnings for dict usage
    tool_class = make_heaven_tool_from_docstring(complex_function)
    tool_instance = tool_class.create()
    
    print(f"Tool name: {tool_instance.name}")
    spec = tool_instance.get_spec()
    
    print("\nGenerated schema:")
    for arg_name, arg_spec in spec['args'].items():
        print(f"  {arg_name}: {arg_spec}")
    
    print("\n✅ Complex type test completed!")


if __name__ == "__main__":
    test_complex_types()