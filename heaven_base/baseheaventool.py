# A metaprogramming wrapper for Langchain BaseTool and Tool construction
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Type, Callable, ClassVar, Literal, List, Union
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model, ConfigDict
try:
    from pydantic import Extra  # For Pydantic 2.10.6
except ImportError:
    Extra = None  # For Pydantic 2.11+
from langchain_core.tools import BaseTool, Tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from abc import abstractmethod, ABC
from langchain_core.utils.json_schema import dereference_refs
import importlib
from collections.abc import Mapping, Iterable


tool_log_path = "/tmp/tool_debug.log"

def schema_to_pydantic_model(model_name: str, schema: dict) -> type:
    """
    Convert a cleaned JSON schema back into a Pydantic model.
    NOTE: Supports only basic 'type', 'properties', 'required', 'description'.
    """
    fields = {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for field_name, field_schema in props.items():
        t = type_map.get(field_schema.get("type"), Any)
        desc = field_schema.get("description", "")
        is_required = field_name in required
        field_def = (t, Field(... if is_required else None, description=desc))
        fields[field_name] = field_def

    return create_model(model_name, **fields)

# These will need a Message class...

@dataclass(frozen=True)
class UserMessage:
    content: str
    user_id: Optional[str] = None
    timestamp: Optional[str] = None
    is_pseudo: Optional[bool] = False
  
@dataclass(frozen=True)
class AgentMessage:
    content: str
    agent_id: Optional[str] = None
    tool_call: Optional["ToolUse"] = None  # If tool triggered in content
    timestamp: Optional[str] = None

@dataclass(frozen=True)
class ToolUse:
    tool_name: str
    arguments: dict
    tool_call_id: Optional[str] = None
    agent_id: Optional[str] = None




if Extra is not None:
    # Pydantic 2.10.6 style
    class ForbidExtraConfig:
        extra = Extra.forbid
else:
    # Pydantic 2.11+ style - will be used as ConfigDict
    ForbidExtraConfig = ConfigDict(extra='forbid')
    

@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""
    output: Optional[str] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        return replace(self, **kwargs)

class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""

class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""

class ToolError(Exception):
    """Raised when a tool encounters an error."""
    def __init__(self, message):
        self.message = f"ERROR!!! {message}"
        super().__init__(message)


def fix_ref_paths(schema: dict) -> dict:
    """Fix $ref paths in schema by replacing #/$defs/ with #/defs/"""
    schema_copy = deepcopy(schema)

    def _fix_refs_recursive(obj):
        if isinstance(obj, dict):
            if "$ref" in obj and isinstance(obj["$ref"], str):
                obj["$ref"] = obj["$ref"].replace("/$defs/", "/defs/")
            for k, v in list(obj.items()):
                if isinstance(v, (dict, list)):
                    _fix_refs_recursive(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _fix_refs_recursive(item)

    _fix_refs_recursive(schema_copy)
    return schema_copy

def flatten_array_anyof(schema: dict) -> dict:
    """
    If the schema has an 'anyOf' that contains one branch with type "array"
    and another with type "null", flatten it to a single array schema with
    'nullable': true.
    """
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        array_branch = None
        null_branch = False
        for branch in schema["anyOf"]:
            if branch.get("type") == "array":
                array_branch = branch
            elif branch.get("type") == "null":
                null_branch = True
        if array_branch and null_branch:
            new_schema = dict(schema)
            new_schema.pop("anyOf")
            new_schema["type"] = "array"
            new_schema["items"] = array_branch.get("items", {})
            if "default" in schema:
                new_schema["default"] = schema["default"]
            new_schema["nullable"] = True
            if "description" in schema:
                new_schema["description"] = schema["description"]
            return new_schema
    return schema

def recursive_flatten(schema: Union[dict, list]) -> Union[dict, list]:
    if isinstance(schema, dict):
        new_schema = flatten_array_anyof(schema)
        for key, value in new_schema.items():
            if isinstance(value, dict) or isinstance(value, list):
                new_schema[key] = recursive_flatten(value)
        return new_schema
    elif isinstance(schema, list):
        return [recursive_flatten(item) if isinstance(item, dict) else item for item in schema]
    else:
        return schema

def fix_empty_object_properties(schema: Union[dict, list]) -> Union[dict, list]:
    """
    Recursively fixes any object-type schema that has an empty 'properties'
    dict by removing 'properties' and adding 'additionalProperties': True.
    """
    if isinstance(schema, dict):
        # Check if this is an object with empty properties.
        if schema.get("type") == "object":
            if "properties" in schema and not schema["properties"]:
                # Remove the empty properties and allow arbitrary keys.
                del schema["properties"]
                schema["additionalProperties"] = True
        # Recurse over dictionary values.
        new_schema = {}
        for key, value in schema.items():
            new_schema[key] = fix_empty_object_properties(value) if isinstance(value, (dict, list)) else value
        return new_schema
    elif isinstance(schema, list):
        return [fix_empty_object_properties(item) if isinstance(item, (dict, list)) else item for item in schema]
    return schema

def generate_dereferenced_schema(schema: Union[dict, Type[BaseModel]]) -> dict:
    """
    Returns a fully dereferenced (flattened) JSON schema.
    If a Pydantic model is passed, generate its JSON schema;
    if a dict is passed, assume it's already a JSON schema.
    Additionally, flatten array schemas that use an "anyOf" and fix empty
    object properties to support Gemini.
    """
    if isinstance(schema, dict):
        raw_schema = schema
    else:
        raw_schema = schema.model_json_schema(ref_template="#/defs/{model}")
    # ADDED FOR ADK COMPLIANCE
    # Fix $ref paths before renaming $defs to defs
    raw_schema = fix_ref_paths(raw_schema)
    ########
    if "$defs" in raw_schema:
        raw_schema["defs"] = raw_schema.pop("$defs")
    inlined = dereference_refs(raw_schema)
    inlined.pop("defs", None)
    # flattened = recursive_flatten(inlined)
    # fixed = fix_empty_object_properties(flattened)
    fixed = fix_empty_object_properties(inlined)
    return fixed



class ToolArgsSchema(BaseModel):
    """Meta-validator for tool arguments ensuring LangChain compatibility"""
    arguments: Dict[str, Dict[str, Any]] = Field(
        ..., description="Validated tool argument specifications"
    )
  
# V6
    @classmethod
    def custom_list_type(cls, item_type: Type, item_type_str: str) -> Type:
        mapping = {str: "string", int: "integer", float: "number", bool: "boolean"}
        json_item_type = mapping.get(item_type, "string")
        
        class CustomList(list):
            @classmethod
            def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
                # Use the default core schema for lists.
                return handler.generate_schema(list)
    
            @classmethod
            def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> Dict[str, Any]:
                # Get the default JSON schema from the core schema.
                json_schema = handler(core_schema)
                # Ensure that the "items" property exists.
                json_schema.setdefault("items", {})
                # Add the "type" to the items if it isn't already provided.
                if "type" not in json_schema["items"]:
                    json_schema["items"]["type"] = json_item_type
                # Add the custom description.
                json_schema["items"]["description"] = f"Item of type {item_type_str}"
                return json_schema
    
        return CustomList



    @classmethod
    def to_pydantic_schema(cls, arguments: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
        """
        Converts argument definitions into a dynamic Pydantic model.
        For list fields with primitive items, we use a custom list type that populates
        the "items" schema field properly.
        """
        schema_fields = {}
        type_mapping = {
            'int': int, 'str': str, 'float': float, 'bool': bool,
            'integer': int, 'string': str, 'number': float, 'boolean': bool,
            'list': list, 'array': list,  # We'll override list fields below.
            'dict': Dict[str, Any], 'object': Dict[str, Any],
        }

        for arg_name, arg_details in arguments.items():
            if not isinstance(arg_details, dict):
                continue

            arg_type_str = arg_details.get('type', 'string')
            description = arg_details.get('description', '')
            is_required = arg_details.get('required', True)
            default_value = arg_details.get('default', None)

            # Determine the field type.
            if arg_type_str in ('dict', 'object'):
                schema_field_type = cls._create_nested_model_recursive(f"Nested_{arg_name}", arg_details)
            elif arg_type_str in ('list', 'array'):
                item_info = arg_details.get('items', {})
                item_type_str = item_info.get('type', 'any')
                if item_type_str == 'any':
                    item_type_str = 'str'
                if item_type_str in ('dict', 'object'):
                    item_model = cls._create_nested_model_recursive(f"ListItem_{arg_name}", item_info)
                    schema_field_type = List[item_model]  # Use standard list of nested model.
                else:
                    primitive_type = type_mapping.get(item_type_str, str)
                    # Instead of List[primitive_type], use a custom list type.
                    schema_field_type = cls.custom_list_type(primitive_type, item_type_str)
            else:
                schema_field_type = type_mapping.get(arg_type_str, str)

            # Append default info into description.
            final_description = description
            if default_value is not None:
                final_description += f" (Defaults to {repr(default_value)})"
                arg_details.pop('default', None)

            field_kwargs = {"description": final_description}
            if not is_required:
                field_kwargs["default"] = None
                schema_field_type = Optional[schema_field_type]

            schema_fields[arg_name] = (schema_field_type, Field(**field_kwargs))

        model_name = f"DynamicArgsSchema_{id(arguments)}"
        
        # Handle both Pydantic 2.10.6 and 2.11+ syntax
        if Extra is not None:
            # Pydantic 2.10.6 - use __config__ with class
            return create_model(model_name, __config__=ForbidExtraConfig, **schema_fields)
        else:
            # Pydantic 2.11+ - use model_config with ConfigDict
            return create_model(model_name, model_config=ForbidExtraConfig, **schema_fields)
    
    @classmethod
    def _create_nested_model_recursive(cls, model_name: str, arg_definition: Dict[str, Any]) -> Type[BaseModel]:
        """
        Recursively creates a nested Pydantic model from the provided definition.
        Now treats 'nested' blocks as true sub-models instead of flattening them.
        """
        from pydantic import create_model, Field, BaseModel
        from typing import Dict, Any, Optional, List

        known_meta = {'name','type','description','required','default','items','additionalProperties','nested'}
        type_map = {
            'integer': int, 'int': int,
            'string': str, 'str': str,
            'number': float, 'float': float,
            'boolean': bool,'bool': bool,
            'list':  list,'array': list,
            'dict':  dict,'object': dict,
        }

        # 1) Handle any direct inline fields (outside of nested)
        schema_fields: Dict[str, Any] = {}
        for key, val in dict(arg_definition).items():
            if key in known_meta: 
                continue
            if isinstance(val, dict) and 'type' in val:
                # primitive or list/object without a nested sub‐block
                field_type = type_map.get(val['type'], str)
                if val['type'] in ('list','array'):
                    # list of primitives or dicts
                    items = val.get('items', {})
                    sub_type_str = items.get('type','string')
                    sub_py = type_map.get(sub_type_str, str)
                    # use your custom list type to keep items schema
                    field_type = cls.custom_list_type(sub_py, sub_type_str)
                schema_fields[key] = ( 
                    Optional[field_type] if not val.get('required',True) else field_type,
                    Field(
                        default=None if not val.get('required',True) else ...,
                        description=val.get('description','')
                    )
                )

        # 2) Now build sub‐models for each group in 'nested'
        nested = arg_definition.get('nested', {}) or {}
        for group_name, group_props in nested.items():
            sub_model = cls._create_nested_model_recursive(f"{model_name}_{group_name}", group_props)
            # include it as a required or optional field
            required = group_props.get('required', True)
            schema_fields[group_name] = (
                Optional[sub_model] if not required else sub_model,
                Field(
                    default=None if not required else ...,
                    description=group_props.get('description','')
                )
            )

        # 3) Create & return the Pydantic model for this level
        # Handle both Pydantic 2.10.6 and 2.11+ syntax
        if Extra is not None:
            # Pydantic 2.10.6 - use __config__ with class
            return create_model(
                model_name,
                __config__ = ForbidExtraConfig,
                **schema_fields
            )
        else:
            # Pydantic 2.11+ - use model_config with ConfigDict
            return create_model(
                model_name,
                model_config = ForbidExtraConfig,
                **schema_fields
            )
   
    @classmethod
    def validate_arguments(cls, arguments: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Validates the argument definitions, ensuring they have required metadata
        and supported types. Returns the validated arguments with defaults moved to descriptions.
        """
        known_meta_keys = {'name', 'type', 'description', 'required', 'default', 'items', 'additionalProperties', 'nested'}
        supported_types = ['int', 'str', 'float', 'bool', 'list', 'dict', 'integer', 'string', 'number', 'boolean', 'array', 'object']
        
        for arg_name, arg_details in arguments.items():
            if not isinstance(arg_details, dict):
                raise ValueError(f"ToolArgsSchema Error: Argument {arg_name} must be a dictionary")
                
            # Ensure required keys are present
            if not all(key in arg_details for key in ['name', 'type', 'description']):
                missing = [k for k in ['name', 'type', 'description'] if k not in arg_details]
                raise ValueError(f"ToolArgsSchema Error: Argument {arg_name} missing required metadata: {missing}")

            # Add default 'required=True' if missing
            if 'required' not in arg_details:
                arg_details['required'] = True
                
            # Validate that 'required' is a boolean
            if not isinstance(arg_details['required'], bool):
                raise ValueError(f"ToolArgsSchema Error: The 'required' field for {arg_name} must be a boolean")
                
            # Validate type is supported
            if arg_details['type'] not in supported_types:
                raise ValueError(f"ToolArgsSchema Error: Unsupported type for {arg_name}: {arg_details['type']}")
                
            # Move default to description if present
            if 'default' in arg_details:
                default_value = arg_details['default']
                arg_details['description'] += f" (Defaults to {repr(default_value)})"
                del arg_details['default']
                
            # Process any nested structure the same way
            if 'nested' in arg_details:
                # Validate the nested structure
                nested = arg_details['nested']
                if not isinstance(nested, dict):
                    raise ValueError(f"ToolArgsSchema Error: 'nested' in {arg_name} must be a dictionary")
                
                # Validate each nested item's arguments
                for item_key, item_args in nested.items():
                    if not isinstance(item_args, dict):
                        raise ValueError(f"ToolArgsSchema Error: Arguments for '{item_key}' in {arg_name} must be a dictionary")
                    
                    # Validate each argument
                    for sub_arg_name, sub_arg_def in item_args.items():
                        if not isinstance(sub_arg_def, dict):
                            raise ValueError(f"ToolArgsSchema Error: Definition for '{arg_name}.{item_key}.{sub_arg_name}' must be a dictionary")
                        
                        # Ensure required metadata is present
                        if 'type' not in sub_arg_def:
                            raise ValueError(f"ToolArgsSchema Error: '{arg_name}.{item_key}.{sub_arg_name}' missing required 'type'")
                        
                        if 'description' not in sub_arg_def:
                            raise ValueError(f"ToolArgsSchema Error: '{arg_name}.{item_key}.{sub_arg_name}' missing required 'description'")
                        
                        # Validate type is supported
                        if sub_arg_def['type'] not in supported_types:
                            raise ValueError(f"ToolArgsSchema Error: Unsupported type for '{arg_name}.{item_key}.{sub_arg_name}': {sub_arg_def['type']}")
                        
                        # Add default 'required=False' if missing for nested properties
                        if 'required' not in sub_arg_def:
                            sub_arg_def['required'] = False
                            
                        # Validate that 'required' is a boolean
                        if not isinstance(sub_arg_def['required'], bool):
                            raise ValueError(f"ToolArgsSchema Error: The 'required' field for '{arg_name}.{item_key}.{sub_arg_name}' must be a boolean")
                        
                        # Move default to description if present
                        if 'default' in sub_arg_def:
                            default_value = sub_arg_def['default']
                            sub_arg_def['description'] += f" (Defaults to {repr(default_value)})"
                            del sub_arg_def['default']
                        
                        # Recursively process any nested structure within this
                        if sub_arg_def['type'] in ['dict', 'object'] and 'nested' in sub_arg_def:
                            cls.validate_arguments({f"{arg_name}.{item_key}.{sub_arg_name}": sub_arg_def})
                        
                        # Handle nested validations for complex types
                        if sub_arg_def['type'] in ['list', 'array'] and 'items' in sub_arg_def:
                            items = sub_arg_def['items']
                            if not isinstance(items, dict):
                                raise ValueError(f"ToolArgsSchema Error: 'items' for '{arg_name}.{item_key}.{sub_arg_name}' must be a dictionary")
                                
                            if 'type' in items and items['type'] not in supported_types and items['type'] != 'any':
                                raise ValueError(f"ToolArgsSchema Error: Unsupported item type for '{arg_name}.{item_key}.{sub_arg_name}': {items['type']}")
                
            # Recursively validate nested structures (regular pattern)
            if arg_details['type'] in ['dict', 'object']:
                # Check for nested definitions and validate them
                for key, value in arg_details.items():
                    if key not in known_meta_keys and isinstance(value, dict) and 'type' in value:
                        # This is a nested property - validate it as well
                        cls._validate_nested_property(f"{arg_name}.{key}", value)
                        
            # Validate list item type if it's a list
            if arg_details['type'] in ['list', 'array']:
                if 'items' not in arg_details:
                    arg_details['items'] = {'type': 'any'}  # Default to Any if not specified
                elif not isinstance(arg_details['items'], dict):
                    raise ValueError(f"ToolArgsSchema Error: 'items' for {arg_name} must be a dictionary")
                elif 'type' not in arg_details['items']:
                    arg_details['items']['type'] = 'any'  # Default to Any if type not specified
                
                items = arg_details['items']
                item_type = items.get('type')
                
                if item_type not in supported_types and item_type != 'any':
                    raise ValueError(f"ToolArgsSchema Error: Unsupported item type for {arg_name}: {item_type}")
                
                if item_type in ['dict', 'object']:
                    # Add description to items if missing
                    if 'description' not in items:
                        items['description'] = f"Item of {arg_name} list"
                    
                    # Recursively validate nested item properties if they exist
                    for key, value in items.items():
                        if key not in known_meta_keys and isinstance(value, dict) and 'type' in value:
                            cls._validate_nested_property(f"{arg_name}[].{key}", value)
        
        return arguments

    @classmethod
    def _validate_nested_property(cls, path: str, prop_details: Dict[str, Any]):
        """
        Validates a nested property definition recursively.
        
        Args:
            path: The path to this property (for error messages)
            prop_details: The property details to validate
        """
        # Check required metadata
        if 'type' not in prop_details:
            raise ValueError(f"ToolArgsSchema Error: Nested property {path} missing required 'type'")
            
        # Check for description
        if 'description' not in prop_details:
            raise ValueError(f"ToolArgsSchema Error: Nested property {path} missing required 'description'")
            
        # Validate type is supported
        supported_types = ['int', 'str', 'float', 'bool', 'list', 'dict', 'integer', 'string', 'number', 'boolean', 'array', 'object']
        if prop_details['type'] not in supported_types:
            raise ValueError(f"ToolArgsSchema Error: Unsupported type for {path}: {prop_details['type']}")
            
        # Add default 'required=False' if missing for nested properties
        if 'required' not in prop_details:
            prop_details['required'] = False
        
        # Validate that 'required' is a boolean
        if not isinstance(prop_details['required'], bool):
            raise ValueError(f"ToolArgsSchema Error: The 'required' field for {path} must be a boolean")
            
        # Move default to description if present
        if 'default' in prop_details:
            default_value = prop_details['default']
            prop_details['description'] += f" (Defaults to {repr(default_value)})"
            del prop_details['default']
            
        # Known metadata keys to skip when looking for nested properties
        known_meta_keys = {'name', 'type', 'description', 'required', 'default', 'items', 'additionalProperties', 'nested'}
            
        # Handle nested structure within this property
        if 'nested' in prop_details and isinstance(prop_details['nested'], dict):
            nested = prop_details['nested']
            for item_key, item_args in nested.items():
                if isinstance(item_args, dict):
                    for sub_arg_name, sub_arg_def in item_args.items():
                        if isinstance(sub_arg_def, dict) and 'type' in sub_arg_def:
                            # Add directly to the parent object without prefixing
                            prop_details[sub_arg_name] = sub_arg_def
            # Once processed, remove the nested key to avoid double-processing
            del prop_details['nested']
            
        # If this is a dict/object, recursively validate its properties
        if prop_details['type'] in ['dict', 'object']:
            # Check for nested definitions and validate them
            for key, value in prop_details.items():
                if key not in known_meta_keys and isinstance(value, dict) and 'type' in value:
                    cls._validate_nested_property(f"{path}.{key}", value)
            
        # If this is a list/array, validate its items
        if prop_details['type'] in ['list', 'array']:
            if 'items' not in prop_details:
                prop_details['items'] = {'type': 'any'}
            elif not isinstance(prop_details['items'], dict):
                raise ValueError(f"ToolArgsSchema Error: 'items' for {path} must be a dictionary")
            elif 'type' not in prop_details['items']:
                prop_details['items']['type'] = 'any'
                
            items = prop_details['items']
            item_type = items.get('type')
            
            if item_type not in supported_types and item_type != 'any':
                raise ValueError(f"ToolArgsSchema Error: Unsupported item type for {path}: {item_type}")
                
            if item_type in ['dict', 'object']:
                # Add description to items if missing
                if 'description' not in items:
                    items['description'] = f"Item of {path}"
                
                # Recursively validate nested item properties if they exist
                for key, value in items.items():
                    if key not in known_meta_keys and isinstance(value, dict) and 'type' in value:
                        cls._validate_nested_property(f"{path}[].{key}", value)


  
class BaseHeavenTool(ABC):
    """Provider-agnostic tool base class with standardized results"""
    # Required class attributes that tools must define
    name: str
    description: str
    func: Callable
    args_schema: Type[ToolArgsSchema] 
    base_tool: BaseTool 
    is_async: bool 

    def __init__(
        self, 
        base_tool: BaseTool,
        args_schema: Type[ToolArgsSchema],
        is_async: bool = False
    ):
        # Set properties (no more super() call since we're not inheriting from BaseTool)
        self.args_schema = args_schema
        self.base_tool = base_tool
        self.is_async = is_async

        # Create instance and validate arguments schema
        schema_instance = args_schema()
        ToolArgsSchema.validate_arguments(schema_instance.arguments)
    
    def _clean_kwargs(self, obj: Any) -> Any:
        """
        Recursively return a *copy* of `obj` in which every mapping key that looks
        like "'some_key'" is replaced by "some_key".

        • Works for any Mapping subclass (dict, OrderedDict, defaultdict, …).
        • Keeps list / tuple / set semantics identical.
        • Primitives and values are left untouched.
        """
        # ---- Handle any mapping (dict-like) ---------------------------
        if isinstance(obj, Mapping):
            cleaned_items = {}
            for k, v in obj.items():
                new_k = k
                if (
                    isinstance(k, str)
                    and k.startswith("'")
                    and k.endswith("'")
                ):
                    stripped = k[1:-1]
                    if stripped != k:
                        # logging.warning(
                        #     "Sanitizing malformed tool-argument key: %r → %r",
                        #     k, stripped
                        # )
                        new_k = stripped

                cleaned_items[new_k] = self._clean_kwargs(v)  # recurse on value

            # Rebuild using the same mapping type when possible
            try:
                return type(obj)(cleaned_items)
            except Exception:
                return cleaned_items

        # ---- Handle iterables that can hold nested mappings -----------
        if isinstance(obj, (list, tuple, set)):
            cleaned_iter = [self._clean_kwargs(item) for item in obj]
            return type(obj)(cleaned_iter)

        # ---- Walk other iterables (e.g., generators) ------------------
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return type(obj)(self._clean_kwargs(item) for item in obj)

        # ---- Base case ------------------------------------------------
        return obj

        
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> ToolResult:
        """Synchronous execution path using wrapped base tool"""
        print(f"\n=== DEBUG _run ===")
        print(f"kwargs received: {kwargs}")
        print(f"is_async: {self.is_async}")
        print(f"run_manager: {run_manager}")
        print(f"config: {config}")
        print(f"base_tool.func: {self.base_tool.func}")
        print(f"base_tool.coroutine: {getattr(self.base_tool, 'coroutine', None)}")
        print(f"base_tool._arun exists: {hasattr(self.base_tool, '_arun')}")
        print(f"base_tool.arun exists: {hasattr(self.base_tool, 'arun')}")
        cleaned_kwargs = self._clean_kwargs(kwargs)
        if cleaned_kwargs:
            kwargs = cleaned_kwargs
        try:
            if self.is_async:
                raise ValueError("Tool marked as async but _run was called")

            if hasattr(self.base_tool, '_run'):
                result = self.base_tool._run(run_manager=run_manager, config=config, **kwargs)
            else:
                # Fallback to func if _run is not available
                result = self.base_tool.func(**kwargs)

   
            if isinstance(result, ToolResult):
                return result  # Already a ToolResult (includes CLIResult)
            return ToolResult(output=str(result))
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"Error in tool '{self.name}': {e}")
          
    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> ToolResult:
        print(f"\n=== DEBUG _arun ===")
        print(f"kwargs received: {kwargs}")
        print(f"is_async: {self.is_async}")
        print(f"run_manager: {run_manager}")
        print(f"config: {config}")
        print(f"base_tool.func: {self.base_tool.func}")
        print(f"base_tool.coroutine: {getattr(self.base_tool, 'coroutine', None)}")
        print(f"base_tool._arun exists: {hasattr(self.base_tool, '_arun')}")
        print(f"base_tool.arun exists: {hasattr(self.base_tool, 'arun')}")
        cleaned_kwargs = self._clean_kwargs(kwargs)
        if cleaned_kwargs:
            kwargs = cleaned_kwargs
        try:
          
            # Improved async handling
            if not self.is_async:
                # Use asyncio.to_thread for sync functions
                import asyncio
                
                result = await asyncio.to_thread(self.base_tool.func, **kwargs)
            else:
                # Check for native async methods
                if hasattr(self.base_tool, '_arun'):
                    result = await self.base_tool._arun(run_manager=run_manager, config=config, **kwargs)
                elif hasattr(self.base_tool, 'arun'):
                    result = await self.base_tool.arun(**kwargs)
                else:
                    
                    # Check if the function itself is async
                    import inspect
                    if inspect.iscoroutinefunction(self.base_tool.func):
                        result = await self.base_tool.func(**kwargs)
                    else:
                        # Fallback to thread-based execution only for sync functions
                        import asyncio
                        result = await asyncio.to_thread(self.base_tool.func, **kwargs)
   
              # Handle result types
            if isinstance(result, ToolResult):
                return result  # Already a ToolResult (includes CLIResult)
            return ToolResult(output=str(result))
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"Error in tool '{self.name}': {e}")
   
    def get_spec(self) -> dict:
        """Get this tool's specification"""
        return {
            "name": self.name,
            "description": self.description,
            "args": self.args_schema().arguments  # That's it. The dictionary is already there.
        }

    @classmethod
    def to_openai_function(cls):
        """Convert the tool's schema to OpenAI function format"""
        print("Entered tool.to_openai_function")
        # Create the Pydantic schema
        schema_instance = cls.args_schema()
        pydantic_schema = cls.args_schema.to_pydantic_schema(schema_instance.arguments)
        print(f"tool Pydantic Schema = {pydantic_schema}")
        # Use LangChain's built-in converter
        openai_function = convert_to_openai_tool(
            Tool(
                name=cls.name,
                description=cls.description,
                func=cls.func,
                args_schema=pydantic_schema
            )
        )
        
        return openai_function
      
    def get_openai_function(self):
        """Get the OpenAI function specification"""
        print("Entered tool.get_openai_function")
        return self.__class__.to_openai_function()   
  
    @classmethod
    def create_adk_tool(cls, func):
        from google.adk.tools import FunctionTool, BaseTool
    
        # Step 1: Get original schema and dynamic Pydantic model
        schema_instance = cls.args_schema()
        DynamicModel = schema_instance.to_pydantic_schema(schema_instance.arguments)
        flat_schema = generate_dereferenced_schema(DynamicModel)
        prop_schemas = flat_schema.get("properties", {})
        top_required = flat_schema.get("required", [])
        original_defs = schema_instance.arguments
    
        # Step 2: Build wrapped function with proper signature (from flattened schema)
        import inspect
        from typing import Optional, get_args, get_origin
    
        param_defs = []
        for name, definition in prop_schemas.items():
            typ = definition.get("type", "string")
            is_required = name in top_required
            default = None if not is_required else inspect._empty
    
            # Very basic type mapping
            type_map = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            resolved_type = type_map.get(typ, str)
            if not is_required:
                resolved_type = Optional[resolved_type]
    
            param_defs.append((name, resolved_type, default))
    
        # Create the function dynamically
        def create_wrapped_function(original_func, func_name, param_defs):
            params = []
            for name, typ, default in param_defs:
                param = inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=typ,
                    default=default
                )
                params.append(param)
    
            sig = inspect.Signature(params)
    
            async def wrapper(**kwargs):
                import inspect, asyncio

                if inspect.iscoroutinefunction(original_func):
                    return await original_func(**kwargs)
                else:
                    # run sync code in thread to avoid blocking
                    return await asyncio.to_thread(original_func, **kwargs)
                
    
            wrapper.__name__ = func_name
            wrapper.__qualname__ = func_name
            wrapper.__signature__ = sig
            return wrapper
    
        wrapped_func = create_wrapped_function(func, cls.name, param_defs)
    
        # Step 3: Create ADK FunctionTool
        orig_tool = FunctionTool(func=wrapped_func)
        orig_tool.description = cls.description
        # Step 4: Schema fixer merge logic
        def merge(adk_prop, pschema, info_def=None):
            enum_cls = type(adk_prop.type)
            if "anyOf" in pschema:
                branches = pschema.pop("anyOf")
                if any(b.get("type") == "null" for b in branches):
                    adk_prop.nullable = True
                arr = next((b for b in branches if b.get("type") == "array"), None)
                chosen = arr or next((b for b in branches if b.get("type") != "null"), {})
                pschema = chosen
    
            if isinstance(info_def, dict):
                if info_def.get("name"):
                    adk_prop.title = info_def["name"]
                if not getattr(adk_prop, "description", None) and info_def.get("description"):
                    adk_prop.description = info_def["description"]
    
            if "description" in pschema:
                adk_prop.description = pschema["description"]
    
            if "items" in pschema or pschema.get("type") == "array":
                adk_prop.type = enum_cls.ARRAY
            elif "type" in pschema:
                t = pschema["type"].upper()
                if hasattr(enum_cls, t):
                    adk_prop.type = getattr(enum_cls, t)
    
            if adk_prop.type == enum_cls.OBJECT:
                adk_prop.properties = adk_prop.properties or {}
                # nested_defs = {}
                # if isinstance(info_def, dict):
                #     if isinstance(info_def.get("nested"), dict):
                #         nested_defs = info_def["nested"]
                #     else:
                #         nested_defs = {
                #             k: v for k, v in info_def.items()
                #             if isinstance(v, dict) and ("type" in v or "nested" in v)
                #         }
                # always allow any dict under info_def[key] as child_info
                for key, child_ps in pschema.get("properties", {}).items():
                    child_info = {}
                    if isinstance(info_def, dict) and isinstance(info_def.get(key), dict):
                        child_info = info_def[key]
                # for key, child_ps in pschema.get("properties", {}).items():
                #     child_info = nested_defs.get(key, {})
                    if key not in adk_prop.properties:
                        Placeholder = type(adk_prop)
                        adk_prop.properties[key] = Placeholder.model_construct(
                            __pydantic_initialised__=True,
                            name=key,
                            title=key,
                            description="",
                            properties=None,
                            items=None,
                            required=[],
                            type=adk_prop.type,
                        )
                    merge(adk_prop.properties[key], child_ps, child_info)
    
                # req = pschema.get("required")
                # if req is None:
                #     req = [k for k, v in nested_defs.items() if isinstance(v, dict) and v.get("required")]
                # adk_prop.required = req or []
                # first try JSON‑schema’s own 'required', otherwise fall back to any dict‑defined 'required' in info_def
                req = pschema.get("required")
                if req is None:
                        req = []
                        if isinstance(info_def, dict):
                                req = [k for k, v in info_def.items() if isinstance(v, dict) and v.get("required")]
                adk_prop.required = req or []
    
            elif adk_prop.type == enum_cls.ARRAY:
                if adk_prop.items is None:
                    Placeholder = type(adk_prop)
                    adk_prop.items = Placeholder.model_construct(
                        __pydantic_initialised__=True,
                        name=adk_prop.title or "",
                        title=adk_prop.title or "",
                        description=pschema.get("items", {}).get("description", ""),
                        properties=None,
                        items=None,
                        required=[],
                        type=adk_prop.type,
                    )
                merge(adk_prop.items, pschema.get("items", {}), None)
#### LITELLM ###
        def normalize_schema_decl(schema):
            from enum import Enum
            # `schema` is a google.genai.types.Schema or similar
            # 1️⃣ normalize this node’s type
            if hasattr(schema, "type") and isinstance(schema.type, Enum):
                schema.type = schema.type.name.lower()
            # 2️⃣ normalize any nested properties
            if getattr(schema, "properties", None):
                for prop in schema.properties.values():
                    normalize_schema_decl(prop)
            # 3️⃣ normalize array items
            if getattr(schema, "items", None):
                normalize_schema_decl(schema.items)
            return schema
#### LITELLM ####
        # Step 5: Wrap in ADK BaseTool with fixed declaration
        class RequiredFieldsFixTool(BaseTool):
            def __init__(self, orig_tool):
                self._orig_tool = orig_tool
                super().__init__(name=orig_tool.name, description=orig_tool.description)
    
            async def run_async(self, **kwargs):
                try:
                    # 1) invoke the underlying tool (may be sync or async)
                    raw = self._orig_tool.run_async(**kwargs)
                    result = await raw if inspect.isawaitable(raw) else raw
        
                    # 2) if it's already a ToolResult or CLIResult, just return it
                    if isinstance(result, ToolResult) or isinstance(result, CLIResult):
                        return result
        
                    # 3) if it's a dict (ADK function tool), dig into the 'response' / 'result' fields
                    if isinstance(result, dict):
                        # ADK wraps tool output under result["response"]["result"]
                        resp = result.get("response", result)
                        payload = resp.get("result", resp) if isinstance(resp, dict) else resp
        
                        # payload may itself be a dict containing output, error, base64_image, etc.
                        if isinstance(payload, dict):
                            return ToolResult(
                                output=payload.get("output", ""),
                                error=payload.get("error", None),
                                base64_image=payload.get("base64_image", None),
                                system=payload.get("system", None),
                            )
                        # otherwise treat payload as a plain string
                        return ToolResult(output=str(payload))
        
                    # 4) if it's just a string, wrap it
                    if isinstance(result, str):
                        return ToolResult(output=result)
        
                    # 5) fallback for anything else
                    return ToolResult(output=str(result))
        
                except ToolError as e:
                    return ToolResult(error=str(e))
                except Exception as e:
                    return ToolResult(error=f"Unhandled error in tool '{self.name}': {e}")
                  
            def _get_declaration(self):
                decl = self._orig_tool._get_declaration()
                
                decl.description = self.description
                if not (decl and decl.parameters and decl.parameters.properties):
                    return decl
    
                decl.parameters.required = top_required
    
                for key, ps in prop_schemas.items():
                    prop = decl.parameters.properties.get(key)
                    if not prop:
                        continue
                    merge(prop, ps, original_defs.get(key))
                ### LITE LLM
                # 2) **Post‑process**: lower‑case enum types into JSON‑schema strings
                # from enum import Enum
                # for prop in decl.parameters.properties.values():
                #    # only convert ADK enum values
                #     if isinstance(prop.type, Enum):
                #        prop.type = prop.type.name.lower()
                #    # handle array items too
                #     if getattr(prop, "items", None) and isinstance(prop.items.type, Enum):
                #         prop.items.type = prop.items.type.name.lower()
                normalize_schema_decl(decl.parameters)
                ###
                return decl
    
        return RequiredFieldsFixTool(orig_tool)


  
    @classmethod
    def create(cls, adk: bool = False):
        """Create a tool instance using class attributes"""
        # with open(tool_log_path, 'a') as f:
        #         f.write("\n\ncreate entered!\n")
       
        # Create an instance of the schema
        
        schema_instance = cls.args_schema()
    
        # # Use your existing Pydantic schema generator
        
        pydantic_schema = cls.args_schema.to_pydantic_schema(schema_instance.arguments)
        
        
        
        if adk:
            
            return cls.create_adk_tool(cls.func) # new method
                   
        #### LANGCHAIN
        else:
            
            schema_instance = cls.args_schema()
            pydantic_schema = cls.args_schema.to_pydantic_schema(schema_instance.arguments)
            if cls.is_async:
                def sync_stub(**kwargs):
                    raise NotImplementedError("This tool is async only")
    
                base_tool = Tool(
                    name=cls.name,
                    description=cls.description,
                    func=sync_stub,
                    coroutine=cls.func,
                    args_schema=pydantic_schema
                )
            else:
                base_tool = Tool(
                    name=cls.name,
                    description=cls.description,
                    func=cls.func,
                    args_schema=pydantic_schema
                )
    
           
            # return instance
            return cls(
                base_tool=base_tool,
                args_schema=cls.args_schema,
                is_async=cls.is_async
            )




### EXAMPLE ###

# ✅ Define Calculator function
def calculator_func(a: int, b: int) -> str:
    return str(a * b)

# ✅ Define ToolArgsSchema for Calculator
class CalculatorArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'a': {
            'name': 'a', 
            'type': 'int', 
            'description': 'First number to multiply',
            'required': True
        },
        'b': {
            'name': 'b', 
            'type': 'int', 
            'description': 'Second number to multiply',
            'required': True
        }
    }

# Calculator Tool
class CalculatorTool(BaseHeavenTool):
    name = "calculator"
    description = "A tool that multiplies two numbers."
    func = calculator_func
    args_schema = CalculatorArgsSchema
    is_async = False


def test_tool_func(message: str) -> str:
    return "The test tool returns this str:`ASDON!@DASN(IJ K@!EDWQACKNSXCBUO!LASDKANSLDALSDLK!@`\n\nMake sure to follow the original instructions while responding to this."

# Updated Test Tool Args Schema with required flags
class TestToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'message': {
            'name': 'message',
            'type': 'str',
            'description': 'Any message to send to the test tool',
            'required': True
        }
    }

class TestTool(BaseHeavenTool):
    name = "TestTool"
    description = "A test tool that performs a test the user needs performed."
    func = test_tool_func
    args_schema = TestToolArgsSchema
    is_async = False


# Example tool definition
# if you need to, add imports
# 
def example_util_func(example_arg_return_dict: bool = False) -> str:
    if return_dict:
        return {"example_result_dict": "example text output"}
    return "Congrats! You and the user successfully tested a tool! This is an example tool result that returns an example string.\n\nThis is example text after 2 line breaks.\n\nThis is an example injected instruction in an example markdown fence:\n\n```markdown\n# Example\n\nContinue\n```"

# from .baseheaventool import BaseHeavenTool, ToolArgsSchema
# from heaven_base.tools.tool_utils.example_util import example_util_func ## Doesnt exist outside of documentation purposes
# Updated Test Tool Args Schema with required flags
class ExampleToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'example_arg_return_dict': {
            'name': 'example_arg_return_dict',
            'type': 'bool',
            'description': 'Set to True to receive a dict and False to get a string',
            'required': True
        }
    }

class ExampleTool(BaseHeavenTool):
    name = "ExampleTool"
    description = "An example tool that returns an example result"
    func = example_util_func
    args_schema = ExampleToolArgsSchema
    is_async = False

# Then on the agent side ie using the tool:
# import the tool
# append it to the HeavenAgentConfig.tools
# Initialize the agent
# Pass that agent to the hermes step in computer_use_demo.tools.base.tool_utils.hermes_utils
# Run it or create a HermesConfig and run it
# Now you understand tools in HEAVEN SDK!

# This is one possible way to allow openAi provider binding to work with tools
class StrictDict(BaseModel):
    class Config:
        extra = Extra.forbid

# ADK Attempt... doesnt work
# class StrictDict(BaseModel):
#     model_config = {"extra": "forbid"}


