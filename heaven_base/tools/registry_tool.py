from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

from ..registry.registry_service import RegistryService
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema

import uuid
import json
from datetime import datetime
import copy

      
class RegistryToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'operation': {
            'name': 'operation',
            'type': 'str',
            'description': 'Operation to perform: create_registry, list_registries, get, get_all, add, update, delete, list_keys',
            'required': True
        },
        'registry_name': {
            'name': 'registry_name',
            'type': 'str',
            'description': 'Name of the registry to operate on',
            'required': False
        },
        'key': {
            'name': 'key',
            'type': 'str',
            'description': 'Key for get, add, update, delete operations',
            'required': False
        },
        'value_str': {
            'name': 'value_str',
            'type': 'str',
            'description': 'String value for add and update operations. Use this for simple string values.',
            'required': False
        },
        'value_dict': {
            'name': 'value_dict',
            'type': 'dict',
            'description': 'Dictionary/object value for add and update operations. Use this for structured data.',
            'required': False
        }
    }



def registry_util_func(operation: str, registry_name: Optional[str] = None, 
                   key: Optional[str] = None, value_str: Optional[str] = None,
                   value_dict: Optional[Dict[str, Any]] = None) -> str:
        """Run the registry tool.
        
        Args:
            operation: Operation to perform
            registry_name: Name of the registry to operate on
            key: Key for get, add, update, delete operations
            value_str: String value for add and update operations
            value_dict: Dictionary value for add and update operations
            
        Returns:
            Result of the operation as a string
        """
        service = RegistryService()
        
        # Determine which value to use
        value = None
        if value_dict is not None:
            value = value_dict
        elif value_str is not None:
            value = value_str
        
        if operation == "create_registry":
            if not registry_name:
                return "Error: registry_name is required for create_registry operation"
            
            result = service.create_registry(registry_name)
            return f"Registry '{registry_name}' created: {result}"
        
        elif operation == "list_registries":
            registries = service.list_registries()
            if not registries:
                return "No registries found"
            
            return f"Available registries: {', '.join(registries)}"
        
        # Operations that require a registry_name
        if not registry_name:
            return "Error: registry_name is required for this operation"
        
        if operation == "get_all":
            items = service.get_all(registry_name)
            if items is None:
                return f"Error: Registry '{registry_name}' not found"
            
            if not items:
                return f"Registry '{registry_name}' is empty"
            
            return f"Items in registry '{registry_name}': {items}"
        
        elif operation == "list_keys":
            keys = service.list_keys(registry_name)
            if keys is None:
                return f"Error: Registry '{registry_name}' not found"
            
            if not keys:
                return f"Registry '{registry_name}' is empty"
            
            return f"Keys in registry '{registry_name}': {', '.join(keys)}"
        
        # Operations that require a key
        if not key:
            return "Error: key is required for this operation"
        
        if operation == "get":
            item = service.get(registry_name, key)
            if item is None:
                return f"Item '{key}' not found in registry '{registry_name}'"
            
            return f"Item '{key}' in registry '{registry_name}': {item}"
        
        elif operation == "delete":
            result = service.delete(registry_name, key)
            return f"Item '{key}' deleted from registry '{registry_name}': {result}"
        
        # Operations that require a value
        if operation in ["add", "update"]:
            if value is None:
                return "Error: either value_str or value_dict is required for this operation"
            
            if operation == "add":
                result = service.add(registry_name, key, value)
                return f"Item '{key}' added to registry '{registry_name}': {result}"
            
            elif operation == "update":
                result = service.update(registry_name, key, value)
                return f"Item '{key}' updated in registry '{registry_name}': {result}"
        
        return f"Error: Unknown operation '{operation}'"
  
class RegistryTool(BaseHeavenTool):
    """Tool for managing registries."""
    
    name = "RegistryTool"
    description = """Tool for managing registries in the system.

    The Registry System provides key-value storage for various types of data across the system.
    Each registry is a separate storage container that can hold multiple key-value pairs.

    Operations and their required parameters:

    1. create_registry:
       - Required: registry_name
       - Creates a new registry with the given name

    2. list_registries:
       - No parameters required
       - Returns a list of all available registries

    3. get:
       - Required: registry_name, key
       - Returns the value associated with the key in the specified registry

    4. get_all:
       - Required: registry_name
       - Returns all key-value pairs in the specified registry

    5. add:
       - Required: registry_name, key, value_str OR value_dict
       - Adds a new key-value pair to the specified registry
       - Use value_str for simple string values, value_dict for structured data

    6. update:
       - Required: registry_name, key, value_str OR value_dict
       - Updates an existing key with a new value in the specified registry
       - Use value_str for simple string values, value_dict for structured data

    7. delete:
       - Required: registry_name, key
       - Removes a key-value pair from the specified registry

    8. list_keys:
       - Required: registry_name
       - Returns a list of all keys in the specified registry

    Examples:
    - Create a registry: operation="create_registry", registry_name="my_registry"
    - Add a string item: operation="add", registry_name="my_registry", key="item1", value_str="value1"
    - Add a dict item: operation="add", registry_name="my_registry", key="config", value_dict={"setting": "value"}
    - Get an item: operation="get", registry_name="my_registry", key="item1"

    Registry-reference syntax (any string field can be a pointer):

    registry_key_ref=<registry_name>:<key>

    • Resolves to the locator string “@<registry>/<key>” (the key itself).

    • Use when a parent record just needs to point at another record.

    Example:

    "title": "registry_key_ref=task_registry:T123"

    --> read-time result: "@task_registry/T123"

    registry_object_ref=<registry_name>:<key>#/<optional/json/pointer>

    • Resolves to the value stored at that key (optionally narrowed by a JSON-pointer path).

    • Recursion continues on the returned value until a non-pointer is reached.

    Example:

    "spec": "registry_object_ref=settings_registry:colors#/header/bg"

    --> read-time result: "blue" (assuming that path exists)

    registry_all_ref=<registry_name>

    • Resolves to the entire contents of the specified registry.

    • All values in the returned registry are also resolved if they contain pointers.

    Example:

    "my_data": "registry_all_ref=knowledge_base"

    --> read-time result: {entire contents of knowledge_base registry}

    Rules & behaviour:

    • Write the reference exactly as a plain string—no braces, no quotes inside.

    • Reads (get, get_all) automatically resolve:

    – key-refs return the locator string.

    – object-refs return the full (or sliced) value.

    • Pointers can chain indefinitely; cycles are detected and depth is capped at 99 hops.

    • Update-guard: if you try update() on an entry whose current value is a pointer string, the operation is refused with an error telling you the target locator—modify the referenced registry/key instead, or replace the pointer string explicitly.
    """
    func = registry_util_func
    args_schema = RegistryToolArgsSchema
    is_async = False
    
    
    