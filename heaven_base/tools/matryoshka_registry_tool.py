from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field

from ..registry.matryoshka_dispatcher import matryoshka_dispatcher
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema


class MatryoshkaRegistryToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'operation': {
            'name': 'operation',
            'type': 'str',
            'description': 'Operation to perform: create_matryoshka, add_to_layer, get_active_layer, switch_active_layer, list_layers, get_all_layers, delete_from_layer, list_layer_keys',
            'required': True
        },
        'matryoshka_name': {
            'name': 'matryoshka_name',
            'type': 'str',
            'description': 'Name of the matryoshka registry',
            'required': False
        },
        'domain': {
            'name': 'domain',
            'type': 'str',
            'description': 'Domain tag for create_matryoshka operation',
            'required': False
        },
        'seed_subdomains': {
            'name': 'seed_subdomains',
            'type': 'list',
            'description': 'List of subdomain names for create_matryoshka operation',
            'required': False
        },
        'subdomain': {
            'name': 'subdomain',
            'type': 'str',
            'description': 'Specific subdomain layer to operate on',
            'required': False
        },
        'key': {
            'name': 'key',
            'type': 'str',
            'description': 'Key for add_to_layer, delete_from_layer operations',
            'required': False
        },
        'value_str': {
            'name': 'value_str',
            'type': 'str',
            'description': 'String value for add_to_layer operation',
            'required': False
        },
        'value_dict': {
            'name': 'value_dict',
            'type': 'dict',
            'description': 'Dictionary value for add_to_layer operation',
            'required': False
        }
    }


def matryoshka_registry_util_func(
    operation: str,
    matryoshka_name: Optional[str] = None,
    domain: Optional[str] = None,
    seed_subdomains: Optional[List[str]] = None,
    subdomain: Optional[str] = None,
    key: Optional[str] = None,
    value_str: Optional[str] = None,
    value_dict: Optional[Dict[str, Any]] = None
) -> str:
    """
    Run the matryoshka registry tool.

    Args:
        operation: Operation to perform
        matryoshka_name: Name of the matryoshka
        domain: Domain tag for create operation
        seed_subdomains: List of subdomain names for create
        subdomain: Specific subdomain to operate on
        key: Key for add/delete operations
        value_str: String value for add operation
        value_dict: Dictionary value for add operation

    Returns:
        Result of the operation as a string
    """
    return matryoshka_dispatcher(
        operation=operation,
        matryoshka_name=matryoshka_name,
        domain=domain,
        seed_subdomains=seed_subdomains,
        subdomain=subdomain,
        key=key,
        value_str=value_str,
        value_dict=value_dict
    )


class MatryoshkaRegistryTool(BaseHeavenTool):
    """Tool for managing matryoshka registries."""

    name = "MatryoshkaRegistryTool"
    description = """Tool for managing matryoshka (nested/hierarchical) registries.

A matryoshka registry is a pattern for organizing related registries into layers:
- Coordinator registry manages multiple subdomain registries
- Each subdomain represents a "layer" (e.g., default, custom, active)
- Uses registry_all_ref pointers for automatic resolution
- Active layer can be switched dynamically

Operations and their required parameters:

1. create_matryoshka:
   - Required: matryoshka_name, domain, seed_subdomains
   - Creates full matryoshka hierarchy with coordinator and subdomain registries
   - Example: operation="create_matryoshka", matryoshka_name="capabilities",
     domain="how_do_i", seed_subdomains=["default", "success_patterns", "custom"]

2. add_to_layer:
   - Required: matryoshka_name, subdomain, key, (value_str OR value_dict)
   - Adds item to specific subdomain layer
   - Example: operation="add_to_layer", matryoshka_name="capabilities",
     subdomain="default", key="starlog", value_dict={"help": "..."}

3. get_active_layer:
   - Required: matryoshka_name
   - Returns active layer contents (automatically resolves registry_all_ref)
   - Example: operation="get_active_layer", matryoshka_name="capabilities"

4. switch_active_layer:
   - Required: matryoshka_name, subdomain
   - Changes which subdomain is the active layer
   - Example: operation="switch_active_layer", matryoshka_name="capabilities",
     subdomain="success_patterns"

5. list_layers:
   - Required: matryoshka_name
   - Lists all available subdomain layers
   - Example: operation="list_layers", matryoshka_name="capabilities"

6. get_all_layers:
   - Required: matryoshka_name
   - Gets contents of all layers (with registry_all_ref resolved)
   - Example: operation="get_all_layers", matryoshka_name="capabilities"

7. delete_from_layer:
   - Required: matryoshka_name, subdomain, key
   - Deletes item from specific layer
   - Example: operation="delete_from_layer", matryoshka_name="capabilities",
     subdomain="custom", key="my_workflow"

8. list_layer_keys:
   - Required: matryoshka_name, subdomain
   - Lists all keys in specific layer
   - Example: operation="list_layer_keys", matryoshka_name="capabilities",
     subdomain="default"

Structure Created by create_matryoshka:

{matryoshka_name}_matryoshka (coordinator)
  ├── root: {name, domain, subdomains, description}
  ├── all_layers: {
  │     subdomain1: "registry_all_ref={matryoshka_name}_{subdomain1}",
  │     subdomain2: "registry_all_ref={matryoshka_name}_{subdomain2}",
  │     ...
  │   }
  └── active: "registry_all_ref={matryoshka_name}_{first_subdomain}"

{matryoshka_name}_{subdomain1}
  ├── _meta: {domain, subdomain, seeded_by, parents_of, matryoshka_name}
  └── <your data>

{matryoshka_name}_{subdomain2}
  ├── _meta: {domain, subdomain, seeded_by, parents_of, matryoshka_name}
  └── <your data>

Use Cases:

1. Capability Catalog with Layers:
   create_matryoshka(name="capabilities", domain="how_do_i",
                    seed_subdomains=["default", "success_patterns", "custom"])

   - default: System components
   - success_patterns: Learned workflows (from TOOT)
   - custom: User additions

2. Environment-Specific Configuration:
   create_matryoshka(name="config", domain="app_settings",
                    seed_subdomains=["development", "staging", "production"])

   switch_active_layer("config", "production")  # Deploy to prod

3. Task Management by Status:
   create_matryoshka(name="tasks", domain="project",
                    seed_subdomains=["planned", "active", "completed"])

   add_to_layer("tasks", "active", "task_001", {...})  # Move task to active

The matryoshka pattern enables:
- Clean separation of concerns across layers
- Dynamic switching between contexts
- Hierarchical organization with domain tags
- Automatic resolution via registry_all_ref pointers
"""
    func = matryoshka_registry_util_func
    args_schema = MatryoshkaRegistryToolArgsSchema
    is_async = False
