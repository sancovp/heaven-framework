"""
Unified dispatcher for matryoshka registry operations.

Provides a single function interface for all matryoshka operations:
- create_matryoshka
- add_to_layer
- get_active_layer
- switch_active_layer
- list_layers
- get_all_layers
- delete_from_layer
- list_layer_keys
"""

from typing import Dict, Any, Optional, List
from .matryoshka_helper import (
    create_matryoshka_registry,
    switch_active_layer,
    get_active_layer,
    list_matryoshka_layers,
    add_to_matryoshka_layer
)
from .registry_service import RegistryService


def matryoshka_dispatcher(
    operation: str,
    matryoshka_name: Optional[str] = None,
    domain: Optional[str] = None,
    seed_subdomains: Optional[List[str]] = None,
    subdomain: Optional[str] = None,
    key: Optional[str] = None,
    value_str: Optional[str] = None,
    value_dict: Optional[Dict[str, Any]] = None,
    registry_dir: Optional[str] = None
) -> str:
    """
    Unified dispatcher for all matryoshka registry operations.

    Args:
        operation: Operation to perform (see below)
        matryoshka_name: Name of the matryoshka
        domain: Domain tag for create operation
        seed_subdomains: List of subdomain names for create
        subdomain: Specific subdomain to operate on
        key: Key for add/get/delete operations
        value_str: String value for add operation
        value_dict: Dictionary value for add operation
        registry_dir: Optional custom registry directory

    Operations:
        create_matryoshka:
            - Required: matryoshka_name, domain, seed_subdomains
            - Creates full matryoshka hierarchy
            - Returns: Summary of created structure

        add_to_layer:
            - Required: matryoshka_name, subdomain, key, (value_str OR value_dict)
            - Adds item to specific subdomain layer
            - Returns: Success message

        get_active_layer:
            - Required: matryoshka_name
            - Returns active layer contents (auto-resolves registry_all_ref)
            - Returns: JSON representation of active layer

        switch_active_layer:
            - Required: matryoshka_name, subdomain
            - Changes which subdomain is active
            - Returns: Success message

        list_layers:
            - Required: matryoshka_name
            - Lists all available subdomain layers
            - Returns: Comma-separated list of subdomains

        get_all_layers:
            - Required: matryoshka_name
            - Gets contents of all layers (resolved)
            - Returns: JSON representation of all layers

        delete_from_layer:
            - Required: matryoshka_name, subdomain, key
            - Deletes item from specific layer
            - Returns: Success message

        list_layer_keys:
            - Required: matryoshka_name, subdomain
            - Lists all keys in specific layer
            - Returns: Comma-separated list of keys

    Returns:
        String result of the operation
    """

    # Determine value to use
    value = None
    if value_dict is not None:
        value = value_dict
    elif value_str is not None:
        value = value_str

    # CREATE_MATRYOSHKA
    if operation == "create_matryoshka":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for create_matryoshka"
        if not domain:
            return "❌ Error: domain is required for create_matryoshka"
        if not seed_subdomains:
            return "❌ Error: seed_subdomains is required for create_matryoshka"

        try:
            result = create_matryoshka_registry(
                name=matryoshka_name,
                domain=domain,
                seed_subdomains=seed_subdomains,
                registry_dir=registry_dir
            )

            return (
                f"✅ Created matryoshka '{matryoshka_name}':\n"
                f"   Coordinator: {result['coordinator']}\n"
                f"   Subdomains: {', '.join(result['subdomains'].keys())}\n"
                f"   Active: {result['active']}\n"
                f"   Domain: {domain}"
            )
        except Exception as e:
            return f"❌ Error creating matryoshka: {str(e)}"

    # ADD_TO_LAYER
    elif operation == "add_to_layer":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for add_to_layer"
        if not subdomain:
            return "❌ Error: subdomain is required for add_to_layer"
        if not key:
            return "❌ Error: key is required for add_to_layer"
        if value is None:
            return "❌ Error: either value_str or value_dict is required for add_to_layer"

        try:
            success = add_to_matryoshka_layer(
                matryoshka_name=matryoshka_name,
                subdomain=subdomain,
                key=key,
                value=value,
                registry_dir=registry_dir
            )

            if success:
                return f"✅ Added '{key}' to {matryoshka_name}/{subdomain} layer"
            else:
                return f"❌ Failed to add '{key}' (key might already exist)"
        except Exception as e:
            return f"❌ Error adding to layer: {str(e)}"

    # GET_ACTIVE_LAYER
    elif operation == "get_active_layer":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for get_active_layer"

        try:
            contents = get_active_layer(
                matryoshka_name=matryoshka_name,
                registry_dir=registry_dir
            )

            if contents is None:
                return f"❌ Matryoshka '{matryoshka_name}' not found"

            import json
            return f"Active layer contents for '{matryoshka_name}':\n{json.dumps(contents, indent=2)}"
        except Exception as e:
            return f"❌ Error getting active layer: {str(e)}"

    # SWITCH_ACTIVE_LAYER
    elif operation == "switch_active_layer":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for switch_active_layer"
        if not subdomain:
            return "❌ Error: subdomain is required for switch_active_layer"

        try:
            success = switch_active_layer(
                matryoshka_name=matryoshka_name,
                new_active_subdomain=subdomain,
                registry_dir=registry_dir
            )

            if success:
                return f"✅ Switched '{matryoshka_name}' active layer to '{subdomain}'"
            else:
                return f"❌ Failed to switch active layer (subdomain '{subdomain}' might not exist)"
        except Exception as e:
            return f"❌ Error switching active layer: {str(e)}"

    # LIST_LAYERS
    elif operation == "list_layers":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for list_layers"

        try:
            layers = list_matryoshka_layers(
                matryoshka_name=matryoshka_name,
                registry_dir=registry_dir
            )

            if layers is None:
                return f"❌ Matryoshka '{matryoshka_name}' not found"

            return f"Layers in '{matryoshka_name}': {', '.join(layers)}"
        except Exception as e:
            return f"❌ Error listing layers: {str(e)}"

    # GET_ALL_LAYERS
    elif operation == "get_all_layers":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for get_all_layers"

        try:
            service = RegistryService(registry_dir)
            coordinator_name = f"{matryoshka_name}_matryoshka"

            # Get all_layers key (this will resolve all registry_all_ref pointers!)
            all_layers = service.get(coordinator_name, "all_layers")

            if all_layers is None:
                return f"❌ Matryoshka '{matryoshka_name}' not found"

            import json
            return f"All layers in '{matryoshka_name}':\n{json.dumps(all_layers, indent=2)}"
        except Exception as e:
            return f"❌ Error getting all layers: {str(e)}"

    # DELETE_FROM_LAYER
    elif operation == "delete_from_layer":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for delete_from_layer"
        if not subdomain:
            return "❌ Error: subdomain is required for delete_from_layer"
        if not key:
            return "❌ Error: key is required for delete_from_layer"

        try:
            service = RegistryService(registry_dir)
            registry_name = f"{matryoshka_name}_{subdomain}"

            success = service.delete(registry_name, key)

            if success:
                return f"✅ Deleted '{key}' from {matryoshka_name}/{subdomain} layer"
            else:
                return f"❌ Failed to delete '{key}' (key might not exist)"
        except Exception as e:
            return f"❌ Error deleting from layer: {str(e)}"

    # LIST_LAYER_KEYS
    elif operation == "list_layer_keys":
        if not matryoshka_name:
            return "❌ Error: matryoshka_name is required for list_layer_keys"
        if not subdomain:
            return "❌ Error: subdomain is required for list_layer_keys"

        try:
            service = RegistryService(registry_dir)
            registry_name = f"{matryoshka_name}_{subdomain}"

            keys = service.list_keys(registry_name)

            if keys is None:
                return f"❌ Layer '{matryoshka_name}/{subdomain}' not found"

            # Filter out _meta
            data_keys = [k for k in keys if k != "_meta"]

            if not data_keys:
                return f"Layer '{matryoshka_name}/{subdomain}' is empty"

            return f"Keys in '{matryoshka_name}/{subdomain}': {', '.join(data_keys)}"
        except Exception as e:
            return f"❌ Error listing layer keys: {str(e)}"

    else:
        return f"❌ Error: Unknown operation '{operation}'"
