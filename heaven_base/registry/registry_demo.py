#!/usr/bin/env python3

"""
Registry System Demo

This script demonstrates how to use the registry system to create registries
and manage registry entries.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import the registry modules
sys.path.append(str(Path(__file__).parent.parent))

from registry.RegistryFactory import RegistryFactory
from registry.registry_service import RegistryService


def main():
    """Main function to demonstrate the registry system."""
    print("Registry System Demo")
    print("-" * 50)
    
    # Create a registry service
    service = RegistryService()
    
    # List existing registries
    print("\nExisting registries:")
    registries = service.list_registries()
    if registries:
        for registry in registries:
            print(f"  - {registry}")
    else:
        print("  No registries found")
    
    # Create a new registry
    registry_name = "cheatsheets"
    print(f"\nCreating registry '{registry_name}'...")
    result = service.create_registry(registry_name)
    print(f"  Result: {result}")
    
    # Add items to the registry
    print(f"\nAdding items to registry '{registry_name}'...")
    items = {
        "heaven_cs": "This is the Heaven cheatsheet content",
        "hermes_cs": "This is the Hermes cheatsheet content",
        "prompt_engineering_cs": "This is the Prompt Engineering cheatsheet content"
    }
    
    for key, value in items.items():
        result = service.add(registry_name, key, value)
        print(f"  Added '{key}': {result}")
    
    # Get all items from the registry
    print(f"\nItems in registry '{registry_name}':")
    all_items = service.get_all(registry_name)
    for key, value in all_items.items():
        print(f"  {key}: {value}")
    
    # Update an item
    update_key = "heaven_cs"
    update_value = "Updated Heaven cheatsheet content"
    print(f"\nUpdating item '{update_key}'...")
    result = service.update(registry_name, update_key, update_value)
    print(f"  Result: {result}")
    
    # Get the updated item
    print(f"\nUpdated item '{update_key}':")
    item = service.get(registry_name, update_key)
    print(f"  {update_key}: {item}")
    
    # List all keys
    print(f"\nKeys in registry '{registry_name}':")
    keys = service.list_keys(registry_name)
    for key in keys:
        print(f"  - {key}")
    
    # Delete an item
    delete_key = "hermes_cs"
    print(f"\nDeleting item '{delete_key}'...")
    result = service.delete(registry_name, delete_key)
    print(f"  Result: {result}")
    
    # List all keys after deletion
    print(f"\nKeys in registry '{registry_name}' after deletion:")
    keys = service.list_keys(registry_name)
    for key in keys:
        print(f"  - {key}")
    
    print("\nRegistry Demo completed successfully!")


if __name__ == "__main__":
    main()
