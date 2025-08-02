"""
Simplified RegistryFactory that works with the new registry system.
No more builder/repository file generation - just JSON files.
"""

import os
import json
from typing import List
from ..utils.get_env_value import EnvConfigUtil
from .simple_registry_service import SimpleRegistryService


class RegistryFactory:
    """Factory class for creating and managing registries - simplified version."""
    
    def __init__(self, base_path: str = None):
        """Initialize the RegistryFactory.
        
        Args:
            base_path: Base directory for storing registry files (defaults to HEAVEN_DATA_DIR/registry)
        """
        if base_path is None:
            base_path = os.path.join(EnvConfigUtil.get_heaven_data_dir(), 'registry')
        self.base_path = base_path
        self.simple_service = SimpleRegistryService(base_path)
    
    def create_registry(self, registry_name: str) -> bool:
        """Create a new registry (just the JSON file, no generated Python files).
        
        Args:
            registry_name: Name of the registry to create
            
        Returns:
            bool: True if created successfully, False if already exists
        """
        return self.simple_service.create_registry(registry_name)
    
    def load_registry(self, registry_name: str) -> Optional[SimpleRegistryService]:
        """Load a registry (returns SimpleRegistryService for compatibility).
        
        Args:
            registry_name: Name of the registry to load
            
        Returns:
            SimpleRegistryService instance or None if not found
        """
        if self.simple_service.registry_exists(registry_name):
            return self.simple_service
        return None
    
    def list_registries(self) -> List[str]:
        """List all available registries.
        
        Returns:
            List of registry names
        """
        return self.simple_service.list_registries()
    
    def registry_exists(self, registry_name: str) -> bool:
        """Check if a registry exists."""
        return self.simple_service.registry_exists(registry_name)
    
    # Backward compatibility methods
    def get_registry_path(self, registry_name: str) -> str:
        """Get the path to a registry JSON file."""
        return os.path.join(self.base_path, f"{registry_name}.json")
    
    def migrate_old_registry(self, registry_name: str) -> bool:
        """Migrate an old-style registry (with _registry.json suffix) to new format.
        
        Args:
            registry_name: Name of the registry to migrate
            
        Returns:
            bool: True if migration successful, False if not needed/failed
        """
        old_path = os.path.join(self.base_path, f"{registry_name}_registry.json")
        new_path = os.path.join(self.base_path, f"{registry_name}.json")
        
        if os.path.exists(old_path) and not os.path.exists(new_path):
            try:
                # Copy the old file to new location
                with open(old_path, 'r') as old_f:
                    data = json.load(old_f)
                with open(new_path, 'w') as new_f:
                    json.dump(data, new_f, indent=2)
                
                print(f"Migrated registry '{registry_name}' from old format to new format")
                return True
            except (IOError, json.JSONDecodeError) as e:
                print(f"Failed to migrate registry '{registry_name}': {e}")
                return False
        
        return False  # No migration needed or failed
    
    def cleanup_old_files(self, registry_name: str) -> bool:
        """Remove old builder/repository files for a registry.
        
        Args:
            registry_name: Name of the registry to clean up
            
        Returns:
            bool: True if cleanup performed, False if no files to clean
        """
        files_to_remove = [
            f"{registry_name}_builder.py",
            f"{registry_name}_repository.py",
            f"{registry_name}_registry.json"  # Old naming convention
        ]
        
        removed_any = False
        for filename in files_to_remove:
            file_path = os.path.join(self.base_path, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed old file: {filename}")
                    removed_any = True
                except OSError as e:
                    print(f"Failed to remove {filename}: {e}")
        
        return removed_any