"""
Simplified RegistryFactory that works with the new registry system.
Uses _registry.json naming for consistency with existing files.
No more builder/repository file generation.
"""

import os
from typing import List, Optional
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