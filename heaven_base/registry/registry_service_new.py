"""
Backward-compatible RegistryService that uses SimpleRegistryService internally.

This maintains the same API as the old RegistryService but eliminates the 
builder/repository wrapper layers.
"""

from typing import Dict, Any, Optional, List
from .simple_registry_service import SimpleRegistryService


class RegistryService:
    """Service for managing registries - now simplified without builder/repository layers."""
    
    def __init__(self, registry_dir: str = None):
        """Initialize the RegistryService."""
        self.simple_service = SimpleRegistryService(registry_dir)
    
    def create_registry(self, registry_name: str) -> bool:
        """Create a new registry."""
        return self.simple_service.create_registry(registry_name)
    
    def get_builder(self, registry_name: str) -> Optional['RegistryBuilder']:
        """Get a builder for a registry (backward compatibility)."""
        if not self.simple_service.registry_exists(registry_name):
            return None
        return RegistryBuilder(self.simple_service, registry_name)
    
    def list_registries(self) -> List[str]:
        """List all available registries."""
        return self.simple_service.list_registries()
    
    # Direct access methods (new, cleaner API)
    def get(self, registry_name: str, key: str) -> Optional[Any]:
        """Get an item from a registry with pointer resolution."""
        return self.simple_service.get(registry_name, key)
    
    def get_all(self, registry_name: str) -> Optional[Dict[str, Any]]:
        """Get all items in a registry with pointer resolution."""
        if not self.simple_service.registry_exists(registry_name):
            return None
        return self.simple_service.get_all(registry_name)
    
    def add(self, registry_name: str, key: str, value: Any) -> bool:
        """Add an item to a registry."""
        return self.simple_service.add(registry_name, key, value)
    
    def update(self, registry_name: str, key: str, value: Any) -> bool:
        """Update an item in a registry."""
        return self.simple_service.update(registry_name, key, value)
    
    def delete(self, registry_name: str, key: str) -> bool:
        """Delete an item from a registry."""
        return self.simple_service.delete(registry_name, key)
    
    def list_keys(self, registry_name: str) -> Optional[List[str]]:
        """List all keys in a registry."""
        if not self.simple_service.registry_exists(registry_name):
            return None
        return self.simple_service.list_keys(registry_name)


class RegistryBuilder:
    """Backward-compatible builder interface that wraps SimpleRegistryService."""
    
    def __init__(self, simple_service: SimpleRegistryService, registry_name: str):
        self.simple_service = simple_service
        self.registry_name = registry_name
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the registry."""
        return self.simple_service.get(self.registry_name, key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all items in the registry."""
        return self.simple_service.get_all(self.registry_name)
    
    def add(self, key: str, value: Any) -> bool:
        """Add an item to the registry."""
        return self.simple_service.add(self.registry_name, key, value)
    
    def update(self, key: str, value: Any) -> bool:
        """Update an item in the registry."""
        return self.simple_service.update(self.registry_name, key, value)
    
    def delete(self, key: str) -> bool:
        """Delete an item from the registry."""
        return self.simple_service.delete(self.registry_name, key)
    
    def list_keys(self) -> List[str]:
        """List all keys in the registry."""
        return self.simple_service.list_keys(self.registry_name)