import os
import json
import importlib.util
from typing import Dict, Any, Optional, List
from ..utils.get_env_value import EnvConfigUtil

class RegistryFactory:
    """Factory class for creating and managing registries."""
    
    def __init__(self, base_path: str = None):
        """Initialize the RegistryFactory.
        
        Args:
            base_path: Base directory for storing registry files (defaults to HEAVEN_DATA_DIR/registry)
        """
        if base_path is None:
            base_path = os.path.join(EnvConfigUtil.get_heaven_data_dir(), 'registry')
        self.base_path = base_path
        self.registries = {}
        self._ensure_base_path_exists()
    
    def _ensure_base_path_exists(self) -> None:
        """Ensure the base path directory exists."""
        os.makedirs(self.base_path, exist_ok=True)
    
    def create_registry(self, registry_name: str) -> bool:
        """Create a new registry.
        
        Args:
            registry_name: Name of the registry to create
            
        Returns:
            bool: True if created successfully, False if already exists
        """
        registry_path = os.path.join(self.base_path, f"{registry_name}_registry.json")
        
        if os.path.exists(registry_path):
            return False
        
        # Create an empty registry file
        with open(registry_path, 'w') as f:
            json.dump({}, f, indent=2)
        
        # Create the repository file
        repo_path = os.path.join(self.base_path, f"{registry_name}_repository.py")
        repo_content = f'''
import os
import json
from typing import Dict, Any, Optional

class {registry_name.capitalize()}Repository:
    """Repository for {registry_name} registry."""
    
    def __init__(self):
        self.registry_path = os.path.join(os.path.dirname(__file__), '{registry_name}_registry.json')
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from file."""
        if not os.path.exists(self.registry_path):
            return {{}}
        
        with open(self.registry_path, 'r') as f:
            return json.load(f)
    
    def _save_registry(self) -> None:
        """Save the registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the registry."""
        return self.registry.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all items in the registry."""
        return self.registry
    
    def add(self, key: str, value: Any) -> bool:
        """Add an item to the registry."""
        if key in self.registry:
            return False
        
        self.registry[key] = value
        self._save_registry()
        return True
    
    def update(self, key: str, value: Any) -> bool:
        """Update an item in the registry."""
        if key not in self.registry:
            return False
        
        self.registry[key] = value
        self._save_registry()
        return True
    
    def delete(self, key: str) -> bool:
        """Delete an item from the registry."""
        if key not in self.registry:
            return False
        
        del self.registry[key]
        self._save_registry()
        return True
'''
        with open(repo_path, 'w') as f:
            f.write(repo_content)
        
        # Create the builder file
        builder_path = os.path.join(self.base_path, f"{registry_name}_builder.py")
        builder_content = f'''
from typing import Dict, Any, Optional, List
from {registry_name}_repository import {registry_name.capitalize()}Repository

class {registry_name.capitalize()}Builder:
    """Builder for {registry_name} registry."""
    
    def __init__(self):
        self.repository = {registry_name.capitalize()}Repository()
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the registry."""
        return self.repository.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all items in the registry."""
        return self.repository.get_all()
    
    def add(self, key: str, value: Any) -> bool:
        """Add an item to the registry."""
        return self.repository.add(key, value)
    
    def update(self, key: str, value: Any) -> bool:
        """Update an item in the registry."""
        return self.repository.update(key, value)
    
    def delete(self, key: str) -> bool:
        """Delete an item from the registry."""
        return self.repository.delete(key)
    
    def list_keys(self) -> List[str]:
        """List all keys in the registry."""
        return list(self.repository.get_all().keys())
'''
        with open(builder_path, 'w') as f:
            f.write(builder_content)
        
        return True
    
    def load_registry(self, registry_name: str) -> Optional[Any]:
        """Load a registry module dynamically.
        
        Args:
            registry_name: Name of the registry to load
            
        Returns:
            The loaded repository class or None if not found
        """
        repo_path = os.path.join(self.base_path, f"{registry_name}_repository.py")
        
        if not os.path.exists(repo_path):
            return None
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(
            f"{registry_name}_repository", repo_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the repository class
        repo_class = getattr(module, f"{registry_name.capitalize()}Repository")
        return repo_class()
    
    def list_registries(self) -> List[str]:
        """List all available registries.
        
        Returns:
            List of registry names
        """
        registries = []
        for filename in os.listdir(self.base_path):
            if filename.endswith('_registry.json'):
                registries.append(filename.replace('_registry.json', ''))
        return registries