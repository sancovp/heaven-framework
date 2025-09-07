import os
import json
import re
from typing import Dict, Any, Optional, List
from ..utils.get_env_value import EnvConfigUtil


# Registry pointer patterns  
REF_KEY_PATTERN = re.compile(r'^registry_key_ref=([^:]+):(.+)$')
REF_OBJ_PATTERN = re.compile(r'^registry_object_ref=([^:]+):(.+)$')
REF_ALL_PATTERN = re.compile(r'^registry_all_ref=([^:]+)$')
MAX_REF_DEPTH = 99


class SimpleRegistryService:
    """Simplified registry service - just JSON CRUD operations with pointer resolution."""
    
    def __init__(self, registry_dir: str = None):
        """Initialize the SimpleRegistryService.
        
        Args:
            registry_dir: Directory containing registry JSON files (defaults to HEAVEN_DATA_DIR/registry)
        """
        if registry_dir is None:
            registry_dir = os.path.join(EnvConfigUtil.get_heaven_data_dir(), 'registry')
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
    
    def _get_registry_path(self, registry_name: str) -> str:
        """Get the file path for a registry with dual-lookup (user dir first, then library)."""
        # Check if name ends with _registry, if not add it
        if registry_name.endswith('_registry'):
            filename = f"{registry_name}.json"
        else:
            filename = f"{registry_name}_registry.json"
        
        # First check user's HEAVEN_DATA_DIR/registry/
        user_path = os.path.join(self.registry_dir, filename)
        if os.path.exists(user_path):
            return user_path
            
        # Then check library-level heaven_base/registry/
        try:
            import heaven_base
            library_registry_dir = os.path.join(os.path.dirname(heaven_base.__file__), 'registry')
            library_path = os.path.join(library_registry_dir, filename)
            if os.path.exists(library_path):
                return library_path
        except ImportError:
            pass
            
        # Default to user path (for creation/writing)
        return user_path
    
    def _load_registry_data(self, registry_name: str) -> Dict[str, Any]:
        """Load registry data from JSON file."""
        registry_path = self._get_registry_path(registry_name)
        if not os.path.exists(registry_path):
            return {}
        
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_registry_data(self, registry_name: str, data: Dict[str, Any]) -> None:
        """Save registry data to JSON file."""
        registry_path = self._get_registry_path(registry_name)
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _resolve_if_pointer(self, value: Any, *, _seen: set = None) -> Any:
        """Resolve registry-pointer strings.
        
        • registry_key_ref=<registry>:<key> → returns the locator string "@<registry>/<key>"
        • registry_object_ref=<registry>:<key> → opens that registry, fetches the value, then continues resolving
        
        Cycles are detected; depth is capped at MAX_REF_DEPTH.
        """
        if not isinstance(value, str):
            return value
        
        _seen = _seen or set()
        ref = value.strip()
        
        if ref in _seen:
            raise ValueError(f"Cyclic registry reference detected: {ref}")
        if len(_seen) >= MAX_REF_DEPTH:
            raise ValueError("Max registry-reference depth exceeded")
        
        # registry_key_ref → locator string only
        m = REF_KEY_PATTERN.match(ref)
        if m:
            registry_name, key = m.groups()
            return f"@{registry_name}/{key}"
        
        # registry_object_ref → fetch value, then recurse
        m = REF_OBJ_PATTERN.match(ref)
        if m:
            registry_name, key = m.groups()
            _seen.add(ref)
            obj = self.get(registry_name, key)  # may itself be a pointer
            return self._resolve_if_pointer(obj, _seen=_seen)
        
        # registry_all_ref → fetch entire registry contents
        m = REF_ALL_PATTERN.match(ref)
        if m:
            registry_name = m.group(1)
            _seen.add(ref)
            all_data = self.get_all(registry_name)  # may contain pointers that get resolved
            return all_data
        
        # not a recognised pointer
        return value
    
    def get(self, registry_name: str, key: str) -> Optional[Any]:
        """Get an item from a registry with pointer resolution."""
        data = self._load_registry_data(registry_name)
        raw_value = data.get(key)
        return self._resolve_if_pointer(raw_value)
    
    def get_all(self, registry_name: str) -> Dict[str, Any]:
        """Get all items in a registry with pointer resolution."""
        data = self._load_registry_data(registry_name)
        return {k: self._resolve_if_pointer(v) for k, v in data.items()}
    
    def get_raw(self, registry_name: str, key: str) -> Optional[Any]:
        """Get raw value without pointer resolution (useful for editing pointers)."""
        data = self._load_registry_data(registry_name)
        return data.get(key)
    
    def get_all_raw(self, registry_name: str) -> Dict[str, Any]:
        """Get all raw values without pointer resolution."""
        return self._load_registry_data(registry_name)
    
    def set(self, registry_name: str, key: str, value: Any) -> None:
        """Set an item in a registry."""
        data = self._load_registry_data(registry_name)
        data[key] = value
        self._save_registry_data(registry_name, data)
    
    def delete(self, registry_name: str, key: str) -> bool:
        """Delete an item from a registry."""
        data = self._load_registry_data(registry_name)
        if key not in data:
            return False
        
        del data[key]
        self._save_registry_data(registry_name, data)
        return True
    
    def update(self, registry_name: str, key: str, value: Any) -> bool:
        """Update an item with pointer validation."""
        data = self._load_registry_data(registry_name)
        current_raw = data.get(key)
        
        # Prevent updating pointers (same logic as original)
        if isinstance(current_raw, str) and current_raw.startswith(
            ("registry_key_ref=", "registry_object_ref=")
        ):
            if current_raw.startswith("registry_key_ref="):
                _, payload = current_raw.split("=", 1)
                target_locator = f"@{payload.replace(':', '/')}"
            else:  # registry_object_ref
                _, payload = current_raw.split("=", 1)
                target_registry, target_key = payload.split(":", 1)
                target_locator = f"@{target_registry}/{target_key}"
            
            raise ValueError(
                f"Cannot update '{registry_name}:{key}'—it is a pointer to "
                f"'{target_locator}'.\n"
                "Update the referenced registry/key instead, or replace the "
                "pointer string explicitly if you intend to repoint it."
            )
        
        if key not in data:
            return False
        
        data[key] = value
        self._save_registry_data(registry_name, data)
        return True
    
    def add(self, registry_name: str, key: str, value: Any) -> bool:
        """Add an item to a registry (fails if key already exists)."""
        data = self._load_registry_data(registry_name)
        if key in data:
            return False
        
        data[key] = value
        self._save_registry_data(registry_name, data)
        return True
    
    def list_keys(self, registry_name: str) -> List[str]:
        """List all keys in a registry."""
        data = self._load_registry_data(registry_name)
        return list(data.keys())
    
    def list_registries(self) -> List[str]:
        """List all available registries."""
        registries = []
        if not os.path.exists(self.registry_dir):
            return registries
        
        for filename in os.listdir(self.registry_dir):
            if filename.endswith('_registry.json'):
                # brain_personas_registry.json -> brain_personas
                registries.append(filename[:-14])  # Remove _registry.json extension
            elif filename.endswith('.json'):
                # Handle files that don't have _registry suffix
                base_name = filename[:-5]  # Remove .json
                if base_name.endswith('_registry'):
                    # This is already a _registry file, extract base name
                    registries.append(base_name[:-9])  # Remove _registry suffix
                else:
                    # This shouldn't happen with our naming convention, but handle it
                    registries.append(base_name)
        return sorted(registries)
    
    def create_registry(self, registry_name: str) -> bool:
        """Create a new empty registry."""
        registry_path = self._get_registry_path(registry_name)
        if os.path.exists(registry_path):
            return False
        
        self._save_registry_data(registry_name, {})
        return True
    
    def registry_exists(self, registry_name: str) -> bool:
        """Check if a registry exists."""
        registry_path = self._get_registry_path(registry_name)
        return os.path.exists(registry_path)