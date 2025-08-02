import os
import importlib.util
import sys
from typing import Dict, Any, Optional, List, Type
import re
from .RegistryFactory import RegistryFactory


REF_KEY_PATTERN = re.compile(r'^registry_key_ref=([^:]+):(.+)$')

REF_OBJ_PATTERN = re.compile(r'^registry_object_ref=([^:]+):(.+)$')

MAX_REF_DEPTH   = 99


class RegistryService:
    """Service for managing registries."""
    
    def __init__(self):
        """Initialize the RegistryService."""
        self.factory = RegistryFactory()
        self.builders = {}
        self._load_existing_registries()
    
    def _load_existing_registries(self) -> None:
        """Load all existing registries."""
        for registry_name in self.factory.list_registries():
            self._load_builder(registry_name)
          
    def _load_builder(self, registry_name: str) -> bool:
        """Load a builder for a registry.
    
        Args:
            registry_name: Name of the registry
    
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        builder_path = os.path.join(self.factory.base_path, f"{registry_name}_builder.py")
    
        if not os.path.exists(builder_path):
            return False
    
        # Add the registry directory to the path so relative imports work
        registry_dir = self.factory.base_path
        if registry_dir not in sys.path:
            sys.path.append(registry_dir)
    
        try:
            # Load the module dynamically without modifying the file
            spec = importlib.util.spec_from_file_location(
                f"{registry_name}_builder", builder_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    
            # Get the builder class
            builder_class = getattr(module, f"{registry_name.capitalize()}Builder")
            self.builders[registry_name] = builder_class()
            return True
        except ImportError as e:
            print(f"Error importing builder for registry '{registry_name}': {str(e)}")
            return False
        except AttributeError as e:
            print(f"Error getting builder class for registry '{registry_name}': {str(e)}")
            return False
        except Exception as e:
            print(f"Unexpected error loading builder for registry '{registry_name}': {str(e)}")
            return False
    
    def create_registry(self, registry_name: str) -> bool:
        """Create a new registry.
        
        Args:
            registry_name: Name of the registry to create
            
        Returns:
            bool: True if created successfully, False if already exists
        """
        result = self.factory.create_registry(registry_name)
        if result:
            self._load_builder(registry_name)
        return result
    
    def get_builder(self, registry_name: str) -> Optional[Any]:
        """Get a builder for a registry.
        
        Args:
            registry_name: Name of the registry
            
        Returns:
            The builder or None if not found
        """
        return self.builders.get(registry_name)
    
    def list_registries(self) -> List[str]:
        """List all available registries.
        
        Returns:
            List of registry names
        """
        return self.factory.list_registries()
    
    # ----------------------------------------------------------------

    # 1.  Pointer-resolver

    # ----------------------------------------------------------------
    # Might not be the exact pattern we want yet
        
    def _resolve_if_pointer(self, value: Any,

                            *, _seen: set[str] | None = None) -> Any:

        """

        Resolve registry-pointer strings.


        • registry_key_ref=<registry>:<key>

            → returns the locator string  "@<registry>/<key>"

            (we do NOT open the registry here).


        • registry_object_ref=<registry>:<key>

            → opens that registry, fetches the value, then continues

            resolving until a non-pointer value is reached.


        Cycles are detected; depth is capped at MAX_REF_DEPTH.

        """

        if not isinstance(value, str):

            return value


        _seen = _seen or set()

        ref   = value.strip()


        if ref in _seen:

            raise ValueError(f"Cyclic registry reference detected: {ref}")

        if len(_seen) >= MAX_REF_DEPTH:

            raise ValueError("Max registry-reference depth exceeded")


        # ───── registry_key_ref  → locator string only

        m = REF_KEY_PATTERN.match(ref)

        if m:

            registry_name, key = m.groups()

            return f"@{registry_name}/{key}"


        # ───── registry_object_ref  → fetch value, then recurse

        m = REF_OBJ_PATTERN.match(ref)

        if m:

            registry_name, key = m.groups()

            _seen.add(ref)

            obj = self.get(registry_name, key)   # may itself be a pointer

            return self._resolve_if_pointer(obj, _seen=_seen)


        # not a recognised pointer

        return value


    # ----------------------------------------------------------------

    # 2.  Override public read helpers to auto-resolve

    # ----------------------------------------------------------------

    def get(self, registry_name: str, key: str) -> Optional[Any]:

        builder = self.get_builder(registry_name)

        if not builder:

            return None

        raw = builder.get(key)

        return self._resolve_if_pointer(raw)


    def get_all(self, registry_name: str) -> Optional[Dict[str, Any]]:

        builder = self.get_builder(registry_name)

        if not builder:

            return None

        raw_dict = builder.get_all()

        if raw_dict is None:

            return None

        # resolve pointers for each value in the returned mapping

        return {k: self._resolve_if_pointer(v) for k, v in raw_dict.items()}
    # # Convenience methods for registry operations
    # def get(self, registry_name: str, key: str) -> Optional[Any]:
    #     """Get an item from a registry."""
    #     builder = self.get_builder(registry_name)
    #     if not builder:
    #         return None
    #     return builder.get(key)
    
    # def get_all(self, registry_name: str) -> Optional[Dict[str, Any]]:
    #     """Get all items in a registry."""
    #     builder = self.get_builder(registry_name)
    #     if not builder:
    #         return None
    #     return builder.get_all()
    
    def add(self, registry_name: str, key: str, value: Any) -> bool:
        """Add an item to a registry."""
        builder = self.get_builder(registry_name)
        if not builder:
            return False
        return builder.add(key, value)
    
    # def update(self, registry_name: str, key: str, value: Any) -> bool:
    #     """Update an item in a registry."""
    #     builder = self.get_builder(registry_name)
    #     if not builder:
    #         return False
    #     return builder.update(key, value)
    def update(self, registry_name: str, key: str, value: Any) -> bool:

        """

        Update an item.

        • If the current value stored at (registry_name, key) is a pointer

        string (registry_key_ref / registry_object_ref) we refuse the

        update and tell the caller the locator of the referenced object.

        • Otherwise we perform the update via the builder.

        """

        builder = self.get_builder(registry_name)

        if not builder:

            return False


        current_raw = builder.get(key)


        if isinstance(current_raw, str) and current_raw.startswith(

            ("registry_key_ref=", "registry_object_ref=")

        ):

            # Tell the caller exactly where the pointer leads

            if current_raw.startswith("registry_key_ref="):

                _, payload = current_raw.split("=", 1)           # <registry>:<ref_key>

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


        # safe – value is not a pointer, proceed with normal update

        return builder.update(key, value)
    
    def delete(self, registry_name: str, key: str) -> bool:
        """Delete an item from a registry."""
        builder = self.get_builder(registry_name)
        if not builder:
            return False
        return builder.delete(key)
    
    def list_keys(self, registry_name: str) -> Optional[List[str]]:
        """List all keys in a registry."""
        builder = self.get_builder(registry_name)
        if not builder:
            return None
        return builder.list_keys()