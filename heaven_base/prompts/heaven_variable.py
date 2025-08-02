
import json

import os

from typing import Any, Optional, Dict, Union

from ..registry.registry_service import RegistryService



class HeavenVariable:

    """

    A class that dynamically loads content from a JSON file at runtime.

    

    This allows for dynamic configuration changes without code redeployment.

    The variable's value is loaded from the specified JSON file each time it's accessed.

    """

    

    def __init__(self, json_path: str, key: Optional[str] = None, default: Any = None):

        """

        Initialize a HeavenVariable.

        

        Args:

            json_path: Path to the JSON file to load

            key: Optional key to extract from the JSON. If None, returns the entire JSON content

            default: Default value to return if the file or key doesn't exist

        """

        self.json_path = json_path

        self.key = key

        self.default = default

        self._last_modified = None

        self._cached_value = None

    

    def _load_json(self) -> Dict[str, Any]:

        """Load the JSON file and return its contents."""

        try:

            if not os.path.exists(self.json_path):

                return {}

            

            with open(self.json_path, 'r') as f:

                return json.load(f)

        except Exception as e:

            print(f"Error loading JSON from {self.json_path}: {e}")

            return {}

    

    def _should_reload(self) -> bool:

        """Check if the file has been modified since last load."""

        if not os.path.exists(self.json_path):

            return self._last_modified is not None  # If we had a value before but file is gone

        

        current_mtime = os.path.getmtime(self.json_path)

        if self._last_modified != current_mtime:

            self._last_modified = current_mtime

            return True

        return False

    

    def get_value(self) -> Any:

        """Get the current value from the JSON file."""

        if self._should_reload() or self._cached_value is None:

            json_data = self._load_json()

            

            if self.key is not None:

                # Extract specific key

                self._cached_value = json_data.get(self.key, self.default)

            else:

                # Return entire JSON

                self._cached_value = json_data or self.default

                

        return self._cached_value

    

    def __str__(self) -> str:

        """String representation of the variable's value."""

        value = self.get_value()

        if isinstance(value, (dict, list)):

            return json.dumps(value)

        return str(value)

    

    def __repr__(self) -> str:

        """Representation of the HeavenVariable instance."""

        return f"HeavenVariable(json_path='{self.json_path}', key={self.key}, value={self.get_value()})"

    

    def __call__(self) -> Any:

        """Make the variable callable to get its current value."""





class RegistryHeavenVariable:

    """A variable that dynamically sources its data from the registry system."""

    

    def __init__(self, registry_name: str, key: Optional[str] = None, default: Any = None):

        """

        Initialize a RegistryHeavenVariable.

        

        Args:

            registry_name: Name of the registry to load from

            key: Optional key within the registry. If None, returns the entire registry content

            default: Default value to return if the registry or key doesn't exist

        """

        self.registry_name = registry_name

        self.key = key

        self.default = default

        self._registry_service = RegistryService()

    

    def _load_from_registry(self):

        """Load data from the registry."""

        try:

            if self.key is not None:

                # Get specific key from registry

                value = self._registry_service.get(self.registry_name, self.key)

                return value if value is not None else self.default

            else:

                # Get entire registry

                items = self._registry_service.get_all(self.registry_name)

                return items if items is not None else self.default

        except Exception as e:

            print(f"Error loading from registry {self.registry_name}: {e}")

            return self.default

    

    def get_value(self):

        """Get the current value from the registry."""

        # Always reload from registry to ensure we have the latest data

        return self._load_from_registry()

    

    def __str__(self) -> str:

        """String representation of the variable's value."""

        value = self.get_value()

        if isinstance(value, (dict, list)):

            return json.dumps(value, indent=2)

        return str(value)

    

    def __repr__(self) -> str:

        """Representation of the RegistryHeavenVariable instance."""

        return f"RegistryHeavenVariable(registry='{self.registry_name}', key={self.key})"

    

    def __call__(self) -> Any:

        """Make the variable callable to get its current value."""

        return self.get_value()



# Helper function to ensure registries exist

def ensure_registry_exists(registry_name: str):

    """Create registry if it doesn't exist."""

    service = RegistryService()

    try:

        service.list_keys(registry_name)

    except:

        service.create_registry(registry_name)

    return True



    # return self.get_value()