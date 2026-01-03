"""
Matryoshka Registry Helper - Create nested registry hierarchies with automatic domain tagging.

A matryoshka registry is a pattern where:
- A coordinator registry manages multiple subdomain registries
- Each subdomain has metadata linking it to the parent
- registry_all_ref pointers enable automatic resolution
- Active layer can be swapped dynamically

Usage:
    from heaven_base.registry.matryoshka_helper import create_matryoshka_registry

    create_matryoshka_registry(
        name="capabilities",
        domain="how_do_i",
        seed_subdomains=["default", "success_patterns", "custom"]
    )
"""

from typing import List, Dict, Any, Optional
from .registry_service import RegistryService


def create_matryoshka_registry(
    name: str,
    domain: str,
    seed_subdomains: List[str],
    registry_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a matryoshka registry pattern with automatic domain tagging.

    This creates:
    1. One subdomain registry for each item in seed_subdomains
    2. Each subdomain has _meta with domain tags and parent refs
    3. A coordinator registry that manages all subdomains
    4. Default active layer pointing to first subdomain

    Args:
        name: Base name for the matryoshka (e.g., "capabilities")
        domain: Domain tag for all registries (e.g., "how_do_i")
        seed_subdomains: List of subdomain names (e.g., ["default", "success_patterns", "custom"])
        registry_dir: Optional custom registry directory (defaults to HEAVEN_DATA_DIR/registry)

    Returns:
        Dict with:
            - coordinator: Name of the coordinator registry
            - subdomains: Dict mapping subdomain names to registry names
            - active: Name of the initially active subdomain

    Example:
        >>> result = create_matryoshka_registry(
        ...     name="capabilities",
        ...     domain="how_do_i",
        ...     seed_subdomains=["default", "success_patterns", "custom"]
        ... )
        >>> print(result)
        {
            'coordinator': 'capabilities_matryoshka',
            'subdomains': {
                'default': 'capabilities_default',
                'success_patterns': 'capabilities_success_patterns',
                'custom': 'capabilities_custom'
            },
            'active': 'default'
        }
    """
    service = RegistryService(registry_dir)

    coordinator_name = f"{name}_matryoshka"
    subdomain_registries = {}

    # Step 1: Create all subdomain registries
    for subdomain in seed_subdomains:
        registry_name = f"{name}_{subdomain}"

        # Create the subdomain registry
        if not service.create_registry(registry_name):
            # Registry might already exist, that's okay
            pass

        # Add metadata to subdomain
        service.add(
            registry_name,
            "_meta",
            {
                "domain": domain,
                "subdomain": subdomain,
                "seeded_by": f"registry_key_ref={coordinator_name}:root",
                "parents_of": [domain],
                "matryoshka_name": name
            }
        )

        subdomain_registries[subdomain] = registry_name

    # Step 2: Create coordinator registry
    if not service.create_registry(coordinator_name):
        # Coordinator might already exist, that's okay
        pass

    # Add root metadata to coordinator
    service.add(
        coordinator_name,
        "root",
        {
            "name": name,
            "domain": domain,
            "subdomains": seed_subdomains,
            "description": f"Matryoshka registry for {domain} domain"
        }
    )

    # Add all_layers with registry_all_ref pointers to each subdomain
    all_layers_refs = {
        subdomain: f"registry_all_ref={registry_name}"
        for subdomain, registry_name in subdomain_registries.items()
    }

    service.add(
        coordinator_name,
        "all_layers",
        all_layers_refs
    )

    # Add active layer pointer (default to first subdomain)
    active_subdomain = seed_subdomains[0]
    active_registry = subdomain_registries[active_subdomain]

    service.add(
        coordinator_name,
        "active",
        f"registry_all_ref={active_registry}"
    )

    return {
        "coordinator": coordinator_name,
        "subdomains": subdomain_registries,
        "active": active_subdomain
    }


def switch_active_layer(
    matryoshka_name: str,
    new_active_subdomain: str,
    registry_dir: Optional[str] = None
) -> bool:
    """
    Switch the active layer of a matryoshka registry.

    Args:
        matryoshka_name: Base name of the matryoshka (e.g., "capabilities")
        new_active_subdomain: Subdomain to make active (e.g., "success_patterns")
        registry_dir: Optional custom registry directory

    Returns:
        True if successful, False otherwise

    Example:
        >>> switch_active_layer("capabilities", "success_patterns")
        True
    """
    service = RegistryService(registry_dir)

    coordinator_name = f"{matryoshka_name}_matryoshka"
    new_active_registry = f"{matryoshka_name}_{new_active_subdomain}"

    try:
        # Verify the target subdomain registry exists
        if not service.simple_service.registry_exists(new_active_registry):
            return False

        # Update active pointer
        result = service.update(
            coordinator_name,
            "active",
            f"registry_all_ref={new_active_registry}"
        )

        return result
    except Exception:
        return False


def get_active_layer(
    matryoshka_name: str,
    registry_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get the currently active layer's contents.

    Args:
        matryoshka_name: Base name of the matryoshka (e.g., "capabilities")
        registry_dir: Optional custom registry directory

    Returns:
        Dict with active layer contents (registry_all_ref resolved), or None if not found

    Example:
        >>> contents = get_active_layer("capabilities")
        >>> print(contents)
        {
            '_meta': {...},
            'starlog': {...},
            'giint': {...}
        }
    """
    service = RegistryService(registry_dir)

    coordinator_name = f"{matryoshka_name}_matryoshka"

    # Get active pointer (this will resolve the registry_all_ref!)
    return service.get(coordinator_name, "active")


def list_matryoshka_layers(
    matryoshka_name: str,
    registry_dir: Optional[str] = None
) -> Optional[List[str]]:
    """
    List all subdomain layers in a matryoshka.

    Args:
        matryoshka_name: Base name of the matryoshka (e.g., "capabilities")
        registry_dir: Optional custom registry directory

    Returns:
        List of subdomain names, or None if matryoshka not found

    Example:
        >>> layers = list_matryoshka_layers("capabilities")
        >>> print(layers)
        ['default', 'success_patterns', 'custom']
    """
    service = RegistryService(registry_dir)

    coordinator_name = f"{matryoshka_name}_matryoshka"

    # Get root metadata
    root = service.get(coordinator_name, "root")
    if not root or "subdomains" not in root:
        return None

    return root["subdomains"]


def add_to_matryoshka_layer(
    matryoshka_name: str,
    subdomain: str,
    key: str,
    value: Any,
    registry_dir: Optional[str] = None
) -> bool:
    """
    Add an item to a specific matryoshka layer.

    Args:
        matryoshka_name: Base name of the matryoshka (e.g., "capabilities")
        subdomain: Subdomain to add to (e.g., "success_patterns")
        key: Key for the new item
        value: Value to store
        registry_dir: Optional custom registry directory

    Returns:
        True if successful, False otherwise

    Example:
        >>> add_to_matryoshka_layer(
        ...     "capabilities",
        ...     "custom",
        ...     "my_workflow",
        ...     {"description": "My custom workflow"}
        ... )
        True
    """
    service = RegistryService(registry_dir)

    registry_name = f"{matryoshka_name}_{subdomain}"

    try:
        return service.add(registry_name, key, value)
    except Exception:
        return False
