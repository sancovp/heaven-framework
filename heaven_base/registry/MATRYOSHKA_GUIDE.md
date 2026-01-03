# Matryoshka Registry Pattern Guide

## What is a Matryoshka Registry?

A matryoshka registry is a hierarchical pattern for organizing related registries:

- **Coordinator Registry**: Manages multiple subdomain registries
- **Subdomain Registries**: Each represents a "layer" (e.g., default, custom, active)
- **Automatic Resolution**: Uses `registry_all_ref` pointers for automatic data resolution
- **Dynamic Switching**: Active layer can be changed at runtime

## Quick Start

```python
from heaven_base.registry import (
    create_matryoshka_registry,
    add_to_matryoshka_layer,
    get_active_layer,
    switch_active_layer,
    list_matryoshka_layers
)

# Create a matryoshka with one function call
result = create_matryoshka_registry(
    name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# Add data to specific layers
add_to_matryoshka_layer(
    "capabilities",
    "default",
    "starlog",
    {"component": "starlog", "help_text": "..."}
)

# Get active layer contents (auto-resolves registry_all_ref)
contents = get_active_layer("capabilities")

# Switch to different layer
switch_active_layer("capabilities", "success_patterns")

# List all available layers
layers = list_matryoshka_layers("capabilities")
```

## How It Works

### Registry Structure Created

```
capabilities_matryoshka (coordinator)
  ├── root: {name, domain, subdomains, description}
  ├── all_layers: {
  │     default: "registry_all_ref=capabilities_default",
  │     success_patterns: "registry_all_ref=capabilities_success_patterns",
  │     custom: "registry_all_ref=capabilities_custom"
  │   }
  └── active: "registry_all_ref=capabilities_default"

capabilities_default
  ├── _meta: {domain, subdomain, seeded_by, parents_of}
  └── <your data>

capabilities_success_patterns
  ├── _meta: {domain, subdomain, seeded_by, parents_of}
  └── <your data>

capabilities_custom
  ├── _meta: {domain, subdomain, seeded_by, parents_of}
  └── <your data>
```

### The Reference Resolution Magic

When you call `get_active_layer("capabilities")`:

1. Queries `capabilities_matryoshka` registry for key "active"
2. Finds value: `"registry_all_ref=capabilities_default"`
3. **Automatically resolves** the `registry_all_ref` pointer
4. Returns the **entire contents** of `capabilities_default` registry

No manual resolution needed - the DSL does it all!

## Domain Tags and Parent Refs

Each subdomain registry has `_meta`:

```python
{
    "domain": "how_do_i",                          # Domain this belongs to
    "subdomain": "default",                        # Which layer this is
    "seeded_by": "registry_key_ref=capabilities_matryoshka:root",  # Parent ref
    "parents_of": ["how_do_i"],                   # Domain hierarchy
    "matryoshka_name": "capabilities"             # Matryoshka it belongs to
}
```

This enables:
- **Domain queries**: Find all registries in a domain
- **Parent tracking**: Know which matryoshka owns this registry
- **Hierarchical organization**: Build domain taxonomies

## Use Cases

### 1. Capability Catalog (how_do_i.seed replacement)

```python
create_matryoshka_registry(
    name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# System components in default
add_to_matryoshka_layer("capabilities", "default", "starlog", {...})

# Learned patterns in success_patterns (from TOOT)
add_to_matryoshka_layer("capabilities", "success_patterns", "write_blog", {...})

# User customizations in custom
add_to_matryoshka_layer("capabilities", "custom", "my_workflow", {...})
```

### 2. Configuration Management

```python
create_matryoshka_registry(
    name="app_config",
    domain="configuration",
    seed_subdomains=["development", "staging", "production"]
)

# Different configs per environment
add_to_matryoshka_layer("app_config", "development", "db_url", "localhost:5432")
add_to_matryoshka_layer("app_config", "production", "db_url", "prod.db:5432")

# Switch environments
switch_active_layer("app_config", "production")
```

### 3. Task Management

```python
create_matryoshka_registry(
    name="tasks",
    domain="project_management",
    seed_subdomains=["planned", "active", "completed"]
)

# Move tasks between layers as status changes
add_to_matryoshka_layer("tasks", "planned", "task_001", {...})
# Later...
add_to_matryoshka_layer("tasks", "active", "task_001", {...})
```

## API Reference

### `create_matryoshka_registry(name, domain, seed_subdomains, registry_dir=None)`

Creates the full matryoshka hierarchy.

**Parameters:**
- `name` (str): Base name (e.g., "capabilities")
- `domain` (str): Domain tag (e.g., "how_do_i")
- `seed_subdomains` (List[str]): Subdomain names (e.g., ["default", "custom"])
- `registry_dir` (Optional[str]): Custom registry directory

**Returns:**
```python
{
    'coordinator': 'capabilities_matryoshka',
    'subdomains': {...},
    'active': 'default'
}
```

### `add_to_matryoshka_layer(matryoshka_name, subdomain, key, value, registry_dir=None)`

Add item to specific layer.

**Parameters:**
- `matryoshka_name` (str): Matryoshka name
- `subdomain` (str): Which layer (e.g., "default")
- `key` (str): Item key
- `value` (Any): Item value
- `registry_dir` (Optional[str]): Custom registry directory

**Returns:** `bool` - Success status

### `get_active_layer(matryoshka_name, registry_dir=None)`

Get active layer contents (auto-resolves `registry_all_ref`).

**Parameters:**
- `matryoshka_name` (str): Matryoshka name
- `registry_dir` (Optional[str]): Custom registry directory

**Returns:** `Dict[str, Any]` - Active layer contents

### `switch_active_layer(matryoshka_name, new_active_subdomain, registry_dir=None)`

Switch which layer is active.

**Parameters:**
- `matryoshka_name` (str): Matryoshka name
- `new_active_subdomain` (str): Subdomain to make active
- `registry_dir` (Optional[str]): Custom registry directory

**Returns:** `bool` - Success status

### `list_matryoshka_layers(matryoshka_name, registry_dir=None)`

List all subdomain layers.

**Parameters:**
- `matryoshka_name` (str): Matryoshka name
- `registry_dir` (Optional[str]): Custom registry directory

**Returns:** `List[str]` - Subdomain names

## Advanced: Manual Registry Access

You can also work with the underlying RegistryService:

```python
from heaven_base.registry import RegistryService

service = RegistryService()

# Direct access to any subdomain
data = service.get_all("capabilities_default")

# Add data directly
service.add("capabilities_custom", "key", "value")

# Use registry_object_ref for cross-registry references
service.add(
    "tasks_planned",
    "depends_on",
    "registry_object_ref=tasks_completed:task_003"
)
```

## Best Practices

1. **Naming Convention**: Use `{name}_{subdomain}` pattern consistently
2. **Domain Tags**: Use clear, hierarchical domain names
3. **Active Layer**: Default to most commonly used layer
4. **_meta**: Always preserve _meta keys in subdomain registries
5. **Switching**: Use `switch_active_layer()` rather than manual updates

## Integration with TOOT

Example: Automatically populate success_patterns layer from TOOT good_job:

```python
def _append_to_matryoshka(name: str, success_data: Dict[str, Any]):
    """Append TOOT success pattern to matryoshka."""
    add_to_matryoshka_layer(
        "capabilities",
        "success_patterns",
        name,
        {
            "type": "success_pattern",
            "domain": success_data["domain"],
            "process": success_data["process"],
            "description": success_data["description"],
            "sequencing": success_data["sequencing"]
        }
    )
```

## Integration with Brain-Agent

Use matryoshka active layer as brain neuron source:

```python
from brain_agent.manager_tools import brain_manager_func

brain_manager_func(
    operation="add",
    brain_id="capabilities_brain",
    name="Capabilities Brain",
    neuron_source_type="registry_keys",
    neuron_source="capabilities_default",  # Or use active layer dynamically
    chunk_max=30000
)
```

## Why Matryoshka?

Like Russian nesting dolls:
- **Outer doll (coordinator)**: Contains and manages inner dolls
- **Inner dolls (subdomains)**: Each a complete entity that can be used independently
- **Nesting**: Hierarchical organization with clear relationships
- **Swappable**: Can change which inner doll is "active"

The pattern enables:
- **Clean separation**: Different concerns in different layers
- **Easy switching**: Change context without moving data
- **Composition**: Combine multiple layers as needed
- **Scalability**: Add new layers without disrupting existing ones
