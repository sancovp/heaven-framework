# Matryoshka Registry System - Complete Implementation

## What Was Built

A complete hierarchical registry system with:

1. **Matryoshka Helper Functions** (`heaven_base/registry/matryoshka_helper.py`)
2. **Unified Dispatcher** (`heaven_base/registry/matryoshka_dispatcher.py`)
3. **MCP Tool** (`heaven_base/tools/matryoshka_registry_tool.py`)
4. **Comprehensive Documentation** (`heaven_base/registry/MATRYOSHKA_GUIDE.md`)

## Three Ways to Use Matryoshka Registries

### 1. Direct Python API (Programmatic)

```python
from heaven_base.registry import (
    create_matryoshka_registry,
    add_to_matryoshka_layer,
    get_active_layer,
    switch_active_layer,
    list_matryoshka_layers
)

# Create
result = create_matryoshka_registry(
    name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# Add data
add_to_matryoshka_layer("capabilities", "default", "starlog", {...})

# Query
contents = get_active_layer("capabilities")

# Switch
switch_active_layer("capabilities", "success_patterns")
```

### 2. Dispatcher Interface (String-based)

```python
from heaven_base.registry import matryoshka_dispatcher

# Create
result = matryoshka_dispatcher(
    operation="create_matryoshka",
    matryoshka_name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# Add data
result = matryoshka_dispatcher(
    operation="add_to_layer",
    matryoshka_name="capabilities",
    subdomain="default",
    key="starlog",
    value_dict={"help": "..."}
)

# Query
result = matryoshka_dispatcher(
    operation="get_active_layer",
    matryoshka_name="capabilities"
)
```

### 3. MCP Tool (For Agent/LLM Use)

```python
from heaven_base.tools import MatryoshkaRegistryTool

tool = MatryoshkaRegistryTool()

# Create
result = tool.func(
    operation="create_matryoshka",
    matryoshka_name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# Add data
result = tool.func(
    operation="add_to_layer",
    matryoshka_name="capabilities",
    subdomain="default",
    key="starlog",
    value_dict={"component": "starlog", "help_text": "..."}
)
```

## Operations Available

### create_matryoshka
- Creates full hierarchy (coordinator + subdomain registries)
- Each subdomain gets `_meta` with domain tags
- Sets up `registry_all_ref` pointers
- Defaults active layer to first subdomain

**Required:** `matryoshka_name`, `domain`, `seed_subdomains`

### add_to_layer
- Adds item to specific subdomain registry
- Supports both `value_str` and `value_dict`

**Required:** `matryoshka_name`, `subdomain`, `key`, (`value_str` OR `value_dict`)

### get_active_layer
- Returns active layer contents
- Automatically resolves `registry_all_ref` pointer
- Returns full subdomain data including `_meta`

**Required:** `matryoshka_name`

### switch_active_layer
- Changes which subdomain is active
- Updates coordinator's "active" pointer

**Required:** `matryoshka_name`, `subdomain`

### list_layers
- Returns list of all available subdomains

**Required:** `matryoshka_name`

### get_all_layers
- Returns all layer contents (resolved)
- Shows `registry_all_ref` pointers for each layer

**Required:** `matryoshka_name`

### delete_from_layer
- Removes item from specific subdomain

**Required:** `matryoshka_name`, `subdomain`, `key`

### list_layer_keys
- Lists all keys in specific subdomain
- Filters out `_meta` key

**Required:** `matryoshka_name`, `subdomain`

## Registry Structure Created

```
{name}_matryoshka (coordinator)
├── root: {
│     name: "capabilities",
│     domain: "how_do_i",
│     subdomains: ["default", "success_patterns", "custom"],
│     description: "Matryoshka registry for how_do_i domain"
│   }
├── all_layers: {
│     default: "registry_all_ref=capabilities_default",
│     success_patterns: "registry_all_ref=capabilities_success_patterns",
│     custom: "registry_all_ref=capabilities_custom"
│   }
└── active: "registry_all_ref=capabilities_default"

{name}_{subdomain} (e.g., capabilities_default)
├── _meta: {
│     domain: "how_do_i",
│     subdomain: "default",
│     seeded_by: "registry_key_ref=capabilities_matryoshka:root",
│     parents_of: ["how_do_i"],
│     matryoshka_name: "capabilities"
│   }
└── <your data keys>
```

## Domain Tags and Parent Refs

Each subdomain registry has `_meta`:
- **domain**: Which domain this belongs to (e.g., "how_do_i")
- **subdomain**: Which layer this is (e.g., "default")
- **seeded_by**: Registry reference to parent coordinator
- **parents_of**: List of parent domains (hierarchy)
- **matryoshka_name**: Which matryoshka owns this

This enables:
- Domain-based queries
- Parent tracking
- Hierarchical organization
- Multi-level matryoshka nesting

## Use Cases

### 1. Capability Catalog (how_do_i.seed replacement)

```python
# Create matryoshka
create_matryoshka_registry(
    name="capabilities",
    domain="how_do_i",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# System components in default layer
add_to_matryoshka_layer("capabilities", "default", "starlog", {
    "component": "starlog",
    "help_text": "📊 STARLOG: Your project workflow memory"
})

# Learned patterns in success_patterns (from TOOT)
add_to_matryoshka_layer("capabilities", "success_patterns", "write_blog", {
    "type": "success_pattern",
    "domain": "content_creation",
    "process": "blogging"
})

# Query based on need
get_active_layer("capabilities")  # Returns default
switch_active_layer("capabilities", "success_patterns")  # Switch context
```

### 2. Environment Configuration

```python
create_matryoshka_registry(
    name="app_config",
    domain="configuration",
    seed_subdomains=["development", "staging", "production"]
)

# Different settings per environment
add_to_matryoshka_layer("app_config", "development", "db_url", "localhost:5432")
add_to_matryoshka_layer("app_config", "production", "db_url", "prod.db:5432")

# Deploy to production
switch_active_layer("app_config", "production")
config = get_active_layer("app_config")
```

### 3. Task Management

```python
create_matryoshka_registry(
    name="tasks",
    domain="project_management",
    seed_subdomains=["planned", "active", "completed"]
)

# Move tasks through workflow
add_to_matryoshka_layer("tasks", "planned", "task_001", {...})
# Later...
add_to_matryoshka_layer("tasks", "active", "task_001", {...})
# Later...
add_to_matryoshka_layer("tasks", "completed", "task_001", {...})
```

## Integration with Existing Systems

### TOOT Integration

Automatically populate success_patterns layer from good_job:

```python
def _append_to_matryoshka(name: str, success_data: Dict[str, Any]):
    """Append TOOT success pattern to capabilities matryoshka."""
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

### Brain-Agent Integration

Use active layer as brain neuron source:

```python
from brain_agent.manager_tools import brain_manager_func

# Get active layer registry name
# If active is "default", use capabilities_default as neuron source
brain_manager_func(
    operation="add",
    brain_id="capabilities_brain",
    name="Capabilities Brain",
    neuron_source_type="registry_keys",
    neuron_source="capabilities_default",  # Or determine from active layer
    chunk_max=30000
)
```

### SEED System Migration

Replace how_do_i.seed file with matryoshka:

```python
# Instead of reading how_do_i.seed file
# Create matryoshka on first run
create_matryoshka_registry(
    name="how_do_i",
    domain="capabilities",
    seed_subdomains=["default", "success_patterns", "custom"]
)

# Migrate existing .seed entries to default layer
# TOOT automatically populates success_patterns
# Users add to custom layer
```

## File Locations

```
heaven-framework-repo/
├── heaven_base/
│   ├── registry/
│   │   ├── matryoshka_helper.py          # Core helper functions
│   │   ├── matryoshka_dispatcher.py      # Unified dispatcher
│   │   ├── MATRYOSHKA_GUIDE.md           # Detailed guide
│   │   └── __init__.py                   # Exports helpers + dispatcher
│   └── tools/
│       ├── matryoshka_registry_tool.py   # MCP tool wrapper
│       └── __init__.py                   # Exports MatryoshkaRegistryTool
└── MATRYOSHKA_IMPLEMENTATION.md          # This file
```

## Next Steps: Hierarchical Brain-Agent Pattern

The matryoshka pattern enables a powerful brain-agent architecture:

```python
# Coarse matryoshka brain - one neuron per layer
create_coarse_matryoshka_brain(
    matryoshka_name="capabilities",
    brain_id="capabilities_coarse_brain"
)
# Creates brain with neurons:
# - default_neuron: all keys in default layer
# - success_patterns_neuron: all keys in success_patterns layer
# - custom_neuron: all keys in custom layer

# Query routes to relevant layer automatically
query_coarse_brain("capabilities_coarse_brain", "how to write blog")
# Routes to success_patterns_neuron if relevant
```

This enables:
- Layer-aware querying
- Hierarchical context compression
- Dynamic routing based on query type
- Parallel neuron processing per layer

## Benefits

1. **One-function creation**: `create_matryoshka_registry()` sets up everything
2. **Automatic resolution**: `registry_all_ref` handles nested data
3. **Domain organization**: Tags enable hierarchical queries
4. **Three interfaces**: Direct API, dispatcher, MCP tool
5. **Clean separation**: Different concerns in different layers
6. **Dynamic switching**: Change context without moving data
7. **Extensible**: Add new layers without disrupting existing ones
8. **Composable**: Matryoshkas can reference each other

## Summary

The matryoshka registry system provides:
- ✅ Pattern for hierarchical registry organization
- ✅ Helper functions for common operations
- ✅ Unified dispatcher for string-based interface
- ✅ MCP tool for agent/LLM integration
- ✅ Complete documentation and examples
- ✅ Domain tags for taxonomy building
- ✅ Ready for TOOT, Brain-Agent, SEED integration

Next: Implement `coarse_matryoshka_brain` pattern for hierarchical brain-agent architecture.
