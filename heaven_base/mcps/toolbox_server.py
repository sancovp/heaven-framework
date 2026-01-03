#!/usr/bin/env python3
"""
HEAVEN Framework Toolbox MCP Server

Exposes HEAVEN Framework tools via MCP interface.
Starting with registry_tool functionality.
"""

import os
import logging
import traceback
from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from pydantic import BaseModel

# Import the utility functions
from ..tools.registry_tool import registry_util_func
from ..tools.matryoshka_registry_tool import matryoshka_registry_util_func
from ..utils.omnitool import omnitool
from ..tools.network_edit_tool import NetworkEditTool
from ..tool_utils.agent_config_test import agent_config_test

# Setup logging
logger = logging.getLogger(__name__)

# Set HEAVEN_DATA_DIR if not already set
if 'HEAVEN_DATA_DIR' not in os.environ:
    os.environ['HEAVEN_DATA_DIR'] = '/tmp/heaven_data'

# Initialize MCP server
mcp = FastMCP("heaven-framework-toolbox")


class OmniToolRequest(BaseModel):
    tool_name: Optional[str] = None
    list_tools: Optional[bool] = False
    get_tool_info: Optional[bool] = False
    parameters: Optional[Dict[str, Any]] = None


@mcp.tool()
async def registry_tool(
    operation: str,
    registry_name: Optional[str] = None,
    key: Optional[str] = None,
    value_str: Optional[str] = None,
    value_dict: Optional[Dict[str, Any]] = None
) -> str:
    """Tool for managing registries in the system.

    The Registry System provides key-value storage for various types of data across the system.
    Each registry is a separate storage container that can hold multiple key-value pairs.

    Operations and their required parameters:

    1. create_registry:
       - Required: registry_name
       - Creates a new registry with the given name

    2. list_registries:
       - No parameters required
       - Returns a list of all available registries

    3. get:
       - Required: registry_name, key
       - Returns the value associated with the key in the specified registry

    4. get_all:
       - Required: registry_name
       - Returns all key-value pairs in the specified registry

    5. add:
       - Required: registry_name, key, value_str OR value_dict
       - Adds a new key-value pair to the specified registry
       - Use value_str for simple string values, value_dict for structured data

    6. update:
       - Required: registry_name, key, value_str OR value_dict
       - Updates an existing key with a new value in the specified registry
       - Use value_str for simple string values, value_dict for structured data

    7. delete:
       - Required: registry_name, key
       - Removes a key-value pair from the specified registry

    8. list_keys:
       - Required: registry_name
       - Returns a list of all keys in the specified registry

    Examples:
    - Create a registry: operation="create_registry", registry_name="my_registry"
    - Add a string item: operation="add", registry_name="my_registry", key="item1", value_str="value1"
    - Add a dict item: operation="add", registry_name="my_registry", key="config", value_dict={"setting": "value"}
    - Get an item: operation="get", registry_name="my_registry", key="item1"

    Registry-reference syntax (any string field can be a pointer):

    registry_key_ref=<registry_name>:<key>

    • Resolves to the locator string "@<registry>/<key>" (the key itself).

    • Use when a parent record just needs to point at another record.

    Example:

    "title": "registry_key_ref=task_registry:T123"

    --> read-time result: "@task_registry/T123"

    registry_object_ref=<registry_name>:<key>#/<optional/json/pointer>

    • Resolves to the value stored at that key (optionally narrowed by a JSON-pointer path).

    • Recursion continues on the returned value until a non-pointer is reached.

    Example:

    "spec": "registry_object_ref=settings_registry:colors#/header/bg"

    --> read-time result: "blue" (assuming that path exists)

    registry_all_ref=<registry_name>

    • Resolves to the entire contents of the specified registry.

    • All values in the returned registry are also resolved if they contain pointers.

    Example:

    "my_data": "registry_all_ref=knowledge_base"

    --> read-time result: {entire contents of knowledge_base registry}

    Rules & behaviour:

    • Write the reference exactly as a plain string—no braces, no quotes inside.

    • Reads (get, get_all) automatically resolve:

    – key-refs return the locator string.

    – object-refs return the full (or sliced) value.

    • Pointers can chain indefinitely; cycles are detected and depth is capped at 99 hops.

    • Update-guard: if you try update() on an entry whose current value is a pointer string, the operation is refused with an error telling you the target locator—modify the referenced registry/key instead, or replace the pointer string explicitly.
    
    Args:
        operation: Operation to perform: create_registry, list_registries, get, get_all, add, update, delete, list_keys
        registry_name: Name of the registry to operate on
        key: Key for get, add, update, delete operations
        value_str: String value for add and update operations. Use this for simple string values.
        value_dict: Dictionary/object value for add and update operations. Use this for structured data.
        
    Returns:
        Result of the operation as a string
    """
    logger.info(f"Registry operation: {operation} on registry: {registry_name}")
    
    return registry_util_func(
        operation=operation,
        registry_name=registry_name,
        key=key,
        value_str=value_str,
        value_dict=value_dict
    )


@mcp.tool()
async def matryoshka_registry_tool(
    operation: str,
    matryoshka_name: Optional[str] = None,
    domain: Optional[str] = None,
    seed_subdomains: Optional[List[str]] = None,
    subdomain: Optional[str] = None,
    key: Optional[str] = None,
    value_str: Optional[str] = None,
    value_dict: Optional[Dict[str, Any]] = None
) -> str:
    """Tool for managing matryoshka (nested/hierarchical) registries.

A matryoshka registry is a pattern for organizing related registries into layers:
- Coordinator registry manages multiple subdomain registries
- Each subdomain represents a "layer" (e.g., default, custom, active)
- Uses registry_all_ref pointers for automatic resolution
- Active layer can be switched dynamically

Operations and their required parameters:

1. create_matryoshka:
   - Required: matryoshka_name, domain, seed_subdomains
   - Creates full matryoshka hierarchy with coordinator and subdomain registries
   - Example: operation="create_matryoshka", matryoshka_name="capabilities",
     domain="how_do_i", seed_subdomains=["default", "success_patterns", "custom"]

2. add_to_layer:
   - Required: matryoshka_name, subdomain, key, (value_str OR value_dict)
   - Adds item to specific subdomain layer
   - Example: operation="add_to_layer", matryoshka_name="capabilities",
     subdomain="default", key="starlog", value_dict={"help": "..."}

3. get_active_layer:
   - Required: matryoshka_name
   - Returns active layer contents (automatically resolves registry_all_ref)
   - Example: operation="get_active_layer", matryoshka_name="capabilities"

4. switch_active_layer:
   - Required: matryoshka_name, subdomain
   - Changes which subdomain is the active layer
   - Example: operation="switch_active_layer", matryoshka_name="capabilities",
     subdomain="success_patterns"

5. list_layers:
   - Required: matryoshka_name
   - Lists all available subdomain layers
   - Example: operation="list_layers", matryoshka_name="capabilities"

6. get_all_layers:
   - Required: matryoshka_name
   - Gets contents of all layers (with registry_all_ref resolved)
   - Example: operation="get_all_layers", matryoshka_name="capabilities"

7. delete_from_layer:
   - Required: matryoshka_name, subdomain, key
   - Deletes item from specific layer
   - Example: operation="delete_from_layer", matryoshka_name="capabilities",
     subdomain="custom", key="my_workflow"

8. list_layer_keys:
   - Required: matryoshka_name, subdomain
   - Lists all keys in specific layer
   - Example: operation="list_layer_keys", matryoshka_name="capabilities",
     subdomain="default"

Use Cases:

1. Capability Catalog with Layers:
   create_matryoshka(name="capabilities", domain="how_do_i",
                    seed_subdomains=["default", "success_patterns", "custom"])

2. Environment-Specific Configuration:
   create_matryoshka(name="config", domain="app_settings",
                    seed_subdomains=["development", "staging", "production"])

3. Task Management by Status:
   create_matryoshka(name="tasks", domain="project",
                    seed_subdomains=["planned", "active", "completed"])

    Args:
        operation: Operation to perform: create_matryoshka, add_to_layer, get_active_layer, switch_active_layer, list_layers, get_all_layers, delete_from_layer, list_layer_keys
        matryoshka_name: Name of the matryoshka registry
        domain: Domain tag for create_matryoshka operation
        seed_subdomains: List of subdomain names for create_matryoshka operation
        subdomain: Specific subdomain layer to operate on
        key: Key for add_to_layer, delete_from_layer operations
        value_str: String value for add_to_layer operation
        value_dict: Dictionary value for add_to_layer operation

    Returns:
        Result of the operation as a string
    """
    logger.info(f"Matryoshka registry operation: {operation} on matryoshka: {matryoshka_name}")

    return matryoshka_registry_util_func(
        operation=operation,
        matryoshka_name=matryoshka_name,
        domain=domain,
        seed_subdomains=seed_subdomains,
        subdomain=subdomain,
        key=key,
        value_str=value_str,
        value_dict=value_dict
    )


@mcp.tool()
async def omni_tool(
    operation: str,
    tool_name: Optional[str] = None,
    parameters: Optional[str] = None
) -> str:
    """Dynamically invoke any tool in `heaven_base.tools` by name with parameters dictionary; can also get_tool_info for tool_name; can also list_tools for all tools in the current HEAVEN build

    Args:
        operation: Operation to perform: list_tools, get_tool_info, or invoke_tool
        tool_name: Name of the tool to invoke (PascalCase or snake_case)
        parameters: JSON string of keyword arguments to pass to the target tool. Must be given if calling a tool
        
    Returns:
        Result of the operation as a string

    Examples:
        List all tools: operation="list_tools"
        Get tool info: operation="get_tool_info", tool_name="BashTool"  
        Invoke tool: operation="invoke_tool", tool_name="ThinkTool", parameters='{"thoughts": "Testing omnitool"}'
    """
    logger.info(f"Omnitool operation: {operation}, tool_name={tool_name}")
    
    # Map operation to omnitool parameters
    if operation == "list_tools":
        result = await omnitool(list_tools=True)
    elif operation == "get_tool_info":
        if not tool_name:
            return "ERROR: get_tool_info requires tool_name parameter"
        result = await omnitool(tool_name=tool_name, get_tool_info=True)
    elif operation == "invoke_tool":
        if not tool_name:
            return "ERROR: invoke_tool requires tool_name parameter"
        
        # Parse parameters JSON string
        params_dict = {}
        if parameters:
            try:
                import json
                params_dict = json.loads(parameters)
            except json.JSONDecodeError:
                import traceback
                return f"ERROR: Invalid JSON in parameters: {traceback.format_exc()}"
        
        result = await omnitool(
            tool_name=tool_name,
            **params_dict
        )
    else:
        return f"ERROR: Unknown operation '{operation}'. Use: list_tools, get_tool_info, or invoke_tool"
    
    return str(result)


@mcp.tool()
async def network_edit_tool(
    command: str,
    path: str,
    command_arguments: Dict[str, Any],
    target_container: Optional[str] = None
) -> str:
    """Custom editing tool for viewing, creating and editing files with optional target container in the docker network. * State is persistent across command calls and discussions with the user
* All command-specific arguments must be provided in the `command_arguments` dictionary
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* The `create` command cannot be used to edit files
* If a command generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* Important directories: 1. `your pwd/core/computer_use_demo/tools/base/tools/` to see tools, 2. `<your pwd>.../base/agents/<specific agent>/memories/...` to view `history_id`s
* Long inputs for create, str_replace, and insert must be chunked into multiple operations (eg. create -> insert; str_replace -> insert; chain of str_replace, etc.)

COMMAND REQUIREMENTS:
When inputting text, the dict must always be properly formatted.
This error `Error in tool 'NetworkEditTool': 'str' object has no attribute 'get'` means the dict was not properly formatted. It must be properly formatted, meaning that the dict itself must ALWAYS be a dict, not a string (ie the curlies of the dict cannot be wrapped in quotes).
Escape Information:
For multiline content in file_text, use \n to indicate line breaks; they will be translated to real newlines.

Inside that JSON string:

• Escape double quotes as \".

• Single quotes need no escaping.

• Backslash must be doubled (\\) to produce a single literal backslash.

str_replace behaves identically: both old_str and new_str are ordinary JSON strings, so same escaping rules apply.

Command Maps:
1. 'view' command:
   - Required: path
   - Optional: view_range (list of 2 integers for line range)

2. 'create' command:
   - Required: path, file_text (cannot be empty - use placeholder word or code comment if needed)

3. 'str_replace' command:
   - Required: path, old_str, new_str

4. 'insert' command:
   - Required: path, new_str, insert_line

5. 'undo_edit' command:
   - Required: path
    """
    logger.info(f"NetworkEditTool operation: command={command}, path={path}, target_container={target_container}")
    
    # Instantiate the NetworkEditTool via LangChain wrapper like omnitool does
    tool = NetworkEditTool.create(adk=False)
    
    # Prepare kwargs for the tool call
    kwargs = {
        'command': command,
        'path': path,
        'command_arguments': command_arguments
    }
    if target_container is not None:
        kwargs['target_container'] = target_container
    
    # Call the tool function
    result = await tool._arun(**kwargs)
    
    # Convert result to string format
    if hasattr(result, 'output'):
        return result.output
    elif hasattr(result, 'error'):
        return f"ERROR: {result.error}"
    else:
        return str(result)


@mcp.tool()
async def agent_config_test_tool(
    test_prompt: str,
    system_prompt: str,
    iterations: int = 1,
    agent_mode: bool = True,
    name: str = "DynamicTestAgent",
    tools: Optional[str] = None,  # JSON string of tool list
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    thinking_budget: Optional[int] = None,
    additional_kws: Optional[str] = None,  # JSON string of additional_kws list
    additional_kw_instructions: str = "",
    known_config_paths: Optional[str] = None,  # JSON string of config paths list
    prompt_suffix_blocks: Optional[str] = None,  # JSON string of suffix blocks list
    max_tool_calls: int = 10,
    orchestrator: bool = False,
    history_id: Optional[str] = None,
    system_prompt_suffix: Optional[str] = None,
    adk: bool = False,
    duo_enabled: bool = False,
    run_on_langchain: bool = False,
    assert_tool_used: Optional[str] = None,
    assert_no_errors: bool = False,
    assert_goal_accomplished: bool = False,
    assert_extracted_keys: Optional[str] = None,  # JSON string of keys list
    assert_extracted_contains: Optional[str] = None,  # JSON string of key-value dict
    assert_min_tool_calls: Optional[int] = None,
    assert_output_contains: Optional[str] = None
) -> str:
    """Test an agent configuration by running it with a test prompt.
    
    Args:
        test_prompt: The prompt to test the agent with
        system_prompt: System prompt for the agent
        iterations: Number of iterations (only used in agent mode)
        agent_mode: Whether to run in agent mode or direct mode
        name: Name for the agent configuration
        tools: JSON string of tool names to include (e.g., '["SafeCodeReaderTool", "BashTool"]')
        provider: AI provider (anthropic, openai, google)
        model: Model name
        temperature: Temperature for AI generation
        max_tokens: Maximum tokens for AI response
        thinking_budget: Thinking budget for reasoning
        additional_kws: JSON string of additional keywords for extraction
        additional_kw_instructions: Instructions for additional keyword extraction
        known_config_paths: JSON string of Hermes config paths for orchestrator mode
        prompt_suffix_blocks: JSON string of prompt suffix block names to append
        max_tool_calls: Maximum number of tool calls allowed
        orchestrator: Whether to run in orchestrator mode with Hermes Switchboard
        history_id: Existing history ID to continue from
        system_prompt_suffix: Additional text to append to the system prompt
        adk: Whether to use Google ADK instead of LangChain
        duo_enabled: Whether to enable DUO prompt injection system
        run_on_langchain: Force LangChain usage even with Google provider
        assert_tool_used: Assert that this specific tool was used
        assert_no_errors: Assert that no tool errors occurred during execution
        assert_goal_accomplished: Assert that the agent marked the goal as accomplished
        assert_extracted_keys: JSON string of keys that must exist in extracted_content
        assert_extracted_contains: JSON string of key-value pairs that must match in extracted_content
        assert_min_tool_calls: Assert minimum number of tool calls made
        assert_output_contains: Assert that final output contains this substring
        
    Returns:
        JSON string with test results including final output, execution info, and assertion results
    """
    logger.info(f"AgentConfigTest: test_prompt={test_prompt[:50]}...")
    
    # Parse JSON string parameters
    import json
    
    tools_list = None
    if tools:
        try:
            tools_list = json.loads(tools)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in tools parameter: {tools}"
    
    additional_kws_list = None
    if additional_kws:
        try:
            additional_kws_list = json.loads(additional_kws)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in additional_kws parameter: {additional_kws}"
    
    known_config_paths_list = None
    if known_config_paths:
        try:
            known_config_paths_list = json.loads(known_config_paths)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in known_config_paths parameter: {known_config_paths}"
    
    prompt_suffix_blocks_list = None
    if prompt_suffix_blocks:
        try:
            prompt_suffix_blocks_list = json.loads(prompt_suffix_blocks)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in prompt_suffix_blocks parameter: {prompt_suffix_blocks}"
    
    assert_extracted_keys_list = None
    if assert_extracted_keys:
        try:
            assert_extracted_keys_list = json.loads(assert_extracted_keys)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in assert_extracted_keys parameter: {assert_extracted_keys}"
    
    assert_extracted_contains_dict = None
    if assert_extracted_contains:
        try:
            assert_extracted_contains_dict = json.loads(assert_extracted_contains)
        except json.JSONDecodeError:
            return f"ERROR: Invalid JSON in assert_extracted_contains parameter: {assert_extracted_contains}"
    
    # Call the agent_config_test function
    try:
        result = await agent_config_test(
            test_prompt=test_prompt,
            system_prompt=system_prompt,
            iterations=iterations,
            agent_mode=agent_mode,
            name=name,
            tools=tools_list,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            additional_kws=additional_kws_list,
            additional_kw_instructions=additional_kw_instructions,
            known_config_paths=known_config_paths_list,
            prompt_suffix_blocks=prompt_suffix_blocks_list,
            max_tool_calls=max_tool_calls,
            orchestrator=orchestrator,
            history_id=history_id,
            system_prompt_suffix=system_prompt_suffix,
            adk=adk,
            duo_enabled=duo_enabled,
            run_on_langchain=run_on_langchain,
            assert_tool_used=assert_tool_used,
            assert_no_errors=assert_no_errors,
            assert_goal_accomplished=assert_goal_accomplished,
            assert_extracted_keys=assert_extracted_keys_list,
            assert_extracted_contains=assert_extracted_contains_dict,
            assert_min_tool_calls=assert_min_tool_calls,
            assert_output_contains=assert_output_contains
        )
        
        # Convert to JSON-serializable format
        def make_serializable(obj):
            if hasattr(obj, '__dict__'):
                # Convert objects with __dict__ to dict
                return {k: make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif hasattr(obj, 'content') and hasattr(obj, 'type'):
                # Handle message objects
                return {"type": str(obj.type), "content": str(obj.content)}
            elif hasattr(obj, '__class__'):
                # Handle other objects by converting to string
                return str(obj)
            else:
                return obj
        
        serializable_result = make_serializable(result)
        return json.dumps(serializable_result, indent=2)
        
    except Exception as e:
        logger.error(f"AgentConfigTest error: {e}")
        return f"ERROR: {e}\n\nTraceback:\n{traceback.format_exc()}"


def main():
    """Entry point for console script."""
    mcp.run()


if __name__ == "__main__":
    main()