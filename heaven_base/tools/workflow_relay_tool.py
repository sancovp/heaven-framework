from typing import Dict, Any, Optional
import json
from datetime import datetime
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema, ToolResult, CLIResult, ToolError
from ..registry.registry_service import RegistryService
# Lazy import PIS to avoid circular dependencies

def get_relay_pattern(pattern_id: str, relay_args: Dict[str, Any], agent_config=None) -> str:
    """
    Generate the actual prompt using PIS and relay_args
    This function constructs the complex prompt that gets injected via {{ relay_pattern_X }}
    """
    try:
        # Lazy import PIS to avoid circular dependencies
        from ..tool_utils.prompt_injection_system_vX1 import (
            PromptInjectionSystemVX1, 
            PromptInjectionSystemConfigVX1,
            PromptStepDefinitionVX1,
            PromptBlockDefinitionVX1,
            BlockTypeVX1
        )
        pis_available = True
    except ImportError:
        pis_available = False
    
    registry_service = RegistryService()
    relay_registry = registry_service.get_registry('relay_patterns')
    pattern = relay_registry.get(pattern_id)
    
    if not pattern:
        return f"Pattern '{pattern_id}' not found"
    
    # If PIS is available and pattern has PIS config, use it
    if pis_available and 'pis_config' in pattern:
        # TODO: Implement full PIS integration
        # For now, fall back to simple template
        pass
    
    # Simple template-based approach
    prompt_template = pattern.get('prompt', 'Continue with your task.')
    
    # Apply relay_args to template
    try:
        return prompt_template.format(**relay_args)
    except KeyError as e:
        return f"Missing template variable: {e}"
    except Exception as e:
        return f"Template error: {e}"


async def workflow_relay_process(
    thought: str,
    relay_pattern_id: Optional[str] = None,
    display_to_user: Optional[str] = None,
    relay_args: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """Execute the workflow relay process"""
    try:
        # Initialize registry service
        registry_service = RegistryService()
        relay_registry = registry_service.get_or_create_registry('relay_patterns')
        
        # Initialize default patterns if registry is empty
        if len(relay_registry.data) == 0:
            _initialize_default_patterns(relay_registry)
        
        # Build output components
        output_parts = []
        
        # Add thought display if requested
        if display_to_user:
            output_parts.append(f"🧠💭 Thought: {thought}")
        
        # Get relay pattern if specified
        if relay_pattern_id:
            pattern = relay_registry.get(relay_pattern_id)
            
            if pattern is None:
                # Pattern not found - didactic error
                available_patterns = list(relay_registry.data.keys())
                return CLIResult(output=f"❌ Pattern '{relay_pattern_id}' not found. Available patterns: {available_patterns}")
            
            # Check if pattern requires specific args structure
            expected_schema = pattern.get('metadata', {}).get('args_schema')
            
            if expected_schema and not relay_args:
                return CLIResult(output=f"❌ Couldn't process thoughts about that without the correct relay_args structure. Pattern '{relay_pattern_id}' expected a dictionary with this schema: {expected_schema}")
            
            if expected_schema and relay_args:
                # Validate args against schema
                required_keys = expected_schema.get('required', [])
                missing_keys = set(required_keys) - set(relay_args.keys())
                if missing_keys:
                    return CLIResult(output=f"❌ Missing required relay_args for pattern '{relay_pattern_id}'. Expected keys: {required_keys}, got: {list(relay_args.keys())}, missing: {list(missing_keys)}")
            
            # Add thought display
            output_parts.append(f"🧠💭 Thought: {thought}")
            
            # Return the relay pattern reference - the actual prompt will be injected via PIS
            output_parts.append(f"{{ {{ relay_pattern_{relay_pattern_id} }} }}")
            
            # Check lifecycle and update if needed
            lifecycle = pattern.get('metadata', {}).get('lifecycle', -1)
            if lifecycle > 0:
                # Decrement lifecycle
                pattern['metadata']['lifecycle'] = lifecycle - 1
                if lifecycle - 1 == 0:
                    # Pattern expired, move to quarantine
                    output_parts.append(f"📦 Pattern '{relay_pattern_id}' has completed its lifecycle and will be quarantined.")
                    # Move to quarantine registry
                    quarantine_registry = registry_service.get_or_create_registry('relay_patterns_quarantine')
                    quarantine_registry.add(relay_pattern_id, pattern)
                    relay_registry.delete(relay_pattern_id)
                else:
                    # Update pattern with new lifecycle
                    relay_registry.update(relay_pattern_id, pattern)
        else:
            # Default ThinkTool behavior
            output_parts.append("Now that I've thought about it, and since the user won't necessarily provide me any logic themselves unless I use WriteBlockReportTool (and I should therefore try to work by myself with the tools at my disposal)...")
        
        # Join all output parts
        output = "\n\n".join(output_parts)
        
        return CLIResult(output=output)
        
    except Exception as e:
        return ToolResult(
            output="",
            error=f"Error in WorkflowRelayTool: {str(e)}"
        )


def _initialize_default_patterns(registry):
    """Initialize some basic relay patterns if they don't exist"""
    default_patterns = {
        'continue_analysis': {
            'prompt': 'Now analyze the dependencies and structure you found, then create a detailed implementation plan. Be specific about the order of operations.',
            'next_pattern': 'start_implementation',
            'metadata': {'type': 'analysis', 'lifecycle': -1}  # -1 means permanent
        },
        'start_implementation': {
            'prompt': 'Begin implementing the first component from your plan. Use appropriate tools to create or modify files as needed.',
            'next_pattern': None,
            'metadata': {'type': 'implementation', 'lifecycle': -1}
        },
        'debug_loop': {
            'prompt': 'The error encountered was: {error}. Debug the issue, identify the root cause, and fix it. Then use WorkflowRelayTool with pattern "verify_fix" to confirm the solution.',
            'next_pattern': 'verify_fix',
            'metadata': {
                'type': 'error_handling', 
                'lifecycle': -1,
                'args_schema': {
                    'required': ['error'],
                    'optional': ['context', 'file_path'],
                    'description': 'Debug loop pattern for error recovery'
                }
            }
        },
        'verify_fix': {
            'prompt': 'Verify that your fix resolved the issue by running appropriate tests or checks. Report the results.',
            'next_pattern': None,
            'metadata': {'type': 'validation', 'lifecycle': -1}
        },
        'recursive_process': {
            'prompt': 'Process the current item: {item}. If more items remain in your list, use WorkflowRelayTool with pattern "recursive_process" and the next item.',
            'next_pattern': 'recursive_process',
            'metadata': {
                'type': 'iteration', 
                'lifecycle': -1,
                'args_schema': {
                    'required': ['item'],
                    'optional': ['remaining_count', 'total_count'],
                    'description': 'Recursive processing pattern for iterating through lists'
                }
            }
        },
        'decision_branch': {
            'prompt': 'Based on your analysis, decide the next step. If condition A, use pattern "branch_a". If condition B, use pattern "branch_b". Otherwise, use pattern "default_branch".',
            'next_pattern': None,  # Dynamic branching
            'metadata': {'type': 'conditional', 'lifecycle': -1}
        }
    }
    
    for pattern_id, pattern_data in default_patterns.items():
        if pattern_id not in registry.data:
            registry.add(pattern_id, pattern_data)


class WorkflowRelayToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'thought': {
            'name': 'thought',
            'type': 'str',
            'description': 'The thought or reasoning to process (similar to ThinkTool)',
            'required': True
        },
        'relay_pattern_id': {
            'name': 'relay_pattern_id',
            'type': 'str',
            'description': 'ID of the relay pattern to use for the next prompt. If not provided, uses default ThinkTool behavior.',
            'required': False
        },
        'display_to_user': {
            'name': 'display_to_user',
            'type': 'str',
            'description': 'Short 3-5 word summary/description for frontend display (optional)',
            'required': False
        },
        'relay_args': {
            'name': 'relay_args',
            'type': 'dict',
            'description': 'Arguments to pass to the relay pattern (structure depends on pattern)',
            'required': False
        }
    }


class WorkflowRelayTool(BaseHeavenTool):
    name = "WorkflowRelayTool"
    description = """An enhanced ThinkTool that supports programmable relay patterns for creating self-directed agent workflows.
    
## Usage
1. **Basic thinking**: Use without relay_pattern_id for standard ThinkTool behavior
2. **Pattern relay**: Specify relay_pattern_id to get a custom next prompt
3. **Hidden orchestration**: Set display_to_user=False for invisible workflow routing
4. **Dynamic patterns**: Use pattern_params to inject values into pattern templates

## Pattern Structure
Relay patterns are stored in the registry under 'relay_patterns/' and contain:
- `prompt`: The instruction to return to the agent
- `next_pattern`: Optional ID of the next pattern in the chain
- `metadata`: Additional pattern information (lifecycle, author, etc.)

## Example Patterns
- `continue_analysis`: Prompts agent to analyze findings and create a plan
- `debug_loop`: Handles error recovery with context
- `recursive_process`: Patterns that can call themselves for iteration

## Creating Patterns
Use RegistryTool to add patterns:
```
RegistryTool(
    operation='add',
    registry_name='relay_patterns',
    key='my_pattern',
    value_dict={
        'prompt': 'Analyze the results and use WorkflowRelayTool with pattern "next_step"',
        'next_pattern': 'next_step',
        'metadata': {'lifecycle': 1, 'author': 'system'}
    }
)
```
"""
    func = workflow_relay_process
    args_schema = WorkflowRelayToolArgsSchema
    is_async = True