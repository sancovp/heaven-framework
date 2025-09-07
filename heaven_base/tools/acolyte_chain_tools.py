"""
Tools generated from acolyte chain functions for the Disciple agent.

Each tool wraps an acolyte chain function to orchestrate multiple acolytes.
"""

from typing import Dict, Any, Optional
import asyncio
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from .acolyte_chains import (
    script_only_chain,
    script_with_configs_chain,
    analysis_improvement_chain,
    full_system_build_chain,
    tool_generation_chain,
    prompt_engineering_chain
)


class ScriptOnlyChainTool(BaseHeavenTool):
    """Tool for generating standalone Python scripts."""
    
    name = "ScriptOnlyChainTool"
    description = """Generate a standalone Python script using the Script Acolyte.
    
    Use this when you need to create a Python script without additional configs or documentation.
    Perfect for simple utilities, one-off scripts, or standalone tools."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'User request describing the script to generate',
                'required': True
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str) -> str:
        result = await script_only_chain(request)
        return f"Generated script: {result['script_name']}\n\n{result['script']}"


class ScriptWithConfigsChainTool(BaseHeavenTool):
    """Tool for generating Python scripts with HermesConfigs."""
    
    name = "ScriptWithConfigsChainTool"
    description = """Generate a Python script and associated HermesConfigs.
    
    Use this when you need a script that will be orchestrated by HEAVEN framework.
    Creates both the implementation and the execution configurations."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'User request describing the script and configs to generate',
                'required': True
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str) -> str:
        result = await script_with_configs_chain(request)
        return f"""Generated script and configs:

SCRIPT ({result['script_name']}):
{result['script']}

HERMESCONFIGS:
{result['hermes_configs']}"""


class AnalysisImprovementChainTool(BaseHeavenTool):
    """Tool for analyzing and improving existing code."""
    
    name = "AnalysisImprovementChainTool"
    description = """Analyze code and generate improvements using specialized acolytes.
    
    Use this to analyze existing code, identify issues, and generate improved versions
    with better patterns, documentation, and structure."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'Description of what to analyze and improve',
                'required': True
            },
            'target_code': {
                'name': 'target_code',
                'type': 'str',
                'description': 'The code to analyze and improve',
                'required': True
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str, target_code: str) -> str:
        result = await analysis_improvement_chain(request, target_code)
        return f"""Analysis and improvements:

ANALYSIS:
{result['analysis']}

IMPROVEMENTS:
{result['improvements']}

IMPROVED CODE:
{result['improved_code']}

DOCUMENTATION:
{result['documentation']}"""


class FullSystemBuildChainTool(BaseHeavenTool):
    """Tool for building complete systems with all components."""
    
    name = "FullSystemBuildChainTool"
    description = """Build a complete system with script, configs, tests, and documentation.
    
    Use this for comprehensive system generation that needs production-ready components
    including implementation, configuration, testing, and documentation."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'High-level description of the system to build',
                'required': True
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str) -> str:
        result = await full_system_build_chain(request)
        return f"""Complete system build:

SCRIPT:
{result['script']}

CONFIGS:
{result['configs']}

TESTS:
{result['tests']}

DOCUMENTATION:
{result['documentation']}

DEPLOYMENT GUIDE:
{result['deployment_guide']}"""


class ToolGenerationChainTool(BaseHeavenTool):
    """Tool for generating new HEAVEN tools."""
    
    name = "ToolGenerationChainTool"
    description = """Generate a new HEAVEN tool with proper structure and documentation.
    
    Creates tools following HEAVEN framework patterns, with support for both
    stateful and stateless tool types."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'Description of the tool to generate',
                'required': True
            },
            'tool_type': {
                'name': 'tool_type',
                'type': 'str',
                'description': 'Either "stateful" or "stateless" tool type',
                'required': False
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str, tool_type: str = "stateless") -> str:
        result = await tool_generation_chain(request, tool_type)
        return f"""Generated tool:

TOOL CODE:
{result['tool_code']}

TOOL TESTS:
{result['tool_tests']}

INTEGRATION GUIDE:
{result['integration_guide']}"""


class PromptEngineeringChainTool(BaseHeavenTool):
    """Tool for generating optimized prompts and prompt blocks."""
    
    name = "PromptEngineeringChainTool"
    description = """Generate optimized prompts and prompt blocks using specialized acolytes.
    
    Creates refined prompts with proper structure, variables, and formatting
    for use in HEAVEN agents."""
    
    class ArgsSchema(ToolArgsSchema):
        arguments: Dict[str, Dict[str, Any]] = {
            'request': {
                'name': 'request',
                'type': 'str',
                'description': 'Description of the prompt to generate',
                'required': True
            },
            'context': {
                'name': 'context',
                'type': 'str',
                'description': 'Optional context about where the prompt will be used',
                'required': False
            }
        }
    
    args_schema = ArgsSchema
    is_async = True
    
    @staticmethod
    async def func(request: str, context: Optional[str] = None) -> str:
        result = await prompt_engineering_chain(request, context)
        return f"""Generated prompt:

PROMPT:
{result['prompt']}

VARIABLES:
{result['variables']}

USAGE GUIDE:
{result['usage_guide']}"""