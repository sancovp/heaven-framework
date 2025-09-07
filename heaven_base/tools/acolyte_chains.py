"""
Acolyte Chain Functions for Disciple Agent

These functions orchestrate different combinations of acolyte agents to 
accomplish complex tasks. Each function becomes a tool for the Disciple agent.
"""

import asyncio
from typing import Dict, Any, Optional
from ..langgraph.foundation import hermes_runner
from ..agents.scripture_writer_agent_config import scripture_writer_agent_config
from ..acolyte_v2.acolyte_agent_config import acolyte_agent_config


async def script_only_chain(request: str) -> Dict[str, Any]:
    """
    Generate a standalone Python script using the Script Acolyte.
    
    Args:
        request: User request describing the script to generate.
        
    Returns:
        Dict containing:
            - script: The generated Python script
            - script_name: Suggested filename for the script
            - description: Brief description of what the script does
            
    Example:
        >>> result = await script_only_chain("Create a script that analyzes Python files")
        >>> print(result['script'])
    """
    # Execute Script Acolyte (ScriptureWriterAgent)
    result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate a Python script for: {request}",
        agent=scripture_writer_agent_config,
        iterations=1
    )
    
    return {
        "script": result.get("formatted_output", ""),
        "script_name": result.get("suggested_filename", "generated_script.py"),
        "description": f"Generated script for: {request}"
    }


async def script_with_configs_chain(request: str) -> Dict[str, Any]:
    """
    Generate a Python script and associated HermesConfigs using multiple acolytes.
    
    This chain first generates a Python script, then creates HermesConfigs
    that can execute or orchestrate that script.
    
    Args:
        request: User request describing the script and configs to generate.
        
    Returns:
        Dict containing:
            - script: The generated Python script
            - script_name: Suggested filename for the script
            - hermes_configs: List of generated HermesConfigs
            - description: Brief description of the complete package
            
    Example:
        >>> result = await script_with_configs_chain("Create code analyzer with configs")
        >>> print(result['script'])
        >>> print(result['hermes_configs'][0])
    """
    # Step 1: Generate script using Script Acolyte
    script_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate a Python script for: {request}",
        agent=scripture_writer_agent_config,
        iterations=1
    )
    
    script_content = script_result.get("formatted_output", "")
    
    # Step 2: Generate HermesConfigs for the script
    config_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate HermesConfigs for this script: {script_content[:500]}... Request: {request}",
        agent=acolyte_agent_config,
        iterations=1
    )
    
    return {
        "script": script_content,
        "script_name": script_result.get("suggested_filename", "generated_script.py"),
        "hermes_configs": config_result.get("formatted_output", ""),
        "description": f"Script and configs for: {request}"
    }


async def analysis_improvement_chain(request: str, target_code: str) -> Dict[str, Any]:
    """
    Analyze code and generate improvements using multiple specialized acolytes.
    
    This chain analyzes existing code, identifies improvements, and generates
    enhanced versions with better patterns and practices.
    
    Args:
        request: Description of what to analyze and improve.
        target_code: The code to analyze and improve.
        
    Returns:
        Dict containing:
            - analysis: Code analysis results
            - improvements: List of suggested improvements
            - improved_code: Enhanced version of the code
            - documentation: Generated documentation
            
    Example:
        >>> code = "def bad_function(): pass"
        >>> result = await analysis_improvement_chain("Improve this function", code)
        >>> print(result['improvements'])
    """
    # Step 1: Analyze code (using specialized analysis acolyte when available)
    analysis_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Analyze this code for issues and patterns: {target_code}",
        agent=acolyte_agent_config,  # TODO: Replace with analysis_acolyte_config
        iterations=1
    )
    
    # Step 2: Generate improvements (using improvement acolyte when available)
    improvement_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Based on analysis: {analysis_result.get('formatted_output', '')[:500]}... Generate improvements for: {request}",
        agent=acolyte_agent_config,  # TODO: Replace with improvement_acolyte_config
        iterations=1
    )
    
    # Step 3: Generate documentation (using documentation acolyte when available)
    doc_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate documentation for the improved code: {improvement_result.get('formatted_output', '')[:500]}",
        agent=acolyte_agent_config,  # TODO: Replace with documentation_acolyte_config
        iterations=1
    )
    
    return {
        "analysis": analysis_result.get("formatted_output", ""),
        "improvements": improvement_result.get("formatted_output", ""),
        "improved_code": improvement_result.get("code", target_code),
        "documentation": doc_result.get("formatted_output", "")
    }


async def full_system_build_chain(request: str) -> Dict[str, Any]:
    """
    Build a complete system with script, configs, tests, and documentation.
    
    This is the most comprehensive chain, orchestrating multiple acolytes
    to create a production-ready system from a single request.
    
    Args:
        request: High-level description of the system to build.
        
    Returns:
        Dict containing:
            - script: Main system script
            - configs: HermesConfigs for execution
            - tests: Generated test suite
            - documentation: Complete system documentation
            - deployment_guide: Instructions for deployment
            
    Example:
        >>> result = await full_system_build_chain("Build a log analysis system")
        >>> print(result['script'])
        >>> print(result['tests'])
    """
    # Step 1: Generate main script
    script_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate main system script for: {request}",
        agent=scripture_writer_agent_config,
        iterations=1
    )
    
    script_content = script_result.get("formatted_output", "")
    
    # Step 2: Generate HermesConfigs
    config_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate HermesConfigs for system: {script_content[:500]}... Request: {request}",
        agent=acolyte_agent_config,
        iterations=1
    )
    
    # Step 3: Generate tests (using test acolyte when available)
    test_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate comprehensive tests for: {script_content[:500]}",
        agent=acolyte_agent_config,  # TODO: Replace with test_acolyte_config
        iterations=1
    )
    
    # Step 4: Generate documentation
    doc_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate complete documentation for system: {request}",
        agent=acolyte_agent_config,  # TODO: Replace with documentation_acolyte_config
        iterations=1
    )
    
    return {
        "script": script_content,
        "configs": config_result.get("formatted_output", ""),
        "tests": test_result.get("formatted_output", ""),
        "documentation": doc_result.get("formatted_output", ""),
        "deployment_guide": f"Deploy system for: {request}"
    }


async def tool_generation_chain(request: str, tool_type: str = "stateless") -> Dict[str, Any]:
    """
    Generate a new HEAVEN tool with proper structure and documentation.
    
    Creates tools following HEAVEN framework patterns, with support for
    both stateful and stateless tool types.
    
    Args:
        request: Description of the tool to generate.
        tool_type: Either "stateful" or "stateless" tool type.
        
    Returns:
        Dict containing:
            - tool_code: Generated tool implementation
            - tool_name: Name of the tool class
            - tool_tests: Test suite for the tool
            - integration_guide: How to integrate the tool
            
    Example:
        >>> result = await tool_generation_chain("Create file search tool", "stateless")
        >>> print(result['tool_code'])
    """
    # Step 1: Generate tool code (using tool acolyte when available)
    tool_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate {tool_type} HEAVEN tool for: {request}",
        agent=acolyte_agent_config,  # TODO: Replace with tool_acolyte_config
        iterations=1
    )
    
    # Step 2: Generate tool tests
    test_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate tests for tool: {tool_result.get('formatted_output', '')[:500]}",
        agent=acolyte_agent_config,  # TODO: Replace with test_acolyte_config
        iterations=1
    )
    
    return {
        "tool_code": tool_result.get("formatted_output", ""),
        "tool_name": tool_result.get("tool_name", "GeneratedTool"),
        "tool_tests": test_result.get("formatted_output", ""),
        "integration_guide": f"Integration guide for {request}"
    }


async def prompt_engineering_chain(request: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate optimized prompts and prompt blocks using specialized acolytes.
    
    Creates refined prompts with proper structure, variables, and formatting
    for use in HEAVEN agents.
    
    Args:
        request: Description of the prompt to generate.
        context: Optional context about where the prompt will be used.
        
    Returns:
        Dict containing:
            - prompt: The generated prompt text
            - prompt_block: WritePromptBlockTool-compatible format
            - variables: List of template variables in the prompt
            - usage_guide: Instructions for using the prompt
            
    Example:
        >>> result = await prompt_engineering_chain("Create code review prompt")
        >>> print(result['prompt'])
    """
    # Generate optimized prompt (using prompt engineering acolyte when available)
    prompt_result = await hermes_runner(
        state={"results": [], "context": {}, "agents": {}},
        goal=f"Generate optimized prompt for: {request}. Context: {context or 'General use'}",
        agent=acolyte_agent_config,  # TODO: Replace with prompt_engineering_acolyte_config
        iterations=1
    )
    
    return {
        "prompt": prompt_result.get("formatted_output", ""),
        "prompt_block": prompt_result.get("prompt_block", {}),
        "variables": prompt_result.get("variables", []),
        "usage_guide": f"Usage guide for prompt: {request}"
    }