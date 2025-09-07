"""
HEAVEN Documentation Pipeline

Orchestrates the documentation and example generation process:
DocstringAgent → DeciderAgent → ExampleMakerAgent
"""

import asyncio
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

from .langgraph.hermes_legos import hermes_node, HermesState
from langgraph.graph import StateGraph, END, START
from .agents.docstring_agent import docstring_agent_config
from .agents.decider import decider_agent_config
from .agents.example_maker import example_maker_agent_config


def build_single_hermes_graph():
    """Build a simple graph with a single hermes_node."""
    graph = StateGraph(HermesState)
    graph.add_node("hermes_execute", hermes_node)
    graph.add_edge(START, "hermes_execute") 
    graph.add_edge("hermes_execute", END)
    return graph.compile()


async def run_documentation_pipeline(
    file_or_obj: Union[str, Path], 
    iterations: int = 1,
    skip_examples: bool = False,
    example_priority_filter: str = "HIGH"
) -> Dict[str, Any]:
    """
    Run the complete documentation pipeline on a file or code object.
    
    This orchestrates the three-stage bootstrap process:
    1. DocstringAgent - generates comprehensive Google docstrings
    2. DeciderAgent - determines which components need examples
    3. ExampleMakerAgent - generates meta-pedagogical examples for selected components
    
    Args:
        file_or_obj: Path to Python file or code object to document
        iterations: Number of iterations for the docstring agent
        skip_examples: If True, only run docstring and decider stages
        example_priority_filter: Generate examples only for this priority level and above
            Options: "HIGH", "MEDIUM", "LOW"
            
    Returns:
        Dict containing:
            - original_file: Path to the original file
            - documented_file: File with enhanced docstrings
            - decision_results: Components that need/don't need examples
            - examples_generated: Meta-pedagogical examples (if not skipped)
            - pipeline_summary: Summary of what was processed
            
    Example:
        >>> result = await run_documentation_pipeline(
        ...     "/path/to/heaven_file.py", 
        ...     iterations=2,
        ...     example_priority_filter="HIGH"
        ... )
        >>> print(result['pipeline_summary'])
    """
    
    # Convert to Path object for consistency
    file_path = Path(file_or_obj)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    pipeline_results = {
        "original_file": str(file_path),
        "documented_file": "",
        "decision_results": {},
        "examples_generated": {},
        "pipeline_summary": {}
    }
    
    print(f"🚀 Starting HEAVEN Documentation Pipeline for: {file_path.name}")
    
    # Stage 1: DocstringAgent - Generate comprehensive docstrings
    print(f"📝 Stage 1: Running DocstringAgent (iterations: {iterations})")
    
    docstring_goal = f"""
    Analyze and enhance the Python file: {file_path}
    
    Generate comprehensive Google-style docstrings for all functions, classes, and methods.
    Focus on HEAVEN framework patterns and integration points.
    
    Iterations: {iterations}
    File: {file_path}
    """
    
    # Build and execute LangGraph for docstring generation
    docstring_graph = build_single_hermes_graph()
    initial_state = {
        "messages": [],
        "heaven_events": [],
        "current_goal": docstring_goal,
        "agent_config": docstring_agent_config,
        "extracted_content": {},
        "iteration_count": 0,
        "max_iterations": iterations,
        "hermes_result": None
    }
    
    docstring_result = await docstring_graph.ainvoke(initial_state)
    
    # Extract documented file content from hermes_result
    hermes_result = docstring_result.get("hermes_result", {})
    documented_content = hermes_result.get("formatted_output", "")
    pipeline_results["documented_file"] = documented_content
    
    print(f"✅ Stage 1 Complete: Enhanced docstrings for {file_path.name}")
    
    # Stage 2: DeciderAgent - Determine which components need examples
    print(f"🤔 Stage 2: Running DeciderAgent")
    
    decider_goal = f"""
    Analyze the documented file and decide which components need meta-pedagogical examples.
    
    Focus on HEAVEN framework components, complex patterns, and teaching-worthy code.
    Skip simple utilities and self-explanatory functions.
    
    File content:
    {documented_content[:2000]}...
    """
    
    # Build and execute LangGraph for decider
    decider_graph = build_single_hermes_graph()
    decider_state = {
        "messages": [],
        "heaven_events": [],
        "current_goal": decider_goal,
        "agent_config": decider_agent_config,
        "extracted_content": {},
        "iteration_count": 0,
        "max_iterations": 1,
        "hermes_result": None
    }
    
    decider_result = await decider_graph.ainvoke(decider_state)
    
    # Parse decision results from hermes_result
    decider_hermes_result = decider_result.get("hermes_result", {})
    decision_content = decider_hermes_result.get("formatted_output", "")
    pipeline_results["decision_results"] = decision_content
    
    # Extract components that need examples with specified priority
    needs_examples = _parse_decision_results(decision_content, example_priority_filter)
    
    print(f"✅ Stage 2 Complete: {len(needs_examples)} components flagged for examples")
    
    # Stage 3: ExampleMakerAgent - Generate meta-pedagogical examples
    if not skip_examples and needs_examples:
        print(f"🎯 Stage 3: Running ExampleMakerAgent for {len(needs_examples)} components")
        
        examples = {}
        for component_name, component_info in needs_examples.items():
            example_goal = f"""
            Generate a meta-pedagogical example for the HEAVEN component: {component_name}
            
            Component context: {component_info}
            
            Create an example that shows:
            1. Complete implementation 
            2. Step-by-step usage
            3. HEAVEN framework integration
            4. Teaching notes and meta-learning
            
            Follow the ExampleTool pattern from baseheaventool.py.
            """
            
            # Build and execute LangGraph for example maker
            example_graph = build_single_hermes_graph()
            example_state = {
                "messages": [],
                "heaven_events": [],
                "current_goal": example_goal,
                "agent_config": example_maker_agent_config,
                "extracted_content": {},
                "iteration_count": 0,
                "max_iterations": 1,
                "hermes_result": None
            }
            
            example_result = await example_graph.ainvoke(example_state)
            
            # Extract example content from hermes_result
            example_hermes_result = example_result.get("hermes_result", {})
            examples[component_name] = example_hermes_result.get("formatted_output", "")
            print(f"   📚 Generated example for: {component_name}")
        
        pipeline_results["examples_generated"] = examples
        print(f"✅ Stage 3 Complete: Generated {len(examples)} meta-pedagogical examples")
    
    else:
        if skip_examples:
            print(f"⏭️  Stage 3 Skipped: skip_examples=True")
        else:
            print(f"⏭️  Stage 3 Skipped: No components flagged for examples")
        pipeline_results["examples_generated"] = {}
    
    # Generate pipeline summary
    pipeline_results["pipeline_summary"] = {
        "file_processed": str(file_path),
        "docstring_iterations": iterations,
        "components_analyzed": len(_extract_components_from_decision(decision_content)),
        "examples_generated": len(pipeline_results["examples_generated"]),
        "priority_filter": example_priority_filter,
        "stages_completed": 3 if not skip_examples else 2
    }
    
    print(f"🎉 Pipeline Complete! Summary: {pipeline_results['pipeline_summary']}")
    return pipeline_results


def _parse_decision_results(decision_content: str, priority_filter: str) -> Dict[str, str]:
    """
    Parse DeciderAgent results to extract components that need examples.
    
    Args:
        decision_content: Output from DeciderAgent
        priority_filter: Only include components with this priority or higher
        
    Returns:
        Dict of component_name -> component_info for components needing examples
    """
    priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_priority = priority_order.get(priority_filter, 3)
    
    needs_examples = {}
    current_component = None
    current_info = []
    
    for line in decision_content.split('\n'):
        line = line.strip()
        
        if line.startswith('COMPONENT:'):
            # Save previous component if it needs examples
            if current_component and current_info:
                info_text = '\n'.join(current_info)
                if 'NEEDS_EXAMPLE' in info_text:
                    # Check priority
                    priority = "LOW"  # default
                    if 'PRIORITY: HIGH' in info_text:
                        priority = "HIGH"
                    elif 'PRIORITY: MEDIUM' in info_text:
                        priority = "MEDIUM"
                    
                    if priority_order.get(priority, 1) >= min_priority:
                        needs_examples[current_component] = info_text
            
            # Start new component
            current_component = line.replace('COMPONENT:', '').strip()
            current_info = [line]
            
        elif current_component:
            current_info.append(line)
    
    # Handle last component
    if current_component and current_info:
        info_text = '\n'.join(current_info)
        if 'NEEDS_EXAMPLE' in info_text:
            priority = "LOW"
            if 'PRIORITY: HIGH' in info_text:
                priority = "HIGH"
            elif 'PRIORITY: MEDIUM' in info_text:
                priority = "MEDIUM"
            
            if priority_order.get(priority, 1) >= min_priority:
                needs_examples[current_component] = info_text
    
    return needs_examples


def _extract_components_from_decision(decision_content: str) -> List[str]:
    """Extract all component names from decision results."""
    components = []
    for line in decision_content.split('\n'):
        if line.strip().startswith('COMPONENT:'):
            component = line.replace('COMPONENT:', '').strip()
            components.append(component)
    return components


# Convenience function for common usage
async def document_heaven_file(file_path: str, iterations: int = 1) -> Dict[str, Any]:
    """
    Convenience function to document a single HEAVEN file with sensible defaults.
    
    Args:
        file_path: Path to the Python file to document
        iterations: Number of docstring enhancement iterations
        
    Returns:
        Pipeline results with documented file and generated examples
    """
    return await run_documentation_pipeline(
        file_path,
        iterations=iterations,
        skip_examples=False,
        example_priority_filter="HIGH"
    )