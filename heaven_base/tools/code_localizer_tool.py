# code_localizer.py



from typing import Dict, Any
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from ..tool_utils.dependency_finder import analyze_dependencies

class CodeLocalizerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'target_name': {
            'name': 'target_name',
            'type': 'str',
            'description': 'Target function or class to search for',
            'required': True
        },
        'search_dirs': {
            'name': 'search_dirs',
            'type': 'list',
            'description': 'list of absolute directory paths that the search is limited to. Defaults to the core directory of the current codebase and current tmp',
            'required': False
        },
        'contextualizer': {
            'name': 'contextualizer',
            'type': 'bool',
            'description': 'Automatically include file contents for the target and dependencies',
            'required': False
        },
        'exclude_from_contextualizer': {
            'name': 'exclude_from_contextualizer',
            'type': 'list',
            'description': 'List of file names (with extensions) to exclude from contextualizer results (e.g. `some_file_to_exclude.py`)',
            'required': False
        }
    }
        
          
        

class CodeLocalizerTool(BaseHeavenTool):
    name = "CodeLocalizerTool"
    description = """Uses Python AST. Does not work for other code types.
* Provides a precise exploration map of a function or class by returning a dependency chain showing exactly what to look at
* Creates a structured learning sequence because dependencies are presented in logical order
* Ensures minimal context pollution by avoiding necessity to view entire files unnecessarily

CodeLocalizerTool enables this workflow:[
1. An agent (you, this AI reading this) encounters code it needs to modify or extend and calls CodeLocalizerTool
2. The agent views the current code and finds the highest level part of it (the top chain function or class) the agent needs to understand before attempting modification
3. The agent calls the CodeLocalizerTool to get a structured sequence of all cross-file dependencies 
4. The agent systematically views each file and range in the recommended sequence returned by CodeLocalizerTool 
5. With this complete understanding, it can now confidently modify or extend the code
]
    """
    func = analyze_dependencies
    args_schema = CodeLocalizerToolArgsSchema
    is_async = False