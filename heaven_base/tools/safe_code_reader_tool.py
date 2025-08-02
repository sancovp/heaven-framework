from typing import Dict, Any
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..tool_utils.safe_code_reader import safe_code_read

class SafeCodeReaderToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'file_path': {
            'name': 'file_path',
            'type': 'str',
            'description': 'Path to the Python file to read and clean',
            'required': True
        }
    }

class SafeCodeReaderTool(BaseHeavenTool):
    name = "SafeCodeReaderTool"
    description = "Reads a Python file and strips inline comments while preserving docstrings and function signatures"
    func = safe_code_read
    args_schema = SafeCodeReaderToolArgsSchema
    is_async = False