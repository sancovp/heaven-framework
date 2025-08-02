# write_block_report_tool.py

from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema, ToolFailure
import json
from typing import Dict, Any, List
from datetime import datetime

class WriteBlockReportToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'completed_tasks': {
            'name': 'completed_tasks',
            'type': 'list',
            'description': 'List of tasks that have already been completed by the agent',
            'items': { 'type': 'string' },
            'required': True
        },
        'current_task': {
            'name': 'current_task',
            'type': 'str',
            'description': 'The current task the agent is stuck on and cannot complete',
            'required': True
        },
        'explanation': {
            'name': 'explanation',
            'type': 'str',
            'description': 'A detailed explanation of the problem encountered and why the agent is blocked',
            'required': True
        },
        'blocked_reason': {
            'name': 'blocked_reason',
            'type': 'str',
            'description': 'A `blocked reason` must explain why you believe you are blocked and cannot solve this problem on your own. For example: "Need extra context. Unsure where to find the context required."',
            'required': True
        }
    }

def write_block_report_func(completed_tasks: List[str], current_task: str, explanation: str, blocked_reason: str) -> str:
    """Create a block report when the agent is stuck and needs assistance."""
    report_data = {
        "completed_tasks": completed_tasks,
        "current_task": current_task,
        "explanation": explanation,
        "blocked_reason": blocked_reason,
        "timestamp": datetime.now().isoformat()
    }
    
    # Use a consistent filepath
    filepath = "/tmp/block_report.json"
    
    # Write report to temp file
    with open(filepath, 'w') as f:
        json.dump(report_data, f)
    
    # Return simple message
    return "⚠️ Block Report created!\n🛑 Agent execution halted.\n🗣️ Say this in response: `I've created a block report and am waiting for the help I need`.\nThat will make it clear to the user."

class WriteBlockReportTool(BaseHeavenTool):
    name = "WriteBlockReportTool"
    description = "Creates a Block Report for the system and stops the interaction. Use this tool ONLY when stuck and need help to proceed and want to end the current interaction completely. When writing a Block Report, provide details about what tasks you've completed, what you're stuck on, and why you need assistance. This tool is only for writing Block Reports."
    func = write_block_report_func
    args_schema = WriteBlockReportToolArgsSchema
    is_async = False