# straightforwardsummarizer_tool

from ..baseheaventool import BaseHeavenTool, ToolArgsSchema, ToolResult
from typing import Dict, Any, List


class StraightforwardSummarizerToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'summary': {
            'name': 'summary',
            'type': 'str',
            'description': 'Overall summary of the conversation',
            'required': True
        },
        'tasks_completed': {
            'name': 'tasks_completed',
            'type': 'list',
            'description': 'List of tasks that were completed',
            'items': {
                'type': 'dict',
                'properties': {
                    'completed_task': {
                        'type': 'str',
                        'description': 'A completed task'
                    }
                }
            },
            'required': True
        },
        'observations': {
            'name': 'observations',
            'type': 'str',
            'description': 'Key observations about the conversation',
            'required': True
        }
    }

def summarizer_tool_func(summary: str, tasks_completed: List[str], observations: str) -> str:
    return f"""Summary Results:
Overall Summary: {summary}
Tasks Completed: {', '.join(tasks_completed)}
Key Observations: {observations}"""

class StraightforwardSummarizerTool(BaseHeavenTool):
    name = "StraightforwardSummarizerTool"
    description = "Creates a structured summary output with overall summary, completed tasks, and key observations"
    func = summarizer_tool_func
    args_schema = StraightforwardSummarizerToolArgsSchema
    is_async = False