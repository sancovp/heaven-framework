# task_system_tool.py

from ..baseheaventool import BaseHeavenTool, ToolResult, CLIResult, ToolError, ToolArgsSchema, ToolFailure
import json
from typing import Dict, Any, List, Optional


class TaskSystemToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'operation': {
            'name': 'operation',
            'type': 'str',
            'description': 'One of: update_tasks, complete_task, goal_accomplished',
            'required': True
        },
        'tasks': {
            'name': 'tasks',
            'type': 'list',
            'description': 'List of task strings. Required for update_tasks operation.',
            'items': {'type': 'string'},
            'required': False
        },
        'task_name': {
            'name': 'task_name',
            'type': 'str',
            'description': 'Name of task to mark complete. Required for complete_task operation.',
            'required': False
        }
    }


# Benign corrective tool result returned when a task-system call arrives with its
# REQUIRED arg missing/null. VERIFIED 2026-06-10 (live minimax probe): the MiniMax
# anthropic-compat endpoint drops/nulls OPTIONAL args whose schema is anyOf[type,null]
# (heaven marks tasks/task_name optional, so they arrive null on EVERY call), even though
# the model's own reasoning contains the value. This is endpoint-side, not a heaven
# build/parse defect (heaven sends a valid schema and faithfully parses the null it gets
# back). Rather than ERROR (which agents perseverate on, burning the tool-call budget),
# return guidance to ignore the task system and proceed with the real goal. Scoped to the
# task-system tool only.
TASK_SYSTEM_NULL_ARG_GUIDANCE = (
    "TASK SYSTEM NON-FUNCTIONAL in this runtime (arguments did not arrive). "
    "Do NOT retry task tools. Ignore the task system entirely and proceed directly "
    "with your goal; finish normally."
)


def task_system_func(operation: str, tasks: list = None, task_name: str = None) -> str:
    """
    Task system CLI. Manages the agent's task list during goal execution.

    Operations:
        update_tasks: Replace the entire task list. Requires 'tasks' list.
        complete_task: Mark a task as done and advance. Requires 'task_name'.
        goal_accomplished: Signal that the goal is fully complete.

    MITIGATION (2026-06-10): when a required arg arrives missing/null (the MiniMax
    endpoint arg-drop, see TASK_SYSTEM_NULL_ARG_GUIDANCE), return benign corrective
    guidance instead of an ERROR so agents stop retrying and go straight to real work.
    """
    if operation == "update_tasks":
        if not tasks or not isinstance(tasks, list):
            return TASK_SYSTEM_NULL_ARG_GUIDANCE
        return f"Task list updated to {len(tasks)} tasks: {tasks}. Current task: {tasks[0]}"

    elif operation == "complete_task":
        if not task_name:
            return TASK_SYSTEM_NULL_ARG_GUIDANCE
        return f"Task '{task_name}' marked complete."

    elif operation == "goal_accomplished":
        return "Goal marked as accomplished. Execution will end."

    else:
        return f"ERROR: Unknown operation '{operation}'. Use: update_tasks, complete_task, goal_accomplished"


class TaskSystemTool(BaseHeavenTool):
    name = "TaskSystemTool"
    description = (
        "Manages your task list during goal execution. Three operations:\n"
        "- update_tasks: Set the full task list. Args: operation='update_tasks', tasks=['task1', 'task2', ...]\n"
        "- complete_task: Mark current task done. Args: operation='complete_task', task_name='the task'\n"
        "- goal_accomplished: Signal goal complete. Args: operation='goal_accomplished'\n"
        "Always use this tool to manage tasks instead of outputting markdown patterns."
    )
    func = task_system_func
    args_schema = TaskSystemToolArgsSchema
    is_async = False
