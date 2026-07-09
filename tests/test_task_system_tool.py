#!/usr/bin/env python3
"""
Tests for TaskSystemTool's null-arg mitigation.

VERIFIED 2026-06-10 (live minimax probe): the MiniMax anthropic-compat endpoint emits
null for OPTIONAL tool args (heaven marks tasks/task_name optional → they arrive null on
EVERY call), so complete_task/update_tasks used to error and agents perseverated until the
tool-call budget killed them. The mitigation: a task-system call whose required-for-operation
arg is missing/null returns BENIGN CORRECTIVE guidance (telling the agent to ignore the task
system and proceed) instead of an error — so it never raises and the agent stops retrying.

These are deterministic unit tests (no live LLM): they call task_system_func directly,
exactly as the tool-execution seam (BaseHeavenTool._run/_arun) invokes it.
"""

from heaven_base.tools.task_system_tool import (
    task_system_func,
    TASK_SYSTEM_NULL_ARG_GUIDANCE,
    TaskSystemTool,
)


def _is_guidance(result: str) -> bool:
    """The corrective result must tell the agent to stop retrying + ignore the task system."""
    return (
        result == TASK_SYSTEM_NULL_ARG_GUIDANCE
        and "Do NOT retry" in result
        and "Ignore the task system" in result
        and "ERROR" not in result
    )


def test_complete_task_null_task_name_returns_corrective_text_not_error():
    # The minimax arg-drop case: complete_task arrives with task_name=None.
    result = task_system_func(operation="complete_task", task_name=None)
    assert _is_guidance(result), result


def test_complete_task_missing_task_name_returns_corrective_text():
    # Same path when the arg is simply absent (defaults to None).
    result = task_system_func(operation="complete_task")
    assert _is_guidance(result), result


def test_complete_task_does_not_raise_on_null_arg():
    # The whole point: never raise — a raise is what agents perseverate on.
    try:
        result = task_system_func(operation="complete_task", task_name=None)
    except Exception as e:  # pragma: no cover - failure path
        raise AssertionError(f"task_system_func raised on null arg: {e!r}")
    assert isinstance(result, str) and result


def test_update_tasks_null_tasks_returns_corrective_text():
    # The other arg-drop case: update_tasks arrives with tasks=None.
    result = task_system_func(operation="update_tasks", tasks=None)
    assert _is_guidance(result), result


def test_complete_task_with_valid_task_name_still_works():
    # Non-dropped (e.g. required-arg or non-minimax) path is unchanged.
    result = task_system_func(operation="complete_task", task_name="write the report")
    assert result == "Task 'write the report' marked complete."
    assert "ERROR" not in result


def test_update_tasks_with_valid_list_still_works():
    result = task_system_func(operation="update_tasks", tasks=["a", "b"])
    assert result.startswith("Task list updated to 2 tasks")
    assert "Current task: a" in result


def test_goal_accomplished_unchanged():
    result = task_system_func(operation="goal_accomplished")
    assert result == "Goal marked as accomplished. Execution will end."


def test_unknown_operation_still_errors():
    # Scope: the mitigation only covers task-arg-drop, not a genuinely bad operation.
    result = task_system_func(operation="bogus_op")
    assert result.startswith("ERROR: Unknown operation")


def test_tool_run_seam_returns_corrective_text_as_output_not_error():
    """Through the real BaseHeavenTool execution seam: a null-arg complete_task comes back
    as ToolResult.output (corrective guidance), NOT ToolResult.error — so the agent sees
    guidance, not an error it would retry."""
    tool = TaskSystemTool.create(adk=False)
    tool_result = tool._run(operation="complete_task", task_name=None)
    assert tool_result.error is None, f"unexpected error: {tool_result.error}"
    assert _is_guidance(tool_result.output), tool_result.output
