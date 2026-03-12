# view_history_tool.py
"""
ViewHistoryTool — Browse, search, and inspect agent conversation histories.

Provides 4 modes:
  1. list_agents  — Show all agents that have saved histories
  2. list_histories — Show all history files for a specific agent (with date filtering)
  3. view — Slice iterations from a specific history by ID (original functionality)
  4. search — Full-text search across history content for a specific agent
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..memory.history import History, get_iteration_view
from ..utils.get_env_value import EnvConfigUtil
from langchain_core.messages import BaseMessage, SystemMessage


def _get_histories_base():
    """Get the base path for agent histories."""
    return os.path.join(EnvConfigUtil.get_heaven_data_dir(), "agents")


def _list_agents_func() -> str:
    """List all agents that have saved history files."""
    base = _get_histories_base()
    if not os.path.exists(base):
        return json.dumps({"error": "No agents directory found", "path": base})

    agents = []
    for agent_dir in sorted(os.listdir(base)):
        hist_dir = os.path.join(base, agent_dir, "memories", "histories")
        if os.path.isdir(hist_dir):
            # Count history files
            count = 0
            latest = None
            for date_dir in os.listdir(hist_dir):
                date_path = os.path.join(hist_dir, date_dir)
                if os.path.isdir(date_path):
                    files = [f for f in os.listdir(date_path) if f.endswith(".json")]
                    count += len(files)
                    if files and (latest is None or date_dir > latest):
                        latest = date_dir

            if count > 0:
                agents.append({
                    "agent": agent_dir,
                    "history_count": count,
                    "latest_date": latest,
                })

    return json.dumps({
        "total_agents": len(agents),
        "agents": agents,
    }, indent=2)


def _list_histories_func(agent_name: str, date_filter: str = None, limit: int = 20) -> str:
    """List history files for a specific agent, optionally filtered by date prefix."""
    base = _get_histories_base()
    hist_dir = os.path.join(base, agent_name, "memories", "histories")

    if not os.path.isdir(hist_dir):
        return json.dumps({"error": f"No histories found for agent '{agent_name}'", "path": hist_dir})

    histories = []
    for date_dir in sorted(os.listdir(hist_dir), reverse=True):
        if date_filter and not date_dir.startswith(date_filter):
            continue
        date_path = os.path.join(hist_dir, date_dir)
        if not os.path.isdir(date_path):
            continue

        for fn in sorted(os.listdir(date_path), reverse=True):
            if not fn.endswith(".json"):
                continue

            fp = os.path.join(date_path, fn)
            history_id = fn.replace(".json", "")

            # Get file size and modification time
            stat = os.stat(fp)
            size_kb = stat.st_size // 1024

            histories.append({
                "history_id": history_id,
                "date": date_dir,
                "size_kb": size_kb,
            })

    # Apply limit
    total = len(histories)
    histories = histories[:limit]

    return json.dumps({
        "agent": agent_name,
        "total_histories": total,
        "showing": len(histories),
        "histories": histories,
    }, indent=2)


def _search_histories_func(
    agent_name: str,
    query: str,
    date_filter: str = None,
    max_results: int = 10,
) -> str:
    """Search through history content for a specific agent.

    Uses simple substring matching on message content.
    Returns matching history IDs with context snippets.
    """
    base = _get_histories_base()
    hist_dir = os.path.join(base, agent_name, "memories", "histories")

    if not os.path.isdir(hist_dir):
        return json.dumps({"error": f"No histories found for agent '{agent_name}'"})

    query_lower = query.lower()
    results = []

    for date_dir in sorted(os.listdir(hist_dir), reverse=True):
        if date_filter and not date_dir.startswith(date_filter):
            continue
        date_path = os.path.join(hist_dir, date_dir)
        if not os.path.isdir(date_path):
            continue

        for fn in sorted(os.listdir(date_path), reverse=True):
            if not fn.endswith(".json") or len(results) >= max_results:
                continue

            fp = os.path.join(date_path, fn)
            history_id = fn.replace(".json", "")

            try:
                with open(fp, "r") as f:
                    data = json.load(f)

                # Search through messages
                matches = []
                messages = data.get("messages", [])
                for i, msg in enumerate(messages):
                    content = ""
                    if isinstance(msg, dict):
                        content = str(msg.get("content", ""))
                    else:
                        content = str(msg)

                    if query_lower in content.lower():
                        # Extract snippet around match
                        idx = content.lower().find(query_lower)
                        start = max(0, idx - 80)
                        end = min(len(content), idx + len(query) + 80)
                        snippet = content[start:end]
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."

                        msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else "unknown"
                        matches.append({
                            "message_index": i,
                            "type": msg_type,
                            "snippet": snippet,
                        })

                if matches:
                    results.append({
                        "history_id": history_id,
                        "date": date_dir,
                        "match_count": len(matches),
                        "matches": matches[:3],  # Show top 3 matches per history
                    })

            except (json.JSONDecodeError, IOError):
                continue

    return json.dumps({
        "agent": agent_name,
        "query": query,
        "total_results": len(results),
        "results": results,
    }, indent=2)


def _view_iterations_func(
    history_id: str,
    start: int,
    end: int,
    include_system: bool = False,
) -> str:
    """Retrieve a slice of iterations from a saved History."""
    history = History.load_from_id(history_id)
    data = get_iteration_view(history, start, end)

    serialized_view: Dict[str, List[Dict[str, Any]]] = {}
    for key, msgs in data["view"].items():
        if include_system:
            filtered = msgs
        else:
            filtered = [msg for msg in msgs if not isinstance(msg, SystemMessage)]
        serialized_view[key] = [
            {"type": type(msg).__name__, "content": msg.content}
            for msg in filtered
        ]

    output = {
        "history_id": data["history_id"],
        "total_iterations": data["total_iterations"],
        "view_range": data["view_range"],
        "view": serialized_view,
    }

    return json.dumps(output, indent=2)


class ViewHistoryToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        "mode": {
            "name": "mode",
            "type": "str",
            "description": (
                "Operation mode. One of: "
                "'list_agents' (show all agents with histories), "
                "'list_histories' (show histories for an agent), "
                "'view' (view iterations from a history), "
                "'search' (search history content for an agent). "
                "Default: 'list_agents'"
            ),
            "required": True,
        },
        "agent_name": {
            "name": "agent_name",
            "type": "str",
            "description": "Agent name. Required for list_histories and search modes.",
            "required": False,
        },
        "history_id": {
            "name": "history_id",
            "type": "str",
            "description": "The unique ID of the History to view. Required for view mode.",
            "required": False,
        },
        "start": {
            "name": "start",
            "type": "int",
            "description": "Starting iteration index (inclusive). Required for view mode.",
            "required": False,
        },
        "end": {
            "name": "end",
            "type": "int",
            "description": "Ending iteration index (inclusive). Required for view mode.",
            "required": False,
        },
        "include_system": {
            "name": "include_system",
            "type": "bool",
            "description": "Include SystemMessage entries in view mode output. Default: false.",
            "required": False,
        },
        "query": {
            "name": "query",
            "type": "str",
            "description": "Search query string. Required for search mode.",
            "required": False,
        },
        "date_filter": {
            "name": "date_filter",
            "type": "str",
            "description": "Date prefix filter (e.g. '2026_03' or '2026_03_05'). Used in list_histories and search modes.",
            "required": False,
        },
        "limit": {
            "name": "limit",
            "type": "int",
            "description": "Max results to return. Default: 20 for list_histories, 10 for search.",
            "required": False,
        },
    }


def view_history_tool_func(
    mode: str = "list_agents",
    agent_name: str = None,
    history_id: str = None,
    start: int = None,
    end: int = None,
    include_system: bool = False,
    query: str = None,
    date_filter: str = None,
    limit: int = None,
) -> str:
    """Unified history browsing tool with 4 modes."""

    if mode == "list_agents":
        return _list_agents_func()

    elif mode == "list_histories":
        if not agent_name:
            return json.dumps({"error": "agent_name is required for list_histories mode"})
        return _list_histories_func(agent_name, date_filter=date_filter, limit=limit or 20)

    elif mode == "view":
        if not history_id:
            return json.dumps({"error": "history_id is required for view mode"})
        if start is None or end is None:
            return json.dumps({"error": "start and end are required for view mode"})
        return _view_iterations_func(history_id, start, end, include_system)

    elif mode == "search":
        if not agent_name:
            return json.dumps({"error": "agent_name is required for search mode"})
        if not query:
            return json.dumps({"error": "query is required for search mode"})
        return _search_histories_func(agent_name, query, date_filter=date_filter, max_results=limit or 10)

    else:
        return json.dumps({"error": f"Unknown mode: {mode}. Use: list_agents, list_histories, view, search"})


class ViewHistoryTool(BaseHeavenTool):
    name = "ViewHistoryTool"
    description = (
        "Browse and search agent conversation histories. Modes: "
        "'list_agents' (discover agents), "
        "'list_histories' (list an agent's histories with optional date filter), "
        "'view' (read specific iterations by history_id), "
        "'search' (full-text search within an agent's histories). "
        "Start with list_agents to discover what's available."
    )
    func = view_history_tool_func
    args_schema = ViewHistoryToolArgsSchema
    is_async = False
