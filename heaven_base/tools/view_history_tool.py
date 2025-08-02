# view_history_tool.py

import json
from typing import Dict, Any, List

from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from ..memory.history import History, get_iteration_view
from langchain_core.messages import BaseMessage, SystemMessage

class ViewHistoryToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'history_id': {
            'name': 'history_id',
            'type': 'str',
            'description': 'The unique ID of the History to view',
            'required': True
        },
        'start': {
            'name': 'start',
            'type': 'int',
            'description': 'Starting iteration index (inclusive)',
            'required': True
        },
        'end': {
            'name': 'end',
            'type': 'int',
            'description': 'Ending iteration index (inclusive)',
            'required': True
        },
        'include_system': {
            'name': 'include_system',
            'type': 'bool',
            'description': 'If true, include SystemMessage entries in each iteration view. Only do this for prompt engineering purposes.',
            'required': False
        }
    }

def view_history_tool_func(
    history_id: str,
    start: int,
    end: int,
    include_system: bool = False
) -> str:
    """
    Retrieve a slice of iterations from a saved History.

    Args:
        history_id (str): ID of the history
        start (int): first iteration index
        end (int): last iteration index
        include_system (bool): whether to include SystemMessage entries

    Returns:
        str: A pretty-printed JSON containing:
          - history_id
          - total_iterations
          - view_range
          - view: { iteration_i: [ {type, content}, ... ], ... }
    """
    # Load the history object
    history = History.load_from_id(history_id)

    # Get the requested iterations slice
    data = get_iteration_view(history, start, end)

    # Serialize messages into JSON-safe dicts
    serialized_view: Dict[str, List[Dict[str, Any]]] = {}
    for key, msgs in data['view'].items():
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
        "view": serialized_view
    }

    # Return pretty JSON
    return json.dumps(output, indent=2)

class ViewHistoryTool(BaseHeavenTool):
    name = "ViewHistoryTool"
    description = (
        "Fetches a consecutive range of iterations from a History by its ID. "
        "Each iteration is the sequence of non-human messages between two human turns. "
        "Set include_system=true to include SystemMessages in the output. Usually false unless prompt engineering or checking if system message is inhibiting intended behavior."
    )
    func = view_history_tool_func
    args_schema = ViewHistoryToolArgsSchema
    is_async = False
