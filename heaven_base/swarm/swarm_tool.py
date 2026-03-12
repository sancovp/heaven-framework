"""
SwarmTool — Team leader's interface for controlling teammates via tmux panes.

The team leader agent gets this tool to:
  - capture(teammate) — read a teammate's pane output
  - send(teammate, message) — send text + enter to a teammate's pane
  - send_all(message) — broadcast to all teammates
  - list() — see all teammates and their pane status
"""

import subprocess
import json
from typing import Dict, Any, Optional

from ..baseheaventool import BaseHeavenTool, ToolArgsSchema


class SwarmToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        "action": {
            "name": "action",
            "type": "str",
            "description": (
                "Action to perform. One of: "
                "'capture' (read teammate's pane output), "
                "'send' (send message to a teammate), "
                "'send_all' (broadcast to all teammates), "
                "'list' (show all teammates and pane status)."
            ),
            "required": True,
        },
        "teammate": {
            "name": "teammate",
            "type": "str",
            "description": "Teammate name. Required for 'capture' and 'send' actions.",
            "required": False,
        },
        "message": {
            "name": "message",
            "type": "str",
            "description": "Message to send. Required for 'send' and 'send_all' actions.",
            "required": False,
        },
        "lines": {
            "name": "lines",
            "type": "int",
            "description": "Number of lines to capture from bottom of pane. Default: 50.",
            "required": False,
        },
    }


def _tmux_cmd(args: list) -> str:
    """Run a tmux command and return stdout."""
    result = subprocess.run(
        ["tmux"] + args,
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip()


def _tmux_send_keys(pane_id: str, text: str, enter: bool = True):
    """Send keys to a tmux pane."""
    subprocess.run(
        ["tmux", "send-keys", "-t", pane_id, text] + (["Enter"] if enter else []),
        capture_output=True, text=True, timeout=10,
    )


def _tmux_capture_pane(pane_id: str, lines: int = 50) -> str:
    """Capture content from a tmux pane."""
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", pane_id, "-p", "-S", f"-{lines}"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout


class SwarmController:
    """Manages the mapping between teammate names and tmux pane IDs.
    
    This is instantiated by the SwarmRunner and injected into the SwarmTool
    via closure so the tool function can resolve teammate names to pane IDs.
    """

    def __init__(self, session_name: str, pane_map: Dict[str, str]):
        """
        Args:
            session_name: tmux session name
            pane_map: {teammate_name: pane_id} mapping
        """
        self.session_name = session_name
        self.pane_map = pane_map  # {"coder": "%1", "analyst": "%2", ...}

    def get_pane_id(self, teammate: str) -> Optional[str]:
        """Resolve teammate name to pane ID."""
        return self.pane_map.get(teammate)

    def get_all_teammates(self) -> list:
        """Get all teammate names."""
        return list(self.pane_map.keys())

    def capture(self, teammate: str, lines: int = 50) -> str:
        """Capture output from a teammate's pane."""
        pane_id = self.get_pane_id(teammate)
        if not pane_id:
            return json.dumps({"error": f"Unknown teammate: {teammate}. Available: {self.get_all_teammates()}"})

        content = _tmux_capture_pane(pane_id, lines)
        return json.dumps({
            "teammate": teammate,
            "pane_id": pane_id,
            "lines_captured": len(content.splitlines()),
            "content": content,
        })

    def send(self, teammate: str, message: str) -> str:
        """Send a message to a teammate's pane."""
        pane_id = self.get_pane_id(teammate)
        if not pane_id:
            return json.dumps({"error": f"Unknown teammate: {teammate}. Available: {self.get_all_teammates()}"})

        _tmux_send_keys(pane_id, message, enter=True)
        return json.dumps({
            "status": "sent",
            "teammate": teammate,
            "pane_id": pane_id,
            "message_length": len(message),
        })

    def send_all(self, message: str) -> str:
        """Broadcast a message to all teammates."""
        results = []
        for name, pane_id in self.pane_map.items():
            _tmux_send_keys(pane_id, message, enter=True)
            results.append({"teammate": name, "pane_id": pane_id, "status": "sent"})

        return json.dumps({
            "status": "broadcast_sent",
            "teammates_reached": len(results),
            "results": results,
        })

    def list_teammates(self) -> str:
        """List all teammates with pane info."""
        teammates = []
        for name, pane_id in self.pane_map.items():
            # Get pane info
            try:
                info = _tmux_cmd([
                    "display-message", "-t", pane_id, "-p",
                    "#{pane_width}x#{pane_height} #{pane_current_command}"
                ])
            except Exception:
                info = "unknown"

            teammates.append({
                "name": name,
                "pane_id": pane_id,
                "info": info,
            })

        return json.dumps({
            "session": self.session_name,
            "teammate_count": len(teammates),
            "teammates": teammates,
        })


def make_swarm_tool_func(controller: SwarmController):
    """Create a closured swarm tool function bound to a specific SwarmController.
    
    This is the 'Closured Tool Pattern' — the controller instance is captured
    in the closure so the tool function can access it without global state.
    """

    def swarm_tool_func(
        action: str,
        teammate: str = None,
        message: str = None,
        lines: int = 50,
    ) -> str:
        """Control teammates in the swarm via tmux panes."""

        if action == "list":
            return controller.list_teammates()

        elif action == "capture":
            if not teammate:
                return json.dumps({"error": "teammate is required for capture action"})
            return controller.capture(teammate, lines=lines)

        elif action == "send":
            if not teammate:
                return json.dumps({"error": "teammate is required for send action"})
            if not message:
                return json.dumps({"error": "message is required for send action"})
            return controller.send(teammate, message)

        elif action == "send_all":
            if not message:
                return json.dumps({"error": "message is required for send_all action"})
            return controller.send_all(message)

        else:
            return json.dumps({
                "error": f"Unknown action: {action}. Use: list, capture, send, send_all"
            })

    return swarm_tool_func


def create_swarm_tool(controller: SwarmController) -> type:
    """Create a SwarmTool class bound to a specific SwarmController.
    
    Returns a class (not instance) that can be passed to HeavenAgentConfig.tools=[].
    """
    func = make_swarm_tool_func(controller)

    class SwarmTool(BaseHeavenTool):
        name = "SwarmTool"
        description = (
            f"Control your team of {len(controller.pane_map)} teammates via tmux panes. "
            f"Teammates: {', '.join(controller.get_all_teammates())}. "
            "Actions: 'list' (show teammates), 'capture' (read pane output), "
            "'send' (message one teammate), 'send_all' (broadcast to all)."
        )
        args_schema = SwarmToolArgsSchema
        is_async = False

    # Bind the closured func
    SwarmTool.func = staticmethod(func)
    return SwarmTool
