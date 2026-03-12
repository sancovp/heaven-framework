"""
SwarmTool — BaseHeavenTool for launching and managing agent swarms.

Uses CodeAgent actors from sanctuary-revolution for inbox-based messaging.
Each teammate gets a tmux session, an inbox, and heartbeat polling.

Usage:
    from heaven_base.tools.swarm_tool import SwarmTool
    
    tools=[BashTool, SwarmTool, ...]
"""

import json
import os
from typing import Optional
from ..make_heaven_tool_from_docstring import make_heaven_tool_from_docstring


async def manage_swarm(
    action: str,
    config_path: Optional[str] = None,
    swarm_name: Optional[str] = None,
    to: Optional[str] = None,
    message: Optional[str] = None,
    priority: int = 0,
) -> str:
    """Manage agent swarms — launch teams, send messages, check status.

    Each swarm teammate runs as a CodeAgent in its own tmux session with
    an inbox for message passing. Use ConstrainedBashTool with tmux commands
    to interact with teammate terminals directly.

    Args:
        action: Action to perform. One of: 'start' (launch swarm from config), 'stop' (shutdown swarm), 'status' (check all agents), 'list' (list all swarms), 'send' (send inbox message to teammate), 'broadcast' (send to all teammates).
        config_path: Path to swarm config JSON file. Required for 'start' action.
        swarm_name: Name of existing swarm. Required for 'stop', 'status', 'send', 'broadcast'.
        to: Teammate name. Required for 'send' action.
        message: Message content. Required for 'send' and 'broadcast' actions.
        priority: Message priority (higher = more urgent). Default: 0.
    """
    from heaven_base.swarm.runner import SwarmRunner

    if action == "list":
        swarms = SwarmRunner.list_swarms()
        if not swarms:
            return json.dumps({"status": "ok", "swarms": [], "message": "No swarms found."})
        return json.dumps({"status": "ok", "swarms": swarms}, indent=2)

    elif action == "start":
        if not config_path:
            return json.dumps({"status": "error", "error": "config_path is required for 'start' action"})
        try:
            runner = SwarmRunner(config_path)
            runner.start()
            return json.dumps({
                "status": "ok",
                "swarm_name": runner.swarm_name,
                "agents": {
                    name: {"tmux_session": agent.config.tmux_session}
                    for name, agent in runner.agents.items()
                },
                "message": f"Swarm '{runner.swarm_name}' launched successfully.",
            }, indent=2)
        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    elif action == "status":
        if not swarm_name:
            return json.dumps({"status": "error", "error": "swarm_name is required for 'status'"})
        # Load from saved state and check
        swarms = SwarmRunner.list_swarms()
        swarm = next((s for s in swarms if s["swarm_name"] == swarm_name), None)
        if not swarm:
            return json.dumps({"status": "error", "error": f"Swarm '{swarm_name}' not found"})
        return json.dumps({"status": "ok", "swarm": swarm}, indent=2)

    elif action == "stop":
        if not swarm_name:
            return json.dumps({"status": "error", "error": "swarm_name is required for 'stop'"})
        # Load state and kill sessions
        import subprocess
        swarms = SwarmRunner.list_swarms()
        swarm = next((s for s in swarms if s["swarm_name"] == swarm_name), None)
        if not swarm:
            return json.dumps({"status": "error", "error": f"Swarm '{swarm_name}' not found"})
        killed = []
        for name, info in swarm.get("agents", {}).items():
            session = info.get("tmux_session", "")
            result = subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, text=True)
            killed.append({"name": name, "session": session, "killed": result.returncode == 0})
        return json.dumps({"status": "ok", "killed": killed}, indent=2)

    elif action == "send":
        if not swarm_name or not to or not message:
            return json.dumps({"status": "error", "error": "swarm_name, to, and message are required for 'send'"})
        # Write message to agent's inbox file
        from pathlib import Path
        inbox_path = Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data")) / "swarm" / swarm_name / f"{to}_inbox.json"
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        # Append to inbox file
        inbox = []
        if inbox_path.exists():
            try:
                inbox = json.loads(inbox_path.read_text())
            except Exception:
                inbox = []
        inbox.append({"content": message, "priority": priority, "from": "leader"})
        inbox_path.write_text(json.dumps(inbox, indent=2))
        return json.dumps({"status": "ok", "sent_to": to, "inbox_size": len(inbox)})

    elif action == "broadcast":
        if not swarm_name or not message:
            return json.dumps({"status": "error", "error": "swarm_name and message are required for 'broadcast'"})
        swarms = SwarmRunner.list_swarms()
        swarm = next((s for s in swarms if s["swarm_name"] == swarm_name), None)
        if not swarm:
            return json.dumps({"status": "error", "error": f"Swarm '{swarm_name}' not found"})
        from pathlib import Path
        sent = []
        for name in swarm.get("agents", {}).keys():
            if name == swarm.get("leader"):
                continue  # Don't broadcast to self
            inbox_path = Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data")) / "swarm" / swarm_name / f"{name}_inbox.json"
            inbox_path.parent.mkdir(parents=True, exist_ok=True)
            inbox = []
            if inbox_path.exists():
                try:
                    inbox = json.loads(inbox_path.read_text())
                except Exception:
                    inbox = []
            inbox.append({"content": message, "priority": priority, "from": "leader"})
            inbox_path.write_text(json.dumps(inbox, indent=2))
            sent.append(name)
        return json.dumps({"status": "ok", "broadcast_to": sent, "message_length": len(message)})

    else:
        return json.dumps({"status": "error", "error": f"Unknown action: {action}. Use: start, stop, status, list, send, broadcast"})


# Auto-generate BaseHeavenTool
SwarmTool = make_heaven_tool_from_docstring(manage_swarm, tool_name="SwarmTool")
