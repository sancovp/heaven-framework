"""
SwarmRunner v2 — Multi-agent team orchestration via CodeAgent actors.

Uses sanctuary-revolution's CodeAgent (Actor model) for:
- Inbox-based message passing between agents
- tmux sessions per agent
- Heartbeat polling for idle detection
- Priority message queuing

Usage:
    from heaven_base.swarm.runner import SwarmRunner
    runner = SwarmRunner("swarm_config.json")
    runner.start()
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# CodeAgent from sanctuary-revolution
from sanctuary_revolution.harness.core.agent import (
    CodeAgent, CodeAgentConfig,
    InboxMessage, UserPromptMessage, SystemEventMessage,
)


class SwarmRunner:
    """Orchestrates a team of CodeAgent actors from a JSON config.
    
    Each agent gets:
    - Its own tmux session (named: swarm_<swarm_name>_<agent_name>)
    - An inbox for receiving messages
    - A heartbeat for idle detection
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)

        self.swarm_name = self.config["swarm_name"]
        self.leader_config = self.config["leader"]
        self.teammates_config = self.config["teammates"]
        self.agents: Dict[str, CodeAgent] = {}  # {name: CodeAgent}

    def _make_agent(self, name: str, agent_config_name: str, 
                    system_prompt_suffix: str = "", is_leader: bool = False) -> CodeAgent:
        """Create a CodeAgent for a team member."""
        session_name = f"swarm_{self.swarm_name}_{name}"
        
        # Build the command to launch the HEAVEN agent
        agent_command = (
            f"HEAVEN_DATA_DIR={os.environ.get('HEAVEN_DATA_DIR', '/tmp/heaven_data')} "
            f"heaven chat --agent {agent_config_name}"
        )

        config = CodeAgentConfig(
            name=name,
            agent_command=agent_command,
            tmux_session=session_name,
            working_directory=os.getcwd(),
            inbox_poll_interval=5.0,
            max_inbox_size=100,
            state_file=str(
                Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data"))
                / "swarm" / self.swarm_name / f"{name}_inbox.json"
            ),
        )

        agent = CodeAgent(config=config)
        return agent

    def create_agents(self):
        """Create CodeAgent instances for all team members."""
        # Leader
        leader = self._make_agent(
            name=self.leader_config["name"],
            agent_config_name=self.leader_config["agent_config"],
            system_prompt_suffix=self.leader_config.get("system_prompt_suffix", ""),
            is_leader=True,
        )
        self.agents[self.leader_config["name"]] = leader

        # Teammates
        for mate in self.teammates_config:
            agent = self._make_agent(
                name=mate["name"],
                agent_config_name=mate["agent_config"],
                system_prompt_suffix=mate.get("system_prompt_suffix", ""),
            )
            self.agents[mate["name"]] = agent

    def launch_all(self):
        """Create tmux sessions and spawn agents for all team members."""
        for name, agent in self.agents.items():
            agent.create_session()
            agent.spawn_agent()
            print(f"  🚀 Launched '{name}' in tmux:{agent.config.tmux_session}")
            time.sleep(0.5)  # Stagger launches

    def send_message(self, to: str, content: str, priority: int = 0):
        """Send a message to a teammate's inbox."""
        agent = self.agents.get(to)
        if not agent:
            raise ValueError(f"Unknown agent: {to}. Available: {list(self.agents.keys())}")

        msg = UserPromptMessage(content=content, priority=priority)
        agent.enqueue(msg)

    def broadcast(self, content: str, exclude: Optional[str] = None, priority: int = 0):
        """Broadcast a message to all agents (optionally excluding one)."""
        for name, agent in self.agents.items():
            if name != exclude:
                msg = UserPromptMessage(content=content, priority=priority)
                agent.enqueue(msg)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all agents in the swarm."""
        status = {
            "swarm_name": self.swarm_name,
            "leader": self.leader_config["name"],
            "agents": {},
        }
        for name, agent in self.agents.items():
            status["agents"][name] = {
                "session": agent.config.tmux_session,
                "session_exists": agent.session_exists(),
                "inbox_count": agent.inbox_count,
            }
        return status

    def save_state(self):
        """Save swarm state to disk."""
        state_dir = Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data")) / "swarm" / self.swarm_name
        state_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "swarm_name": self.swarm_name,
            "leader": self.leader_config["name"],
            "agents": {
                name: {
                    "tmux_session": agent.config.tmux_session,
                    "agent_command": agent.config.agent_command,
                }
                for name, agent in self.agents.items()
            },
            "teammates": [t["name"] for t in self.teammates_config],
        }

        (state_dir / "swarm_state.json").write_text(json.dumps(state, indent=2))
        print(f"  💾 State saved to {state_dir / 'swarm_state.json'}")

    def shutdown(self, agent_name: Optional[str] = None):
        """Shutdown one or all agents."""
        targets = [agent_name] if agent_name else list(self.agents.keys())
        for name in targets:
            agent = self.agents.get(name)
            if agent and agent.session_exists():
                agent.kill_session()
                print(f"  🛑 Killed '{name}' ({agent.config.tmux_session})")

    def start(self):
        """Full startup: create agents → launch → save state."""
        print(f"🐝 Starting swarm: {self.swarm_name}")
        print(f"   Leader: {self.leader_config['name']}")
        print(f"   Teammates: {[t['name'] for t in self.teammates_config]}")
        print()

        self.create_agents()

        print("📡 Launching agents...")
        self.launch_all()

        self.save_state()

        print(f"\n✅ Swarm '{self.swarm_name}' is running!")
        for name, agent in self.agents.items():
            role = "👑" if name == self.leader_config["name"] else "🤖"
            print(f"   {role} {name}: tmux attach -t {agent.config.tmux_session}")
        print(f"\n   Kill all: heaven swarm stop {self.swarm_name}")
        return True

    @staticmethod
    def list_swarms() -> List[Dict[str, Any]]:
        """List all saved swarms."""
        swarm_dir = Path(os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data")) / "swarm"
        if not swarm_dir.exists():
            return []

        swarms = []
        for d in swarm_dir.iterdir():
            if d.is_dir():
                state_file = d / "swarm_state.json"
                if state_file.exists():
                    try:
                        swarms.append(json.loads(state_file.read_text()))
                    except Exception:
                        continue
        return swarms


def main():
    """CLI entrypoint."""
    if len(sys.argv) < 2:
        print("Usage: python -m heaven_base.swarm.runner <config.json>")
        print("       python -m heaven_base.swarm.runner list")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        swarms = SwarmRunner.list_swarms()
        if not swarms:
            print("No swarms found.")
        else:
            for s in swarms:
                print(f"  🐝 {s['swarm_name']} — leader: {s['leader']}, " 
                      f"agents: {list(s['agents'].keys())}")
    else:
        runner = SwarmRunner(cmd)
        runner.start()


if __name__ == "__main__":
    main()
