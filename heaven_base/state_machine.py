"""KeywordBasedStateMachine — a Heaven-native agent state machine.

A standalone component you pass to HeavenAgentConfig. Any BaseHeavenAgent
can use it. The state names ARE the additional_kws. The agent outputs
<STATE_NAME>reason</STATE_NAME> to signal a transition. The existing
additional_kws extraction catches it. The state machine inspects
extracted_content after each iteration to detect and execute transitions.

This runs INSIDE the agent's iteration loop — _process_agent_response
fires every iteration. Config swaps take effect on the next iteration.

Higher-level state machines work by having orchestrator agents that
wrap inner agents. It's turtles all the way up.

Usage:
    from heaven_base.state_machine import KeywordBasedStateMachine, StateConfig

    sm = KeywordBasedStateMachine(
        name="coder_sm",
        states={
            "WRITE": StateConfig(
                goal="Write the .🐙 file from the spec.",
                tools=["write_file", "read_file"],
                prompt_suffix="You are writing code.",
            ),
            "TEST": StateConfig(
                goal="Run the test suite.",
                tools=["run_tests"],
                prompt_suffix="Execute tests and report.",
            ),
            "DONE": StateConfig(goal="Return result."),
        },
        initial_state="WRITE",
        terminal_states={"DONE"},
        transitions={
            "WRITE": ["TEST"],
            "TEST": ["WRITE", "DONE"],
        },
    )

    config = HeavenAgentConfig(
        name="octopus_coder",  # REQUIRED — unnamed agents raise
        state_machine=sm,
        ...
    )

Mechanism:
    1. __init__ registers ALL state names as additional_kws
    2. System prompt gets <STATE_MACHINE> block listing states and valid transitions
    3. Agent outputs <ANNEAL>reason</ANNEAL> at end of its work
    4. additional_kws extraction catches "ANNEAL" in extracted_content
    5. State machine checks: is any state name in extracted_content?
    6. If yes and transition is valid → swap config, persist state
    7. Next iteration runs with new goal/tools/prompt

Persistence:
    State is saved to heaven_data/agents/{agent_name}/sm_{name}.json
    on every transition. On agent init, saved state is loaded if it exists.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default heaven data directory
HEAVEN_DATA_DIR = os.environ.get("HEAVEN_DATA_DIR", "/tmp/heaven_data")


@dataclass
class StateConfig:
    """Configuration active during a specific state.

    When the agent transitions to this state, these values override
    the agent's runtime config for the next iteration.
    """
    goal: str = ""                                   # Replaces the agent's goal
    tools: List[str] = field(default_factory=list)   # Tools available in this state
    prompt_suffix: str = ""                          # Injected as prompt suffix
    additional_kws: List[str] = field(default_factory=list)  # Extra kws for this state
    metadata: Dict[str, Any] = field(default_factory=dict)   # Arbitrary per-state data


class KeywordBasedStateMachine:
    """A state machine driven by additional_kws keyword detection.

    State names are registered as additional_kws. The agent outputs
    <STATE_NAME>reason</STATE_NAME> to transition. The existing
    additional_kws extraction system catches it. This class inspects
    the extracted_content dict after each iteration.

    IMPORTANT: The agent MUST be named (not "default" or None).
    Raises ValueError on unnamed agents because state persistence
    requires a unique agent directory.
    """

    UNNAMED_AGENTS = {None, "", "default", "unnamed", "agent"}

    def __init__(
        self,
        name: str,
        states: Dict[str, StateConfig],
        initial_state: str,
        terminal_states: Optional[Set[str]] = None,
        transitions: Optional[Dict[str, List[str]]] = None,
        heaven_data_dir: Optional[str] = None,
    ):
        """Initialize the state machine.

        Args:
            name: Unique name for this state machine
            states: Dict mapping state name → StateConfig
            initial_state: Starting state
            terminal_states: States that signal completion
            transitions: Dict mapping state → list of valid next states.
                         If None, all transitions are valid.
            heaven_data_dir: Override for HEAVEN_DATA_DIR
        """
        if not states:
            raise ValueError(
                "KeywordBasedStateMachine requires at least one state. "
                "Got empty states dict."
            )
        if initial_state not in states:
            raise ValueError(
                f"Initial state '{initial_state}' not in states {set(states.keys())}. "
                f"The initial state must be a key in the states dict."
            )

        self.name = name
        self.states = states
        self.initial_state = initial_state
        self.terminal_states = terminal_states or set()
        self.current_state = initial_state
        self.cycles_completed = 0
        self._heaven_data_dir = heaven_data_dir or HEAVEN_DATA_DIR

        # Validate terminal states
        bad_terminal = self.terminal_states - set(states.keys())
        if bad_terminal:
            raise ValueError(
                f"Terminal states {bad_terminal} not in states {set(states.keys())}. "
                f"All terminal states must be keys in the states dict."
            )

        # Build transition map
        if transitions is not None:
            self._transitions = {k: set(v) for k, v in transitions.items()}
            for from_state, to_states in self._transitions.items():
                if from_state not in states:
                    raise ValueError(
                        f"Transition source '{from_state}' not in states. "
                        f"Valid states: {set(states.keys())}"
                    )
                bad_targets = to_states - set(states.keys())
                if bad_targets:
                    raise ValueError(
                        f"Transition targets {bad_targets} from '{from_state}' not in states. "
                        f"Valid states: {set(states.keys())}"
                    )
        else:
            non_terminal = set(states.keys()) - self.terminal_states
            self._transitions = {
                s: set(states.keys()) - {s}
                for s in non_terminal
            }

        self._history: List[Dict[str, str]] = []

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_terminal(self) -> bool:
        """True if current state is terminal."""
        return self.current_state in self.terminal_states

    @property
    def config(self) -> StateConfig:
        """Get the StateConfig for the current state."""
        return self.states[self.current_state]

    @property
    def valid_transitions(self) -> List[str]:
        """List of valid next states from current state."""
        return sorted(self._transitions.get(self.current_state, set()))

    @property
    def state_keywords(self) -> List[str]:
        """All state names — these become additional_kws."""
        return sorted(self.states.keys())

    # =========================================================================
    # Transitions
    # =========================================================================

    def transition(self, new_state: str, reason: str = "") -> StateConfig:
        """Validate and execute a state transition.

        Returns the StateConfig for the new state.
        Raises ValueError if the transition is invalid.
        """
        if new_state not in self.states:
            raise ValueError(
                f"State '{new_state}' does not exist. "
                f"Valid states: {sorted(self.states.keys())}."
            )

        if self.is_terminal:
            raise ValueError(
                f"Cannot transition from terminal state '{self.current_state}'. "
                f"Terminal states: {sorted(self.terminal_states)}."
            )

        valid = self._transitions.get(self.current_state, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: '{self.current_state}' → '{new_state}'. "
                f"Valid from '{self.current_state}': {sorted(valid)}."
            )

        old_state = self.current_state
        self.current_state = new_state

        self._history.append({
            "from": old_state,
            "to": new_state,
            "reason": reason,
        })

        logger.info(
            "SM '%s': %s → %s (reason: %s)",
            self.name, old_state, new_state, reason or "none",
        )
        return self.config

    def reset(self) -> None:
        """Reset to initial state and increment cycle counter."""
        self.cycles_completed += 1
        self.current_state = self.initial_state
        self._history.clear()

    # =========================================================================
    # KV Processing — called by _process_agent_response
    # =========================================================================

    def process_kvs(self, extracted_content: Dict[str, Any]) -> List[str]:
        """Process extracted keywords and execute any valid transition.

        Called from _process_agent_response after additional_kws extraction.
        Checks if any key in extracted_content is a valid transition target.
        If yes, executes the transition internally.

        The SM does NOT touch the agent. The caller reads sm.config
        after this returns and sets its own goal/tools.

        Args:
            extracted_content: The agent's _current_extracted_content dict.

        Returns:
            List of consumed keys (state name keys that triggered a transition).
            The caller should delete these from the stash.
        """
        if not extracted_content:
            return []

        consumed = []

        # Check valid transitions first
        for target_state in self.valid_transitions:
            if target_state in extracted_content:
                reason = extracted_content[target_state]
                if isinstance(reason, list):
                    reason = reason[-1] if reason else ""
                reason = str(reason)

                try:
                    self.transition(target_state, reason)
                    consumed.append(target_state)
                except ValueError as e:
                    logger.warning("SM '%s': transition failed: %s", self.name, e)

                # Only one transition per call
                break

        # Warn about invalid transition attempts (don't consume them)
        if not consumed:
            for state_name in self.states:
                if state_name in extracted_content and state_name != self.current_state:
                    logger.warning(
                        "SM '%s': agent output <%s> but %s → %s is not valid. "
                        "Valid from '%s': %s",
                        self.name, state_name, self.current_state, state_name,
                        self.current_state, self.valid_transitions,
                    )

        return consumed

    # =========================================================================
    # Prompt Injection — system prompt block
    # =========================================================================

    def build_kw_instructions(self) -> str:
        """Build the additional_kw_instructions string.

        This tells the agent WHAT the keywords mean and HOW to use them.
        Gets merged with any existing additional_kw_instructions.
        """
        current = self.current_state
        valid = self.valid_transitions
        terminal = sorted(self.terminal_states)

        state_list = []
        for name, cfg in sorted(self.states.items()):
            markers = []
            if name == current:
                markers.append("CURRENT")
            if name in self.terminal_states:
                markers.append("TERMINAL")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            state_list.append(f"  {name}: {cfg.goal}{marker_str}")

        return (
            f"STATE MACHINE '{self.name}':\n"
            f"You are in state: {current}\n"
            f"Valid next states: {valid}\n"
            f"Cycles completed: {self.cycles_completed}\n\n"
            "States:\n" + "\n".join(state_list) + "\n\n"
            "When you complete your current phase, output the NEXT STATE as an XML tag "
            "at the END of your response. The tag content should explain WHY you're transitioning.\n\n"
            "Example: <ANNEAL>I've finished writing the code, ready to anneal</ANNEAL>\n\n"
            f"Terminal states ({terminal}) signal that you're DONE. "
            "Only transition to a terminal state when the work is truly complete."
        )

    def build_transition_prompt(self) -> str:
        """Generate the full STATE_MACHINE prompt section.

        Injected into the system prompt at agent init time.
        """
        return f"<STATE_MACHINE>\n{self.build_kw_instructions()}\n</STATE_MACHINE>"

    # =========================================================================
    # Persistence
    # =========================================================================

    def _state_file_path(self, agent_name: str) -> Path:
        """Get the persistence path for this state machine."""
        return Path(self._heaven_data_dir) / "agents" / agent_name / f"sm_{self.name}.json"

    def save_state(self, agent_name: str) -> None:
        """Save current state to disk.

        Raises ValueError if agent_name is unnamed/default.
        """
        if agent_name in self.UNNAMED_AGENTS:
            raise ValueError(
                f"Cannot save state for unnamed agent '{agent_name}'. "
                f"KeywordBasedStateMachine requires a named agent. "
                f"Set a unique name on HeavenAgentConfig "
                f"(not: {sorted(self.UNNAMED_AGENTS, key=str)})."
            )

        path = self._state_file_path(agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.name,
            "current_state": self.current_state,
            "initial_state": self.initial_state,
            "cycles_completed": self.cycles_completed,
            "history": self._history,
        }

        path.write_text(json.dumps(data, indent=2))
        logger.debug("SM '%s': saved to %s", self.name, path)

    def load_state(self, agent_name: str) -> bool:
        """Load state from disk if it exists. Returns True if loaded."""
        if agent_name in self.UNNAMED_AGENTS:
            return False

        path = self._state_file_path(agent_name)
        if not path.exists():
            return False

        try:
            data = json.loads(path.read_text())
            saved_state = data.get("current_state", self.initial_state)

            if saved_state in self.states:
                self.current_state = saved_state
                self.cycles_completed = data.get("cycles_completed", 0)
                self._history = data.get("history", [])
                logger.info("SM '%s': loaded state '%s' (cycle %d) from %s", self.name, saved_state, self.cycles_completed, path)
                return True
            else:
                logger.warning("SM '%s': saved state '%s' invalid, starting fresh", self.name, saved_state)
                return False
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("SM '%s': failed to load from %s: %s", self.name, path, e)
            return False

    def clear_state(self, agent_name: str) -> None:
        """Delete saved state file."""
        path = self._state_file_path(agent_name)
        if path.exists():
            path.unlink()
            logger.info("SM '%s': cleared saved state", self.name)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "current_state": self.current_state,
            "initial_state": self.initial_state,
            "cycles_completed": self.cycles_completed,
            "terminal_states": sorted(self.terminal_states),
            "states": {
                name: {
                    "goal": cfg.goal,
                    "tools": cfg.tools,
                    "prompt_suffix": cfg.prompt_suffix,
                    "additional_kws": cfg.additional_kws,
                    "metadata": cfg.metadata,
                }
                for name, cfg in self.states.items()
            },
            "transitions": {
                k: sorted(v) for k, v in self._transitions.items()
            },
            "history": self._history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeywordBasedStateMachine":
        """Deserialize from dict."""
        states = {
            name: StateConfig(**cfg)
            for name, cfg in data["states"].items()
        }
        sm = cls(
            name=data["name"],
            states=states,
            initial_state=data["initial_state"],
            terminal_states=set(data.get("terminal_states", [])),
            transitions=data.get("transitions"),
        )
        sm.current_state = data.get("current_state", data["initial_state"])
        sm.cycles_completed = data.get("cycles_completed", 0)
        sm._history = data.get("history", [])
        return sm

    @classmethod
    def from_json_file(cls, path: str) -> "KeywordBasedStateMachine":
        """Load SM config from a JSON file.

        The JSON structure matches to_dict() output:
        {
            "name": "coder",
            "initial_state": "WRITE",
            "terminal_states": ["DONE"],
            "states": {
                "WRITE": {"goal": "Write code", "tools": ["write_file"]},
                "TEST":  {"goal": "Run tests",  "tools": ["run_tests"]},
                "DONE":  {"goal": "Complete"}
            },
            "transitions": {
                "WRITE": ["TEST"],
                "TEST": ["WRITE", "DONE"]
            }
        }
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SM config not found: {path}")
        data = json.loads(p.read_text())
        return cls.from_dict(data)

    def to_json_file(self, path: str) -> None:
        """Save SM config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    # =========================================================================
    # Mermaid
    # =========================================================================

    def to_mermaid(self) -> str:
        """Generate mermaid state diagram."""
        lines = ["stateDiagram-v2"]
        lines.append(f"    [*] --> {self.initial_state}")

        for from_state, to_states in sorted(self._transitions.items()):
            for to_state in sorted(to_states):
                lines.append(f"    {from_state} --> {to_state}")

        for ts in sorted(self.terminal_states):
            lines.append(f"    {ts} --> [*]")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"KeywordBasedStateMachine('{self.name}', "
            f"state='{self.current_state}', "
            f"states={sorted(self.states.keys())}, "
            f"terminal={sorted(self.terminal_states)})"
        )
