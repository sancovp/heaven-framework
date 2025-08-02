import sys
import uuid
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)
from pydantic import BaseModel, Field
from .base_piece import BasePiece
from .history import History, AgentStatus
from .heaven_event import HeavenEvent, is_heaven_event_dict, convert_langchain_to_heaven, convert_heaven_to_langchain
from ..utils.name_utils import normalize_agent_name

# Define SystemPrompt class for HEAVEN events
class SystemPrompt(BaseModel):
    """Represents a system prompt in the HEAVEN system.
    
    System prompts provide instructions, context, and guidelines to agents.
    They influence the agent's behavior and responses.
    """
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_langchain(self) -> SystemMessage:
        """Convert to LangChain SystemMessage."""
        return SystemMessage(content=self.content, additional_kwargs=self.metadata)
    
    @classmethod
    def from_langchain(cls, message: SystemMessage) -> 'SystemPrompt':
        """Create from LangChain SystemMessage."""
        return cls(
            content=message.content,
            metadata=message.additional_kwargs
        )
    
    def to_heaven_event(self) -> Dict[str, Any]:
        """Convert to HEAVEN event format."""
        return HeavenEvent(
            event_type="SYSTEM_MESSAGE",
            data={
                "content": self.content,
                "metadata": self.metadata
            }
        ).to_dict()
    
    @classmethod
    def from_heaven_event(cls, event: Dict[str, Any]) -> Optional['SystemPrompt']:
        """Create from HEAVEN event."""
        if not is_heaven_event_dict(event) or event.get("event_type") != "SYSTEM_MESSAGE":
            return None
        
        data = event.get("data", {})
        return cls(
            content=data.get("content", ""),
            metadata=data.get("metadata", {})
        )

class HeavenHistory(History):
    """Enhanced History class that supports both LangChain messages and HEAVEN events"""
    # We keep the original messages field for backward compatibility
    messages: List[BaseMessage]
    # Add a new field for HEAVEN events
    heaven_events: List[Dict[str, Any]] = Field(default_factory=list)
    # Add a field for system prompts
    system_prompts: List[SystemPrompt] = Field(default_factory=list)
    # Flag to track the primary storage format
    primary_format: Literal["langchain", "heaven"] = "langchain"
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize heaven_events if not provided
        if not hasattr(self, "heaven_events") or not self.heaven_events:
            # Convert existing messages to HEAVEN events
            self.heaven_events = convert_langchain_to_heaven(self.messages)
        
        # Initialize system_prompts if not provided
        if not hasattr(self, "system_prompts") or not self.system_prompts:
            # Extract system messages and convert to SystemPrompt objects
            system_messages = [msg for msg in self.messages if isinstance(msg, SystemMessage)]
            self.system_prompts = [SystemPrompt.from_langchain(msg) for msg in system_messages]
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a LangChain message to the history."""
        self.messages.append(message)
        
        # Also add as HEAVEN event
        heaven_event = HeavenEvent.from_langchain_message(message).to_dict()
        self.heaven_events.append(heaven_event)
        
        # If it's a system message, also add to system_prompts
        if isinstance(message, SystemMessage):
            system_prompt = SystemPrompt.from_langchain(message)
            self.system_prompts.append(system_prompt)
    
    def add_heaven_event(self, event: Union[Dict[str, Any], HeavenEvent]) -> None:
        """Add a HEAVEN event to the history."""
        # Convert to dict if it's a HeavenEvent object
        if isinstance(event, HeavenEvent):
            event_dict = event.to_dict()
        else:
            event_dict = event
        
        # Add to heaven_events
        self.heaven_events.append(event_dict)
        
        # Check if it's a system message event
        if is_heaven_event_dict(event_dict) and event_dict.get("event_type") == "SYSTEM_MESSAGE":
            # Create a SystemPrompt from the event
            system_prompt = SystemPrompt.from_heaven_event(event_dict)
            if system_prompt:
                self.system_prompts.append(system_prompt)
        
        # Convert to LangChain message if possible and add to messages
        if is_heaven_event_dict(event_dict):
            heaven_event = HeavenEvent.from_dict(event_dict)
            message = heaven_event.to_langchain_message()
            if message:
                self.messages.append(message)
    
    def get_heaven_events(self) -> List[Dict[str, Any]]:
        """Get all HEAVEN events in the history."""
        return self.heaven_events
    
    def get_langchain_messages(self) -> List[BaseMessage]:
        """Get all LangChain messages in the history."""
        return self.messages
    
    def get_system_prompts(self) -> List[SystemPrompt]:
        """Get all system prompts in the history."""
        return self.system_prompts
    
    def get_combined_system_prompt(self) -> str:
        """Get all system prompts combined into a single string."""
        return "\n\n".join([sp.content for sp in self.system_prompts if sp.content])
    
    def add_system_prompt(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a system prompt to the history."""
        # Create SystemPrompt object
        system_prompt = SystemPrompt(content=content, metadata=metadata or {})
        self.system_prompts.append(system_prompt)
        
        # Add as LangChain message
        self.messages.append(system_prompt.to_langchain())
        
        # Add as HEAVEN event
        self.heaven_events.append(system_prompt.to_heaven_event())
    
    def to_json(self) -> dict:
        """Convert to JSON format with added HEAVEN events and system prompts."""
        base_json = super().to_json()
        base_json["heaven_events"] = self.heaven_events
        base_json["primary_format"] = self.primary_format
        base_json["system_prompts"] = [sp.dict() for sp in self.system_prompts]
        return base_json
    
    @classmethod
    def from_json(cls, data: dict, project: str = None) -> "HeavenHistory":
        """Create HeavenHistory from JSON."""
        # First create a standard History object
        history = super().from_json(data, project)
        
        # Extract heaven_events if present
        heaven_events = data.get("heaven_events", [])
        primary_format = data.get("primary_format", "langchain")
        
        # Extract system_prompts if present
        system_prompts_data = data.get("system_prompts", [])
        system_prompts = [SystemPrompt(**sp_data) for sp_data in system_prompts_data]
        
        # Create and return a HeavenHistory
        return cls(
            messages=history.messages,
            created_datetime=history.created_datetime,
            metadata=history.metadata,
            project=history.project,
            agent_status=history.agent_status,
            history_id=history.history_id,
            json_md_path=history.json_md_path,
            heaven_events=heaven_events,
            primary_format=primary_format,
            system_prompts=system_prompts
        )
    
    @classmethod
    def from_history(cls, history: History) -> "HeavenHistory":
        """Convert a standard History to HeavenHistory."""
        # Convert messages to HEAVEN events
        heaven_events = convert_langchain_to_heaven(history.messages)
        
        # Extract system messages
        system_messages = [msg for msg in history.messages if isinstance(msg, SystemMessage)]
        system_prompts = [SystemPrompt.from_langchain(msg) for msg in system_messages]
        
        # Create and return a HeavenHistory
        return cls(
            messages=history.messages,
            created_datetime=history.created_datetime,
            metadata=history.metadata,
            project=history.project,
            agent_status=history.agent_status,
            history_id=history.history_id,
            json_md_path=history.json_md_path,
            heaven_events=heaven_events,
            system_prompts=system_prompts,
            primary_format="langchain"
        )
    
    @classmethod
    def from_heaven_events(cls, events: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> "HeavenHistory":
        """Create a HeavenHistory from a list of HEAVEN events."""
        # Convert HEAVEN events to LangChain messages
        messages = convert_heaven_to_langchain(events)
        
        # Extract system message events
        system_events = [event for event in events 
                        if is_heaven_event_dict(event) and event.get("event_type") == "SYSTEM_MESSAGE"]
        system_prompts = [SystemPrompt.from_heaven_event(event) for event in system_events 
                         if SystemPrompt.from_heaven_event(event) is not None]
        
        # Create and return a HeavenHistory
        return cls(
            messages=messages,
            created_datetime=datetime.now(),
            metadata=metadata or {},
            heaven_events=events,
            system_prompts=system_prompts,
            primary_format="heaven"
        )
    
    def _compute_heaven_iterations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Split heaven_events into iterations."""
        iterations: Dict[str, List[Dict[str, Any]]] = {}
        current: List[Dict[str, Any]] = []
        idx = 0
        
        for event in self.heaven_events:
            if is_heaven_event_dict(event) and event.get("event_type") == "USER_MESSAGE":
                if current:
                    iterations[f"iteration_{idx}"] = current
                    idx += 1
                    current = []
            else:
                current.append(event)
        
        if current:
            iterations[f"iteration_{idx}"] = current
        
        return iterations
    
    @property
    def heaven_iterations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns a dict of all iterations in HEAVEN event format."""
        return self._compute_heaven_iterations()


def get_heaven_iteration_view(
    history: HeavenHistory,
    start: int,
    end: int
) -> Dict[str, Any]:
    """Get a view of iterations in HEAVEN event format."""
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    
    all_iters = history.heaven_iterations
    total = len(all_iters)
    
    # Build the view
    view: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(start, end + 1):
        key = f"iteration_{i}"
        if key not in all_iters:
            raise KeyError(f"Missing iteration: {key}")
        view[key] = all_iters[key]
    
    return {
        "history_id": history.history_id,
        "total_iterations": total,
        "view_range": {"start": start, "end": end},
        "view": view
    }
