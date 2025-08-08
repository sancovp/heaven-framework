from __future__ import annotations
import sys
import uuid
import json
import os
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)
from pydantic import BaseModel, Field
from .base_piece import BasePiece
from ..utils.name_utils import normalize_agent_name
from ..utils.get_env_value import EnvConfigUtil


from google.adk.sessions.session import Session as ADKSession
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.events.event import Event as ADKEvent
from google.adk.events.event_actions import EventActions
from google.genai import types as adk_types


# from google.adk import types as adk_types
# from google.adk.events.event import Event as ADKEvent
# from google.adk.sessions import Session as ADKSession

class AgentStatus(BaseModel):
    goal: Optional[str] = None
    task_list: List[str] = Field(default_factory=list)
    current_task: Optional[str] = None
    completed: bool = False
    extracted_content: Dict[str, str] = Field(default_factory=dict)



class History(BasePiece):
    """Single conversation with metadata, backed by an ADK Session."""
    # Legacy fallback messages
    messages: List[Union[BaseMessage, ADKEvent]]
    history_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    agent_status: Optional[AgentStatus] = None
    json_md_path: Optional[str] = None

    # The single source of truth for ADK-based histories
    adk_session: Optional[ADKSession] = None

    @property
    def events(self) -> List[ADKEvent]:
        """All ADK events in this history (or fallback to any stored `messages`)."""
        if self.adk_session:
            return self.adk_session.events
        return [m for m in self.messages if isinstance(m, ADKEvent)]

    def to_markdown(self) -> str:
        """Convert to markdown with metadata and a human-readable dump of ADK events."""
        md_parts: List[str] = []
        md_parts.append("===[METADATA]===")
        md_parts.append(f"datetime: {self.created_datetime.isoformat()}")
        md_parts.append(f"history_id: {self.history_id}")
        md_parts.append(f"json_md_path: {self.json_md_path}")
        for k, v in self.metadata.items():
            md_parts.append(f"{k}: {v}")
        md_parts.append("===[CONTENT]===\n")

        # If we have an ADK session, render its events
        for ev in self.events:
            md_parts.append(f"--- [ADK Event] id={ev.id}, author={ev.author}, timestamp={datetime.fromtimestamp(ev.timestamp)}")
            # Render any function calls/responses/text
            for part in getattr(ev.content, "parts", []):
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    md_parts.append(f"*Function Call*: {fc.name}({fc.args})")
                if getattr(part, "function_response", None):
                    fr = part.function_response
                    md_parts.append(f"*Function Response*: {fr.response}")
                if getattr(part, "text", None):
                    md_parts.append(f"{part.text}")
            md_parts.append("")  # blank line between events

        # Fallback: if no ADK session, render legacy BaseMessages
        if not self.adk_session:
            for msg in self.messages:
                md_parts.append("===[MESSAGE]===")
                if isinstance(msg, SystemMessage):
                    md_parts.append(f"**System**: {msg.content}")
                elif isinstance(msg, HumanMessage):
                    md_parts.append(f"**Human**: {msg.content}")
                elif isinstance(msg, AIMessage):
                    md_parts.append(f"**AI**: {msg.content}")
                elif isinstance(msg, ToolMessage):
                    md_parts.append(f"**Tool** (id: {msg.tool_call_id})\n```{msg.content}```")

        return "\n".join(md_parts)

    def to_json(self) -> dict:
        """Convert to JSON format."""
        def _make_jsonable(x):
            if isinstance(x, dict):
                return {k: _make_jsonable(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_make_jsonable(v) for v in x]
            if isinstance(x, set):
                return [_make_jsonable(v) for v in x]
            return x

        msgs: list[dict[str, Any]] = []
        for msg in self.messages:
            if isinstance(msg, ADKEvent):
                raw_evt = msg.model_dump()
                msgs.append({"adk_event": _make_jsonable(raw_evt)})
            else:
                msgs.append({
                    "type": type(msg).__name__,
                    "content": msg.content,
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                    "additional_kwargs": getattr(msg, "additional_kwargs", None),
                    "tool_calls": getattr(msg, "tool_calls", None), 
                })

        out: dict[str, Any] = {
            "history_id": self.history_id,
            "created_datetime": self.created_datetime.isoformat(),
            "metadata": self.metadata,
            "messages": msgs,
            "agent_status": self.agent_status.dict() if self.agent_status else None,
            "json_md_path": self.json_md_path,
        }

        if self.adk_session:
            raw_sess = self.adk_session.model_dump()
            out["adk_session"] = _make_jsonable(raw_sess)

        return out


    @classmethod
    def _load_history_file(cls, history_id: str) -> History:
        """Load a saved history + ADK session from disk."""
        # Use HEAVEN_DATA_DIR for loading agent histories
        base_path = os.path.join(EnvConfigUtil.get_heaven_data_dir(), "agents")
        date_str = "_".join(history_id.split("_")[:3])

        for agent_dir in os.listdir(base_path):
            fn = os.path.join(base_path, agent_dir, "memories", "histories", date_str, f"{history_id}.json")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    data = json.load(f)
                hist = cls.from_json(data)
                hist.json_md_path = os.path.dirname(fn)
                # restore ADK session
                if data.get("adk_session"):
                    hist.adk_session = ADKSession.model_validate(data["adk_session"])
                return hist

        raise FileNotFoundError(f"No history file found for ID {history_id}")

    @classmethod
    def load_from_id(cls, history_id: str) -> History:
        """Continue an existing history (or start fresh if None)."""
        if history_id is None:
            return cls(messages=[], history_id=str(uuid.uuid4()))

        orig = cls._load_history_file(history_id)
        # bump continuation suffix
        if "_continued_" in history_id:
            m = re.search(r"_continued_(\d+)$", history_id)
            num = int(m.group(1)) + 1 if m else 1
            new_id = history_id.split("_continued_")[0] + f"_continued_{num}"
        else:
            new_id = f"{history_id}_continued_1"

        return History(
            messages=orig.messages.copy(),
            adk_session=orig.adk_session,
            created_datetime=datetime.now(),
            metadata=orig.metadata.copy(),
            agent_status=orig.agent_status,
            history_id=new_id,
            json_md_path=orig.json_md_path,
        )

    # def save(self, agent_name: str) -> str:
    #     """Persist both .json and .md to disk."""
    #     base_dir = os.path.dirname(os.path.abspath(__file__))
    #     base_path = os.path.join(os.path.dirname(base_dir), "agents")
    #     nm = normalize_agent_name(agent_name)
    #     now = datetime.now()
    #     date_dir = now.strftime("%Y_%m_%d")
    #     dt_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    #     hist_id = f"{dt_str}_{nm}"
    #     path = os.path.join(base_path, nm, "memories", "histories", date_dir)
    #     os.makedirs(path, exist_ok=True)

    #     existing = [f for f in os.listdir(path) if f.startswith(hist_id)]
    #     if existing:
    #         nums = [int(f.split("_continued_")[1].split(".")[0]) for f in existing if "_continued_" in f]
    #         nxt = max(nums, default=0) + 1
    #         hist_id = f"{hist_id}_continued_{nxt}"
    #     self.history_id = hist_id
    #     json_fp = os.path.join(path, f"{hist_id}.json")
    #     md_fp   = os.path.join(path, f"{hist_id}.md")

    #     with open(json_fp, "w") as f:
    #         json.dump(self.to_json(), f, indent=2)
    #     with open(md_fp, "w") as f:
    #         f.write(self.to_markdown())

    #     return hist_id
    def save(self, agent_name: str) -> str:
        """Persist both .json and .md to disk, overwriting if history_id exists."""
        # Use HEAVEN_DATA_DIR for saving agent histories
        base_path = os.path.join(EnvConfigUtil.get_heaven_data_dir(), "agents")
        nm = normalize_agent_name(agent_name)
        
        # If we already have a history_id, use it
        if self.history_id:
            # Extract date directory from existing history_id
            date_str = "_".join(self.history_id.split("_")[:3])
            path = os.path.join(base_path, nm, "memories", "histories", date_str)
            hist_id = self.history_id
        else:
            # Create new history_id for new conversation
            now = datetime.now()
            date_dir = now.strftime("%Y_%m_%d")
            hist_id = f"{now.strftime('%Y_%m_%d_%H_%M_%S')}_{nm}"
            path = os.path.join(base_path, nm, "memories", "histories", date_dir)
        
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)
        
        # Use existing or new history_id
        self.history_id = hist_id
        json_fp = os.path.join(path, f"{hist_id}.json")
        md_fp = os.path.join(path, f"{hist_id}.md")
        
        # Write files, overwriting existing ones
        with open(json_fp, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        with open(md_fp, "w") as f:
            f.write(self.to_markdown())
        
        return hist_id
      
    @classmethod
    def from_json(cls, data: dict, project: str = None) -> History:
        msgs: List[Union[BaseMessage, ADKEvent]] = []
        for m in data["messages"]:
            if "adk_event" in m:
                evt = ADKEvent.model_validate(m["adk_event"])
                msgs.append(evt)
            else:
                t = m["type"]
                if t == "SystemMessage":
                    msgs.append(SystemMessage(content=m["content"]))
                elif t == "HumanMessage":
                    msgs.append(HumanMessage(content=m["content"]))
                elif t == "AIMessage":
                    msgs.append(AIMessage(content=m["content"], additional_kwargs=m.get("additional_kwargs", {}), tool_calls=m.get("tool_calls", None)))
                elif t == "ToolMessage":
                    msgs.append(ToolMessage(content=m["content"], tool_call_id=m.get("tool_call_id")))

        status = AgentStatus(**data["agent_status"]) if data.get("agent_status") else None

        hist = cls(
            messages=msgs,
            created_datetime=datetime.fromisoformat(data["created_datetime"]),
            metadata=data.get("metadata", {}),
            project=project,
            agent_status=status,
            history_id=data.get("history_id"),
            json_md_path=data.get("json_md_path"),
        )

        # rehydrate session if present
        if "adk_session" in data:
            hist.adk_session = ADKSession.model_validate(data["adk_session"])

        return hist

      
    @classmethod
    def from_markdown(cls, md_content: str, project: str = None) -> "History":
        print("Parsing markdown:", md_content)  # Debug
        """Parse markdown back into History"""
        # Split into metadata and content sections
        parts = md_content.split("===[CONTENT]===")
        if len(parts) == 2:
            metadata_section = parts[0].split("===[METADATA]===")[1].strip()
            content = parts[1].strip()
            metadata_lines = metadata_section.split("\n")
        else:
            metadata_lines = []
            content = md_content
            
        # Parse metadata
        metadata = {}
        datetime_val = None
        history_id = None
        for line in metadata_lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "datetime":
                    datetime_val = datetime.fromisoformat(value)
                elif key == "history_id":  
                    history_id = value
                elif key == "json_md_path":  # Add this condition
                    json_md_path = value

                else:
                    metadata[key] = value

        # Parse messages by splitting on message separator
        messages = []
        message_blocks = content.split("===[MESSAGE]===")
        
        for block in message_blocks:
            block = block.strip()
            if not block:  # Skip empty blocks
                continue
            
            lines = block.split("\n")
            current_type = None
            current_lines = []
            
            # Special handling for tool messages
            if block.startswith("**Tool**"):
                # Extract tool ID
                header = lines[0]
                tool_id = header.split("id:", 1)[1].split(")", 1)[0].strip()
                
                # Extract content between backticks
                content_lines = []
                in_code_block = False
                for line in lines[1:]:  # Skip header
                    if line.strip() == "```":
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        content_lines.append(line)
                
                content = "\n".join(content_lines)
                messages.append(ToolMessage(content=content, tool_call_id=tool_id))
                continue
            
            # Normal message handling
            for line in lines:
                if line.startswith("**") and "**:" in line:
                    current_type = line.split("**:", 1)[0].strip("*").strip()
                    current_lines = [line.split("**:", 1)[1].strip()]
                else:
                    current_lines.append(line)
                    
            # Process accumulated message at end of block
            if current_lines and current_type:
                msg_content = "\n".join(current_lines).strip()
                if msg_content:  # Only add if there's actual content
                    if current_type == "System":
                        messages.append(SystemMessage(content=msg_content))
                    elif current_type == "Human":
                        messages.append(HumanMessage(content=msg_content))
                    elif current_type == "AI":
                        messages.append(AIMessage(content=msg_content))
                    elif current_type.startswith("Tool"):
                        # Extract tool_call_id if present
                        tool_id = None
                        if "(id:" in current_type:
                            tool_id = current_type.split("id:", 1)[1].split(")", 1)[0].strip()
                        
                        # Process tool message more carefully
                        try:
                            print(f"Raw tool message content: {msg_content}")  # Debug
                            sys.stdout.flush()
                            lines = [line for line in msg_content.split("\n") if line.strip()]  # Remove empty lines
                            print(f"Non-empty lines: {lines}")  # Debug
                            
                            # Extract content between ``` markers
                            content_start = lines.index("```") + 1
                            content_end = lines.index("```", content_start)
                            tool_content = "\n".join(lines[content_start:content_end])
                            
                            print(f"Extracted tool content: {tool_content}")  # Debug
                            messages.append(ToolMessage(content=tool_content, tool_call_id=tool_id))
                        except Exception as e:
                            print(f"Error processing tool message: {e}")  # Debug
        
        # Process blocks complete

        return cls(
            messages=messages,
            created_datetime=datetime_val or datetime.now(),
            project=project,
            metadata=metadata,
            history_id=history_id,
            json_md_path=json_md_path
        )
      

    @classmethod
    def from_adk_session(cls, session: ADKSession) -> "History":
        """
        Wrap an ADKSession without mutating history.messages at all.
        """
        hist = cls(
            messages=[],                                      # untouched
            history_id=None,                                  # will be set on save()
            created_datetime=datetime.fromtimestamp(session.last_update_time),
            metadata={},
            agent_status=None,
        )
        hist.adk_session = session
        return hist



    def to_adk_session(
        self,
        app_name: str,
        user_id: str,
    ) -> ADKSession:
        """
        Return the exact same ADKSession we stored—state and events intact.
        """
        if not hasattr(self, "adk_session") or self.adk_session is None:
            raise RuntimeError("No ADK session attached to this History")
        return self.adk_session

    # def _compute_iterations(self) -> Dict[str, List[ADKEvent]]:
    #     """
    #     Split events into “iterations”: chunks between user turns.
    #     """
    #     iters: Dict[str, List[ADKEvent]] = {}
    #     current: List[ADKEvent] = []
    #     idx = 0
    #     for ev in self.events:
    #         if ev.author == "user":
    #             if current:
    #                 iters[f"iteration_{idx}"] = current
    #                 idx += 1
    #                 current = []
    #         else:
    #             current.append(ev)
    #     if current:
    #         iters[f"iteration_{idx}"] = current
    #     return iters

    # @property
    # def iterations(self) -> Dict[str, List[ADKEvent]]:
    #     return self._compute_iterations()


    def to_uni_messages(self) -> List[Dict[str, Any]]:
        """Convert dictionary messages to uni-api format: dictionary → LangChain → uni-api"""
        import json
        
        # Convert dictionary messages (stored in self.messages) to LangChain objects first
        msgs_as_dicts = self.to_json()["messages"]
        
        langchain_messages = []
        for m in msgs_as_dicts:
            if "adk_event" in m:
                continue  # Skip ADK events
            else:
                t = m["type"]
                if t == "SystemMessage":
                    langchain_messages.append(SystemMessage(content=m["content"]))
                elif t == "HumanMessage":
                    langchain_messages.append(HumanMessage(content=m["content"]))
                elif t == "AIMessage":
                    langchain_messages.append(AIMessage(content=m["content"], additional_kwargs=m.get("additional_kwargs", {}), tool_calls=m.get("tool_calls", None)))
                elif t == "ToolMessage":
                    langchain_messages.append(ToolMessage(content=m["content"], tool_call_id=m.get("tool_call_id")))
        
        # Convert LangChain objects to uni-api format
        uni_messages = []
        
        for msg in langchain_messages:
            if isinstance(msg, SystemMessage):
                uni_messages.append({"role": "system", "content": msg.content})
            
            elif isinstance(msg, HumanMessage):
                uni_messages.append({"role": "user", "content": msg.content})
            
            elif isinstance(msg, AIMessage):
                uni_msg = {"role": "assistant", "content": msg.content or ""}
                
                # Preserve additional_kwargs
                if msg.additional_kwargs:
                    uni_msg["additional_kwargs"] = msg.additional_kwargs
                
                # Handle tool calls in additional_kwargs (OpenAI style)
                if msg.additional_kwargs.get("tool_calls"):
                    uni_msg["tool_calls"] = msg.additional_kwargs["tool_calls"]
                
                # Handle tool calls as direct attribute
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    uni_msg["tool_calls"] = msg.tool_calls
                
                # Handle Anthropic style tool_use in content
                elif isinstance(msg.content, list):
                    tool_calls = []
                    content_parts = []
                    
                    for item in msg.content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            # Convert Anthropic format to OpenAI format
                            tool_calls.append({
                                "id": item.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": json.dumps(item.get("input", {}))
                                }
                            })
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            content_parts.append(item.get('text', ''))
                        elif isinstance(item, str):
                            content_parts.append(item)
                    
                    if tool_calls:
                        uni_msg["tool_calls"] = tool_calls
                    if content_parts:
                        uni_msg["content"] = ' '.join(content_parts)
                
                uni_messages.append(uni_msg)
            
            elif isinstance(msg, ToolMessage):
                uni_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content
                })
        
        return uni_messages

    @classmethod
    def from_uni_messages(
        cls, 
        uni_messages: List[Dict[str, Any]], 
        history_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "History":
        """Create History from uni-api message format: uni-api → LangChain → dictionary"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        
        # First: uni-api → LangChain (original logic)
        messages = []
        
        for uni_msg in uni_messages:
            role = uni_msg["role"]
            content = uni_msg.get("content", "")
            
            if role == "system":
                messages.append(SystemMessage(content=content))
            
            elif role == "user":
                messages.append(HumanMessage(content=content))
            
            elif role == "assistant":
                # Restore additional_kwargs from uni-api format
                additional_kwargs = uni_msg.get("additional_kwargs", {})
                
                if uni_msg.get("tool_calls"):
                    # Transform OpenAI format to LangChain format
                    langchain_tool_calls = []
                    for tc in uni_msg["tool_calls"]:
                        if "function" in tc:
                            # Convert OpenAI format to LangChain format
                            langchain_tool_calls.append({
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "args": json.loads(tc["function"]["arguments"]),
                                "type": "tool_call"
                            })
                        else:
                            # Already in LangChain format
                            langchain_tool_calls.append(tc)
                    
                    # Merge tool_calls into additional_kwargs
                    additional_kwargs["tool_calls"] = uni_msg["tool_calls"]
                    messages.append(AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        tool_calls=langchain_tool_calls  # Use converted format
                    ))
                else:
                    messages.append(AIMessage(content=content, additional_kwargs=additional_kwargs))
            
            elif role == "tool":
                messages.append(ToolMessage(
                    content=content,
                    tool_call_id=uni_msg["tool_call_id"]
                ))
        
        # Convert LangChain messages to dictionary format for storage
        message_dicts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                message_dicts.append({
                    "type": "SystemMessage",
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, 'additional_kwargs', {})
                })
            elif isinstance(msg, HumanMessage):
                message_dicts.append({
                    "type": "HumanMessage", 
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, 'additional_kwargs', {})
                })
            elif isinstance(msg, AIMessage):
                message_dicts.append({
                    "type": "AIMessage",
                    "content": msg.content,
                    "additional_kwargs": getattr(msg, 'additional_kwargs', {}),
                    "tool_calls": getattr(msg, 'tool_calls', None)
                })
            elif isinstance(msg, ToolMessage):
                message_dicts.append({
                    "type": "ToolMessage",
                    "content": msg.content,
                    "tool_call_id": getattr(msg, 'tool_call_id', None)
                })
        
        # Create History using from_json to properly convert dictionaries to LangChain objects
        json_data = {
            "messages": message_dicts,
            "history_id": history_id,
            "metadata": metadata or {},
            "created_datetime": datetime.now().isoformat(),
            "agent_status": None,
            "json_md_path": None
        }
        return cls.from_json(json_data)

    def save_with_uni_context(self, agent_name: str, uni_api_used: bool = False) -> str:
        """
        Enhanced save method that adds uni-api context to metadata.
        This is optional - the regular save() method will work fine too.
        """
        if uni_api_used:
            self.metadata["uni_api_used"] = True
            self.metadata["provider_unified"] = True
        
        return self.save(agent_name)

    def get_last_n_uni_messages(self, n: int) -> List[Dict[str, Any]]:
        """Get last N messages in uni-api format"""
        uni_messages = self.to_uni_messages()
        return uni_messages[-n:] if len(uni_messages) >= n else uni_messages

    def append_uni_messages(self, uni_messages: List[Dict[str, Any]]):
        """Append uni-api messages to history after converting to LangChain format"""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        
        for uni_msg in uni_messages:
            role = uni_msg["role"]
            content = uni_msg.get("content", "")
            
            if role == "system":
                self.messages.append(SystemMessage(content=content))
            
            elif role == "user":
                self.messages.append(HumanMessage(content=content))
            
            elif role == "assistant":
                if uni_msg.get("tool_calls"):
                    self.messages.append(AIMessage(
                        content=content,
                        additional_kwargs={"tool_calls": uni_msg["tool_calls"]}
                    ))
                else:
                    self.messages.append(AIMessage(content=content))
            
            elif role == "tool":
                self.messages.append(ToolMessage(
                    content=content,
                    tool_call_id=uni_msg["tool_call_id"]
                ))


# def get_iteration_view(
#     history: History,
#     start: int,
#     end: int
# ) -> Dict[str, Any]:
#     """
#     Return a dict containing:
#       • history_id
#       • total_iterations
#       • view_range: {'start', 'end'}
#       • view: sub-dict iteration_i → List[ADKEvent]
#     """
#     if start > end:
#         raise ValueError(f"start ({start}) must be <= end ({end})")
#     all_iters = history.iterations
#     total = len(all_iters)

#     view: Dict[str, List[ADKEvent]] = {}
#     for i in range(start, end + 1):
#         key = f"iteration_{i}"
#         if key not in all_iters:
#             raise KeyError(f"Missing iteration: {key}")
#         view[key] = all_iters[key]

#     return {
#         "history_id": history.history_id,
#         "total_iterations": total,
#         "view_range": {"start": start, "end": end},
#         "view": view,
#     }


# Should add utils like this
# def get_last_tool_msg(history) -> Optional[str]:
#     """
#     Scan history.messages to find the most recent tool use and return its ToolMessage.content.

#     Args:
#         history: The agent's history object with a .messages attribute.

#     Returns:
#         Optional[str]: The content of the last ToolMessage, or None if none found.
#     """
#     for i, msg in enumerate(history.messages):
#         if isinstance(msg, AIMessage) and isinstance(msg.content, list):
#             for item in msg.content:
#                 if isinstance(item, dict) and item.get('type') == 'tool_use':
#                     # Next message should be the tool result
#                     if i + 1 < len(history.messages) and isinstance(history.messages[i + 1], ToolMessage):
#                         return history.messages[i + 1].content
#     return None



  
# class History(BasePiece):
#     """Single conversation with metadata"""
#     # messages: List[BaseMessage] # old
#     messages: List[Union[BaseMessage, ADKEvent]]
#     history_id: Optional[str] = None
#     # Using created_datetime from BasePiece
#     metadata: Dict = {}
#     agent_status: Optional[AgentStatus] = None
#     json_md_path: Optional[str] = None
#     adk_session: Optional[ADKSession] = None
    
#     def to_markdown(self) -> str:
#         """Convert to markdown with metadata header"""
#         md_parts = []
        
#         # Add metadata header with unique separator
#         md_parts.append("===[METADATA]===")
#         md_parts.append(f"datetime: {self.created_datetime.isoformat()}")
#         md_parts.append(f"history_id: {self.history_id}")
#         md_parts.append(f"json_md_path: {self.json_md_path}")
#         for k, v in self.metadata.items():
#             md_parts.append(f"{k}: {v}")
#         md_parts.append("===[CONTENT]===\n")
        
#         # Add messages
#         for msg in self.messages:
#             # This needs work
#             if isinstance(msg, ADKEvent):
#                 md_parts.append(f"**[ADK Event {msg.id}]** {msg.actions=}, …")  # however you’d like to render it
#             else:
#                 # Add a separator between messages
#                 md_parts.append("===[MESSAGE]===")
                
#                 if isinstance(msg, SystemMessage):
#                     md_parts.append(f"**System**: {msg.content}")
#                 elif isinstance(msg, HumanMessage):
#                     md_parts.append(f"**Human**: {msg.content}")
#                 elif isinstance(msg, AIMessage):
#                     md_parts.append(f"**AI**: {msg.content}")
#                 elif isinstance(msg, ToolMessage):
#                     # Special format for tool messages to make parsing easier
#                     tool_msg = [
#                         "**Tool** (id: {tool_id})".format(tool_id=msg.tool_call_id),
#                         "```",
#                         msg.content,
#                         "```"
#                     ]
#                     md_parts.append("\n".join(tool_msg))
        
#         return "\n\n".join(md_parts)
    
#     def to_json(self) -> dict:
#         """Convert to JSON format"""
#         msgs = []
#         for msg in self.messages:
#             if isinstance(msg, ADKEvent):
#                 # ADK Event knows how to JSON‐serialize itself
#                 # dump to dict and convert any sets -> lists
#                 raw = msg.model_dump()
#                 def _make_jsonable(x):
#                     if isinstance(x, dict):
#                         return {k: _make_jsonable(v) for k,v in x.items()}
#                     if isinstance(x, list):
#                         return [_make_jsonable(v) for v in x]
#                     if isinstance(x, set):
#                         return [_make_jsonable(v) for v in x]
#                     return x
#                 clean = _make_jsonable(raw)
#                 msgs.append({"adk_event": clean})
#             else:
#                 # existing BaseMessage branch
#                 msgs.append({
#                   "type": type(msg).__name__,
#                   "content": msg.content,
#                   "tool_call_id": getattr(msg, "tool_call_id", None),
#                   "additional_kwargs": getattr(msg, "additional_kwargs", None),
#                 })
        
#         return {
#             "history_id": self.history_id,
#             "created_datetime": self.created_datetime.isoformat(),
#             "metadata": self.metadata,
#             "messages": msgs,
#             "agent_status": self.agent_status.dict() if self.agent_status else None,
#             "json_md_path": self.json_md_path
#         }


#     @classmethod
#     def _load_history_file(cls, history_id: str) -> 'History':
#         """Load history file from disk"""
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         base_path = os.path.join(os.path.dirname(base_dir), "agents")
    
#         if not os.path.exists(base_path):
#             raise FileNotFoundError(f"Agents directory not found at {base_path}")
    
#         # Get date from history_id for narrowing search
#         date_str = '_'.join(history_id.split('_')[:3])
    
#         # Search for the history file in all agent directories
#         for agent_dir in os.listdir(base_path):
#             history_path = os.path.join(base_path, agent_dir, "memories", "histories", date_str, f"{history_id}.json")
#             print(f"Checking path: {history_path}")  # Debug print
#             if os.path.exists(history_path):
#                 print(f"Found history file at: {history_path}")  # Debug print
#                 with open(history_path, 'r') as f:
#                     history_data = json.load(f)
#                     history = cls.from_json(history_data)
#                     history.json_md_path = os.path.dirname(history_path)
#                     history.adk_session = ADKSession.model_validate(history_data["adk_session"])

#                     return history
    
#         raise FileNotFoundError(f"No history file found for ID {history_id} in any agent directory")
          
#     @classmethod
#     def load_from_id(cls, history_id: str) -> 'History':
#         """Load history from ID with proper continuation handling"""
#         if history_id is None:
#             # Brand new conversation
#             return cls(messages=[], history_id=str(uuid.uuid4()))
    
#         try:
#             # Load existing history
#             original = cls._load_history_file(history_id)
    
#             # Create continuation ID
#             if "_continued_" in history_id:
#                 pattern = r"_continued_(\d+)$"
#                 match = re.search(pattern, history_id)
#                 if match:
#                     current_num = int(match.group(1))
#                     base_id = history_id.split("_continued_")[0]
#                     new_history_id = f"{base_id}_continued_{current_num + 1}"
#                 else:
#                     raise ValueError(f"Invalid continuation ID format: {history_id}")
#             else:
#                 new_history_id = f"{history_id}_continued_1"
#             # Create a cleaned copy of agent_status
#             clean_agent_status = None
#             if original.agent_status:
#                 # Create a copy by converting to dict and back
#                 agent_status_dict = original.agent_status.dict()
                
#                 # Clean any block reports from extracted_content
#                 if 'extracted_content' in agent_status_dict and agent_status_dict['extracted_content']:
#                     if 'block_report' in agent_status_dict['extracted_content']:
#                         del agent_status_dict['extracted_content']['block_report']
                
#                 # Create new AgentStatus with cleaned data
#                 clean_agent_status = AgentStatus(**agent_status_dict)

#             # Create new history with same messages but new ID
#             return cls(
#                 messages=original.messages.copy(),
#                 created_datetime=datetime.now(),
#                 metadata=original.metadata.copy(),
#                 history_id=new_history_id,
#                 agent_status=clean_agent_status,
#                 json_md_path=original.json_md_path
#             )
#         except FileNotFoundError:
#             raise ValueError(f"No history found with ID {history_id}")
      
#     def save(self, agent_name: str):
#         # Get base path
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         base_path = os.path.join(os.path.dirname(base_dir), "agents")
    
#         # Normalize agent name to match AgentMakerTool's format
#         normalized_agent_name = normalize_agent_name(agent_name)  # Using our new function
    
#         # Generate history_id based on datetime and agent name
#         now = datetime.now()
#         date_str = now.strftime('%Y_%m_%d')  # For directory
#         datetime_str = now.strftime('%Y_%m_%d_%H_%M_%S')  # For file name
#         history_id = f"{datetime_str}_{normalized_agent_name}"
    
#         # Create path including date directory
#         path = os.path.join(base_path, normalized_agent_name, "memories", "histories", date_str)
#         self.json_md_path = path
#         os.makedirs(path, exist_ok=True)
    
#         # Look for existing files with same base name
#         existing_files = [f for f in os.listdir(path) if f.startswith(history_id)]
#         if existing_files:
#             # If files exist, this is a continuation
#             max_cont = 0
#             for f in existing_files:
#                 if "_continued_" in f:
#                     cont_num = int(f.split("_continued_")[1].split(".")[0])
#                     max_cont = max(max_cont, cont_num)
#             history_id = f"{history_id}_continued_{max_cont + 1}"
#         self.history_id = history_id
#         # Save both JSON and Markdown versions
#         json_filepath = os.path.join(path, f"{history_id}.json")
#         md_filepath = os.path.join(path, f"{history_id}.md")
        
#         # Save JSON
#         with open(json_filepath, 'w') as f:
#             json.dump(self.to_json(), f)
    
#         # Save Markdown
#         with open(md_filepath, 'w') as f:
#             f.write(self.to_markdown())
    
#         return history_id
      
#     @classmethod
#     def from_json(cls, data: dict, project: str = None) -> "History":
#         """Create History from JSON"""
#         messages = []
#         for msg in data["messages"]:
#             if msg["type"] == "SystemMessage":
#                 messages.append(SystemMessage(content=msg["content"]))
#             elif msg["type"] == "HumanMessage":
#                 messages.append(HumanMessage(content=msg["content"]))
#             elif msg["type"] == "AIMessage":
#                 messages.append(AIMessage(
#                     content=msg["content"],
#                     additional_kwargs=msg["additional_kwargs"]
#                 ))
#             elif msg["type"] == "ToolMessage":
#                 messages.append(ToolMessage(
#                     content=msg["content"],
#                     tool_call_id=msg["tool_call_id"]
#                 ))
    
#         # Handle status if it exists
#         agent_status = None
#         if data.get("agent_status"):
#             agent_status = AgentStatus(**data["agent_status"])
    
#         return cls(
#             messages=messages,
#             created_datetime=datetime.fromisoformat(data["created_datetime"]),
#             metadata=data["metadata"],
#             project=project,
#             agent_status=agent_status,
#             history_id=data.get("history_id"),
#             json_md_path=data.get("json_md_path")
#         )
      
#     @classmethod
#     def from_markdown(cls, md_content: str, project: str = None) -> "History":
#         print("Parsing markdown:", md_content)  # Debug
#         """Parse markdown back into History"""
#         # Split into metadata and content sections
#         parts = md_content.split("===[CONTENT]===")
#         if len(parts) == 2:
#             metadata_section = parts[0].split("===[METADATA]===")[1].strip()
#             content = parts[1].strip()
#             metadata_lines = metadata_section.split("\n")
#         else:
#             metadata_lines = []
#             content = md_content
            
#         # Parse metadata
#         metadata = {}
#         datetime_val = None
#         history_id = None
#         for line in metadata_lines:
#             if ":" in line:
#                 key, value = line.split(":", 1)
#                 key = key.strip()
#                 value = value.strip()
#                 if key == "datetime":
#                     datetime_val = datetime.fromisoformat(value)
#                 elif key == "history_id":  
#                     history_id = value
#                 elif key == "json_md_path":  # Add this condition
#                     json_md_path = value

#                 else:
#                     metadata[key] = value

#         # Parse messages by splitting on message separator
#         messages = []
#         message_blocks = content.split("===[MESSAGE]===")
        
#         for block in message_blocks:
#             block = block.strip()
#             if not block:  # Skip empty blocks
#                 continue
            
#             lines = block.split("\n")
#             current_type = None
#             current_lines = []
            
#             # Special handling for tool messages
#             if block.startswith("**Tool**"):
#                 # Extract tool ID
#                 header = lines[0]
#                 tool_id = header.split("id:", 1)[1].split(")", 1)[0].strip()
                
#                 # Extract content between backticks
#                 content_lines = []
#                 in_code_block = False
#                 for line in lines[1:]:  # Skip header
#                     if line.strip() == "```":
#                         in_code_block = not in_code_block
#                         continue
#                     if in_code_block:
#                         content_lines.append(line)
                
#                 content = "\n".join(content_lines)
#                 messages.append(ToolMessage(content=content, tool_call_id=tool_id))
#                 continue
            
#             # Normal message handling
#             for line in lines:
#                 if line.startswith("**") and "**:" in line:
#                     current_type = line.split("**:", 1)[0].strip("*").strip()
#                     current_lines = [line.split("**:", 1)[1].strip()]
#                 else:
#                     current_lines.append(line)
                    
#             # Process accumulated message at end of block
#             if current_lines and current_type:
#                 msg_content = "\n".join(current_lines).strip()
#                 if msg_content:  # Only add if there's actual content
#                     if current_type == "System":
#                         messages.append(SystemMessage(content=msg_content))
#                     elif current_type == "Human":
#                         messages.append(HumanMessage(content=msg_content))
#                     elif current_type == "AI":
#                         messages.append(AIMessage(content=msg_content))
#                     elif current_type.startswith("Tool"):
#                         # Extract tool_call_id if present
#                         tool_id = None
#                         if "(id:" in current_type:
#                             tool_id = current_type.split("id:", 1)[1].split(")", 1)[0].strip()
                        
#                         # Process tool message more carefully
#                         try:
#                             print(f"Raw tool message content: {msg_content}")  # Debug
#                             sys.stdout.flush()
#                             lines = [line for line in msg_content.split("\n") if line.strip()]  # Remove empty lines
#                             print(f"Non-empty lines: {lines}")  # Debug
                            
#                             # Extract content between ``` markers
#                             content_start = lines.index("```") + 1
#                             content_end = lines.index("```", content_start)
#                             tool_content = "\n".join(lines[content_start:content_end])
                            
#                             print(f"Extracted tool content: {tool_content}")  # Debug
#                             messages.append(ToolMessage(content=tool_content, tool_call_id=tool_id))
#                         except Exception as e:
#                             print(f"Error processing tool message: {e}")  # Debug
        
#         # Process blocks complete

#         return cls(
#             messages=messages,
#             created_datetime=datetime_val or datetime.now(),
#             project=project,
#             metadata=metadata,
#             history_id=history_id,
#             json_md_path=json_md_path
#         )
      
#     @classmethod
#     def from_adk_session(cls, session: ADKSession) -> "History":
#         return cls(
#             messages=session.events,        # now a list of ADKEvent
#             history_id=session.id,
#             created_datetime=datetime.fromtimestamp(session.last_update_time),
#             metadata={},                    # or deserialize session.state if you like
#             # agent_status left None or reconstructed from state
#         )

#     def to_adk_session(
#         self,
#         app_name: str,
#         user_id: str,
#     ) -> ADKSession:
#         return ADKSession(
#             id=self.history_id or "",
#             app_name=app_name,
#             user_id=user_id,
#             state={},                      # serialize .agent_status or metadata if needed
#             events=[m for m in self.messages if isinstance(m, ADKEvent)],
#             last_update_time=self.created_datetime.timestamp(),
#         )
#     # Old
#     # def _compute_iterations(self) -> Dict[str, List[BaseMessage]]:
#     #     """
#     #     Split self.messages into “iterations”:
#     #     each iteration is the sequence of messages *between* two human turns.
#     #     HumanMessages themselves are not included.
#     #     """
#     #     iterations: Dict[str, List[BaseMessage]] = {}
#     #     current: List[BaseMessage] = []
#     #     idx = 0

#     #     for msg in self.messages:
#     #         if isinstance(msg, HumanMessage):
#     #             if current:
#     #                 iterations[f"iteration_{idx}"] = current
#     #                 idx += 1
#     #                 current = []
#     #         else:
#     #             current.append(msg)

#     #     if current:
#     #         iterations[f"iteration_{idx}"] = current

#     #     return iterations
  # unsure if this handles ADK correctly
    def _compute_iterations(self) -> Dict[str, List[Union[BaseMessage, ADKEvent]]]:
        """
        Split self.messages into “iterations”:
        each iteration is the sequence of messages *between* two user turns.
        HumanMessage and ADKEvent(author="user") both start a new iteration.
        """
        iterations: Dict[str, List[Union[BaseMessage, ADKEvent]]] = {}
        current: List[Union[BaseMessage, ADKEvent]] = []
        idx = 0
    
        for msg in self.messages:
            is_user = (
                isinstance(msg, HumanMessage)
                or (isinstance(msg, ADKEvent) and getattr(msg, "author", None) == "user")
            )
            if is_user:
                if current:
                    iterations[f"iteration_{idx}"] = current
                    idx += 1
                    current = []
            else:
                current.append(msg)
    
        if current:
            iterations[f"iteration_{idx}"] = current
    
        return iterations
      
    @property
    def iterations(self) -> Dict[str, List[Union[BaseMessage, ADKEvent]]]: # changed return annotation
        """
        Returns a dict of all “iterations” in this history,
        where each iteration is the sequence of non‑HumanMessage
        messages between two human turns.
        """
        return self._compute_iterations()




def get_iteration_view(
    history: History,
    start: int,
    end: int
) -> Dict[str, Any]:
    """
    Return a dict containing:
      • history_id            – the ID of this history
      • total_iterations      – how many iterations the history has
      • view_range            – dict with 'start' and 'end'
      • view                  – sub‑dict of iteration_i → List[BaseMessage]
    Raises:
      • ValueError if start > end
      • KeyError   if any iteration_i in that range is missing
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    all_iters = history.iterations
    total = len(all_iters)

    # Build the view
    view: Dict[str, List[BaseMessage]] = {}
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

# from google.genai import types
    # def to_contents(self) -> list[types.Content]:
    #     contents: list[types.Content] = []
    #     for msg in self.messages:
    #         if isinstance(msg, SystemMessage):
    #             role = "system";  text = msg.content
    #         elif isinstance(msg, HumanMessage):
    #             role = "user";    text = msg.content
    #         elif isinstance(msg, AIMessage):
    #             role = "assistant"
    #             text = (
    #               msg.content
    #               if isinstance(msg.content, str)
    #               else "".join(
    #                  b.get("text","") for b in msg.content if b.get("type")=="text"
    #               )
    #             )
    #         elif isinstance(msg, ToolMessage):
    #             role = "tool";    text = msg.content
    #         else:
    #             continue

    #         contents.append(types.Content(
    #             role=role,
    #             parts=[types.Part(text=text)]
    #         ))
    #     return contents

    # def update_from_contents(self, contents: list[types.Content]):
    #     new_msgs: list[BaseMessage] = []
    #     for c in contents:
    #         text = "".join(p.text or "" for p in c.parts)
    #         if c.role == "system":
    #             new_msgs.append(SystemMessage(content=text))
    #         elif c.role == "user":
    #             new_msgs.append(HumanMessage(content=text))
    #         elif c.role == "assistant":
    #             new_msgs.append(AIMessage(content=text))
    #         elif c.role == "tool":
    #             new_msgs.append(ToolMessage(content=text))
    #     self.messages = new_msgs