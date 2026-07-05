#!/usr/bin/env python3
"""
HEAVEN Chat Branching — chat engineering as prompt engineering.

A Chat IS A sequence of messages (a heaven History). This module gives that
object the same lifecycle a prompt has in a prompt-engineering system:

  fork      — copy a past History (optionally only messages[:k]) into a NEW
              History with its own history_id; the source is never mutated.
  template  — freeze a fork as a named, reusable ChatTemplate (a chat lifted
              out of its run; supports {{variable}} parameterization).
  rehydrate — instantiate a template back into a live History (+ optionally a
              new conversation), ready for BaseHeavenAgent(history_id=...).
  compose   — concatenate templates into bigger templates, and export any
              template as plain dict shapes (`to_context_dict`,
              `to_parsed_messages`) consumable by sdna's ContextEngineeringLib
              (inject(context=...) / ParsedMessage) WITHOUT any cross-import.

Storage:
  forks     → the canonical History path
              ($HEAVEN_DATA_DIR/agents/{agent}/memories/histories/{date}/...)
              so a fork is immediately continuable via History.load_from_id /
              BaseHeavenAgent(history_id=...).
  templates → $HEAVEN_DATA_DIR/chat_templates/{template_id}.json
              (flat files + glob listing, same convention as conversations).

Vocabulary note: this module deliberately uses `fork`/`template`/`rehydrate`
(NOT `weave`) to avoid colliding with ContextManager's woven_* synthetic
histories, which live in a separate, non-canonical store.
"""

from __future__ import annotations

import glob
import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from ..utils.get_env_value import EnvConfigUtil
from ..utils.name_utils import normalize_agent_name
from .conversations import ConversationManager, get_latest_history
from .history import History

# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

_ROLE_BY_TYPE = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "ToolMessage": "tool",
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "template"


def _load_source_history(history_id: str) -> History:
    """Raw load — NO continuation bump (load_from_id would mint _continued_N)."""
    return History._load_history_file(history_id)


def _owning_agent(history: History) -> Optional[str]:
    """Extract the owning agent dir from a loaded history's json_md_path
    (.../agents/{agent}/memories/histories/{date})."""
    if not history.json_md_path:
        return None
    parts = os.path.normpath(history.json_md_path).split(os.sep)
    try:
        return parts[parts.index("agents") + 1]
    except (ValueError, IndexError):
        return None


def _safe_cut(messages: list, upto: Optional[int]) -> list:
    """Slice messages[:upto] without leaving a dangling AIMessage whose
    tool_calls got their ToolMessage responses cut off (providers reject
    unpaired tool calls on resume)."""
    sliced = list(messages) if upto is None else list(messages[:upto])
    while sliced:
        last = sliced[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            sliced.pop()
        else:
            break
    return sliced


def _substitute(obj: Any, mapping: Dict[str, str]) -> Any:
    """Recursively replace {{name}} placeholders in every string, for the
    registered variable names only (chat content is full of literal braces —
    str.format would explode)."""
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = re.sub(r"\{\{" + re.escape(k) + r"\}\}", v, obj)
        return obj
    if isinstance(obj, list):
        return [_substitute(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, mapping) for k, v in obj.items()}
    return obj


def _messages_to_dicts(messages: list) -> List[dict]:
    """Serialize langchain messages to the History.to_json message shape."""
    return History(messages=list(messages)).to_json()["messages"]


def _dicts_to_history(message_dicts: List[dict], metadata: Optional[dict] = None) -> History:
    return History.from_json({
        "messages": message_dicts,
        "created_datetime": datetime.now().isoformat(),
        "metadata": metadata or {},
    })


def _create_conversation(
    title: str,
    first_history_id: str,
    agent_name: str,
    tags: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a conversation record in ConversationManager's exact on-disk
    shape, but with a COLLISION-PROOF id. ConversationManager's own
    _generate_conversation_id is second-granular, so start_chat silently
    OVERWRITES any conversation created in the same second — a fork/rehydrate
    typically lands in the same second as its source. The uuid suffix keeps
    _get_conversation_file_path's month/day parsing (parts[1:3]) intact."""
    conversation_id = f"{_now_stamp()}_{uuid.uuid4().hex[:4]}"
    now = datetime.now()
    data = {
        "conversation_id": conversation_id,
        "title": title,
        "created_datetime": now.isoformat(),
        "last_updated": now.isoformat(),
        "history_chain": [first_history_id],
        "metadata": {
            "agent_name": agent_name,
            "total_exchanges": 1,
            "tags": tags or [],
            **(extra_metadata or {}),
        },
    }
    path = ConversationManager._get_conversation_file_path(conversation_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def _templates_dir() -> str:
    d = os.path.join(EnvConfigUtil.get_heaven_data_dir(), "chat_templates")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# fork
# --------------------------------------------------------------------------

def fork_history(
    history_id: str,
    upto: Optional[int] = None,
    agent_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    keep_agent_status: bool = False,
) -> History:
    """Fork a saved History into a NEW History (source untouched).

    Args:
        history_id: source history to fork.
        upto: copy only messages[:upto] (None = full copy). The cut is
            tool-call-safe: a trailing AIMessage with unpaired tool_calls is
            trimmed off.
        agent_name: agent to save the fork under (default: the source's
            owning agent).
        metadata: extra metadata merged onto the fork.
        keep_agent_status: forks are new threads, so agent_status is dropped
            unless this is True.

    Returns:
        The saved fork (its .history_id is continuable via
        History.load_from_id / BaseHeavenAgent(history_id=...)).
    """
    source = _load_source_history(history_id)
    agent = agent_name or _owning_agent(source)
    if not agent:
        raise ValueError(
            f"Could not derive owning agent for {history_id}; pass agent_name."
        )
    nm = normalize_agent_name(agent)

    sliced = _safe_cut(source.messages, upto)
    fork_meta = dict(source.metadata or {})
    fork_meta.update({
        "forked_from": history_id,
        "fork_point": len(sliced),
        "forked_at": datetime.now().isoformat(),
    })
    fork_meta.update(metadata or {})

    fork = History(
        messages=[m for m in sliced],
        metadata=fork_meta,
        agent_status=source.agent_status if keep_agent_status else None,
        history_id=f"{_now_stamp()}_{nm}_fork_{uuid.uuid4().hex[:6]}",
    )
    fork.save(nm)
    return fork


def fork_conversation(
    conversation_id: str,
    at_history_id: Optional[str] = None,
    upto: Optional[int] = None,
    title: Optional[str] = None,
    agent_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fork a conversation: fork its (latest or given) history, then start a
    NEW conversation whose history_chain begins at the fork. Lineage is
    recorded in the new conversation's metadata.

    Returns {"conversation_id", "history_id", "title", "fork_point"}.
    """
    source_conv = ConversationManager.load_conversation(conversation_id)
    if not source_conv:
        raise FileNotFoundError(f"Conversation {conversation_id} not found")

    source_hid = at_history_id or get_latest_history(conversation_id)
    if not source_hid:
        raise ValueError(f"Conversation {conversation_id} has no histories")

    fork = fork_history(source_hid, upto=upto, agent_name=agent_name)
    fork_title = title or f"Fork of {source_conv.get('title', conversation_id)}"
    agent = agent_name or source_conv.get("metadata", {}).get("agent_name", "agent")

    new_conv = _create_conversation(
        fork_title, fork.history_id, agent, tags,
        extra_metadata={
            "forked_from_conversation": conversation_id,
            "forked_from_history": source_hid,
            "fork_point": fork.metadata.get("fork_point"),
        },
    )

    return {
        "conversation_id": new_conv["conversation_id"],
        "history_id": fork.history_id,
        "title": fork_title,
        "fork_point": fork.metadata.get("fork_point"),
    }


# convenience alias matching the start_chat/continue_chat family
fork_chat = fork_conversation


# --------------------------------------------------------------------------
# ChatTemplate
# --------------------------------------------------------------------------

class ChatTemplate(BaseModel):
    """A frozen, reusable chat context — a conversation lifted out of its run.

    messages are stored in the History.to_json message shape; string contents
    may carry {{variable}} placeholders filled at rehydrate/render time.
    """

    template_id: Optional[str] = None
    name: str
    description: str = ""
    created_datetime: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = Field(default_factory=list)
    messages: List[dict] = Field(default_factory=list)
    variables: Dict[str, str] = Field(default_factory=dict)  # name -> default
    source: Dict[str, Any] = Field(default_factory=dict)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_history(
        cls,
        history: Union[History, str],
        name: str,
        upto: Optional[int] = None,
        include_system: bool = False,
        description: str = "",
        tags: Optional[List[str]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> "ChatTemplate":
        """Freeze a History (object or history_id) as a template.

        include_system=False (default) drops a leading SystemMessage: on
        rehydrate, BaseHeavenAgent stamps the live agent's own system prompt
        at index 0 anyway.
        """
        if isinstance(history, str):
            source_ref: Dict[str, Any] = {"history_id": history}
            history = _load_source_history(history)
        else:
            source_ref = {"history_id": history.history_id}

        sliced = _safe_cut(history.messages, upto)
        if not include_system and sliced and type(sliced[0]).__name__ == "SystemMessage":
            sliced = sliced[1:]
        source_ref["fork_point"] = len(sliced)

        return cls(
            name=name,
            description=description,
            tags=tags or [],
            messages=_messages_to_dicts(sliced),
            variables=variables or {},
            source=source_ref,
        )

    @classmethod
    def from_conversation(
        cls,
        conversation_id: str,
        name: str,
        upto: Optional[int] = None,
        include_system: bool = False,
        description: str = "",
        tags: Optional[List[str]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> "ChatTemplate":
        hid = get_latest_history(conversation_id)
        if not hid:
            raise ValueError(f"Conversation {conversation_id} has no histories")
        tmpl = cls.from_history(
            hid, name, upto=upto, include_system=include_system,
            description=description, tags=tags, variables=variables,
        )
        tmpl.source["conversation_id"] = conversation_id
        return tmpl

    # -- rendering / composition ---------------------------------------------

    def _merged_vars(self, variables: Optional[Dict[str, str]]) -> Dict[str, str]:
        merged = dict(self.variables)
        merged.update(variables or {})
        return merged

    def render_messages(self, variables: Optional[Dict[str, str]] = None) -> List[dict]:
        """Message dicts with {{variable}} placeholders substituted."""
        return _substitute(self.messages, self._merged_vars(variables))

    def to_history(self, variables: Optional[Dict[str, str]] = None) -> History:
        """Instantiate as an (unsaved) History."""
        return _dicts_to_history(
            self.render_messages(variables),
            metadata={"rehydrated_from_template": self.template_id or self.name},
        )

    def to_parsed_messages(self, variables: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Export as sdna ParsedMessage-shaped dicts: {index, role, content}.
        This is the compose seam toward sdna.context_engineering — plain
        dicts, no import either direction."""
        out = []
        for i, m in enumerate(self.render_messages(variables)):
            role = _ROLE_BY_TYPE.get(m.get("type"), "assistant")
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            out.append({"index": i, "role": role, "content": content})
        return out

    def to_context_dict(self, variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Export as a named-section context dict consumable by
        sdna.context_engineering.ContextEngineeringLib.inject(context=...)."""
        return {
            "chat_template": self.name,
            "description": self.description,
            "conversation": [
                f"{p['role']}: {p['content']}" for p in self.to_parsed_messages(variables)
            ],
        }

    def compose(self, *others: "ChatTemplate", name: Optional[str] = None) -> "ChatTemplate":
        """Concatenate this template with others (in order) into a new
        template. Variables merge left-to-right (later wins)."""
        parts = (self,) + others
        variables: Dict[str, str] = {}
        messages: List[dict] = []
        for p in parts:
            variables.update(p.variables)
            messages.extend(p.messages)
        return ChatTemplate(
            name=name or " + ".join(p.name for p in parts),
            description=f"Composed from: {', '.join(p.name for p in parts)}",
            tags=sorted({t for p in parts for t in p.tags}),
            messages=messages,
            variables=variables,
            source={"composed_from": [p.template_id or p.name for p in parts]},
        )

    # -- persistence -----------------------------------------------------------

    def save(self) -> str:
        """Persist to $HEAVEN_DATA_DIR/chat_templates/{template_id}.json."""
        if not self.template_id:
            self.template_id = (
                f"tmpl_{_now_stamp()}_{_slugify(self.name)}_{uuid.uuid4().hex[:4]}"
            )
        path = os.path.join(_templates_dir(), f"{self.template_id}.json")
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
        return self.template_id

    # -- rehydration -----------------------------------------------------------

    def rehydrate(
        self,
        agent_name: str,
        variables: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        start_conversation: bool = True,
    ) -> Dict[str, Any]:
        """Instantiate this template as a LIVE saved History (+ optionally a
        new conversation). The returned history_id is ready for
        BaseHeavenAgent(history_id=...)."""
        nm = normalize_agent_name(agent_name)
        hist = self.to_history(variables)
        hist.history_id = f"{_now_stamp()}_{nm}_rehydrated_{uuid.uuid4().hex[:6]}"
        hist.save(nm)

        conversation_id = None
        if start_conversation:
            conv = _create_conversation(
                title or f"From template: {self.name}", hist.history_id, nm, tags,
                extra_metadata={
                    "rehydrated_from_template": self.template_id or self.name,
                },
            )
            conversation_id = conv["conversation_id"]

        return {
            "history_id": hist.history_id,
            "conversation_id": conversation_id,
            "template_id": self.template_id,
            "message_count": len(hist.messages),
        }


# --------------------------------------------------------------------------
# template store (module-level convenience API, conversations.py style)
# --------------------------------------------------------------------------

def save_chat_template(template: ChatTemplate) -> str:
    """Persist a template; returns its template_id."""
    return template.save()


def get_chat_template(name_or_id: str) -> Optional[ChatTemplate]:
    """Load a template by template_id (exact) or name (newest match)."""
    d = _templates_dir()
    exact = os.path.join(d, f"{name_or_id}.json")
    if os.path.exists(exact):
        with open(exact) as f:
            return ChatTemplate(**json.load(f))
    matches = [t for t in _iter_templates() if t.name == name_or_id]
    if matches:
        return sorted(matches, key=lambda t: t.created_datetime, reverse=True)[0]
    return None


def _iter_templates() -> List[ChatTemplate]:
    out = []
    for fp in glob.glob(os.path.join(_templates_dir(), "*.json")):
        try:
            with open(fp) as f:
                out.append(ChatTemplate(**json.load(f)))
        except Exception as e:
            print(f"Error loading chat template {fp}: {e}")
    return out


def list_chat_templates(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Summaries (no messages), newest first."""
    templates = sorted(_iter_templates(), key=lambda t: t.created_datetime, reverse=True)
    if limit:
        templates = templates[:limit]
    return [
        {
            "template_id": t.template_id,
            "name": t.name,
            "description": t.description,
            "created_datetime": t.created_datetime,
            "tags": t.tags,
            "message_count": len(t.messages),
            "variables": t.variables,
            "source": t.source,
        }
        for t in templates
    ]


def search_chat_templates(query: str) -> List[Dict[str, Any]]:
    q = query.lower()
    return [
        s for s in list_chat_templates()
        if q in s["name"].lower()
        or q in s["description"].lower()
        or any(q in t.lower() for t in s["tags"])
    ]


def delete_chat_template(name_or_id: str) -> bool:
    tmpl = get_chat_template(name_or_id)
    if not tmpl or not tmpl.template_id:
        return False
    path = os.path.join(_templates_dir(), f"{tmpl.template_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def compose_chat_templates(
    names_or_ids: List[str], name: str, save: bool = True,
) -> ChatTemplate:
    """Compose stored templates (in order) into a new template."""
    loaded = []
    for ref in names_or_ids:
        t = get_chat_template(ref)
        if not t:
            raise FileNotFoundError(f"Chat template not found: {ref}")
        loaded.append(t)
    composed = loaded[0].compose(*loaded[1:], name=name) if len(loaded) > 1 else loaded[0].model_copy(
        update={"template_id": None, "name": name}
    )
    if save:
        composed.save()
    return composed


def rehydrate_chat(
    name_or_id: str,
    agent_name: str,
    variables: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One-call rehydration: template name/id -> live {history_id, conversation_id}."""
    tmpl = get_chat_template(name_or_id)
    if not tmpl:
        raise FileNotFoundError(f"Chat template not found: {name_or_id}")
    return tmpl.rehydrate(agent_name, variables=variables, title=title, tags=tags)
