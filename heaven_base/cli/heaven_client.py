#!/usr/bin/env python3
"""
HEAVEN Client - Non-Interactive HTTP Client for HEAVEN Framework

Programmatic API for interacting with the HEAVEN HTTP server.
No input() calls, no interactive prompts — designed for automation,
agent-to-agent communication, scripting, and embedding.

Usage:
    # Async context manager (recommended)
    async with HeavenClient(agent_config=config) as client:
        result = await client.send_message("Hello!")
        print(result.agent_response)

    # Sync one-shot
    client = HeavenClient(agent_config=config)
    result = client.send_message_sync("Hello!")

    # CLI one-shot
    $ python -m heaven_base.cli.heaven_client --agent MyAgent --message "Hello"
"""

import asyncio
import aiohttp
import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from datetime import datetime

from ..baseheavenagent import HeavenAgentConfig
from ..memory.conversations import (
    ConversationManager,
    start_chat,
    continue_chat,
    list_chats,
    search_chats,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured result types
# ---------------------------------------------------------------------------

@dataclass
class HeavenEvent:
    """A single parsed SSE event from the HEAVEN server."""
    event_type: str
    data: Dict[str, Any]
    history_id: Optional[str] = None
    raw: Optional[str] = None


@dataclass
class MessageResult:
    """Structured result of a send_message call."""
    agent_response: str = ""
    thinking: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    events: List[HeavenEvent] = field(default_factory=list)
    history_id: Optional[str] = None
    conversation_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "success": self.success,
            "agent_response": self.agent_response,
            "thinking": self.thinking,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "history_id": self.history_id,
            "conversation_id": self.conversation_id,
            "error": self.error,
            "event_count": len(self.events),
        }


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class HeavenClient:
    """Non-interactive HTTP client for the HEAVEN server.

    Provides the same capabilities as HeavenCLI but as a programmatic API:
    - Session management
    - Message send + SSE event collection
    - Conversation CRUD (create, continue, list, search)
    - Async-first with sync convenience wrappers
    """

    def __init__(
        self,
        agent_config: Union[HeavenAgentConfig, str, type],
        # TRIGGERS: CAVE/sancrev via HTTP to localhost:8080
        server_url: str = "http://localhost:8080",
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        history_id: Optional[str] = None,
        event_callback: Optional[Callable[[HeavenEvent], None]] = None,
    ):
        """Initialize the client.

        Args:
            agent_config: HeavenAgentConfig instance, path string, or agent class.
            server_url: URL of HEAVEN HTTP server.
            session_id: Existing session to reuse (None = auto-create).
            conversation_id: Existing conversation to continue.
            history_id: Existing history chain to continue.
            event_callback: Optional callback invoked for each SSE event as it
                arrives. Useful for streaming output without blocking on the
                full result.
        """
        # --- resolve agent config ---
        if isinstance(agent_config, HeavenAgentConfig):
            self.agent_config = agent_config
            self.agent_name = agent_config.name
            self.tools = agent_config.tools
        elif isinstance(agent_config, str):
            raise NotImplementedError("Config file loading not yet implemented")
        elif isinstance(agent_config, type):
            self.agent_config = None
            self.agent_name = agent_config.__name__
            self.tools = getattr(agent_config, "tools", [])
        else:
            raise ValueError(
                "agent_config must be HeavenAgentConfig, path string, or agent class"
            )

        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.history_id = history_id
        self.conversation_id = conversation_id
        self.conversation_title: Optional[str] = None
        self.event_callback = event_callback

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "HeavenClient":
        await self.ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass  # session lives on the server; nothing to close client-side

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def ensure_session(self) -> str:
        """Ensure we have an active session, creating one if needed.

        Returns:
            The session_id.
        """
        if self.session_id:
            return self.session_id

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.server_url}/api/session/start"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.session_id = data.get("session_id")
                        logger.info("Session started: %s", self.session_id)
                        return self.session_id
                    else:
                        raise ConnectionError(
                            f"Failed to start session: HTTP {resp.status}"
                        )
            except aiohttp.ClientConnectorError as exc:
                raise ConnectionError(
                    f"Cannot connect to HEAVEN server at {self.server_url}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Core messaging
    # ------------------------------------------------------------------

    async def send_message(
        self,
        message: str,
        *,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        save_conversation: bool = True,
    ) -> MessageResult:
        """Send a message and collect the full response.

        Args:
            message: The user message to send.
            title: Conversation title (used when auto-creating a conversation).
            tags: Tags for conversation metadata.
            save_conversation: If True, auto-create/update conversation records.

        Returns:
            MessageResult with the complete structured response.
        """
        await self.ensure_session()

        result = MessageResult()

        # --- POST the message ---
        request_data = {
            "text": message,
            "agent": self.agent_name,
            "tools": [str(tool) for tool in self.tools] if self.tools else [],
            "history_id": self.history_id,
        }

        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{self.server_url}/api/session/{self.session_id}/message",
                json=request_data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    result.error = f"Message failed ({resp.status}): {error_text}"
                    logger.error(result.error)
                    return result

            # --- Stream SSE events ---
            async with http.get(
                f"{self.server_url}/api/session/{self.session_id}/stream"
            ) as stream_resp:
                if stream_resp.status != 200:
                    result.error = f"Stream failed: HTTP {stream_resp.status}"
                    return result

                async for line in stream_resp.content:
                    if not line:
                        continue

                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue

                    try:
                        event_data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        logger.debug("Skipped malformed SSE event")
                        continue

                    event = HeavenEvent(
                        event_type=event_data.get("event_type", "UNKNOWN"),
                        data=event_data.get("data", {}),
                        history_id=event_data.get("history_id"),
                        raw=line_str,
                    )

                    # Track history_id
                    if event.history_id:
                        self.history_id = event.history_id
                        result.history_id = event.history_id

                    # Fire callback if provided
                    if self.event_callback:
                        try:
                            self.event_callback(event)
                        except Exception as cb_err:
                            logger.warning("Event callback error: %s", cb_err)

                    result.events.append(event)

                    # Accumulate by type
                    if event.event_type == "AGENT_MESSAGE":
                        content = event.data.get("content", "")
                        result.agent_response += content
                    elif event.event_type == "THINKING":
                        content = event.data.get("content", "")
                        result.thinking += content
                    elif event.event_type == "TOOL_USE":
                        result.tool_calls.append(event.data)
                    elif event.event_type == "TOOL_RESULT":
                        result.tool_results.append(event.data)
                    elif event.event_type == "CONVERSATION_COMPLETE":
                        break

        # --- Conversation bookkeeping ---
        if save_conversation:
            if not self.conversation_id and self.history_id:
                auto_title = title or (
                    self.conversation_title
                    or f"Chat with {self.agent_name} - "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                try:
                    conv_data = start_chat(
                        title=auto_title,
                        first_history_id=self.history_id,
                        agent_name=self.agent_name,
                        tags=tags or [],
                    )
                    self.conversation_id = conv_data["conversation_id"]
                    self.conversation_title = auto_title
                    logger.info("Conversation created: %s", self.conversation_id)
                except Exception as exc:
                    logger.warning("Could not create conversation: %s", exc)

            elif self.conversation_id and self.history_id:
                try:
                    continue_chat(self.conversation_id, self.history_id)
                except Exception as exc:
                    logger.warning("Could not update conversation: %s", exc)

        result.conversation_id = self.conversation_id
        return result

    async def send_messages(
        self,
        messages: List[str],
        **kwargs,
    ) -> List[MessageResult]:
        """Send multiple messages sequentially in the same conversation.

        Each message waits for the previous to complete, maintaining history.

        Args:
            messages: List of user messages.
            **kwargs: Passed to send_message().

        Returns:
            List of MessageResult, one per message.
        """
        results = []
        for msg in messages:
            result = await self.send_message(msg, **kwargs)
            results.append(result)
            if not result.success:
                break  # stop on first error
        return results

    # ------------------------------------------------------------------
    # Event streaming (async generator)
    # ------------------------------------------------------------------

    async def stream_message(
        self,
        message: str,
    ) -> AsyncIterator[HeavenEvent]:
        """Send a message and yield events as they arrive.

        This is the streaming counterpart to send_message(). Each event
        is yielded as soon as it is received from the SSE stream.

        Args:
            message: The user message to send.

        Yields:
            HeavenEvent instances.
        """
        await self.ensure_session()

        request_data = {
            "text": message,
            "agent": self.agent_name,
            "tools": [str(tool) for tool in self.tools] if self.tools else [],
            "history_id": self.history_id,
        }

        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{self.server_url}/api/session/{self.session_id}/message",
                json=request_data,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(
                        f"Message failed ({resp.status}): {error_text}"
                    )

            async with http.get(
                f"{self.server_url}/api/session/{self.session_id}/stream"
            ) as stream_resp:
                if stream_resp.status != 200:
                    raise RuntimeError(f"Stream failed: HTTP {stream_resp.status}")

                async for line in stream_resp.content:
                    if not line:
                        continue
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue

                    try:
                        event_data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        continue

                    event = HeavenEvent(
                        event_type=event_data.get("event_type", "UNKNOWN"),
                        data=event_data.get("data", {}),
                        history_id=event_data.get("history_id"),
                        raw=line_str,
                    )

                    if event.history_id:
                        self.history_id = event.history_id

                    yield event

                    if event.event_type == "CONVERSATION_COMPLETE":
                        return

    # ------------------------------------------------------------------
    # Conversation management (no input() anywhere)
    # ------------------------------------------------------------------

    async def new_conversation(
        self,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Reset state to start a new conversation.

        Does NOT prompt the user — just resets internal state.
        The conversation record is created on first send_message().

        Args:
            title: Optional conversation title.
            tags: Optional tags for metadata.
        """
        self.conversation_id = None
        self.history_id = None
        self.conversation_title = title

    async def continue_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """Load an existing conversation by ID.

        Args:
            conversation_id: The conversation to continue.

        Returns:
            True if the conversation was loaded successfully.
        """
        self.conversation_id = conversation_id
        latest_history_id = ConversationManager.get_conversation_latest_history(
            conversation_id
        )
        if latest_history_id:
            self.history_id = latest_history_id
        return True

    @staticmethod
    def list_conversations(limit: int = 15) -> List[Dict[str, Any]]:
        """List recent conversations.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts.
        """
        return list_chats(limit=limit)

    @staticmethod
    def search_conversations(query: str) -> List[Dict[str, Any]]:
        """Search conversations by query string.

        Args:
            query: Search query.

        Returns:
            List of matching conversation dicts.
        """
        return search_chats(query)

    # ------------------------------------------------------------------
    # Sync convenience wrappers
    # ------------------------------------------------------------------

    def send_message_sync(self, message: str, **kwargs) -> MessageResult:
        """Synchronous wrapper for send_message()."""
        return asyncio.run(self.send_message(message, **kwargs))

    def send_messages_sync(
        self, messages: List[str], **kwargs
    ) -> List[MessageResult]:
        """Synchronous wrapper for send_messages()."""
        return asyncio.run(self.send_messages(messages, **kwargs))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(
    agent_config: Union[HeavenAgentConfig, str, type],
    # TRIGGERS: CAVE/sancrev via HTTP to localhost:8080
    server_url: str = "http://localhost:8080",
    **kwargs,
) -> HeavenClient:
    """Create a configured HeavenClient instance.

    Args:
        agent_config: HeavenAgentConfig instance, path to config, or agent class.
        server_url: URL of HEAVEN HTTP server.
        **kwargs: Forwarded to HeavenClient constructor.

    Returns:
        Configured HeavenClient instance.
    """
    return HeavenClient(
        agent_config=agent_config,
        server_url=server_url,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry point: python -m heaven_base.cli.heaven_client
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heaven-client",
        description="HEAVEN non-interactive client — send messages from the shell",
    )
    parser.add_argument(
        # TRIGGERS: CAVE/sancrev via HTTP to localhost:8080
        "--server", default="http://localhost:8080", help="HEAVEN server URL"
    )
    parser.add_argument(
        "--agent", required=True, help="Agent name to invoke"
    )
    parser.add_argument(
        "--session-id", default=None, help="Existing session ID to reuse"
    )
    parser.add_argument(
        "--conversation-id", default=None, help="Existing conversation to continue"
    )
    parser.add_argument(
        "--history-id", default=None, help="Existing history ID to continue"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- send ---
    send_p = sub.add_parser("send", help="Send a message and print the response")
    send_p.add_argument("message", help="The message to send")
    send_p.add_argument("--title", default=None, help="Conversation title")
    send_p.add_argument("--tags", default=None, help="Comma-separated tags")
    send_p.add_argument(
        "--json", dest="output_json", action="store_true",
        help="Output full JSON result instead of just the agent response",
    )

    # --- list ---
    list_p = sub.add_parser("list", help="List recent conversations")
    list_p.add_argument("--limit", type=int, default=15)

    # --- search ---
    search_p = sub.add_parser("search", help="Search conversations")
    search_p.add_argument("query", help="Search query")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "send":
        # Build a minimal "type-like" agent config stand-in
        # (we only need agent_name and empty tools for the HTTP request)
        class _CLIAgent:
            pass
        _CLIAgent.__name__ = args.agent
        _CLIAgent.tools = []

        client = HeavenClient(
            agent_config=_CLIAgent,
            server_url=args.server,
            session_id=args.session_id,
            conversation_id=args.conversation_id,
            history_id=args.history_id,
        )

        tags = (
            [t.strip() for t in args.tags.split(",") if t.strip()]
            if args.tags
            else None
        )

        result = client.send_message_sync(
            args.message, title=args.title, tags=tags
        )

        if args.output_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(result.agent_response)
            else:
                print(f"ERROR: {result.error}", file=sys.stderr)
                return 1

    elif args.command == "list":
        convos = HeavenClient.list_conversations(limit=args.limit)
        if not convos:
            print("No conversations found.")
            return 0
        for i, conv in enumerate(convos, 1):
            meta = conv.get("metadata", {})
            print(
                f"{i:2d}. {conv['title']}  "
                f"[{meta.get('total_exchanges', '?')} msgs, "
                f"{conv.get('last_updated', '?')[:19]}]"
            )

    elif args.command == "search":
        results = HeavenClient.search_conversations(args.query)
        if not results:
            print(f"No conversations matching '{args.query}'.")
            return 0
        for i, conv in enumerate(results, 1):
            meta = conv.get("metadata", {})
            print(
                f"{i:2d}. {conv['title']}  "
                f"[{meta.get('total_exchanges', '?')} msgs, "
                f"{conv.get('last_updated', '?')[:19]}]"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
