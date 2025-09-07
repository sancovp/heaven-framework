#!/usr/bin/env python3
"""
HEAVEN CLI - HTTP Client Implementation
"""

import asyncio
import aiohttp
import json
import sys
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from ..baseheavenagent import HeavenAgentConfig
from ..memory.heaven_event import HeavenEvent
from ..memory.conversations import ConversationManager, start_chat, continue_chat, list_chats, search_chats

# Set up logger
logger = logging.getLogger(__name__)


class HeavenCLI:
    """CLI client that talks to HEAVEN HTTP server."""
    
    def __init__(
        self,
        agent_config: Union[HeavenAgentConfig, str, type],
        server_url: str = "http://localhost:8080",
        session_id: Optional[str] = None
    ):
        # Handle different agent_config types
        if isinstance(agent_config, HeavenAgentConfig):
            self.agent_config = agent_config
            self.agent_name = agent_config.name
            self.tools = agent_config.tools
        elif isinstance(agent_config, str):
            # Path to config file - we'll need to implement loading
            raise NotImplementedError("Config file loading not yet implemented")
        elif isinstance(agent_config, type):
            # Agent class - extract name and tools
            self.agent_config = None
            self.agent_name = agent_config.__name__
            # Try to get tools from the agent class if it has them
            self.tools = getattr(agent_config, 'tools', [])
        else:
            raise ValueError("agent_config must be HeavenAgentConfig, path string, or agent class")
            
        self.server_url = server_url.rstrip('/')
        self.session_id = session_id
        self.history_id = None
        self.conversation_id = None
        self.conversation_title = None
        
    async def _ensure_session(self):
        """Ensure we have an active session."""
        if self.session_id:
            return
            
        # Start a new session
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.server_url}/api/session/start") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.session_id = data.get("session_id")
                        logger.info(f"Connected to server (session: {self.session_id})")
                        print(f"📡 Connected to server (session: {self.session_id[:8]}...)")
                    else:
                        logger.error(f"Failed to start session: {resp.status}")
                        raise Exception(f"Failed to start session: {resp.status}")
            except aiohttp.ClientConnectorError:
                raise Exception(f"Cannot connect to HEAVEN server at {self.server_url}. Is it running?")
    
    def _render_agent_message(self, data: Dict[str, Any]) -> str:
        """Render agent message event."""
        content = data.get("content", "")
        return f"\n🤖 Assistant:\n   {content}\n"
    
    def _render_thinking(self, data: Dict[str, Any]) -> str:
        """Render thinking event."""
        content = data.get("content", "")
        return f"\n💭 Thinking:\n   {content}\n"
    
    def _render_tool_use(self, data: Dict[str, Any]) -> str:
        """Render tool use event."""
        tool_name = data.get("name", "unknown")
        tool_input = data.get("input", {})
        input_str = json.dumps(tool_input) if tool_input else "(no input)"
        return f"\n🔧 Tool Call: {tool_name}\n   Input: {input_str}\n"
    
    def _render_tool_result(self, data: Dict[str, Any]) -> str:
        """Render tool result event."""
        output = data.get("output", "")
        # Truncate long outputs
        if len(output) > 500:
            output = output[:500] + "... (truncated)"
        return f"\n📋 Tool Result:\n   {output}\n"
    
    def _render_event(self, event_data: Dict[str, Any]) -> Optional[str]:
        """Render a HEAVEN event for display."""
        event_type = event_data.get("event_type", "UNKNOWN")
        data = event_data.get("data", {})
        
        renderers = {
            "USER_MESSAGE": lambda d: None,  # Don't echo user input
            "AGENT_MESSAGE": self._render_agent_message,
            "THINKING": self._render_thinking,
            "TOOL_USE": self._render_tool_use,
            "TOOL_RESULT": self._render_tool_result,
            "SYSTEM_MESSAGE": lambda d: f"\n🔧 System: {d.get('content', '')}\n",
            "CONVERSATION_COMPLETE": lambda d: None  # Don't render, just signals completion
        }
        
        renderer = renderers.get(event_type)
        return renderer(data) if renderer else None
    
    async def _send_message_and_stream(self, message: str) -> None:
        """Send message and stream back events."""
        await self._ensure_session()
        
        # Prepare request data
        request_data = {
            "text": message,
            "agent": self.agent_name,
            "tools": [str(tool) for tool in self.tools] if self.tools else [],
            "history_id": self.history_id
        }
        
        async with aiohttp.ClientSession() as session:
            # Send message
            async with session.post(
                f"{self.server_url}/api/session/{self.session_id}/message",
                json=request_data
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Message failed ({resp.status}): {error_text}")
                    raise Exception(f"Message failed ({resp.status}): {error_text}")
            
            # Stream events
            logger.debug("Starting event stream")
            print("⚡ Processing...", flush=True)
            
            async with session.get(
                f"{self.server_url}/api/session/{self.session_id}/stream"
            ) as stream_resp:
                if stream_resp.status != 200:
                    raise Exception(f"Stream failed: {stream_resp.status}")
                
                async for line in stream_resp.content:
                    if not line:
                        continue
                        
                    line_str = line.decode('utf-8').strip()
                    if not line_str or not line_str.startswith('data: '):
                        continue
                    
                    try:
                        # Parse SSE data
                        event_data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                        
                        # Extract history_id if present
                        if event_data.get("history_id"):
                            self.history_id = event_data["history_id"]
                            logger.debug(f"Updated history_id: {self.history_id}")
                        
                        # Check if this is a completion event
                        if event_data.get("event_type") == "CONVERSATION_COMPLETE":
                            logger.debug("Received CONVERSATION_COMPLETE, ending stream")
                            break
                        
                        # Render the event
                        rendered = self._render_event(event_data)
                        if rendered:
                            print(rendered, end="", flush=True)
                            
                    except json.JSONDecodeError:
                        logger.debug("Skipped malformed SSE event")
                        continue  # Skip malformed events
                    except Exception as e:
                        logger.error(f"Error processing event: {e}", exc_info=True)
                        print(f"\n❌ Error processing event: {e}")
    
    def _show_main_menu(self):
        """Display main menu for conversation options."""
        print("\n" + "="*60)
        print("🌟 " + " HEAVEN CLI Chat Interface ".center(56) + "🌟")
        print("="*60)
        print(f"\n🤖 Agent: {self.agent_name}")
        if self.tools:
            tool_names = [str(t).split('.')[-1] if hasattr(t, '__name__') else str(t) for t in self.tools]
            print(f"🛠️  Tools: {', '.join(tool_names)}")
        else:
            print("🛠️  Tools: None configured")
        print(f"📡 Server: {self.server_url}")
        print("\n" + "="*60)
        print("📋 CONVERSATION MENU")
        print("="*60)
        print("1️⃣  Start New Conversation")
        print("2️⃣  Continue Existing Conversation")
        print("3️⃣  List Recent Conversations")
        print("4️⃣  Search Conversations")
        print("5️⃣  Exit")
        print("\n" + "="*60 + "\n")

    def _show_chat_header(self):
        """Display chat session header."""
        print("\n" + "="*60)
        if self.conversation_title:
            print(f"💬 {self.conversation_title}")
        else:
            print("💬 Chat Session")
        print(f"🤖 Agent: {self.agent_name}")
        if self.conversation_id:
            print(f"📝 Conversation ID: {self.conversation_id}")
        print("="*60)
        print("💡 Instructions:")
        print("  • Type your message and press Enter")
        print("  • Press Ctrl+C to return to main menu")
        print("\n" + "─"*60 + "\n")

    async def _handle_new_conversation(self):
        """Handle starting a new conversation."""
        print("\n🆕 Starting New Conversation")
        print("─"*40)
        
        title = input("📝 Enter conversation title (or press Enter for auto-title): ").strip()
        if not title:
            title = f"Chat with {self.agent_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        tags_input = input("🏷️  Enter tags (comma-separated, optional): ").strip()
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
        
        self.conversation_title = title
        print(f"\n✅ Ready to start: {title}")
        return tags

    async def _handle_continue_conversation(self):
        """Handle continuing an existing conversation."""
        print("\n📖 Continue Existing Conversation")
        print("─"*40)
        
        # List recent conversations
        recent = list_chats(limit=10)
        if not recent:
            print("❌ No existing conversations found.")
            return False
        
        print("\n📋 Recent Conversations:")
        for i, conv in enumerate(recent, 1):
            print(f"{i:2d}. {conv['title']}")
            print(f"    💬 {conv['metadata']['total_exchanges']} messages | 📅 {conv['last_updated'][:19]}")
        
        try:
            choice = input("\n🔢 Enter conversation number (or 'q' to go back): ").strip()
            if choice.lower() == 'q':
                return False
            
            conv_idx = int(choice) - 1
            if 0 <= conv_idx < len(recent):
                selected_conv = recent[conv_idx]
                self.conversation_id = selected_conv['conversation_id']
                self.conversation_title = selected_conv['title']
                
                # Load the latest history for context
                latest_history_id = ConversationManager.get_conversation_latest_history(self.conversation_id)
                if latest_history_id:
                    self.history_id = latest_history_id
                
                print(f"\n✅ Continuing: {self.conversation_title}")
                return True
            else:
                print("❌ Invalid selection.")
                return False
                
        except ValueError:
            print("❌ Invalid input.")
            return False

    async def _handle_list_conversations(self):
        """Handle listing conversations."""
        print("\n📋 Recent Conversations")
        print("─"*40)
        
        conversations = list_chats(limit=15)
        if not conversations:
            print("❌ No conversations found.")
            return
        
        for i, conv in enumerate(conversations, 1):
            print(f"\n{i:2d}. 📝 {conv['title']}")
            print(f"    🤖 Agent: {conv['metadata']['agent_name']}")
            print(f"    💬 Messages: {conv['metadata']['total_exchanges']}")
            print(f"    📅 Last updated: {conv['last_updated'][:19]}")
            if conv['metadata'].get('tags'):
                print(f"    🏷️  Tags: {', '.join(conv['metadata']['tags'])}")
        
        input("\n⏎ Press Enter to continue...")

    async def _handle_search_conversations(self):
        """Handle searching conversations."""
        print("\n🔍 Search Conversations")
        print("─"*40)
        
        query = input("🔎 Enter search query: ").strip()
        if not query:
            print("❌ No search query provided.")
            return
        
        results = search_chats(query)
        if not results:
            print(f"❌ No conversations found matching '{query}'.")
            return
        
        print(f"\n🎯 Found {len(results)} results for '{query}':")
        for i, conv in enumerate(results, 1):
            print(f"\n{i:2d}. 📝 {conv['title']}")
            print(f"    💬 {conv['metadata']['total_exchanges']} messages | 📅 {conv['last_updated'][:19]}")
            if conv['metadata'].get('tags'):
                print(f"    🏷️  Tags: {', '.join(conv['metadata']['tags'])}")
        
        input("\n⏎ Press Enter to continue...")
    
    async def _chat_loop(self, tags: Optional[List[str]] = None):
        """Run the main chat interaction loop."""
        self._show_chat_header()
        
        print("\n" + "─"*60 + "\n")
        print("💡 Type your message to start the conversation, or 'back' to return to menu")
        print("─"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("💭 Your message: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['back', 'exit', 'quit', 'menu']:
                    print("\n🔙 Returning to main menu...\n")
                    break
                
                # Send message and stream response
                await self._send_message_and_stream(user_input)
                
                # If this was the first message and we got a history_id, create conversation record
                if not self.conversation_id and self.history_id:
                    conv_data = start_chat(
                        title=self.conversation_title or f"Chat with {self.agent_name}",
                        first_history_id=self.history_id,
                        agent_name=self.agent_name,
                        tags=tags or []
                    )
                    self.conversation_id = conv_data['conversation_id']
                    print(f"\n📝 Conversation saved as: {self.conversation_id}")
                
                # Update conversation with new history if we have one
                if self.conversation_id and self.history_id:
                    try:
                        continue_chat(self.conversation_id, self.history_id)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not update conversation: {e}")
                
                print("\n" + "─"*50)
                
            except KeyboardInterrupt:
                print("\n\n🔙 Returning to main menu...\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Continuing...\n")
    
    async def run(self):
        """Run the CLI with main menu system."""
        print("\n🌟 Welcome to HEAVEN CLI Chat! 🌟")
        
        while True:
            try:
                self._show_main_menu()
                choice = input("🔢 Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    # Start new conversation
                    tags = await self._handle_new_conversation()
                    # Reset conversation state
                    self.conversation_id = None
                    self.history_id = None
                    await self._chat_loop(tags)
                    
                elif choice == "2":
                    # Continue existing conversation
                    if await self._handle_continue_conversation():
                        await self._chat_loop()
                    
                elif choice == "3":
                    # List conversations
                    await self._handle_list_conversations()
                    
                elif choice == "4":
                    # Search conversations
                    await self._handle_search_conversations()
                    
                elif choice == "5" or choice.lower() == "quit" or choice.lower() == "exit":
                    # Exit
                    print("\n" + "="*60)
                    print("👋 Thanks for using HEAVEN CLI! Goodbye!")
                    print("="*60 + "\n")
                    break
                    
                else:
                    print("\n❌ Invalid choice. Please select 1-5.\n")
                    
            except KeyboardInterrupt:
                print("\n\n" + "="*60)
                print("👋 Thanks for using HEAVEN CLI! Goodbye!")
                print("="*60 + "\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Returning to main menu...\n")
    
    def run_sync(self):
        """Run the CLI synchronously."""
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            pass


def make_cli(
    agent_config: Union[HeavenAgentConfig, str, type],
    server_url: str = "http://localhost:8080"
) -> HeavenCLI:
    """Create a configured CLI instance.
    
    Args:
        agent_config: HeavenAgentConfig instance, path to config file, or agent class
        server_url: URL of HEAVEN HTTP server
    
    Returns:
        Configured HeavenCLI instance
    """
    return HeavenCLI(
        agent_config=agent_config,
        server_url=server_url
    )