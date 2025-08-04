#!/usr/bin/env python3
"""
HEAVEN Conversation Management Utilities

These utilities manage conversations as collections of related history snapshots.
Each conversation tracks the terminal history_ids from continued chains.

Directory structure:
/conversations/{month}/{day}/{datetime}.json

Each conversation file contains:
{
  "conversation_id": "2025_08_04_17_30_00",
  "title": "User-provided title or auto-generated",
  "created_datetime": "2025-08-04T17:30:00.123456",
  "last_updated": "2025-08-04T17:45:15.789012",
  "history_chain": [
    "2025_08_04_17_30_00_agent_continued_3",  // Last in continuation chain
    "2025_08_04_17_45_15_agent"              // Single history (no continuations)
  ],
  "metadata": {
    "agent_name": "ChatAgent",
    "total_exchanges": 5,
    "tags": ["astronomy", "learning"]
  }
}

NOTE: These utilities are designed to be moved to heaven-framework eventually.
"""

import os
import json
import glob
from datetime import datetime
from typing import Optional, List, Dict, Any
from heaven_base.utils.get_env_value import EnvConfigUtil


class ConversationManager:
    """Manages HEAVEN conversations as collections of history snapshots."""
    
    @staticmethod
    def _get_conversations_dir() -> str:
        """Get the conversations directory path."""
        data_dir = EnvConfigUtil.get_heaven_data_dir()
        conversations_dir = os.path.join(data_dir, "conversations")
        os.makedirs(conversations_dir, exist_ok=True)
        return conversations_dir
    
    @staticmethod
    def _generate_conversation_id() -> str:
        """Generate a conversation ID based on current datetime."""
        now = datetime.now()
        return now.strftime("%Y_%m_%d_%H_%M_%S")
    
    @staticmethod
    def _get_conversation_file_path(conversation_id: str) -> str:
        """Get the file path for a conversation ID."""
        conversations_dir = ConversationManager._get_conversations_dir()
        
        # Extract month and day from conversation_id (format: YYYY_MM_DD_HH_MM_SS)
        parts = conversation_id.split("_")
        if len(parts) >= 3:
            month = parts[1]  # MM
            day = parts[2]    # DD
        else:
            # Fallback to current month/day
            now = datetime.now()
            month = now.strftime("%m")
            day = now.strftime("%d")
        
        # Create directory structure
        conv_path = os.path.join(conversations_dir, month, day)
        os.makedirs(conv_path, exist_ok=True)
        
        return os.path.join(conv_path, f"{conversation_id}.json")
    
    @staticmethod
    def start_conversation(title: str, first_history_id: str, agent_name: str, 
                          tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Start a new conversation with the first history.
        
        Args:
            title: Human-readable conversation title
            first_history_id: ID of the first history in this conversation
            agent_name: Name of the agent involved
            tags: Optional tags for categorization
            
        Returns:
            Dictionary with conversation data
        """
        conversation_id = ConversationManager._generate_conversation_id()
        now = datetime.now()
        
        conversation_data = {
            "conversation_id": conversation_id,
            "title": title,
            "created_datetime": now.isoformat(),
            "last_updated": now.isoformat(),
            "history_chain": [first_history_id],
            "metadata": {
                "agent_name": agent_name,
                "total_exchanges": 1,
                "tags": tags or []
            }
        }
        
        # Save to file
        file_path = ConversationManager._get_conversation_file_path(conversation_id)
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        return conversation_data
    
    @staticmethod
    def continue_conversation(conversation_id: str, new_history_id: str) -> Dict[str, Any]:
        """
        Continue an existing conversation by adding a new history.
        
        Args:
            conversation_id: ID of the conversation to continue
            new_history_id: ID of the new history to add
            
        Returns:
            Updated conversation data
            
        Raises:
            FileNotFoundError: If conversation doesn't exist
        """
        file_path = ConversationManager._get_conversation_file_path(conversation_id)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Conversation {conversation_id} not found")
        
        # Load existing conversation
        with open(file_path, 'r') as f:
            conversation_data = json.load(f)
        
        # Add new history to chain
        conversation_data["history_chain"].append(new_history_id)
        conversation_data["last_updated"] = datetime.now().isoformat()
        conversation_data["metadata"]["total_exchanges"] += 1
        
        # Save updated conversation
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        return conversation_data
    
    @staticmethod
    def load_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to load
            
        Returns:
            Conversation data dictionary, or None if not found
        """
        try:
            file_path = ConversationManager._get_conversation_file_path(conversation_id)
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return None
    
    @staticmethod
    def list_conversations(limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all conversations, sorted by last_updated (newest first).
        
        Args:
            limit: Optional limit on number of conversations to return
            
        Returns:
            List of conversation data dictionaries
        """
        conversations_dir = ConversationManager._get_conversations_dir()
        
        # Find all conversation files
        pattern = os.path.join(conversations_dir, "**", "*.json")
        conv_files = glob.glob(pattern, recursive=True)
        
        conversations = []
        for file_path in conv_files:
            try:
                with open(file_path, 'r') as f:
                    conv_data = json.load(f)
                    conversations.append(conv_data)
            except Exception as e:
                print(f"Error loading conversation file {file_path}: {e}")
                continue
        
        # Sort by last_updated (newest first)
        conversations.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        
        if limit:
            conversations = conversations[:limit]
        
        return conversations
    
    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """
        Delete a conversation file.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            file_path = ConversationManager._get_conversation_file_path(conversation_id)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    @staticmethod
    def get_conversation_histories(conversation_id: str) -> List[str]:
        """
        Get all history IDs from a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of history IDs in the conversation chain
        """
        conversation_data = ConversationManager.load_conversation(conversation_id)
        if not conversation_data:
            return []
        
        return conversation_data.get("history_chain", [])
    
    @staticmethod
    def get_conversation_latest_history(conversation_id: str) -> Optional[str]:
        """
        Get the latest/terminal history ID from a conversation for context loading.
        This avoids loading redundant intermediate snapshots.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Latest history ID (most complete snapshot), or None if not found
        """
        conversation_data = ConversationManager.load_conversation(conversation_id)
        if not conversation_data or not conversation_data.get("history_chain"):
            return None
        
        # Return the last history ID (most complete snapshot)
        return conversation_data["history_chain"][-1]
    
    @staticmethod
    def search_conversations(query: str, search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search conversations by text in specified fields.
        
        Args:
            query: Search query string
            search_fields: Fields to search in (default: ["title", "tags"])
            
        Returns:
            List of matching conversation data dictionaries
        """
        if search_fields is None:
            search_fields = ["title", "tags"]
        
        query_lower = query.lower()
        all_conversations = ConversationManager.list_conversations()
        
        matching_conversations = []
        for conv in all_conversations:
            match_found = False
            
            for field in search_fields:
                if field == "title" and query_lower in conv.get("title", "").lower():
                    match_found = True
                    break
                elif field == "tags":
                    tags = conv.get("metadata", {}).get("tags", [])
                    if any(query_lower in tag.lower() for tag in tags):
                        match_found = True
                        break
                elif field in conv and query_lower in str(conv[field]).lower():
                    match_found = True
                    break
            
            if match_found:
                matching_conversations.append(conv)
        
        return matching_conversations


# Convenience functions for easier usage
def start_chat(title: str, first_history_id: str, agent_name: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function to start a new conversation."""
    return ConversationManager.start_conversation(title, first_history_id, agent_name, tags)


def continue_chat(conversation_id: str, new_history_id: str) -> Dict[str, Any]:
    """Convenience function to continue an existing conversation."""
    return ConversationManager.continue_conversation(conversation_id, new_history_id)


def load_chat(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to load a conversation."""
    return ConversationManager.load_conversation(conversation_id)


def list_chats(limit: Optional[int] = 10) -> List[Dict[str, Any]]:
    """Convenience function to list conversations."""
    return ConversationManager.list_conversations(limit)


def search_chats(query: str) -> List[Dict[str, Any]]:
    """Convenience function to search conversations."""
    return ConversationManager.search_conversations(query)


def get_latest_history(conversation_id: str) -> Optional[str]:
    """Convenience function to get the latest history ID from a conversation."""
    return ConversationManager.get_conversation_latest_history(conversation_id)


# Example usage and testing
if __name__ == "__main__":
    # Test the conversation management system
    print("🧪 Testing HEAVEN Conversation Management")
    print("=" * 45)
    
    # Start a new conversation
    conv_data = start_chat(
        title="Astronomy Discussion", 
        first_history_id="2025_08_04_17_30_00_agent",
        agent_name="ChatAgent",
        tags=["astronomy", "learning"]
    )
    print(f"✅ Started conversation: {conv_data['conversation_id']}")
    
    # Continue the conversation
    continued = continue_chat(
        conv_data['conversation_id'], 
        "2025_08_04_17_35_15_agent_continued_1"
    )
    print(f"✅ Continued conversation: {len(continued['history_chain'])} histories")
    
    # List conversations
    conversations = list_chats(limit=5)
    print(f"✅ Found {len(conversations)} conversations")
    
    # Search conversations
    matches = search_chats("astronomy")
    print(f"✅ Found {len(matches)} conversations matching 'astronomy'")