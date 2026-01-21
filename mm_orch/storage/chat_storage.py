"""
Chat Storage Implementation.

This module implements the ChatStorage class for managing conversation
history with support for:
- Session creation and management
- Message persistence
- Sliding window strategy for history retrieval
- File-based persistence

Requirements:
- 4.2: Retrieve conversation history for context
- 4.4: Persist at least 10 recent conversation turns
- 4.5: Sliding window strategy for history management
- 14.1: Persist conversation history to storage

Properties verified:
- Property 7: 对话会话唯一性
- Property 9: 对话历史持久化
- Property 10: 对话历史滑动窗口
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json
import os
import time
import uuid
import threading
from pathlib import Path

from mm_orch.schemas import ChatMessage, ChatSession
from mm_orch.exceptions import StorageError, ValidationError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


# Singleton instance
_chat_storage_instance: Optional['ChatStorage'] = None
_storage_lock = threading.Lock()


def get_chat_storage(storage_path: Optional[str] = None) -> 'ChatStorage':
    """
    Get the singleton ChatStorage instance.
    
    Args:
        storage_path: Optional path for storage. Only used on first call.
    
    Returns:
        ChatStorage singleton instance
    """
    global _chat_storage_instance
    
    with _storage_lock:
        if _chat_storage_instance is None:
            _chat_storage_instance = ChatStorage(storage_path=storage_path)
        return _chat_storage_instance


def reset_chat_storage() -> None:
    """Reset the singleton instance (mainly for testing)."""
    global _chat_storage_instance
    with _storage_lock:
        _chat_storage_instance = None


class ChatStorage:
    """
    Chat history storage manager.
    
    Manages conversation sessions and messages with support for:
    - Creating and retrieving sessions
    - Adding messages to sessions
    - Sliding window history retrieval
    - File-based persistence
    
    Attributes:
        storage_path: Path to the storage directory
        max_history_length: Maximum messages to keep per session
        window_size: Default sliding window size for retrieval
        auto_save: Whether to auto-save after each operation
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_history_length: int = 100,
        window_size: int = 10,
        auto_save: bool = True
    ):
        """
        Initialize the ChatStorage.
        
        Args:
            storage_path: Path to storage directory (default: data/chat_history)
            max_history_length: Maximum messages to keep per session
            window_size: Default sliding window size for retrieval
            auto_save: Whether to auto-save after modifications
        """
        self.storage_path = storage_path or "data/chat_history"
        self.max_history_length = max_history_length
        self.window_size = window_size
        self.auto_save = auto_save
        
        # In-memory session cache
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        
        # Ensure storage directory exists
        self._ensure_storage_dir()
        
        logger.info(
            "ChatStorage initialized",
            storage_path=self.storage_path,
            max_history_length=max_history_length,
            window_size=window_size
        )

    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise StorageError(f"Failed to create storage directory: {e}")
    
    def _get_session_file_path(self, session_id: str) -> str:
        """Get the file path for a session."""
        return os.path.join(self.storage_path, f"{session_id}.json")
    
    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> ChatSession:
        """
        Create a new chat session with a unique ID.
        
        Property 7: 对话会话唯一性 - Each session gets a globally unique ID.
        
        Args:
            metadata: Optional session metadata
        
        Returns:
            New ChatSession with unique session_id
        """
        with self._lock:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Ensure uniqueness (extremely unlikely to collide, but verify)
            while session_id in self._sessions:
                session_id = str(uuid.uuid4())
            
            session = ChatSession(
                session_id=session_id,
                metadata=metadata
            )
            
            self._sessions[session_id] = session
            
            logger.info(f"Created new session: {session_id}")
            
            if self.auto_save:
                self._save_session(session)
            
            return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session identifier
        
        Returns:
            ChatSession if found, None otherwise
        """
        with self._lock:
            # Check in-memory cache first
            if session_id in self._sessions:
                return self._sessions[session_id]
            
            # Try to load from disk
            session = self._load_session(session_id)
            if session:
                self._sessions[session_id] = session
            
            return session
    
    def get_or_create_session(
        self, 
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID to retrieve
            metadata: Metadata for new session if created
        
        Returns:
            Existing or new ChatSession
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(metadata=metadata)
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add a message to a session.
        
        Property 9: 对话历史持久化 - Messages are persisted after adding.
        
        Args:
            session_id: The session to add the message to
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata
        
        Returns:
            The created ChatMessage
        
        Raises:
            ValidationError: If session not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                # Try to load from disk
                session = self._load_session(session_id)
                if session:
                    self._sessions[session_id] = session
                else:
                    raise ValidationError(f"Session not found: {session_id}")
            
            # Create and add message
            message = session.add_message(role=role, content=content, metadata=metadata)
            
            # Apply sliding window if needed
            self._apply_sliding_window(session)
            
            logger.debug(
                f"Added message to session {session_id}",
                role=role,
                content_length=len(content)
            )
            
            if self.auto_save:
                self._save_session(session)
            
            return message
    
    def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a user message to a session."""
        return self.add_message(session_id, "user", content, metadata)
    
    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add an assistant message to a session."""
        return self.add_message(session_id, "assistant", content, metadata)
    
    def add_system_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a system message to a session."""
        return self.add_message(session_id, "system", content, metadata)

    def get_chat_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get chat history for a session with sliding window.
        
        Property 10: 对话历史滑动窗口 - Returns only the most recent N messages.
        
        Args:
            session_id: The session identifier
            limit: Maximum messages to return (default: window_size)
        
        Returns:
            List of recent ChatMessages
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        limit = limit or self.window_size
        return session.get_recent_messages(limit=limit)
    
    def get_full_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get full chat history for a session (up to max_history_length).
        
        Args:
            session_id: The session identifier
        
        Returns:
            List of all stored ChatMessages
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        return session.messages.copy()
    
    def get_history_count(self, session_id: str) -> int:
        """
        Get the number of messages in a session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            Number of messages
        """
        session = self.get_session(session_id)
        return len(session.messages) if session else 0
    
    def _apply_sliding_window(self, session: ChatSession) -> None:
        """
        Apply sliding window to keep history within max_history_length.
        
        Property 10: When history exceeds max length, oldest messages are removed.
        
        Args:
            session: The session to apply sliding window to
        """
        if len(session.messages) > self.max_history_length:
            # Keep only the most recent messages
            excess = len(session.messages) - self.max_history_length
            session.messages = session.messages[excess:]
            
            logger.debug(
                f"Applied sliding window to session {session.session_id}",
                removed_count=excess,
                remaining_count=len(session.messages)
            )
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session.
        
        Args:
            session_id: The session to clear
        
        Returns:
            True if session was cleared, False if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if not session:
                session = self._load_session(session_id)
                if not session:
                    return False
                self._sessions[session_id] = session
            
            session.messages = []
            session.updated_at = time.time()
            
            if self.auto_save:
                self._save_session(session)
            
            logger.info(f"Cleared session: {session_id}")
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: The session to delete
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            # Remove from memory
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            # Remove from disk
            file_path = self._get_session_file_path(session_id)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted session: {session_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete session file: {e}")
                    raise StorageError(f"Failed to delete session: {e}")
            
            return False
    
    def list_sessions(self) -> List[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        session_ids = set(self._sessions.keys())
        
        # Also check disk
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json
                    session_ids.add(session_id)
        except Exception as e:
            logger.warning(f"Failed to list sessions from disk: {e}")
        
        return list(session_ids)
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: The session identifier
        
        Returns:
            True if session exists
        """
        if session_id in self._sessions:
            return True
        
        file_path = self._get_session_file_path(session_id)
        return os.path.exists(file_path)

    def _save_session(self, session: ChatSession) -> None:
        """
        Save a session to disk.
        
        Args:
            session: The session to save
        """
        file_path = self._get_session_file_path(session.session_id)
        
        try:
            data = self._session_to_dict(session)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved session to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise StorageError(f"Failed to save session: {e}")
    
    def _load_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Load a session from disk.
        
        Args:
            session_id: The session identifier
        
        Returns:
            ChatSession if found, None otherwise
        """
        file_path = self._get_session_file_path(session_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = self._dict_to_session(data)
            logger.debug(f"Loaded session from {file_path}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def _session_to_dict(self, session: ChatSession) -> Dict[str, Any]:
        """
        Convert a ChatSession to a dictionary for serialization.
        
        Args:
            session: The session to convert
        
        Returns:
            Dictionary representation
        """
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "metadata": session.metadata,
            "messages": [
                {
                    "message_id": msg.message_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ]
        }
    
    def _dict_to_session(self, data: Dict[str, Any]) -> ChatSession:
        """
        Convert a dictionary to a ChatSession.
        
        Args:
            data: Dictionary representation
        
        Returns:
            ChatSession object
        """
        messages = [
            ChatMessage(
                message_id=msg.get("message_id"),
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp", time.time()),
                metadata=msg.get("metadata")
            )
            for msg in data.get("messages", [])
        ]
        
        return ChatSession(
            session_id=data["session_id"],
            messages=messages,
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata")
        )
    
    def save_all(self) -> int:
        """
        Save all sessions to disk.
        
        Returns:
            Number of sessions saved
        """
        count = 0
        with self._lock:
            for session in self._sessions.values():
                try:
                    self._save_session(session)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to save session {session.session_id}: {e}")
        
        logger.info(f"Saved {count} sessions")
        return count
    
    def load_all(self) -> int:
        """
        Load all sessions from disk into memory.
        
        Returns:
            Number of sessions loaded
        """
        count = 0
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]
                    if session_id not in self._sessions:
                        session = self._load_session(session_id)
                        if session:
                            self._sessions[session_id] = session
                            count += 1
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
        
        logger.info(f"Loaded {count} sessions")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        total_messages = sum(
            len(session.messages) 
            for session in self._sessions.values()
        )
        
        return {
            "sessions_in_memory": len(self._sessions),
            "total_messages_in_memory": total_messages,
            "storage_path": self.storage_path,
            "max_history_length": self.max_history_length,
            "window_size": self.window_size
        }
    
    def format_history_for_model(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_system: bool = True
    ) -> str:
        """
        Format chat history as a string for model input.
        
        Args:
            session_id: The session identifier
            limit: Maximum messages to include
            include_system: Whether to include system messages
        
        Returns:
            Formatted history string
        """
        messages = self.get_chat_history(session_id, limit=limit)
        
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        
        formatted_parts = []
        for msg in messages:
            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "system": "System"
            }.get(msg.role, msg.role.capitalize())
            
            formatted_parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(formatted_parts)
