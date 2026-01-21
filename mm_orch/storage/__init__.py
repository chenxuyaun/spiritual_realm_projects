"""
Storage module for MuAI Multi-Model Orchestration System.

This module provides storage implementations for various data types
including chat history, consciousness state, and vector databases.
"""

from mm_orch.storage.chat_storage import ChatStorage, get_chat_storage

__all__ = [
    "ChatStorage",
    "get_chat_storage"
]
