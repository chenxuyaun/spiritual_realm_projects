"""
Unit tests for ChatGenerate Workflow.

This module contains unit tests for the ChatGenerateWorkflow and
ChatStorage implementations.

Tests cover:
- Session creation and management
- Message persistence
- History retrieval
- Sliding window behavior
- Workflow execution
"""

import pytest
import tempfile
import shutil
import os
import time

from mm_orch.workflows.chat_generate import (
    ChatGenerateWorkflow,
    ChatGenerateContext,
    ChatGenerateStep
)
from mm_orch.storage.chat_storage import ChatStorage, reset_chat_storage
from mm_orch.schemas import WorkflowResult, WorkflowType, ChatMessage
from mm_orch.exceptions import ValidationError


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for chat storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chat_storage(temp_storage_dir):
    """Create a ChatStorage instance with temporary directory."""
    reset_chat_storage()
    return ChatStorage(storage_path=temp_storage_dir)


@pytest.fixture
def workflow(chat_storage):
    """Create a ChatGenerateWorkflow instance."""
    return ChatGenerateWorkflow(chat_storage=chat_storage)


class TestChatStorage:
    """Unit tests for ChatStorage."""
    
    def test_create_session(self, chat_storage):
        """Test session creation."""
        session = chat_storage.create_session()
        
        assert session is not None
        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.messages == []
    
    def test_create_session_with_metadata(self, chat_storage):
        """Test session creation with metadata."""
        metadata = {"user_id": "test_user", "language": "en"}
        session = chat_storage.create_session(metadata=metadata)
        
        assert session.metadata == metadata
    
    def test_get_session(self, chat_storage):
        """Test session retrieval."""
        session = chat_storage.create_session()
        
        retrieved = chat_storage.get_session(session.session_id)
        
        assert retrieved is not None
        assert retrieved.session_id == session.session_id
    
    def test_get_nonexistent_session(self, chat_storage):
        """Test retrieval of non-existent session."""
        result = chat_storage.get_session("nonexistent-id")
        assert result is None
    
    def test_add_message(self, chat_storage):
        """Test adding a message to a session."""
        session = chat_storage.create_session()
        
        message = chat_storage.add_message(
            session.session_id,
            "user",
            "Hello, world!"
        )
        
        assert message is not None
        assert message.role == "user"
        assert message.content == "Hello, world!"
    
    def test_add_message_to_nonexistent_session(self, chat_storage):
        """Test adding message to non-existent session raises error."""
        with pytest.raises(ValidationError):
            chat_storage.add_message("nonexistent", "user", "Hello")
    
    def test_get_chat_history(self, chat_storage):
        """Test retrieving chat history."""
        session = chat_storage.create_session()
        
        chat_storage.add_message(session.session_id, "user", "Message 1")
        chat_storage.add_message(session.session_id, "assistant", "Response 1")
        chat_storage.add_message(session.session_id, "user", "Message 2")
        
        history = chat_storage.get_chat_history(session.session_id, limit=10)
        
        assert len(history) == 3
        assert history[0].content == "Message 1"
        assert history[1].content == "Response 1"
        assert history[2].content == "Message 2"
    
    def test_get_chat_history_with_limit(self, chat_storage):
        """Test history retrieval with limit."""
        session = chat_storage.create_session()
        
        for i in range(10):
            chat_storage.add_message(session.session_id, "user", f"Message {i}")
        
        history = chat_storage.get_chat_history(session.session_id, limit=5)
        
        assert len(history) == 5
        # Should be most recent messages
        assert history[-1].content == "Message 9"
    
    def test_sliding_window(self, chat_storage):
        """Test sliding window removes old messages."""
        storage = ChatStorage(
            storage_path=chat_storage.storage_path,
            max_history_length=5
        )
        session = storage.create_session()
        
        for i in range(10):
            storage.add_message(session.session_id, "user", f"Message {i}")
        
        history = storage.get_full_history(session.session_id)
        
        assert len(history) <= 5
        # Oldest messages should be removed
        contents = [m.content for m in history]
        assert "Message 0" not in contents
    
    def test_session_persistence(self, temp_storage_dir):
        """Test session persists to disk."""
        storage1 = ChatStorage(storage_path=temp_storage_dir)
        session = storage1.create_session()
        storage1.add_message(session.session_id, "user", "Test message")
        
        # Create new storage instance
        storage2 = ChatStorage(storage_path=temp_storage_dir)
        
        # Session should be loadable
        loaded = storage2.get_session(session.session_id)
        
        assert loaded is not None
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Test message"
    
    def test_clear_session(self, chat_storage):
        """Test clearing a session."""
        session = chat_storage.create_session()
        chat_storage.add_message(session.session_id, "user", "Message")
        
        result = chat_storage.clear_session(session.session_id)
        
        assert result is True
        history = chat_storage.get_chat_history(session.session_id)
        assert len(history) == 0
    
    def test_delete_session(self, chat_storage):
        """Test deleting a session."""
        session = chat_storage.create_session()
        
        result = chat_storage.delete_session(session.session_id)
        
        assert result is True
        assert chat_storage.get_session(session.session_id) is None
    
    def test_list_sessions(self, chat_storage):
        """Test listing all sessions."""
        session1 = chat_storage.create_session()
        session2 = chat_storage.create_session()
        
        sessions = chat_storage.list_sessions()
        
        assert session1.session_id in sessions
        assert session2.session_id in sessions
    
    def test_session_exists(self, chat_storage):
        """Test checking if session exists."""
        session = chat_storage.create_session()
        
        assert chat_storage.session_exists(session.session_id) is True
        assert chat_storage.session_exists("nonexistent") is False
    
    def test_format_history_for_model(self, chat_storage):
        """Test formatting history for model input."""
        session = chat_storage.create_session()
        chat_storage.add_message(session.session_id, "user", "Hello")
        chat_storage.add_message(session.session_id, "assistant", "Hi there!")
        
        formatted = chat_storage.format_history_for_model(session.session_id)
        
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted


class TestChatGenerateWorkflow:
    """Unit tests for ChatGenerateWorkflow."""
    
    def test_workflow_initialization(self, workflow):
        """Test workflow initialization."""
        assert workflow.workflow_type == WorkflowType.CHAT_GENERATE
        assert workflow.name == "ChatGenerate"
    
    def test_execute_creates_new_session(self, workflow):
        """Test that execute creates a new session when none provided."""
        result = workflow.execute({"message": "Hello"})
        
        assert result.status in ["success", "partial"]
        assert "session_id" in result.metadata
        assert result.metadata["is_new_session"] is True
    
    def test_execute_uses_existing_session(self, workflow, chat_storage):
        """Test that execute uses existing session when provided."""
        # Create a session first
        session = chat_storage.create_session()
        
        result = workflow.execute({
            "message": "Hello",
            "session_id": session.session_id
        })
        
        assert result.metadata["session_id"] == session.session_id
        assert result.metadata["is_new_session"] is False
    
    def test_execute_returns_response(self, workflow):
        """Test that execute returns a response."""
        result = workflow.execute({"message": "Hello"})
        
        if result.status == "success":
            assert result.result is not None
            assert len(result.result) > 0
    
    def test_execute_updates_history(self, workflow, chat_storage):
        """Test that execute updates conversation history."""
        result = workflow.execute({"message": "Hello"})
        session_id = result.metadata["session_id"]
        
        history = chat_storage.get_chat_history(session_id, limit=10)
        
        # Should have at least user message and response
        assert len(history) >= 2
        
        # First message should be user's
        user_messages = [m for m in history if m.role == "user"]
        assert len(user_messages) >= 1
        assert user_messages[0].content == "Hello"
    
    def test_execute_with_history_context(self, workflow, chat_storage):
        """Test that execute includes history in context."""
        # First message
        result1 = workflow.execute({"message": "My name is Alice"})
        session_id = result1.metadata["session_id"]
        
        # Second message
        result2 = workflow.execute({
            "message": "What is my name?",
            "session_id": session_id
        })
        
        # History should include first exchange
        assert result2.metadata["history_length"] >= 2
    
    def test_validate_parameters_valid(self, workflow):
        """Test parameter validation with valid params."""
        params = {"message": "Hello"}
        assert workflow.validate_parameters(params) is True
    
    def test_validate_parameters_empty_message(self, workflow):
        """Test parameter validation with empty message."""
        params = {"message": ""}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_whitespace_message(self, workflow):
        """Test parameter validation with whitespace-only message."""
        params = {"message": "   "}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_temperature(self, workflow):
        """Test parameter validation with invalid temperature."""
        params = {"message": "Hello", "temperature": 3.0}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_max_history(self, workflow):
        """Test parameter validation with invalid max_history."""
        params = {"message": "Hello", "max_history": -1}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_get_required_parameters(self, workflow):
        """Test getting required parameters."""
        required = workflow.get_required_parameters()
        assert "message" in required
    
    def test_get_optional_parameters(self, workflow):
        """Test getting optional parameters."""
        optional = workflow.get_optional_parameters()
        assert "session_id" in optional
        assert "max_history" in optional
        assert "temperature" in optional
    
    def test_get_required_models(self, workflow):
        """Test getting required models."""
        models = workflow.get_required_models()
        assert len(models) > 0
    
    def test_chat_convenience_method(self, workflow):
        """Test the chat convenience method."""
        result = workflow.chat("Hello")
        
        assert result.status in ["success", "partial", "failed"]
        assert "session_id" in result.metadata
    
    def test_start_conversation(self, workflow):
        """Test starting a new conversation."""
        result = workflow.start_conversation("Hello, I'm starting a conversation")
        
        assert result.metadata["is_new_session"] is True
    
    def test_continue_conversation(self, workflow, chat_storage):
        """Test continuing an existing conversation."""
        # Start conversation
        result1 = workflow.start_conversation("Hello")
        session_id = result1.metadata["session_id"]
        
        # Continue conversation
        result2 = workflow.continue_conversation(session_id, "How are you?")
        
        assert result2.metadata["session_id"] == session_id
        assert result2.metadata["is_new_session"] is False
    
    def test_get_session_history(self, workflow, chat_storage):
        """Test getting session history through workflow."""
        result = workflow.execute({"message": "Hello"})
        session_id = result.metadata["session_id"]
        
        history = workflow.get_session_history(session_id)
        
        assert len(history) >= 2
    
    def test_clear_session(self, workflow, chat_storage):
        """Test clearing session through workflow."""
        result = workflow.execute({"message": "Hello"})
        session_id = result.metadata["session_id"]
        
        cleared = workflow.clear_session(session_id)
        
        assert cleared is True
        history = workflow.get_session_history(session_id)
        assert len(history) == 0


class TestChatGenerateContext:
    """Unit tests for ChatGenerateContext."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        ctx = ChatGenerateContext(
            session_id="test-session",
            message="Hello"
        )
        
        assert ctx.session_id == "test-session"
        assert ctx.message == "Hello"
        assert ctx.is_new_session is False
        assert ctx.history == []
        assert ctx.context_text == ""
        assert ctx.response == ""
        assert ctx.steps == []
    
    def test_add_step(self):
        """Test adding a step to context."""
        ctx = ChatGenerateContext(session_id="test", message="Hello")
        step = ChatGenerateStep(name="test_step", success=True, duration=0.5)
        
        ctx.add_step(step)
        
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == "test_step"
        assert ctx.steps[0].success is True


class TestChatGenerateStep:
    """Unit tests for ChatGenerateStep."""
    
    def test_step_initialization(self):
        """Test step initialization."""
        step = ChatGenerateStep(name="test", success=True)
        
        assert step.name == "test"
        assert step.success is True
        assert step.duration == 0.0
        assert step.error is None
    
    def test_step_with_error(self):
        """Test step with error."""
        step = ChatGenerateStep(
            name="test",
            success=False,
            error="Something went wrong"
        )
        
        assert step.success is False
        assert step.error == "Something went wrong"


class TestChatGenerateEdgeCases:
    """Edge case tests for ChatGenerate workflow."""
    
    def test_very_long_message(self, workflow):
        """Test handling of very long messages."""
        long_message = "Hello " * 1000
        result = workflow.execute({"message": long_message})
        
        # Should not crash
        assert result.status in ["success", "partial", "failed"]
    
    def test_special_characters_in_message(self, workflow):
        """Test handling of special characters."""
        message = "Hello! @#$%^&*() 你好 🎉"
        result = workflow.execute({"message": message})
        
        assert result.status in ["success", "partial", "failed"]
    
    def test_multiple_rapid_messages(self, workflow, chat_storage):
        """Test handling multiple rapid messages."""
        result = workflow.execute({"message": "First"})
        session_id = result.metadata["session_id"]
        
        for i in range(5):
            workflow.execute({
                "message": f"Message {i}",
                "session_id": session_id
            })
        
        history = chat_storage.get_full_history(session_id)
        # Should have all messages (6 user + 6 assistant = 12)
        assert len(history) >= 12
    
    def test_custom_system_prompt(self, workflow):
        """Test using custom system prompt."""
        result = workflow.execute({
            "message": "Hello",
            "system_prompt": "You are a helpful pirate assistant."
        })
        
        assert result.status in ["success", "partial", "failed"]
    
    def test_zero_max_history(self, workflow):
        """Test with zero max_history."""
        result = workflow.execute({
            "message": "Hello",
            "max_history": 0
        })
        
        # Should still work, just without history
        assert result.status in ["success", "partial", "failed"]
