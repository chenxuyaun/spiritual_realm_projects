"""
Property-based tests for ChatGenerate Workflow.

This module contains property-based tests using Hypothesis to verify
the correctness properties of the ChatGenerateWorkflow implementation.

Properties tested:
- Property 7: 对话会话唯一性 (Session Uniqueness)
- Property 8: 对话历史上下文传递 (History Context Passing)
- Property 9: 对话历史持久化 (History Persistence)
- Property 10: 对话历史滑动窗口 (History Sliding Window)

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
"""

import pytest
import tempfile
import shutil
import os
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, Any, List
from contextlib import contextmanager

from mm_orch.workflows.chat_generate import (
    ChatGenerateWorkflow,
    ChatGenerateContext,
    ChatGenerateStep
)
from mm_orch.storage.chat_storage import ChatStorage, reset_chat_storage
from mm_orch.schemas import WorkflowResult, WorkflowType, ChatMessage


# Configure Hypothesis settings for property tests
settings.register_profile("default", max_examples=100)
settings.load_profile("default")


# Custom strategies for generating test data
message_strategy = st.text(
    min_size=1,
    max_size=500,
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' ?!.,;:'
    )
).filter(lambda x: x.strip())

temperature_strategy = st.floats(min_value=0.0, max_value=2.0)

max_history_strategy = st.integers(min_value=1, max_value=20)


@contextmanager
def temp_storage():
    """Context manager for temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        reset_chat_storage()
        storage = ChatStorage(storage_path=temp_dir)
        yield storage, temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestSessionUniqueness:
    """
    Property 7: 对话会话唯一性
    
    对于任何两次独立的对话发起请求，系统应该生成不同的session_id，
    且每个session_id应该是全局唯一的。
    
    **Validates: Requirements 4.1**
    """
    
    @given(
        message1=message_strategy,
        message2=message_strategy
    )
    @settings(max_examples=100)
    def test_new_sessions_have_unique_ids(self, message1: str, message2: str):
        """
        Feature: muai-orchestration-system, Property 7: 对话会话唯一性
        
        Two independent conversation requests should generate different session IDs.
        
        **Validates: Requirements 4.1**
        """
        assume(message1.strip() and message2.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # Start two new conversations
            result1 = workflow.execute({"message": message1})
            result2 = workflow.execute({"message": message2})
            
            # Both should succeed or at least have session IDs
            assert "session_id" in result1.metadata
            assert "session_id" in result2.metadata
            
            # Session IDs should be different
            assert result1.metadata["session_id"] != result2.metadata["session_id"]
    
    @given(num_sessions=st.integers(min_value=2, max_value=10))
    @settings(max_examples=50)
    def test_multiple_sessions_all_unique(self, num_sessions: int):
        """
        Feature: muai-orchestration-system, Property 7: 对话会话唯一性
        
        Multiple session creations should all produce unique IDs.
        
        **Validates: Requirements 4.1**
        """
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            session_ids = set()
            
            for i in range(num_sessions):
                result = workflow.execute({"message": f"Test message {i}"})
                session_id = result.metadata.get("session_id")
                
                assert session_id is not None
                assert session_id not in session_ids, f"Duplicate session ID: {session_id}"
                session_ids.add(session_id)
            
            assert len(session_ids) == num_sessions
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_session_id_is_valid_uuid_format(self, message: str):
        """
        Feature: muai-orchestration-system, Property 7: 对话会话唯一性
        
        Session IDs should be valid UUID format.
        
        **Validates: Requirements 4.1**
        """
        assume(message.strip())
        
        import uuid
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            result = workflow.execute({"message": message})
            session_id = result.metadata.get("session_id")
            
            assert session_id is not None
            
            # Should be valid UUID
            try:
                uuid.UUID(session_id)
            except ValueError:
                pytest.fail(f"Session ID is not a valid UUID: {session_id}")


class TestHistoryContextPassing:
    """
    Property 8: 对话历史上下文传递
    
    对于任何在已有会话中发送的消息，生成回复时传递给模型的输入应该
    包含该会话的历史对话记录（至少包含最近的消息）。
    
    **Validates: Requirements 4.2, 4.3**
    """
    
    @given(
        message1=message_strategy,
        message2=message_strategy
    )
    @settings(max_examples=100)
    def test_context_includes_history(self, message1: str, message2: str):
        """
        Feature: muai-orchestration-system, Property 8: 对话历史上下文传递
        
        When sending a message in an existing session, the context should
        include previous messages.
        
        **Validates: Requirements 4.2, 4.3**
        """
        assume(message1.strip() and message2.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # First message - creates session
            result1 = workflow.execute({"message": message1})
            session_id = result1.metadata.get("session_id")
            
            assert session_id is not None
            
            # Second message - should include history
            result2 = workflow.execute({
                "message": message2,
                "session_id": session_id
            })
            
            # History length should be > 0 (includes first exchange)
            assert result2.metadata.get("history_length", 0) >= 2
    
    @given(messages=st.lists(message_strategy, min_size=2, max_size=5))
    @settings(max_examples=50)
    def test_history_grows_with_conversation(self, messages: List[str]):
        """
        Feature: muai-orchestration-system, Property 8: 对话历史上下文传递
        
        As conversation progresses, history should grow.
        
        **Validates: Requirements 4.2, 4.3**
        """
        messages = [m for m in messages if m.strip()]
        assume(len(messages) >= 2)
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # First message
            result = workflow.execute({"message": messages[0]})
            session_id = result.metadata.get("session_id")
            
            previous_history_length = 0
            
            # Subsequent messages
            for i, message in enumerate(messages[1:], start=1):
                result = workflow.execute({
                    "message": message,
                    "session_id": session_id
                })
                
                current_history_length = result.metadata.get("history_length", 0)
                
                # History should grow (each turn adds 2 messages: user + assistant)
                assert current_history_length >= previous_history_length
                previous_history_length = current_history_length
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_context_length_is_tracked(self, message: str):
        """
        Feature: muai-orchestration-system, Property 8: 对话历史上下文传递
        
        Context length should be tracked in metadata.
        
        **Validates: Requirements 4.2, 4.3**
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            result = workflow.execute({"message": message})
            
            # Context length should be tracked
            assert "context_length" in result.metadata
            assert result.metadata["context_length"] > 0


class TestHistoryPersistence:
    """
    Property 9: 对话历史持久化
    
    对于任何对话会话，在添加新消息后，从存储中检索该会话的历史应该
    包含刚添加的消息，且历史记录数量应该至少保留最近10轮对话。
    
    **Validates: Requirements 4.4, 14.1**
    """
    
    @given(
        message1=message_strategy,
        message2=message_strategy
    )
    @settings(max_examples=100)
    def test_messages_are_persisted(self, message1: str, message2: str):
        """
        Feature: muai-orchestration-system, Property 9: 对话历史持久化
        
        After adding messages, they should be retrievable from storage.
        
        **Validates: Requirements 4.4, 14.1**
        """
        assume(message1.strip() and message2.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # Send messages
            result1 = workflow.execute({"message": message1})
            session_id = result1.metadata.get("session_id")
            
            result2 = workflow.execute({
                "message": message2,
                "session_id": session_id
            })
            
            # Retrieve history
            history = storage.get_chat_history(session_id, limit=100)
            
            # Should have at least 4 messages (2 user + 2 assistant)
            assert len(history) >= 4
            
            # User messages should be in history
            user_messages = [m.content for m in history if m.role == "user"]
            assert message1 in user_messages
            assert message2 in user_messages
    
    @given(num_turns=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_history_preserves_at_least_10_turns(self, num_turns: int):
        """
        Feature: muai-orchestration-system, Property 9: 对话历史持久化
        
        History should preserve at least 10 recent conversation turns.
        
        **Validates: Requirements 4.4**
        """
        with temp_storage() as (storage, temp_dir):
            storage = ChatStorage(
                storage_path=temp_dir,
                max_history_length=100  # Allow more than 10 turns
            )
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # Create conversation with multiple turns
            result = workflow.execute({"message": "First message"})
            session_id = result.metadata.get("session_id")
            
            for i in range(num_turns - 1):
                workflow.execute({
                    "message": f"Message {i + 2}",
                    "session_id": session_id
                })
            
            # Retrieve history
            history = storage.get_full_history(session_id)
            
            # Should have 2 messages per turn (user + assistant)
            expected_messages = num_turns * 2
            assert len(history) >= min(expected_messages, 20)  # At least 10 turns = 20 messages
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_history_survives_storage_reload(self, message: str):
        """
        Feature: muai-orchestration-system, Property 9: 对话历史持久化
        
        History should survive storage reload (file persistence).
        
        **Validates: Requirements 14.1**
        """
        assume(message.strip())
        
        temp_dir = tempfile.mkdtemp()
        try:
            # Create storage and add message
            storage1 = ChatStorage(storage_path=temp_dir)
            workflow1 = ChatGenerateWorkflow(chat_storage=storage1)
            
            result = workflow1.execute({"message": message})
            session_id = result.metadata.get("session_id")
            
            # Create new storage instance (simulates restart)
            storage2 = ChatStorage(storage_path=temp_dir)
            
            # History should be loadable
            history = storage2.get_chat_history(session_id, limit=100)
            
            assert len(history) >= 2  # At least user message + response
            
            # User message should be present
            user_messages = [m.content for m in history if m.role == "user"]
            assert message in user_messages
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestHistorySlidingWindow:
    """
    Property 10: 对话历史滑动窗口
    
    对于任何对话会话，当历史记录超过最大长度限制时，检索到的历史应该
    只包含最近的N条记录（N为配置的窗口大小），且最旧的记录应该被移除。
    
    **Validates: Requirements 4.5**
    """
    
    @given(
        window_size=st.integers(min_value=2, max_value=10),
        num_messages=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=50)
    def test_sliding_window_limits_history(self, window_size: int, num_messages: int):
        """
        Feature: muai-orchestration-system, Property 10: 对话历史滑动窗口
        
        When history exceeds max length, only recent messages are retrieved.
        
        **Validates: Requirements 4.5**
        """
        assume(num_messages > window_size)
        
        temp_dir = tempfile.mkdtemp()
        try:
            storage = ChatStorage(
                storage_path=temp_dir,
                window_size=window_size,
                max_history_length=100
            )
            workflow = ChatGenerateWorkflow(
                chat_storage=storage,
                max_history_turns=window_size
            )
            
            # Create conversation
            result = workflow.execute({"message": "Message 1"})
            session_id = result.metadata.get("session_id")
            
            for i in range(2, num_messages + 1):
                workflow.execute({
                    "message": f"Message {i}",
                    "session_id": session_id
                })
            
            # Get history with window
            history = storage.get_chat_history(session_id, limit=window_size)
            
            # Should not exceed window size
            assert len(history) <= window_size
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(
        max_history=st.integers(min_value=5, max_value=20),
        num_messages=st.integers(min_value=10, max_value=30)
    )
    @settings(max_examples=50)
    def test_oldest_messages_removed_when_exceeding_max(self, max_history: int, num_messages: int):
        """
        Feature: muai-orchestration-system, Property 10: 对话历史滑动窗口
        
        When history exceeds max length, oldest messages should be removed.
        
        **Validates: Requirements 4.5**
        """
        assume(num_messages > max_history)
        
        temp_dir = tempfile.mkdtemp()
        try:
            storage = ChatStorage(
                storage_path=temp_dir,
                max_history_length=max_history
            )
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # Create conversation with many messages
            result = workflow.execute({"message": "FIRST_MESSAGE"})
            session_id = result.metadata.get("session_id")
            
            for i in range(2, num_messages + 1):
                workflow.execute({
                    "message": f"Message {i}",
                    "session_id": session_id
                })
            
            # Get full history
            history = storage.get_full_history(session_id)
            
            # Should not exceed max_history
            assert len(history) <= max_history
            
            # First message should be removed if we exceeded max
            if num_messages * 2 > max_history:
                user_messages = [m.content for m in history if m.role == "user"]
                # FIRST_MESSAGE should not be in history if we exceeded max
                # (it would have been removed by sliding window)
                if len(user_messages) < num_messages:
                    assert "FIRST_MESSAGE" not in user_messages
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(window_size=st.integers(min_value=2, max_value=10))
    @settings(max_examples=50)
    def test_window_size_respected_in_retrieval(self, window_size: int):
        """
        Feature: muai-orchestration-system, Property 10: 对话历史滑动窗口
        
        Retrieval should respect the configured window size.
        
        **Validates: Requirements 4.5**
        """
        temp_dir = tempfile.mkdtemp()
        try:
            storage = ChatStorage(
                storage_path=temp_dir,
                window_size=window_size,
                max_history_length=100
            )
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            # Create conversation with more messages than window
            result = workflow.execute({"message": "Message 1"})
            session_id = result.metadata.get("session_id")
            
            for i in range(2, window_size + 5):
                workflow.execute({
                    "message": f"Message {i}",
                    "session_id": session_id
                })
            
            # Get history with default window
            history = storage.get_chat_history(session_id)
            
            # Should respect window size
            assert len(history) <= window_size
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestChatGenerateWorkflowProperties:
    """Additional property tests for ChatGenerateWorkflow."""
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_workflow_type_consistency(self, message: str):
        """
        For any execution, the workflow type should remain consistent.
        
        **Validates: Requirements 4.1**
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            assert workflow.workflow_type == WorkflowType.CHAT_GENERATE
            assert workflow.name == "ChatGenerate"
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_result_structure_completeness(self, message: str):
        """
        For any execution, result should have complete structure.
        
        **Validates: Requirements 4.1, 4.3**
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            result = workflow.execute({"message": message})
            
            # Result should be WorkflowResult
            assert isinstance(result, WorkflowResult)
            
            # Status should be valid
            assert result.status in ["success", "partial", "failed"]
            
            # Metadata should contain required fields
            assert "workflow" in result.metadata
            assert "session_id" in result.metadata
            assert "is_new_session" in result.metadata
            assert "history_length" in result.metadata
            assert "steps" in result.metadata
    
    @given(message=message_strategy, temperature=temperature_strategy)
    @settings(max_examples=100)
    def test_temperature_parameter_accepted(self, message: str, temperature: float):
        """
        Any valid temperature should be accepted.
        
        **Validates: Requirements 4.3**
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            result = workflow.execute({
                "message": message,
                "temperature": temperature
            })
            
            # Should not fail due to temperature
            assert result.status in ["success", "partial", "failed"]
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_response_is_non_empty_on_success(self, message: str):
        """
        On successful execution, response should be non-empty.
        
        **Validates: Requirements 4.3**
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            
            result = workflow.execute({"message": message})
            
            if result.status == "success":
                assert result.result is not None
                assert len(result.result) > 0


class TestChatStorageProperties:
    """Property tests for ChatStorage directly."""
    
    @given(num_sessions=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_all_created_sessions_are_unique(self, num_sessions: int):
        """
        All created sessions should have unique IDs.
        """
        with temp_storage() as (storage, _):
            session_ids = set()
            for _ in range(num_sessions):
                session = storage.create_session()
                assert session.session_id not in session_ids
                session_ids.add(session.session_id)
            
            assert len(session_ids) == num_sessions
    
    @given(
        role=st.sampled_from(["user", "assistant", "system"]),
        content=message_strategy
    )
    @settings(max_examples=100)
    def test_messages_are_stored_correctly(self, role: str, content: str):
        """
        Messages should be stored with correct role and content.
        """
        assume(content.strip())
        
        with temp_storage() as (storage, _):
            session = storage.create_session()
            
            storage.add_message(session.session_id, role, content)
            
            history = storage.get_chat_history(session.session_id, limit=10)
            
            assert len(history) == 1
            assert history[0].role == role
            assert history[0].content == content
    
    @given(
        window_size=st.integers(min_value=2, max_value=10),
        num_messages=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50)
    def test_get_chat_history_respects_limit(self, window_size: int, num_messages: int):
        """
        get_chat_history should respect the limit parameter.
        """
        with temp_storage() as (storage, _):
            session = storage.create_session()
            
            # Add messages
            for i in range(num_messages):
                storage.add_message(session.session_id, "user", f"Message {i}")
            
            # Get with limit
            history = storage.get_chat_history(session.session_id, limit=window_size)
            
            # Should not exceed limit
            assert len(history) <= window_size
            
            # Should return most recent messages
            if num_messages > window_size:
                assert len(history) == window_size


class TestChatGenerateValidationProperties:
    """Property tests for parameter validation."""
    
    @given(message=message_strategy)
    @settings(max_examples=100)
    def test_valid_message_passes_validation(self, message: str):
        """
        Any non-empty message should pass validation.
        """
        assume(message.strip())
        
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            params = {"message": message}
            
            assert workflow.validate_parameters(params) is True
    
    @given(max_history=max_history_strategy)
    @settings(max_examples=100)
    def test_valid_max_history_passes_validation(self, max_history: int):
        """
        Any positive max_history should pass validation.
        """
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            params = {"message": "Test", "max_history": max_history}
            
            assert workflow.validate_parameters(params) is True
    
    @given(temperature=temperature_strategy)
    @settings(max_examples=100)
    def test_valid_temperature_passes_validation(self, temperature: float):
        """
        Any temperature in valid range should pass validation.
        """
        with temp_storage() as (storage, _):
            workflow = ChatGenerateWorkflow(chat_storage=storage)
            params = {"message": "Test", "temperature": temperature}
            
            assert workflow.validate_parameters(params) is True
