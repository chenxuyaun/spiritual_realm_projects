"""
Unit tests for MuAI data models (schemas).

Tests cover:
- Data class instantiation and field validation
- Serialization and deserialization
- Edge cases and error conditions
"""

import pytest
import time
import json
import numpy as np
from dataclasses import asdict

from mm_orch.schemas import (
    WorkflowType,
    IntentType,
    WorkflowStatus,
    MessageRole,
    DevelopmentStage,
    UserRequest,
    WorkflowSelection,
    WorkflowResult,
    Document,
    ChatMessage,
    ChatSession,
    ConsciousnessState,
    LessonPack,
    StrategySuggestion,
    SystemEvent,
    Task,
    Evaluation,
    ModelConfig,
    SystemConfig,
    ErrorResponse,
)


class TestEnums:
    """Tests for enum types."""
    
    def test_workflow_type_values(self):
        """Test WorkflowType enum has all expected values."""
        assert WorkflowType.SEARCH_QA.value == "search_qa"
        assert WorkflowType.LESSON_PACK.value == "lesson_pack"
        assert WorkflowType.CHAT_GENERATE.value == "chat_generate"
        assert WorkflowType.RAG_QA.value == "rag_qa"
        assert WorkflowType.SELF_ASK_SEARCH_QA.value == "self_ask_search_qa"
    
    def test_intent_type_values(self):
        """Test IntentType enum has all expected values."""
        assert IntentType.QUESTION_ANSWERING.value == "qa"
        assert IntentType.TEACHING.value == "teaching"
        assert IntentType.CONVERSATION.value == "conversation"
    
    def test_development_stage_values(self):
        """Test DevelopmentStage enum has all expected values."""
        assert DevelopmentStage.INFANT.value == "infant"
        assert DevelopmentStage.CHILD.value == "child"
        assert DevelopmentStage.ADOLESCENT.value == "adolescent"
        assert DevelopmentStage.ADULT.value == "adult"


class TestUserRequest:
    """Tests for UserRequest data class."""
    
    def test_basic_creation(self):
        """Test basic UserRequest creation."""
        request = UserRequest(query="What is Python?")
        assert request.query == "What is Python?"
        assert request.context is None
        assert request.session_id is None
        assert request.preferences is None
    
    def test_full_creation(self):
        """Test UserRequest with all fields."""
        request = UserRequest(
            query="Explain machine learning",
            context={"topic": "AI"},
            session_id="session-123",
            preferences={"language": "zh"}
        )
        assert request.query == "Explain machine learning"
        assert request.context == {"topic": "AI"}
        assert request.session_id == "session-123"
        assert request.preferences == {"language": "zh"}
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            UserRequest(query="")
    
    def test_whitespace_query_raises_error(self):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            UserRequest(query="   ")



class TestWorkflowSelection:
    """Tests for WorkflowSelection data class."""
    
    def test_basic_creation(self):
        """Test basic WorkflowSelection creation."""
        selection = WorkflowSelection(
            workflow_type=WorkflowType.SEARCH_QA,
            confidence=0.85,
            parameters={"query": "test"}
        )
        assert selection.workflow_type == WorkflowType.SEARCH_QA
        assert selection.confidence == 0.85
        assert selection.parameters == {"query": "test"}
        assert selection.alternatives is None
    
    def test_with_alternatives(self):
        """Test WorkflowSelection with alternatives."""
        alt = WorkflowSelection(
            workflow_type=WorkflowType.RAG_QA,
            confidence=0.6,
            parameters={}
        )
        selection = WorkflowSelection(
            workflow_type=WorkflowType.SEARCH_QA,
            confidence=0.55,
            parameters={},
            alternatives=[alt]
        )
        assert len(selection.alternatives) == 1
        assert selection.alternatives[0].workflow_type == WorkflowType.RAG_QA
    
    def test_confidence_boundary_values(self):
        """Test confidence at boundary values."""
        # Valid boundaries
        WorkflowSelection(workflow_type=WorkflowType.SEARCH_QA, confidence=0.0, parameters={})
        WorkflowSelection(workflow_type=WorkflowType.SEARCH_QA, confidence=1.0, parameters={})
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            WorkflowSelection(
                workflow_type=WorkflowType.SEARCH_QA,
                confidence=1.5,
                parameters={}
            )
        with pytest.raises(ValueError, match="Confidence must be between"):
            WorkflowSelection(
                workflow_type=WorkflowType.SEARCH_QA,
                confidence=-0.1,
                parameters={}
            )


class TestWorkflowResult:
    """Tests for WorkflowResult data class."""
    
    def test_success_result(self):
        """Test successful WorkflowResult."""
        result = WorkflowResult(
            result="Answer to the question",
            metadata={"source": "web"},
            status="success"
        )
        assert result.result == "Answer to the question"
        assert result.status == "success"
        assert result.error is None
    
    def test_partial_result(self):
        """Test partial WorkflowResult."""
        result = WorkflowResult(
            result="Partial answer",
            metadata={},
            status="partial",
            error="Some sources unavailable"
        )
        assert result.status == "partial"
        assert result.error == "Some sources unavailable"
    
    def test_failed_result(self):
        """Test failed WorkflowResult."""
        result = WorkflowResult(
            result=None,
            metadata={},
            status="failed",
            error="Network error"
        )
        assert result.status == "failed"
        assert result.result is None
    
    def test_invalid_status_raises_error(self):
        """Test that invalid status raises ValueError."""
        with pytest.raises(ValueError, match="Status must be one of"):
            WorkflowResult(result="test", metadata={}, status="unknown")
    
    def test_success_with_none_result_raises_error(self):
        """Test that success status with None result raises ValueError."""
        with pytest.raises(ValueError, match="Result cannot be None"):
            WorkflowResult(result=None, metadata={}, status="success")


class TestDocument:
    """Tests for Document data class."""
    
    def test_basic_creation(self):
        """Test basic Document creation."""
        doc = Document(
            content="This is a test document.",
            metadata={"source": "test.txt"}
        )
        assert doc.content == "This is a test document."
        assert doc.metadata == {"source": "test.txt"}
        assert doc.doc_id is not None  # Auto-generated
        assert doc.embedding is None
    
    def test_with_embedding(self):
        """Test Document with embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        doc = Document(
            content="Test content",
            metadata={},
            embedding=embedding
        )
        assert np.array_equal(doc.embedding, embedding)
    
    def test_custom_doc_id(self):
        """Test Document with custom doc_id."""
        doc = Document(
            content="Test",
            metadata={},
            doc_id="custom-id-123"
        )
        assert doc.doc_id == "custom-id-123"
    
    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Document content cannot be empty"):
            Document(content="", metadata={})



class TestChatMessage:
    """Tests for ChatMessage data class."""
    
    def test_basic_creation(self):
        """Test basic ChatMessage creation."""
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.timestamp > 0
        assert msg.message_id is not None
    
    def test_all_valid_roles(self):
        """Test all valid message roles."""
        for role in ["user", "assistant", "system"]:
            msg = ChatMessage(role=role, content="Test")
            assert msg.role == role
    
    def test_invalid_role_raises_error(self):
        """Test that invalid role raises ValueError."""
        with pytest.raises(ValueError, match="Role must be one of"):
            ChatMessage(role="admin", content="Test")
    
    def test_custom_timestamp(self):
        """Test ChatMessage with custom timestamp."""
        ts = 1700000000.0
        msg = ChatMessage(role="user", content="Test", timestamp=ts)
        assert msg.timestamp == ts


class TestChatSession:
    """Tests for ChatSession data class."""
    
    def test_create_new_session(self):
        """Test creating a new chat session."""
        session = ChatSession.create_new()
        assert session.session_id is not None
        assert len(session.messages) == 0
        assert session.created_at > 0
    
    def test_add_message(self):
        """Test adding messages to session."""
        session = ChatSession.create_new()
        msg = session.add_message("user", "Hello!")
        
        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello!"
        assert msg.role == "user"
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        session = ChatSession.create_new()
        for i in range(15):
            session.add_message("user", f"Message {i}")
        
        recent = session.get_recent_messages(limit=10)
        assert len(recent) == 10
        assert recent[0].content == "Message 5"
        assert recent[-1].content == "Message 14"
    
    def test_get_recent_messages_empty_session(self):
        """Test getting recent messages from empty session."""
        session = ChatSession.create_new()
        recent = session.get_recent_messages()
        assert recent == []
    
    def test_updated_at_changes_on_add(self):
        """Test that updated_at changes when adding messages."""
        session = ChatSession.create_new()
        original_updated = session.updated_at
        time.sleep(0.01)  # Small delay
        session.add_message("user", "Test")
        assert session.updated_at > original_updated


class TestConsciousnessState:
    """Tests for ConsciousnessState data class."""
    
    def test_default_creation(self):
        """Test ConsciousnessState with defaults."""
        state = ConsciousnessState()
        assert state.self_state == {}
        assert state.world_state == {}
        assert state.emotion_state == {"valence": 0.0, "arousal": 0.5}
        assert state.development_stage == "adult"
    
    def test_custom_emotion_state(self):
        """Test ConsciousnessState with custom emotion."""
        state = ConsciousnessState(
            emotion_state={"valence": 0.5, "arousal": 0.8}
        )
        assert state.emotion_state["valence"] == 0.5
        assert state.emotion_state["arousal"] == 0.8
    
    def test_invalid_development_stage_raises_error(self):
        """Test that invalid development stage raises ValueError."""
        with pytest.raises(ValueError, match="Development stage must be one of"):
            ConsciousnessState(development_stage="teenager")
    
    def test_invalid_emotion_value_raises_error(self):
        """Test that invalid emotion values raise ValueError."""
        with pytest.raises(ValueError, match="valence must be between"):
            ConsciousnessState(emotion_state={"valence": 2.0, "arousal": 0.5})



class TestLessonPack:
    """Tests for LessonPack data class."""
    
    def test_basic_creation(self):
        """Test basic LessonPack creation."""
        pack = LessonPack(
            topic="Python Basics",
            plan="1. Introduction\n2. Variables\n3. Functions",
            explanation="Python is a programming language...",
            exercises=[
                {"question": "What is Python?", "answer": "A programming language"}
            ]
        )
        assert pack.topic == "Python Basics"
        assert len(pack.exercises) == 1
    
    def test_empty_topic_raises_error(self):
        """Test that empty topic raises ValueError."""
        with pytest.raises(ValueError, match="Topic cannot be empty"):
            LessonPack(
                topic="",
                plan="Plan",
                explanation="Explanation",
                exercises=[]
            )
    
    def test_invalid_exercise_structure_raises_error(self):
        """Test that invalid exercise structure raises ValueError."""
        with pytest.raises(ValueError, match="must have 'question' and 'answer'"):
            LessonPack(
                topic="Test",
                plan="Plan",
                explanation="Explanation",
                exercises=[{"question": "Q1"}]  # Missing answer
            )


class TestStrategySuggestion:
    """Tests for StrategySuggestion data class."""
    
    def test_basic_creation(self):
        """Test basic StrategySuggestion creation."""
        suggestion = StrategySuggestion(
            strategy="use_cache",
            confidence=0.9,
            reasoning="Previous similar query found in cache"
        )
        assert suggestion.strategy == "use_cache"
        assert suggestion.confidence == 0.9
    
    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            StrategySuggestion(
                strategy="test",
                confidence=1.5,
                reasoning="test"
            )


class TestTask:
    """Tests for Task data class."""
    
    def test_create_task(self):
        """Test creating a task using factory method."""
        task = Task.create(
            task_type="search",
            parameters={"query": "test"},
            priority=5
        )
        assert task.task_id is not None
        assert task.task_type == "search"
        assert task.priority == 5
    
    def test_task_has_timestamp(self):
        """Test that task has creation timestamp."""
        task = Task.create(task_type="test", parameters={})
        assert task.created_at > 0


class TestEvaluation:
    """Tests for Evaluation data class."""
    
    def test_basic_creation(self):
        """Test basic Evaluation creation."""
        evaluation = Evaluation(
            success=True,
            score=0.85,
            feedback="Good result"
        )
        assert evaluation.success is True
        assert evaluation.score == 0.85
    
    def test_invalid_score_raises_error(self):
        """Test that invalid score raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between"):
            Evaluation(success=True, score=1.5, feedback="test")


class TestModelConfig:
    """Tests for ModelConfig data class."""
    
    def test_basic_creation(self):
        """Test basic ModelConfig creation."""
        config = ModelConfig(
            name="qwen-chat",
            model_path="Qwen/Qwen-7B-Chat"
        )
        assert config.name == "qwen-chat"
        assert config.device == "auto"
        assert config.quantization is None
    
    def test_with_quantization(self):
        """Test ModelConfig with quantization."""
        config = ModelConfig(
            name="test",
            model_path="test/model",
            quantization="8bit"
        )
        assert config.quantization == "8bit"
    
    def test_invalid_device_raises_error(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Device must be one of"):
            ModelConfig(name="test", model_path="test", device="tpu")
    
    def test_invalid_quantization_raises_error(self):
        """Test that invalid quantization raises ValueError."""
        with pytest.raises(ValueError, match="Quantization must be one of"):
            ModelConfig(name="test", model_path="test", quantization="16bit")


class TestSystemConfig:
    """Tests for SystemConfig data class."""
    
    def test_basic_creation(self):
        """Test basic SystemConfig creation."""
        model_config = ModelConfig(name="test", model_path="test/model")
        config = SystemConfig(
            models={"test": model_config},
            vector_db_path="./data/vectors",
            storage_path="./data/storage"
        )
        assert config.log_level == "INFO"
        assert config.max_cached_models == 3
        assert config.development_stage == "adult"
    
    def test_invalid_log_level_raises_error(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            SystemConfig(
                models={},
                vector_db_path="./data",
                storage_path="./data",
                log_level="VERBOSE"
            )


class TestErrorResponse:
    """Tests for ErrorResponse data class."""
    
    def test_basic_creation(self):
        """Test basic ErrorResponse creation."""
        error = ErrorResponse(
            error_type="ValidationError",
            message="Invalid input"
        )
        assert error.error_type == "ValidationError"
        assert error.message == "Invalid input"
        assert error.timestamp > 0
    
    def test_with_details(self):
        """Test ErrorResponse with details."""
        error = ErrorResponse(
            error_type="NetworkError",
            message="Connection failed",
            details={"url": "http://example.com", "status": 500}
        )
        assert error.details["status"] == 500


class TestSerialization:
    """Tests for data class serialization."""
    
    def test_user_request_to_dict(self):
        """Test UserRequest serialization to dict."""
        request = UserRequest(query="Test query")
        data = asdict(request)
        assert data["query"] == "Test query"
        assert "context" in data
    
    def test_workflow_result_to_dict(self):
        """Test WorkflowResult serialization to dict."""
        result = WorkflowResult(
            result="Answer",
            metadata={"source": "web"},
            status="success"
        )
        data = asdict(result)
        assert data["result"] == "Answer"
        assert data["status"] == "success"
    
    def test_chat_message_to_dict(self):
        """Test ChatMessage serialization to dict."""
        msg = ChatMessage(role="user", content="Hello")
        data = asdict(msg)
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert "timestamp" in data
        assert "message_id" in data
