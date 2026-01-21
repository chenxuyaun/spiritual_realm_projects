"""
Core data models for MuAI Multi-Model Orchestration System.

This module defines all the core data structures used throughout the system,
including user requests, workflow results, documents, chat messages, and
consciousness state.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time
import uuid
import numpy as np


class WorkflowType(Enum):
    """Supported workflow types in the system."""
    SEARCH_QA = "search_qa"
    LESSON_PACK = "lesson_pack"
    CHAT_GENERATE = "chat_generate"
    RAG_QA = "rag_qa"
    SELF_ASK_SEARCH_QA = "self_ask_search_qa"


class IntentType(Enum):
    """User intent classification types."""
    QUESTION_ANSWERING = "qa"
    TEACHING = "teaching"
    CONVERSATION = "conversation"
    KNOWLEDGE_QUERY = "knowledge"
    COMPLEX_REASONING = "reasoning"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class MessageRole(Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DevelopmentStage(Enum):
    """Development stages for the consciousness system."""
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"



@dataclass
class UserRequest:
    """
    User request data structure.
    
    Attributes:
        query: The user's query text
        context: Optional additional context for the request
        session_id: Optional session identifier for multi-turn conversations
        preferences: Optional user preferences for response generation
    """
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")


@dataclass
class WorkflowSelection:
    """
    Workflow selection result from the Router.
    
    Attributes:
        workflow_type: The selected workflow type
        confidence: Confidence score (0.0 to 1.0)
        parameters: Parameters to pass to the workflow
        alternatives: Alternative workflow selections when confidence is low
    """
    workflow_type: WorkflowType
    confidence: float
    parameters: Dict[str, Any]
    alternatives: Optional[List['WorkflowSelection']] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class WorkflowResult:
    """
    Workflow execution result.
    
    Attributes:
        result: The main result of the workflow execution
        metadata: Additional metadata about the execution
        status: Execution status (success, partial, failed)
        error: Error message if status is failed or partial
        execution_time: Time taken to execute the workflow in seconds
    """
    result: Any
    metadata: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        valid_statuses = {"success", "partial", "failed"}
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        if self.status == "success" and self.result is None:
            raise ValueError("Result cannot be None for successful execution")


@dataclass
class Document:
    """
    Document fragment for RAG system.
    
    Attributes:
        content: The text content of the document
        metadata: Document metadata (source, title, etc.)
        embedding: Optional vector embedding of the content
        doc_id: Unique document identifier
    """
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = str(uuid.uuid4())
        if not self.content:
            raise ValueError("Document content cannot be empty")



@dataclass
class ChatMessage:
    """
    Chat message in a conversation.
    
    Attributes:
        role: The role of the message sender (user, assistant, system)
        content: The message content
        timestamp: Unix timestamp of when the message was created
        metadata: Optional additional metadata
        message_id: Unique message identifier
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None
    message_id: Optional[str] = None
    
    def __post_init__(self):
        valid_roles = {"user", "assistant", "system"}
        if self.role not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}")
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())


@dataclass
class ChatSession:
    """
    Chat session containing conversation history.
    
    Attributes:
        session_id: Unique session identifier
        messages: List of chat messages in the session
        created_at: Unix timestamp of session creation
        updated_at: Unix timestamp of last update
        metadata: Optional session metadata
    """
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Add a new message to the session."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        self.updated_at = time.time()
        return message
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get the most recent messages from the session."""
        return self.messages[-limit:] if self.messages else []
    
    @staticmethod
    def create_new() -> 'ChatSession':
        """Create a new chat session with a unique ID."""
        return ChatSession(session_id=str(uuid.uuid4()))


@dataclass
class ConsciousnessState:
    """
    Consciousness module state.
    
    Attributes:
        self_state: Self model state (capabilities, status, performance)
        world_state: World model state (environment knowledge)
        emotion_state: Emotion state (valence, arousal)
        motivation_state: Motivation state (goals, priorities)
        development_stage: Current development stage
        metacognition_metrics: Metacognition metrics
    """
    self_state: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    emotion_state: Dict[str, float] = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.5})
    motivation_state: Dict[str, Any] = field(default_factory=dict)
    development_stage: str = "adult"
    metacognition_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        valid_stages = {"infant", "child", "adolescent", "adult"}
        if self.development_stage not in valid_stages:
            raise ValueError(f"Development stage must be one of {valid_stages}")
        # Validate emotion state values
        for key in ["valence", "arousal"]:
            if key in self.emotion_state:
                value = self.emotion_state[key]
                if not -1.0 <= value <= 1.0:
                    raise ValueError(f"{key} must be between -1.0 and 1.0")



@dataclass
class LessonPack:
    """
    Teaching content package.
    
    Attributes:
        topic: The lesson topic
        plan: Teaching plan/outline
        explanation: Detailed explanation content
        exercises: List of exercises with questions and answers
        metadata: Additional metadata
    """
    topic: str
    plan: str
    explanation: str
    exercises: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.topic:
            raise ValueError("Topic cannot be empty")
        if not self.plan:
            raise ValueError("Plan cannot be empty")
        if not self.explanation:
            raise ValueError("Explanation cannot be empty")
        # Validate exercises structure
        for i, exercise in enumerate(self.exercises):
            if "question" not in exercise or "answer" not in exercise:
                raise ValueError(f"Exercise {i} must have 'question' and 'answer' fields")


@dataclass
class StrategySuggestion:
    """
    Strategy suggestion from metacognition module.
    
    Attributes:
        strategy: Suggested strategy name
        confidence: Confidence in the suggestion (0.0 to 1.0)
        reasoning: Explanation for the suggestion
        parameters: Suggested parameters for the strategy
    """
    strategy: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class SystemEvent:
    """
    System event for consciousness state updates.
    
    Attributes:
        event_type: Type of the event
        data: Event data
        timestamp: Unix timestamp of the event
        source: Source component of the event
    """
    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None


@dataclass
class Task:
    """
    Task representation for consciousness processing.
    
    Attributes:
        task_id: Unique task identifier
        task_type: Type of the task
        parameters: Task parameters
        priority: Task priority (higher = more important)
        created_at: Unix timestamp of task creation
    """
    task_id: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    
    @staticmethod
    def create(task_type: str, parameters: Dict[str, Any], priority: int = 0) -> 'Task':
        """Create a new task with a unique ID."""
        return Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            parameters=parameters,
            priority=priority
        )



@dataclass
class Evaluation:
    """
    Task result evaluation from consciousness.
    
    Attributes:
        success: Whether the task was successful
        score: Evaluation score (0.0 to 1.0)
        feedback: Evaluation feedback
        emotion_impact: Impact on emotion state
        motivation_impact: Impact on motivation state
    """
    success: bool
    score: float
    feedback: str
    emotion_impact: Dict[str, float] = field(default_factory=dict)
    motivation_impact: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")


@dataclass
class ModelConfig:
    """
    Model configuration.
    
    Attributes:
        name: Model identifier name
        model_path: Path to the model or HuggingFace model ID
        device: Device to load the model on ('auto', 'cuda', 'cpu')
        quantization: Quantization type ('8bit', '4bit', None)
        max_length: Maximum sequence length
        temperature: Generation temperature
    """
    name: str
    model_path: str
    device: str = "auto"
    quantization: Optional[str] = None
    max_length: int = 512
    temperature: float = 0.7
    
    def __post_init__(self):
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        valid_quantizations = {None, "8bit", "4bit"}
        if self.quantization not in valid_quantizations:
            raise ValueError(f"Quantization must be one of {valid_quantizations}")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")


@dataclass
class SystemConfig:
    """
    System-wide configuration.
    
    Attributes:
        models: Dictionary of model configurations
        vector_db_path: Path to vector database storage
        storage_path: Path to data storage
        log_level: Logging level
        max_cached_models: Maximum number of models to cache
        development_stage: Current development stage
    """
    models: Dict[str, ModelConfig]
    vector_db_path: str
    storage_path: str
    log_level: str = "INFO"
    max_cached_models: int = 3
    development_stage: str = "adult"
    
    def __post_init__(self):
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")
        valid_stages = {"infant", "child", "adolescent", "adult"}
        if self.development_stage not in valid_stages:
            raise ValueError(f"Development stage must be one of {valid_stages}")


@dataclass
class ErrorResponse:
    """
    Structured error response.
    
    Attributes:
        error_type: Type of the error
        message: Error message
        details: Additional error details
        timestamp: Unix timestamp of the error
        context: Context information when error occurred
    """
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None


# Type aliases for convenience
DocumentList = List[Document]
MessageList = List[ChatMessage]
ExerciseList = List[Dict[str, str]]
