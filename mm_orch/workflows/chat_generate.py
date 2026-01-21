"""
ChatGenerate Workflow Implementation.

This module implements the ChatGenerate workflow for multi-turn
conversations with:
1. Session creation and ID generation
2. History retrieval and context building
3. Response generation
4. History update

Requirements:
- 4.1: Create new session with unique ID
- 4.2: Retrieve conversation history
- 4.3: Generate context-aware responses
- 4.4: Persist at least 10 recent turns
- 4.5: Sliding window strategy

Properties verified:
- Property 7: 对话会话唯一性
- Property 8: 对话历史上下文传递
- Property 9: 对话历史持久化
- Property 10: 对话历史滑动窗口
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import uuid

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType, ChatMessage, ChatSession
from mm_orch.storage.chat_storage import ChatStorage, get_chat_storage
from mm_orch.exceptions import ValidationError, WorkflowError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ChatGenerateStep:
    """Tracks the execution of a workflow step."""
    name: str
    success: bool
    duration: float = 0.0
    error: Optional[str] = None


@dataclass
class ChatGenerateContext:
    """Context for ChatGenerate workflow execution."""
    session_id: str
    message: str
    is_new_session: bool = False
    history: List[ChatMessage] = field(default_factory=list)
    context_text: str = ""
    response: str = ""
    steps: List[ChatGenerateStep] = field(default_factory=list)
    
    def add_step(self, step: ChatGenerateStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)


class ChatGenerateWorkflow(BaseWorkflow):
    """
    ChatGenerate Workflow: Retrieve History → Build Context → Generate → Update
    
    This workflow handles multi-turn conversations by:
    1. Creating or retrieving a chat session
    2. Retrieving conversation history
    3. Building context from history
    4. Generating a response
    5. Updating the conversation history
    
    The workflow ensures:
    - Unique session IDs (Property 7)
    - Context-aware responses using history (Property 8)
    - Persistent storage of history (Property 9)
    - Sliding window for history management (Property 10)
    
    Attributes:
        workflow_type: WorkflowType.CHAT_GENERATE
        name: "ChatGenerate"
    """
    
    workflow_type = WorkflowType.CHAT_GENERATE
    name = "ChatGenerate"
    description = "Multi-turn conversation workflow"
    
    def __init__(
        self,
        chat_storage: Optional[ChatStorage] = None,
        model_manager: Optional[Any] = None,
        generator_model: str = "gpt2",
        max_history_turns: int = 10,
        max_context_length: int = 2000,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the ChatGenerate workflow.
        
        Args:
            chat_storage: Chat storage instance for history management
            model_manager: Model manager for response generation
            generator_model: Model name for response generation
            max_history_turns: Maximum history turns to include in context
            max_context_length: Maximum context length in characters
            system_prompt: Optional system prompt for the conversation
        """
        super().__init__()
        self.chat_storage = chat_storage or get_chat_storage()
        self.model_manager = model_manager
        self.generator_model = generator_model
        self.max_history_turns = max_history_turns
        self.max_context_length = max_context_length
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        """Return the default system prompt."""
        return (
            "You are a helpful AI assistant. Respond to the user's messages "
            "in a helpful, accurate, and friendly manner. Consider the "
            "conversation history when generating responses."
        )
    
    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["message"]
    
    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {
            "session_id": None,
            "max_history": self.max_history_turns,
            "temperature": 0.7,
            "system_prompt": None
        }
    
    def get_required_models(self) -> List[str]:
        """Return the list of models required by this workflow."""
        return [self.generator_model]
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate workflow parameters.
        
        Args:
            parameters: Parameters to validate
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If parameters are invalid
        """
        self._validate_required_parameters(parameters)
        
        message = parameters.get("message", "")
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")
        
        max_history = parameters.get("max_history", self.max_history_turns)
        if not isinstance(max_history, int) or max_history < 0:
            raise ValidationError("max_history must be a non-negative integer")
        
        temperature = parameters.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
            raise ValidationError("temperature must be between 0.0 and 2.0")
        
        return True
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the ChatGenerate workflow.
        
        Steps:
        1. Session: Get or create chat session
        2. History: Retrieve conversation history
        3. Context: Build context from history
        4. Generate: Generate response
        5. Update: Update conversation history
        
        Args:
            parameters: Workflow parameters including 'message'
        
        Returns:
            WorkflowResult with response and metadata
        """
        message = parameters["message"]
        session_id = parameters.get("session_id")
        max_history = parameters.get("max_history", self.max_history_turns)
        temperature = parameters.get("temperature", 0.7)
        custom_system_prompt = parameters.get("system_prompt")
        
        # Initialize context
        ctx = ChatGenerateContext(
            session_id=session_id or "",
            message=message
        )
        
        try:
            # Step 1: Get or create session
            ctx = self._step_get_session(ctx, session_id)
            
            # Step 2: Retrieve history
            ctx = self._step_retrieve_history(ctx, max_history)
            
            # Step 3: Build context
            ctx = self._step_build_context(ctx, custom_system_prompt)
            
            # Step 4: Generate response
            ctx = self._step_generate_response(ctx, temperature)
            
            if not ctx.response:
                return self._create_result(
                    ctx,
                    status="failed",
                    error="Failed to generate response"
                )
            
            # Step 5: Update history
            ctx = self._step_update_history(ctx)
            
            return self._create_result(ctx, status="success")
            
        except Exception as e:
            logger.error(
                "ChatGenerate workflow failed",
                error=str(e),
                session_id=ctx.session_id
            )
            return self._create_result(
                ctx,
                status="partial" if ctx.response else "failed",
                error=str(e)
            )
    
    def _step_get_session(
        self, 
        ctx: ChatGenerateContext, 
        session_id: Optional[str]
    ) -> ChatGenerateContext:
        """
        Step 1: Get or create a chat session.
        
        Property 7: 对话会话唯一性 - New sessions get unique IDs.
        
        Args:
            ctx: Workflow context
            session_id: Optional existing session ID
        
        Returns:
            Updated context with session
        """
        start_time = time.time()
        step = ChatGenerateStep(name="get_session", success=False)
        
        try:
            if session_id and self.chat_storage.session_exists(session_id):
                # Use existing session
                ctx.session_id = session_id
                ctx.is_new_session = False
                logger.info(f"Using existing session: {session_id}")
            else:
                # Create new session
                session = self.chat_storage.create_session()
                ctx.session_id = session.session_id
                ctx.is_new_session = True
                logger.info(f"Created new session: {session.session_id}")
            
            step.success = True
            
        except Exception as e:
            step.error = str(e)
            logger.error(f"Failed to get/create session: {e}")
            raise WorkflowError(f"Session management failed: {e}")
        
        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_retrieve_history(
        self, 
        ctx: ChatGenerateContext, 
        max_history: int
    ) -> ChatGenerateContext:
        """
        Step 2: Retrieve conversation history.
        
        Property 8: 对话历史上下文传递 - History is retrieved for context.
        Property 10: 对话历史滑动窗口 - Only recent messages are retrieved.
        
        Args:
            ctx: Workflow context
            max_history: Maximum history turns to retrieve
        
        Returns:
            Updated context with history
        """
        start_time = time.time()
        step = ChatGenerateStep(name="retrieve_history", success=False)
        
        try:
            # Retrieve history with sliding window
            history = self.chat_storage.get_chat_history(
                session_id=ctx.session_id,
                limit=max_history * 2  # Each turn has 2 messages (user + assistant)
            )
            
            ctx.history = history
            step.success = True
            
            logger.info(
                f"Retrieved {len(history)} messages from history",
                session_id=ctx.session_id
            )
            
        except Exception as e:
            step.error = str(e)
            logger.warning(f"Failed to retrieve history: {e}")
            # Continue with empty history
            ctx.history = []
            step.success = True  # Not a fatal error
        
        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx
    
    def _step_build_context(
        self, 
        ctx: ChatGenerateContext,
        custom_system_prompt: Optional[str]
    ) -> ChatGenerateContext:
        """
        Step 3: Build context from history for model input.
        
        Property 8: 对话历史上下文传递 - Context includes history.
        
        Args:
            ctx: Workflow context
            custom_system_prompt: Optional custom system prompt
        
        Returns:
            Updated context with context text
        """
        start_time = time.time()
        step = ChatGenerateStep(name="build_context", success=False)
        
        try:
            system_prompt = custom_system_prompt or self.system_prompt
            
            # Build context string
            context_parts = []
            
            # Add system prompt
            if system_prompt:
                context_parts.append(f"System: {system_prompt}")
            
            # Add history
            for msg in ctx.history:
                role_label = {
                    "user": "User",
                    "assistant": "Assistant",
                    "system": "System"
                }.get(msg.role, msg.role.capitalize())
                context_parts.append(f"{role_label}: {msg.content}")
            
            # Add current message
            context_parts.append(f"User: {ctx.message}")
            context_parts.append("Assistant:")
            
            # Join and truncate if needed
            context_text = "\n".join(context_parts)
            
            if len(context_text) > self.max_context_length:
                # Truncate from the beginning (keep recent context)
                context_text = self._truncate_context(
                    context_parts, 
                    self.max_context_length
                )
            
            ctx.context_text = context_text
            step.success = True
            
            logger.debug(
                f"Built context of length {len(context_text)}",
                history_messages=len(ctx.history)
            )
            
        except Exception as e:
            step.error = str(e)
            logger.error(f"Failed to build context: {e}")
            # Fallback: use just the current message
            ctx.context_text = f"User: {ctx.message}\nAssistant:"
            step.success = True
        
        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx
    
    def _truncate_context(
        self, 
        context_parts: List[str], 
        max_length: int
    ) -> str:
        """
        Truncate context to fit within max_length.
        
        Keeps the system prompt, most recent messages, and current message.
        
        Args:
            context_parts: List of context parts
            max_length: Maximum length
        
        Returns:
            Truncated context string
        """
        if not context_parts:
            return ""
        
        # Always keep the last two parts (current message and "Assistant:")
        essential_parts = context_parts[-2:]
        essential_length = sum(len(p) + 1 for p in essential_parts)
        
        if essential_length >= max_length:
            # Just return essential parts
            return "\n".join(essential_parts)
        
        remaining_length = max_length - essential_length
        
        # Try to keep system prompt if present
        result_parts = []
        if context_parts and context_parts[0].startswith("System:"):
            system_part = context_parts[0]
            if len(system_part) < remaining_length // 2:
                result_parts.append(system_part)
                remaining_length -= len(system_part) + 1
                context_parts = context_parts[1:-2]
            else:
                context_parts = context_parts[:-2]
        else:
            context_parts = context_parts[:-2]
        
        # Add as many recent messages as possible
        for part in reversed(context_parts):
            if len(part) + 1 <= remaining_length:
                result_parts.insert(len(result_parts), part)
                remaining_length -= len(part) + 1
            else:
                break
        
        result_parts.extend(essential_parts)
        return "\n".join(result_parts)
