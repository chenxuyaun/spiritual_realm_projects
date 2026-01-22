"""
ChatGenerate Workflow Implementation.

This module implements the ChatGenerate workflow for multi-turn
conversations with:
1. Session creation and ID generation
2. History retrieval and context building
3. Response generation
4. History update

Supports both mock model manager and real model integration via
RealModelManager, InferenceEngine, and ConversationManager.

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

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import time
import uuid

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType, ChatMessage, ChatSession
from mm_orch.storage.chat_storage import ChatStorage, get_chat_storage
from mm_orch.exceptions import ValidationError, WorkflowError
from mm_orch.logger import get_logger

if TYPE_CHECKING:
    from mm_orch.runtime.real_model_manager import RealModelManager
    from mm_orch.runtime.inference_engine import InferenceEngine
    from mm_orch.runtime.conversation import ConversationManager


logger = get_logger(__name__)


# Default system prompts for different languages
DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a helpful AI assistant. Respond to the user's messages "
    "in a helpful, accurate, and friendly manner. Consider the "
    "conversation history when generating responses."
)

DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是一个有帮助的AI助手。请以友好、准确、有帮助的方式回应用户的消息。"
    "在生成回复时，请考虑对话历史。"
)


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

    Supports real model integration via RealModelManager, InferenceEngine,
    and ConversationManager.

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
        real_model_manager: Optional["RealModelManager"] = None,
        inference_engine: Optional["InferenceEngine"] = None,
        conversation_manager: Optional["ConversationManager"] = None,
        generator_model: str = "gpt2",
        model_type: str = "gpt2",
        max_history_turns: int = 10,
        max_context_length: int = 2000,
        system_prompt: Optional[str] = None,
        use_real_models: bool = False,
        language: str = "en",
    ):
        """
        Initialize the ChatGenerate workflow.

        Args:
            chat_storage: Chat storage instance for history management
            model_manager: Model manager for response generation (mock)
            real_model_manager: Real model manager for actual LLM inference
            inference_engine: Inference engine for real model generation
            conversation_manager: Conversation manager for prompt building
            generator_model: Model name for response generation
            model_type: Model type for conversation format ("qwen-chat", "gpt2", etc.)
            max_history_turns: Maximum history turns to include in context
            max_context_length: Maximum context length in characters
            system_prompt: Optional system prompt for the conversation
            use_real_models: Whether to use real models instead of mock
            language: Output language ("en" or "zh")
        """
        super().__init__()
        self.chat_storage = chat_storage or get_chat_storage()
        self.model_manager = model_manager
        self.real_model_manager = real_model_manager
        self.inference_engine = inference_engine
        self.conversation_manager = conversation_manager
        self.generator_model = generator_model
        self.model_type = model_type
        self.max_history_turns = max_history_turns
        self.max_context_length = max_context_length
        self.use_real_models = use_real_models
        self.language = language
        
        # Set default system prompt based on language
        if system_prompt:
            self.system_prompt = system_prompt
        elif language == "zh":
            self.system_prompt = DEFAULT_SYSTEM_PROMPT_ZH
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT_EN

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["message"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {
            "session_id": None,
            "max_history": self.max_history_turns,
            "temperature": 0.7,
            "system_prompt": None,
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
        ctx = ChatGenerateContext(session_id=session_id or "", message=message)

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
                    ctx, status="failed", error="Failed to generate response"
                )

            # Step 5: Update history
            ctx = self._step_update_history(ctx)

            return self._create_result(ctx, status="success")

        except Exception as e:
            logger.error("ChatGenerate workflow failed", error=str(e), session_id=ctx.session_id)
            return self._create_result(
                ctx, status="partial" if ctx.response else "failed", error=str(e)
            )

    def _step_get_session(
        self, ctx: ChatGenerateContext, session_id: Optional[str]
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
        self, ctx: ChatGenerateContext, max_history: int
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
                limit=max_history * 2,  # Each turn has 2 messages (user + assistant)
            )

            ctx.history = history
            step.success = True

            logger.info(
                f"Retrieved {len(history)} messages from history", session_id=ctx.session_id
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
        self, ctx: ChatGenerateContext, custom_system_prompt: Optional[str]
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
                role_label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                    msg.role, msg.role.capitalize()
                )
                context_parts.append(f"{role_label}: {msg.content}")

            # Add current message
            context_parts.append(f"User: {ctx.message}")
            context_parts.append("Assistant:")

            # Join and truncate if needed
            context_text = "\n".join(context_parts)

            if len(context_text) > self.max_context_length:
                # Truncate from the beginning (keep recent context)
                context_text = self._truncate_context(context_parts, self.max_context_length)

            ctx.context_text = context_text
            step.success = True

            logger.debug(
                f"Built context of length {len(context_text)}", history_messages=len(ctx.history)
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

    def _truncate_context(self, context_parts: List[str], max_length: int) -> str:
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

    def _step_generate_response(
        self, ctx: ChatGenerateContext, temperature: float
    ) -> ChatGenerateContext:
        """
        Step 4: Generate response using the model.

        Args:
            ctx: Workflow context with context text
            temperature: Generation temperature

        Returns:
            Updated context with response
        """
        start_time = time.time()
        step = ChatGenerateStep(name="generate_response", success=False)

        try:
            logger.info("Generating response")

            response = self._generate_response(ctx.context_text, temperature)
            ctx.response = response
            step.success = bool(response)

            if not response:
                step.error = "Empty response generated"

            logger.info(f"Generated response of length {len(response)}")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Response generation failed: {e}")
            # Try fallback response
            ctx.response = self._fallback_response(ctx.message)
            if ctx.response:
                step.success = True

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _generate_response(self, context: str, temperature: float) -> str:
        """
        Generate a response using the model.

        Args:
            context: Full context including history and current message
            temperature: Generation temperature

        Returns:
            Generated response text
        """
        # Use real models if available and enabled
        if self.use_real_models and self.inference_engine:
            return self._generate_with_real_model(context, temperature)

        if self.model_manager:
            try:
                response = self.model_manager.infer(
                    self.generator_model, context, max_new_tokens=300, temperature=temperature
                )

                # Clean up the response
                response = self._clean_response(response, context)
                return response

            except Exception as e:
                logger.warning(f"Model generation failed: {e}")

        # Fallback: simple echo response
        return self._simple_response(context)

    def _generate_with_real_model(self, context: str, temperature: float) -> str:
        """
        Generate response using real model via InferenceEngine.

        Args:
            context: Full context including history and current message
            temperature: Generation temperature

        Returns:
            Generated response text
        """
        try:
            # If we have a conversation manager, use it to build proper prompt
            if self.conversation_manager:
                prompt = self.conversation_manager.build_prompt(
                    system_prompt=self.system_prompt,
                    include_generation_prompt=True
                )
            else:
                prompt = context

            # Generate using inference engine
            from mm_orch.runtime.inference_engine import GenerationConfig
            
            config = GenerationConfig(
                max_new_tokens=512,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1
            )

            result = self.inference_engine.generate(prompt, config=config)
            response = result.text.strip()

            # Clean up the response
            response = self._clean_response(response, prompt)

            logger.info(
                f"Generated response with real model: {len(response)} chars, "
                f"{result.tokens_per_second:.1f} tokens/s"
            )
            return response

        except Exception as e:
            logger.error(f"Real model generation failed: {e}")
            # Fallback to simple response
            return self._simple_response(context)

    def _step_build_context_with_conversation_manager(
        self, ctx: ChatGenerateContext, custom_system_prompt: Optional[str]
    ) -> ChatGenerateContext:
        """
        Build context using ConversationManager for proper model-specific formatting.

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

            # Create or reset conversation manager
            if self.conversation_manager is None:
                from mm_orch.runtime.conversation import ConversationManager
                self.conversation_manager = ConversationManager(
                    model_type=self.model_type,
                    max_history_tokens=self.max_context_length,
                    system_prompt=system_prompt
                )
            else:
                self.conversation_manager.clear_history()
                self.conversation_manager.set_system_prompt(system_prompt)

            # Add history to conversation manager
            for msg in ctx.history:
                self.conversation_manager.add_turn(msg.role, msg.content)

            # Add current message
            self.conversation_manager.add_user_input(ctx.message)

            # Build prompt
            ctx.context_text = self.conversation_manager.build_prompt(
                system_prompt=system_prompt,
                include_generation_prompt=True
            )

            step.success = True

            logger.debug(
                f"Built context with conversation manager: {len(ctx.context_text)} chars",
                history_messages=len(ctx.history)
            )

        except Exception as e:
            step.error = str(e)
            logger.error(f"Failed to build context with conversation manager: {e}")
            # Fallback to simple context building
            ctx.context_text = f"User: {ctx.message}\nAssistant:"
            step.success = True

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _clean_response(self, response: str, context: str) -> str:
        """
        Clean up the generated response.

        Args:
            response: Raw generated response
            context: Original context (to remove if echoed)

        Returns:
            Cleaned response
        """
        if not response:
            return ""

        # Remove the context if it was echoed
        if response.startswith(context):
            response = response[len(context) :]

        # Remove "Assistant:" prefix if present
        if response.strip().startswith("Assistant:"):
            response = response.strip()[10:]

        # Remove any trailing incomplete sentences
        response = response.strip()

        # Remove any "User:" or "System:" that might indicate continuation
        for marker in ["User:", "System:", "\nUser:", "\nSystem:"]:
            if marker in response:
                response = response.split(marker)[0]

        return response.strip()

    def _simple_response(self, context: str) -> str:
        """
        Generate a simple response without using models.

        Args:
            context: The conversation context

        Returns:
            Simple response
        """
        # Extract the user's message from context
        lines = context.strip().split("\n")
        user_message = ""

        for line in reversed(lines):
            if line.startswith("User:"):
                user_message = line[5:].strip()
                break

        if not user_message:
            return "I understand. How can I help you?"

        # Generate a simple acknowledgment
        if "?" in user_message:
            return "That's an interesting question. Let me think about that."
        elif any(word in user_message.lower() for word in ["hello", "hi", "hey"]):
            return "Hello! How can I assist you today?"
        elif any(word in user_message.lower() for word in ["thank", "thanks"]):
            return "You're welcome! Is there anything else I can help with?"
        else:
            return "I understand. Please tell me more about what you need."

    def _fallback_response(self, message: str) -> str:
        """
        Generate a fallback response when generation fails.

        Args:
            message: The user's message

        Returns:
            Fallback response
        """
        return (
            "I apologize, but I'm having trouble generating a response right now. "
            "Could you please try again or rephrase your message?"
        )

    def _step_update_history(self, ctx: ChatGenerateContext) -> ChatGenerateContext:
        """
        Step 5: Update conversation history.

        Property 9: 对话历史持久化 - Messages are persisted.

        Args:
            ctx: Workflow context with response

        Returns:
            Updated context
        """
        start_time = time.time()
        step = ChatGenerateStep(name="update_history", success=False)

        try:
            # Add user message
            self.chat_storage.add_user_message(session_id=ctx.session_id, content=ctx.message)

            # Add assistant response
            self.chat_storage.add_assistant_message(session_id=ctx.session_id, content=ctx.response)

            step.success = True

            logger.debug(
                f"Updated history for session {ctx.session_id}",
                message_length=len(ctx.message),
                response_length=len(ctx.response),
            )

        except Exception as e:
            step.error = str(e)
            logger.error(f"Failed to update history: {e}")
            # Not a fatal error - response was still generated
            step.success = True

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _create_result(
        self, ctx: ChatGenerateContext, status: str = "success", error: Optional[str] = None
    ) -> WorkflowResult:
        """
        Create the workflow result.

        Args:
            ctx: Workflow context
            status: Result status
            error: Error message if any

        Returns:
            WorkflowResult object
        """
        metadata = {
            "workflow": self.name,
            "session_id": ctx.session_id,
            "is_new_session": ctx.is_new_session,
            "history_length": len(ctx.history),
            "context_length": len(ctx.context_text),
            "response_length": len(ctx.response) if ctx.response else 0,
            "steps": [
                {"name": s.name, "success": s.success, "duration": s.duration, "error": s.error}
                for s in ctx.steps
            ],
        }

        return WorkflowResult(
            result=ctx.response if ctx.response else None,
            metadata=metadata,
            status=status,
            error=error,
        )

    # Convenience methods for direct usage

    def chat(self, message: str, session_id: Optional[str] = None, **kwargs) -> WorkflowResult:
        """
        Convenience method for chatting.

        Args:
            message: User message
            session_id: Optional session ID
            **kwargs: Additional parameters

        Returns:
            WorkflowResult with response
        """
        parameters = {"message": message, "session_id": session_id, **kwargs}
        return self.run(parameters)

    def start_conversation(
        self, initial_message: str, system_prompt: Optional[str] = None, **kwargs
    ) -> WorkflowResult:
        """
        Start a new conversation.

        Args:
            initial_message: First message in the conversation
            system_prompt: Optional custom system prompt
            **kwargs: Additional parameters

        Returns:
            WorkflowResult with response and new session_id in metadata
        """
        parameters = {
            "message": initial_message,
            "session_id": None,  # Force new session
            "system_prompt": system_prompt,
            **kwargs,
        }
        return self.run(parameters)

    def continue_conversation(self, session_id: str, message: str, **kwargs) -> WorkflowResult:
        """
        Continue an existing conversation.

        Args:
            session_id: Existing session ID
            message: New message
            **kwargs: Additional parameters

        Returns:
            WorkflowResult with response
        """
        parameters = {"message": message, "session_id": session_id, **kwargs}
        return self.run(parameters)

    def get_session_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get the history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return

        Returns:
            List of ChatMessages
        """
        return self.chat_storage.get_chat_history(
            session_id=session_id, limit=limit or self.max_history_turns * 2
        )

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session's history.

        Args:
            session_id: Session to clear

        Returns:
            True if cleared
        """
        return self.chat_storage.clear_session(session_id)
