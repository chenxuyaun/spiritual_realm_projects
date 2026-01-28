"""
Compatibility Module - Helpers for Phase A/B interoperability.

This module provides utilities to ensure Phase A workflows continue
to work in Phase B environment, including:
- Runtime context creation
- State/parameter conversion
- Workflow execution wrappers
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.orchestration.state import State
from mm_orch.orchestration.legacy_adapter import LegacyWorkflowAdapter
from mm_orch.schemas import WorkflowResult
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class LegacyRuntime:
    """
    Runtime context for legacy workflows.

    This provides a compatible runtime environment that Phase A workflows
    expect, while working within the Phase B execution model.

    Attributes:
        model_manager: Model manager for inference
        real_model_manager: Real model manager for actual LLMs
        inference_engine: Inference engine for generation
        conversation_manager: Conversation manager for chat
        search_tool: Web search tool
        fetch_tool: URL fetch tool
        vector_db: Vector database for RAG
        chat_storage: Chat history storage
    """

    model_manager: Optional[Any] = None
    real_model_manager: Optional[Any] = None
    inference_engine: Optional[Any] = None
    conversation_manager: Optional[Any] = None
    search_tool: Optional[Any] = None
    fetch_tool: Optional[Any] = None
    vector_db: Optional[Any] = None
    chat_storage: Optional[Any] = None


def execute_legacy_workflow(
    workflow: BaseWorkflow, state: State, runtime: Optional[LegacyRuntime] = None
) -> State:
    """
    Execute a Phase A workflow using Phase B state.

    This function provides a bridge for executing legacy workflows
    in the new execution model without requiring code changes.

    Args:
        workflow: Phase A BaseWorkflow instance
        state: Phase B State object
        runtime: Optional runtime context

    Returns:
        Updated State with workflow results

    Example:
        from mm_orch.workflows.search_qa import SearchQAWorkflow

        workflow = SearchQAWorkflow()
        state = {"question": "What is Python?", "meta": {}}
        runtime = LegacyRuntime(model_manager=my_model_manager)

        result_state = execute_legacy_workflow(workflow, state, runtime)
        print(result_state["final_answer"])
    """
    # Create adapter
    adapter = LegacyWorkflowAdapter(workflow)

    # Execute using adapter
    runtime_obj = runtime or LegacyRuntime()
    return adapter.run(state, runtime_obj)


def convert_parameters_to_state(
    parameters: Dict[str, Any], workflow_type: Optional[str] = None
) -> State:
    """
    Convert Phase A workflow parameters to Phase B State.

    This helper creates a State object from legacy parameter dictionaries,
    mapping common parameter names to State fields.

    Args:
        parameters: Legacy workflow parameters
        workflow_type: Optional workflow type hint for better mapping

    Returns:
        State object with parameters mapped to appropriate fields

    Example:
        params = {"query": "What is Python?", "max_results": 5}
        state = convert_parameters_to_state(params, "search_qa")
        # state = {"question": "What is Python?", "meta": {"max_results": 5}}
    """
    state: State = {"meta": {}}

    # Common parameter mappings
    if "query" in parameters:
        state["question"] = parameters["query"]
    elif "question" in parameters:
        state["question"] = parameters["question"]
    elif "message" in parameters:
        state["question"] = parameters["message"]
    elif "topic" in parameters:
        state["lesson_topic"] = parameters["topic"]
        state["question"] = parameters["topic"]  # Also set question for consistency

    # Session/conversation parameters
    if "session_id" in parameters:
        state["conversation_id"] = parameters["session_id"]

    # Store other parameters in metadata
    for key, value in parameters.items():
        if key not in ["query", "question", "message", "topic", "session_id"]:
            state["meta"][key] = value

    # Add workflow type hint if provided
    if workflow_type:
        state["meta"]["workflow_type"] = workflow_type

    return state


def convert_state_to_parameters(
    state: State, required_params: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert Phase B State to Phase A workflow parameters.

    This is the reverse of convert_parameters_to_state, extracting
    parameters from State for legacy workflow execution.

    Args:
        state: Phase B State object
        required_params: Optional list of required parameter names

    Returns:
        Dictionary of parameters for legacy workflow

    Example:
        state = {"question": "What is Python?", "meta": {"max_results": 5}}
        params = convert_state_to_parameters(state, ["query"])
        # params = {"query": "What is Python?", "max_results": 5}
    """
    parameters: Dict[str, Any] = {}

    # Map State fields to common parameter names
    if "question" in state:
        parameters["query"] = state["question"]
        parameters["question"] = state["question"]
        parameters["message"] = state["question"]

    if "lesson_topic" in state:
        parameters["topic"] = state["lesson_topic"]

    if "conversation_id" in state:
        parameters["session_id"] = state["conversation_id"]

    # Extract parameters from metadata
    if "meta" in state:
        for key, value in state["meta"].items():
            if key not in ["workflow_type", "workflow_name", "workflow_status"]:
                parameters[key] = value

    # Filter to required params if specified
    if required_params:
        parameters = {k: v for k, v in parameters.items() if k in required_params}

    return parameters


def convert_workflow_result_to_state(
    result: WorkflowResult, original_state: Optional[State] = None
) -> State:
    """
    Convert Phase A WorkflowResult to Phase B State.

    This helper extracts data from WorkflowResult and creates
    a State object with appropriate field mappings.

    Args:
        result: WorkflowResult from legacy workflow
        original_state: Optional original state to preserve fields

    Returns:
        State object with workflow results

    Example:
        result = workflow.run({"query": "What is Python?"})
        state = convert_workflow_result_to_state(result)
        print(state["final_answer"])
    """
    state: State = original_state.copy() if original_state else {}

    # Set final answer
    if result.result is not None:
        state["final_answer"] = str(result.result)

    # Extract metadata
    if not state.get("meta"):
        state["meta"] = {}

    state["meta"]["workflow_status"] = result.status
    state["meta"]["workflow_execution_time"] = result.execution_time

    if result.error:
        state["meta"]["workflow_error"] = result.error

    # Extract workflow-specific data from metadata
    if result.metadata:
        # Sources/citations
        if "sources" in result.metadata:
            state["citations"] = [
                source.get("url", source.get("doc_id", "")) for source in result.metadata["sources"]
            ]

        # Session info
        if "session_id" in result.metadata:
            state["conversation_id"] = result.metadata["session_id"]

        # Store full metadata
        state["meta"]["workflow_metadata"] = result.metadata

    return state


class LegacyWorkflowExecutor:
    """
    Executor for running Phase A workflows in Phase B environment.

    This class provides a high-level interface for executing legacy
    workflows with automatic state conversion and runtime management.

    Example:
        executor = LegacyWorkflowExecutor(runtime)

        # Execute workflow
        result_state = executor.execute(
            workflow=SearchQAWorkflow(),
            parameters={"query": "What is Python?"}
        )

        print(result_state["final_answer"])
    """

    def __init__(self, runtime: Optional[LegacyRuntime] = None):
        """
        Initialize executor with runtime context.

        Args:
            runtime: Optional runtime context for workflows
        """
        self.runtime = runtime or LegacyRuntime()
        logger.debug("Initialized LegacyWorkflowExecutor")

    def execute(
        self,
        workflow: BaseWorkflow,
        parameters: Optional[Dict[str, Any]] = None,
        state: Optional[State] = None,
    ) -> State:
        """
        Execute a legacy workflow with automatic conversion.

        Accepts either parameters (Phase A style) or state (Phase B style).

        Args:
            workflow: Phase A workflow to execute
            parameters: Optional Phase A parameters
            state: Optional Phase B state

        Returns:
            Phase B State with results

        Raises:
            ValueError: If neither parameters nor state is provided
        """
        # Convert parameters to state if needed
        if state is None:
            if parameters is None:
                raise ValueError("Must provide either parameters or state")
            state = convert_parameters_to_state(parameters, workflow.workflow_type.value)

        # Execute workflow
        logger.info(f"Executing legacy workflow '{workflow.name}'")
        result_state = execute_legacy_workflow(workflow, state, self.runtime)

        logger.info(
            f"Legacy workflow '{workflow.name}' completed",
            status=result_state.get("meta", {}).get("workflow_status", "unknown"),
        )

        return result_state

    def execute_with_result(
        self, workflow: BaseWorkflow, parameters: Dict[str, Any]
    ) -> WorkflowResult:
        """
        Execute workflow and return Phase A WorkflowResult.

        This method maintains full Phase A compatibility by returning
        the original WorkflowResult format.

        Args:
            workflow: Phase A workflow to execute
            parameters: Phase A parameters

        Returns:
            WorkflowResult in Phase A format
        """
        # Direct execution using Phase A interface
        logger.info(f"Executing legacy workflow '{workflow.name}' (Phase A mode)")

        # Inject runtime dependencies
        self._inject_dependencies(workflow)

        # Execute
        result = workflow.run(parameters)

        logger.info(
            f"Legacy workflow '{workflow.name}' completed (Phase A mode)", status=result.status
        )

        return result

    def _inject_dependencies(self, workflow: BaseWorkflow) -> None:
        """Inject runtime dependencies into workflow."""
        if hasattr(workflow, "model_manager") and workflow.model_manager is None:
            workflow.model_manager = self.runtime.model_manager

        if hasattr(workflow, "real_model_manager") and workflow.real_model_manager is None:
            workflow.real_model_manager = self.runtime.real_model_manager

        if hasattr(workflow, "inference_engine") and workflow.inference_engine is None:
            workflow.inference_engine = self.runtime.inference_engine

        if hasattr(workflow, "conversation_manager") and workflow.conversation_manager is None:
            workflow.conversation_manager = self.runtime.conversation_manager

        if hasattr(workflow, "search_tool") and workflow.search_tool is None:
            workflow.search_tool = self.runtime.search_tool

        if hasattr(workflow, "fetch_tool") and workflow.fetch_tool is None:
            workflow.fetch_tool = self.runtime.fetch_tool

        if hasattr(workflow, "vector_db") and workflow.vector_db is None:
            workflow.vector_db = self.runtime.vector_db

        if hasattr(workflow, "chat_storage") and workflow.chat_storage is None:
            workflow.chat_storage = self.runtime.chat_storage
