"""
Legacy Workflow Adapter - Compatibility layer for Phase A workflows.

This module provides adapters that allow Phase A workflows (BaseWorkflow subclasses)
to work seamlessly in the Phase B environment (Step-based graph execution).

The adapter wraps Phase A workflows and exposes them as Phase B Steps, handling
the translation between the two execution models:
- Phase A: BaseWorkflow.execute(parameters) -> WorkflowResult
- Phase B: Step.run(state, runtime) -> State

This ensures backward compatibility while enabling gradual migration to Phase B.
"""

from typing import Any, Dict, List, Optional
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.state import State
from mm_orch.schemas import WorkflowResult
from mm_orch.logger import get_logger


logger = get_logger(__name__)


class LegacyWorkflowAdapter(BaseStep):
    """
    Adapter that wraps a Phase A BaseWorkflow as a Phase B Step.

    This adapter:
    1. Extracts parameters from State based on workflow requirements
    2. Calls the legacy workflow's execute() method
    3. Converts WorkflowResult back to State updates
    4. Preserves all workflow metadata and error handling

    Example:
        # Wrap a Phase A workflow
        from mm_orch.workflows.search_qa import SearchQAWorkflow

        legacy_workflow = SearchQAWorkflow()
        adapter = LegacyWorkflowAdapter(legacy_workflow)

        # Use as a Phase B step
        state = {"question": "What is Python?", "meta": {}}
        result_state = adapter.run(state, runtime)
    """

    def __init__(self, workflow: BaseWorkflow, parameter_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the adapter with a legacy workflow.

        Args:
            workflow: Phase A BaseWorkflow instance to wrap
            parameter_mapping: Optional mapping from State keys to workflow parameter names
                              If None, uses automatic mapping (question->query, etc.)
                              Example: {"question": "query"} maps state["question"] to params["query"]
        """
        self.workflow = workflow

        # Set up parameter mapping with defaults
        if parameter_mapping is None:
            # Auto-detect common mappings
            required_params = workflow.get_required_parameters()
            self.parameter_mapping = {}

            # Common mappings
            if "query" in required_params:
                self.parameter_mapping["question"] = "query"
            if "message" in required_params:
                self.parameter_mapping["question"] = "message"
            if "topic" in required_params:
                self.parameter_mapping["lesson_topic"] = "topic"
        else:
            self.parameter_mapping = parameter_mapping

        # Set Step protocol attributes based on workflow
        self.name = workflow.name

        # Determine input keys - use State keys (left side of mapping)
        required_params = workflow.get_required_parameters()
        self.input_keys = []

        for param in required_params:
            # Find State key that maps to this parameter
            state_key = param
            for sk, pn in self.parameter_mapping.items():
                if pn == param:
                    state_key = sk
                    break
            self.input_keys.append(state_key)

        # Output keys depend on workflow type
        # Most workflows produce "final_answer", but we'll be flexible
        self.output_keys = ["final_answer", "workflow_result"]

        logger.debug(
            f"Created legacy adapter for workflow '{self.name}'",
            input_keys=self.input_keys,
            output_keys=self.output_keys,
            parameter_mapping=self.parameter_mapping,
        )

    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Execute the legacy workflow and convert result to State updates.

        Args:
            state: Current workflow state
            runtime: Runtime context (may contain model_manager, etc.)

        Returns:
            Dictionary of updates to merge into state
        """
        # Extract parameters from state
        parameters = self._extract_parameters(state)

        # Inject runtime dependencies if workflow needs them
        self._inject_runtime_dependencies(runtime)

        # Execute legacy workflow
        logger.info(f"Executing legacy workflow '{self.name}'", parameters=list(parameters.keys()))

        try:
            result: WorkflowResult = self.workflow.run(parameters)

            # Convert WorkflowResult to State updates
            updates = self._convert_result_to_state(result, state)

            logger.info(
                f"Legacy workflow '{self.name}' completed",
                status=result.status,
                execution_time=result.execution_time,
            )

            return updates

        except Exception as e:
            logger.error(f"Legacy workflow '{self.name}' failed", error=str(e))

            # Return partial state with error information
            return {
                "final_answer": None,
                "workflow_result": {"status": "failed", "error": str(e), "workflow": self.name},
                "meta": {
                    **state.get("meta", {}),
                    "workflow_error": str(e),
                    "workflow_status": "failed",
                },
            }

    def _extract_parameters(self, state: State) -> Dict[str, Any]:
        """
        Extract workflow parameters from State.

        Uses parameter_mapping if provided, otherwise direct mapping.
        Also includes optional parameters with defaults.

        Args:
            state: Current state

        Returns:
            Dictionary of parameters for workflow.execute()
        """
        parameters = {}

        # Extract required parameters
        for state_key in self.input_keys:
            # Apply mapping if specified
            param_name = self.parameter_mapping.get(state_key, state_key)

            if state_key in state:
                parameters[param_name] = state[state_key]
            else:
                logger.warning(
                    f"Required parameter '{param_name}' not found in state",
                    workflow=self.name,
                    available_keys=list(state.keys()),
                )

        # Add optional parameters from state if present
        optional_params = self.workflow.get_optional_parameters()
        for param_name, default_value in optional_params.items():
            # Check if this parameter is in state (reverse mapping)
            state_key = param_name
            for sk, pn in self.parameter_mapping.items():
                if pn == param_name:
                    state_key = sk
                    break

            if state_key in state:
                parameters[param_name] = state[state_key]
            # Don't add defaults here - workflow will handle them

        # Add metadata if workflow supports it
        if "meta" in state:
            # Some workflows may use metadata for mode, language, etc.
            meta = state["meta"]
            if "mode" in meta and "mode" not in parameters:
                parameters["mode"] = meta["mode"]
            if "language" in meta and "language" not in parameters:
                parameters["language"] = meta["language"]

        return parameters

    def _inject_runtime_dependencies(self, runtime: Any) -> None:
        """
        Inject runtime dependencies into the workflow if needed.

        Phase A workflows may need access to model_manager, tools, etc.
        This method updates the workflow instance with runtime dependencies.

        Args:
            runtime: Runtime context
        """
        # Inject model_manager if workflow has the attribute and runtime provides it
        if hasattr(self.workflow, "model_manager") and hasattr(runtime, "model_manager"):
            if self.workflow.model_manager is None:
                self.workflow.model_manager = runtime.model_manager
                logger.debug(f"Injected model_manager into workflow '{self.name}'")

        # Inject real model components if available
        if hasattr(self.workflow, "real_model_manager") and hasattr(runtime, "real_model_manager"):
            if self.workflow.real_model_manager is None:
                self.workflow.real_model_manager = runtime.real_model_manager
                logger.debug(f"Injected real_model_manager into workflow '{self.name}'")

        if hasattr(self.workflow, "inference_engine") and hasattr(runtime, "inference_engine"):
            if self.workflow.inference_engine is None:
                self.workflow.inference_engine = runtime.inference_engine
                logger.debug(f"Injected inference_engine into workflow '{self.name}'")

        if hasattr(self.workflow, "conversation_manager") and hasattr(
            runtime, "conversation_manager"
        ):
            if self.workflow.conversation_manager is None:
                self.workflow.conversation_manager = runtime.conversation_manager
                logger.debug(f"Injected conversation_manager into workflow '{self.name}'")

        # Inject tools if workflow needs them
        if hasattr(self.workflow, "search_tool") and hasattr(runtime, "search_tool"):
            if self.workflow.search_tool is None:
                self.workflow.search_tool = runtime.search_tool

        if hasattr(self.workflow, "fetch_tool") and hasattr(runtime, "fetch_tool"):
            if self.workflow.fetch_tool is None:
                self.workflow.fetch_tool = runtime.fetch_tool

        # Inject vector_db for RAG workflows
        if hasattr(self.workflow, "vector_db") and hasattr(runtime, "vector_db"):
            if self.workflow.vector_db is None:
                self.workflow.vector_db = runtime.vector_db

        # Inject chat_storage for chat workflows
        if hasattr(self.workflow, "chat_storage") and hasattr(runtime, "chat_storage"):
            if self.workflow.chat_storage is None:
                self.workflow.chat_storage = runtime.chat_storage

    def _convert_result_to_state(
        self, result: WorkflowResult, original_state: State
    ) -> Dict[str, Any]:
        """
        Convert WorkflowResult to State updates.

        This method maps the workflow result to appropriate State fields
        based on the workflow type and result content.

        Args:
            result: WorkflowResult from legacy workflow
            original_state: Original state (for preserving fields)

        Returns:
            Dictionary of State updates
        """
        updates: Dict[str, Any] = {}

        # Store the complete workflow result for reference
        updates["workflow_result"] = {
            "result": result.result,
            "metadata": result.metadata,
            "status": result.status,
            "error": result.error,
            "execution_time": result.execution_time,
        }

        # Map result to appropriate State field based on workflow type
        # Always set final_answer, even if None
        updates["final_answer"] = str(result.result) if result.result is not None else None

        if result.result is not None:
            # For specific workflow types, also populate specialized fields
            workflow_type = result.metadata.get("workflow", self.name).lower()

            # Check if this is a search workflow (by name or type)
            is_search_workflow = (
                "search" in workflow_type
                or "searchqa" in workflow_type.replace("_", "").replace("-", "")
                or hasattr(self.workflow, "workflow_type")
                and self.workflow.workflow_type
                and "search" in str(self.workflow.workflow_type.value).lower()
            )

            if is_search_workflow:
                # SearchQA workflow - extract sources
                if "sources" in result.metadata:
                    updates["citations"] = [
                        source.get("url", "")
                        for source in result.metadata["sources"]
                        if source.get("url")
                    ]

            elif "lesson" in workflow_type.lower():
                # LessonPack workflow - extract structured content
                if isinstance(result.result, dict):
                    if "plan" in result.result:
                        updates["lesson_outline"] = result.result["plan"]
                    if "explanation" in result.result:
                        updates["teaching_text"] = result.result["explanation"]
                    if "exercises" in result.result:
                        updates["exercises"] = result.result["exercises"]

            elif "rag" in workflow_type.lower():
                # RAG workflow - extract sources
                if "sources" in result.metadata:
                    updates["kb_sources"] = result.metadata["sources"]
                    updates["citations"] = [
                        source.get("doc_id", "") for source in result.metadata["sources"]
                    ]

            elif "chat" in workflow_type.lower():
                # Chat workflow - preserve session info
                if "session_id" in result.metadata:
                    updates["conversation_id"] = result.metadata["session_id"]

        # Update metadata
        meta = original_state.get("meta", {}).copy()
        meta["workflow_name"] = self.name
        meta["workflow_status"] = result.status
        meta["workflow_execution_time"] = result.execution_time

        if result.error:
            meta["workflow_error"] = result.error

        # Merge workflow metadata
        if result.metadata:
            meta["workflow_metadata"] = result.metadata

        updates["meta"] = meta

        return updates


def create_legacy_workflow_step(
    workflow: BaseWorkflow, parameter_mapping: Optional[Dict[str, str]] = None
) -> LegacyWorkflowAdapter:
    """
    Factory function to create a legacy workflow adapter.

    This is a convenience function for creating adapters with a cleaner API.

    Args:
        workflow: Phase A BaseWorkflow instance
        parameter_mapping: Optional parameter name mapping

    Returns:
        LegacyWorkflowAdapter instance that can be used as a Phase B Step

    Example:
        from mm_orch.workflows.search_qa import SearchQAWorkflow

        search_qa = SearchQAWorkflow()
        search_qa_step = create_legacy_workflow_step(search_qa)

        # Register in step registry
        step_registry["search_qa"] = search_qa_step
    """
    return LegacyWorkflowAdapter(workflow, parameter_mapping)


def register_legacy_workflows(
    step_registry: Dict[str, Any],
    workflows: List[BaseWorkflow],
    parameter_mappings: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """
    Register multiple legacy workflows in a step registry.

    This helper function creates adapters for all provided workflows
    and registers them in the step registry using their workflow names.

    Args:
        step_registry: Step registry to populate
        workflows: List of Phase A workflows to register
        parameter_mappings: Optional dict mapping workflow names to parameter mappings

    Example:
        from mm_orch.workflows.search_qa import SearchQAWorkflow
        from mm_orch.workflows.lesson_pack import LessonPackWorkflow

        workflows = [
            SearchQAWorkflow(),
            LessonPackWorkflow()
        ]

        step_registry = {}
        register_legacy_workflows(step_registry, workflows)

        # Now step_registry contains:
        # {"SearchQA": <adapter>, "LessonPack": <adapter>}
    """
    parameter_mappings = parameter_mappings or {}

    for workflow in workflows:
        workflow_name = workflow.name
        mapping = parameter_mappings.get(workflow_name)

        adapter = create_legacy_workflow_step(workflow, mapping)
        step_registry[workflow_name] = adapter

        logger.info(
            f"Registered legacy workflow '{workflow_name}' as step",
            input_keys=adapter.input_keys,
            output_keys=adapter.output_keys,
        )
