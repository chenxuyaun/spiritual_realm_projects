"""
Phase B Orchestrator - Enhanced orchestrator with Phase B components.

This module provides an enhanced orchestrator that integrates:
- Graph Executor for workflow execution
- Workflow Registry for workflow discovery
- Router v3/v2/v1 with automatic fallback
- Tracer for comprehensive observability
- Configuration fallback to Phase A

Requirements: 22.1, 22.2, 23.1
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from mm_orch.schemas import UserRequest, WorkflowResult, WorkflowType
from mm_orch.orchestration.graph_executor import GraphExecutor, SimpleTracer
from mm_orch.orchestration.state import State
from mm_orch.orchestration.config_fallback import get_config_manager, ConfigFallbackResult
from mm_orch.orchestration.compatibility import LegacyRuntime
from mm_orch.registries.workflow_registry import WorkflowRegistry
from mm_orch.logger import get_logger
from mm_orch.exceptions import OrchestrationError


logger = get_logger(__name__)


@dataclass
class PhaseBComponents:
    """
    Container for Phase B components.

    Attributes:
        graph_executor: GraphExecutor for workflow execution
        workflow_registry: WorkflowRegistry for workflow discovery
        router: Router instance (v3, v2, or v1)
        tracer: Tracer for observability
        step_registry: Registry of available steps
    """

    graph_executor: Optional[GraphExecutor] = None
    workflow_registry: Optional[WorkflowRegistry] = None
    router: Optional[Any] = None
    tracer: Optional[Any] = None
    step_registry: Optional[Dict[str, Any]] = None


class PhaseBOrchestrator:
    """
    Enhanced orchestrator with Phase B components and Phase A fallback.

    This orchestrator:
    1. Attempts to use Phase B components (Graph Executor, Workflow Registry, Router v3)
    2. Falls back to Phase A behavior if Phase B components unavailable
    3. Ensures all executions are traced for observability
    4. Maintains backward compatibility with existing workflows

    Example:
        # Create orchestrator with automatic component detection
        orchestrator = PhaseBOrchestrator()

        # Process request (uses Phase B if available, Phase A otherwise)
        result = orchestrator.process_request(request)
    """

    def __init__(
        self,
        phase_b_components: Optional[PhaseBComponents] = None,
        legacy_runtime: Optional[LegacyRuntime] = None,
        config_dir: Optional[str] = None,
    ):
        """
        Initialize Phase B orchestrator.

        Args:
            phase_b_components: Optional Phase B components (auto-detected if None)
            legacy_runtime: Optional legacy runtime for Phase A workflows
            config_dir: Optional custom config directory
        """
        self.config_manager = get_config_manager(config_dir)
        self.legacy_runtime = legacy_runtime or LegacyRuntime()

        # Try to initialize Phase B components
        if phase_b_components:
            self.phase_b = phase_b_components
        else:
            self.phase_b = self._initialize_phase_b_components()

        # Track which mode we're using
        self.using_phase_b = self._check_phase_b_available()

        logger.info(
            "PhaseBOrchestrator initialized",
            using_phase_b=self.using_phase_b,
            has_graph_executor=self.phase_b.graph_executor is not None,
            has_workflow_registry=self.phase_b.workflow_registry is not None,
            has_router=self.phase_b.router is not None,
        )

    def _initialize_phase_b_components(self) -> PhaseBComponents:
        """
        Initialize Phase B components with configuration fallback.

        Returns:
            PhaseBComponents with initialized components (or None for unavailable)
        """
        components = PhaseBComponents()

        # Try to initialize tracer
        try:
            tracer_config = self.config_manager.load_tracer_config()
            if tracer_config.config.get("enabled", True):
                # Use simple tracer for now (full tracer in Phase B4)
                components.tracer = SimpleTracer()
                logger.info("Initialized tracer", config_source=tracer_config.config_source)
        except Exception as e:
            logger.warning(f"Failed to initialize tracer: {e}")

        # Try to initialize step registry
        try:
            from mm_orch.orchestration.workflow_steps import get_step_registry

            components.step_registry = get_step_registry()
            logger.info("Initialized step registry", num_steps=len(components.step_registry))
        except Exception as e:
            logger.warning(f"Failed to initialize step registry: {e}")

        # Try to initialize workflow registry
        if components.step_registry:
            try:
                components.workflow_registry = WorkflowRegistry(components.step_registry)

                # Register workflows from config
                workflow_config = self.config_manager.load_workflow_registry_config()
                self._register_workflows_from_config(components.workflow_registry, workflow_config)
                logger.info(
                    "Initialized workflow registry",
                    config_source=workflow_config.config_source,
                    num_workflows=len(components.workflow_registry.list_all()),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize workflow registry: {e}")

        # Try to initialize graph executor
        if components.step_registry:
            try:
                components.graph_executor = GraphExecutor(
                    components.step_registry, components.tracer
                )
                logger.info("Initialized graph executor")
            except Exception as e:
                logger.warning(f"Failed to initialize graph executor: {e}")

        # Try to initialize router (v3 -> v2 -> v1 fallback)
        try:
            router_config = self.config_manager.load_router_config()
            components.router = self._initialize_router(router_config)
            logger.info(
                "Initialized router",
                version=router_config.config.get("router_version", "unknown"),
                config_source=router_config.config_source,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize router: {e}")

        return components

    def _initialize_router(self, router_config: ConfigFallbackResult) -> Optional[Any]:
        """
        Initialize router with fallback from v3 -> v2 -> v1.

        Args:
            router_config: Router configuration result

        Returns:
            Router instance or None
        """
        config = router_config.config
        version = config.get("router_version", "v1")

        # Try Router v3 (cost-aware)
        if version == "v3":
            try:
                from mm_orch.routing.router_v3 import RouterV3

                vectorizer_path = config.get("vectorizer_path")
                classifier_path = config.get("classifier_path")
                costs_path = config.get("costs_path")

                if all([vectorizer_path, classifier_path, costs_path]):
                    if all(
                        Path(p).exists() for p in [vectorizer_path, classifier_path, costs_path]
                    ):
                        return RouterV3(
                            vectorizer_path,
                            classifier_path,
                            costs_path,
                            lambda_cost=config.get("lambda_cost", 0.1),
                        )
                    else:
                        logger.warning("Router v3 model files not found, falling back to v2")
                        version = "v2"
            except Exception as e:
                logger.warning(f"Failed to initialize Router v3: {e}, falling back to v2")
                version = "v2"

        # Try Router v2 (classifier-based)
        if version == "v2":
            try:
                from mm_orch.routing.router_v2 import RouterV2

                vectorizer_path = config.get("vectorizer_path")
                classifier_path = config.get("classifier_path")

                if all([vectorizer_path, classifier_path]):
                    if all(Path(p).exists() for p in [vectorizer_path, classifier_path]):
                        return RouterV2(vectorizer_path, classifier_path)
                    else:
                        logger.warning("Router v2 model files not found, falling back to v1")
                        version = "v1"
            except Exception as e:
                logger.warning(f"Failed to initialize Router v2: {e}, falling back to v1")
                version = "v1"

        # Fall back to Router v1 (rule-based)
        if version == "v1":
            try:
                from mm_orch.routing.router_v1 import RouterV1

                rules = config.get("rules", [])
                return RouterV1(rules)
            except Exception as e:
                logger.error(f"Failed to initialize Router v1: {e}")
                return None

        return None

    def _register_workflows_from_config(
        self, registry: WorkflowRegistry, config: ConfigFallbackResult
    ) -> None:
        """
        Register workflows from configuration.

        Args:
            registry: WorkflowRegistry to populate
            config: Workflow configuration result
        """
        workflows = config.config.get("workflows", [])

        for workflow_config in workflows:
            if not workflow_config.get("enabled", True):
                continue

            workflow_name = workflow_config.get("name")
            if not workflow_name:
                continue

            try:
                # Try to get workflow definition
                from mm_orch.registries.workflow_definitions import get_workflow_definition

                definition = get_workflow_definition(workflow_name)
                if definition:
                    registry.register(definition)
                    logger.debug(f"Registered workflow '{workflow_name}'")
            except Exception as e:
                logger.warning(f"Failed to register workflow '{workflow_name}': {e}")

    def _check_phase_b_available(self) -> bool:
        """
        Check if Phase B components are available.

        Returns:
            True if Phase B can be used, False if must fall back to Phase A
        """
        return all(
            [
                self.phase_b.graph_executor is not None,
                self.phase_b.workflow_registry is not None,
                self.phase_b.step_registry is not None,
            ]
        )

    def process_request(self, request: UserRequest) -> WorkflowResult:
        """
        Process user request using Phase B or Phase A.

        Args:
            request: User request to process

        Returns:
            WorkflowResult with execution results
        """
        start_time = time.time()

        # Convert request to State
        state: State = {
            "question": request.query,
            "meta": {
                "mode": request.mode if hasattr(request, "mode") else "default",
                "request_id": str(time.time()),
                "start_time": start_time,
            },
        }

        try:
            # Route request
            workflow_name, confidence = self._route_request(request, state)

            logger.info(
                "Request routed",
                workflow=workflow_name,
                confidence=confidence,
                using_phase_b=self.using_phase_b,
            )

            # Execute workflow
            if self.using_phase_b:
                result_state = self._execute_phase_b_workflow(workflow_name, state)
            else:
                result_state = self._execute_phase_a_workflow(workflow_name, state)

            # Convert state to WorkflowResult
            execution_time = time.time() - start_time

            return WorkflowResult(
                result=result_state.get("final_answer"),
                metadata={
                    "workflow": workflow_name,
                    "confidence": confidence,
                    "execution_time": execution_time,
                    "using_phase_b": self.using_phase_b,
                    "state_keys": list(result_state.keys()),
                },
                status="success" if result_state.get("final_answer") else "partial",
                error=result_state.get("meta", {}).get("workflow_error"),
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Request processing failed: {e}")

            return WorkflowResult(
                result=None,
                metadata={
                    "error_type": type(e).__name__,
                    "execution_time": execution_time,
                    "using_phase_b": self.using_phase_b,
                },
                status="failed",
                error=str(e),
                execution_time=execution_time,
            )

    def _route_request(self, request: UserRequest, state: State) -> Tuple[str, float]:
        """
        Route request to appropriate workflow.

        Args:
            request: User request
            state: Current state

        Returns:
            Tuple of (workflow_name, confidence)
        """
        # Try Phase B router
        if self.phase_b.router:
            try:
                workflow, confidence, candidates = self.phase_b.router.route(request.query, state)
                return workflow, confidence
            except Exception as e:
                logger.warning(f"Phase B router failed: {e}, using fallback")

        # Fall back to Phase A router
        try:
            from mm_orch.router import get_router

            router = get_router()
            selection = router.route(request)

            return selection.workflow_type.value, selection.confidence
        except Exception as e:
            logger.error(f"Phase A router failed: {e}, using default")
            return "search_qa", 0.5

    def _execute_phase_b_workflow(self, workflow_name: str, state: State) -> State:
        """
        Execute workflow using Phase B components.

        Args:
            workflow_name: Name of workflow to execute
            state: Initial state

        Returns:
            Final state after execution
        """
        # Get workflow definition
        workflow_def = self.phase_b.workflow_registry.get(workflow_name)
        if not workflow_def:
            raise OrchestrationError(f"Workflow '{workflow_name}' not found in registry")

        # Execute using graph executor
        result_state = self.phase_b.graph_executor.execute(
            workflow_def.graph, state, self.legacy_runtime
        )

        return result_state

    def _execute_phase_a_workflow(self, workflow_name: str, state: State) -> State:
        """
        Execute workflow using Phase A components.

        Args:
            workflow_name: Name of workflow to execute
            state: Initial state

        Returns:
            Final state after execution
        """
        # Get Phase A workflow
        from mm_orch.orchestrator import get_orchestrator

        orchestrator = get_orchestrator()

        # Map workflow name to WorkflowType
        workflow_type_map = {
            "search_qa": WorkflowType.SEARCH_QA,
            "lesson_pack": WorkflowType.LESSON_PACK,
            "chat_generate": WorkflowType.CHAT_GENERATE,
            "rag_qa": WorkflowType.RAG_QA,
            "self_ask_search_qa": WorkflowType.SELF_ASK_SEARCH_QA,
        }

        workflow_type = workflow_type_map.get(workflow_name, WorkflowType.SEARCH_QA)

        # Convert state to parameters
        from mm_orch.orchestration.compatibility import convert_state_to_parameters

        parameters = convert_state_to_parameters(state)

        # Execute
        result = orchestrator.execute_workflow(workflow_type, parameters)

        # Convert result back to state
        from mm_orch.orchestration.compatibility import convert_workflow_result_to_state

        result_state = convert_workflow_result_to_state(result, state)

        return result_state

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "using_phase_b": self.using_phase_b,
            "has_graph_executor": self.phase_b.graph_executor is not None,
            "has_workflow_registry": self.phase_b.workflow_registry is not None,
            "has_router": self.phase_b.router is not None,
            "has_tracer": self.phase_b.tracer is not None,
        }

        # Add workflow registry stats
        if self.phase_b.workflow_registry:
            stats["registered_workflows"] = len(self.phase_b.workflow_registry.list_all())
            stats["workflow_names"] = self.phase_b.workflow_registry.list_all()

        # Add tracer stats
        if self.phase_b.tracer and hasattr(self.phase_b.tracer, "get_traces"):
            stats["trace_count"] = len(self.phase_b.tracer.get_traces())

        return stats


# Singleton instance
_phase_b_orchestrator: Optional[PhaseBOrchestrator] = None


def get_phase_b_orchestrator(
    phase_b_components: Optional[PhaseBComponents] = None,
    legacy_runtime: Optional[LegacyRuntime] = None,
    config_dir: Optional[str] = None,
) -> PhaseBOrchestrator:
    """
    Get singleton Phase B orchestrator instance.

    Args:
        phase_b_components: Optional Phase B components
        legacy_runtime: Optional legacy runtime
        config_dir: Optional custom config directory

    Returns:
        PhaseBOrchestrator instance
    """
    global _phase_b_orchestrator
    if _phase_b_orchestrator is None:
        _phase_b_orchestrator = PhaseBOrchestrator(phase_b_components, legacy_runtime, config_dir)
    return _phase_b_orchestrator


def reset_phase_b_orchestrator() -> None:
    """Reset singleton Phase B orchestrator instance."""
    global _phase_b_orchestrator
    _phase_b_orchestrator = None
