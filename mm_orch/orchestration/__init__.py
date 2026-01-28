"""
Orchestration layer for Phase B extensible workflow execution.

This module provides the core abstractions for graph-based workflow execution:
- Step: Protocol defining the interface for workflow steps
- State: TypedDict containing all workflow execution data
- BaseStep: Abstract base class with validation and helper methods
- GraphExecutor: Executes workflow graphs with tracing support
- GraphNode: Node definition for workflow graphs
- Workflow Steps: Refactored implementations of common workflow steps
- State Utilities: Helper functions for State creation with mode settings
"""

from mm_orch.orchestration.step import Step
from mm_orch.orchestration.state import State
from mm_orch.orchestration.base_step import BaseStep, FunctionStep
from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode, SimpleTracer
from mm_orch.orchestration.workflow_steps import (
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep,
)
from mm_orch.orchestration.state_utils import (
    create_state,
    get_mode_from_state,
    set_mode_in_state,
    is_chat_mode,
)

__all__ = [
    "Step",
    "State",
    "BaseStep",
    "FunctionStep",
    "GraphExecutor",
    "GraphNode",
    "SimpleTracer",
    "WebSearchStep",
    "FetchUrlStep",
    "SummarizeStep",
    "AnswerGenerateStep",
    "create_state",
    "get_mode_from_state",
    "set_mode_in_state",
    "is_chat_mode",
]
