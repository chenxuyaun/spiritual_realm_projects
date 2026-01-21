"""
MuAI多模型编排系统 (MuAI Multi-Model Orchestration System)

A general-purpose AI system with consciousness modules, multi-workflow orchestration,
and teaching capabilities.
"""

__version__ = "0.1.0"

from mm_orch.router import Router, RoutingRule, get_router, create_router, reset_router
from mm_orch.orchestrator import (
    WorkflowOrchestrator,
    get_orchestrator,
    create_orchestrator,
    reset_orchestrator,
)

__all__ = [
    "Router",
    "RoutingRule",
    "get_router",
    "create_router",
    "reset_router",
    "WorkflowOrchestrator",
    "get_orchestrator",
    "create_orchestrator",
    "reset_orchestrator",
]
