"""
MuAI多模型编排系统 (MuAI Multi-Model Orchestration System)

A general-purpose AI system with consciousness modules, multi-workflow orchestration,
and teaching capabilities.
"""

from mm_orch.version import (
    __version__,
    __version_info__,
    get_version,
    get_version_info,
    get_full_version,
    get_release_info,
    is_feature_enabled,
)

from mm_orch.router import Router, RoutingRule, get_router, create_router, reset_router
from mm_orch.orchestrator import (
    WorkflowOrchestrator,
    get_orchestrator,
    create_orchestrator,
    reset_orchestrator,
)

__all__ = [
    # Version information
    "__version__",
    "__version_info__",
    "get_version",
    "get_version_info",
    "get_full_version",
    "get_release_info",
    "is_feature_enabled",
    # Router
    "Router",
    "RoutingRule",
    "get_router",
    "create_router",
    "reset_router",
    # Orchestrator
    "WorkflowOrchestrator",
    "get_orchestrator",
    "create_orchestrator",
    "reset_orchestrator",
]
