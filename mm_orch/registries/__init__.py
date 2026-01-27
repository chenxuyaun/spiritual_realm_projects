"""
Registry system for tools, models, and workflows.

This module provides centralized registration and discovery
for external tools, ML models, and workflow definitions.
"""

from mm_orch.registries.tool_registry import ToolRegistry, ToolMetadata
from mm_orch.registries.model_registry import ModelRegistry, ModelMetadata
from mm_orch.registries.workflow_registry import (
    WorkflowRegistry,
    WorkflowDefinition,
    get_workflow_registry,
    reset_workflow_registry
)
from mm_orch.registries.workflow_definitions import (
    create_search_qa_workflow,
    create_rag_qa_workflow,
    create_lesson_pack_workflow,
    create_chat_generate_workflow,
    register_default_workflows
)

__all__ = [
    "ToolRegistry",
    "ToolMetadata",
    "ModelRegistry",
    "ModelMetadata",
    "WorkflowRegistry",
    "WorkflowDefinition",
    "get_workflow_registry",
    "reset_workflow_registry",
    "create_search_qa_workflow",
    "create_rag_qa_workflow",
    "create_lesson_pack_workflow",
    "create_chat_generate_workflow",
    "register_default_workflows",
]
