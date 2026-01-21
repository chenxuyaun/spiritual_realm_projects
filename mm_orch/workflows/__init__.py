"""
Workflow implementations for different task types.

This module provides workflow implementations for various AI tasks
including search-based Q&A, lesson generation, chat, and RAG.
"""

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.workflows.search_qa import SearchQAWorkflow

__all__ = [
    "BaseWorkflow",
    "SearchQAWorkflow"
]
