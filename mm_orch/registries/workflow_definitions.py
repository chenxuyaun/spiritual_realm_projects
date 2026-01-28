"""
Workflow Definitions - Pre-defined workflow graphs for common tasks.

This module provides factory functions to create WorkflowDefinition instances
for the existing workflows: search_qa, rag_qa, lesson_pack, and chat_generate.
"""

from mm_orch.orchestration.graph_executor import GraphNode
from mm_orch.registries.workflow_registry import WorkflowDefinition


def create_search_qa_workflow() -> WorkflowDefinition:
    """
    Create the search_qa workflow definition.

    Flow: web_search → fetch_url → summarize → answer_generate

    Returns:
        WorkflowDefinition for search-based question answering
    """
    return WorkflowDefinition(
        name="search_qa",
        description="Search-based question answering with web search, content extraction, summarization, and answer generation",
        graph={
            "start": GraphNode(step_name="web_search", next_nodes=["fetch"]),
            "fetch": GraphNode(step_name="fetch_url", next_nodes=["summarize"]),
            "summarize": GraphNode(step_name="summarize", next_nodes=["answer"]),
            "answer": GraphNode(step_name="answer_generate", next_nodes=["end"]),
            # Note: "end" is a special node name that signals completion
            # It doesn't need to be in the graph or reference a step
        },
        required_capabilities=["search", "fetch", "summarize", "generate"],
        tags=["qa", "search", "web"],
        metadata={"typical_latency_ms": 5000, "typical_vram_mb": 2000, "complexity": "medium"},
    )


def create_rag_qa_workflow() -> WorkflowDefinition:
    """
    Create the rag_qa workflow definition.

    Flow: embed_query → retrieve_docs → answer_generate

    Note: This is a placeholder. The actual RAG workflow steps
    need to be implemented in the orchestration layer.

    Returns:
        WorkflowDefinition for RAG-based question answering
    """
    return WorkflowDefinition(
        name="rag_qa",
        description="Retrieval-augmented generation question answering using knowledge base",
        graph={
            "start": GraphNode(step_name="embed_query", next_nodes=["retrieve"]),
            "retrieve": GraphNode(step_name="retrieve_docs", next_nodes=["answer"]),
            "answer": GraphNode(step_name="answer_generate", next_nodes=["end"]),
        },
        required_capabilities=["embed", "retrieve", "generate"],
        tags=["qa", "rag", "knowledge_base"],
        metadata={"typical_latency_ms": 2000, "typical_vram_mb": 1500, "complexity": "medium"},
    )


def create_lesson_pack_workflow() -> WorkflowDefinition:
    """
    Create the lesson_pack workflow definition.

    Flow: lesson_plan → lesson_explain → lesson_exercises

    Note: This is a placeholder. The actual lesson pack workflow steps
    need to be implemented in the orchestration layer.

    Returns:
        WorkflowDefinition for lesson pack generation
    """
    return WorkflowDefinition(
        name="lesson_pack",
        description="Generate structured teaching content including lesson plan, explanations, and exercises",
        graph={
            "start": GraphNode(step_name="lesson_plan", next_nodes=["explain"]),
            "explain": GraphNode(step_name="lesson_explain", next_nodes=["exercises"]),
            "exercises": GraphNode(step_name="lesson_exercises", next_nodes=["end"]),
        },
        required_capabilities=["generate", "structure"],
        tags=["education", "teaching", "lesson"],
        metadata={"typical_latency_ms": 8000, "typical_vram_mb": 3000, "complexity": "high"},
    )


def create_chat_generate_workflow() -> WorkflowDefinition:
    """
    Create the chat_generate workflow definition.

    Flow: load_context → generate_response

    Note: This is a placeholder. The actual chat workflow steps
    need to be implemented in the orchestration layer.

    Returns:
        WorkflowDefinition for conversational generation
    """
    return WorkflowDefinition(
        name="chat_generate",
        description="Context-aware conversational response generation",
        graph={
            "start": GraphNode(step_name="load_context", next_nodes=["generate"]),
            "generate": GraphNode(step_name="generate_response", next_nodes=["end"]),
        },
        required_capabilities=["generate", "context"],
        tags=["chat", "conversation", "fast"],
        metadata={"typical_latency_ms": 1500, "typical_vram_mb": 2000, "complexity": "low"},
    )


def register_default_workflows(workflow_registry) -> None:
    """
    Register all default workflows in the registry.

    This function registers:
    - search_qa: Search-based question answering
    - search_qa_fast: Fast search-based QA (reduced summarization)
    - search_qa_strict_citations: Search-based QA with citation validation
    - summarize_url: Single URL summarization
    - rag_qa: RAG-based question answering
    - lesson_pack: Lesson content generation
    - chat_generate: Conversational generation

    Args:
        workflow_registry: WorkflowRegistry instance to register workflows in

    Note:
        Some workflows reference steps that may not yet be implemented.
        The registry will validate that all steps exist, so ensure the
        step registry is properly populated before calling this function.
    """
    workflows = [
        create_search_qa_workflow(),
        create_search_qa_fast_workflow(),
        create_search_qa_strict_citations_workflow(),
        create_summarize_url_workflow(),
        create_rag_qa_workflow(),
        create_lesson_pack_workflow(),
        create_chat_generate_workflow(),
    ]

    for workflow in workflows:
        try:
            workflow_registry.register(workflow)
        except (ValueError, KeyError) as e:
            # Log but don't fail - some workflows may reference unimplemented steps
            from mm_orch.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Could not register workflow '{workflow.name}': {e}. "
                f"This is expected if the workflow's steps are not yet implemented."
            )


def create_summarize_url_workflow() -> WorkflowDefinition:
    """
    Create the summarize_url workflow definition.

    Flow: fetch_single_url → summarize_to_answer

    Returns:
        WorkflowDefinition for URL summarization
    """
    return WorkflowDefinition(
        name="summarize_url",
        description="Fetch and summarize content from a single URL",
        graph={
            "start": GraphNode(step_name="fetch_single_url", next_nodes=["summarize"]),
            "summarize": GraphNode(step_name="summarize_to_answer", next_nodes=["end"]),
        },
        required_capabilities=["fetch", "summarize"],
        tags=["summarize", "url", "fast"],
        metadata={"typical_latency_ms": 3000, "typical_vram_mb": 1500, "complexity": "low"},
    )


def create_search_qa_fast_workflow() -> WorkflowDefinition:
    """
    Create the search_qa_fast workflow definition.

    Flow: web_search → fetch_top_n(n=2) → answer_generate_from_docs

    Skips summarization for faster execution.

    Returns:
        WorkflowDefinition for fast search-based question answering
    """
    return WorkflowDefinition(
        name="search_qa_fast",
        description="Fast search-based question answering with reduced summarization",
        graph={
            "start": GraphNode(step_name="web_search", next_nodes=["fetch"]),
            "fetch": GraphNode(step_name="fetch_top_n", next_nodes=["answer"]),
            "answer": GraphNode(step_name="answer_generate_from_docs", next_nodes=["end"]),
        },
        required_capabilities=["search", "fetch", "generate"],
        tags=["qa", "search", "fast"],
        metadata={"typical_latency_ms": 3500, "typical_vram_mb": 1800, "complexity": "low"},
    )


def create_search_qa_strict_citations_workflow() -> WorkflowDefinition:
    """
    Create the search_qa_strict_citations workflow definition.

    Flow: web_search → fetch_url → summarize → answer_generate → citation_validation

    Enforces strict citation formatting with validation.

    Returns:
        WorkflowDefinition for search-based QA with strict citation validation
    """
    return WorkflowDefinition(
        name="search_qa_strict_citations",
        description="Search-based question answering with enforced citation formatting",
        graph={
            "start": GraphNode(step_name="web_search", next_nodes=["fetch"]),
            "fetch": GraphNode(step_name="fetch_url", next_nodes=["summarize"]),
            "summarize": GraphNode(step_name="summarize", next_nodes=["answer"]),
            "answer": GraphNode(step_name="answer_generate", next_nodes=["validate"]),
            "validate": GraphNode(step_name="citation_validation", next_nodes=["end"]),
        },
        required_capabilities=["search", "fetch", "summarize", "generate"],
        tags=["qa", "search", "citations", "strict"],
        metadata={"typical_latency_ms": 5500, "typical_vram_mb": 2000, "complexity": "medium"},
    )
