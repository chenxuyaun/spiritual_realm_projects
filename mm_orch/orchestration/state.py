"""
State TypedDict - Unified data container for workflow execution.

This module defines the State TypedDict that carries all data through workflow execution.
State is the single source of truth for workflow data flow, containing inputs, intermediate
results, and final outputs.
"""

from typing import TypedDict, Dict, List, Any, Optional


class State(TypedDict, total=False):
    """
    Workflow execution state containing all data flowing through steps.

    This TypedDict uses total=False to make all fields optional, allowing
    workflows to use only the fields they need. Steps should validate that
    required input_keys exist before accessing them.

    Core Fields:
        question: User's input question or request
        search_results: List of search result dicts from web search
        docs: Mapping of URL to fetched content
        summaries: Mapping of URL to generated summary
        final_answer: Generated answer to the question
        citations: List of source URLs used in the answer

    Lesson Pack Fields:
        lesson_topic: Topic for the lesson plan
        lesson_objectives: Learning objectives for the lesson
        lesson_outline: Structured outline of lesson sections
        board_plan: Blackboard/whiteboard content plan
        lesson_explain_structured: JSON-structured lesson content
        teaching_text: Plain text teaching content
        exercises: List of practice exercises with solutions

    RAG Fields:
        kb_sources: Knowledge base documents retrieved
        memory_context: Conversation history context

    Metadata:
        meta: Dictionary for execution metadata (mode, turn_index, router_version, etc.)

    Example:
        state: State = {
            "question": "What is Python?",
            "meta": {"mode": "default", "router_version": "v3"}
        }

        # After web search step
        state["search_results"] = [{"title": "...", "url": "...", "snippet": "..."}]

        # After fetch step
        state["docs"] = {"https://example.com": "content..."}
    """

    # Core fields
    question: str
    search_results: List[Dict[str, str]]
    docs: Dict[str, str]  # url -> content
    summaries: Dict[str, str]  # url -> summary
    final_answer: str
    citations: List[str]

    # Lesson pack fields
    lesson_topic: str
    lesson_objectives: List[str]
    lesson_outline: List[str]
    board_plan: List[str]
    lesson_explain_structured: Optional[Dict[str, Any]]  # NEW: Structured JSON lesson content
    teaching_text: str
    exercises: List[Dict[str, str]]

    # RAG fields
    kb_sources: List[Dict[str, Any]]
    memory_context: str

    # Chat fields
    conversation_id: Optional[str]
    turn_index: Optional[int]

    # Metadata
    meta: Dict[str, Any]
