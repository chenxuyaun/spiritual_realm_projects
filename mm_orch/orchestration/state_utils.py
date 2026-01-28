"""
State Utilities - Helper functions for State creation and manipulation.

This module provides utility functions for creating and managing State objects
with proper mode settings for routing.

Requirements: 21.1
"""

from typing import Optional, Dict, Any
from mm_orch.orchestration.state import State


def create_state(
    question: str,
    mode: str = "default",
    conversation_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    **kwargs,
) -> State:
    """
    Create a State object with proper mode setting.

    This function ensures that State.meta["mode"] is properly set for routing.
    Mode affects workflow selection in Router v3:
    - mode="chat": Increases preference for chat_generate and lesson_pack
    - mode="default": Standard routing for single-shot queries

    Args:
        question: User's question or input
        mode: Execution mode ("chat" or "default")
        conversation_id: Optional conversation ID for chat mode
        turn_index: Optional turn index for chat mode
        **kwargs: Additional State fields

    Returns:
        State object with mode properly set in meta

    Requirements:
        - 21.1: State.meta["mode"] must be set for routing

    Example:
        # Single-shot query
        state = create_state("What is Python?", mode="default")

        # Chat interaction
        state = create_state(
            "Tell me more",
            mode="chat",
            conversation_id="abc123",
            turn_index=2
        )
    """
    # Create base state
    state: State = {"question": question, "meta": {"mode": mode}}

    # Add conversation fields if provided
    if conversation_id is not None:
        state["conversation_id"] = conversation_id

    if turn_index is not None:
        state["turn_index"] = turn_index

    # Add any additional fields
    for key, value in kwargs.items():
        if key not in state:
            state[key] = value  # type: ignore

    return state


def get_mode_from_state(state: State) -> str:
    """
    Extract mode from State.

    Args:
        state: State object

    Returns:
        Mode string ("chat" or "default")

    Requirements:
        - 21.1: Mode must be extracted from State.meta
    """
    meta = state.get("meta", {})
    return meta.get("mode", "default")


def set_mode_in_state(state: State, mode: str) -> State:
    """
    Set mode in State.meta.

    Args:
        state: State object
        mode: Mode to set ("chat" or "default")

    Returns:
        Updated State object

    Requirements:
        - 21.1: Mode must be set in State.meta
    """
    if "meta" not in state:
        state["meta"] = {}

    state["meta"]["mode"] = mode
    return state


def is_chat_mode(state: State) -> bool:
    """
    Check if State is in chat mode.

    Args:
        state: State object

    Returns:
        True if mode is "chat", False otherwise
    """
    return get_mode_from_state(state) == "chat"
