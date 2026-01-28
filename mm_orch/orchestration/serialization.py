"""
State Serialization Utilities - Robust JSON serialization for State objects.

This module provides utilities for serializing and deserializing State objects
to/from JSON format. This is essential for trace logging, persistence, and
debugging workflows.

Requirements:
    - 24.1: State serialization to JSON preserving all field values
    - 24.2: State deserialization from JSON producing equivalent State
    - 24.3: Handle nested structures and optional fields correctly
    - 24.4: Descriptive error messages for serialization failures

Properties:
    - Property 31: State Serialization Round Trip
    - Property 32: Nested Structure Preservation
"""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import is_dataclass, asdict

from mm_orch.orchestration.state import State


class StateSerializationError(Exception):
    """Raised when State serialization fails."""

    pass


class StateDeserializationError(Exception):
    """Raised when State deserialization fails."""

    pass


def state_to_json(state: State, indent: Optional[int] = None) -> str:
    """
    Serialize State object to JSON string.

    This function handles all State fields including nested structures,
    optional fields, and special types. It ensures that the serialized
    JSON can be deserialized back to an equivalent State object.

    Args:
        state: State object to serialize
        indent: Optional indentation for pretty-printing (None for compact)

    Returns:
        JSON string representation of the State

    Raises:
        StateSerializationError: If serialization fails with descriptive message

    Requirements:
        - 24.1: Preserve all field values during serialization
        - 24.3: Handle nested structures correctly
        - 24.4: Descriptive error messages

    Properties:
        - Property 31: State Serialization Round Trip
        - Property 32: Nested Structure Preservation

    Example:
        >>> state: State = {"question": "What is Python?", "meta": {"mode": "default"}}
        >>> json_str = state_to_json(state)
        >>> restored = json_to_state(json_str)
        >>> assert restored == state
    """
    try:
        # Convert State to a regular dict for JSON serialization
        state_dict = dict(state)

        # Recursively convert any non-serializable objects
        serializable_dict = _make_serializable(state_dict)

        # Serialize to JSON
        return json.dumps(serializable_dict, indent=indent, ensure_ascii=False)

    except TypeError as e:
        # Identify the problematic field
        problematic_field = _find_problematic_field(state)
        raise StateSerializationError(
            f"Failed to serialize State to JSON. "
            f"Problematic field: {problematic_field}. "
            f"Error: {str(e)}"
        ) from e
    except Exception as e:
        raise StateSerializationError(
            f"Unexpected error during State serialization: {str(e)}"
        ) from e


def json_to_state(json_str: str) -> State:
    """
    Deserialize State object from JSON string.

    This function reconstructs a State object from its JSON representation,
    preserving all field values, nested structures, and optional fields.

    Args:
        json_str: JSON string to deserialize

    Returns:
        State object reconstructed from JSON

    Raises:
        StateDeserializationError: If deserialization fails with descriptive message

    Requirements:
        - 24.2: Produce equivalent State object from JSON
        - 24.3: Handle nested structures and optional fields
        - 24.4: Descriptive error messages

    Properties:
        - Property 31: State Serialization Round Trip
        - Property 32: Nested Structure Preservation

    Example:
        >>> json_str = '{"question": "What is Python?", "meta": {"mode": "default"}}'
        >>> state = json_to_state(json_str)
        >>> assert state["question"] == "What is Python?"
    """
    try:
        # Parse JSON string
        data = json.loads(json_str)

        # Validate that it's a dictionary
        if not isinstance(data, dict):
            raise StateDeserializationError(
                f"Expected JSON object (dict), got {type(data).__name__}"
            )

        # Convert to State (TypedDict is just a dict at runtime)
        # We don't need to validate field types here as TypedDict is for static analysis
        state: State = data  # type: ignore

        return state

    except json.JSONDecodeError as e:
        raise StateDeserializationError(
            f"Invalid JSON string. Error at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e
    except Exception as e:
        raise StateDeserializationError(
            f"Unexpected error during State deserialization: {str(e)}"
        ) from e


def state_to_dict(state: State) -> Dict[str, Any]:
    """
    Convert State to a plain dictionary suitable for JSON serialization.

    This is a convenience function that converts State to a dict without
    serializing to JSON string. Useful for intermediate processing.

    Args:
        state: State object to convert

    Returns:
        Dictionary representation of State

    Raises:
        StateSerializationError: If conversion fails

    Example:
        >>> state: State = {"question": "test", "meta": {}}
        >>> d = state_to_dict(state)
        >>> assert isinstance(d, dict)
    """
    try:
        state_dict = dict(state)
        return _make_serializable(state_dict)
    except Exception as e:
        raise StateSerializationError(f"Failed to convert State to dict: {str(e)}") from e


def dict_to_state(data: Dict[str, Any]) -> State:
    """
    Convert a plain dictionary to State.

    This is a convenience function that validates and converts a dict to State.

    Args:
        data: Dictionary to convert

    Returns:
        State object

    Raises:
        StateDeserializationError: If conversion fails

    Example:
        >>> d = {"question": "test", "meta": {}}
        >>> state = dict_to_state(d)
        >>> assert state["question"] == "test"
    """
    if not isinstance(data, dict):
        raise StateDeserializationError(f"Expected dict, got {type(data).__name__}")

    state: State = data  # type: ignore
    return state


def _make_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serializable form.

    This function handles:
    - Nested dictionaries and lists
    - Dataclasses (converted to dicts)
    - datetime objects (converted to ISO format strings)
    - None values (preserved)
    - Empty collections (preserved)

    Args:
        obj: Object to make serializable

    Returns:
        JSON-serializable version of the object

    Requirements:
        - 24.3: Handle nested structures correctly

    Properties:
        - Property 32: Nested Structure Preservation
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic JSON types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Handle dataclasses
    if is_dataclass(obj):
        return _make_serializable(asdict(obj))

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]

    # Handle sets
    if isinstance(obj, set):
        return [_make_serializable(item) for item in obj]

    # For other types, try to convert to string
    try:
        return str(obj)
    except Exception:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _find_problematic_field(state: State) -> str:
    """
    Identify which field in State is causing serialization issues.

    This function attempts to serialize each field individually to identify
    the problematic one, providing better error messages.

    Args:
        state: State object with serialization issues

    Returns:
        Name of the problematic field, or "unknown" if not found

    Requirements:
        - 24.4: Descriptive error messages
    """
    for key, value in state.items():
        try:
            json.dumps(_make_serializable(value))
        except (TypeError, ValueError):
            return f"{key} (type: {type(value).__name__})"

    return "unknown"


def validate_state_serializable(state: State) -> tuple[bool, Optional[str]]:
    """
    Validate that a State object can be serialized to JSON.

    This function checks if the State can be serialized without actually
    performing the serialization. Useful for validation before logging.

    Args:
        state: State object to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if State can be serialized, False otherwise
        - error_message: None if valid, error description if invalid

    Example:
        >>> state: State = {"question": "test"}
        >>> is_valid, error = validate_state_serializable(state)
        >>> assert is_valid
        >>> assert error is None
    """
    try:
        state_to_json(state)
        return True, None
    except StateSerializationError as e:
        return False, str(e)


def serialize_state_safely(state: State, fallback: str = "{}") -> str:
    """
    Serialize State to JSON with fallback on error.

    This function attempts to serialize the State, but returns a fallback
    value if serialization fails. Useful for logging where we don't want
    to crash on serialization errors.

    Args:
        state: State object to serialize
        fallback: Fallback JSON string if serialization fails

    Returns:
        JSON string representation of State, or fallback on error

    Example:
        >>> state: State = {"question": "test"}
        >>> json_str = serialize_state_safely(state)
        >>> assert "question" in json_str
    """
    try:
        return state_to_json(state)
    except StateSerializationError:
        return fallback
