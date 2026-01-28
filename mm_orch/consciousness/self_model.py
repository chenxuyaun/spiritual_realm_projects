"""
Self Model module for consciousness system.

The SelfModel maintains the system's self-awareness including:
- Capability inventory (what the system can do)
- Current state information
- Performance metrics and history
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass
class Capability:
    """Represents a system capability."""

    name: str
    description: str
    enabled: bool = True
    performance_score: float = 1.0  # 0.0 to 1.0
    usage_count: int = 0
    last_used: Optional[float] = None


class SelfModel:
    """
    Self Model maintains the system's self-awareness.

    Tracks capabilities, current state, and performance history.
    Implements requirement 6.3: maintain capability inventory and state information.
    """

    def __init__(self):
        """Initialize the self model with default state."""
        self._capabilities: Dict[str, Capability] = {}
        self._state: Dict[str, Any] = {
            "status": "idle",
            "current_task": None,
            "load": 0.0,
            "health": 1.0,
        }
        self._performance_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 100
        self._initialized_at: float = time.time()

        # Register default capabilities
        self._register_default_capabilities()

    def _register_default_capabilities(self) -> None:
        """Register the default system capabilities."""
        default_capabilities = [
            Capability("search_qa", "Search-based question answering"),
            Capability("lesson_pack", "Educational content generation"),
            Capability("chat_generate", "Multi-turn conversation"),
            Capability("rag_qa", "Knowledge base question answering"),
            Capability("self_ask_search_qa", "Complex reasoning with search"),
            Capability("summarization", "Text summarization"),
            Capability("embedding", "Text embedding generation"),
        ]
        for cap in default_capabilities:
            self._capabilities[cap.name] = cap

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current self model state.

        Returns:
            Dictionary containing current state information.
        """
        return {
            **self._state,
            "uptime": time.time() - self._initialized_at,
            "capability_count": len(self._capabilities),
            "enabled_capabilities": sum(1 for c in self._capabilities.values() if c.enabled),
        }

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the self model state.

        Args:
            updates: Dictionary of state updates to apply.
        """
        self._state.update(updates)

    def get_capabilities(self) -> Dict[str, Capability]:
        """
        Get all registered capabilities.

        Returns:
            Dictionary of capability name to Capability object.
        """
        return self._capabilities.copy()

    def get_capability(self, name: str) -> Optional[Capability]:
        """
        Get a specific capability by name.

        Args:
            name: The capability name.

        Returns:
            The Capability object or None if not found.
        """
        return self._capabilities.get(name)

    def is_capability_enabled(self, name: str) -> bool:
        """
        Check if a capability is enabled.

        Args:
            name: The capability name.

        Returns:
            True if the capability exists and is enabled.
        """
        cap = self._capabilities.get(name)
        return cap is not None and cap.enabled

    def enable_capability(self, name: str) -> bool:
        """
        Enable a capability.

        Args:
            name: The capability name.

        Returns:
            True if the capability was enabled, False if not found.
        """
        if name in self._capabilities:
            self._capabilities[name].enabled = True
            return True
        return False

    def disable_capability(self, name: str) -> bool:
        """
        Disable a capability.

        Args:
            name: The capability name.

        Returns:
            True if the capability was disabled, False if not found.
        """
        if name in self._capabilities:
            self._capabilities[name].enabled = False
            return True
        return False

    def register_capability(self, capability: Capability) -> None:
        """
        Register a new capability.

        Args:
            capability: The Capability object to register.
        """
        self._capabilities[capability.name] = capability

    def record_capability_usage(
        self, name: str, success: bool, performance_score: float = 1.0
    ) -> None:
        """
        Record usage of a capability.

        Args:
            name: The capability name.
            success: Whether the usage was successful.
            performance_score: Performance score for this usage (0.0 to 1.0).
        """
        if name in self._capabilities:
            cap = self._capabilities[name]
            cap.usage_count += 1
            cap.last_used = time.time()
            # Update performance score with exponential moving average
            alpha = 0.3
            cap.performance_score = alpha * performance_score + (1 - alpha) * cap.performance_score

            # Record in performance history
            self._add_performance_record(
                {
                    "capability": name,
                    "success": success,
                    "performance_score": performance_score,
                    "timestamp": time.time(),
                }
            )

    def _add_performance_record(self, record: Dict[str, Any]) -> None:
        """Add a performance record to history, maintaining max size."""
        self._performance_history.append(record)
        if len(self._performance_history) > self._max_history_size:
            self._performance_history = self._performance_history[-self._max_history_size :]

    def get_performance_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get performance history records.

        Args:
            limit: Maximum number of records to return (most recent).

        Returns:
            List of performance records.
        """
        if limit is None:
            return self._performance_history.copy()
        return self._performance_history[-limit:]

    def get_average_performance(self, capability: Optional[str] = None) -> float:
        """
        Get average performance score.

        Args:
            capability: Optional capability name to filter by.

        Returns:
            Average performance score (0.0 to 1.0).
        """
        records = self._performance_history
        if capability:
            records = [r for r in records if r.get("capability") == capability]

        if not records:
            return 1.0

        return sum(r.get("performance_score", 1.0) for r in records) / len(records)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the self model to a dictionary.

        Returns:
            Dictionary representation of the self model.
        """
        return {
            "state": self._state.copy(),
            "capabilities": {
                name: {
                    "name": cap.name,
                    "description": cap.description,
                    "enabled": cap.enabled,
                    "performance_score": cap.performance_score,
                    "usage_count": cap.usage_count,
                    "last_used": cap.last_used,
                }
                for name, cap in self._capabilities.items()
            },
            "performance_history": self._performance_history.copy(),
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the self model from a dictionary.

        Args:
            data: Dictionary representation of the self model.
        """
        if "state" in data:
            self._state = data["state"]

        if "capabilities" in data:
            self._capabilities = {}
            for name, cap_data in data["capabilities"].items():
                self._capabilities[name] = Capability(
                    name=cap_data["name"],
                    description=cap_data["description"],
                    enabled=cap_data.get("enabled", True),
                    performance_score=cap_data.get("performance_score", 1.0),
                    usage_count=cap_data.get("usage_count", 0),
                    last_used=cap_data.get("last_used"),
                )

        if "performance_history" in data:
            self._performance_history = data["performance_history"]

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]
