"""
Intrinsic Motivation Engine module for consciousness system.

This module implements the IntrinsicMotivationEngine which manages intrinsic
motivation and curiosity-driven exploration. It calculates curiosity rewards
based on prediction error, tracks novelty and familiarity, and provides
exploration bonuses for action selection.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import time


@dataclass
class IntrinsicMotivationConfig:
    """Configuration for the intrinsic motivation engine."""
    base_curiosity_reward: float = 0.5
    prediction_error_weight: float = 0.7
    novelty_threshold: float = 0.5
    familiarity_decay_rate: float = 0.1
    curiosity_decay_rate: float = 0.15
    exploration_weight: float = 0.3
    max_familiarity_entries: int = 10000
    information_gain_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.base_curiosity_reward <= 1.0):
            raise ValueError("base_curiosity_reward must be between 0.0 and 1.0")
        if not (0.0 <= self.prediction_error_weight <= 1.0):
            raise ValueError("prediction_error_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.novelty_threshold <= 1.0):
            raise ValueError("novelty_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.familiarity_decay_rate <= 1.0):
            raise ValueError("familiarity_decay_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.curiosity_decay_rate <= 1.0):
            raise ValueError("curiosity_decay_rate must be between 0.0 and 1.0")
        if not (0.0 <= self.exploration_weight <= 1.0):
            raise ValueError("exploration_weight must be between 0.0 and 1.0")
        if self.max_familiarity_entries < 1:
            raise ValueError("max_familiarity_entries must be at least 1")
        if self.information_gain_scale <= 0:
            raise ValueError("information_gain_scale must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "base_curiosity_reward": self.base_curiosity_reward,
            "prediction_error_weight": self.prediction_error_weight,
            "novelty_threshold": self.novelty_threshold,
            "familiarity_decay_rate": self.familiarity_decay_rate,
            "curiosity_decay_rate": self.curiosity_decay_rate,
            "exploration_weight": self.exploration_weight,
            "max_familiarity_entries": self.max_familiarity_entries,
            "information_gain_scale": self.information_gain_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntrinsicMotivationConfig":
        """Create config from dictionary representation."""
        return cls(
            base_curiosity_reward=data.get("base_curiosity_reward", 0.5),
            prediction_error_weight=data.get("prediction_error_weight", 0.7),
            novelty_threshold=data.get("novelty_threshold", 0.5),
            familiarity_decay_rate=data.get("familiarity_decay_rate", 0.1),
            curiosity_decay_rate=data.get("curiosity_decay_rate", 0.15),
            exploration_weight=data.get("exploration_weight", 0.3),
            max_familiarity_entries=data.get("max_familiarity_entries", 10000),
            information_gain_scale=data.get("information_gain_scale", 1.0),
        )


@dataclass
class FamiliarityEntry:
    """Tracks familiarity with a specific stimulus."""
    stimulus_hash: str
    encounter_count: int = 1
    familiarity_score: float = 0.0
    curiosity_level: float = 1.0
    first_encountered: float = field(default_factory=time.time)
    last_encountered: float = field(default_factory=time.time)
    cumulative_reward: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "stimulus_hash": self.stimulus_hash,
            "encounter_count": self.encounter_count,
            "familiarity_score": self.familiarity_score,
            "curiosity_level": self.curiosity_level,
            "first_encountered": self.first_encountered,
            "last_encountered": self.last_encountered,
            "cumulative_reward": self.cumulative_reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FamiliarityEntry":
        """Create entry from dictionary representation."""
        return cls(
            stimulus_hash=data["stimulus_hash"],
            encounter_count=data.get("encounter_count", 1),
            familiarity_score=data.get("familiarity_score", 0.0),
            curiosity_level=data.get("curiosity_level", 1.0),
            first_encountered=data.get("first_encountered", time.time()),
            last_encountered=data.get("last_encountered", time.time()),
            cumulative_reward=data.get("cumulative_reward", 0.0),
        )


@dataclass
class ActionExplorationInfo:
    """Tracks exploration information for an action."""
    action: str
    attempt_count: int = 0
    contexts_seen: List[str] = field(default_factory=list)
    average_information_gain: float = 0.5
    last_attempted: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action": self.action,
            "attempt_count": self.attempt_count,
            "contexts_seen": self.contexts_seen.copy(),
            "average_information_gain": self.average_information_gain,
            "last_attempted": self.last_attempted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionExplorationInfo":
        """Create info from dictionary representation."""
        return cls(
            action=data["action"],
            attempt_count=data.get("attempt_count", 0),
            contexts_seen=data.get("contexts_seen", []),
            average_information_gain=data.get("average_information_gain", 0.5),
            last_attempted=data.get("last_attempted"),
        )
