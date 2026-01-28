"""
PAD Emotion Model for consciousness system.

This module implements the PADEmotionModel which represents emotional states
using the three-dimensional PAD (Pleasure-Arousal-Dominance) model.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import math


@dataclass
class PADState:
    """PAD emotional state vector."""

    pleasure: float
    arousal: float
    dominance: float

    def __post_init__(self) -> None:
        if not isinstance(self.pleasure, (int, float)):
            raise ValueError("pleasure must be a number")
        if not isinstance(self.arousal, (int, float)):
            raise ValueError("arousal must be a number")
        if not isinstance(self.dominance, (int, float)):
            raise ValueError("dominance must be a number")
        self.pleasure = max(-1.0, min(1.0, float(self.pleasure)))
        self.arousal = max(0.0, min(1.0, float(self.arousal)))
        self.dominance = max(-1.0, min(1.0, float(self.dominance)))

    def to_dict(self) -> Dict[str, float]:
        return {"pleasure": self.pleasure, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PADState":
        return cls(data.get("pleasure", 0.0), data.get("arousal", 0.4), data.get("dominance", 0.0))

    def distance_to(self, other: "PADState") -> float:
        return math.sqrt(
            (self.pleasure - other.pleasure) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PADState):
            return False
        return (
            abs(self.pleasure - other.pleasure) < 1e-6
            and abs(self.arousal - other.arousal) < 1e-6
            and abs(self.dominance - other.dominance) < 1e-6
        )


# PAD coordinates for discrete emotions
EMOTION_PAD_MAPPING: Dict[str, PADState] = {
    "happy": PADState(0.7, 0.6, 0.5),
    "excited": PADState(0.6, 0.8, 0.6),
    "content": PADState(0.6, 0.3, 0.4),
    "relaxed": PADState(0.5, 0.2, 0.4),
    "proud": PADState(0.6, 0.5, 0.7),
    "hopeful": PADState(0.5, 0.5, 0.3),
    "sad": PADState(-0.6, 0.3, -0.3),
    "depressed": PADState(-0.7, 0.2, -0.5),
    "angry": PADState(-0.5, 0.8, 0.6),
    "frustrated": PADState(-0.5, 0.6, 0.3),
    "fearful": PADState(-0.6, 0.7, -0.5),
    "anxious": PADState(-0.4, 0.7, -0.3),
    "bored": PADState(-0.3, 0.1, -0.2),
    "disappointed": PADState(-0.5, 0.4, -0.2),
    "surprised": PADState(0.2, 0.8, 0.0),
    "neutral": PADState(0.0, 0.4, 0.0),
    "curious": PADState(0.3, 0.6, 0.2),
    "determined": PADState(0.2, 0.7, 0.6),
}


@dataclass
class PADEmotionConfig:
    """Configuration for the PAD emotion model."""

    decay_rate: float = 0.95
    decay_interval: float = 60.0
    baseline_pleasure: float = 0.0
    baseline_arousal: float = 0.4
    baseline_dominance: float = 0.0
    min_intensity_threshold: float = 0.1

    def __post_init__(self) -> None:
        if not (0.0 <= self.decay_rate <= 1.0):
            raise ValueError("decay_rate must be between 0.0 and 1.0")
        if self.decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        if not (-1.0 <= self.baseline_pleasure <= 1.0):
            raise ValueError("baseline_pleasure must be between -1.0 and 1.0")
        if not (0.0 <= self.baseline_arousal <= 1.0):
            raise ValueError("baseline_arousal must be between 0.0 and 1.0")
        if not (-1.0 <= self.baseline_dominance <= 1.0):
            raise ValueError("baseline_dominance must be between -1.0 and 1.0")
        if self.min_intensity_threshold < 0:
            raise ValueError("min_intensity_threshold must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "decay_interval": self.decay_interval,
            "baseline_pleasure": self.baseline_pleasure,
            "baseline_arousal": self.baseline_arousal,
            "baseline_dominance": self.baseline_dominance,
            "min_intensity_threshold": self.min_intensity_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PADEmotionConfig":
        return cls(
            decay_rate=data.get("decay_rate", 0.95),
            decay_interval=data.get("decay_interval", 60.0),
            baseline_pleasure=data.get("baseline_pleasure", 0.0),
            baseline_arousal=data.get("baseline_arousal", 0.4),
            baseline_dominance=data.get("baseline_dominance", 0.0),
            min_intensity_threshold=data.get("min_intensity_threshold", 0.1),
        )


class PADEmotionModel:
    """PAD-based emotion model with three dimensions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self._config = PADEmotionConfig.from_dict(config)
        else:
            self._config = PADEmotionConfig()

        self._state = PADState(
            self._config.baseline_pleasure,
            self._config.baseline_arousal,
            self._config.baseline_dominance,
        )
        self._baseline = PADState(
            self._config.baseline_pleasure,
            self._config.baseline_arousal,
            self._config.baseline_dominance,
        )
        self._last_decay_time: float = time.time()
        self._state_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 100
        self._total_updates: int = 0
        self._total_decays: int = 0
        self._initialized_at: float = time.time()

    def get_state(self) -> PADState:
        """Get current PAD state."""
        return PADState(self._state.pleasure, self._state.arousal, self._state.dominance)

    def set_state(self, state: PADState) -> None:
        """Directly set PAD state."""
        if not isinstance(state, PADState):
            raise TypeError("state must be a PADState instance")
        self._record_state_change("set", state)
        self._state = PADState(state.pleasure, state.arousal, state.dominance)

    def update_state(
        self,
        pleasure_delta: float = 0.0,
        arousal_delta: float = 0.0,
        dominance_delta: float = 0.0,
    ) -> PADState:
        """Update PAD state with deltas."""
        new_state = PADState(
            self._state.pleasure + pleasure_delta,
            self._state.arousal + arousal_delta,
            self._state.dominance + dominance_delta,
        )
        self._record_state_change(
            "update",
            new_state,
            {
                "pleasure_delta": pleasure_delta,
                "arousal_delta": arousal_delta,
                "dominance_delta": dominance_delta,
            },
        )
        self._state = new_state
        self._total_updates += 1
        return self.get_state()

    def apply_decay(self, decay_rate: Optional[float] = None) -> PADState:
        """Apply decay toward baseline."""
        rate = decay_rate if decay_rate is not None else self._config.decay_rate
        if not (0.0 <= rate <= 1.0):
            raise ValueError("decay_rate must be between 0.0 and 1.0")

        new_pleasure = self._baseline.pleasure + rate * (
            self._state.pleasure - self._baseline.pleasure
        )
        new_arousal = self._baseline.arousal + rate * (self._state.arousal - self._baseline.arousal)
        new_dominance = self._baseline.dominance + rate * (
            self._state.dominance - self._baseline.dominance
        )

        new_state = PADState(new_pleasure, new_arousal, new_dominance)
        self._record_state_change("decay", new_state, {"decay_rate": rate})
        self._state = new_state
        self._last_decay_time = time.time()
        self._total_decays += 1
        return self.get_state()

    def apply_time_based_decay(self) -> PADState:
        """Apply decay based on elapsed time since last decay."""
        elapsed = time.time() - self._last_decay_time
        intervals = int(elapsed / self._config.decay_interval)
        if intervals > 0:
            effective_rate = self._config.decay_rate**intervals
            return self.apply_decay(effective_rate)
        return self.get_state()

    def get_dominant_emotion(self) -> str:
        """Get the closest discrete emotion label using nearest neighbor."""
        min_distance = float("inf")
        dominant = "neutral"
        for emotion, pad_coords in EMOTION_PAD_MAPPING.items():
            distance = self._state.distance_to(pad_coords)
            if distance < min_distance:
                min_distance = distance
                dominant = emotion
        return dominant

    def get_emotion_intensity(self) -> float:
        """Get overall emotional intensity (distance from neutral)."""
        neutral = EMOTION_PAD_MAPPING["neutral"]
        return self._state.distance_to(neutral)

    def is_emotional(self) -> bool:
        """Check if current state is significantly different from neutral."""
        return self.get_emotion_intensity() >= self._config.min_intensity_threshold

    def map_to_valence_arousal(self) -> Tuple[float, float]:
        """Map to legacy valence-arousal format for compatibility."""
        return (self._state.pleasure, self._state.arousal)

    def get_emotion_probabilities(self) -> Dict[str, float]:
        """Get probability distribution over discrete emotions."""
        distances: Dict[str, float] = {}
        for emotion, pad_coords in EMOTION_PAD_MAPPING.items():
            distances[emotion] = self._state.distance_to(pad_coords)
        max_dist = max(distances.values()) if distances else 1.0
        if max_dist == 0:
            max_dist = 1.0
        similarities = {e: 1.0 - (d / max_dist) for e, d in distances.items()}
        total = sum(similarities.values())
        if total > 0:
            return {e: s / total for e, s in similarities.items()}
        return {e: 1.0 / len(EMOTION_PAD_MAPPING) for e in EMOTION_PAD_MAPPING}

    def _record_state_change(
        self,
        change_type: str,
        new_state: PADState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record state change in history."""
        entry = {
            "timestamp": time.time(),
            "type": change_type,
            "old_state": self._state.to_dict(),
            "new_state": new_state.to_dict(),
            "metadata": metadata or {},
        }
        self._state_history.append(entry)
        if len(self._state_history) > self._max_history_size:
            self._state_history = self._state_history[-self._max_history_size :]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the emotion model."""
        return {
            "current_state": self._state.to_dict(),
            "baseline": self._baseline.to_dict(),
            "dominant_emotion": self.get_dominant_emotion(),
            "emotion_intensity": self.get_emotion_intensity(),
            "is_emotional": self.is_emotional(),
            "total_updates": self._total_updates,
            "total_decays": self._total_decays,
            "history_size": len(self._state_history),
            "uptime_seconds": time.time() - self._initialized_at,
        }

    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state change history."""
        if limit is not None:
            return self._state_history[-limit:]
        return self._state_history.copy()

    def reset_to_baseline(self) -> PADState:
        """Reset state to baseline."""
        self._record_state_change("reset", self._baseline)
        self._state = PADState(
            self._baseline.pleasure,
            self._baseline.arousal,
            self._baseline.dominance,
        )
        return self.get_state()

    def set_baseline(self, baseline: PADState) -> None:
        """Set a new baseline state."""
        if not isinstance(baseline, PADState):
            raise TypeError("baseline must be a PADState instance")
        self._baseline = PADState(baseline.pleasure, baseline.arousal, baseline.dominance)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the emotion model state to dictionary representation."""
        return {
            "config": self._config.to_dict(),
            "state": self._state.to_dict(),
            "baseline": self._baseline.to_dict(),
            "last_decay_time": self._last_decay_time,
            "statistics": {
                "total_updates": self._total_updates,
                "total_decays": self._total_decays,
                "initialized_at": self._initialized_at,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary representation."""
        if "config" in state:
            self._config = PADEmotionConfig.from_dict(state["config"])
        if "state" in state:
            self._state = PADState.from_dict(state["state"])
        if "baseline" in state:
            self._baseline = PADState.from_dict(state["baseline"])
        if "last_decay_time" in state:
            self._last_decay_time = state["last_decay_time"]
        if "statistics" in state:
            stats = state["statistics"]
            self._total_updates = stats.get("total_updates", 0)
            self._total_decays = stats.get("total_decays", 0)
            self._initialized_at = stats.get("initialized_at", time.time())

    def from_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a dictionary representation.

        Alias for load_state() for consistency with other modules.

        Args:
            state: Dictionary containing saved state.
        """
        self.load_state(state)

    def clear_history(self) -> None:
        """Clear state change history."""
        self._state_history.clear()


def get_emotion_pad_mapping() -> Dict[str, PADState]:
    """Get the emotion to PAD mapping dictionary."""
    return EMOTION_PAD_MAPPING.copy()


def get_pad_for_emotion(emotion: str) -> Optional[PADState]:
    """Get PAD coordinates for a discrete emotion label."""
    return EMOTION_PAD_MAPPING.get(emotion.lower())
