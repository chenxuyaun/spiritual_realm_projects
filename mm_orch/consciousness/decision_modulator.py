"""
Decision Modulator for consciousness system.

This module implements the DecisionModulator which adjusts decision-making
parameters based on the current emotional state (PAD model).

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from mm_orch.consciousness.pad_emotion import PADEmotionModel, PADState


@dataclass
class DecisionModifiers:
    """Modifiers for decision-making parameters."""

    risk_tolerance: float  # -0.5 to 0.5 adjustment
    deliberation_time: float  # multiplier (0.5 to 2.0)
    exploration_bias: float  # -0.3 to 0.3 adjustment
    confidence_threshold: float  # -0.2 to 0.2 adjustment

    def __post_init__(self) -> None:
        """Validate modifier ranges."""
        if not (-0.5 <= self.risk_tolerance <= 0.5):
            raise ValueError("risk_tolerance must be between -0.5 and 0.5")
        if not (0.5 <= self.deliberation_time <= 2.0):
            raise ValueError("deliberation_time must be between 0.5 and 2.0")
        if not (-0.3 <= self.exploration_bias <= 0.3):
            raise ValueError("exploration_bias must be between -0.3 and 0.3")
        if not (-0.2 <= self.confidence_threshold <= 0.2):
            raise ValueError("confidence_threshold must be between -0.2 and 0.2")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "risk_tolerance": self.risk_tolerance,
            "deliberation_time": self.deliberation_time,
            "exploration_bias": self.exploration_bias,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionModifiers":
        """Create from dictionary."""
        return cls(
            risk_tolerance=data.get("risk_tolerance", 0.0),
            deliberation_time=data.get("deliberation_time", 1.0),
            exploration_bias=data.get("exploration_bias", 0.0),
            confidence_threshold=data.get("confidence_threshold", 0.0),
        )


@dataclass
class DecisionLog:
    """Log entry for a decision with emotional context."""

    timestamp: float
    decision: str
    emotional_state: Dict[str, float]  # PAD values
    modifiers: Dict[str, float]
    outcome: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "decision": self.decision,
            "emotional_state": self.emotional_state,
            "modifiers": self.modifiers,
            "outcome": self.outcome,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionLog":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            decision=data["decision"],
            emotional_state=data["emotional_state"],
            modifiers=data["modifiers"],
            outcome=data.get("outcome"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DecisionModulatorConfig:
    """Configuration for the decision modulator."""

    # Risk tolerance parameters
    dominance_risk_scale: float = 0.4  # Scale factor for dominance -> risk
    base_risk_tolerance: float = 0.0

    # Deliberation time parameters
    arousal_deliberation_scale: float = 0.8  # Scale factor for arousal -> deliberation
    base_deliberation_time: float = 1.0

    # Conservative strategy threshold
    conservative_pleasure_threshold: float = -0.3

    # Confidence adjustment parameters
    pleasure_confidence_scale: float = 0.15
    dominance_confidence_scale: float = 0.1

    # Logging
    max_log_size: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.dominance_risk_scale < 0:
            raise ValueError("dominance_risk_scale must be non-negative")
        if self.arousal_deliberation_scale < 0:
            raise ValueError("arousal_deliberation_scale must be non-negative")
        if not (-1.0 <= self.conservative_pleasure_threshold <= 1.0):
            raise ValueError("conservative_pleasure_threshold must be between -1.0 and 1.0")
        if self.max_log_size < 0:
            raise ValueError("max_log_size must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dominance_risk_scale": self.dominance_risk_scale,
            "base_risk_tolerance": self.base_risk_tolerance,
            "arousal_deliberation_scale": self.arousal_deliberation_scale,
            "base_deliberation_time": self.base_deliberation_time,
            "conservative_pleasure_threshold": self.conservative_pleasure_threshold,
            "pleasure_confidence_scale": self.pleasure_confidence_scale,
            "dominance_confidence_scale": self.dominance_confidence_scale,
            "max_log_size": self.max_log_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionModulatorConfig":
        """Create from dictionary."""
        return cls(
            dominance_risk_scale=data.get("dominance_risk_scale", 0.4),
            base_risk_tolerance=data.get("base_risk_tolerance", 0.0),
            arousal_deliberation_scale=data.get("arousal_deliberation_scale", 0.8),
            base_deliberation_time=data.get("base_deliberation_time", 1.0),
            conservative_pleasure_threshold=data.get("conservative_pleasure_threshold", -0.3),
            pleasure_confidence_scale=data.get("pleasure_confidence_scale", 0.15),
            dominance_confidence_scale=data.get("dominance_confidence_scale", 0.1),
            max_log_size=data.get("max_log_size", 1000),
        )


class DecisionModulator:
    """Modulates decision-making based on emotional state."""

    def __init__(
        self,
        pad_model: PADEmotionModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the decision modulator.

        Args:
            pad_model: Reference to the PAD emotion model
            config: Optional configuration dictionary
        """
        if not isinstance(pad_model, PADEmotionModel):
            raise TypeError("pad_model must be a PADEmotionModel instance")

        self._pad_model = pad_model
        self._config = (
            DecisionModulatorConfig.from_dict(config) if config else DecisionModulatorConfig()
        )
        self._decision_log: List[DecisionLog] = []
        self._total_decisions: int = 0
        self._initialized_at: float = time.time()

    def get_modifiers(self) -> DecisionModifiers:
        """
        Get current decision modifiers based on emotional state.

        Returns:
            DecisionModifiers with adjustments based on PAD state
        """
        state = self._pad_model.get_state()

        # Calculate risk tolerance based on dominance
        risk_tolerance = self.adjust_risk_tolerance(self._config.base_risk_tolerance)

        # Calculate deliberation time based on arousal
        deliberation_time = self.adjust_deliberation(self._config.base_deliberation_time)

        # Calculate exploration bias based on pleasure and arousal
        # Low pleasure + high arousal = more exploration (seeking improvement)
        exploration_bias = max(-0.3, min(0.3, -state.pleasure * 0.15 + (state.arousal - 0.5) * 0.2))

        # Calculate confidence threshold adjustment
        # High pleasure and dominance = lower threshold (more confident)
        confidence_threshold = max(
            -0.2,
            min(
                0.2,
                -state.pleasure * self._config.pleasure_confidence_scale
                - state.dominance * self._config.dominance_confidence_scale,
            ),
        )

        return DecisionModifiers(
            risk_tolerance=risk_tolerance,
            deliberation_time=deliberation_time,
            exploration_bias=exploration_bias,
            confidence_threshold=confidence_threshold,
        )

    def adjust_risk_tolerance(self, base_tolerance: float) -> float:
        """
        Adjust risk tolerance based on dominance.

        Higher dominance increases risk tolerance, lower dominance decreases it.

        Args:
            base_tolerance: Base risk tolerance value

        Returns:
            Adjusted risk tolerance in range [-0.5, 0.5]
        """
        state = self._pad_model.get_state()
        # Dominance ranges from -1 to 1, scale to -0.5 to 0.5
        adjustment = state.dominance * self._config.dominance_risk_scale
        adjusted = base_tolerance + adjustment
        return max(-0.5, min(0.5, adjusted))

    def adjust_deliberation(self, base_time: float) -> float:
        """
        Adjust deliberation time based on arousal.

        High arousal decreases deliberation time (faster decisions),
        low arousal increases it (more deliberate).

        Args:
            base_time: Base deliberation time

        Returns:
            Deliberation time multiplier in range [0.5, 2.0]
        """
        state = self._pad_model.get_state()
        # Arousal ranges from 0 to 1
        # High arousal (>0.7) -> faster (< 1.0)
        # Low arousal (<0.3) -> slower (> 1.0)
        # Neutral arousal (0.4-0.6) -> normal (~1.0)

        # Map arousal to multiplier: arousal 0 -> 2.0, arousal 1 -> 0.5
        multiplier = 2.0 - (state.arousal * self._config.arousal_deliberation_scale * 1.5)
        multiplier = max(0.5, min(2.0, multiplier))

        return base_time * multiplier

    def adjust_strategy_confidence(
        self,
        strategy: str,
        base_confidence: float,
    ) -> float:
        """
        Adjust strategy confidence based on emotional state.

        Args:
            strategy: Strategy name
            base_confidence: Base confidence score

        Returns:
            Adjusted confidence in range [0.0, 1.0]
        """
        modifiers = self.get_modifiers()
        adjusted = base_confidence + modifiers.confidence_threshold
        return max(0.0, min(1.0, adjusted))

    def should_use_conservative_strategy(self) -> bool:
        """
        Check if emotional state suggests conservative approach.

        Returns:
            True if pleasure is below the conservative threshold
        """
        state = self._pad_model.get_state()
        return state.pleasure < self._config.conservative_pleasure_threshold

    def log_decision(
        self,
        decision: str,
        emotional_state: PADState,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log decision with emotional context for analysis.

        Args:
            decision: Description of the decision made
            emotional_state: PAD state at time of decision
            outcome: Optional outcome of the decision
            metadata: Optional additional metadata
        """
        modifiers = self.get_modifiers()

        log_entry = DecisionLog(
            timestamp=time.time(),
            decision=decision,
            emotional_state=emotional_state.to_dict(),
            modifiers=modifiers.to_dict(),
            outcome=outcome,
            metadata=metadata or {},
        )

        self._decision_log.append(log_entry)
        self._total_decisions += 1

        # Prune log if it exceeds max size
        if len(self._decision_log) > self._config.max_log_size:
            self._decision_log = self._decision_log[-self._config.max_log_size :]

    def get_decision_log(
        self,
        limit: Optional[int] = None,
        since: Optional[float] = None,
    ) -> List[DecisionLog]:
        """
        Get decision log entries.

        Args:
            limit: Maximum number of entries to return (most recent)
            since: Only return entries after this timestamp

        Returns:
            List of decision log entries
        """
        logs = self._decision_log

        if since is not None:
            logs = [log for log in logs if log.timestamp >= since]

        if limit is not None:
            logs = logs[-limit:]

        return logs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about decision modulation.

        Returns:
            Dictionary with statistics
        """
        state = self._pad_model.get_state()
        modifiers = self.get_modifiers()

        return {
            "total_decisions": self._total_decisions,
            "log_size": len(self._decision_log),
            "current_emotional_state": state.to_dict(),
            "current_modifiers": modifiers.to_dict(),
            "conservative_mode": self.should_use_conservative_strategy(),
            "uptime_seconds": time.time() - self._initialized_at,
        }

    def analyze_decision_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in decision-making.

        Returns:
            Dictionary with analysis results
        """
        if not self._decision_log:
            return {
                "total_decisions": 0,
                "average_risk_tolerance": 0.0,
                "average_deliberation_time": 1.0,
                "conservative_decisions": 0,
                "decisions_by_emotion": {},
            }

        total_risk = 0.0
        total_delib = 0.0
        conservative_count = 0
        emotion_counts: Dict[str, int] = {}

        for log in self._decision_log:
            total_risk += log.modifiers.get("risk_tolerance", 0.0)
            total_delib += log.modifiers.get("deliberation_time", 1.0)

            # Check if it was a conservative decision
            pleasure = log.emotional_state.get("pleasure", 0.0)
            if pleasure < self._config.conservative_pleasure_threshold:
                conservative_count += 1

            # Count decisions by dominant emotion (approximate from PAD)
            # This is a simplified categorization
            if pleasure > 0.3:
                emotion = "positive"
            elif pleasure < -0.3:
                emotion = "negative"
            else:
                emotion = "neutral"
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        n = len(self._decision_log)

        return {
            "total_decisions": n,
            "average_risk_tolerance": total_risk / n,
            "average_deliberation_time": total_delib / n,
            "conservative_decisions": conservative_count,
            "conservative_ratio": conservative_count / n,
            "decisions_by_emotion": emotion_counts,
        }

    def clear_log(self) -> None:
        """Clear the decision log."""
        self._decision_log.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with modulator state
        """
        return {
            "config": self._config.to_dict(),
            "statistics": {
                "total_decisions": self._total_decisions,
                "initialized_at": self._initialized_at,
            },
            "decision_log": [log.to_dict() for log in self._decision_log],
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from dictionary representation.

        Args:
            state: Dictionary with modulator state
        """
        if "config" in state:
            self._config = DecisionModulatorConfig.from_dict(state["config"])

        if "statistics" in state:
            stats = state["statistics"]
            self._total_decisions = stats.get("total_decisions", 0)
            self._initialized_at = stats.get("initialized_at", time.time())

        if "decision_log" in state:
            self._decision_log = [DecisionLog.from_dict(log) for log in state["decision_log"]]

    def from_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a dictionary representation.

        Alias for load_state() for consistency with other modules.

        Args:
            state: Dictionary containing saved state.
        """
        self.load_state(state)
