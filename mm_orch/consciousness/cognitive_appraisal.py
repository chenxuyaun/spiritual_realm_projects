"""
Cognitive Appraisal System for consciousness system.

This module implements the CognitiveAppraisalSystem which evaluates events
using cognitive appraisal theory to generate emotional responses.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math

from mm_orch.consciousness.motivation import MotivationSystem, Goal, GoalStatus
from mm_orch.consciousness.self_model import SelfModel


@dataclass
class AppraisalResult:
    """Result of cognitive appraisal."""
    relevance: float  # 0.0 to 1.0 - how relevant to agent
    goal_congruence: float  # -1.0 to 1.0 - helps or hinders goals
    coping_potential: float  # 0.0 to 1.0 - ability to handle
    norm_compatibility: float  # -1.0 to 1.0 - fits expectations
    
    def __post_init__(self) -> None:
        """Validate and clamp values to valid ranges."""
        if not isinstance(self.relevance, (int, float)):
            raise ValueError("relevance must be a number")
        if not isinstance(self.goal_congruence, (int, float)):
            raise ValueError("goal_congruence must be a number")
        if not isinstance(self.coping_potential, (int, float)):
            raise ValueError("coping_potential must be a number")
        if not isinstance(self.norm_compatibility, (int, float)):
            raise ValueError("norm_compatibility must be a number")
        
        self.relevance = max(0.0, min(1.0, float(self.relevance)))
        self.goal_congruence = max(-1.0, min(1.0, float(self.goal_congruence)))
        self.coping_potential = max(0.0, min(1.0, float(self.coping_potential)))
        self.norm_compatibility = max(-1.0, min(1.0, float(self.norm_compatibility)))

    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "relevance": self.relevance,
            "goal_congruence": self.goal_congruence,
            "coping_potential": self.coping_potential,
            "norm_compatibility": self.norm_compatibility,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppraisalResult":
        """Create from dictionary representation."""
        return cls(
            relevance=data.get("relevance", 0.5),
            goal_congruence=data.get("goal_congruence", 0.0),
            coping_potential=data.get("coping_potential", 0.5),
            norm_compatibility=data.get("norm_compatibility", 0.0),
        )
    
    def to_pad_delta(self) -> Dict[str, float]:
        """
        Convert appraisal to PAD state changes.
        
        Mapping logic:
        - Pleasure: Primarily driven by goal_congruence
        - Arousal: Driven by relevance and absolute goal_congruence
        - Dominance: Driven by coping_potential
        """
        # Pleasure is primarily determined by goal congruence
        # Positive goal congruence -> positive pleasure
        pleasure_delta = self.goal_congruence * self.relevance * 0.5
        
        # Arousal increases with relevance and emotional intensity
        # High relevance + strong goal congruence (positive or negative) -> high arousal
        intensity = abs(self.goal_congruence)
        arousal_delta = self.relevance * intensity * 0.4
        
        # Dominance is determined by coping potential
        # High coping -> positive dominance, low coping -> negative dominance
        dominance_delta = (self.coping_potential - 0.5) * self.relevance * 0.6
        
        return {
            "pleasure_delta": pleasure_delta,
            "arousal_delta": arousal_delta,
            "dominance_delta": dominance_delta,
        }


@dataclass
class CognitiveAppraisalConfig:
    """Configuration for the cognitive appraisal system."""
    relevance_threshold: float = 0.3  # Minimum relevance to trigger emotional response
    goal_importance_weight: float = 0.4  # Weight for goal importance in appraisal
    capability_weight: float = 0.3  # Weight for capability in coping potential
    history_weight: float = 0.3  # Weight for historical performance
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.relevance_threshold <= 1.0):
            raise ValueError("relevance_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.goal_importance_weight <= 1.0):
            raise ValueError("goal_importance_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.capability_weight <= 1.0):
            raise ValueError("capability_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.history_weight <= 1.0):
            raise ValueError("history_weight must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "relevance_threshold": self.relevance_threshold,
            "goal_importance_weight": self.goal_importance_weight,
            "capability_weight": self.capability_weight,
            "history_weight": self.history_weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveAppraisalConfig":
        """Create from dictionary representation."""
        return cls(
            relevance_threshold=data.get("relevance_threshold", 0.3),
            goal_importance_weight=data.get("goal_importance_weight", 0.4),
            capability_weight=data.get("capability_weight", 0.3),
            history_weight=data.get("history_weight", 0.3),
        )


# Appraisal pattern to emotion mapping
# Based on cognitive appraisal theory
APPRAISAL_EMOTION_MAPPING: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Emotion: {dimension: (min, max)}
    "happy": {
        "goal_congruence": (0.3, 1.0),
        "coping_potential": (0.4, 1.0),
    },
    "excited": {
        "goal_congruence": (0.4, 1.0),
        "relevance": (0.6, 1.0),
        "coping_potential": (0.5, 1.0),
    },
    "content": {
        "goal_congruence": (0.2, 0.6),
        "coping_potential": (0.5, 1.0),
    },
    "proud": {
        "goal_congruence": (0.5, 1.0),
        "coping_potential": (0.7, 1.0),
    },
    "hopeful": {
        "goal_congruence": (0.1, 0.5),
        "coping_potential": (0.4, 0.7),
    },
    "sad": {
        "goal_congruence": (-1.0, -0.3),
        "coping_potential": (0.0, 0.4),
    },
    "depressed": {
        "goal_congruence": (-1.0, -0.5),
        "coping_potential": (0.0, 0.3),
        "relevance": (0.5, 1.0),
    },
    "angry": {
        "goal_congruence": (-1.0, -0.3),
        "coping_potential": (0.6, 1.0),
        "relevance": (0.5, 1.0),
    },
    "frustrated": {
        "goal_congruence": (-0.7, -0.2),
        "coping_potential": (0.3, 0.6),
        "relevance": (0.4, 1.0),
    },
    "fearful": {
        "goal_congruence": (-1.0, -0.3),
        "coping_potential": (0.0, 0.3),
        "relevance": (0.6, 1.0),
    },
    "anxious": {
        "goal_congruence": (-0.6, -0.1),
        "coping_potential": (0.2, 0.5),
    },
    "surprised": {
        "norm_compatibility": (-1.0, -0.3),
        "relevance": (0.5, 1.0),
        "goal_congruence": (-0.3, 0.3),  # Surprised is more neutral on goal congruence
    },
    "curious": {
        "relevance": (0.4, 1.0),
        "norm_compatibility": (-0.5, 0.5),
        "goal_congruence": (-0.2, 0.5),  # Curious is more neutral/positive on goal congruence
    },
    "determined": {
        "goal_congruence": (-0.5, 0.5),
        "coping_potential": (0.6, 1.0),
        "relevance": (0.5, 1.0),
    },
    "neutral": {
        "relevance": (0.0, 0.3),
    },
}


class CognitiveAppraisalSystem:
    """
    Evaluates events using cognitive appraisal theory.
    
    The cognitive appraisal system evaluates events along multiple dimensions:
    - Relevance: How relevant is the event to the agent?
    - Goal congruence: Does the event help or hinder current goals?
    - Coping potential: Can the agent handle the event?
    - Norm compatibility: Does the event fit expectations?
    
    These appraisals are then mapped to emotional responses.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    """
    
    def __init__(
        self,
        motivation_system: Optional[MotivationSystem] = None,
        self_model: Optional[SelfModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the cognitive appraisal system.
        
        Args:
            motivation_system: Reference to the motivation system for goal information.
            self_model: Reference to the self model for capability information.
            config: Optional configuration dictionary.
        """
        if config is not None:
            self._config = CognitiveAppraisalConfig.from_dict(config)
        else:
            self._config = CognitiveAppraisalConfig()
        
        self._motivation_system = motivation_system
        self._self_model = self_model
        
        # Statistics
        self._total_appraisals: int = 0
        self._appraisal_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 100
        self._initialized_at: float = time.time()
        
        # Event type relevance mapping
        self._event_relevance: Dict[str, float] = {
            "task_complete": 0.8,
            "task_error": 0.9,
            "user_feedback": 0.9,
            "user_message": 0.7,
            "system_event": 0.4,
            "workflow_start": 0.6,
            "workflow_end": 0.7,
            "model_loaded": 0.3,
            "model_unloaded": 0.2,
            "unknown": 0.5,
        }
        
        # Norm expectations for different event types
        self._norm_expectations: Dict[str, Dict[str, Any]] = {
            "task_complete": {"success": True, "score_min": 0.7},
            "task_error": {"expected": False},
            "user_feedback": {"positive": True},
        }

    
    def appraise_event(
        self,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AppraisalResult:
        """
        Appraise an event along multiple dimensions.
        
        Evaluates relevance, goal congruence, coping potential, and norm compatibility.
        
        Args:
            event: The event to appraise. Should contain at least 'type' key.
            context: Optional additional context for appraisal.
        
        Returns:
            AppraisalResult with scores for each dimension.
        
        Requirements: 7.1, 7.2
        """
        context = context or {}
        
        # Calculate each appraisal dimension
        relevance = self.calculate_relevance(event, context)
        goal_congruence = self.calculate_goal_congruence(event, context)
        coping_potential = self.calculate_coping_potential(event, context)
        norm_compatibility = self.calculate_norm_compatibility(event, context)
        
        result = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=norm_compatibility,
        )
        
        # Record appraisal
        self._record_appraisal(event, result, context)
        self._total_appraisals += 1
        
        return result
    
    def calculate_relevance(
        self,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate how relevant an event is to the agent.
        
        Args:
            event: The event to evaluate.
            context: Optional additional context.
        
        Returns:
            Relevance score from 0.0 to 1.0.
        
        Requirements: 7.1
        """
        context = context or {}
        
        # Base relevance from event type
        event_type = event.get("type", "unknown")
        base_relevance = self._event_relevance.get(event_type, 0.5)
        
        # Adjust based on event properties
        adjustments = 0.0
        
        # User-related events are more relevant
        if event.get("user_initiated", False):
            adjustments += 0.1
        
        # Events with high importance are more relevant
        importance = event.get("importance", 0.5)
        adjustments += (importance - 0.5) * 0.2
        
        # Events related to active goals are more relevant
        if self._motivation_system:
            active_goals = self._motivation_system.get_active_goals()
            if self._is_event_goal_related(event, active_goals):
                adjustments += 0.15
        
        # Context-based adjustments
        if context.get("urgent", False):
            adjustments += 0.2
        
        return max(0.0, min(1.0, base_relevance + adjustments))

    
    def calculate_goal_congruence(
        self,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate whether event helps or hinders current goals.
        
        Args:
            event: The event to evaluate.
            context: Optional additional context.
        
        Returns:
            Goal congruence score from -1.0 (hinders) to 1.0 (helps).
        
        Requirements: 7.2
        """
        context = context or {}
        event_type = event.get("type", "unknown")
        
        # Direct success/failure indicators
        if event.get("success") is True:
            base_congruence = 0.6
        elif event.get("success") is False:
            base_congruence = -0.6
        else:
            base_congruence = 0.0
        
        # Event type specific adjustments
        if event_type == "task_complete":
            # Use score if available, otherwise use success flag
            if "score" in event:
                score = event.get("score", 0.5)
                base_congruence = (score - 0.5) * 2  # Map 0-1 to -1 to 1
            # If no score but success flag is set, keep the base_congruence from above
        elif event_type == "task_error":
            base_congruence = -0.7
        elif event_type == "user_feedback":
            sentiment = event.get("sentiment", 0.0)  # -1 to 1
            base_congruence = sentiment * 0.8
        
        # Adjust based on goal importance
        if self._motivation_system:
            active_goals = self._motivation_system.get_active_goals()
            related_goals = [g for g in active_goals if self._is_event_goal_related(event, [g])]
            
            if related_goals:
                # Weight by goal priority
                max_priority = max(g.priority for g in related_goals)
                base_congruence *= (0.5 + max_priority * 0.5)
        
        return max(-1.0, min(1.0, base_congruence))
    
    def calculate_coping_potential(
        self,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate agent's ability to handle the event.
        
        Args:
            event: The event to evaluate.
            context: Optional additional context.
        
        Returns:
            Coping potential score from 0.0 (cannot cope) to 1.0 (can easily cope).
        
        Requirements: 7.1
        """
        context = context or {}
        event_type = event.get("type", "unknown")
        
        # Base coping potential
        base_coping = 0.5
        
        # Check capability-based coping
        if self._self_model:
            task_type = event.get("task_type") or event.get("workflow_type")
            if task_type:
                capability = self._self_model.get_capability(task_type)
                if capability:
                    # Higher performance score = higher coping potential
                    base_coping = capability.performance_score * 0.7 + 0.3
                    if not capability.enabled:
                        base_coping *= 0.5
        
        # Adjust based on event difficulty
        difficulty = event.get("difficulty", 0.5)
        base_coping -= (difficulty - 0.5) * 0.3
        
        # Historical performance adjustment
        if self._self_model:
            avg_performance = self._self_model.get_average_performance()
            base_coping = base_coping * 0.7 + avg_performance * 0.3
        
        # Context-based adjustments
        if context.get("resources_available", True):
            base_coping += 0.1
        else:
            base_coping -= 0.2
        
        return max(0.0, min(1.0, base_coping))

    
    def calculate_norm_compatibility(
        self,
        event: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate whether event fits expectations.
        
        Args:
            event: The event to evaluate.
            context: Optional additional context.
        
        Returns:
            Norm compatibility score from -1.0 (unexpected) to 1.0 (expected).
        
        Requirements: 7.1
        """
        context = context or {}
        event_type = event.get("type", "unknown")
        
        # Check against norm expectations
        expectations = self._norm_expectations.get(event_type, {})
        
        if not expectations:
            # No specific expectations, neutral
            return 0.0
        
        compatibility_scores = []
        
        # Check success expectation
        if "success" in expectations:
            expected_success = expectations["success"]
            actual_success = event.get("success")
            if actual_success is not None:
                if actual_success == expected_success:
                    compatibility_scores.append(0.5)
                else:
                    compatibility_scores.append(-0.5)
        
        # Check score expectation
        if "score_min" in expectations:
            expected_min = expectations["score_min"]
            actual_score = event.get("score", 0.5)
            if actual_score >= expected_min:
                compatibility_scores.append(0.3)
            else:
                compatibility_scores.append(-0.3)
        
        # Check expected flag
        if "expected" in expectations:
            if expectations["expected"] is False:
                # This type of event is unexpected
                compatibility_scores.append(-0.6)
        
        # Check positive expectation
        if "positive" in expectations:
            sentiment = event.get("sentiment", 0.0)
            if sentiment > 0:
                compatibility_scores.append(0.4)
            elif sentiment < 0:
                compatibility_scores.append(-0.4)
        
        if compatibility_scores:
            return max(-1.0, min(1.0, sum(compatibility_scores) / len(compatibility_scores)))
        
        return 0.0
    
    def appraisal_to_emotion(self, appraisal: AppraisalResult) -> str:
        """
        Map appraisal pattern to discrete emotion.
        
        Args:
            appraisal: The appraisal result to map.
        
        Returns:
            The emotion label that best matches the appraisal pattern.
        
        Requirements: 7.3
        """
        best_emotion = "neutral"
        best_score = -float('inf')
        
        for emotion, criteria in APPRAISAL_EMOTION_MAPPING.items():
            score = self._calculate_emotion_match_score(appraisal, criteria)
            if score > best_score:
                best_score = score
                best_emotion = emotion
        
        return best_emotion

    
    def _calculate_emotion_match_score(
        self,
        appraisal: AppraisalResult,
        criteria: Dict[str, Tuple[float, float]],
    ) -> float:
        """
        Calculate how well an appraisal matches emotion criteria.
        
        Args:
            appraisal: The appraisal result.
            criteria: Dictionary of dimension -> (min, max) ranges.
        
        Returns:
            Match score (higher is better match).
        """
        if not criteria:
            return 0.0
        
        total_score = 0.0
        num_criteria = len(criteria)
        
        for dimension, (min_val, max_val) in criteria.items():
            value = getattr(appraisal, dimension, 0.0)
            
            if min_val <= value <= max_val:
                # Value is within range - calculate how centered it is
                range_center = (min_val + max_val) / 2
                range_size = max_val - min_val
                if range_size > 0:
                    distance_from_center = abs(value - range_center) / (range_size / 2)
                    score = 1.0 - distance_from_center * 0.5  # 0.5 to 1.0
                else:
                    score = 1.0
                total_score += score
            else:
                # Value is outside range - penalize based on distance
                if value < min_val:
                    distance = min_val - value
                else:
                    distance = value - max_val
                total_score -= distance
        
        return total_score / num_criteria
    
    def _is_event_goal_related(
        self,
        event: Dict[str, Any],
        goals: List[Goal],
    ) -> bool:
        """Check if an event is related to any of the given goals."""
        event_type = event.get("type", "")
        task_type = event.get("task_type", "")
        
        # Simple keyword matching
        event_keywords = set()
        if event_type:
            event_keywords.update(event_type.lower().split("_"))
        if task_type:
            event_keywords.update(task_type.lower().split("_"))
        
        for goal in goals:
            goal_text = f"{goal.name} {goal.description}".lower()
            for keyword in event_keywords:
                if keyword in goal_text:
                    return True
        
        return False
    
    def _record_appraisal(
        self,
        event: Dict[str, Any],
        result: AppraisalResult,
        context: Dict[str, Any],
    ) -> None:
        """Record an appraisal in history."""
        entry = {
            "timestamp": time.time(),
            "event_type": event.get("type", "unknown"),
            "appraisal": result.to_dict(),
            "emotion": self.appraisal_to_emotion(result),
            "context_keys": list(context.keys()),
        }
        self._appraisal_history.append(entry)
        
        if len(self._appraisal_history) > self._max_history_size:
            self._appraisal_history = self._appraisal_history[-self._max_history_size:]

    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the appraisal system."""
        return {
            "total_appraisals": self._total_appraisals,
            "history_size": len(self._appraisal_history),
            "uptime_seconds": time.time() - self._initialized_at,
            "has_motivation_system": self._motivation_system is not None,
            "has_self_model": self._self_model is not None,
        }
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get appraisal history."""
        if limit is not None:
            return self._appraisal_history[-limit:]
        return self._appraisal_history.copy()
    
    def clear_history(self) -> None:
        """Clear appraisal history."""
        self._appraisal_history.clear()
    
    def set_event_relevance(self, event_type: str, relevance: float) -> None:
        """Set the base relevance for an event type."""
        self._event_relevance[event_type] = max(0.0, min(1.0, relevance))
    
    def set_norm_expectation(
        self,
        event_type: str,
        expectations: Dict[str, Any],
    ) -> None:
        """Set norm expectations for an event type."""
        self._norm_expectations[event_type] = expectations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the appraisal system state to dictionary representation."""
        return {
            "config": self._config.to_dict(),
            "statistics": {
                "total_appraisals": self._total_appraisals,
                "initialized_at": self._initialized_at,
            },
            "event_relevance": self._event_relevance.copy(),
            "norm_expectations": self._norm_expectations.copy(),
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load state from a dictionary representation."""
        if "config" in state:
            self._config = CognitiveAppraisalConfig.from_dict(state["config"])
        if "statistics" in state:
            stats = state["statistics"]
            self._total_appraisals = stats.get("total_appraisals", 0)
            self._initialized_at = stats.get("initialized_at", time.time())
        if "event_relevance" in state:
            self._event_relevance.update(state["event_relevance"])
        if "norm_expectations" in state:
            self._norm_expectations.update(state["norm_expectations"])
    
    def from_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore state from a dictionary representation.
        
        Alias for load_state() for consistency with other modules.
        
        Args:
            state: Dictionary containing saved state.
        """
        self.load_state(state)


def get_appraisal_emotion_mapping() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Get the appraisal to emotion mapping dictionary."""
    return APPRAISAL_EMOTION_MAPPING.copy()
