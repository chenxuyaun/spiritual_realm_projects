"""
Emotion System module for consciousness system.

The EmotionSystem manages the system's emotional state:
- Emotion state calculation (valence, arousal)
- Emotion impact on processing strategy
- Emotion decay and regulation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import math


@dataclass
class EmotionEvent:
    """Records an emotion-triggering event."""
    event_type: str
    valence_impact: float  # -1.0 to 1.0
    arousal_impact: float  # -1.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmotionSystem:
    """
    Emotion System manages the system's emotional state.
    
    Calculates emotion state based on events and influences processing strategy.
    Implements requirements 7.3, 7.4: calculate emotion state and influence strategy.
    """
    
    # Emotion impact mappings for different event types
    EVENT_IMPACTS = {
        "task_success": {"valence": 0.2, "arousal": 0.1},
        "task_failure": {"valence": -0.3, "arousal": 0.2},
        "user_positive_feedback": {"valence": 0.3, "arousal": 0.15},
        "user_negative_feedback": {"valence": -0.25, "arousal": 0.2},
        "complex_task_start": {"valence": 0.0, "arousal": 0.3},
        "task_timeout": {"valence": -0.2, "arousal": 0.25},
        "resource_constraint": {"valence": -0.15, "arousal": 0.2},
        "learning_progress": {"valence": 0.15, "arousal": 0.05},
        "idle": {"valence": 0.0, "arousal": -0.1},
    }
    
    # Strategy modifiers based on emotion state
    STRATEGY_MODIFIERS = {
        "high_valence_high_arousal": {
            "temperature": 0.1,  # More creative
            "verbosity": 0.1,   # More detailed
            "risk_tolerance": 0.1,  # More willing to try new approaches
        },
        "high_valence_low_arousal": {
            "temperature": 0.0,
            "verbosity": 0.0,
            "risk_tolerance": 0.0,
        },
        "low_valence_high_arousal": {
            "temperature": -0.1,  # More conservative
            "verbosity": -0.1,   # More concise
            "risk_tolerance": -0.15,  # Prefer safe approaches
        },
        "low_valence_low_arousal": {
            "temperature": -0.05,
            "verbosity": -0.05,
            "risk_tolerance": -0.05,
        },
    }
    
    def __init__(self):
        """Initialize the emotion system."""
        self._valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
        self._arousal: float = 0.5  # 0.0 (calm) to 1.0 (excited)
        self._event_history: List[EmotionEvent] = []
        self._max_history_size: int = 100
        self._decay_rate: float = 0.95  # Emotion decay per time unit
        self._last_update: float = time.time()
        self._initialized_at: float = time.time()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current emotion state.
        
        Returns:
            Dictionary containing current emotion state.
        """
        self._apply_decay()
        return {
            "valence": self._valence,
            "arousal": self._arousal,
            "emotion_label": self._get_emotion_label(),
            "event_count": len(self._event_history),
            "uptime": time.time() - self._initialized_at,
        }
    
    def get_emotion_values(self) -> Tuple[float, float]:
        """
        Get the current valence and arousal values.
        
        Returns:
            Tuple of (valence, arousal).
        """
        self._apply_decay()
        return (self._valence, self._arousal)
    
    def _get_emotion_label(self) -> str:
        """Get a human-readable emotion label based on current state."""
        if self._valence > 0.3:
            if self._arousal > 0.6:
                return "excited"
            elif self._arousal > 0.3:
                return "happy"
            else:
                return "content"
        elif self._valence < -0.3:
            if self._arousal > 0.6:
                return "stressed"
            elif self._arousal > 0.3:
                return "frustrated"
            else:
                return "sad"
        else:
            if self._arousal > 0.6:
                return "alert"
            elif self._arousal > 0.3:
                return "neutral"
            else:
                return "calm"
    
    def _apply_decay(self) -> None:
        """Apply emotion decay based on time elapsed."""
        now = time.time()
        elapsed = now - self._last_update
        
        if elapsed > 0:
            # Decay towards neutral (valence=0, arousal=0.5)
            decay_factor = math.pow(self._decay_rate, elapsed / 60.0)  # Decay per minute
            self._valence *= decay_factor
            self._arousal = 0.5 + (self._arousal - 0.5) * decay_factor
            self._last_update = now
    
    def process_event(self, event_type: str, source: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Process an emotion-triggering event.
        
        Args:
            event_type: Type of the event.
            source: Optional source of the event.
            metadata: Optional event metadata.
            
        Returns:
            Dictionary with the new emotion state.
        """
        self._apply_decay()
        
        # Get impact values
        impacts = self.EVENT_IMPACTS.get(event_type, {"valence": 0.0, "arousal": 0.0})
        valence_impact = impacts["valence"]
        arousal_impact = impacts["arousal"]
        
        # Apply custom impacts from metadata
        if metadata:
            valence_impact = metadata.get("valence_impact", valence_impact)
            arousal_impact = metadata.get("arousal_impact", arousal_impact)
        
        # Record event
        event = EmotionEvent(
            event_type=event_type,
            valence_impact=valence_impact,
            arousal_impact=arousal_impact,
            source=source,
            metadata=metadata or {},
        )
        self._add_event(event)
        
        # Update emotion state
        self._update_emotion(valence_impact, arousal_impact)
        
        return {
            "valence": self._valence,
            "arousal": self._arousal,
            "emotion_label": self._get_emotion_label(),
        }
    
    def _update_emotion(self, valence_delta: float, arousal_delta: float) -> None:
        """Update emotion values with bounds checking."""
        self._valence = max(-1.0, min(1.0, self._valence + valence_delta))
        self._arousal = max(0.0, min(1.0, self._arousal + arousal_delta))
        self._last_update = time.time()
    
    def _add_event(self, event: EmotionEvent) -> None:
        """Add an event to history, maintaining max size."""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def set_emotion(self, valence: float, arousal: float) -> None:
        """
        Directly set emotion values.
        
        Args:
            valence: New valence value (-1.0 to 1.0).
            arousal: New arousal value (0.0 to 1.0).
        """
        self._valence = max(-1.0, min(1.0, valence))
        self._arousal = max(0.0, min(1.0, arousal))
        self._last_update = time.time()
    
    def get_strategy_modifiers(self) -> Dict[str, float]:
        """
        Get strategy modifiers based on current emotion state.
        
        Returns:
            Dictionary of strategy parameter modifiers.
        """
        self._apply_decay()
        
        # Determine emotion quadrant
        high_valence = self._valence > 0
        high_arousal = self._arousal > 0.5
        
        if high_valence and high_arousal:
            key = "high_valence_high_arousal"
        elif high_valence and not high_arousal:
            key = "high_valence_low_arousal"
        elif not high_valence and high_arousal:
            key = "low_valence_high_arousal"
        else:
            key = "low_valence_low_arousal"
        
        # Scale modifiers by emotion intensity
        base_modifiers = self.STRATEGY_MODIFIERS[key]
        intensity = (abs(self._valence) + abs(self._arousal - 0.5)) / 2
        
        return {
            k: v * intensity for k, v in base_modifiers.items()
        }
    
    def get_response_style(self) -> Dict[str, Any]:
        """
        Get response style parameters based on emotion state.
        
        Returns:
            Dictionary of response style parameters.
        """
        self._apply_decay()
        modifiers = self.get_strategy_modifiers()
        
        # Base parameters
        base_temperature = 0.7
        base_verbosity = 0.5
        
        return {
            "temperature": max(0.1, min(1.5, base_temperature + modifiers.get("temperature", 0))),
            "verbosity": max(0.0, min(1.0, base_verbosity + modifiers.get("verbosity", 0))),
            "tone": self._get_tone(),
            "risk_tolerance": max(0.0, min(1.0, 0.5 + modifiers.get("risk_tolerance", 0))),
        }
    
    def _get_tone(self) -> str:
        """Get the appropriate tone based on emotion state."""
        if self._valence > 0.3:
            return "enthusiastic" if self._arousal > 0.6 else "friendly"
        elif self._valence < -0.3:
            return "careful" if self._arousal > 0.6 else "empathetic"
        else:
            return "professional"
    
    def on_task_result(self, success: bool, score: float, 
                       user_feedback: Optional[str] = None) -> Dict[str, float]:
        """
        Process task result and update emotion state.
        
        Args:
            success: Whether the task was successful.
            score: Performance score (0.0 to 1.0).
            user_feedback: Optional user feedback ('positive', 'negative', None).
            
        Returns:
            Dictionary with the new emotion state.
        """
        # Process task outcome
        if success:
            event_type = "task_success"
            # Boost impact based on score
            metadata = {
                "valence_impact": 0.1 + 0.2 * score,
                "arousal_impact": 0.05 + 0.1 * score,
            }
        else:
            event_type = "task_failure"
            metadata = {
                "valence_impact": -0.2 - 0.1 * (1 - score),
                "arousal_impact": 0.15,
            }
        
        result = self.process_event(event_type, source="task_result", metadata=metadata)
        
        # Process user feedback if provided
        if user_feedback == "positive":
            result = self.process_event("user_positive_feedback", source="user")
        elif user_feedback == "negative":
            result = self.process_event("user_negative_feedback", source="user")
        
        return result
    
    def get_event_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get emotion event history.
        
        Args:
            limit: Maximum number of events to return.
            
        Returns:
            List of event dictionaries.
        """
        events = self._event_history[-limit:] if limit else self._event_history
        return [
            {
                "event_type": e.event_type,
                "valence_impact": e.valence_impact,
                "arousal_impact": e.arousal_impact,
                "timestamp": e.timestamp,
                "source": e.source,
            }
            for e in events
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the emotion system to a dictionary.
        
        Returns:
            Dictionary representation of the emotion system.
        """
        return {
            "valence": self._valence,
            "arousal": self._arousal,
            "event_history": [
                {
                    "event_type": e.event_type,
                    "valence_impact": e.valence_impact,
                    "arousal_impact": e.arousal_impact,
                    "timestamp": e.timestamp,
                    "source": e.source,
                    "metadata": e.metadata,
                }
                for e in self._event_history
            ],
            "last_update": self._last_update,
            "initialized_at": self._initialized_at,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the emotion system from a dictionary.
        
        Args:
            data: Dictionary representation of the emotion system.
        """
        if "valence" in data:
            self._valence = data["valence"]
        if "arousal" in data:
            self._arousal = data["arousal"]
        
        if "event_history" in data:
            self._event_history = [
                EmotionEvent(
                    event_type=e["event_type"],
                    valence_impact=e["valence_impact"],
                    arousal_impact=e["arousal_impact"],
                    timestamp=e.get("timestamp", time.time()),
                    source=e.get("source"),
                    metadata=e.get("metadata", {}),
                )
                for e in data["event_history"]
            ]
        
        if "last_update" in data:
            self._last_update = data["last_update"]
        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]
