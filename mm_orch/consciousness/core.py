"""
Consciousness Core module - the central integration point for all consciousness modules.

The ConsciousnessCore integrates and coordinates:
- SelfModel: Self-awareness and capability tracking
- WorldModel: Environment and knowledge representation
- Metacognition: Task monitoring and strategy suggestion
- MotivationSystem: Goal management and drives
- EmotionSystem: Emotional state and response style
- DevelopmentSystem: Growth stages and feature management
"""

from typing import Any, Dict, Optional
import time
import json
import os

from mm_orch.schemas import (
    SystemEvent, Task, StrategySuggestion, Evaluation, ConsciousnessState
)
from mm_orch.consciousness.self_model import SelfModel
from mm_orch.consciousness.world_model import WorldModel
from mm_orch.consciousness.metacognition import Metacognition
from mm_orch.consciousness.motivation import MotivationSystem
from mm_orch.consciousness.emotion import EmotionSystem
from mm_orch.consciousness.development import DevelopmentSystem


class ConsciousnessCore:
    """
    Consciousness Core integrates all consciousness modules.
    
    Provides unified interface for consciousness operations and coordinates
    information flow between modules.
    Implements requirements 6.1, 6.2: initialize modules and coordinate interactions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the consciousness core with all sub-modules.
        
        Args:
            config: Optional configuration dictionary.
        """
        config = config or {}
        
        # Initialize all sub-modules
        self.self_model = SelfModel()
        self.world_model = WorldModel()
        self.metacognition = Metacognition()
        self.motivation = MotivationSystem()
        self.emotion = EmotionSystem()
        self.development = DevelopmentSystem(
            initial_stage=config.get("development_stage", "adult")
        )
        
        self._initialized_at: float = time.time()
        self._state_path: Optional[str] = config.get("state_path")
        self._auto_save_interval: float = config.get("auto_save_interval", 300.0)  # 5 minutes
        self._last_save_time: float = time.time()
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the consciousness system status.
        
        Returns:
            Dictionary containing status summary of all modules.
        """
        return {
            "self_model": self.self_model.get_state(),
            "world_model": self.world_model.get_state(),
            "metacognition": self.metacognition.get_state(),
            "motivation": self.motivation.get_state(),
            "emotion": self.emotion.get_state(),
            "development": self.development.get_state(),
            "uptime": time.time() - self._initialized_at,
        }
    
    def update_state(self, event: SystemEvent) -> None:
        """
        Update consciousness state based on a system event.
        
        Args:
            event: The SystemEvent to process.
        """
        event_type = event.event_type
        data = event.data
        
        # Route event to appropriate modules
        if event_type == "task_start":
            self._handle_task_start(data)
        elif event_type == "task_complete":
            self._handle_task_complete(data)
        elif event_type == "task_error":
            self._handle_task_error(data)
        elif event_type == "user_interaction":
            self._handle_user_interaction(data)
        elif event_type == "resource_update":
            self._handle_resource_update(data)
        elif event_type == "knowledge_update":
            self._handle_knowledge_update(data)
        
        # Check for auto-save
        self._check_auto_save()
    
    def _handle_task_start(self, data: Dict[str, Any]) -> None:
        """Handle task start event."""
        task = data.get("task")
        if task:
            # Update self model status
            self.self_model.update_state({
                "status": "processing",
                "current_task": task.task_id if hasattr(task, 'task_id') else str(task),
            })
            
            # Start metacognition monitoring
            if isinstance(task, Task):
                self.metacognition.start_monitoring(task)
            
            # Update emotion (task start can increase arousal)
            self.emotion.process_event("complex_task_start", source="task_handler")
    
    def _handle_task_complete(self, data: Dict[str, Any]) -> None:
        """Handle task completion event."""
        task_id = data.get("task_id")
        success = data.get("success", True)
        score = data.get("score", 1.0 if success else 0.0)
        task_type = data.get("task_type", "unknown")
        
        # Update self model
        self.self_model.update_state({
            "status": "idle",
            "current_task": None,
        })
        
        # Record capability usage
        self.self_model.record_capability_usage(task_type, success, score)
        
        # Complete metacognition monitoring
        if task_id:
            self.metacognition.complete_task(task_id, success, {"score": score})
        
        # Update motivation
        self.motivation.on_task_completed(task_type, success, score)
        
        # Update emotion
        self.emotion.on_task_result(success, score)
        
        # Record for development
        self.development.record_task_result(success, score, task_type)
    
    def _handle_task_error(self, data: Dict[str, Any]) -> None:
        """Handle task error event."""
        task_id = data.get("task_id")
        error_type = data.get("error_type", "unknown")
        
        # Update self model
        self.self_model.update_state({
            "status": "error_recovery",
            "last_error": error_type,
        })
        
        # Complete metacognition with failure
        if task_id:
            self.metacognition.complete_task(task_id, False, {"error": error_type})
        
        # Update emotion (errors cause negative valence)
        self.emotion.process_event("task_failure", source="error_handler")
    
    def _handle_user_interaction(self, data: Dict[str, Any]) -> None:
        """Handle user interaction event."""
        user_id = data.get("user_id", "default")
        interaction_type = data.get("type", "message")
        feedback = data.get("feedback")
        
        # Update world model with user interaction
        self.world_model.record_user_interaction(user_id, {
            "type": interaction_type,
            "data": data,
        })
        
        # Process feedback for emotion
        if feedback == "positive":
            self.emotion.process_event("user_positive_feedback", source="user")
        elif feedback == "negative":
            self.emotion.process_event("user_negative_feedback", source="user")
    
    def _handle_resource_update(self, data: Dict[str, Any]) -> None:
        """Handle resource update event."""
        resource_type = data.get("type")
        status = data.get("status")
        
        # Update world model with resource info
        self.world_model.update_state({
            "environment": {
                "resources": {resource_type: status},
            },
        })
        
        # Update self model health if resource constrained
        if status == "constrained":
            self.self_model.update_state({"health": 0.8})
            self.emotion.process_event("resource_constraint", source="resource_monitor")
    
    def _handle_knowledge_update(self, data: Dict[str, Any]) -> None:
        """Handle knowledge update event."""
        domain = data.get("domain", "facts")
        key = data.get("key")
        value = data.get("value")
        
        if key and value is not None:
            self.world_model.update_knowledge(domain, key, value)
            self.emotion.process_event("learning_progress", source="knowledge_system")
    
    def get_strategy_suggestion(self, task: Task) -> StrategySuggestion:
        """
        Get a strategy suggestion for a task.
        
        Combines metacognition analysis with emotion-based modifiers.
        
        Args:
            task: The Task to get suggestion for.
            
        Returns:
            A StrategySuggestion with recommended approach.
        """
        # Check if feature is enabled for current development stage
        task_type = task.task_type
        if not self.development.is_feature_enabled(task_type):
            access_info = self.development.check_feature_access(task_type)
            return StrategySuggestion(
                strategy="feature_restricted",
                confidence=1.0,
                reasoning=access_info.get("message", f"Feature '{task_type}' not available"),
                parameters={"access_info": access_info},
            )
        
        # Get base suggestion from metacognition
        suggestion = self.metacognition.get_strategy_suggestion(task)
        
        # Apply emotion-based modifiers
        emotion_modifiers = self.emotion.get_strategy_modifiers()
        response_style = self.emotion.get_response_style()
        
        # Enhance suggestion parameters
        suggestion.parameters["emotion_modifiers"] = emotion_modifiers
        suggestion.parameters["response_style"] = response_style
        suggestion.parameters["max_complexity"] = self.development.get_max_complexity()
        
        return suggestion
    
    def evaluate_result(self, result: Any, expected: Any = None, 
                       task: Optional[Task] = None) -> Evaluation:
        """
        Evaluate a task result and update consciousness state.
        
        Args:
            result: The actual result.
            expected: Optional expected result for comparison.
            task: Optional Task object for context.
            
        Returns:
            An Evaluation object with assessment.
        """
        # Determine success and score
        if expected is not None:
            success = result == expected
            score = 1.0 if success else 0.5
        else:
            # Assume success if no expected value
            success = result is not None
            score = 0.8 if success else 0.2
        
        # Generate feedback
        if success:
            feedback = "Task completed successfully"
        else:
            feedback = "Task completed with issues"
        
        # Calculate emotion impact
        emotion_state = self.emotion.get_state()
        emotion_impact = {
            "valence_delta": 0.1 if success else -0.1,
            "arousal_delta": -0.05,  # Completion reduces arousal
        }
        
        # Calculate motivation impact
        motivation_impact = {}
        if task:
            motivation_impact = self.motivation.on_task_completed(
                task.task_type, success, score
            )
        
        # Create evaluation
        evaluation = Evaluation(
            success=success,
            score=score,
            feedback=feedback,
            emotion_impact=emotion_impact,
            motivation_impact=motivation_impact,
        )
        
        # Update emotion based on evaluation
        self.emotion.on_task_result(success, score)
        
        return evaluation
    
    def get_consciousness_state(self) -> ConsciousnessState:
        """
        Get the complete consciousness state.
        
        Returns:
            A ConsciousnessState object with all module states.
        """
        valence, arousal = self.emotion.get_emotion_values()
        
        return ConsciousnessState(
            self_state=self.self_model.get_state(),
            world_state=self.world_model.get_state(),
            emotion_state={"valence": valence, "arousal": arousal},
            motivation_state=self.motivation.get_state(),
            development_stage=self.development.get_current_stage().value,
            metacognition_metrics=self.metacognition.get_state(),
        )
    
    def _check_auto_save(self) -> None:
        """Check if auto-save should be triggered."""
        if self._state_path and (time.time() - self._last_save_time) > self._auto_save_interval:
            self.save_state(self._state_path)
    
    def save_state(self, path: Optional[str] = None, force: bool = False) -> bool:
        """
        Save consciousness state to disk.
        
        Args:
            path: Path to save state to.
            force: If True, save even if recently saved.
            
        Returns:
            True if save was successful.
        """
        path = path or self._state_path
        if not path:
            return False
        
        if not force and (time.time() - self._last_save_time) < 60:
            return False  # Don't save too frequently
        
        try:
            state = self.to_dict()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self._last_save_time = time.time()
            return True
        except Exception:
            return False
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load consciousness state from disk.
        
        Args:
            path: Path to load state from.
            
        Returns:
            True if load was successful.
        """
        path = path or self._state_path
        if not path or not os.path.exists(path):
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.from_dict(state)
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the consciousness core to a dictionary.
        
        Returns:
            Dictionary representation of the consciousness core.
        """
        return {
            "self_model": self.self_model.to_dict(),
            "world_model": self.world_model.to_dict(),
            "metacognition": self.metacognition.to_dict(),
            "motivation": self.motivation.to_dict(),
            "emotion": self.emotion.to_dict(),
            "development": self.development.to_dict(),
            "initialized_at": self._initialized_at,
            "saved_at": time.time(),
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the consciousness core from a dictionary.
        
        Args:
            data: Dictionary representation of the consciousness core.
        """
        if "self_model" in data:
            self.self_model.from_dict(data["self_model"])
        if "world_model" in data:
            self.world_model.from_dict(data["world_model"])
        if "metacognition" in data:
            self.metacognition.from_dict(data["metacognition"])
        if "motivation" in data:
            self.motivation.from_dict(data["motivation"])
        if "emotion" in data:
            self.emotion.from_dict(data["emotion"])
        if "development" in data:
            self.development.from_dict(data["development"])
        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]


# Singleton instance
_consciousness_instance: Optional[ConsciousnessCore] = None


def get_consciousness(config: Optional[Dict[str, Any]] = None) -> ConsciousnessCore:
    """
    Get the singleton consciousness instance.
    
    Args:
        config: Optional configuration for first initialization.
        
    Returns:
        The ConsciousnessCore singleton instance.
    """
    global _consciousness_instance
    if _consciousness_instance is None:
        _consciousness_instance = ConsciousnessCore(config)
    return _consciousness_instance


def save_consciousness(path: Optional[str] = None, force: bool = False) -> bool:
    """
    Save the consciousness state.
    
    Args:
        path: Path to save to.
        force: If True, save even if recently saved.
        
    Returns:
        True if save was successful.
    """
    if _consciousness_instance:
        return _consciousness_instance.save_state(path, force)
    return False


def load_consciousness(path: str, config: Optional[Dict[str, Any]] = None) -> ConsciousnessCore:
    """
    Load consciousness state from disk.
    
    Args:
        path: Path to load from.
        config: Optional configuration.
        
    Returns:
        The loaded ConsciousnessCore instance.
    """
    global _consciousness_instance
    _consciousness_instance = ConsciousnessCore(config)
    _consciousness_instance.load_state(path)
    return _consciousness_instance
