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
import uuid

from mm_orch.schemas import SystemEvent, Task, StrategySuggestion, Evaluation, ConsciousnessState
from mm_orch.consciousness.self_model import SelfModel
from mm_orch.consciousness.world_model import WorldModel
from mm_orch.consciousness.metacognition import Metacognition
from mm_orch.consciousness.motivation import MotivationSystem
from mm_orch.consciousness.emotion import EmotionSystem
from mm_orch.consciousness.development import DevelopmentSystem
from mm_orch.consciousness.curriculum import CurriculumLearningSystem, CapabilityDimension
from mm_orch.consciousness.intrinsic_motivation import IntrinsicMotivationEngine
from mm_orch.consciousness.experience_replay import ExperienceReplayBuffer, Experience
from mm_orch.consciousness.episodic_memory import EpisodicMemory
from mm_orch.consciousness.semantic_memory import SemanticMemory
from mm_orch.consciousness.symbol_grounding import SymbolGroundingModule
from mm_orch.consciousness.pad_emotion import PADEmotionModel
from mm_orch.consciousness.cognitive_appraisal import CognitiveAppraisalSystem
from mm_orch.consciousness.decision_modulator import DecisionModulator


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

        # Initialize existing modules
        self.self_model = SelfModel()
        self.world_model = WorldModel()
        self.metacognition = Metacognition()
        self.motivation = MotivationSystem()
        self.emotion = EmotionSystem()
        self.development = DevelopmentSystem(initial_stage=config.get("development_stage", "adult"))

        # Initialize curriculum learning layer
        self.curriculum = CurriculumLearningSystem(
            development_system=self.development,
            config=config.get("curriculum", {})
        )
        self.intrinsic_motivation = IntrinsicMotivationEngine(
            config=config.get("intrinsic_motivation", {})
        )
        self.experience_replay = ExperienceReplayBuffer(
            max_size=config.get("experience_replay", {}).get("max_size", 10000),
            config=config.get("experience_replay", {})
        )

        # Initialize dual memory layer
        self.episodic_memory = EpisodicMemory(
            max_episodes=config.get("episodic_memory", {}).get("max_episodes", 5000),
            config=config.get("episodic_memory", {})
        )
        self.semantic_memory = SemanticMemory(
            config=config.get("semantic_memory", {})
        )
        self.symbol_grounding = SymbolGroundingModule(
            episodic_memory=self.episodic_memory,
            semantic_memory=self.semantic_memory,
            config=config.get("symbol_grounding", {})
        )

        # Initialize enhanced emotion layer
        self.pad_emotion = PADEmotionModel(
            config=config.get("pad_emotion", {})
        )
        self.cognitive_appraisal = CognitiveAppraisalSystem(
            motivation_system=self.motivation,
            self_model=self.self_model,
            config=config.get("cognitive_appraisal", {})
        )
        self.decision_modulator = DecisionModulator(
            pad_model=self.pad_emotion,
            config=config.get("decision_modulator", {})
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
            # Existing modules
            "self_model": self.self_model.get_state(),
            "world_model": self.world_model.get_state(),
            "metacognition": self.metacognition.get_state(),
            "motivation": self.motivation.get_state(),
            "emotion": self.emotion.get_state(),
            "development": self.development.get_state(),
            # Curriculum learning layer
            "curriculum": {
                "capabilities": {
                    dim.value: self.curriculum.get_capability_level(dim.value)
                    for dim in CapabilityDimension
                }
            },
            "intrinsic_motivation": {
                "active": True
            },
            "experience_replay": {
                "size": len(self.experience_replay._experiences),
                "task_distribution": self.experience_replay.get_task_type_distribution()
            },
            # Dual memory layer
            "episodic_memory": {
                "episode_count": len(self.episodic_memory._episodes)
            },
            "semantic_memory": {
                "concept_count": len(self.semantic_memory.knowledge_graph._nodes)
            },
            "symbol_grounding": {
                "grounded_symbols": len(self.symbol_grounding._groundings)
            },
            # Enhanced emotion layer
            "pad_emotion": self.pad_emotion.get_state().to_dict(),
            "decision_modulator": {
                "modifiers": self.decision_modulator.get_modifiers().__dict__
            },
            # System info
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

        # Route event to cognitive appraisal and PAD emotion system
        # Convert SystemEvent to dict format expected by cognitive appraisal
        event_dict = {
            "type": event_type,
            **data
        }
        appraisal_result = self.cognitive_appraisal.appraise_event(event_dict, context=data)
        pad_delta = appraisal_result.to_pad_delta()
        self.pad_emotion.update_state(**pad_delta)

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
            self.self_model.update_state(
                {
                    "status": "processing",
                    "current_task": task.task_id if hasattr(task, "task_id") else str(task),
                }
            )

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
        self.self_model.update_state(
            {
                "status": "idle",
                "current_task": None,
            }
        )

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

        # Update curriculum learning system
        self.curriculum.update_capabilities(task_type, success, score)

        # Calculate intrinsic motivation rewards
        predicted_outcome = data.get("predicted_outcome")
        actual_outcome = data.get("actual_outcome", {"success": success, "score": score})
        if predicted_outcome:
            curiosity_reward = self.intrinsic_motivation.calculate_curiosity_reward(
                predicted_outcome, actual_outcome
            )
            data["curiosity_reward"] = curiosity_reward

        # Store experience in replay buffer
        experience = Experience(
            experience_id=f"exp_{task_id}_{time.time()}",
            task_type=task_type,
            context=data.get("context", {}),
            action=data.get("action", "complete_task"),
            outcome={"success": success, "score": score},
            reward=score,
            priority=1.0 if not success else 0.5,  # Prioritize failures
        )
        self.experience_replay.store(experience)

        # Create episode in episodic memory (significant event)
        self.episodic_memory.create_episode(
            context=data.get("context", {"task_type": task_type}),
            events=[{"type": "task_execution", "task_type": task_type}],
            emotional_state=self.pad_emotion.get_state().to_dict(),
            importance=0.7 if success else 0.9,  # Failures are more important
            metadata={"task_id": task_id, "success": success, "score": score}
        )

    def _handle_task_error(self, data: Dict[str, Any]) -> None:
        """Handle task error event."""
        task_id = data.get("task_id")
        error_type = data.get("error_type", "unknown")
        task_type = data.get("task_type", "unknown")

        # Update self model
        self.self_model.update_state(
            {
                "status": "error_recovery",
                "last_error": error_type,
            }
        )

        # Complete metacognition with failure
        if task_id:
            self.metacognition.complete_task(task_id, False, {"error": error_type})

        # Update emotion (errors cause negative valence)
        self.emotion.process_event("task_failure", source="error_handler")

        # Create episode for error (significant event)
        self.episodic_memory.create_episode(
            context=data.get("context", {"task_type": task_type, "error_type": error_type}),
            events=[{"type": "task_error", "task_type": task_type, "error": error_type}],
            emotional_state=self.pad_emotion.get_state().to_dict(),
            importance=0.95,  # Errors are very important for learning
            metadata={"task_id": task_id, "error_type": error_type}
        )

        # Store high-priority experience for learning from errors
        experience = Experience(
            experience_id=f"exp_error_{task_id}_{time.time()}",
            task_type=task_type,
            context=data.get("context", {}),
            action="task_execution",
            outcome={"success": False, "error": error_type},
            reward=0.0,
            priority=2.0,  # High priority for errors
        )
        self.experience_replay.store(experience)

    def _handle_user_interaction(self, data: Dict[str, Any]) -> None:
        """Handle user interaction event."""
        user_id = data.get("user_id", "default")
        interaction_type = data.get("type", "message")
        feedback = data.get("feedback")

        # Update world model with user interaction
        self.world_model.record_user_interaction(
            user_id,
            {
                "type": interaction_type,
                "data": data,
            },
        )

        # Process feedback for emotion
        if feedback == "positive":
            self.emotion.process_event("user_positive_feedback", source="user")
        elif feedback == "negative":
            self.emotion.process_event("user_negative_feedback", source="user")

        # Create episode for user feedback (significant event)
        if feedback:
            self.episodic_memory.create_episode(
                context={"user_id": user_id, "interaction_type": interaction_type},
                events=[{"type": "user_feedback", "feedback": feedback}],
                emotional_state=self.pad_emotion.get_state().to_dict(),
                importance=0.8,  # User feedback is important
                metadata={"user_id": user_id, "feedback": feedback}
            )

    def _handle_resource_update(self, data: Dict[str, Any]) -> None:
        """Handle resource update event."""
        resource_type = data.get("type")
        status = data.get("status")

        # Update world model with resource info
        self.world_model.update_state(
            {
                "environment": {
                    "resources": {resource_type: status},
                },
            }
        )

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
            # Update existing world model
            self.world_model.update_knowledge(domain, key, value)
            
            # Integrate into semantic memory
            knowledge_data = {
                "concept": key,
                "attributes": {"value": value, "domain": domain},
                "source": "knowledge_update"
            }
            self.semantic_memory.integrate_knowledge(knowledge_data, source="world_model")
            
            self.emotion.process_event("learning_progress", source="knowledge_system")

    def get_strategy_suggestion(self, task: Task) -> StrategySuggestion:
        """
        Get a strategy suggestion for a task.

        Combines metacognition analysis with emotion-based modifiers,
        curriculum difficulty assessment, and decision modulation.

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

        # Get curriculum difficulty assessment
        difficulty = self.curriculum.estimate_task_difficulty(task)
        zpd_assessment = self.curriculum.is_in_zpd(task)

        # Check if task is too difficult
        if not zpd_assessment.in_zpd and zpd_assessment.difficulty_gap > 0:
            scaffolding = self.curriculum.suggest_scaffolding(task)
            return StrategySuggestion(
                strategy="scaffolding_required",
                confidence=0.9,
                reasoning=f"Task difficulty ({difficulty.overall_difficulty:.2f}) exceeds capability. Scaffolding recommended.",
                parameters={
                    "difficulty": difficulty.__dict__,
                    "zpd_assessment": zpd_assessment.__dict__,
                    "scaffolding": scaffolding,
                },
            )

        # Get base suggestion from metacognition
        suggestion = self.metacognition.get_strategy_suggestion(task)

        # Apply emotion-based modifiers from legacy system
        emotion_modifiers = self.emotion.get_strategy_modifiers()
        response_style = self.emotion.get_response_style()

        # Apply decision modulation from PAD emotion system
        decision_modifiers = self.decision_modulator.get_modifiers()

        # Calculate intrinsic motivation exploration bonus
        exploration_bonus = self.intrinsic_motivation.get_exploration_bonus(
            action=task_type,
            context={"task": task}
        )

        # Enhance suggestion parameters
        suggestion.parameters["emotion_modifiers"] = emotion_modifiers
        suggestion.parameters["response_style"] = response_style
        suggestion.parameters["max_complexity"] = self.development.get_max_complexity()
        suggestion.parameters["difficulty_assessment"] = {
            "overall_difficulty": difficulty.overall_difficulty,
            "in_zpd": zpd_assessment.in_zpd,
            "difficulty_gap": zpd_assessment.difficulty_gap,
            "recommended_difficulty": self.curriculum.get_recommended_difficulty(task_type),
        }
        suggestion.parameters["decision_modifiers"] = {
            "risk_tolerance": decision_modifiers.risk_tolerance,
            "deliberation_time": decision_modifiers.deliberation_time,
            "exploration_bias": decision_modifiers.exploration_bias,
            "confidence_threshold": decision_modifiers.confidence_threshold,
        }
        suggestion.parameters["exploration_bonus"] = exploration_bonus

        # Adjust confidence based on decision modifiers
        adjusted_confidence = suggestion.confidence + decision_modifiers.confidence_threshold
        suggestion.confidence = max(0.0, min(1.0, adjusted_confidence))

        return suggestion

    def evaluate_result(
        self, result: Any, expected: Any = None, task: Optional[Task] = None
    ) -> Evaluation:
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
        emotion_impact = {
            "valence_delta": 0.1 if success else -0.1,
            "arousal_delta": -0.05,  # Completion reduces arousal
        }

        # Calculate motivation impact
        motivation_impact = {}
        if task:
            motivation_impact = self.motivation.on_task_completed(task.task_type, success, score)

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

    def set_development_stage(self, stage: str) -> None:
        """
        Set the development stage and adjust curriculum thresholds accordingly.

        Args:
            stage: The new development stage (infant, child, adolescent, adult).
        """
        # Update development system
        self.development.set_stage(stage)

        # Adjust curriculum thresholds based on stage
        stage_config = {
            "infant": {
                "zpd_lower_threshold": 0.1,
                "zpd_upper_threshold": 0.3,
                "capability_growth_rate": 0.1,
                "capability_decay_rate": 0.01,
            },
            "child": {
                "zpd_lower_threshold": 0.15,
                "zpd_upper_threshold": 0.35,
                "capability_growth_rate": 0.08,
                "capability_decay_rate": 0.015,
            },
            "adolescent": {
                "zpd_lower_threshold": 0.2,
                "zpd_upper_threshold": 0.4,
                "capability_growth_rate": 0.05,
                "capability_decay_rate": 0.02,
            },
            "adult": {
                "zpd_lower_threshold": 0.2,
                "zpd_upper_threshold": 0.4,
                "capability_growth_rate": 0.05,
                "capability_decay_rate": 0.02,
            },
        }

        if stage in stage_config:
            config = stage_config[stage]
            self.curriculum._config.zpd_lower_threshold = config["zpd_lower_threshold"]
            self.curriculum._config.zpd_upper_threshold = config["zpd_upper_threshold"]
            self.curriculum._config.capability_growth_rate = config["capability_growth_rate"]
            self.curriculum._config.capability_decay_rate = config["capability_decay_rate"]

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

            with open(path, "w", encoding="utf-8") as f:
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
            with open(path, "r", encoding="utf-8") as f:
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
            # Existing modules
            "self_model": self.self_model.to_dict(),
            "world_model": self.world_model.to_dict(),
            "metacognition": self.metacognition.to_dict(),
            "motivation": self.motivation.to_dict(),
            "emotion": self.emotion.to_dict(),
            "development": self.development.to_dict(),
            # Curriculum learning layer
            "curriculum": self.curriculum.to_dict(),
            "intrinsic_motivation": self.intrinsic_motivation.to_dict(),
            "experience_replay": self.experience_replay.to_dict(),
            # Dual memory layer
            "episodic_memory": self.episodic_memory.to_dict(),
            "semantic_memory": self.semantic_memory.to_dict(),
            "symbol_grounding": self.symbol_grounding.to_dict(),
            # Enhanced emotion layer
            "pad_emotion": self.pad_emotion.to_dict(),
            "cognitive_appraisal": self.cognitive_appraisal.to_dict(),
            "decision_modulator": self.decision_modulator.to_dict(),
            # System metadata
            "initialized_at": self._initialized_at,
            "saved_at": time.time(),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the consciousness core from a dictionary.

        Args:
            data: Dictionary representation of the consciousness core.
        """
        # Existing modules
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
        
        # Curriculum learning layer
        if "curriculum" in data:
            self.curriculum.from_dict(data["curriculum"])
        if "intrinsic_motivation" in data:
            self.intrinsic_motivation.from_dict(data["intrinsic_motivation"])
        if "experience_replay" in data:
            self.experience_replay.from_dict(data["experience_replay"])
        
        # Dual memory layer
        if "episodic_memory" in data:
            self.episodic_memory.load_state(data["episodic_memory"])
        if "semantic_memory" in data:
            self.semantic_memory.from_dict(data["semantic_memory"])
        if "symbol_grounding" in data:
            self.symbol_grounding.from_dict(data["symbol_grounding"])
        
        # Enhanced emotion layer
        if "pad_emotion" in data:
            self.pad_emotion.from_dict(data["pad_emotion"])
        if "cognitive_appraisal" in data:
            self.cognitive_appraisal.from_dict(data["cognitive_appraisal"])
        if "decision_modulator" in data:
            self.decision_modulator.from_dict(data["decision_modulator"])
        
        # System metadata
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
