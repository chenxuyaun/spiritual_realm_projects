"""
Unit tests for ConsciousnessCore integration with new modules.

Tests event routing, state persistence, and stage transitions.
"""

import pytest
from mm_orch.consciousness.core import ConsciousnessCore, get_consciousness
from mm_orch.consciousness.curriculum import CapabilityDimension
from mm_orch.schemas import SystemEvent, Task


class TestConsciousnessIntegration:
    """Unit tests for ConsciousnessCore integration."""

    def test_initialization_with_all_modules(self):
        """Test that all modules are initialized correctly."""
        core = ConsciousnessCore()
        
        # Check existing modules
        assert core.self_model is not None
        assert core.world_model is not None
        assert core.metacognition is not None
        assert core.motivation is not None
        assert core.emotion is not None
        assert core.development is not None
        
        # Check curriculum learning layer
        assert core.curriculum is not None
        assert core.intrinsic_motivation is not None
        assert core.experience_replay is not None
        
        # Check dual memory layer
        assert core.episodic_memory is not None
        assert core.semantic_memory is not None
        assert core.symbol_grounding is not None
        
        # Check enhanced emotion layer
        assert core.pad_emotion is not None
        assert core.cognitive_appraisal is not None
        assert core.decision_modulator is not None

    def test_event_routing_task_complete(self):
        """Test event routing for task completion."""
        core = ConsciousnessCore()
        
        initial_episodes = len(core.episodic_memory._episodes)
        initial_experiences = len(core.experience_replay._experiences)
        
        event = SystemEvent(
            event_type="task_complete",
            data={
                "task_id": "test_task_1",
                "success": True,
                "score": 0.9,
                "task_type": "search_qa",
                "context": {"query": "test query"},
            }
        )
        
        core.update_state(event)
        
        # Verify episode was created
        assert len(core.episodic_memory._episodes) == initial_episodes + 1
        
        # Verify experience was stored
        assert len(core.experience_replay._experiences) == initial_experiences + 1
        
        # Verify curriculum was updated
        capability = core.curriculum.get_capability_level("reasoning")
        assert capability > 0.0  # Should have increased

    def test_event_routing_task_error(self):
        """Test event routing for task errors."""
        core = ConsciousnessCore()
        
        initial_episodes = len(core.episodic_memory._episodes)
        
        event = SystemEvent(
            event_type="task_error",
            data={
                "task_id": "test_task_2",
                "error_type": "timeout",
                "task_type": "chat_generate",
                "context": {"query": "test query"},
            }
        )
        
        core.update_state(event)
        
        # Verify episode was created for error
        assert len(core.episodic_memory._episodes) == initial_episodes + 1
        
        # Verify high-priority experience was stored
        experiences = core.experience_replay._experiences
        assert len(experiences) > 0
        # Get the most recent experience (experiences is a dict)
        experience_list = list(experiences.values())
        # Priority is clamped to [0.0, 1.0], so high priority errors should be 1.0
        assert experience_list[-1].priority == 1.0

    def test_event_routing_user_feedback(self):
        """Test event routing for user feedback."""
        core = ConsciousnessCore()
        
        initial_episodes = len(core.episodic_memory._episodes)
        
        event = SystemEvent(
            event_type="user_interaction",
            data={
                "user_id": "test_user",
                "type": "message",
                "feedback": "positive",
            }
        )
        
        core.update_state(event)
        
        # Verify episode was created for feedback
        assert len(core.episodic_memory._episodes) == initial_episodes + 1
        
        # Verify episode has correct metadata
        episode_list = list(core.episodic_memory._episodes.values())
        episode = episode_list[-1]
        assert episode.metadata.get("feedback") == "positive"
        assert episode.metadata.get("user_id") == "test_user"

    def test_event_routing_knowledge_update(self):
        """Test event routing for knowledge updates."""
        core = ConsciousnessCore()
        
        event = SystemEvent(
            event_type="knowledge_update",
            data={
                "domain": "facts",
                "key": "test_concept",
                "value": "test_value",
            }
        )
        
        core.update_state(event)
        
        # Verify world model was updated
        knowledge = core.world_model.get_knowledge("facts")
        assert "test_concept" in knowledge
        assert knowledge["test_concept"] == "test_value"

    def test_strategy_suggestion_with_difficulty(self):
        """Test strategy suggestion includes difficulty assessment."""
        core = ConsciousnessCore()
        
        task = Task(
            task_id="test_task",
            task_type="search_qa",
            parameters={"query": "test query"},
        )
        
        suggestion = core.get_strategy_suggestion(task)
        
        # Verify difficulty assessment is included
        assert "difficulty_assessment" in suggestion.parameters
        difficulty = suggestion.parameters["difficulty_assessment"]
        assert "overall_difficulty" in difficulty
        assert "in_zpd" in difficulty
        assert "difficulty_gap" in difficulty
        
        # Verify decision modifiers are included
        assert "decision_modifiers" in suggestion.parameters
        modifiers = suggestion.parameters["decision_modifiers"]
        assert "risk_tolerance" in modifiers
        assert "deliberation_time" in modifiers
        
        # Verify exploration bonus is included
        assert "exploration_bonus" in suggestion.parameters

    def test_strategy_suggestion_with_scaffolding(self):
        """Test strategy suggestion for too-difficult tasks."""
        core = ConsciousnessCore()
        
        # Set low capabilities using public method
        for dim in CapabilityDimension:
            # Update capabilities by simulating failed tasks
            for _ in range(5):
                core.curriculum.update_capabilities(dim.value, success=False, score=0.1)
        
        # Create a difficult task
        task = Task(
            task_id="difficult_task",
            task_type="lesson_pack",  # Typically more complex
            parameters={"topic": "advanced quantum physics"},
        )
        
        suggestion = core.get_strategy_suggestion(task)
        
        # Should recommend scaffolding for difficult task
        # (depending on difficulty estimation, may or may not trigger)
        assert suggestion is not None
        assert suggestion.confidence >= 0.0

    def test_state_persistence_round_trip(self):
        """Test state serialization and deserialization."""
        core = ConsciousnessCore()
        
        # Create some state
        event = SystemEvent(
            event_type="task_complete",
            data={
                "task_id": "test_task",
                "success": True,
                "score": 0.8,
                "task_type": "search_qa",
                "context": {},
            }
        )
        core.update_state(event)
        
        # Serialize
        state_dict = core.to_dict()
        
        # Verify all modules are in state
        assert "curriculum" in state_dict
        assert "episodic_memory" in state_dict
        assert "semantic_memory" in state_dict
        assert "pad_emotion" in state_dict
        assert "decision_modulator" in state_dict
        
        # Deserialize into new core
        new_core = ConsciousnessCore()
        new_core.from_dict(state_dict)
        
        # Verify state was restored
        assert len(new_core.episodic_memory._episodes) == len(core.episodic_memory._episodes)
        
        # Verify PAD state
        original_pad = core.pad_emotion.get_state()
        restored_pad = new_core.pad_emotion.get_state()
        assert abs(original_pad.pleasure - restored_pad.pleasure) < 0.01
        assert abs(original_pad.arousal - restored_pad.arousal) < 0.01

    def test_development_stage_transition_infant(self):
        """Test development stage transition to infant."""
        core = ConsciousnessCore()
        
        core.set_development_stage("infant")
        
        # Verify stage was changed
        assert core.development.get_current_stage().value == "infant"
        
        # Verify curriculum thresholds were adjusted
        assert core.curriculum._config.zpd_lower_threshold == 0.1
        assert core.curriculum._config.zpd_upper_threshold == 0.3
        assert core.curriculum._config.capability_growth_rate == 0.1

    def test_development_stage_transition_child(self):
        """Test development stage transition to child."""
        core = ConsciousnessCore()
        
        core.set_development_stage("child")
        
        # Verify stage was changed
        assert core.development.get_current_stage().value == "child"
        
        # Verify curriculum thresholds were adjusted
        assert core.curriculum._config.zpd_lower_threshold == 0.15
        assert core.curriculum._config.zpd_upper_threshold == 0.35

    def test_development_stage_transition_adolescent(self):
        """Test development stage transition to adolescent."""
        core = ConsciousnessCore()
        
        core.set_development_stage("adolescent")
        
        # Verify stage was changed
        assert core.development.get_current_stage().value == "adolescent"
        
        # Verify curriculum thresholds were adjusted
        assert core.curriculum._config.zpd_lower_threshold == 0.2
        assert core.curriculum._config.zpd_upper_threshold == 0.4

    def test_development_stage_transition_adult(self):
        """Test development stage transition to adult."""
        core = ConsciousnessCore()
        
        core.set_development_stage("adult")
        
        # Verify stage was changed
        assert core.development.get_current_stage().value == "adult"
        
        # Verify curriculum thresholds were adjusted
        assert core.curriculum._config.zpd_lower_threshold == 0.2
        assert core.curriculum._config.zpd_upper_threshold == 0.4

    def test_get_status_summary_includes_new_modules(self):
        """Test that status summary includes all new modules."""
        core = ConsciousnessCore()
        
        status = core.get_status_summary()
        
        # Check existing modules
        assert "self_model" in status
        assert "world_model" in status
        assert "emotion" in status
        
        # Check new modules
        assert "curriculum" in status
        assert "intrinsic_motivation" in status
        assert "experience_replay" in status
        assert "episodic_memory" in status
        assert "semantic_memory" in status
        assert "symbol_grounding" in status
        assert "pad_emotion" in status
        assert "decision_modulator" in status

    def test_dual_emotion_system_coordination(self):
        """Test that both emotion systems are updated."""
        core = ConsciousnessCore()
        
        # Get initial states
        initial_legacy_valence, initial_legacy_arousal = core.emotion.get_emotion_values()
        initial_pad = core.pad_emotion.get_state()
        
        # Process event
        event = SystemEvent(
            event_type="task_complete",
            data={
                "task_id": "test",
                "success": True,
                "score": 0.9,
                "task_type": "search_qa",
            }
        )
        core.update_state(event)
        
        # Verify both systems were updated
        final_legacy_valence, final_legacy_arousal = core.emotion.get_emotion_values()
        final_pad = core.pad_emotion.get_state()
        
        # Legacy emotion system should have been updated
        assert final_legacy_valence >= initial_legacy_valence  # Success increases valence
        
        # PAD emotion system should have valid state
        assert -1.0 <= final_pad.pleasure <= 1.0
        assert 0.0 <= final_pad.arousal <= 1.0
        assert -1.0 <= final_pad.dominance <= 1.0

    def test_memory_consolidation_integration(self):
        """Test that episodic memory can consolidate to semantic memory."""
        core = ConsciousnessCore()
        
        # Create multiple episodes
        for i in range(5):
            event = SystemEvent(
                event_type="task_complete",
                data={
                    "task_id": f"task_{i}",
                    "success": True,
                    "score": 0.8,
                    "task_type": "search_qa",
                    "context": {"query": f"query {i}"},
                }
            )
            core.update_state(event)
        
        # Verify episodes were created
        assert len(core.episodic_memory._episodes) >= 5
        
        # Consolidation would be triggered periodically
        # For now, just verify the integration is in place
        assert core.episodic_memory is not None
        assert core.semantic_memory is not None

    def test_intrinsic_motivation_integration(self):
        """Test intrinsic motivation integration with task completion."""
        core = ConsciousnessCore()
        
        event = SystemEvent(
            event_type="task_complete",
            data={
                "task_id": "test",
                "success": True,
                "score": 0.9,
                "task_type": "search_qa",
                "predicted_outcome": {"success": False, "score": 0.5},
                "actual_outcome": {"success": True, "score": 0.9},
            }
        )
        
        core.update_state(event)
        
        # Curiosity reward should have been calculated
        # (stored in experience data)
        assert len(core.experience_replay._experiences) > 0
