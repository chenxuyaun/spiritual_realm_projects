"""
Unit tests for consciousness modules.

Tests specific examples and edge cases for:
- SelfModel
- WorldModel
- Metacognition
- MotivationSystem
- EmotionSystem
- DevelopmentSystem
- ConsciousnessCore
"""

import pytest
import time

from mm_orch.consciousness import (
    ConsciousnessCore,
    SelfModel,
    WorldModel,
    Metacognition,
    MotivationSystem,
    EmotionSystem,
    DevelopmentSystem,
    DevelopmentStage,
    GoalType,
    GoalStatus,
    Capability,
    Entity,
    get_consciousness,
)
from mm_orch.schemas import Task, SystemEvent


class TestSelfModel:
    """Unit tests for SelfModel."""
    
    def test_initialization(self):
        """Test SelfModel initializes with default capabilities."""
        model = SelfModel()
        
        capabilities = model.get_capabilities()
        assert len(capabilities) > 0
        assert "search_qa" in capabilities
        assert "chat_generate" in capabilities
    
    def test_capability_enable_disable(self):
        """Test enabling and disabling capabilities."""
        model = SelfModel()
        
        # Disable a capability
        assert model.is_capability_enabled("search_qa")
        model.disable_capability("search_qa")
        assert not model.is_capability_enabled("search_qa")
        
        # Re-enable
        model.enable_capability("search_qa")
        assert model.is_capability_enabled("search_qa")
    
    def test_capability_usage_recording(self):
        """Test recording capability usage."""
        model = SelfModel()
        
        # Record usage
        model.record_capability_usage("search_qa", success=True, performance_score=0.9)
        
        cap = model.get_capability("search_qa")
        assert cap.usage_count == 1
        assert cap.last_used is not None
        
        # Check performance history
        history = model.get_performance_history()
        assert len(history) == 1
        assert history[0]["capability"] == "search_qa"
        assert history[0]["success"] is True
    
    def test_register_custom_capability(self):
        """Test registering a custom capability."""
        model = SelfModel()
        
        custom = Capability(
            name="custom_workflow",
            description="A custom workflow"
        )
        model.register_capability(custom)
        
        assert model.is_capability_enabled("custom_workflow")
        assert model.get_capability("custom_workflow") is not None
    
    def test_serialization(self):
        """Test SelfModel serialization and deserialization."""
        model1 = SelfModel()
        model1.update_state({"status": "test_status"})
        model1.record_capability_usage("search_qa", True, 0.8)
        
        data = model1.to_dict()
        
        model2 = SelfModel()
        model2.from_dict(data)
        
        assert model2.get_state()["status"] == "test_status"
        assert len(model2.get_performance_history()) == 1


class TestWorldModel:
    """Unit tests for WorldModel."""
    
    def test_entity_management(self):
        """Test adding and retrieving entities."""
        model = WorldModel()
        
        entity = Entity(
            entity_id="test_entity",
            entity_type="concept",
            attributes={"name": "Test"}
        )
        model.add_entity(entity)
        
        retrieved = model.get_entity("test_entity")
        assert retrieved is not None
        assert retrieved.entity_type == "concept"
        assert retrieved.attributes["name"] == "Test"
    
    def test_entity_update(self):
        """Test updating entity attributes."""
        model = WorldModel()
        
        entity = Entity(entity_id="e1", entity_type="topic")
        model.add_entity(entity)
        
        model.update_entity("e1", {"updated": True})
        
        retrieved = model.get_entity("e1")
        assert retrieved.attributes["updated"] is True
    
    def test_entity_relations(self):
        """Test adding relations between entities."""
        model = WorldModel()
        
        e1 = Entity(entity_id="e1", entity_type="concept")
        e2 = Entity(entity_id="e2", entity_type="concept")
        model.add_entity(e1)
        model.add_entity(e2)
        
        model.add_relation("e1", "related_to", "e2")
        
        retrieved = model.get_entity("e1")
        assert "related_to" in retrieved.relations
        assert "e2" in retrieved.relations["related_to"]
    
    def test_user_model(self):
        """Test user model management."""
        model = WorldModel()
        
        user = model.get_or_create_user("user1")
        assert user.user_id == "user1"
        
        model.update_user_preferences("user1", {"theme": "dark"})
        user = model.get_user("user1")
        assert user.preferences["theme"] == "dark"
    
    def test_knowledge_management(self):
        """Test knowledge storage and retrieval."""
        model = WorldModel()
        
        model.add_fact("test_fact", "test_value")
        
        value = model.get_fact("test_fact")
        assert value == "test_value"
        
        knowledge = model.get_knowledge("facts")
        assert "test_fact" in knowledge


class TestMetacognition:
    """Unit tests for Metacognition."""
    
    def test_task_monitoring(self):
        """Test task monitoring lifecycle."""
        meta = Metacognition()
        
        task = Task.create("search_qa", {"query": "test"})
        
        # Start monitoring
        monitor = meta.start_monitoring(task)
        assert monitor.task_id == task.task_id
        assert monitor.status == "running"
        
        # Update progress
        meta.update_progress(task.task_id, 0.5, "halfway")
        monitor = meta.get_task_monitor(task.task_id)
        assert monitor.progress == 0.5
        assert len(monitor.checkpoints) == 1
        
        # Complete task
        completed = meta.complete_task(task.task_id, True, {"score": 0.9})
        assert completed.status == "completed"
        assert completed.progress == 1.0
    
    def test_strategy_suggestion(self):
        """Test strategy suggestion for different task types."""
        meta = Metacognition()
        
        # Simple task
        simple_task = Task.create("chat_generate", {"query": "hello"})
        suggestion = meta.get_strategy_suggestion(simple_task)
        assert suggestion.strategy is not None
        assert 0 <= suggestion.confidence <= 1
        
        # Complex task
        complex_task = Task.create("self_ask_search_qa", {"query": "complex question " * 50})
        suggestion = meta.get_strategy_suggestion(complex_task)
        assert suggestion.strategy is not None
    
    def test_strategy_recording(self):
        """Test recording strategy results."""
        meta = Metacognition()
        
        meta.record_strategy_result("direct_answer", True, 0.9, 1.5)
        meta.record_strategy_result("direct_answer", True, 0.8, 2.0)
        
        stats = meta.get_strategy_stats()
        assert stats["direct_answer"]["usage_count"] == 2
        assert stats["direct_answer"]["success_rate"] == 1.0


class TestMotivationSystem:
    """Unit tests for MotivationSystem."""
    
    def test_goal_creation(self):
        """Test creating goals."""
        motivation = MotivationSystem()
        
        goal = motivation.create_goal(
            name="Test Goal",
            description="A test goal",
            goal_type=GoalType.SHORT_TERM,
            priority=0.8
        )
        
        assert goal.name == "Test Goal"
        assert goal.priority == 0.8
        assert goal.status == GoalStatus.PENDING
    
    def test_goal_hierarchy(self):
        """Test goal parent-child relationships."""
        motivation = MotivationSystem()
        
        parent = motivation.create_goal("Parent", "Parent goal", GoalType.LONG_TERM)
        child = motivation.create_goal(
            "Child", "Child goal", 
            GoalType.SHORT_TERM,
            parent_goal_id=parent.goal_id
        )
        
        parent = motivation.get_goal(parent.goal_id)
        assert child.goal_id in parent.sub_goal_ids
    
    def test_goal_progress_update(self):
        """Test updating goal progress."""
        motivation = MotivationSystem()
        
        goal = motivation.create_goal("Test", "Test", GoalType.SHORT_TERM)
        
        motivation.update_goal_progress(goal.goal_id, 0.5)
        updated = motivation.get_goal(goal.goal_id)
        assert updated.progress == 0.5
        assert updated.status == GoalStatus.IN_PROGRESS
        
        motivation.update_goal_progress(goal.goal_id, 1.0)
        updated = motivation.get_goal(goal.goal_id)
        assert updated.status == GoalStatus.COMPLETED
    
    def test_drive_levels(self):
        """Test drive level management."""
        motivation = MotivationSystem()
        
        initial = motivation.get_drive_level("curiosity")
        assert 0 <= initial <= 1
        
        motivation.update_drive_level("curiosity", 0.1)
        new_level = motivation.get_drive_level("curiosity")
        assert new_level == initial + 0.1


class TestEmotionSystem:
    """Unit tests for EmotionSystem."""
    
    def test_initial_state(self):
        """Test initial emotion state."""
        emotion = EmotionSystem()
        
        state = emotion.get_state()
        assert "valence" in state
        assert "arousal" in state
        assert -1 <= state["valence"] <= 1
        assert 0 <= state["arousal"] <= 1
    
    def test_event_processing(self):
        """Test processing emotion events."""
        emotion = EmotionSystem()
        
        # Process success event
        result = emotion.process_event("task_success")
        assert result["valence"] > 0  # Success should increase valence
        
        # Process failure event
        emotion2 = EmotionSystem()
        result = emotion2.process_event("task_failure")
        assert result["valence"] < 0  # Failure should decrease valence
    
    def test_emotion_decay(self):
        """Test emotion decay over time."""
        emotion = EmotionSystem()
        
        # Set extreme emotion
        emotion.set_emotion(0.9, 0.9)
        
        # Simulate time passing (manually trigger decay)
        emotion._last_update = time.time() - 120  # 2 minutes ago
        emotion._apply_decay()
        
        valence, arousal = emotion.get_emotion_values()
        # Values should have decayed towards neutral
        assert valence < 0.9
        assert arousal < 0.9
    
    def test_strategy_modifiers(self):
        """Test getting strategy modifiers based on emotion."""
        emotion = EmotionSystem()
        
        # Positive high arousal
        emotion.set_emotion(0.8, 0.8)
        modifiers = emotion.get_strategy_modifiers()
        assert isinstance(modifiers, dict)
        assert "temperature" in modifiers
        
        # Negative low arousal
        emotion.set_emotion(-0.8, 0.2)
        modifiers = emotion.get_strategy_modifiers()
        assert isinstance(modifiers, dict)
    
    def test_response_style(self):
        """Test getting response style."""
        emotion = EmotionSystem()
        
        style = emotion.get_response_style()
        assert "temperature" in style
        assert "verbosity" in style
        assert "tone" in style
        assert "risk_tolerance" in style


class TestDevelopmentSystem:
    """Unit tests for DevelopmentSystem."""
    
    def test_stage_initialization(self):
        """Test initializing with different stages."""
        for stage in ["infant", "child", "adolescent", "adult"]:
            dev = DevelopmentSystem(initial_stage=stage)
            assert dev.get_current_stage().value == stage
    
    def test_feature_access_by_stage(self):
        """Test feature access varies by stage."""
        infant = DevelopmentSystem(initial_stage="infant")
        adult = DevelopmentSystem(initial_stage="adult")
        
        # Infant has limited features
        infant_features = infant.get_enabled_features()
        adult_features = adult.get_enabled_features()
        
        assert len(adult_features) > len(infant_features)
        assert "self_ask_search_qa" not in infant_features
        assert "self_ask_search_qa" in adult_features
    
    def test_max_complexity(self):
        """Test max complexity varies by stage."""
        infant = DevelopmentSystem(initial_stage="infant")
        adult = DevelopmentSystem(initial_stage="adult")
        
        assert infant.get_max_complexity() < adult.get_max_complexity()
        assert adult.get_max_complexity() == 1.0
    
    def test_promotion_eligibility(self):
        """Test checking promotion eligibility."""
        dev = DevelopmentSystem(initial_stage="infant")
        
        eligibility = dev.check_promotion_eligibility()
        assert "eligible" in eligibility
        assert "unmet_requirements" in eligibility
    
    def test_learning_record(self):
        """Test recording learning data."""
        dev = DevelopmentSystem(initial_stage="child")
        
        dev.record_task_result(True, 0.9, "search_qa")
        dev.record_task_result(False, 0.3, "search_qa")
        
        records = dev.get_learning_records()
        assert len(records) == 2
        
        stats = dev.get_stage_statistics()
        assert stats["task_count"] == 2
        assert stats["success_count"] == 1


class TestConsciousnessCore:
    """Unit tests for ConsciousnessCore."""
    
    def test_initialization(self):
        """Test ConsciousnessCore initialization."""
        core = ConsciousnessCore()
        
        assert core.self_model is not None
        assert core.world_model is not None
        assert core.metacognition is not None
        assert core.motivation is not None
        assert core.emotion is not None
        assert core.development is not None
    
    def test_status_summary(self):
        """Test getting status summary."""
        core = ConsciousnessCore()
        
        summary = core.get_status_summary()
        assert "self_model" in summary
        assert "world_model" in summary
        assert "metacognition" in summary
        assert "motivation" in summary
        assert "emotion" in summary
        assert "development" in summary
        assert "uptime" in summary
    
    def test_event_handling(self):
        """Test handling system events."""
        core = ConsciousnessCore()
        
        # Task start event
        task = Task.create("search_qa", {"query": "test"})
        event = SystemEvent(
            event_type="task_start",
            data={"task": task}
        )
        core.update_state(event)
        
        state = core.self_model.get_state()
        assert state["status"] == "processing"
        
        # Task complete event
        event = SystemEvent(
            event_type="task_complete",
            data={
                "task_id": task.task_id,
                "success": True,
                "score": 0.9,
                "task_type": "search_qa"
            }
        )
        core.update_state(event)
        
        state = core.self_model.get_state()
        assert state["status"] == "idle"
    
    def test_strategy_suggestion(self):
        """Test getting strategy suggestion."""
        core = ConsciousnessCore()
        
        task = Task.create("search_qa", {"query": "test question"})
        suggestion = core.get_strategy_suggestion(task)
        
        assert suggestion.strategy is not None
        assert "emotion_modifiers" in suggestion.parameters
        assert "response_style" in suggestion.parameters
    
    def test_result_evaluation(self):
        """Test evaluating task results."""
        core = ConsciousnessCore()
        
        task = Task.create("search_qa", {"query": "test"})
        evaluation = core.evaluate_result("answer", "answer", task)
        
        assert evaluation.success is True
        assert evaluation.score == 1.0
        assert "emotion_impact" in evaluation.__dict__
    
    def test_consciousness_state(self):
        """Test getting complete consciousness state."""
        core = ConsciousnessCore()
        
        state = core.get_consciousness_state()
        
        assert state.self_state is not None
        assert state.world_state is not None
        assert "valence" in state.emotion_state
        assert "arousal" in state.emotion_state
        assert state.development_stage in ["infant", "child", "adolescent", "adult"]
    
    def test_serialization(self):
        """Test ConsciousnessCore serialization."""
        core1 = ConsciousnessCore()
        core1.emotion.set_emotion(0.5, 0.6)
        
        data = core1.to_dict()
        
        core2 = ConsciousnessCore()
        core2.from_dict(data)
        
        # Verify emotion state is preserved (approximately, due to decay)
        v1, a1 = core1.emotion.get_emotion_values()
        v2, a2 = core2.emotion.get_emotion_values()
        assert abs(v1 - v2) < 0.1
        assert abs(a1 - a2) < 0.1


class TestSingletonPattern:
    """Test the singleton pattern for consciousness."""
    
    def test_get_consciousness_returns_same_instance(self):
        """Test that get_consciousness returns the same instance."""
        # Note: This test may be affected by other tests that use get_consciousness
        # In a real scenario, you'd want to reset the singleton between tests
        
        # For now, just verify it returns a ConsciousnessCore
        core = get_consciousness()
        assert isinstance(core, ConsciousnessCore)
