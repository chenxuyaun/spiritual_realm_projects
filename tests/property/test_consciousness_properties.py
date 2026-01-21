"""
Property-based tests for consciousness modules.

Tests properties 14-22 from the design document:
- Property 14: 意识模块初始化完整性
- Property 15: 意识模块状态维护
- Property 16: 元认知策略建议
- Property 17: 动机系统目标更新
- Property 18: 情感状态计算
- Property 19: 情感影响处理策略
- Property 20: 发展阶段功能限制
- Property 21: 发展阶段晋升
- Property 22: 发展数据记录
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
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
)
from mm_orch.schemas import Task, StrategySuggestion


# Strategies for generating test data
task_type_strategy = st.sampled_from([
    "search_qa", "lesson_pack", "chat_generate", "rag_qa", 
    "self_ask_search_qa", "simple_qa", "summarization"
])

development_stage_strategy = st.sampled_from(["infant", "child", "adolescent", "adult"])

score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

valence_strategy = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)

arousal_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


class TestConsciousnessInitialization:
    """Tests for Property 14: 意识模块初始化完整性"""
    
    @given(stage=development_stage_strategy)
    @settings(max_examples=100)
    def test_consciousness_core_has_all_modules(self, stage: str):
        """
        Feature: muai-orchestration-system, Property 14: 意识模块初始化完整性
        
        对于任何ConsciousnessCore的实例化，该实例应该包含所有六个子模块的非空引用。
        **Validates: Requirements 6.1**
        """
        core = ConsciousnessCore(config={"development_stage": stage})
        
        # Verify all six modules are present and non-null
        assert core.self_model is not None, "self_model should not be None"
        assert core.world_model is not None, "world_model should not be None"
        assert core.metacognition is not None, "metacognition should not be None"
        assert core.motivation is not None, "motivation should not be None"
        assert core.emotion is not None, "emotion should not be None"
        assert core.development is not None, "development should not be None"
        
        # Verify correct types
        assert isinstance(core.self_model, SelfModel)
        assert isinstance(core.world_model, WorldModel)
        assert isinstance(core.metacognition, Metacognition)
        assert isinstance(core.motivation, MotivationSystem)
        assert isinstance(core.emotion, EmotionSystem)
        assert isinstance(core.development, DevelopmentSystem)


class TestConsciousnessStateMaintenance:
    """Tests for Property 15: 意识模块状态维护"""
    
    @given(
        status=st.sampled_from(["idle", "processing", "error"]),
        load=score_strategy,
        health=score_strategy
    )
    @settings(max_examples=100)
    def test_self_model_state_update_and_query(self, status: str, load: float, health: float):
        """
        Feature: muai-orchestration-system, Property 15: 意识模块状态维护
        
        对于任何意识模块，在系统运行过程中，这些模块应该维护可访问的状态字典，
        且状态应该可以被更新和查询。
        **Validates: Requirements 6.3**
        """
        self_model = SelfModel()
        
        # Update state
        self_model.update_state({
            "status": status,
            "load": load,
            "health": health,
        })
        
        # Query state
        state = self_model.get_state()
        
        # Verify state is accessible and contains updated values
        assert isinstance(state, dict)
        assert state["status"] == status
        assert state["load"] == load
        assert state["health"] == health
    
    @given(
        entity_type=st.sampled_from(["concept", "topic", "tool"]),
        entity_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N')))
    )
    @settings(max_examples=100)
    def test_world_model_state_update_and_query(self, entity_type: str, entity_id: str):
        """
        Feature: muai-orchestration-system, Property 15: 意识模块状态维护
        
        对于任何意识模块，状态应该可以被更新和查询。
        **Validates: Requirements 6.4**
        """
        from mm_orch.consciousness.world_model import Entity
        
        world_model = WorldModel()
        
        # Add entity
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            attributes={"test": True}
        )
        world_model.add_entity(entity)
        
        # Query state
        state = world_model.get_state()
        retrieved = world_model.get_entity(entity_id)
        
        # Verify state is accessible
        assert isinstance(state, dict)
        assert state["entity_count"] >= 1
        assert retrieved is not None
        assert retrieved.entity_id == entity_id
        assert retrieved.entity_type == entity_type


class TestMetacognitionStrategySuggestion:
    """Tests for Property 16: 元认知策略建议"""
    
    @given(
        task_type=task_type_strategy,
        query=st.text(min_size=1, max_size=500)
    )
    @settings(max_examples=100)
    def test_metacognition_returns_strategy_suggestion(self, task_type: str, query: str):
        """
        Feature: muai-orchestration-system, Property 16: 元认知策略建议
        
        对于任何任务请求，当调用意识核心的get_strategy_suggestion方法时，
        应该返回一个StrategySuggestion对象，包含对任务处理的建议。
        **Validates: Requirements 6.5**
        """
        metacognition = Metacognition()
        
        task = Task.create(
            task_type=task_type,
            parameters={"query": query}
        )
        
        suggestion = metacognition.get_strategy_suggestion(task)
        
        # Verify suggestion is valid
        assert isinstance(suggestion, StrategySuggestion)
        assert suggestion.strategy is not None
        assert len(suggestion.strategy) > 0
        assert 0.0 <= suggestion.confidence <= 1.0
        assert suggestion.reasoning is not None
        assert isinstance(suggestion.parameters, dict)


class TestMotivationGoalUpdate:
    """Tests for Property 17: 动机系统目标更新"""
    
    @given(
        task_type=task_type_strategy,
        success=st.booleans(),
        score=score_strategy
    )
    @settings(max_examples=100)
    def test_motivation_goal_update_on_task_completion(self, task_type: str, success: bool, score: float):
        """
        Feature: muai-orchestration-system, Property 17: 动机系统目标更新
        
        对于任何任务完成事件，动机系统的目标状态应该被更新，
        且相关目标的完成状态或优先级应该发生变化。
        **Validates: Requirements 7.1, 7.2**
        """
        motivation = MotivationSystem()
        
        # Get initial state
        initial_state = motivation.get_state()
        initial_drive = motivation.get_drive_level("helpfulness")
        
        # Complete a task
        changes = motivation.on_task_completed(task_type, success, score)
        
        # Verify state was updated
        assert isinstance(changes, dict)
        
        # Drive levels should change
        new_drive = motivation.get_drive_level("helpfulness")
        if success:
            assert new_drive >= initial_drive or abs(new_drive - initial_drive) < 0.01
        
        # State should reflect the update
        new_state = motivation.get_state()
        assert isinstance(new_state, dict)
        assert "drive_levels" in new_state


class TestEmotionStateCalculation:
    """Tests for Property 18: 情感状态计算"""
    
    @given(
        success=st.booleans(),
        score=score_strategy
    )
    @settings(max_examples=100)
    def test_emotion_state_calculation(self, success: bool, score: float):
        """
        Feature: muai-orchestration-system, Property 18: 情感状态计算
        
        对于任何任务结果，情感系统应该计算新的情感状态值（valence和arousal），
        且这些值应该在合理的范围内。
        **Validates: Requirements 7.3**
        """
        emotion = EmotionSystem()
        
        # Process task result
        result = emotion.on_task_result(success, score)
        
        # Verify emotion state is calculated
        assert isinstance(result, dict)
        assert "valence" in result
        assert "arousal" in result
        
        # Verify values are in valid range
        assert -1.0 <= result["valence"] <= 1.0, f"Valence {result['valence']} out of range"
        assert 0.0 <= result["arousal"] <= 1.0, f"Arousal {result['arousal']} out of range"
        
        # Verify emotion label is present
        assert "emotion_label" in result
        assert isinstance(result["emotion_label"], str)


class TestEmotionInfluenceOnStrategy:
    """Tests for Property 19: 情感影响处理策略"""
    
    @given(
        valence1=valence_strategy,
        arousal1=arousal_strategy,
        valence2=valence_strategy,
        arousal2=arousal_strategy
    )
    @settings(max_examples=100)
    def test_different_emotions_produce_different_strategies(
        self, valence1: float, arousal1: float, valence2: float, arousal2: float
    ):
        """
        Feature: muai-orchestration-system, Property 19: 情感影响处理策略
        
        对于任何两个具有不同情感状态的系统状态，在相同的任务输入下，
        处理策略（如响应风格、温度参数）应该有所不同。
        **Validates: Requirements 7.4, 7.5**
        """
        # Skip if emotions are in the same quadrant (same strategy category)
        # The emotion system uses quadrants: high/low valence x high/low arousal
        same_valence_sign = (valence1 > 0) == (valence2 > 0)
        same_arousal_region = (arousal1 > 0.5) == (arousal2 > 0.5)
        
        # Only test when emotions are in different quadrants
        assume(not (same_valence_sign and same_arousal_region))
        
        emotion1 = EmotionSystem()
        emotion2 = EmotionSystem()
        
        # Set different emotion states
        emotion1.set_emotion(valence1, arousal1)
        emotion2.set_emotion(valence2, arousal2)
        
        # Get strategy modifiers
        modifiers1 = emotion1.get_strategy_modifiers()
        modifiers2 = emotion2.get_strategy_modifiers()
        
        # Get response styles
        style1 = emotion1.get_response_style()
        style2 = emotion2.get_response_style()
        
        # Verify modifiers are dictionaries
        assert isinstance(modifiers1, dict)
        assert isinstance(modifiers2, dict)
        assert isinstance(style1, dict)
        assert isinstance(style2, dict)
        
        # When emotions are in different quadrants, at least one parameter should differ
        # Check if any modifier or style parameter differs
        modifiers_differ = any(
            abs(modifiers1.get(k, 0) - modifiers2.get(k, 0)) > 0.001
            for k in set(modifiers1.keys()) | set(modifiers2.keys())
        )
        styles_differ = any(
            style1.get(k) != style2.get(k)
            for k in set(style1.keys()) | set(style2.keys())
        )
        assert modifiers_differ or styles_differ, \
            f"Different emotion quadrants should produce different strategies: " \
            f"({valence1:.2f}, {arousal1:.2f}) vs ({valence2:.2f}, {arousal2:.2f})"


class TestDevelopmentStageFeatureRestriction:
    """Tests for Property 20: 发展阶段功能限制"""
    
    @given(stage=development_stage_strategy)
    @settings(max_examples=100)
    def test_feature_restriction_by_stage(self, stage: str):
        """
        Feature: muai-orchestration-system, Property 20: 发展阶段功能限制
        
        对于任何发展阶段，当系统处于该阶段时，尝试使用该阶段未解锁的功能应该被拒绝，
        而已解锁的功能应该可以正常使用。
        **Validates: Requirements 8.3**
        """
        development = DevelopmentSystem(initial_stage=stage)
        
        # Get enabled features for this stage
        enabled = development.get_enabled_features()
        
        # Verify enabled features are accessible
        for feature in enabled:
            assert development.is_feature_enabled(feature), \
                f"Feature '{feature}' should be enabled in stage '{stage}'"
            access = development.check_feature_access(feature)
            assert access["enabled"] is True
        
        # Define features that should be restricted in earlier stages
        all_features = {
            "chat_generate", "simple_qa", "search_qa", "summarization",
            "rag_qa", "lesson_pack", "embedding", "self_ask_search_qa",
            "complex_reasoning", "multi_step"
        }
        
        # Check that non-enabled features are restricted
        restricted = all_features - enabled
        for feature in restricted:
            assert not development.is_feature_enabled(feature), \
                f"Feature '{feature}' should be restricted in stage '{stage}'"
            access = development.check_feature_access(feature)
            assert access["enabled"] is False


class TestDevelopmentStagePromotion:
    """Tests for Property 21: 发展阶段晋升"""
    
    def test_stage_promotion_with_force(self):
        """
        Feature: muai-orchestration-system, Property 21: 发展阶段晋升
        
        对于任何发展阶段，当满足该阶段的晋升条件时，调用发展系统的评估方法
        应该触发阶段升级，且当前阶段应该变更为下一个阶段。
        **Validates: Requirements 8.4**
        """
        # Test promotion from each stage (except adult)
        stages = ["infant", "child", "adolescent"]
        next_stages = ["child", "adolescent", "adult"]
        
        for current, expected_next in zip(stages, next_stages):
            development = DevelopmentSystem(initial_stage=current)
            
            # Force promotion
            result = development.promote(force=True)
            
            assert result["success"] is True, f"Promotion from {current} should succeed"
            assert result["from_stage"] == current
            assert result["to_stage"] == expected_next
            assert development.get_current_stage().value == expected_next
    
    def test_adult_cannot_promote(self):
        """Adult stage cannot be promoted further."""
        development = DevelopmentSystem(initial_stage="adult")
        
        result = development.promote(force=True)
        
        assert result["success"] is False
        assert "maximum" in result["reason"].lower() or "already" in result["reason"].lower()


class TestDevelopmentDataRecording:
    """Tests for Property 22: 发展数据记录"""
    
    @given(
        stage=development_stage_strategy,
        success=st.booleans(),
        score=score_strategy,
        task_type=task_type_strategy
    )
    @settings(max_examples=100)
    def test_learning_data_recording(self, stage: str, success: bool, score: float, task_type: str):
        """
        Feature: muai-orchestration-system, Property 22: 发展数据记录
        
        对于任何发展阶段，系统应该记录该阶段的学习数据和性能指标，
        且这些数据应该可以被查询和导出。
        **Validates: Requirements 8.5**
        """
        development = DevelopmentSystem(initial_stage=stage)
        
        # Record task result
        development.record_task_result(success, score, task_type)
        
        # Query learning records
        records = development.get_learning_records()
        
        # Verify records are accessible
        assert isinstance(records, list)
        assert len(records) >= 1
        
        # Verify record structure
        latest = records[-1]
        assert "record_id" in latest
        assert "stage" in latest
        assert latest["stage"] == stage
        assert "metric_type" in latest
        assert "value" in latest
        assert "timestamp" in latest
        
        # Verify statistics are available
        stats = development.get_stage_statistics()
        assert isinstance(stats, dict)
        assert stats["task_count"] >= 1
        assert "success_rate" in stats
        assert "average_score" in stats


class TestConsciousnessCoreSerialization:
    """Tests for consciousness state persistence."""
    
    @given(stage=development_stage_strategy)
    @settings(max_examples=50)
    def test_consciousness_serialization_roundtrip(self, stage: str):
        """
        Test that consciousness state can be serialized and deserialized.
        **Validates: Requirements 14.3, 14.4**
        """
        # Create and modify consciousness
        core1 = ConsciousnessCore(config={"development_stage": stage})
        
        # Make some state changes
        core1.self_model.update_state({"status": "test"})
        core1.emotion.set_emotion(0.5, 0.7)
        
        # Serialize
        data = core1.to_dict()
        
        # Create new instance and deserialize
        core2 = ConsciousnessCore()
        core2.from_dict(data)
        
        # Verify state is preserved
        assert core2.self_model.get_state()["status"] == "test"
        valence, arousal = core2.emotion.get_emotion_values()
        assert abs(valence - 0.5) < 0.1  # Allow for decay
        assert abs(arousal - 0.7) < 0.1
