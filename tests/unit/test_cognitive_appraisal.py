"""
Unit tests for Cognitive Appraisal System.

Tests the CognitiveAppraisalSystem module which evaluates events
using cognitive appraisal theory to generate emotional responses.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.cognitive_appraisal import (
    CognitiveAppraisalSystem,
    AppraisalResult,
    CognitiveAppraisalConfig,
    APPRAISAL_EMOTION_MAPPING,
    get_appraisal_emotion_mapping,
)
from mm_orch.consciousness.motivation import MotivationSystem, GoalType
from mm_orch.consciousness.self_model import SelfModel


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def motivation_system():
    """Create a motivation system for testing."""
    return MotivationSystem()


@pytest.fixture
def self_model():
    """Create a self model for testing."""
    return SelfModel()


@pytest.fixture
def appraisal_system():
    """Create a basic appraisal system without dependencies."""
    return CognitiveAppraisalSystem()


@pytest.fixture
def full_appraisal_system(motivation_system, self_model):
    """Create an appraisal system with all dependencies."""
    return CognitiveAppraisalSystem(
        motivation_system=motivation_system,
        self_model=self_model,
    )


# =============================================================================
# AppraisalResult Tests
# =============================================================================

class TestAppraisalResult:
    """Tests for AppraisalResult dataclass."""

    def test_create_valid_appraisal_result(self):
        """Test creating a valid AppraisalResult."""
        result = AppraisalResult(
            relevance=0.8,
            goal_congruence=0.5,
            coping_potential=0.7,
            norm_compatibility=0.3,
        )
        assert result.relevance == 0.8
        assert result.goal_congruence == 0.5
        assert result.coping_potential == 0.7
        assert result.norm_compatibility == 0.3

    def test_appraisal_result_clamps_relevance(self):
        """Test that relevance is clamped to [0.0, 1.0]."""
        result = AppraisalResult(
            relevance=1.5,
            goal_congruence=0.0,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        assert result.relevance == 1.0
        
        result2 = AppraisalResult(
            relevance=-0.5,
            goal_congruence=0.0,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        assert result2.relevance == 0.0

    def test_appraisal_result_clamps_goal_congruence(self):
        """Test that goal_congruence is clamped to [-1.0, 1.0]."""
        result = AppraisalResult(
            relevance=0.5,
            goal_congruence=1.5,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        assert result.goal_congruence == 1.0
        
        result2 = AppraisalResult(
            relevance=0.5,
            goal_congruence=-1.5,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        assert result2.goal_congruence == -1.0

    def test_appraisal_result_clamps_coping_potential(self):
        """Test that coping_potential is clamped to [0.0, 1.0]."""
        result = AppraisalResult(
            relevance=0.5,
            goal_congruence=0.0,
            coping_potential=1.5,
            norm_compatibility=0.0,
        )
        assert result.coping_potential == 1.0

    def test_appraisal_result_clamps_norm_compatibility(self):
        """Test that norm_compatibility is clamped to [-1.0, 1.0]."""
        result = AppraisalResult(
            relevance=0.5,
            goal_congruence=0.0,
            coping_potential=0.5,
            norm_compatibility=1.5,
        )
        assert result.norm_compatibility == 1.0

    def test_appraisal_result_to_dict(self):
        """Test converting AppraisalResult to dictionary."""
        result = AppraisalResult(
            relevance=0.8,
            goal_congruence=0.5,
            coping_potential=0.7,
            norm_compatibility=0.3,
        )
        d = result.to_dict()
        assert d["relevance"] == 0.8
        assert d["goal_congruence"] == 0.5
        assert d["coping_potential"] == 0.7
        assert d["norm_compatibility"] == 0.3

    def test_appraisal_result_from_dict(self):
        """Test creating AppraisalResult from dictionary."""
        d = {
            "relevance": 0.8,
            "goal_congruence": 0.5,
            "coping_potential": 0.7,
            "norm_compatibility": 0.3,
        }
        result = AppraisalResult.from_dict(d)
        assert result.relevance == 0.8
        assert result.goal_congruence == 0.5
        assert result.coping_potential == 0.7
        assert result.norm_compatibility == 0.3


    def test_appraisal_result_to_pad_delta_positive(self):
        """Test PAD delta for positive appraisal."""
        result = AppraisalResult(
            relevance=1.0,
            goal_congruence=1.0,
            coping_potential=1.0,
            norm_compatibility=1.0,
        )
        delta = result.to_pad_delta()
        assert "pleasure_delta" in delta
        assert "arousal_delta" in delta
        assert "dominance_delta" in delta
        assert delta["pleasure_delta"] > 0  # Positive goal congruence -> positive pleasure
        assert delta["dominance_delta"] > 0  # High coping -> positive dominance

    def test_appraisal_result_to_pad_delta_negative(self):
        """Test PAD delta for negative appraisal."""
        result = AppraisalResult(
            relevance=1.0,
            goal_congruence=-1.0,
            coping_potential=0.0,
            norm_compatibility=-1.0,
        )
        delta = result.to_pad_delta()
        assert delta["pleasure_delta"] < 0  # Negative goal congruence -> negative pleasure
        assert delta["dominance_delta"] < 0  # Low coping -> negative dominance

    def test_appraisal_result_invalid_type(self):
        """Test that invalid types raise ValueError."""
        with pytest.raises(ValueError):
            AppraisalResult(
                relevance="invalid",
                goal_congruence=0.0,
                coping_potential=0.5,
                norm_compatibility=0.0,
            )


# =============================================================================
# CognitiveAppraisalConfig Tests
# =============================================================================

class TestCognitiveAppraisalConfig:
    """Tests for CognitiveAppraisalConfig dataclass."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = CognitiveAppraisalConfig()
        assert config.relevance_threshold == 0.3
        assert config.goal_importance_weight == 0.4
        assert config.capability_weight == 0.3
        assert config.history_weight == 0.3

    def test_create_custom_config(self):
        """Test creating custom configuration."""
        config = CognitiveAppraisalConfig(
            relevance_threshold=0.5,
            goal_importance_weight=0.5,
            capability_weight=0.25,
            history_weight=0.25,
        )
        assert config.relevance_threshold == 0.5
        assert config.goal_importance_weight == 0.5

    def test_config_invalid_relevance_threshold(self):
        """Test that invalid relevance_threshold raises ValueError."""
        with pytest.raises(ValueError):
            CognitiveAppraisalConfig(relevance_threshold=1.5)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CognitiveAppraisalConfig()
        d = config.to_dict()
        assert "relevance_threshold" in d
        assert "goal_importance_weight" in d

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        d = {"relevance_threshold": 0.5, "goal_importance_weight": 0.6}
        config = CognitiveAppraisalConfig.from_dict(d)
        assert config.relevance_threshold == 0.5
        assert config.goal_importance_weight == 0.6


# =============================================================================
# CognitiveAppraisalSystem Initialization Tests
# =============================================================================

class TestCognitiveAppraisalSystemInit:
    """Tests for CognitiveAppraisalSystem initialization."""

    def test_create_basic_system(self):
        """Test creating a basic appraisal system."""
        system = CognitiveAppraisalSystem()
        assert system is not None
        assert system._motivation_system is None
        assert system._self_model is None

    def test_create_system_with_dependencies(self, motivation_system, self_model):
        """Test creating system with dependencies."""
        system = CognitiveAppraisalSystem(
            motivation_system=motivation_system,
            self_model=self_model,
        )
        assert system._motivation_system is motivation_system
        assert system._self_model is self_model

    def test_create_system_with_config(self):
        """Test creating system with custom config."""
        config = {"relevance_threshold": 0.5}
        system = CognitiveAppraisalSystem(config=config)
        assert system._config.relevance_threshold == 0.5

    def test_initial_statistics(self, appraisal_system):
        """Test initial statistics."""
        stats = appraisal_system.get_statistics()
        assert stats["total_appraisals"] == 0
        assert stats["history_size"] == 0


# =============================================================================
# Appraise Event Tests
# =============================================================================

class TestAppraiseEvent:
    """Tests for appraise_event method."""

    def test_appraise_basic_event(self, appraisal_system):
        """Test appraising a basic event."""
        event = {"type": "task_complete", "success": True, "score": 0.8}
        result = appraisal_system.appraise_event(event)
        
        assert isinstance(result, AppraisalResult)
        assert 0.0 <= result.relevance <= 1.0
        assert -1.0 <= result.goal_congruence <= 1.0
        assert 0.0 <= result.coping_potential <= 1.0
        assert -1.0 <= result.norm_compatibility <= 1.0

    def test_appraise_increments_counter(self, appraisal_system):
        """Test that appraising increments the counter."""
        event = {"type": "task_complete"}
        appraisal_system.appraise_event(event)
        
        stats = appraisal_system.get_statistics()
        assert stats["total_appraisals"] == 1

    def test_appraise_records_history(self, appraisal_system):
        """Test that appraising records history."""
        event = {"type": "task_complete"}
        appraisal_system.appraise_event(event)
        
        history = appraisal_system.get_history()
        assert len(history) == 1
        assert history[0]["event_type"] == "task_complete"

    def test_appraise_with_context(self, appraisal_system):
        """Test appraising with context."""
        event = {"type": "task_complete"}
        context = {"urgent": True}
        result = appraisal_system.appraise_event(event, context)
        
        # Urgent context should increase relevance
        assert result.relevance > 0.5

    def test_appraise_unknown_event_type(self, appraisal_system):
        """Test appraising unknown event type."""
        event = {"type": "unknown_type"}
        result = appraisal_system.appraise_event(event)
        
        # Should still return valid result
        assert isinstance(result, AppraisalResult)


# =============================================================================
# Calculate Relevance Tests
# =============================================================================

class TestCalculateRelevance:
    """Tests for calculate_relevance method."""

    def test_task_complete_relevance(self, appraisal_system):
        """Test relevance for task_complete event."""
        event = {"type": "task_complete"}
        relevance = appraisal_system.calculate_relevance(event)
        assert relevance == pytest.approx(0.8, abs=0.1)

    def test_task_error_relevance(self, appraisal_system):
        """Test relevance for task_error event."""
        event = {"type": "task_error"}
        relevance = appraisal_system.calculate_relevance(event)
        assert relevance == pytest.approx(0.9, abs=0.1)

    def test_user_initiated_increases_relevance(self, appraisal_system):
        """Test that user_initiated flag increases relevance."""
        event1 = {"type": "system_event", "user_initiated": False}
        event2 = {"type": "system_event", "user_initiated": True}
        
        relevance1 = appraisal_system.calculate_relevance(event1)
        relevance2 = appraisal_system.calculate_relevance(event2)
        
        assert relevance2 > relevance1

    def test_high_importance_increases_relevance(self, appraisal_system):
        """Test that high importance increases relevance."""
        event1 = {"type": "system_event", "importance": 0.3}
        event2 = {"type": "system_event", "importance": 0.9}
        
        relevance1 = appraisal_system.calculate_relevance(event1)
        relevance2 = appraisal_system.calculate_relevance(event2)
        
        assert relevance2 > relevance1

    def test_urgent_context_increases_relevance(self, appraisal_system):
        """Test that urgent context increases relevance."""
        event = {"type": "system_event"}
        
        relevance1 = appraisal_system.calculate_relevance(event, {"urgent": False})
        relevance2 = appraisal_system.calculate_relevance(event, {"urgent": True})
        
        assert relevance2 > relevance1


# =============================================================================
# Calculate Goal Congruence Tests
# =============================================================================

class TestCalculateGoalCongruence:
    """Tests for calculate_goal_congruence method."""

    def test_successful_event_positive_congruence(self, appraisal_system):
        """Test that successful events have positive goal congruence."""
        event = {"type": "task_complete", "success": True}
        congruence = appraisal_system.calculate_goal_congruence(event)
        assert congruence > 0

    def test_failed_event_negative_congruence(self, appraisal_system):
        """Test that failed events have negative goal congruence."""
        event = {"type": "task_complete", "success": False}
        congruence = appraisal_system.calculate_goal_congruence(event)
        assert congruence < 0

    def test_task_error_negative_congruence(self, appraisal_system):
        """Test that task_error has negative goal congruence."""
        event = {"type": "task_error"}
        congruence = appraisal_system.calculate_goal_congruence(event)
        assert congruence < 0

    def test_high_score_positive_congruence(self, appraisal_system):
        """Test that high score leads to positive congruence."""
        event = {"type": "task_complete", "score": 0.9}
        congruence = appraisal_system.calculate_goal_congruence(event)
        assert congruence > 0

    def test_low_score_negative_congruence(self, appraisal_system):
        """Test that low score leads to negative congruence."""
        event = {"type": "task_complete", "score": 0.2}
        congruence = appraisal_system.calculate_goal_congruence(event)
        assert congruence < 0

    def test_user_feedback_sentiment(self, appraisal_system):
        """Test user feedback sentiment affects congruence."""
        positive_event = {"type": "user_feedback", "sentiment": 0.8}
        negative_event = {"type": "user_feedback", "sentiment": -0.8}
        
        positive_congruence = appraisal_system.calculate_goal_congruence(positive_event)
        negative_congruence = appraisal_system.calculate_goal_congruence(negative_event)
        
        assert positive_congruence > 0
        assert negative_congruence < 0


# =============================================================================
# Calculate Coping Potential Tests
# =============================================================================

class TestCalculateCopingPotential:
    """Tests for calculate_coping_potential method."""

    def test_basic_coping_potential(self, appraisal_system):
        """Test basic coping potential calculation."""
        event = {"type": "task_complete"}
        coping = appraisal_system.calculate_coping_potential(event)
        assert 0.0 <= coping <= 1.0

    def test_high_difficulty_reduces_coping(self, appraisal_system):
        """Test that high difficulty reduces coping potential."""
        easy_event = {"type": "task_complete", "difficulty": 0.2}
        hard_event = {"type": "task_complete", "difficulty": 0.9}
        
        easy_coping = appraisal_system.calculate_coping_potential(easy_event)
        hard_coping = appraisal_system.calculate_coping_potential(hard_event)
        
        assert easy_coping > hard_coping

    def test_resources_available_increases_coping(self, appraisal_system):
        """Test that available resources increase coping potential."""
        event = {"type": "task_complete"}
        
        coping_with = appraisal_system.calculate_coping_potential(
            event, {"resources_available": True}
        )
        coping_without = appraisal_system.calculate_coping_potential(
            event, {"resources_available": False}
        )
        
        assert coping_with > coping_without

    def test_coping_with_self_model(self, full_appraisal_system):
        """Test coping potential with self model."""
        event = {"type": "task_complete", "task_type": "search_qa"}
        coping = full_appraisal_system.calculate_coping_potential(event)
        
        # Should use capability performance score
        assert 0.0 <= coping <= 1.0


# =============================================================================
# Calculate Norm Compatibility Tests
# =============================================================================

class TestCalculateNormCompatibility:
    """Tests for calculate_norm_compatibility method."""

    def test_expected_success_compatible(self, appraisal_system):
        """Test that expected success is norm compatible."""
        event = {"type": "task_complete", "success": True}
        compatibility = appraisal_system.calculate_norm_compatibility(event)
        assert compatibility > 0

    def test_unexpected_failure_incompatible(self, appraisal_system):
        """Test that unexpected failure is norm incompatible."""
        event = {"type": "task_complete", "success": False}
        compatibility = appraisal_system.calculate_norm_compatibility(event)
        assert compatibility < 0

    def test_task_error_incompatible(self, appraisal_system):
        """Test that task_error is norm incompatible."""
        event = {"type": "task_error"}
        compatibility = appraisal_system.calculate_norm_compatibility(event)
        assert compatibility < 0

    def test_unknown_event_neutral(self, appraisal_system):
        """Test that unknown event type has neutral compatibility."""
        event = {"type": "unknown_type"}
        compatibility = appraisal_system.calculate_norm_compatibility(event)
        assert compatibility == 0.0


# =============================================================================
# Appraisal to Emotion Tests
# =============================================================================

class TestAppraisalToEmotion:
    """Tests for appraisal_to_emotion method."""

    def test_positive_appraisal_positive_emotion(self, appraisal_system):
        """Test that positive appraisal maps to positive emotion."""
        appraisal = AppraisalResult(
            relevance=0.8,
            goal_congruence=0.8,
            coping_potential=0.8,
            norm_compatibility=0.5,
        )
        emotion = appraisal_system.appraisal_to_emotion(appraisal)
        assert emotion in ["happy", "excited", "content", "proud", "hopeful"]

    def test_negative_appraisal_low_coping_fear_sadness(self, appraisal_system):
        """Test that negative appraisal with low coping maps to fear/sadness."""
        appraisal = AppraisalResult(
            relevance=0.8,
            goal_congruence=-0.8,
            coping_potential=0.2,
            norm_compatibility=-0.5,
        )
        emotion = appraisal_system.appraisal_to_emotion(appraisal)
        assert emotion in ["fearful", "sad", "depressed", "anxious"]

    def test_negative_appraisal_high_coping_anger(self, appraisal_system):
        """Test that negative appraisal with high coping maps to anger."""
        appraisal = AppraisalResult(
            relevance=0.8,
            goal_congruence=-0.8,
            coping_potential=0.9,
            norm_compatibility=-0.5,
        )
        emotion = appraisal_system.appraisal_to_emotion(appraisal)
        assert emotion in ["angry", "frustrated", "determined"]

    def test_low_relevance_neutral(self, appraisal_system):
        """Test that low relevance maps to neutral."""
        appraisal = AppraisalResult(
            relevance=0.1,
            goal_congruence=0.0,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        emotion = appraisal_system.appraisal_to_emotion(appraisal)
        assert emotion == "neutral"

    def test_emotion_is_from_mapping(self, appraisal_system):
        """Test that returned emotion is from the mapping."""
        appraisal = AppraisalResult(
            relevance=0.5,
            goal_congruence=0.3,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        emotion = appraisal_system.appraisal_to_emotion(appraisal)
        assert emotion in APPRAISAL_EMOTION_MAPPING


# =============================================================================
# Integration Tests with Dependencies
# =============================================================================

class TestIntegrationWithDependencies:
    """Tests for integration with motivation system and self model."""

    def test_goal_related_event_higher_relevance(self, full_appraisal_system, motivation_system):
        """Test that goal-related events have higher relevance."""
        # Create a goal related to accuracy
        motivation_system.create_goal(
            name="Improve Accuracy",
            description="Improve answer accuracy",
            goal_type=GoalType.SHORT_TERM,
            priority=0.9,
        )
        
        # Event related to accuracy
        related_event = {"type": "task_complete", "task_type": "search_qa"}
        unrelated_event = {"type": "system_event"}
        
        related_relevance = full_appraisal_system.calculate_relevance(related_event)
        unrelated_relevance = full_appraisal_system.calculate_relevance(unrelated_event)
        
        # Related event should have higher relevance
        assert related_relevance >= unrelated_relevance

    def test_capability_affects_coping(self, full_appraisal_system, self_model):
        """Test that capability performance affects coping potential."""
        # Record good performance for search_qa
        self_model.record_capability_usage("search_qa", True, 0.95)
        
        event = {"type": "task_complete", "task_type": "search_qa"}
        coping = full_appraisal_system.calculate_coping_potential(event)
        
        # Should have high coping due to good performance
        assert coping > 0.5


# =============================================================================
# Statistics and History Tests
# =============================================================================

class TestStatisticsAndHistory:
    """Tests for statistics and history methods."""

    def test_get_statistics(self, appraisal_system):
        """Test getting statistics."""
        stats = appraisal_system.get_statistics()
        assert "total_appraisals" in stats
        assert "history_size" in stats
        assert "uptime_seconds" in stats

    def test_get_history(self, appraisal_system):
        """Test getting history."""
        event = {"type": "task_complete"}
        appraisal_system.appraise_event(event)
        
        history = appraisal_system.get_history()
        assert len(history) == 1

    def test_get_history_with_limit(self, appraisal_system):
        """Test getting history with limit."""
        for i in range(5):
            appraisal_system.appraise_event({"type": "task_complete"})
        
        history = appraisal_system.get_history(limit=3)
        assert len(history) == 3

    def test_clear_history(self, appraisal_system):
        """Test clearing history."""
        appraisal_system.appraise_event({"type": "task_complete"})
        appraisal_system.clear_history()
        
        history = appraisal_system.get_history()
        assert len(history) == 0


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration methods."""

    def test_set_event_relevance(self, appraisal_system):
        """Test setting event relevance."""
        appraisal_system.set_event_relevance("custom_event", 0.9)
        
        event = {"type": "custom_event"}
        relevance = appraisal_system.calculate_relevance(event)
        assert relevance == pytest.approx(0.9, abs=0.1)

    def test_set_norm_expectation(self, appraisal_system):
        """Test setting norm expectation."""
        appraisal_system.set_norm_expectation(
            "custom_event",
            {"success": True, "score_min": 0.8}
        )
        
        event = {"type": "custom_event", "success": True, "score": 0.9}
        compatibility = appraisal_system.calculate_norm_compatibility(event)
        assert compatibility > 0


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for serialization methods."""

    def test_to_dict(self, appraisal_system):
        """Test converting to dictionary."""
        d = appraisal_system.to_dict()
        assert "config" in d
        assert "statistics" in d
        assert "event_relevance" in d
        assert "norm_expectations" in d

    def test_load_state(self, appraisal_system):
        """Test loading state from dictionary."""
        state = {
            "config": {"relevance_threshold": 0.5},
            "statistics": {"total_appraisals": 10},
            "event_relevance": {"custom": 0.9},
        }
        appraisal_system.load_state(state)
        
        assert appraisal_system._config.relevance_threshold == 0.5
        assert appraisal_system._total_appraisals == 10
        assert appraisal_system._event_relevance["custom"] == 0.9

    def test_serialization_roundtrip(self, appraisal_system):
        """Test serialization round-trip."""
        appraisal_system.appraise_event({"type": "task_complete"})
        
        state = appraisal_system.to_dict()
        
        new_system = CognitiveAppraisalSystem()
        new_system.load_state(state)
        
        assert new_system._total_appraisals == appraisal_system._total_appraisals


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_appraisal_emotion_mapping(self):
        """Test getting appraisal emotion mapping."""
        mapping = get_appraisal_emotion_mapping()
        assert isinstance(mapping, dict)
        assert "happy" in mapping
        assert "sad" in mapping
        assert "angry" in mapping
        assert "neutral" in mapping

    def test_mapping_has_valid_criteria(self):
        """Test that mapping has valid criteria."""
        mapping = get_appraisal_emotion_mapping()
        valid_dimensions = {"relevance", "goal_congruence", "coping_potential", "norm_compatibility"}
        
        for emotion, criteria in mapping.items():
            for dimension in criteria.keys():
                assert dimension in valid_dimensions, \
                    f"Invalid dimension '{dimension}' for emotion '{emotion}'"
