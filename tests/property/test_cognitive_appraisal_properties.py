"""
Property-based tests for Cognitive Appraisal System.

Tests properties 30-35 from the consciousness-system-deepening design document:
- Property 30: Appraisal Dimension Completeness
- Property 31: Goal-Based Appraisal Correlation
- Property 32: Appraisal-Emotion Mapping Determinism
- Property 33: Goal-Congruent Positive Emotion Scaling
- Property 34: Low-Coping Threat Response
- Property 35: High-Coping Activating Response

Validates: Requirements 7.1-7.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional
import math

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
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for relevance values [0.0, 1.0]
relevance_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for goal_congruence values [-1.0, 1.0]
goal_congruence_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for coping_potential values [0.0, 1.0]
coping_potential_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for norm_compatibility values [-1.0, 1.0]
norm_compatibility_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


# Strategy for event types
event_type_strategy = st.sampled_from([
    "task_complete", "task_error", "user_feedback", "user_message",
    "system_event", "workflow_start", "workflow_end", "unknown"
])

# Strategy for score values [0.0, 1.0]
score_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for sentiment values [-1.0, 1.0]
sentiment_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


@st.composite
def appraisal_result_strategy(draw):
    """Generate a valid AppraisalResult."""
    return AppraisalResult(
        relevance=draw(relevance_strategy),
        goal_congruence=draw(goal_congruence_strategy),
        coping_potential=draw(coping_potential_strategy),
        norm_compatibility=draw(norm_compatibility_strategy),
    )


@st.composite
def event_strategy(draw):
    """Generate a valid event dictionary."""
    event_type = draw(event_type_strategy)
    event = {"type": event_type}
    
    # Add type-specific fields
    if event_type == "task_complete":
        if draw(st.booleans()):
            event["score"] = draw(score_strategy)
        else:
            event["success"] = draw(st.booleans())
    elif event_type == "task_error":
        event["error_type"] = draw(st.sampled_from(["timeout", "validation", "runtime"]))
    elif event_type == "user_feedback":
        event["sentiment"] = draw(sentiment_strategy)
    
    # Optional fields
    if draw(st.booleans()):
        event["importance"] = draw(score_strategy)
    if draw(st.booleans()):
        event["user_initiated"] = draw(st.booleans())
    if draw(st.booleans()):
        event["difficulty"] = draw(score_strategy)
    
    return event


@st.composite
def context_strategy(draw):
    """Generate a valid context dictionary."""
    context = {}
    if draw(st.booleans()):
        context["urgent"] = draw(st.booleans())
    if draw(st.booleans()):
        context["resources_available"] = draw(st.booleans())
    return context


def create_appraisal_system():
    """Create a CognitiveAppraisalSystem for testing."""
    return CognitiveAppraisalSystem()


def create_full_appraisal_system():
    """Create a CognitiveAppraisalSystem with dependencies."""
    return CognitiveAppraisalSystem(
        motivation_system=MotivationSystem(),
        self_model=SelfModel(),
    )


# =============================================================================
# Property 30: Appraisal Dimension Completeness
# =============================================================================

class TestAppraisalDimensionCompleteness:
    """
    Tests for Property 30: Appraisal Dimension Completeness
    
    *For any* event appraised by CognitiveAppraisalSystem, the AppraisalResult 
    SHALL contain all four dimensions: relevance, goal_congruence, coping_potential, 
    and norm_compatibility.
    
    **Validates: Requirements 7.1**
    """

    @given(event=event_strategy(), context=context_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_appraisal_has_all_dimensions(self, event: Dict[str, Any], context: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Property 30: Appraisal Dimension Completeness
        
        For any event appraised, the result SHALL contain all four dimensions.
        
        **Validates: Requirements 7.1**
        """
        system = create_appraisal_system()
        result = system.appraise_event(event, context)
        
        assert hasattr(result, "relevance"), "Result should have relevance"
        assert hasattr(result, "goal_congruence"), "Result should have goal_congruence"
        assert hasattr(result, "coping_potential"), "Result should have coping_potential"
        assert hasattr(result, "norm_compatibility"), "Result should have norm_compatibility"

    @given(event=event_strategy(), context=context_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_appraisal_dimensions_in_valid_range(self, event: Dict[str, Any], context: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Property 30: Appraisal Dimension Completeness
        
        For any event appraised, all dimensions SHALL be in valid ranges.
        
        **Validates: Requirements 7.1**
        """
        system = create_appraisal_system()
        result = system.appraise_event(event, context)
        
        assert 0.0 <= result.relevance <= 1.0, \
            f"Relevance should be in [0.0, 1.0], got {result.relevance}"
        assert -1.0 <= result.goal_congruence <= 1.0, \
            f"Goal congruence should be in [-1.0, 1.0], got {result.goal_congruence}"
        assert 0.0 <= result.coping_potential <= 1.0, \
            f"Coping potential should be in [0.0, 1.0], got {result.coping_potential}"
        assert -1.0 <= result.norm_compatibility <= 1.0, \
            f"Norm compatibility should be in [-1.0, 1.0], got {result.norm_compatibility}"

    @given(event=event_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_appraisal_to_dict_has_all_dimensions(self, event: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Property 30: Appraisal Dimension Completeness
        
        For any appraisal result, to_dict() SHALL contain all four dimensions.
        
        **Validates: Requirements 7.1**
        """
        system = create_appraisal_system()
        result = system.appraise_event(event)
        d = result.to_dict()
        
        assert "relevance" in d
        assert "goal_congruence" in d
        assert "coping_potential" in d
        assert "norm_compatibility" in d


# =============================================================================
# Property 31: Goal-Based Appraisal Correlation
# =============================================================================

class TestGoalBasedAppraisalCorrelation:
    """
    Tests for Property 31: Goal-Based Appraisal Correlation
    
    *For any* event that directly affects an active goal, the goal_congruence 
    score SHALL be positive if the event helps the goal and negative if it hinders it.
    
    **Validates: Requirements 7.2**
    """

    @given(score=score_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_score_positive_congruence(self, score: float):
        """
        Feature: consciousness-system-deepening, Property 31: Goal-Based Appraisal Correlation
        
        For any task_complete event with high score (>0.5), goal_congruence SHALL be positive.
        
        **Validates: Requirements 7.2**
        """
        assume(score > 0.5)
        system = create_appraisal_system()
        event = {"type": "task_complete", "score": score}
        result = system.appraise_event(event)
        
        assert result.goal_congruence > 0, \
            f"High score ({score}) should result in positive goal_congruence, got {result.goal_congruence}"

    @given(score=score_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_low_score_negative_congruence(self, score: float):
        """
        Feature: consciousness-system-deepening, Property 31: Goal-Based Appraisal Correlation
        
        For any task_complete event with low score (<0.5), goal_congruence SHALL be negative.
        
        **Validates: Requirements 7.2**
        """
        assume(score < 0.5)
        system = create_appraisal_system()
        event = {"type": "task_complete", "score": score}
        result = system.appraise_event(event)
        
        assert result.goal_congruence < 0, \
            f"Low score ({score}) should result in negative goal_congruence, got {result.goal_congruence}"

    @given(sentiment=sentiment_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_positive_feedback_positive_congruence(self, sentiment: float):
        """
        Feature: consciousness-system-deepening, Property 31: Goal-Based Appraisal Correlation
        
        For any user_feedback with positive sentiment, goal_congruence SHALL be positive.
        
        **Validates: Requirements 7.2**
        """
        assume(sentiment > 0.1)
        system = create_appraisal_system()
        event = {"type": "user_feedback", "sentiment": sentiment}
        result = system.appraise_event(event)
        
        assert result.goal_congruence > 0, \
            f"Positive sentiment ({sentiment}) should result in positive goal_congruence"

    @given(sentiment=sentiment_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_negative_feedback_negative_congruence(self, sentiment: float):
        """
        Feature: consciousness-system-deepening, Property 31: Goal-Based Appraisal Correlation
        
        For any user_feedback with negative sentiment, goal_congruence SHALL be negative.
        
        **Validates: Requirements 7.2**
        """
        assume(sentiment < -0.1)
        system = create_appraisal_system()
        event = {"type": "user_feedback", "sentiment": sentiment}
        result = system.appraise_event(event)
        
        assert result.goal_congruence < 0, \
            f"Negative sentiment ({sentiment}) should result in negative goal_congruence"

    def test_task_error_always_negative_congruence(self):
        """
        Feature: consciousness-system-deepening, Property 31: Goal-Based Appraisal Correlation
        
        For any task_error event, goal_congruence SHALL be negative.
        
        **Validates: Requirements 7.2**
        """
        system = create_appraisal_system()
        event = {"type": "task_error"}
        result = system.appraise_event(event)
        
        assert result.goal_congruence < 0, \
            f"Task error should result in negative goal_congruence, got {result.goal_congruence}"


# =============================================================================
# Property 32: Appraisal-Emotion Mapping Determinism
# =============================================================================

class TestAppraisalEmotionMappingDeterminism:
    """
    Tests for Property 32: Appraisal-Emotion Mapping Determinism
    
    *For any* AppraisalResult, the mapped emotion SHALL be deterministic and 
    consistent with the appraisal pattern.
    
    **Validates: Requirements 7.3**
    """

    @given(appraisal=appraisal_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_mapping_is_deterministic(self, appraisal: AppraisalResult):
        """
        Feature: consciousness-system-deepening, Property 32: Appraisal-Emotion Mapping Determinism
        
        For any appraisal, calling appraisal_to_emotion multiple times SHALL
        return the same emotion.
        
        **Validates: Requirements 7.3**
        """
        system = create_appraisal_system()
        
        emotion1 = system.appraisal_to_emotion(appraisal)
        emotion2 = system.appraisal_to_emotion(appraisal)
        emotion3 = system.appraisal_to_emotion(appraisal)
        
        assert emotion1 == emotion2 == emotion3, \
            f"Emotion mapping should be deterministic, got {emotion1}, {emotion2}, {emotion3}"

    @given(appraisal=appraisal_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_is_from_mapping(self, appraisal: AppraisalResult):
        """
        Feature: consciousness-system-deepening, Property 32: Appraisal-Emotion Mapping Determinism
        
        For any appraisal, the mapped emotion SHALL be from APPRAISAL_EMOTION_MAPPING.
        
        **Validates: Requirements 7.3**
        """
        system = create_appraisal_system()
        emotion = system.appraisal_to_emotion(appraisal)
        
        assert emotion in APPRAISAL_EMOTION_MAPPING, \
            f"Emotion '{emotion}' should be in APPRAISAL_EMOTION_MAPPING"

    @given(
        relevance=st.floats(min_value=0.6, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        coping_potential=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_positive_appraisal_positive_emotion(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 32: Appraisal-Emotion Mapping Determinism
        
        For any highly positive appraisal, the emotion SHALL be positive.
        
        **Validates: Requirements 7.3**
        """
        system = create_appraisal_system()
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=0.5,
        )
        emotion = system.appraisal_to_emotion(appraisal)
        
        positive_emotions = {"happy", "excited", "content", "proud", "hopeful", "curious", "determined"}
        assert emotion in positive_emotions, \
            f"Positive appraisal should map to positive emotion, got '{emotion}'"


# =============================================================================
# Property 33: Goal-Congruent Positive Emotion Scaling
# =============================================================================

class TestGoalCongruentPositiveEmotionScaling:
    """
    Tests for Property 33: Goal-Congruent Positive Emotion Scaling
    
    *For any* goal-congruent event (goal_congruence > 0), the resulting pleasure 
    delta SHALL be positive and proportional to the goal's importance.
    
    **Validates: Requirements 7.4**
    """

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_positive_congruence_positive_pleasure_delta(
        self, relevance: float, goal_congruence: float
    ):
        """
        Feature: consciousness-system-deepening, Property 33: Goal-Congruent Positive Emotion Scaling
        
        For any goal-congruent appraisal, the pleasure delta SHALL be positive.
        
        **Validates: Requirements 7.4**
        """
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        delta = appraisal.to_pad_delta()
        
        assert delta["pleasure_delta"] > 0, \
            f"Positive goal_congruence ({goal_congruence}) should result in positive pleasure_delta"

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence1=st.floats(min_value=0.1, max_value=0.4, allow_nan=False),
        goal_congruence2=st.floats(min_value=0.6, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_higher_congruence_higher_pleasure_delta(
        self, relevance: float, goal_congruence1: float, goal_congruence2: float
    ):
        """
        Feature: consciousness-system-deepening, Property 33: Goal-Congruent Positive Emotion Scaling
        
        For any two goal-congruent appraisals, higher goal_congruence SHALL result
        in higher pleasure_delta.
        
        **Validates: Requirements 7.4**
        """
        appraisal1 = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence1,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        appraisal2 = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence2,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        
        delta1 = appraisal1.to_pad_delta()
        delta2 = appraisal2.to_pad_delta()
        
        assert delta2["pleasure_delta"] > delta1["pleasure_delta"], \
            f"Higher goal_congruence should result in higher pleasure_delta"

    @given(
        relevance1=st.floats(min_value=0.2, max_value=0.4, allow_nan=False),
        relevance2=st.floats(min_value=0.7, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_higher_relevance_higher_pleasure_delta(
        self, relevance1: float, relevance2: float, goal_congruence: float
    ):
        """
        Feature: consciousness-system-deepening, Property 33: Goal-Congruent Positive Emotion Scaling
        
        For any two goal-congruent appraisals, higher relevance SHALL result
        in higher pleasure_delta magnitude.
        
        **Validates: Requirements 7.4**
        """
        appraisal1 = AppraisalResult(
            relevance=relevance1,
            goal_congruence=goal_congruence,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        appraisal2 = AppraisalResult(
            relevance=relevance2,
            goal_congruence=goal_congruence,
            coping_potential=0.5,
            norm_compatibility=0.0,
        )
        
        delta1 = appraisal1.to_pad_delta()
        delta2 = appraisal2.to_pad_delta()
        
        assert delta2["pleasure_delta"] > delta1["pleasure_delta"], \
            f"Higher relevance should result in higher pleasure_delta"


# =============================================================================
# Property 34: Low-Coping Threat Response
# =============================================================================

class TestLowCopingThreatResponse:
    """
    Tests for Property 34: Low-Coping Threat Response
    
    *For any* goal-incongruent event with coping_potential < 0.3, the resulting 
    emotion SHALL be in the fear/sadness category (negative pleasure, variable 
    arousal, negative dominance).
    
    **Validates: Requirements 7.5**
    """

    @given(
        relevance=st.floats(min_value=0.6, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.5, allow_nan=False),
        coping_potential=st.floats(min_value=0.0, max_value=0.25, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_low_coping_threat_emotion(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 34: Low-Coping Threat Response
        
        For any goal-incongruent event with low coping potential, the emotion
        SHALL be in the fear/sadness category.
        
        **Validates: Requirements 7.5**
        """
        system = create_appraisal_system()
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=-0.5,
        )
        emotion = system.appraisal_to_emotion(appraisal)
        
        threat_emotions = {"fearful", "sad", "depressed", "anxious"}
        assert emotion in threat_emotions, \
            f"Low coping threat should map to fear/sadness, got '{emotion}'"

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.3, allow_nan=False),
        coping_potential=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_low_coping_negative_dominance_delta(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 34: Low-Coping Threat Response
        
        For any low coping appraisal, the dominance delta SHALL be negative.
        
        **Validates: Requirements 7.5**
        """
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=0.0,
        )
        delta = appraisal.to_pad_delta()
        
        assert delta["dominance_delta"] < 0, \
            f"Low coping ({coping_potential}) should result in negative dominance_delta"

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.3, allow_nan=False),
        coping_potential=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_low_coping_negative_pleasure_delta(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 34: Low-Coping Threat Response
        
        For any goal-incongruent low coping appraisal, the pleasure delta SHALL be negative.
        
        **Validates: Requirements 7.5**
        """
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=0.0,
        )
        delta = appraisal.to_pad_delta()
        
        assert delta["pleasure_delta"] < 0, \
            f"Goal-incongruent appraisal should result in negative pleasure_delta"


# =============================================================================
# Property 35: High-Coping Activating Response
# =============================================================================

class TestHighCopingActivatingResponse:
    """
    Tests for Property 35: High-Coping Activating Response
    
    *For any* goal-incongruent event with coping_potential > 0.7, the resulting 
    emotion SHALL be in the anger/determination category (negative pleasure, 
    high arousal, positive dominance).
    
    **Validates: Requirements 7.6**
    """

    @given(
        relevance=st.floats(min_value=0.6, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.5, allow_nan=False),
        coping_potential=st.floats(min_value=0.75, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_coping_activating_emotion(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 35: High-Coping Activating Response
        
        For any goal-incongruent event with high coping potential, the emotion
        SHALL be in the anger/determination category.
        
        **Validates: Requirements 7.6**
        """
        system = create_appraisal_system()
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=-0.5,
        )
        emotion = system.appraisal_to_emotion(appraisal)
        
        activating_emotions = {"angry", "frustrated", "determined"}
        assert emotion in activating_emotions, \
            f"High coping threat should map to anger/determination, got '{emotion}'"

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.3, allow_nan=False),
        coping_potential=st.floats(min_value=0.7, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_coping_positive_dominance_delta(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 35: High-Coping Activating Response
        
        For any high coping appraisal, the dominance delta SHALL be positive.
        
        **Validates: Requirements 7.6**
        """
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=0.0,
        )
        delta = appraisal.to_pad_delta()
        
        assert delta["dominance_delta"] > 0, \
            f"High coping ({coping_potential}) should result in positive dominance_delta"

    @given(
        relevance=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        goal_congruence=st.floats(min_value=-1.0, max_value=-0.5, allow_nan=False),
        coping_potential=st.floats(min_value=0.7, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_high_coping_high_arousal_delta(
        self, relevance: float, goal_congruence: float, coping_potential: float
    ):
        """
        Feature: consciousness-system-deepening, Property 35: High-Coping Activating Response
        
        For any high coping goal-incongruent appraisal, the arousal delta SHALL be positive.
        
        **Validates: Requirements 7.6**
        """
        appraisal = AppraisalResult(
            relevance=relevance,
            goal_congruence=goal_congruence,
            coping_potential=coping_potential,
            norm_compatibility=0.0,
        )
        delta = appraisal.to_pad_delta()
        
        # Arousal is driven by relevance and intensity of goal congruence
        assert delta["arousal_delta"] > 0, \
            f"High relevance goal-incongruent appraisal should result in positive arousal_delta"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestCognitiveAppraisalIntegration:
    """
    Integration tests for CognitiveAppraisalSystem combining multiple properties.
    """

    @given(event=event_strategy(), context=context_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_full_appraisal_lifecycle(self, event: Dict[str, Any], context: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Properties 30-35 Integration
        
        For any event, the full appraisal lifecycle SHALL maintain all invariants.
        
        **Validates: Requirements 7.1-7.6**
        """
        system = create_full_appraisal_system()
        
        # Appraise the event
        result = system.appraise_event(event, context)
        
        # Property 30: All dimensions present and valid
        assert 0.0 <= result.relevance <= 1.0
        assert -1.0 <= result.goal_congruence <= 1.0
        assert 0.0 <= result.coping_potential <= 1.0
        assert -1.0 <= result.norm_compatibility <= 1.0
        
        # Property 32: Emotion mapping is deterministic
        emotion1 = system.appraisal_to_emotion(result)
        emotion2 = system.appraisal_to_emotion(result)
        assert emotion1 == emotion2
        assert emotion1 in APPRAISAL_EMOTION_MAPPING
        
        # PAD delta is valid
        delta = result.to_pad_delta()
        assert "pleasure_delta" in delta
        assert "arousal_delta" in delta
        assert "dominance_delta" in delta

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_appraisal_history_consistency(self, data):
        """
        Feature: consciousness-system-deepening, Properties 30-35 Integration
        
        For any sequence of appraisals, the history SHALL be consistent.
        
        **Validates: Requirements 7.1-7.6**
        """
        system = create_appraisal_system()
        num_events = data.draw(st.integers(min_value=1, max_value=10))
        
        for _ in range(num_events):
            event = data.draw(event_strategy())
            system.appraise_event(event)
        
        stats = system.get_statistics()
        history = system.get_history()
        
        assert stats["total_appraisals"] == num_events
        assert len(history) == num_events

    @given(appraisal=appraisal_result_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_appraisal_result_serialization_roundtrip(self, appraisal: AppraisalResult):
        """
        Feature: consciousness-system-deepening, Properties 30-35 Integration
        
        For any AppraisalResult, serialization round-trip SHALL preserve values.
        
        **Validates: Requirements 7.1-7.6**
        """
        d = appraisal.to_dict()
        restored = AppraisalResult.from_dict(d)
        
        assert abs(appraisal.relevance - restored.relevance) < 1e-6
        assert abs(appraisal.goal_congruence - restored.goal_congruence) < 1e-6
        assert abs(appraisal.coping_potential - restored.coping_potential) < 1e-6
        assert abs(appraisal.norm_compatibility - restored.norm_compatibility) < 1e-6
