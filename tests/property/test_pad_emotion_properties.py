"""
Property-based tests for PAD Emotion Model.

Tests properties 25-29 from the consciousness-system-deepening design document:
- Property 25: PAD State Bounds Validity
- Property 26: Emotion Label PAD Mapping Consistency
- Property 27: Event-Driven PAD Updates
- Property 28: Emotion Decay Toward Baseline
- Property 29: Dominant Emotion Nearest Neighbor

Validates: Requirements 6.1-6.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional, Tuple
import math

from mm_orch.consciousness.pad_emotion import (
    PADEmotionModel,
    PADState,
    PADEmotionConfig,
    EMOTION_PAD_MAPPING,
    get_emotion_pad_mapping,
    get_pad_for_emotion,
)


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for pleasure values [-1.0, 1.0]
pleasure_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for arousal values [0.0, 1.0]
arousal_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for dominance values [-1.0, 1.0]
dominance_strategy = st.floats(
    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


# Strategy for out-of-bounds pleasure values
out_of_bounds_pleasure_strategy = st.one_of(
    st.floats(min_value=-10.0, max_value=-1.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)

# Strategy for out-of-bounds arousal values
out_of_bounds_arousal_strategy = st.one_of(
    st.floats(min_value=-10.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)

# Strategy for out-of-bounds dominance values
out_of_bounds_dominance_strategy = st.one_of(
    st.floats(min_value=-10.0, max_value=-1.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False),
)

# Strategy for delta values (can be any reasonable float)
delta_strategy = st.floats(
    min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False
)

# Strategy for decay rates [0.0, 1.0]
decay_rate_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for emotion labels from the mapping
emotion_label_strategy = st.sampled_from(list(EMOTION_PAD_MAPPING.keys()))


@st.composite
def pad_state_strategy(draw):
    """Generate a valid PADState."""
    return PADState(
        pleasure=draw(pleasure_strategy),
        arousal=draw(arousal_strategy),
        dominance=draw(dominance_strategy),
    )


@st.composite
def pad_config_strategy(draw):
    """Generate a valid PADEmotionConfig."""
    baseline_pleasure = draw(pleasure_strategy)
    baseline_arousal = draw(arousal_strategy)
    baseline_dominance = draw(dominance_strategy)
    return PADEmotionConfig(
        decay_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        decay_interval=draw(st.floats(min_value=1.0, max_value=3600.0, allow_nan=False)),
        baseline_pleasure=baseline_pleasure,
        baseline_arousal=baseline_arousal,
        baseline_dominance=baseline_dominance,
        min_intensity_threshold=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


# =============================================================================
# Property 25: PAD State Bounds Validity
# =============================================================================

class TestPADStateBoundsValidity:
    """
    Tests for Property 25: PAD State Bounds Validity
    
    *For any* PADState, pleasure SHALL be in [-1.0, 1.0], arousal SHALL be 
    in [0.0, 1.0], and dominance SHALL be in [-1.0, 1.0].
    
    **Validates: Requirements 6.1**
    """

    @given(pleasure=pleasure_strategy, arousal=arousal_strategy, dominance=dominance_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valid_pad_state_creation(self, pleasure: float, arousal: float, dominance: float):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any valid input values, PADState SHALL be created with values
        within the valid bounds.
        
        **Validates: Requirements 6.1**
        """
        state = PADState(pleasure, arousal, dominance)
        
        assert -1.0 <= state.pleasure <= 1.0, \
            f"Pleasure should be in [-1.0, 1.0], got {state.pleasure}"
        assert 0.0 <= state.arousal <= 1.0, \
            f"Arousal should be in [0.0, 1.0], got {state.arousal}"
        assert -1.0 <= state.dominance <= 1.0, \
            f"Dominance should be in [-1.0, 1.0], got {state.dominance}"

    @given(pleasure=out_of_bounds_pleasure_strategy, arousal=arousal_strategy, dominance=dominance_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_out_of_bounds_pleasure_clamped(self, pleasure: float, arousal: float, dominance: float):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any out-of-bounds pleasure value, PADState SHALL clamp it to [-1.0, 1.0].
        
        **Validates: Requirements 6.1**
        """
        state = PADState(pleasure, arousal, dominance)
        
        assert -1.0 <= state.pleasure <= 1.0, \
            f"Pleasure should be clamped to [-1.0, 1.0], got {state.pleasure}"

    @given(pleasure=pleasure_strategy, arousal=out_of_bounds_arousal_strategy, dominance=dominance_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_out_of_bounds_arousal_clamped(self, pleasure: float, arousal: float, dominance: float):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any out-of-bounds arousal value, PADState SHALL clamp it to [0.0, 1.0].
        
        **Validates: Requirements 6.1**
        """
        state = PADState(pleasure, arousal, dominance)
        
        assert 0.0 <= state.arousal <= 1.0, \
            f"Arousal should be clamped to [0.0, 1.0], got {state.arousal}"


    @given(pleasure=pleasure_strategy, arousal=arousal_strategy, dominance=out_of_bounds_dominance_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_out_of_bounds_dominance_clamped(self, pleasure: float, arousal: float, dominance: float):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any out-of-bounds dominance value, PADState SHALL clamp it to [-1.0, 1.0].
        
        **Validates: Requirements 6.1**
        """
        state = PADState(pleasure, arousal, dominance)
        
        assert -1.0 <= state.dominance <= 1.0, \
            f"Dominance should be clamped to [-1.0, 1.0], got {state.dominance}"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pad_state_to_dict_preserves_bounds(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any PADState, to_dict() SHALL return values within valid bounds.
        
        **Validates: Requirements 6.1**
        """
        d = state.to_dict()
        
        assert -1.0 <= d["pleasure"] <= 1.0
        assert 0.0 <= d["arousal"] <= 1.0
        assert -1.0 <= d["dominance"] <= 1.0

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_pad_state_from_dict_round_trip(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any PADState, from_dict(to_dict()) SHALL produce an equivalent state.
        
        **Validates: Requirements 6.1**
        """
        d = state.to_dict()
        restored = PADState.from_dict(d)
        
        assert abs(state.pleasure - restored.pleasure) < 1e-6
        assert abs(state.arousal - restored.arousal) < 1e-6
        assert abs(state.dominance - restored.dominance) < 1e-6

    @given(
        pleasure_delta=delta_strategy,
        arousal_delta=delta_strategy,
        dominance_delta=delta_strategy,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_model_update_maintains_bounds(
        self, pleasure_delta: float, arousal_delta: float, dominance_delta: float
    ):
        """
        Feature: consciousness-system-deepening, Property 25: PAD State Bounds Validity
        
        For any update deltas applied to PADEmotionModel, the resulting state
        SHALL remain within valid bounds.
        
        **Validates: Requirements 6.1**
        """
        model = PADEmotionModel()
        
        new_state = model.update_state(
            pleasure_delta=pleasure_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
        )
        
        assert -1.0 <= new_state.pleasure <= 1.0, \
            f"Pleasure should be in [-1.0, 1.0] after update, got {new_state.pleasure}"
        assert 0.0 <= new_state.arousal <= 1.0, \
            f"Arousal should be in [0.0, 1.0] after update, got {new_state.arousal}"
        assert -1.0 <= new_state.dominance <= 1.0, \
            f"Dominance should be in [-1.0, 1.0] after update, got {new_state.dominance}"


# =============================================================================
# Property 26: Emotion Label PAD Mapping Consistency
# =============================================================================

class TestEmotionLabelPADMappingConsistency:
    """
    Tests for Property 26: Emotion Label PAD Mapping Consistency
    
    *For any* discrete emotion label in the mapping, querying the dominant 
    emotion for its PAD coordinates SHALL return that same label (or a 
    semantically equivalent one in the same PAD region).
    
    **Validates: Requirements 6.2**
    """

    @given(emotion=emotion_label_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_mapping_returns_same_or_similar_emotion(self, emotion: str):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        For any emotion label, setting the model to its PAD coordinates SHALL
        return that same label as the dominant emotion.
        
        **Validates: Requirements 6.2**
        """
        model = PADEmotionModel()
        
        # Get the PAD coordinates for this emotion
        pad_coords = EMOTION_PAD_MAPPING[emotion]
        
        # Set the model state to these coordinates
        model.set_state(pad_coords)
        
        # Get the dominant emotion
        dominant = model.get_dominant_emotion()
        
        # The dominant emotion should be the same as the input emotion
        # (since we set the state to exactly that emotion's coordinates)
        assert dominant == emotion, \
            f"Setting state to '{emotion}' coordinates should return '{emotion}', got '{dominant}'"

    @given(emotion=emotion_label_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_pad_mapping_has_valid_coordinates(self, emotion: str):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        For any emotion in the mapping, its PAD coordinates SHALL be within
        valid bounds.
        
        **Validates: Requirements 6.2**
        """
        pad_coords = EMOTION_PAD_MAPPING[emotion]
        
        assert -1.0 <= pad_coords.pleasure <= 1.0, \
            f"Emotion '{emotion}' pleasure should be in [-1.0, 1.0], got {pad_coords.pleasure}"
        assert 0.0 <= pad_coords.arousal <= 1.0, \
            f"Emotion '{emotion}' arousal should be in [0.0, 1.0], got {pad_coords.arousal}"
        assert -1.0 <= pad_coords.dominance <= 1.0, \
            f"Emotion '{emotion}' dominance should be in [-1.0, 1.0], got {pad_coords.dominance}"

    @given(emotion=emotion_label_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_get_pad_for_emotion_returns_correct_coordinates(self, emotion: str):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        For any emotion label, get_pad_for_emotion() SHALL return the same
        coordinates as in EMOTION_PAD_MAPPING.
        
        **Validates: Requirements 6.2**
        """
        expected = EMOTION_PAD_MAPPING[emotion]
        actual = get_pad_for_emotion(emotion)
        
        assert actual is not None, f"get_pad_for_emotion('{emotion}') should not return None"
        assert actual == expected, \
            f"get_pad_for_emotion('{emotion}') should match EMOTION_PAD_MAPPING"


    @given(emotion=emotion_label_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_mapping_case_insensitive(self, emotion: str):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        For any emotion label, get_pad_for_emotion() SHALL be case-insensitive.
        
        **Validates: Requirements 6.2**
        """
        lower_result = get_pad_for_emotion(emotion.lower())
        upper_result = get_pad_for_emotion(emotion.upper())
        mixed_result = get_pad_for_emotion(emotion.capitalize())
        
        assert lower_result is not None
        assert upper_result is not None
        assert mixed_result is not None
        assert lower_result == upper_result == mixed_result

    def test_all_emotions_in_mapping_are_unique(self):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        All emotion labels in the mapping SHALL be unique.
        
        **Validates: Requirements 6.2**
        """
        emotions = list(EMOTION_PAD_MAPPING.keys())
        assert len(emotions) == len(set(emotions)), \
            "All emotion labels should be unique"

    def test_emotion_mapping_contains_neutral(self):
        """
        Feature: consciousness-system-deepening, Property 26: Emotion Label PAD Mapping Consistency
        
        The emotion mapping SHALL contain a 'neutral' emotion.
        
        **Validates: Requirements 6.2**
        """
        assert "neutral" in EMOTION_PAD_MAPPING, \
            "Emotion mapping should contain 'neutral'"


# =============================================================================
# Property 27: Event-Driven PAD Updates
# =============================================================================

class TestEventDrivenPADUpdates:
    """
    Tests for Property 27: Event-Driven PAD Updates
    
    *For any* event processed by PADEmotionModel, the state SHALL change 
    according to the appraisal-derived deltas, and the resulting state 
    SHALL remain within valid bounds.
    
    **Validates: Requirements 6.3**
    """

    @given(
        pleasure_delta=delta_strategy,
        arousal_delta=delta_strategy,
        dominance_delta=delta_strategy,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_update_changes_state_by_deltas(
        self, pleasure_delta: float, arousal_delta: float, dominance_delta: float
    ):
        """
        Feature: consciousness-system-deepening, Property 27: Event-Driven PAD Updates
        
        For any update deltas, the state SHALL change according to the deltas
        (before clamping).
        
        **Validates: Requirements 6.3**
        """
        model = PADEmotionModel()
        initial_state = model.get_state()
        
        new_state = model.update_state(
            pleasure_delta=pleasure_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
        )
        
        # Calculate expected values (before clamping)
        expected_pleasure = initial_state.pleasure + pleasure_delta
        expected_arousal = initial_state.arousal + arousal_delta
        expected_dominance = initial_state.dominance + dominance_delta
        
        # Clamp expected values
        expected_pleasure = max(-1.0, min(1.0, expected_pleasure))
        expected_arousal = max(0.0, min(1.0, expected_arousal))
        expected_dominance = max(-1.0, min(1.0, expected_dominance))
        
        assert abs(new_state.pleasure - expected_pleasure) < 1e-6, \
            f"Pleasure should be {expected_pleasure}, got {new_state.pleasure}"
        assert abs(new_state.arousal - expected_arousal) < 1e-6, \
            f"Arousal should be {expected_arousal}, got {new_state.arousal}"
        assert abs(new_state.dominance - expected_dominance) < 1e-6, \
            f"Dominance should be {expected_dominance}, got {new_state.dominance}"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_updates_accumulate(self, data):
        """
        Feature: consciousness-system-deepening, Property 27: Event-Driven PAD Updates
        
        For any sequence of updates, the state SHALL accumulate the deltas
        (with clamping applied at each step).
        
        **Validates: Requirements 6.3**
        """
        model = PADEmotionModel()
        num_updates = data.draw(st.integers(min_value=1, max_value=10))
        
        for _ in range(num_updates):
            pleasure_delta = data.draw(delta_strategy)
            arousal_delta = data.draw(delta_strategy)
            dominance_delta = data.draw(delta_strategy)
            
            new_state = model.update_state(
                pleasure_delta=pleasure_delta,
                arousal_delta=arousal_delta,
                dominance_delta=dominance_delta,
            )
            
            # State should always be within bounds
            assert -1.0 <= new_state.pleasure <= 1.0
            assert 0.0 <= new_state.arousal <= 1.0
            assert -1.0 <= new_state.dominance <= 1.0

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_set_state_updates_model(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 27: Event-Driven PAD Updates
        
        For any valid PADState, set_state() SHALL update the model to that state.
        
        **Validates: Requirements 6.3**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        current = model.get_state()
        
        assert abs(current.pleasure - state.pleasure) < 1e-6
        assert abs(current.arousal - state.arousal) < 1e-6
        assert abs(current.dominance - state.dominance) < 1e-6

    @given(
        pleasure_delta=delta_strategy,
        arousal_delta=delta_strategy,
        dominance_delta=delta_strategy,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_update_increments_total_updates_counter(
        self, pleasure_delta: float, arousal_delta: float, dominance_delta: float
    ):
        """
        Feature: consciousness-system-deepening, Property 27: Event-Driven PAD Updates
        
        For any update, the total_updates counter SHALL increment by 1.
        
        **Validates: Requirements 6.3**
        """
        model = PADEmotionModel()
        initial_stats = model.get_statistics()
        initial_updates = initial_stats["total_updates"]
        
        model.update_state(
            pleasure_delta=pleasure_delta,
            arousal_delta=arousal_delta,
            dominance_delta=dominance_delta,
        )
        
        final_stats = model.get_statistics()
        final_updates = final_stats["total_updates"]
        
        assert final_updates == initial_updates + 1, \
            f"total_updates should increment by 1, got {final_updates - initial_updates}"


# =============================================================================
# Property 28: Emotion Decay Toward Baseline
# =============================================================================

class TestEmotionDecayTowardBaseline:
    """
    Tests for Property 28: Emotion Decay Toward Baseline
    
    *For any* non-baseline PAD state, applying decay SHALL move the state 
    closer to baseline, and repeated decay applications SHALL converge to baseline.
    
    **Validates: Requirements 6.4**
    """

    @given(state=pad_state_strategy(), decay_rate=decay_rate_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_decay_moves_toward_baseline(self, state: PADState, decay_rate: float):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For any non-baseline state, applying decay SHALL move the state
        closer to (or equal to) the baseline.
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        baseline = PADState(
            model._baseline.pleasure,
            model._baseline.arousal,
            model._baseline.dominance,
        )
        
        initial_distance = state.distance_to(baseline)
        
        model.apply_decay(decay_rate)
        new_state = model.get_state()
        new_distance = new_state.distance_to(baseline)
        
        # After decay, distance to baseline should be <= initial distance
        # (with small tolerance for floating point)
        assert new_distance <= initial_distance + 1e-6, \
            f"Distance to baseline should decrease or stay same after decay, " \
            f"was {initial_distance}, now {new_distance}"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_repeated_decay_converges_to_baseline(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For any non-baseline state, repeated decay applications SHALL
        converge to baseline.
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        baseline = PADState(
            model._baseline.pleasure,
            model._baseline.arousal,
            model._baseline.dominance,
        )
        
        # Apply decay many times
        for _ in range(100):
            model.apply_decay(0.9)  # Use a reasonable decay rate
        
        final_state = model.get_state()
        final_distance = final_state.distance_to(baseline)
        
        # After many decays, should be very close to baseline
        assert final_distance < 0.01, \
            f"After 100 decays, should be very close to baseline, distance is {final_distance}"


    @given(decay_rate=decay_rate_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_decay_at_baseline_stays_at_baseline(self, decay_rate: float):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For a state at baseline, applying decay SHALL keep the state at baseline.
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        # Model starts at baseline by default
        
        baseline = model.get_state()
        model.apply_decay(decay_rate)
        after_decay = model.get_state()
        
        assert abs(after_decay.pleasure - baseline.pleasure) < 1e-6
        assert abs(after_decay.arousal - baseline.arousal) < 1e-6
        assert abs(after_decay.dominance - baseline.dominance) < 1e-6

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_decay_rate_zero_keeps_state_unchanged(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For any state, applying decay with rate 0.0 SHALL keep the state
        at baseline (since decay_rate=0 means immediate jump to baseline).
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        model.apply_decay(0.0)
        after_decay = model.get_state()
        
        # With decay_rate=0, state should jump to baseline
        baseline = model._baseline
        assert abs(after_decay.pleasure - baseline.pleasure) < 1e-6
        assert abs(after_decay.arousal - baseline.arousal) < 1e-6
        assert abs(after_decay.dominance - baseline.dominance) < 1e-6

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_decay_rate_one_keeps_state_unchanged(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For any state, applying decay with rate 1.0 SHALL keep the state unchanged.
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        model.apply_decay(1.0)
        after_decay = model.get_state()
        
        # With decay_rate=1, state should stay the same
        assert abs(after_decay.pleasure - state.pleasure) < 1e-6
        assert abs(after_decay.arousal - state.arousal) < 1e-6
        assert abs(after_decay.dominance - state.dominance) < 1e-6

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_decay_increments_total_decays_counter(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 28: Emotion Decay Toward Baseline
        
        For any decay application, the total_decays counter SHALL increment by 1.
        
        **Validates: Requirements 6.4**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        initial_stats = model.get_statistics()
        initial_decays = initial_stats["total_decays"]
        
        model.apply_decay()
        
        final_stats = model.get_statistics()
        final_decays = final_stats["total_decays"]
        
        assert final_decays == initial_decays + 1, \
            f"total_decays should increment by 1, got {final_decays - initial_decays}"


# =============================================================================
# Property 29: Dominant Emotion Nearest Neighbor
# =============================================================================

class TestDominantEmotionNearestNeighbor:
    """
    Tests for Property 29: Dominant Emotion Nearest Neighbor
    
    *For any* PAD state, the dominant emotion label SHALL be the emotion 
    whose PAD coordinates are closest (by Euclidean distance) to the current state.
    
    **Validates: Requirements 6.5**
    """

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dominant_emotion_is_nearest_neighbor(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, the dominant emotion SHALL be the emotion with
        the smallest Euclidean distance to the current state.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        dominant = model.get_dominant_emotion()
        dominant_distance = state.distance_to(EMOTION_PAD_MAPPING[dominant])
        
        # Check that no other emotion is closer
        for emotion, coords in EMOTION_PAD_MAPPING.items():
            distance = state.distance_to(coords)
            assert distance >= dominant_distance - 1e-6, \
                f"Emotion '{emotion}' (distance {distance}) is closer than " \
                f"dominant '{dominant}' (distance {dominant_distance})"

    @given(emotion=emotion_label_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_exact_emotion_coordinates_return_that_emotion(self, emotion: str):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any emotion's exact PAD coordinates, the dominant emotion SHALL
        be that emotion.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        coords = EMOTION_PAD_MAPPING[emotion]
        model.set_state(coords)
        
        dominant = model.get_dominant_emotion()
        
        assert dominant == emotion, \
            f"At exact '{emotion}' coordinates, dominant should be '{emotion}', got '{dominant}'"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dominant_emotion_is_from_mapping(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, the dominant emotion SHALL be one of the emotions
        in EMOTION_PAD_MAPPING.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        dominant = model.get_dominant_emotion()
        
        assert dominant in EMOTION_PAD_MAPPING, \
            f"Dominant emotion '{dominant}' should be in EMOTION_PAD_MAPPING"


    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_intensity_is_distance_from_neutral(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, the emotion intensity SHALL be the Euclidean
        distance from the neutral emotion coordinates.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        intensity = model.get_emotion_intensity()
        neutral = EMOTION_PAD_MAPPING["neutral"]
        expected_intensity = state.distance_to(neutral)
        
        assert abs(intensity - expected_intensity) < 1e-6, \
            f"Intensity should be {expected_intensity}, got {intensity}"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_probabilities_sum_to_one(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, the emotion probabilities SHALL sum to 1.0.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        probabilities = model.get_emotion_probabilities()
        total = sum(probabilities.values())
        
        assert abs(total - 1.0) < 1e-6, \
            f"Emotion probabilities should sum to 1.0, got {total}"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_emotion_probabilities_all_non_negative(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, all emotion probabilities SHALL be non-negative.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        probabilities = model.get_emotion_probabilities()
        
        for emotion, prob in probabilities.items():
            assert prob >= 0.0, \
                f"Probability for '{emotion}' should be >= 0, got {prob}"

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dominant_emotion_has_highest_probability(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Property 29: Dominant Emotion Nearest Neighbor
        
        For any PAD state, the dominant emotion SHALL have the highest
        (or tied for highest) probability.
        
        **Validates: Requirements 6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        dominant = model.get_dominant_emotion()
        probabilities = model.get_emotion_probabilities()
        
        dominant_prob = probabilities[dominant]
        max_prob = max(probabilities.values())
        
        # Dominant emotion should have the highest probability (or tied)
        assert dominant_prob >= max_prob - 1e-6, \
            f"Dominant '{dominant}' probability {dominant_prob} should be >= max {max_prob}"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestPADEmotionModelIntegration:
    """
    Integration tests for PADEmotionModel combining multiple properties.
    """

    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_full_lifecycle_maintains_invariants(self, data):
        """
        Feature: consciousness-system-deepening, Properties 25-29 Integration
        
        For any sequence of operations, all PAD invariants SHALL be maintained.
        
        **Validates: Requirements 6.1-6.5**
        """
        model = PADEmotionModel()
        
        num_operations = data.draw(st.integers(min_value=5, max_value=20))
        
        for _ in range(num_operations):
            operation = data.draw(st.sampled_from(["update", "decay", "set"]))
            
            if operation == "update":
                model.update_state(
                    pleasure_delta=data.draw(delta_strategy),
                    arousal_delta=data.draw(delta_strategy),
                    dominance_delta=data.draw(delta_strategy),
                )
            elif operation == "decay":
                model.apply_decay(data.draw(decay_rate_strategy))
            else:  # set
                model.set_state(data.draw(pad_state_strategy()))
            
            # Check invariants after each operation
            state = model.get_state()
            
            # Property 25: Bounds validity
            assert -1.0 <= state.pleasure <= 1.0
            assert 0.0 <= state.arousal <= 1.0
            assert -1.0 <= state.dominance <= 1.0
            
            # Property 29: Dominant emotion is from mapping
            dominant = model.get_dominant_emotion()
            assert dominant in EMOTION_PAD_MAPPING
            
            # Property 29: Probabilities sum to 1
            probs = model.get_emotion_probabilities()
            assert abs(sum(probs.values()) - 1.0) < 1e-6


    @given(data=st.data())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_serialization_round_trip_preserves_state(self, data):
        """
        Feature: consciousness-system-deepening, Properties 25-29 Integration
        
        For any model state, serialization and deserialization SHALL preserve
        the state exactly.
        
        **Validates: Requirements 6.1-6.5**
        """
        model = PADEmotionModel()
        
        # Apply some operations
        num_operations = data.draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_operations):
            model.update_state(
                pleasure_delta=data.draw(delta_strategy),
                arousal_delta=data.draw(delta_strategy),
                dominance_delta=data.draw(delta_strategy),
            )
        
        # Serialize
        serialized = model.to_dict()
        
        # Create new model and load state
        new_model = PADEmotionModel()
        new_model.load_state(serialized)
        
        # Compare states
        original_state = model.get_state()
        restored_state = new_model.get_state()
        
        assert abs(original_state.pleasure - restored_state.pleasure) < 1e-6
        assert abs(original_state.arousal - restored_state.arousal) < 1e-6
        assert abs(original_state.dominance - restored_state.dominance) < 1e-6

    @given(state=pad_state_strategy())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_valence_arousal_mapping_consistency(self, state: PADState):
        """
        Feature: consciousness-system-deepening, Properties 25-29 Integration
        
        For any PAD state, the valence-arousal mapping SHALL return
        (pleasure, arousal) values.
        
        **Validates: Requirements 6.1-6.5**
        """
        model = PADEmotionModel()
        model.set_state(state)
        
        valence, arousal = model.map_to_valence_arousal()
        
        # Valence should equal pleasure
        assert abs(valence - state.pleasure) < 1e-6, \
            f"Valence should equal pleasure ({state.pleasure}), got {valence}"
        
        # Arousal should equal arousal
        assert abs(arousal - state.arousal) < 1e-6, \
            f"Arousal should equal arousal ({state.arousal}), got {arousal}"
