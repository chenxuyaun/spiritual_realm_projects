"""
Unit tests for the PAD Emotion Model module.

Tests PAD state creation, updates, decay, emotion detection,
and serialization functionality.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.pad_emotion import (
    PADEmotionModel,
    PADState,
    PADEmotionConfig,
    EMOTION_PAD_MAPPING,
    get_emotion_pad_mapping,
    get_pad_for_emotion,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def pad_model():
    """Create a PAD emotion model for testing."""
    return PADEmotionModel()


@pytest.fixture
def configured_model():
    """Create a PAD emotion model with custom config."""
    config = {
        "decay_rate": 0.9,
        "decay_interval": 30.0,
        "baseline_pleasure": 0.1,
        "baseline_arousal": 0.5,
        "baseline_dominance": 0.1,
    }
    return PADEmotionModel(config=config)


# =============================================================================
# Tests for PADState dataclass
# =============================================================================

class TestPADState:
    """Tests for the PADState dataclass."""

    def test_state_creation_with_valid_values(self):
        """Test creating a PAD state with valid values."""
        state = PADState(pleasure=0.5, arousal=0.6, dominance=0.3)
        assert state.pleasure == 0.5
        assert state.arousal == 0.6
        assert state.dominance == 0.3

    def test_state_pleasure_clamping_high(self):
        """Test that pleasure is clamped to 1.0 maximum."""
        state = PADState(pleasure=1.5, arousal=0.5, dominance=0.0)
        assert state.pleasure == 1.0

    def test_state_pleasure_clamping_low(self):
        """Test that pleasure is clamped to -1.0 minimum."""
        state = PADState(pleasure=-1.5, arousal=0.5, dominance=0.0)
        assert state.pleasure == -1.0

    def test_state_arousal_clamping_high(self):
        """Test that arousal is clamped to 1.0 maximum."""
        state = PADState(pleasure=0.0, arousal=1.5, dominance=0.0)
        assert state.arousal == 1.0

    def test_state_arousal_clamping_low(self):
        """Test that arousal is clamped to 0.0 minimum."""
        state = PADState(pleasure=0.0, arousal=-0.5, dominance=0.0)
        assert state.arousal == 0.0

    def test_state_dominance_clamping_high(self):
        """Test that dominance is clamped to 1.0 maximum."""
        state = PADState(pleasure=0.0, arousal=0.5, dominance=1.5)
        assert state.dominance == 1.0

    def test_state_dominance_clamping_low(self):
        """Test that dominance is clamped to -1.0 minimum."""
        state = PADState(pleasure=0.0, arousal=0.5, dominance=-1.5)
        assert state.dominance == -1.0

    def test_state_validation_invalid_pleasure(self):
        """Test that non-numeric pleasure raises ValueError."""
        with pytest.raises(ValueError, match="pleasure must be a number"):
            PADState(pleasure="invalid", arousal=0.5, dominance=0.0)

    def test_state_validation_invalid_arousal(self):
        """Test that non-numeric arousal raises ValueError."""
        with pytest.raises(ValueError, match="arousal must be a number"):
            PADState(pleasure=0.0, arousal="invalid", dominance=0.0)

    def test_state_validation_invalid_dominance(self):
        """Test that non-numeric dominance raises ValueError."""
        with pytest.raises(ValueError, match="dominance must be a number"):
            PADState(pleasure=0.0, arousal=0.5, dominance="invalid")

    def test_state_to_dict(self):
        """Test serializing a PAD state to dictionary."""
        state = PADState(pleasure=0.5, arousal=0.6, dominance=0.3)
        data = state.to_dict()
        assert data["pleasure"] == 0.5
        assert data["arousal"] == 0.6
        assert data["dominance"] == 0.3

    def test_state_from_dict(self):
        """Test deserializing a PAD state from dictionary."""
        data = {"pleasure": 0.5, "arousal": 0.6, "dominance": 0.3}
        state = PADState.from_dict(data)
        assert state.pleasure == 0.5
        assert state.arousal == 0.6
        assert state.dominance == 0.3

    def test_state_from_dict_with_defaults(self):
        """Test deserializing with missing fields uses defaults."""
        data = {}
        state = PADState.from_dict(data)
        assert state.pleasure == 0.0
        assert state.arousal == 0.4
        assert state.dominance == 0.0

    def test_state_distance_to(self):
        """Test calculating distance between two PAD states."""
        state1 = PADState(0.0, 0.0, 0.0)
        state2 = PADState(1.0, 0.0, 0.0)
        assert state1.distance_to(state2) == 1.0

    def test_state_distance_to_self(self):
        """Test that distance to self is zero."""
        state = PADState(0.5, 0.6, 0.3)
        assert state.distance_to(state) == 0.0

    def test_state_equality(self):
        """Test PAD state equality comparison."""
        state1 = PADState(0.5, 0.6, 0.3)
        state2 = PADState(0.5, 0.6, 0.3)
        assert state1 == state2

    def test_state_inequality(self):
        """Test PAD state inequality comparison."""
        state1 = PADState(0.5, 0.6, 0.3)
        state2 = PADState(0.5, 0.6, 0.4)
        assert state1 != state2

    def test_state_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = PADState(0.5, 0.6, 0.3)
        data = original.to_dict()
        restored = PADState.from_dict(data)
        assert restored == original



# =============================================================================
# Tests for PADEmotionConfig dataclass
# =============================================================================

class TestPADEmotionConfig:
    """Tests for the PADEmotionConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = PADEmotionConfig()
        assert config.decay_rate == 0.95
        assert config.decay_interval == 60.0
        assert config.baseline_pleasure == 0.0
        assert config.baseline_arousal == 0.4
        assert config.baseline_dominance == 0.0

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = PADEmotionConfig(
            decay_rate=0.9,
            decay_interval=30.0,
            baseline_pleasure=0.1,
            baseline_arousal=0.5,
            baseline_dominance=0.1,
        )
        assert config.decay_rate == 0.9
        assert config.decay_interval == 30.0
        assert config.baseline_pleasure == 0.1

    def test_config_validation_decay_rate_high(self):
        """Test that decay_rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be between"):
            PADEmotionConfig(decay_rate=1.5)

    def test_config_validation_decay_rate_low(self):
        """Test that decay_rate < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be between"):
            PADEmotionConfig(decay_rate=-0.1)

    def test_config_validation_decay_interval(self):
        """Test that non-positive decay_interval raises ValueError."""
        with pytest.raises(ValueError, match="decay_interval must be positive"):
            PADEmotionConfig(decay_interval=0)

    def test_config_validation_baseline_pleasure(self):
        """Test that invalid baseline_pleasure raises ValueError."""
        with pytest.raises(ValueError, match="baseline_pleasure must be between"):
            PADEmotionConfig(baseline_pleasure=1.5)

    def test_config_validation_baseline_arousal(self):
        """Test that invalid baseline_arousal raises ValueError."""
        with pytest.raises(ValueError, match="baseline_arousal must be between"):
            PADEmotionConfig(baseline_arousal=1.5)

    def test_config_validation_baseline_dominance(self):
        """Test that invalid baseline_dominance raises ValueError."""
        with pytest.raises(ValueError, match="baseline_dominance must be between"):
            PADEmotionConfig(baseline_dominance=1.5)

    def test_config_to_dict(self):
        """Test serializing config to dictionary."""
        config = PADEmotionConfig(decay_rate=0.9, decay_interval=30.0)
        data = config.to_dict()
        assert data["decay_rate"] == 0.9
        assert data["decay_interval"] == 30.0

    def test_config_from_dict(self):
        """Test deserializing config from dictionary."""
        data = {"decay_rate": 0.9, "decay_interval": 30.0}
        config = PADEmotionConfig.from_dict(data)
        assert config.decay_rate == 0.9
        assert config.decay_interval == 30.0

    def test_config_serialization_roundtrip(self):
        """Test that serialization and deserialization preserves data."""
        original = PADEmotionConfig(
            decay_rate=0.85,
            decay_interval=45.0,
            baseline_pleasure=0.1,
            baseline_arousal=0.5,
            baseline_dominance=0.1,
        )
        data = original.to_dict()
        restored = PADEmotionConfig.from_dict(data)
        assert restored.decay_rate == original.decay_rate
        assert restored.decay_interval == original.decay_interval


# =============================================================================
# Tests for PADEmotionModel initialization
# =============================================================================

class TestPADEmotionModelInit:
    """Tests for PADEmotionModel initialization."""

    def test_model_initialization_default(self):
        """Test default model initialization."""
        model = PADEmotionModel()
        state = model.get_state()
        assert state.pleasure == 0.0
        assert state.arousal == 0.4
        assert state.dominance == 0.0

    def test_model_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = {"baseline_pleasure": 0.2, "baseline_arousal": 0.5}
        model = PADEmotionModel(config=config)
        state = model.get_state()
        assert state.pleasure == 0.2
        assert state.arousal == 0.5

    def test_model_starts_at_baseline(self):
        """Test that model starts at baseline state."""
        config = {
            "baseline_pleasure": 0.1,
            "baseline_arousal": 0.5,
            "baseline_dominance": 0.1,
        }
        model = PADEmotionModel(config=config)
        state = model.get_state()
        assert state.pleasure == 0.1
        assert state.arousal == 0.5
        assert state.dominance == 0.1



# =============================================================================
# Tests for get_state() and set_state()
# =============================================================================

class TestGetSetState:
    """Tests for get_state() and set_state() methods."""

    def test_get_state_returns_copy(self, pad_model):
        """Test that get_state returns a copy, not the internal state."""
        state1 = pad_model.get_state()
        state2 = pad_model.get_state()
        assert state1 == state2
        assert state1 is not state2

    def test_set_state_updates_state(self, pad_model):
        """Test that set_state updates the internal state."""
        new_state = PADState(0.5, 0.6, 0.3)
        pad_model.set_state(new_state)
        current = pad_model.get_state()
        assert current == new_state

    def test_set_state_invalid_type(self, pad_model):
        """Test that set_state with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="state must be a PADState instance"):
            pad_model.set_state({"pleasure": 0.5, "arousal": 0.6, "dominance": 0.3})

    def test_set_state_clamps_values(self, pad_model):
        """Test that set_state clamps values to valid ranges."""
        new_state = PADState(1.5, 1.5, 1.5)  # Will be clamped
        pad_model.set_state(new_state)
        current = pad_model.get_state()
        assert current.pleasure == 1.0
        assert current.arousal == 1.0
        assert current.dominance == 1.0


# =============================================================================
# Tests for update_state()
# =============================================================================

class TestUpdateState:
    """Tests for the update_state() method."""

    def test_update_state_with_deltas(self, pad_model):
        """Test updating state with deltas."""
        initial = pad_model.get_state()
        pad_model.update_state(pleasure_delta=0.2, arousal_delta=0.1, dominance_delta=0.1)
        updated = pad_model.get_state()
        
        assert updated.pleasure == initial.pleasure + 0.2
        assert updated.arousal == initial.arousal + 0.1
        assert updated.dominance == initial.dominance + 0.1

    def test_update_state_clamps_values(self, pad_model):
        """Test that update_state clamps values to valid ranges."""
        pad_model.set_state(PADState(0.9, 0.9, 0.9))
        pad_model.update_state(pleasure_delta=0.5, arousal_delta=0.5, dominance_delta=0.5)
        updated = pad_model.get_state()
        
        assert updated.pleasure == 1.0
        assert updated.arousal == 1.0
        assert updated.dominance == 1.0

    def test_update_state_negative_deltas(self, pad_model):
        """Test updating state with negative deltas."""
        pad_model.set_state(PADState(0.5, 0.5, 0.5))
        pad_model.update_state(pleasure_delta=-0.3, arousal_delta=-0.2, dominance_delta=-0.1)
        updated = pad_model.get_state()
        
        assert abs(updated.pleasure - 0.2) < 1e-6
        assert abs(updated.arousal - 0.3) < 1e-6
        assert abs(updated.dominance - 0.4) < 1e-6

    def test_update_state_returns_new_state(self, pad_model):
        """Test that update_state returns the new state."""
        result = pad_model.update_state(pleasure_delta=0.1)
        current = pad_model.get_state()
        assert result == current

    def test_update_state_increments_counter(self, pad_model):
        """Test that update_state increments the update counter."""
        initial_stats = pad_model.get_statistics()
        initial_count = initial_stats["total_updates"]
        
        pad_model.update_state(pleasure_delta=0.1)
        pad_model.update_state(arousal_delta=0.1)
        
        final_stats = pad_model.get_statistics()
        assert final_stats["total_updates"] == initial_count + 2


# =============================================================================
# Tests for apply_decay()
# =============================================================================

class TestApplyDecay:
    """Tests for the apply_decay() method."""

    def test_apply_decay_moves_toward_baseline(self, pad_model):
        """Test that decay moves state toward baseline."""
        pad_model.set_state(PADState(0.8, 0.8, 0.8))
        baseline = PADState(0.0, 0.4, 0.0)
        
        pad_model.apply_decay(decay_rate=0.5)
        decayed = pad_model.get_state()
        
        # State should be closer to baseline
        assert decayed.pleasure < 0.8
        assert decayed.arousal < 0.8
        assert decayed.dominance < 0.8

    def test_apply_decay_with_custom_rate(self, pad_model):
        """Test decay with custom decay rate."""
        pad_model.set_state(PADState(1.0, 1.0, 1.0))
        
        pad_model.apply_decay(decay_rate=0.0)  # Instant decay to baseline
        decayed = pad_model.get_state()
        
        assert decayed.pleasure == 0.0
        assert decayed.arousal == 0.4
        assert decayed.dominance == 0.0

    def test_apply_decay_invalid_rate(self, pad_model):
        """Test that invalid decay rate raises ValueError."""
        with pytest.raises(ValueError, match="decay_rate must be between"):
            pad_model.apply_decay(decay_rate=1.5)

    def test_apply_decay_increments_counter(self, pad_model):
        """Test that apply_decay increments the decay counter."""
        initial_stats = pad_model.get_statistics()
        initial_count = initial_stats["total_decays"]
        
        pad_model.apply_decay()
        pad_model.apply_decay()
        
        final_stats = pad_model.get_statistics()
        assert final_stats["total_decays"] == initial_count + 2

    def test_repeated_decay_converges_to_baseline(self, pad_model):
        """Test that repeated decay converges to baseline."""
        pad_model.set_state(PADState(1.0, 1.0, 1.0))
        
        for _ in range(100):
            pad_model.apply_decay(decay_rate=0.9)
        
        final = pad_model.get_state()
        baseline = PADState(0.0, 0.4, 0.0)
        
        assert abs(final.pleasure - baseline.pleasure) < 0.01
        assert abs(final.arousal - baseline.arousal) < 0.01
        assert abs(final.dominance - baseline.dominance) < 0.01



# =============================================================================
# Tests for get_dominant_emotion()
# =============================================================================

class TestGetDominantEmotion:
    """Tests for the get_dominant_emotion() method."""

    def test_dominant_emotion_neutral(self, pad_model):
        """Test dominant emotion for neutral state."""
        pad_model.set_state(PADState(0.0, 0.4, 0.0))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "neutral"

    def test_dominant_emotion_happy(self, pad_model):
        """Test dominant emotion for happy state."""
        pad_model.set_state(PADState(0.7, 0.6, 0.5))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "happy"

    def test_dominant_emotion_sad(self, pad_model):
        """Test dominant emotion for sad state."""
        pad_model.set_state(PADState(-0.6, 0.3, -0.3))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "sad"

    def test_dominant_emotion_angry(self, pad_model):
        """Test dominant emotion for angry state."""
        pad_model.set_state(PADState(-0.5, 0.8, 0.6))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "angry"

    def test_dominant_emotion_fearful(self, pad_model):
        """Test dominant emotion for fearful state."""
        pad_model.set_state(PADState(-0.6, 0.7, -0.5))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "fearful"

    def test_dominant_emotion_excited(self, pad_model):
        """Test dominant emotion for excited state."""
        pad_model.set_state(PADState(0.6, 0.8, 0.6))
        emotion = pad_model.get_dominant_emotion()
        assert emotion == "excited"

    def test_dominant_emotion_returns_string(self, pad_model):
        """Test that dominant emotion is always a string."""
        pad_model.set_state(PADState(0.5, 0.5, 0.5))
        emotion = pad_model.get_dominant_emotion()
        assert isinstance(emotion, str)
        assert emotion in EMOTION_PAD_MAPPING


# =============================================================================
# Tests for get_emotion_intensity()
# =============================================================================

class TestGetEmotionIntensity:
    """Tests for the get_emotion_intensity() method."""

    def test_intensity_at_neutral(self, pad_model):
        """Test intensity at neutral state is zero."""
        pad_model.set_state(PADState(0.0, 0.4, 0.0))
        intensity = pad_model.get_emotion_intensity()
        assert intensity == 0.0

    def test_intensity_increases_with_distance(self, pad_model):
        """Test that intensity increases with distance from neutral."""
        pad_model.set_state(PADState(0.5, 0.5, 0.5))
        intensity1 = pad_model.get_emotion_intensity()
        
        pad_model.set_state(PADState(1.0, 1.0, 1.0))
        intensity2 = pad_model.get_emotion_intensity()
        
        assert intensity2 > intensity1

    def test_intensity_is_non_negative(self, pad_model):
        """Test that intensity is always non-negative."""
        pad_model.set_state(PADState(-1.0, 0.0, -1.0))
        intensity = pad_model.get_emotion_intensity()
        assert intensity >= 0.0


# =============================================================================
# Tests for map_to_valence_arousal()
# =============================================================================

class TestMapToValenceArousal:
    """Tests for the map_to_valence_arousal() method."""

    def test_map_returns_tuple(self, pad_model):
        """Test that map returns a tuple of two floats."""
        result = pad_model.map_to_valence_arousal()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_map_valence_equals_pleasure(self, pad_model):
        """Test that valence equals pleasure."""
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        valence, arousal = pad_model.map_to_valence_arousal()
        assert valence == 0.5

    def test_map_arousal_preserved(self, pad_model):
        """Test that arousal is preserved in mapping."""
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        valence, arousal = pad_model.map_to_valence_arousal()
        assert arousal == 0.6


# =============================================================================
# Tests for get_emotion_probabilities()
# =============================================================================

class TestGetEmotionProbabilities:
    """Tests for the get_emotion_probabilities() method."""

    def test_probabilities_sum_to_one(self, pad_model):
        """Test that emotion probabilities sum to 1.0."""
        probs = pad_model.get_emotion_probabilities()
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_probabilities_all_positive(self, pad_model):
        """Test that all probabilities are non-negative."""
        probs = pad_model.get_emotion_probabilities()
        for prob in probs.values():
            assert prob >= 0.0

    def test_probabilities_cover_all_emotions(self, pad_model):
        """Test that probabilities cover all mapped emotions."""
        probs = pad_model.get_emotion_probabilities()
        for emotion in EMOTION_PAD_MAPPING:
            assert emotion in probs

    def test_dominant_emotion_has_high_probability(self, pad_model):
        """Test that dominant emotion has relatively high probability."""
        pad_model.set_state(PADState(0.7, 0.6, 0.5))  # Happy
        probs = pad_model.get_emotion_probabilities()
        dominant = pad_model.get_dominant_emotion()
        
        # Dominant emotion should have one of the higher probabilities
        assert probs[dominant] > 0.0



# =============================================================================
# Tests for reset_to_baseline() and set_baseline()
# =============================================================================

class TestBaselineOperations:
    """Tests for baseline-related operations."""

    def test_reset_to_baseline(self, pad_model):
        """Test resetting state to baseline."""
        pad_model.set_state(PADState(0.8, 0.8, 0.8))
        pad_model.reset_to_baseline()
        state = pad_model.get_state()
        
        assert state.pleasure == 0.0
        assert state.arousal == 0.4
        assert state.dominance == 0.0

    def test_set_baseline(self, pad_model):
        """Test setting a new baseline."""
        new_baseline = PADState(0.2, 0.5, 0.1)
        pad_model.set_baseline(new_baseline)
        
        pad_model.set_state(PADState(0.8, 0.8, 0.8))
        pad_model.reset_to_baseline()
        state = pad_model.get_state()
        
        assert state.pleasure == 0.2
        assert state.arousal == 0.5
        assert state.dominance == 0.1

    def test_set_baseline_invalid_type(self, pad_model):
        """Test that set_baseline with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="baseline must be a PADState instance"):
            pad_model.set_baseline({"pleasure": 0.2})


# =============================================================================
# Tests for serialization
# =============================================================================

class TestSerialization:
    """Tests for serialization and deserialization."""

    def test_to_dict(self, pad_model):
        """Test serializing model to dictionary."""
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        data = pad_model.to_dict()
        
        assert "config" in data
        assert "state" in data
        assert "baseline" in data
        assert data["state"]["pleasure"] == 0.5

    def test_load_state(self, pad_model):
        """Test loading state from dictionary."""
        state_data = {
            "state": {"pleasure": 0.5, "arousal": 0.6, "dominance": 0.3},
            "baseline": {"pleasure": 0.1, "arousal": 0.5, "dominance": 0.1},
        }
        pad_model.load_state(state_data)
        state = pad_model.get_state()
        
        assert state.pleasure == 0.5
        assert state.arousal == 0.6
        assert state.dominance == 0.3

    def test_serialization_roundtrip(self, pad_model):
        """Test that serialization and deserialization preserves state."""
        pad_model.set_state(PADState(0.5, 0.6, 0.3))
        pad_model.update_state(pleasure_delta=0.1)
        
        data = pad_model.to_dict()
        
        new_model = PADEmotionModel()
        new_model.load_state(data)
        
        original_state = pad_model.get_state()
        restored_state = new_model.get_state()
        
        assert original_state == restored_state


# =============================================================================
# Tests for statistics and history
# =============================================================================

class TestStatisticsAndHistory:
    """Tests for statistics and history tracking."""

    def test_get_statistics(self, pad_model):
        """Test getting statistics."""
        stats = pad_model.get_statistics()
        
        assert "current_state" in stats
        assert "baseline" in stats
        assert "dominant_emotion" in stats
        assert "emotion_intensity" in stats
        assert "total_updates" in stats
        assert "total_decays" in stats

    def test_get_history(self, pad_model):
        """Test getting state change history."""
        pad_model.update_state(pleasure_delta=0.1)
        pad_model.update_state(arousal_delta=0.1)
        
        history = pad_model.get_history()
        assert len(history) >= 2

    def test_get_history_with_limit(self, pad_model):
        """Test getting limited history."""
        for _ in range(10):
            pad_model.update_state(pleasure_delta=0.01)
        
        history = pad_model.get_history(limit=5)
        assert len(history) == 5

    def test_clear_history(self, pad_model):
        """Test clearing history."""
        pad_model.update_state(pleasure_delta=0.1)
        pad_model.clear_history()
        
        history = pad_model.get_history()
        assert len(history) == 0


# =============================================================================
# Tests for helper functions
# =============================================================================

class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_emotion_pad_mapping(self):
        """Test getting the emotion PAD mapping."""
        mapping = get_emotion_pad_mapping()
        assert "happy" in mapping
        assert "sad" in mapping
        assert "neutral" in mapping
        assert isinstance(mapping["happy"], PADState)

    def test_get_pad_for_emotion_valid(self):
        """Test getting PAD for a valid emotion."""
        pad = get_pad_for_emotion("happy")
        assert pad is not None
        assert isinstance(pad, PADState)

    def test_get_pad_for_emotion_invalid(self):
        """Test getting PAD for an invalid emotion."""
        pad = get_pad_for_emotion("nonexistent")
        assert pad is None

    def test_get_pad_for_emotion_case_insensitive(self):
        """Test that emotion lookup is case insensitive."""
        pad1 = get_pad_for_emotion("Happy")
        pad2 = get_pad_for_emotion("HAPPY")
        pad3 = get_pad_for_emotion("happy")
        
        assert pad1 == pad2 == pad3


# =============================================================================
# Tests for is_emotional()
# =============================================================================

class TestIsEmotional:
    """Tests for the is_emotional() method."""

    def test_is_emotional_at_neutral(self, pad_model):
        """Test is_emotional at neutral state."""
        pad_model.set_state(PADState(0.0, 0.4, 0.0))
        assert pad_model.is_emotional() is False

    def test_is_emotional_with_strong_emotion(self, pad_model):
        """Test is_emotional with strong emotion."""
        pad_model.set_state(PADState(0.8, 0.8, 0.8))
        assert pad_model.is_emotional() is True
