"""
Unit tests for Intrinsic Motivation Engine.

Tests specific examples and edge cases for:
- IntrinsicMotivationConfig creation and validation
- FamiliarityEntry and ActionExplorationInfo dataclasses
- IntrinsicMotivationEngine initialization
- calculate_curiosity_reward() with various prediction error types
- get_novelty_score() for new and familiar stimuli
- update_familiarity() and familiarity tracking
- decay_curiosity() behavior
- get_intrinsic_reward() for novel and familiar stimuli
- get_exploration_bonus() for explored and unexplored actions
- select_action_with_exploration() with different known rewards
- Serialization/deserialization (to_dict/from_dict)

Requirements: 2.1-2.5
"""

import pytest
import time
from typing import Dict, Any

from mm_orch.consciousness.intrinsic_motivation import (
    IntrinsicMotivationEngine,
    IntrinsicMotivationConfig,
    FamiliarityEntry,
    ActionExplorationInfo,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_engine() -> IntrinsicMotivationEngine:
    """Create an IntrinsicMotivationEngine with default configuration."""
    return IntrinsicMotivationEngine()


@pytest.fixture
def custom_config() -> IntrinsicMotivationConfig:
    """Create a custom IntrinsicMotivationConfig for testing."""
    return IntrinsicMotivationConfig(
        base_curiosity_reward=0.4,
        prediction_error_weight=0.8,
        novelty_threshold=0.6,
        familiarity_decay_rate=0.15,
        curiosity_decay_rate=0.2,
        exploration_weight=0.4,
        max_familiarity_entries=5000,
        information_gain_scale=1.5,
    )


@pytest.fixture
def custom_engine(custom_config: IntrinsicMotivationConfig) -> IntrinsicMotivationEngine:
    """Create an IntrinsicMotivationEngine with custom configuration."""
    return IntrinsicMotivationEngine(config=custom_config.to_dict())


@pytest.fixture
def sample_context() -> Dict[str, Any]:
    """Create a sample context dictionary for testing."""
    return {
        "task_type": "reasoning",
        "difficulty": 0.5,
        "user_id": "test_user",
    }


@pytest.fixture
def sample_stimulus() -> str:
    """Create a sample stimulus for testing."""
    return "test_stimulus_content"


# =============================================================================
# Tests for IntrinsicMotivationConfig Dataclass
# =============================================================================

class TestIntrinsicMotivationConfigDataclass:
    """Unit tests for IntrinsicMotivationConfig dataclass creation and validation."""
    
    def test_default_config_creation(self):
        """Test creating IntrinsicMotivationConfig with default values."""
        config = IntrinsicMotivationConfig()
        
        assert config.base_curiosity_reward == 0.5
        assert config.prediction_error_weight == 0.7
        assert config.novelty_threshold == 0.5
        assert config.familiarity_decay_rate == 0.1
        assert config.curiosity_decay_rate == 0.15
        assert config.exploration_weight == 0.3
        assert config.max_familiarity_entries == 10000
        assert config.information_gain_scale == 1.0
    
    def test_custom_config_creation(self, custom_config: IntrinsicMotivationConfig):
        """Test creating IntrinsicMotivationConfig with custom values."""
        assert custom_config.base_curiosity_reward == 0.4
        assert custom_config.prediction_error_weight == 0.8
        assert custom_config.novelty_threshold == 0.6
        assert custom_config.familiarity_decay_rate == 0.15
        assert custom_config.curiosity_decay_rate == 0.2
        assert custom_config.exploration_weight == 0.4
        assert custom_config.max_familiarity_entries == 5000
        assert custom_config.information_gain_scale == 1.5
    
    def test_config_boundary_values(self):
        """Test IntrinsicMotivationConfig with boundary values."""
        # Minimum valid values
        min_config = IntrinsicMotivationConfig(
            base_curiosity_reward=0.0,
            prediction_error_weight=0.0,
            novelty_threshold=0.0,
            familiarity_decay_rate=0.0,
            curiosity_decay_rate=0.0,
            exploration_weight=0.0,
            max_familiarity_entries=1,
            information_gain_scale=0.001,
        )
        assert min_config.base_curiosity_reward == 0.0
        
        # Maximum valid values
        max_config = IntrinsicMotivationConfig(
            base_curiosity_reward=1.0,
            prediction_error_weight=1.0,
            novelty_threshold=1.0,
            familiarity_decay_rate=1.0,
            curiosity_decay_rate=1.0,
            exploration_weight=1.0,
        )
        assert max_config.base_curiosity_reward == 1.0

    def test_config_invalid_base_curiosity_reward_raises_error(self):
        """Test that invalid base_curiosity_reward values raise ValueError."""
        with pytest.raises(ValueError, match="base_curiosity_reward must be between"):
            IntrinsicMotivationConfig(base_curiosity_reward=1.5)
        
        with pytest.raises(ValueError, match="base_curiosity_reward must be between"):
            IntrinsicMotivationConfig(base_curiosity_reward=-0.1)
    
    def test_config_invalid_prediction_error_weight_raises_error(self):
        """Test that invalid prediction_error_weight values raise ValueError."""
        with pytest.raises(ValueError, match="prediction_error_weight must be between"):
            IntrinsicMotivationConfig(prediction_error_weight=1.2)
        
        with pytest.raises(ValueError, match="prediction_error_weight must be between"):
            IntrinsicMotivationConfig(prediction_error_weight=-0.5)
    
    def test_config_invalid_novelty_threshold_raises_error(self):
        """Test that invalid novelty_threshold values raise ValueError."""
        with pytest.raises(ValueError, match="novelty_threshold must be between"):
            IntrinsicMotivationConfig(novelty_threshold=2.0)
    
    def test_config_invalid_familiarity_decay_rate_raises_error(self):
        """Test that invalid familiarity_decay_rate values raise ValueError."""
        with pytest.raises(ValueError, match="familiarity_decay_rate must be between"):
            IntrinsicMotivationConfig(familiarity_decay_rate=-0.1)
    
    def test_config_invalid_curiosity_decay_rate_raises_error(self):
        """Test that invalid curiosity_decay_rate values raise ValueError."""
        with pytest.raises(ValueError, match="curiosity_decay_rate must be between"):
            IntrinsicMotivationConfig(curiosity_decay_rate=1.5)
    
    def test_config_invalid_exploration_weight_raises_error(self):
        """Test that invalid exploration_weight values raise ValueError."""
        with pytest.raises(ValueError, match="exploration_weight must be between"):
            IntrinsicMotivationConfig(exploration_weight=-0.2)
    
    def test_config_invalid_max_familiarity_entries_raises_error(self):
        """Test that invalid max_familiarity_entries values raise ValueError."""
        with pytest.raises(ValueError, match="max_familiarity_entries must be at least 1"):
            IntrinsicMotivationConfig(max_familiarity_entries=0)
        
        with pytest.raises(ValueError, match="max_familiarity_entries must be at least 1"):
            IntrinsicMotivationConfig(max_familiarity_entries=-10)
    
    def test_config_invalid_information_gain_scale_raises_error(self):
        """Test that invalid information_gain_scale values raise ValueError."""
        with pytest.raises(ValueError, match="information_gain_scale must be positive"):
            IntrinsicMotivationConfig(information_gain_scale=0)
        
        with pytest.raises(ValueError, match="information_gain_scale must be positive"):
            IntrinsicMotivationConfig(information_gain_scale=-1.0)
    
    def test_config_to_dict(self, custom_config: IntrinsicMotivationConfig):
        """Test IntrinsicMotivationConfig serialization to dictionary."""
        data = custom_config.to_dict()
        
        assert data["base_curiosity_reward"] == 0.4
        assert data["prediction_error_weight"] == 0.8
        assert data["novelty_threshold"] == 0.6
        assert data["familiarity_decay_rate"] == 0.15
        assert data["curiosity_decay_rate"] == 0.2
        assert data["exploration_weight"] == 0.4
        assert data["max_familiarity_entries"] == 5000
        assert data["information_gain_scale"] == 1.5
    
    def test_config_from_dict(self):
        """Test IntrinsicMotivationConfig deserialization from dictionary."""
        data = {
            "base_curiosity_reward": 0.6,
            "prediction_error_weight": 0.5,
            "novelty_threshold": 0.4,
            "familiarity_decay_rate": 0.2,
            "curiosity_decay_rate": 0.1,
            "exploration_weight": 0.5,
            "max_familiarity_entries": 8000,
            "information_gain_scale": 2.0,
        }
        
        config = IntrinsicMotivationConfig.from_dict(data)
        
        assert config.base_curiosity_reward == 0.6
        assert config.prediction_error_weight == 0.5
        assert config.novelty_threshold == 0.4
        assert config.familiarity_decay_rate == 0.2
        assert config.curiosity_decay_rate == 0.1
        assert config.exploration_weight == 0.5
        assert config.max_familiarity_entries == 8000
        assert config.information_gain_scale == 2.0
    
    def test_config_from_dict_with_defaults(self):
        """Test IntrinsicMotivationConfig deserialization with missing keys uses defaults."""
        data = {"base_curiosity_reward": 0.3}
        
        config = IntrinsicMotivationConfig.from_dict(data)
        
        assert config.base_curiosity_reward == 0.3
        assert config.prediction_error_weight == 0.7  # Default
        assert config.novelty_threshold == 0.5  # Default


# =============================================================================
# Tests for FamiliarityEntry Dataclass
# =============================================================================

class TestFamiliarityEntryDataclass:
    """Unit tests for FamiliarityEntry dataclass creation and validation."""
    
    def test_valid_familiarity_entry_creation(self):
        """Test creating a valid FamiliarityEntry object."""
        entry = FamiliarityEntry(
            stimulus_hash="abc123def456",
            encounter_count=5,
            familiarity_score=0.7,
            curiosity_level=0.3,
            first_encountered=1000.0,
            last_encountered=2000.0,
            cumulative_reward=1.5,
        )
        
        assert entry.stimulus_hash == "abc123def456"
        assert entry.encounter_count == 5
        assert entry.familiarity_score == 0.7
        assert entry.curiosity_level == 0.3
        assert entry.first_encountered == 1000.0
        assert entry.last_encountered == 2000.0
        assert entry.cumulative_reward == 1.5
    
    def test_familiarity_entry_default_values(self):
        """Test FamiliarityEntry creation with default values."""
        entry = FamiliarityEntry(stimulus_hash="test_hash")
        
        assert entry.stimulus_hash == "test_hash"
        assert entry.encounter_count == 1
        assert entry.familiarity_score == 0.0
        assert entry.curiosity_level == 1.0
        assert entry.cumulative_reward == 0.0
        # first_encountered and last_encountered should be set to current time
        assert entry.first_encountered > 0
        assert entry.last_encountered > 0
    
    def test_familiarity_entry_to_dict(self):
        """Test FamiliarityEntry serialization to dictionary."""
        entry = FamiliarityEntry(
            stimulus_hash="hash123",
            encounter_count=3,
            familiarity_score=0.5,
            curiosity_level=0.5,
            first_encountered=100.0,
            last_encountered=200.0,
            cumulative_reward=0.8,
        )
        
        data = entry.to_dict()
        
        assert data["stimulus_hash"] == "hash123"
        assert data["encounter_count"] == 3
        assert data["familiarity_score"] == 0.5
        assert data["curiosity_level"] == 0.5
        assert data["first_encountered"] == 100.0
        assert data["last_encountered"] == 200.0
        assert data["cumulative_reward"] == 0.8
    
    def test_familiarity_entry_from_dict(self):
        """Test FamiliarityEntry deserialization from dictionary."""
        data = {
            "stimulus_hash": "restored_hash",
            "encounter_count": 10,
            "familiarity_score": 0.9,
            "curiosity_level": 0.1,
            "first_encountered": 500.0,
            "last_encountered": 600.0,
            "cumulative_reward": 2.5,
        }
        
        entry = FamiliarityEntry.from_dict(data)
        
        assert entry.stimulus_hash == "restored_hash"
        assert entry.encounter_count == 10
        assert entry.familiarity_score == 0.9
        assert entry.curiosity_level == 0.1
        assert entry.first_encountered == 500.0
        assert entry.last_encountered == 600.0
        assert entry.cumulative_reward == 2.5
    
    def test_familiarity_entry_from_dict_with_defaults(self):
        """Test FamiliarityEntry deserialization with missing keys uses defaults."""
        data = {"stimulus_hash": "minimal_hash"}
        
        entry = FamiliarityEntry.from_dict(data)
        
        assert entry.stimulus_hash == "minimal_hash"
        assert entry.encounter_count == 1  # Default
        assert entry.familiarity_score == 0.0  # Default
        assert entry.curiosity_level == 1.0  # Default


# =============================================================================
# Tests for ActionExplorationInfo Dataclass
# =============================================================================

class TestActionExplorationInfoDataclass:
    """Unit tests for ActionExplorationInfo dataclass creation and validation."""
    
    def test_valid_action_exploration_info_creation(self):
        """Test creating a valid ActionExplorationInfo object."""
        info = ActionExplorationInfo(
            action="search_action",
            attempt_count=15,
            contexts_seen=["ctx1", "ctx2", "ctx3"],
            average_information_gain=0.6,
            last_attempted=1500.0,
        )
        
        assert info.action == "search_action"
        assert info.attempt_count == 15
        assert info.contexts_seen == ["ctx1", "ctx2", "ctx3"]
        assert info.average_information_gain == 0.6
        assert info.last_attempted == 1500.0
    
    def test_action_exploration_info_default_values(self):
        """Test ActionExplorationInfo creation with default values."""
        info = ActionExplorationInfo(action="new_action")
        
        assert info.action == "new_action"
        assert info.attempt_count == 0
        assert info.contexts_seen == []
        assert info.average_information_gain == 0.5
        assert info.last_attempted is None
    
    def test_action_exploration_info_to_dict(self):
        """Test ActionExplorationInfo serialization to dictionary."""
        info = ActionExplorationInfo(
            action="test_action",
            attempt_count=5,
            contexts_seen=["a", "b"],
            average_information_gain=0.7,
            last_attempted=300.0,
        )
        
        data = info.to_dict()
        
        assert data["action"] == "test_action"
        assert data["attempt_count"] == 5
        assert data["contexts_seen"] == ["a", "b"]
        assert data["average_information_gain"] == 0.7
        assert data["last_attempted"] == 300.0
    
    def test_action_exploration_info_from_dict(self):
        """Test ActionExplorationInfo deserialization from dictionary."""
        data = {
            "action": "restored_action",
            "attempt_count": 20,
            "contexts_seen": ["x", "y", "z"],
            "average_information_gain": 0.8,
            "last_attempted": 400.0,
        }
        
        info = ActionExplorationInfo.from_dict(data)
        
        assert info.action == "restored_action"
        assert info.attempt_count == 20
        assert info.contexts_seen == ["x", "y", "z"]
        assert info.average_information_gain == 0.8
        assert info.last_attempted == 400.0
    
    def test_action_exploration_info_from_dict_with_defaults(self):
        """Test ActionExplorationInfo deserialization with missing keys uses defaults."""
        data = {"action": "minimal_action"}
        
        info = ActionExplorationInfo.from_dict(data)
        
        assert info.action == "minimal_action"
        assert info.attempt_count == 0  # Default
        assert info.contexts_seen == []  # Default
        assert info.average_information_gain == 0.5  # Default
        assert info.last_attempted is None  # Default


# =============================================================================
# Tests for IntrinsicMotivationEngine Initialization
# =============================================================================

class TestIntrinsicMotivationEngineInitialization:
    """Unit tests for IntrinsicMotivationEngine initialization."""
    
    def test_default_initialization(self, default_engine: IntrinsicMotivationEngine):
        """Test initialization with default configuration."""
        assert default_engine is not None
        
        state = default_engine.get_state()
        assert state["total_stimuli_encountered"] == 0
        assert state["total_curiosity_rewards"] == 0.0
        assert state["tracked_stimuli_count"] == 0
        assert state["tracked_actions_count"] == 0
        assert state["uptime"] >= 0
    
    def test_initialization_with_custom_config(self, custom_engine: IntrinsicMotivationEngine):
        """Test initialization with custom configuration."""
        state = custom_engine.get_state()
        config = state["config"]
        
        assert config["base_curiosity_reward"] == 0.4
        assert config["prediction_error_weight"] == 0.8
        assert config["novelty_threshold"] == 0.6
        assert config["exploration_weight"] == 0.4
    
    def test_initialization_with_config_dict(self):
        """Test initialization with config as dictionary."""
        config_dict = {
            "base_curiosity_reward": 0.6,
            "exploration_weight": 0.5,
        }
        
        engine = IntrinsicMotivationEngine(config=config_dict)
        state = engine.get_state()
        
        assert state["config"]["base_curiosity_reward"] == 0.6
        assert state["config"]["exploration_weight"] == 0.5
    
    def test_initialization_with_none_config(self):
        """Test initialization with None config uses defaults."""
        engine = IntrinsicMotivationEngine(config=None)
        state = engine.get_state()
        
        assert state["config"]["base_curiosity_reward"] == 0.5
        assert state["config"]["prediction_error_weight"] == 0.7
    
    def test_initial_state_is_empty(self, default_engine: IntrinsicMotivationEngine):
        """Test that initial state has no tracked stimuli or actions."""
        state = default_engine.get_state()
        
        assert state["tracked_stimuli_count"] == 0
        assert state["tracked_actions_count"] == 0
        assert state["total_stimuli_encountered"] == 0


# =============================================================================
# Tests for calculate_curiosity_reward() - Various Prediction Error Types
# =============================================================================

class TestCalculateCuriosityReward:
    """Unit tests for calculate_curiosity_reward() method."""
    
    def test_curiosity_reward_with_numeric_prediction_error(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test curiosity reward calculation with numeric prediction errors."""
        # Perfect prediction - low reward
        reward_perfect = default_engine.calculate_curiosity_reward(1.0, 1.0)
        assert 0.0 <= reward_perfect <= 1.0
        
        # Small error
        reward_small = default_engine.calculate_curiosity_reward(1.0, 1.2)
        assert 0.0 <= reward_small <= 1.0
        
        # Large error - higher reward
        reward_large = default_engine.calculate_curiosity_reward(1.0, 5.0)
        assert 0.0 <= reward_large <= 1.0
        
        # Larger errors should produce larger rewards
        assert reward_large >= reward_small
    
    def test_curiosity_reward_with_string_prediction_error(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test curiosity reward calculation with string prediction errors."""
        # Identical strings - low reward
        reward_same = default_engine.calculate_curiosity_reward("hello", "hello")
        assert 0.0 <= reward_same <= 1.0
        
        # Similar strings
        reward_similar = default_engine.calculate_curiosity_reward("hello", "hallo")
        assert 0.0 <= reward_similar <= 1.0
        
        # Completely different strings - higher reward
        reward_different = default_engine.calculate_curiosity_reward("hello", "world")
        assert 0.0 <= reward_different <= 1.0
        
        # Different strings should produce higher reward than identical
        assert reward_different >= reward_same
    
    def test_curiosity_reward_with_dict_prediction_error(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test curiosity reward calculation with dictionary prediction errors."""
        # Identical dicts
        reward_same = default_engine.calculate_curiosity_reward(
            {"a": 1, "b": 2}, {"a": 1, "b": 2}
        )
        assert 0.0 <= reward_same <= 1.0
        
        # Partially different dicts
        reward_partial = default_engine.calculate_curiosity_reward(
            {"a": 1, "b": 2}, {"a": 1, "b": 3}
        )
        assert 0.0 <= reward_partial <= 1.0
        
        # Completely different dicts
        reward_different = default_engine.calculate_curiosity_reward(
            {"a": 1}, {"x": 9, "y": 8}
        )
        assert 0.0 <= reward_different <= 1.0
    
    def test_curiosity_reward_with_list_prediction_error(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test curiosity reward calculation with list prediction errors."""
        # Identical lists
        reward_same = default_engine.calculate_curiosity_reward([1, 2, 3], [1, 2, 3])
        assert 0.0 <= reward_same <= 1.0
        
        # Partially different lists
        reward_partial = default_engine.calculate_curiosity_reward([1, 2, 3], [1, 2, 4])
        assert 0.0 <= reward_partial <= 1.0
        
        # Different length lists
        reward_diff_len = default_engine.calculate_curiosity_reward([1, 2], [1, 2, 3, 4])
        assert 0.0 <= reward_diff_len <= 1.0
    
    def test_curiosity_reward_with_none_values(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test curiosity reward calculation with None values."""
        # Both None - no error
        reward_both_none = default_engine.calculate_curiosity_reward(None, None)
        assert 0.0 <= reward_both_none <= 1.0
        
        # One None - maximum error
        reward_one_none = default_engine.calculate_curiosity_reward(None, "value")
        assert 0.0 <= reward_one_none <= 1.0
        
        reward_other_none = default_engine.calculate_curiosity_reward("value", None)
        assert 0.0 <= reward_other_none <= 1.0
    
    def test_curiosity_reward_is_bounded(self, default_engine: IntrinsicMotivationEngine):
        """Test that curiosity reward is always in [0.0, 1.0] range."""
        test_cases = [
            (0, 0),
            (100, 0),
            (-100, 100),
            ("", "very long string" * 100),
            ({}, {"a": 1, "b": 2, "c": 3}),
            ([], list(range(100))),
        ]
        
        for predicted, actual in test_cases:
            reward = default_engine.calculate_curiosity_reward(predicted, actual)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of bounds for {predicted}, {actual}"
    
    def test_curiosity_reward_accumulates_total(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that curiosity rewards accumulate in total."""
        initial_state = default_engine.get_state()
        initial_total = initial_state["total_curiosity_rewards"]
        
        reward1 = default_engine.calculate_curiosity_reward(1.0, 2.0)
        reward2 = default_engine.calculate_curiosity_reward("a", "b")
        
        final_state = default_engine.get_state()
        expected_total = initial_total + reward1 + reward2
        
        assert abs(final_state["total_curiosity_rewards"] - expected_total) < 0.001


# =============================================================================
# Tests for get_novelty_score() - New and Familiar Stimuli
# =============================================================================

class TestGetNoveltyScore:
    """Unit tests for get_novelty_score() method."""
    
    def test_novelty_score_for_new_stimulus(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that new stimuli have maximum novelty score of 1.0."""
        novelty = default_engine.get_novelty_score(sample_stimulus)
        
        assert novelty == 1.0
    
    def test_novelty_score_decreases_after_familiarity_update(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that novelty decreases after updating familiarity."""
        initial_novelty = default_engine.get_novelty_score(sample_stimulus)
        
        # Update familiarity
        default_engine.update_familiarity(sample_stimulus)
        
        updated_novelty = default_engine.get_novelty_score(sample_stimulus)
        
        assert updated_novelty < initial_novelty
    
    def test_novelty_score_continues_decreasing_with_repeated_encounters(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that novelty continues to decrease with repeated encounters."""
        novelty_scores = []
        
        for _ in range(5):
            novelty = default_engine.get_novelty_score(sample_stimulus)
            novelty_scores.append(novelty)
            default_engine.update_familiarity(sample_stimulus)
        
        # Each subsequent novelty should be less than or equal to previous
        for i in range(1, len(novelty_scores)):
            assert novelty_scores[i] <= novelty_scores[i - 1]
    
    def test_novelty_score_is_bounded(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that novelty score is always in [0.0, 1.0] range."""
        # New stimulus
        novelty_new = default_engine.get_novelty_score(sample_stimulus)
        assert 0.0 <= novelty_new <= 1.0
        
        # After many encounters
        for _ in range(20):
            default_engine.update_familiarity(sample_stimulus)
            default_engine.decay_curiosity(sample_stimulus)
        
        novelty_familiar = default_engine.get_novelty_score(sample_stimulus)
        assert 0.0 <= novelty_familiar <= 1.0
    
    def test_novelty_score_for_different_stimuli_types(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test novelty scoring for different stimulus types."""
        stimuli = [
            "string_stimulus",
            123,
            {"key": "value"},
            ["list", "items"],
            (1, 2, 3),
        ]
        
        for stimulus in stimuli:
            novelty = default_engine.get_novelty_score(stimulus)
            assert novelty == 1.0, f"New stimulus {stimulus} should have novelty 1.0"
    
    def test_novelty_score_different_stimuli_are_independent(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that different stimuli have independent novelty tracking."""
        stimulus_a = "stimulus_a"
        stimulus_b = "stimulus_b"
        
        # Make stimulus_a familiar
        for _ in range(5):
            default_engine.update_familiarity(stimulus_a)
        
        # stimulus_b should still be novel
        novelty_a = default_engine.get_novelty_score(stimulus_a)
        novelty_b = default_engine.get_novelty_score(stimulus_b)
        
        assert novelty_b == 1.0
        assert novelty_a < novelty_b


# =============================================================================
# Tests for update_familiarity() - Familiarity Tracking
# =============================================================================

class TestUpdateFamiliarity:
    """Unit tests for update_familiarity() method."""
    
    def test_update_familiarity_creates_entry_for_new_stimulus(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that update_familiarity creates entry for new stimulus."""
        # Initially no entry
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry is None
        
        # Update familiarity
        default_engine.update_familiarity(sample_stimulus)
        
        # Now entry should exist
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry is not None
        assert entry.encounter_count == 1
    
    def test_update_familiarity_increments_encounter_count(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that update_familiarity increments encounter count."""
        default_engine.update_familiarity(sample_stimulus)
        default_engine.update_familiarity(sample_stimulus)
        default_engine.update_familiarity(sample_stimulus)
        
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry.encounter_count == 3
    
    def test_update_familiarity_increases_familiarity_score(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that update_familiarity increases familiarity score."""
        default_engine.update_familiarity(sample_stimulus)
        entry1 = default_engine.get_familiarity_entry(sample_stimulus)
        score1 = entry1.familiarity_score
        
        default_engine.update_familiarity(sample_stimulus)
        entry2 = default_engine.get_familiarity_entry(sample_stimulus)
        score2 = entry2.familiarity_score
        
        assert score2 > score1
    
    def test_update_familiarity_score_approaches_one(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that familiarity score approaches 1.0 with many encounters."""
        for _ in range(50):
            default_engine.update_familiarity(sample_stimulus)
        
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry.familiarity_score > 0.9
        assert entry.familiarity_score <= 1.0
    
    def test_update_familiarity_updates_last_encountered(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that update_familiarity updates last_encountered timestamp."""
        default_engine.update_familiarity(sample_stimulus)
        entry1 = default_engine.get_familiarity_entry(sample_stimulus)
        time1 = entry1.last_encountered
        
        time.sleep(0.01)  # Small delay
        
        default_engine.update_familiarity(sample_stimulus)
        entry2 = default_engine.get_familiarity_entry(sample_stimulus)
        time2 = entry2.last_encountered
        
        assert time2 >= time1
    
    def test_update_familiarity_increments_total_stimuli_encountered(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that update_familiarity increments total stimuli counter."""
        initial_state = default_engine.get_state()
        initial_count = initial_state["total_stimuli_encountered"]
        
        default_engine.update_familiarity("stimulus_1")
        default_engine.update_familiarity("stimulus_2")
        default_engine.update_familiarity("stimulus_1")  # Same stimulus again
        
        final_state = default_engine.get_state()
        assert final_state["total_stimuli_encountered"] == initial_count + 3
    
    def test_update_familiarity_prunes_when_over_limit(self):
        """Test that familiarity tracker prunes entries when over limit."""
        # Create engine with small limit
        engine = IntrinsicMotivationEngine(config={"max_familiarity_entries": 5})
        
        # Add more stimuli than the limit
        for i in range(10):
            engine.update_familiarity(f"stimulus_{i}")
        
        state = engine.get_state()
        assert state["tracked_stimuli_count"] <= 5


# =============================================================================
# Tests for decay_curiosity() - Curiosity Decay Behavior
# =============================================================================

class TestDecayCuriosity:
    """Unit tests for decay_curiosity() method."""
    
    def test_decay_curiosity_reduces_curiosity_level(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that decay_curiosity reduces curiosity level."""
        # First create an entry
        default_engine.update_familiarity(sample_stimulus)
        entry1 = default_engine.get_familiarity_entry(sample_stimulus)
        curiosity1 = entry1.curiosity_level
        
        # Apply decay
        default_engine.decay_curiosity(sample_stimulus)
        
        entry2 = default_engine.get_familiarity_entry(sample_stimulus)
        curiosity2 = entry2.curiosity_level
        
        assert curiosity2 < curiosity1
    
    def test_decay_curiosity_on_unknown_stimulus_does_nothing(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that decay_curiosity on unknown stimulus has no effect."""
        # Should not raise error
        default_engine.decay_curiosity("unknown_stimulus")
        
        # No entry should be created
        entry = default_engine.get_familiarity_entry("unknown_stimulus")
        assert entry is None
    
    def test_decay_curiosity_approaches_zero(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that curiosity level approaches zero with repeated decay."""
        default_engine.update_familiarity(sample_stimulus)
        
        for _ in range(50):
            default_engine.decay_curiosity(sample_stimulus)
        
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry.curiosity_level < 0.01
        assert entry.curiosity_level >= 0.0
    
    def test_decay_curiosity_never_goes_negative(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that curiosity level never goes below zero."""
        default_engine.update_familiarity(sample_stimulus)
        
        for _ in range(100):
            default_engine.decay_curiosity(sample_stimulus)
        
        entry = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry.curiosity_level >= 0.0
    
    def test_decay_curiosity_rate_is_configurable(self):
        """Test that curiosity decay rate is configurable."""
        # High decay rate
        fast_engine = IntrinsicMotivationEngine(config={"curiosity_decay_rate": 0.5})
        fast_engine.update_familiarity("stimulus")
        fast_engine.decay_curiosity("stimulus")
        fast_entry = fast_engine.get_familiarity_entry("stimulus")
        
        # Low decay rate
        slow_engine = IntrinsicMotivationEngine(config={"curiosity_decay_rate": 0.05})
        slow_engine.update_familiarity("stimulus")
        slow_engine.decay_curiosity("stimulus")
        slow_entry = slow_engine.get_familiarity_entry("stimulus")
        
        # Fast decay should result in lower curiosity
        assert fast_entry.curiosity_level < slow_entry.curiosity_level


# =============================================================================
# Tests for get_intrinsic_reward() - Novel and Familiar Stimuli
# =============================================================================

class TestGetIntrinsicReward:
    """Unit tests for get_intrinsic_reward() method."""
    
    def test_intrinsic_reward_for_novel_stimulus(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that novel stimuli generate high intrinsic reward."""
        reward = default_engine.get_intrinsic_reward("completely_new_stimulus")
        
        assert reward > 0.0
        assert reward <= 1.0
    
    def test_intrinsic_reward_decreases_for_familiar_stimulus(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that intrinsic reward decreases for familiar stimuli."""
        stimulus = "repeated_stimulus"
        
        reward1 = default_engine.get_intrinsic_reward(stimulus)
        reward2 = default_engine.get_intrinsic_reward(stimulus)
        reward3 = default_engine.get_intrinsic_reward(stimulus)
        
        # Rewards should decrease with familiarity
        assert reward2 <= reward1
        assert reward3 <= reward2
    
    def test_intrinsic_reward_is_bounded(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that intrinsic reward is always in [0.0, 1.0] range."""
        # Novel stimulus
        reward_novel = default_engine.get_intrinsic_reward("novel")
        assert 0.0 <= reward_novel <= 1.0
        
        # Very familiar stimulus
        stimulus = "very_familiar"
        for _ in range(20):
            default_engine.get_intrinsic_reward(stimulus)
        
        reward_familiar = default_engine.get_intrinsic_reward(stimulus)
        assert 0.0 <= reward_familiar <= 1.0
    
    def test_intrinsic_reward_updates_familiarity(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that get_intrinsic_reward updates familiarity tracking."""
        stimulus = "tracked_stimulus"
        
        # Initially no entry
        entry = default_engine.get_familiarity_entry(stimulus)
        assert entry is None
        
        # Get reward
        default_engine.get_intrinsic_reward(stimulus)
        
        # Now entry should exist
        entry = default_engine.get_familiarity_entry(stimulus)
        assert entry is not None
    
    def test_intrinsic_reward_applies_curiosity_decay(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that get_intrinsic_reward applies curiosity decay."""
        stimulus = "decaying_stimulus"
        
        default_engine.get_intrinsic_reward(stimulus)
        entry1 = default_engine.get_familiarity_entry(stimulus)
        curiosity1 = entry1.curiosity_level
        
        default_engine.get_intrinsic_reward(stimulus)
        entry2 = default_engine.get_familiarity_entry(stimulus)
        curiosity2 = entry2.curiosity_level
        
        assert curiosity2 < curiosity1
    
    def test_intrinsic_reward_accumulates_cumulative_reward(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that intrinsic rewards accumulate in cumulative_reward."""
        stimulus = "cumulative_stimulus"
        
        reward1 = default_engine.get_intrinsic_reward(stimulus)
        entry1 = default_engine.get_familiarity_entry(stimulus)
        
        reward2 = default_engine.get_intrinsic_reward(stimulus)
        entry2 = default_engine.get_familiarity_entry(stimulus)
        
        assert abs(entry2.cumulative_reward - (reward1 + reward2)) < 0.001
    
    def test_intrinsic_reward_higher_for_novel_than_familiar(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that novel stimuli get higher rewards than familiar ones."""
        # Make one stimulus very familiar
        familiar_stimulus = "familiar"
        for _ in range(10):
            default_engine.get_intrinsic_reward(familiar_stimulus)
        
        # Get reward for familiar vs novel
        reward_familiar = default_engine.get_intrinsic_reward(familiar_stimulus)
        reward_novel = default_engine.get_intrinsic_reward("brand_new")
        
        assert reward_novel > reward_familiar


# =============================================================================
# Tests for get_exploration_bonus() - Explored and Unexplored Actions
# =============================================================================

class TestGetExplorationBonus:
    """Unit tests for get_exploration_bonus() method."""
    
    def test_exploration_bonus_for_new_action(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that new actions have maximum exploration bonus of 1.0."""
        bonus = default_engine.get_exploration_bonus("new_action", sample_context)
        
        assert bonus == 1.0
    
    def test_exploration_bonus_decreases_with_attempts(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that exploration bonus decreases with more attempts."""
        action = "repeated_action"
        
        bonus1 = default_engine.get_exploration_bonus(action, sample_context)
        
        # Update exploration tracking
        default_engine.update_action_exploration(action, sample_context, 0.5)
        
        bonus2 = default_engine.get_exploration_bonus(action, sample_context)
        
        assert bonus2 < bonus1
    
    def test_exploration_bonus_higher_for_new_context(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that exploration bonus is higher for new contexts."""
        action = "context_action"
        context1 = {"type": "context_1"}
        context2 = {"type": "context_2"}
        
        # Explore action in context1
        default_engine.update_action_exploration(action, context1, 0.5)
        
        # Bonus should be higher for new context
        bonus_old_context = default_engine.get_exploration_bonus(action, context1)
        bonus_new_context = default_engine.get_exploration_bonus(action, context2)
        
        assert bonus_new_context > bonus_old_context
    
    def test_exploration_bonus_is_bounded(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that exploration bonus is always in [0.0, 1.0] range."""
        action = "bounded_action"
        
        # New action
        bonus_new = default_engine.get_exploration_bonus(action, sample_context)
        assert 0.0 <= bonus_new <= 1.0
        
        # After many attempts
        for i in range(50):
            default_engine.update_action_exploration(
                action, {"iteration": i}, 0.3
            )
        
        bonus_explored = default_engine.get_exploration_bonus(action, sample_context)
        assert 0.0 <= bonus_explored <= 1.0
    
    def test_exploration_bonus_considers_information_gain(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that exploration bonus considers average information gain."""
        action_high_gain = "high_gain_action"
        action_low_gain = "low_gain_action"
        
        # Update with different information gains
        default_engine.update_action_exploration(action_high_gain, sample_context, 0.9)
        default_engine.update_action_exploration(action_low_gain, sample_context, 0.1)
        
        bonus_high = default_engine.get_exploration_bonus(action_high_gain, sample_context)
        bonus_low = default_engine.get_exploration_bonus(action_low_gain, sample_context)
        
        # Higher information gain should contribute to higher bonus
        # Note: The exact relationship depends on implementation weights
        assert bonus_high != bonus_low


# =============================================================================
# Tests for update_action_exploration()
# =============================================================================

class TestUpdateActionExploration:
    """Unit tests for update_action_exploration() method."""
    
    def test_update_action_exploration_creates_entry(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that update_action_exploration creates entry for new action."""
        action = "new_action"
        
        # Initially no entry
        info = default_engine.get_action_exploration_info(action)
        assert info is None
        
        # Update exploration
        default_engine.update_action_exploration(action, sample_context, 0.5)
        
        # Now entry should exist
        info = default_engine.get_action_exploration_info(action)
        assert info is not None
        assert info.action == action
        assert info.attempt_count == 1
    
    def test_update_action_exploration_increments_attempt_count(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that update_action_exploration increments attempt count."""
        action = "counted_action"
        
        default_engine.update_action_exploration(action, sample_context, 0.5)
        default_engine.update_action_exploration(action, sample_context, 0.6)
        default_engine.update_action_exploration(action, sample_context, 0.7)
        
        info = default_engine.get_action_exploration_info(action)
        assert info.attempt_count == 3
    
    def test_update_action_exploration_tracks_contexts(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that update_action_exploration tracks seen contexts."""
        action = "context_tracked_action"
        context1 = {"type": "a"}
        context2 = {"type": "b"}
        
        default_engine.update_action_exploration(action, context1, 0.5)
        default_engine.update_action_exploration(action, context2, 0.5)
        default_engine.update_action_exploration(action, context1, 0.5)  # Duplicate
        
        info = default_engine.get_action_exploration_info(action)
        assert len(info.contexts_seen) == 2  # Only unique contexts
    
    def test_update_action_exploration_updates_average_information_gain(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that update_action_exploration updates average information gain."""
        action = "avg_gain_action"
        
        default_engine.update_action_exploration(action, sample_context, 0.4)
        info1 = default_engine.get_action_exploration_info(action)
        
        default_engine.update_action_exploration(action, sample_context, 0.8)
        info2 = default_engine.get_action_exploration_info(action)
        
        # Average should be (0.4 + 0.8) / 2 = 0.6
        assert abs(info2.average_information_gain - 0.6) < 0.01
    
    def test_update_action_exploration_updates_last_attempted(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that update_action_exploration updates last_attempted timestamp."""
        action = "timed_action"
        
        default_engine.update_action_exploration(action, sample_context, 0.5)
        info1 = default_engine.get_action_exploration_info(action)
        time1 = info1.last_attempted
        
        time.sleep(0.01)
        
        default_engine.update_action_exploration(action, sample_context, 0.5)
        info2 = default_engine.get_action_exploration_info(action)
        time2 = info2.last_attempted
        
        assert time2 >= time1


# =============================================================================
# Tests for select_action_with_exploration() - Different Known Rewards
# =============================================================================

class TestSelectActionWithExploration:
    """Unit tests for select_action_with_exploration() method."""
    
    def test_select_action_with_no_known_rewards(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test action selection when no known rewards are provided."""
        actions = ["action_a", "action_b", "action_c"]
        
        selected, score = default_engine.select_action_with_exploration(
            actions, sample_context
        )
        
        assert selected in actions
        assert 0.0 <= score <= 1.0
    
    def test_select_action_with_known_rewards(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test action selection with known rewards."""
        actions = ["low_reward", "high_reward", "medium_reward"]
        known_rewards = {
            "low_reward": 0.1,
            "high_reward": 0.9,
            "medium_reward": 0.5,
        }
        
        selected, score = default_engine.select_action_with_exploration(
            actions, sample_context, known_rewards
        )
        
        assert selected in actions
        assert 0.0 <= score <= 1.0
    
    def test_select_action_prefers_high_reward_with_low_exploration(self):
        """Test that low exploration weight prefers high known rewards."""
        # Low exploration weight = prefer exploitation
        engine = IntrinsicMotivationEngine(config={"exploration_weight": 0.1})
        context = {"test": True}
        
        actions = ["low", "high"]
        known_rewards = {"low": 0.1, "high": 0.9}
        
        # Run multiple times to check tendency
        high_count = 0
        for _ in range(10):
            selected, _ = engine.select_action_with_exploration(
                actions, context, known_rewards
            )
            if selected == "high":
                high_count += 1
        
        # Should mostly select high reward action
        assert high_count >= 7
    
    def test_select_action_explores_with_high_exploration_weight(self):
        """Test that high exploration weight considers exploration bonus."""
        # High exploration weight
        engine = IntrinsicMotivationEngine(config={"exploration_weight": 0.9})
        context = {"test": True}
        
        actions = ["explored", "unexplored"]
        known_rewards = {"explored": 0.9, "unexplored": 0.1}
        
        # Make "explored" very familiar
        for _ in range(20):
            engine.update_action_exploration("explored", context, 0.3)
        
        # With high exploration weight, unexplored action should be competitive
        selected, _ = engine.select_action_with_exploration(
            actions, context, known_rewards
        )
        
        # The unexplored action should have a chance due to exploration bonus
        assert selected in actions
    
    def test_select_action_with_empty_actions_raises_error(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that empty actions list raises ValueError."""
        with pytest.raises(ValueError, match="No actions provided"):
            default_engine.select_action_with_exploration([], sample_context)
    
    def test_select_action_with_single_action(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test action selection with single action."""
        actions = ["only_action"]
        
        selected, score = default_engine.select_action_with_exploration(
            actions, sample_context
        )
        
        assert selected == "only_action"
        assert 0.0 <= score <= 1.0
    
    def test_select_action_returns_consistent_best_score(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test that selected action has the best score."""
        actions = ["a", "b", "c"]
        known_rewards = {"a": 0.3, "b": 0.6, "c": 0.4}
        
        selected, best_score = default_engine.select_action_with_exploration(
            actions, sample_context, known_rewards
        )
        
        # Verify the selected action's score matches the returned score
        exploration_bonus = default_engine.get_exploration_bonus(selected, sample_context)
        known_reward = known_rewards.get(selected, 0.5)
        config = default_engine.get_state()["config"]
        expected_score = (
            config["exploration_weight"] * exploration_bonus +
            (1 - config["exploration_weight"]) * known_reward
        )
        
        assert abs(best_score - expected_score) < 0.01
    
    def test_select_action_with_partial_known_rewards(
        self, default_engine: IntrinsicMotivationEngine, sample_context: Dict[str, Any]
    ):
        """Test action selection with partial known rewards."""
        actions = ["known", "unknown"]
        known_rewards = {"known": 0.8}  # "unknown" not in rewards
        
        selected, score = default_engine.select_action_with_exploration(
            actions, sample_context, known_rewards
        )
        
        assert selected in actions
        assert 0.0 <= score <= 1.0


# =============================================================================
# Tests for reset_curiosity()
# =============================================================================

class TestResetCuriosity:
    """Unit tests for reset_curiosity() method."""
    
    def test_reset_curiosity_restores_curiosity_level(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that reset_curiosity restores curiosity level to 1.0."""
        # Make stimulus familiar
        for _ in range(10):
            default_engine.get_intrinsic_reward(sample_stimulus)
        
        entry_before = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry_before.curiosity_level < 1.0
        
        # Reset curiosity
        default_engine.reset_curiosity(sample_stimulus)
        
        entry_after = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry_after.curiosity_level == 1.0
    
    def test_reset_curiosity_clears_familiarity_score(
        self, default_engine: IntrinsicMotivationEngine, sample_stimulus: str
    ):
        """Test that reset_curiosity clears familiarity score."""
        # Make stimulus familiar
        for _ in range(10):
            default_engine.update_familiarity(sample_stimulus)
        
        entry_before = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry_before.familiarity_score > 0.0
        
        # Reset curiosity
        default_engine.reset_curiosity(sample_stimulus)
        
        entry_after = default_engine.get_familiarity_entry(sample_stimulus)
        assert entry_after.familiarity_score == 0.0
    
    def test_reset_curiosity_on_unknown_stimulus_does_nothing(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that reset_curiosity on unknown stimulus has no effect."""
        # Should not raise error
        default_engine.reset_curiosity("unknown_stimulus")
        
        # No entry should be created
        entry = default_engine.get_familiarity_entry("unknown_stimulus")
        assert entry is None


# =============================================================================
# Tests for Serialization/Deserialization (to_dict/from_dict)
# =============================================================================

class TestSerialization:
    """Unit tests for to_dict() and from_dict() serialization methods."""
    
    def test_to_dict_includes_all_fields(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that to_dict includes all required fields."""
        # Add some state
        default_engine.update_familiarity("stimulus_1")
        default_engine.update_familiarity("stimulus_2")
        default_engine.update_action_exploration("action_1", {"ctx": 1}, 0.5)
        default_engine.calculate_curiosity_reward(1.0, 2.0)
        
        data = default_engine.to_dict()
        
        assert "config" in data
        assert "familiarity_tracker" in data
        assert "action_exploration" in data
        assert "total_curiosity_rewards" in data
        assert "total_stimuli_encountered" in data
        assert "initialized_at" in data
    
    def test_to_dict_serializes_config(
        self, custom_engine: IntrinsicMotivationEngine
    ):
        """Test that to_dict correctly serializes config."""
        data = custom_engine.to_dict()
        
        assert data["config"]["base_curiosity_reward"] == 0.4
        assert data["config"]["prediction_error_weight"] == 0.8
        assert data["config"]["novelty_threshold"] == 0.6
    
    def test_to_dict_serializes_familiarity_tracker(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that to_dict correctly serializes familiarity tracker."""
        default_engine.update_familiarity("test_stimulus")
        
        data = default_engine.to_dict()
        
        assert len(data["familiarity_tracker"]) == 1
        # Check that entries are properly serialized
        for key, entry in data["familiarity_tracker"].items():
            assert "stimulus_hash" in entry
            assert "encounter_count" in entry
            assert "familiarity_score" in entry
    
    def test_to_dict_serializes_action_exploration(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that to_dict correctly serializes action exploration."""
        default_engine.update_action_exploration("test_action", {"ctx": 1}, 0.7)
        
        data = default_engine.to_dict()
        
        assert "test_action" in data["action_exploration"]
        action_data = data["action_exploration"]["test_action"]
        assert action_data["action"] == "test_action"
        assert action_data["attempt_count"] == 1
    
    def test_from_dict_restores_config(self):
        """Test that from_dict correctly restores config."""
        engine = IntrinsicMotivationEngine()
        
        data = {
            "config": {
                "base_curiosity_reward": 0.3,
                "prediction_error_weight": 0.6,
                "novelty_threshold": 0.4,
            }
        }
        
        engine.from_dict(data)
        state = engine.get_state()
        
        assert state["config"]["base_curiosity_reward"] == 0.3
        assert state["config"]["prediction_error_weight"] == 0.6
        assert state["config"]["novelty_threshold"] == 0.4
    
    def test_from_dict_restores_familiarity_tracker(self):
        """Test that from_dict correctly restores familiarity tracker."""
        engine = IntrinsicMotivationEngine()
        
        data = {
            "familiarity_tracker": {
                "hash123": {
                    "stimulus_hash": "hash123",
                    "encounter_count": 5,
                    "familiarity_score": 0.7,
                    "curiosity_level": 0.3,
                    "first_encountered": 1000.0,
                    "last_encountered": 2000.0,
                    "cumulative_reward": 1.5,
                }
            }
        }
        
        engine.from_dict(data)
        
        # The entry should be restored (we need to check via internal state)
        state = engine.get_state()
        assert state["tracked_stimuli_count"] == 1
    
    def test_from_dict_restores_action_exploration(self):
        """Test that from_dict correctly restores action exploration."""
        engine = IntrinsicMotivationEngine()
        
        data = {
            "action_exploration": {
                "restored_action": {
                    "action": "restored_action",
                    "attempt_count": 10,
                    "contexts_seen": ["ctx1", "ctx2"],
                    "average_information_gain": 0.6,
                    "last_attempted": 500.0,
                }
            }
        }
        
        engine.from_dict(data)
        
        info = engine.get_action_exploration_info("restored_action")
        assert info is not None
        assert info.action == "restored_action"
        assert info.attempt_count == 10
        assert info.contexts_seen == ["ctx1", "ctx2"]
    
    def test_from_dict_restores_counters(self):
        """Test that from_dict correctly restores counters."""
        engine = IntrinsicMotivationEngine()
        
        data = {
            "total_curiosity_rewards": 5.5,
            "total_stimuli_encountered": 100,
            "initialized_at": 12345.0,
        }
        
        engine.from_dict(data)
        state = engine.get_state()
        
        assert state["total_curiosity_rewards"] == 5.5
        assert state["total_stimuli_encountered"] == 100
    
    def test_serialization_round_trip(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that serialization and deserialization preserves state."""
        # Add various state
        default_engine.update_familiarity("stimulus_a")
        default_engine.update_familiarity("stimulus_b")
        default_engine.update_familiarity("stimulus_a")  # Repeat
        default_engine.update_action_exploration("action_x", {"ctx": 1}, 0.5)
        default_engine.update_action_exploration("action_y", {"ctx": 2}, 0.8)
        default_engine.calculate_curiosity_reward(1.0, 3.0)
        default_engine.get_intrinsic_reward("stimulus_c")
        
        # Serialize
        data = default_engine.to_dict()
        
        # Create new engine and restore
        new_engine = IntrinsicMotivationEngine()
        new_engine.from_dict(data)
        
        # Compare states
        original_state = default_engine.get_state()
        restored_state = new_engine.get_state()
        
        assert original_state["tracked_stimuli_count"] == restored_state["tracked_stimuli_count"]
        assert original_state["tracked_actions_count"] == restored_state["tracked_actions_count"]
        assert abs(original_state["total_curiosity_rewards"] - restored_state["total_curiosity_rewards"]) < 0.001
        assert original_state["total_stimuli_encountered"] == restored_state["total_stimuli_encountered"]
    
    def test_from_dict_with_empty_data(self):
        """Test that from_dict handles empty data gracefully."""
        engine = IntrinsicMotivationEngine()
        original_state = engine.get_state()
        
        # Empty data should not crash
        engine.from_dict({})
        
        # State should remain unchanged
        new_state = engine.get_state()
        assert new_state["config"] == original_state["config"]
    
    def test_from_dict_with_partial_data(self):
        """Test that from_dict handles partial data gracefully."""
        engine = IntrinsicMotivationEngine()
        
        # Only provide some fields
        data = {
            "total_stimuli_encountered": 50,
        }
        
        engine.from_dict(data)
        state = engine.get_state()
        
        assert state["total_stimuli_encountered"] == 50
        # Other fields should retain defaults
        assert state["tracked_stimuli_count"] == 0


# =============================================================================
# Tests for get_state()
# =============================================================================

class TestGetState:
    """Unit tests for get_state() method."""
    
    def test_get_state_returns_complete_state(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that get_state returns complete state information."""
        state = default_engine.get_state()
        
        assert "total_stimuli_encountered" in state
        assert "total_curiosity_rewards" in state
        assert "tracked_stimuli_count" in state
        assert "tracked_actions_count" in state
        assert "uptime" in state
        assert "config" in state
    
    def test_get_state_reflects_updates(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that get_state reflects state updates."""
        initial_state = default_engine.get_state()
        
        default_engine.update_familiarity("new_stimulus")
        default_engine.update_action_exploration("new_action", {}, 0.5)
        
        updated_state = default_engine.get_state()
        
        assert updated_state["tracked_stimuli_count"] == initial_state["tracked_stimuli_count"] + 1
        assert updated_state["tracked_actions_count"] == initial_state["tracked_actions_count"] + 1
        assert updated_state["total_stimuli_encountered"] == initial_state["total_stimuli_encountered"] + 1
    
    def test_get_state_uptime_increases(
        self, default_engine: IntrinsicMotivationEngine
    ):
        """Test that uptime increases over time."""
        state1 = default_engine.get_state()
        time.sleep(0.01)
        state2 = default_engine.get_state()
        
        assert state2["uptime"] >= state1["uptime"]
