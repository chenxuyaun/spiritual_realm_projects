"""
Property-based tests for Intrinsic Motivation Engine.

Tests properties 6-9 from the consciousness-system-deepening design document:
- Property 6: Curiosity Reward Proportionality
- Property 7: Novelty Reward Generation
- Property 8: Novelty Decay with Repeated Exposure
- Property 9: Exploration-Exploitation Balance

Validates: Requirements 2.1-2.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Any, Optional

from mm_orch.consciousness.intrinsic_motivation import (
    IntrinsicMotivationEngine,
    IntrinsicMotivationConfig,
    FamiliarityEntry,
    ActionExplorationInfo,
)


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for valid float values in [0.0, 1.0]
unit_float_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for numeric outcomes (for prediction error calculation)
numeric_outcome_strategy = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Strategy for string outcomes
string_outcome_strategy = st.text(min_size=0, max_size=50)

# Strategy for dictionary outcomes
dict_outcome_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
    min_size=0,
    max_size=5
)

# Strategy for list outcomes
list_outcome_strategy = st.lists(
    st.one_of(st.integers(), st.text(max_size=10)),
    min_size=0,
    max_size=10
)

# Strategy for any outcome type
outcome_strategy = st.one_of(
    numeric_outcome_strategy,
    string_outcome_strategy,
    dict_outcome_strategy,
    list_outcome_strategy,
    st.none()
)

# Strategy for stimulus (can be string, dict, or other hashable types)
stimulus_strategy = st.one_of(
    st.text(min_size=1, max_size=50),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.integers(), st.text(max_size=20)),
        min_size=1,
        max_size=5
    ),
    st.integers()
)

# Strategy for action names
action_strategy = st.text(
    min_size=1, max_size=30,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Pd', 'Pc'))
)

# Strategy for context dictionaries
context_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15),
    values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
    min_size=0,
    max_size=5
)

# Strategy for known rewards dictionary
known_rewards_strategy = st.dictionaries(
    keys=action_strategy,
    values=unit_float_strategy,
    min_size=0,
    max_size=5
)


# Strategy for generating valid IntrinsicMotivationConfig objects
@st.composite
def intrinsic_motivation_config_strategy(draw):
    """Generate a valid IntrinsicMotivationConfig object."""
    base_curiosity = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    pred_error_weight = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    novelty_threshold = draw(st.floats(min_value=0.3, max_value=0.7, allow_nan=False))
    familiarity_decay = draw(st.floats(min_value=0.05, max_value=0.3, allow_nan=False))
    curiosity_decay = draw(st.floats(min_value=0.05, max_value=0.3, allow_nan=False))
    exploration_weight = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    
    return IntrinsicMotivationConfig(
        base_curiosity_reward=base_curiosity,
        prediction_error_weight=pred_error_weight,
        novelty_threshold=novelty_threshold,
        familiarity_decay_rate=familiarity_decay,
        curiosity_decay_rate=curiosity_decay,
        exploration_weight=exploration_weight,
        max_familiarity_entries=1000,
        information_gain_scale=1.0,
    )


# =============================================================================
# Property 6: Curiosity Reward Proportionality
# =============================================================================

class TestCuriosityRewardProportionality:
    """
    Tests for Property 6: Curiosity Reward Proportionality
    
    *For any* prediction-outcome pair, the curiosity reward SHALL be proportional 
    to the absolute prediction error, with larger errors producing larger rewards 
    within [0.0, 1.0].
    
    **Validates: Requirements 2.1**
    """

    @given(
        predicted=numeric_outcome_strategy,
        actual=numeric_outcome_strategy
    )
    @settings(max_examples=100)
    def test_curiosity_reward_in_valid_range(self, predicted: float, actual: float):
        """
        Feature: consciousness-system-deepening, Property 6: Curiosity Reward Proportionality
        
        For any prediction-outcome pair, the curiosity reward SHALL be in range [0.0, 1.0].
        
        **Validates: Requirements 2.1**
        """
        engine = IntrinsicMotivationEngine()
        
        reward = engine.calculate_curiosity_reward(predicted, actual)
        
        assert 0.0 <= reward <= 1.0, \
            f"Curiosity reward {reward} must be in [0.0, 1.0]"

    @given(
        predicted=numeric_outcome_strategy,
        actual=numeric_outcome_strategy
    )
    @settings(max_examples=100)
    def test_curiosity_reward_is_deterministic(self, predicted: float, actual: float):
        """
        Feature: consciousness-system-deepening, Property 6: Curiosity Reward Proportionality
        
        For the same prediction-outcome pair, the curiosity reward SHALL be deterministic.
        
        **Validates: Requirements 2.1**
        """
        engine = IntrinsicMotivationEngine()
        
        reward1 = engine.calculate_curiosity_reward(predicted, actual)
        reward2 = engine.calculate_curiosity_reward(predicted, actual)
        
        assert reward1 == reward2, \
            f"Curiosity reward must be deterministic: {reward1} != {reward2}"

    @given(
        base_value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
        small_error=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
        large_error=st.floats(min_value=10.0, max_value=50.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_larger_prediction_error_produces_larger_reward(
        self, base_value: float, small_error: float, large_error: float
    ):
        """
        Feature: consciousness-system-deepening, Property 6: Curiosity Reward Proportionality
        
        For any prediction-outcome pairs, larger prediction errors SHALL produce 
        larger (or equal) curiosity rewards.
        
        **Validates: Requirements 2.1**
        """
        # Ensure large_error is significantly larger than small_error
        assume(large_error > small_error * 2)
        
        engine = IntrinsicMotivationEngine()
        
        # Calculate rewards for small and large prediction errors
        predicted = base_value
        actual_small_error = base_value + small_error
        actual_large_error = base_value + large_error
        
        reward_small = engine.calculate_curiosity_reward(predicted, actual_small_error)
        reward_large = engine.calculate_curiosity_reward(predicted, actual_large_error)
        
        assert reward_large >= reward_small, \
            f"Larger prediction error should produce larger reward: " \
            f"small_error={small_error} -> reward={reward_small}, " \
            f"large_error={large_error} -> reward={reward_large}"

    @given(value=numeric_outcome_strategy)
    @settings(max_examples=100)
    def test_zero_prediction_error_produces_minimum_reward(self, value: float):
        """
        Feature: consciousness-system-deepening, Property 6: Curiosity Reward Proportionality
        
        For any prediction-outcome pair with zero error (predicted == actual),
        the curiosity reward SHALL be at or near the minimum (base reward).
        
        **Validates: Requirements 2.1**
        """
        engine = IntrinsicMotivationEngine()
        
        # Same predicted and actual value = zero prediction error
        reward = engine.calculate_curiosity_reward(value, value)
        
        # With zero error, reward should be at or near base_curiosity_reward
        # (accounting for the weighted formula)
        max_expected = engine._config.base_curiosity_reward + 0.1
        
        assert reward <= max_expected, \
            f"Zero prediction error should produce near-minimum reward: {reward}"

    @given(
        predicted=outcome_strategy,
        actual=outcome_strategy,
        config=intrinsic_motivation_config_strategy()
    )
    @settings(max_examples=100)
    def test_curiosity_reward_respects_config(
        self, predicted: Any, actual: Any, config: IntrinsicMotivationConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 6: Curiosity Reward Proportionality
        
        For any configuration, the curiosity reward SHALL still be in valid range [0.0, 1.0].
        
        **Validates: Requirements 2.1**
        """
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        reward = engine.calculate_curiosity_reward(predicted, actual)
        
        assert 0.0 <= reward <= 1.0, \
            f"Curiosity reward {reward} must be in [0.0, 1.0] for any config"


# =============================================================================
# Property 7: Novelty Reward Generation
# =============================================================================

class TestNoveltyRewardGeneration:
    """
    Tests for Property 7: Novelty Reward Generation
    
    *For any* stimulus, the intrinsic reward SHALL be higher when novelty_score 
    is higher, and novel stimuli (novelty > 0.5) SHALL always generate positive rewards.
    
    **Validates: Requirements 2.2**
    """

    @given(stimulus=stimulus_strategy)
    @settings(max_examples=100)
    def test_novel_stimulus_has_high_novelty_score(self, stimulus: Any):
        """
        Feature: consciousness-system-deepening, Property 7: Novelty Reward Generation
        
        For any stimulus never seen before, the novelty_score SHALL be 1.0 (maximum).
        
        **Validates: Requirements 2.2**
        """
        engine = IntrinsicMotivationEngine()
        
        # First encounter - should be maximally novel
        novelty = engine.get_novelty_score(stimulus)
        
        assert novelty == 1.0, \
            f"First encounter novelty should be 1.0, got {novelty}"

    @given(stimulus=stimulus_strategy)
    @settings(max_examples=100)
    def test_novel_stimuli_generate_positive_rewards(self, stimulus: Any):
        """
        Feature: consciousness-system-deepening, Property 7: Novelty Reward Generation
        
        For any novel stimulus (novelty > 0.5), the intrinsic reward SHALL be positive.
        
        **Validates: Requirements 2.2**
        """
        engine = IntrinsicMotivationEngine()
        
        # Check novelty before getting reward
        novelty = engine.get_novelty_score(stimulus)
        assume(novelty > 0.5)  # Only test novel stimuli
        
        # Get intrinsic reward
        reward = engine.get_intrinsic_reward(stimulus)
        
        assert reward > 0.0, \
            f"Novel stimulus (novelty={novelty}) should generate positive reward, got {reward}"

    @given(stimulus=stimulus_strategy)
    @settings(max_examples=100)
    def test_intrinsic_reward_in_valid_range(self, stimulus: Any):
        """
        Feature: consciousness-system-deepening, Property 7: Novelty Reward Generation
        
        For any stimulus, the intrinsic reward SHALL be in range [0.0, 1.0].
        
        **Validates: Requirements 2.2**
        """
        engine = IntrinsicMotivationEngine()
        
        reward = engine.get_intrinsic_reward(stimulus)
        
        assert 0.0 <= reward <= 1.0, \
            f"Intrinsic reward {reward} must be in [0.0, 1.0]"

    @given(
        stimulus1=stimulus_strategy,
        stimulus2=stimulus_strategy,
        num_encounters=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_higher_novelty_produces_higher_reward(
        self, stimulus1: Any, stimulus2: Any, num_encounters: int
    ):
        """
        Feature: consciousness-system-deepening, Property 7: Novelty Reward Generation
        
        For any two stimuli, the one with higher novelty_score SHALL generate
        higher (or equal) intrinsic reward.
        
        **Validates: Requirements 2.2**
        """
        # Ensure stimuli are different
        assume(str(stimulus1) != str(stimulus2))
        
        engine = IntrinsicMotivationEngine()
        
        # Make stimulus1 familiar by encountering it multiple times
        for _ in range(num_encounters):
            engine.update_familiarity(stimulus1)
            engine.decay_curiosity(stimulus1)
        
        # Get novelty scores
        novelty1 = engine.get_novelty_score(stimulus1)
        novelty2 = engine.get_novelty_score(stimulus2)  # Still novel
        
        # stimulus2 should be more novel
        assume(novelty2 > novelty1 + 0.1)  # Ensure meaningful difference
        
        # Create fresh engines to get rewards without side effects
        engine_for_reward = IntrinsicMotivationEngine()
        
        # Copy familiarity state
        engine_for_reward._familiarity_tracker = engine._familiarity_tracker.copy()
        
        reward1 = engine_for_reward.get_intrinsic_reward(stimulus1)
        
        # Reset for stimulus2
        engine_for_reward2 = IntrinsicMotivationEngine()
        engine_for_reward2._familiarity_tracker = engine._familiarity_tracker.copy()
        reward2 = engine_for_reward2.get_intrinsic_reward(stimulus2)
        
        assert reward2 >= reward1, \
            f"Higher novelty ({novelty2}) should produce higher reward ({reward2}) " \
            f"than lower novelty ({novelty1}) with reward ({reward1})"


# =============================================================================
# Property 8: Novelty Decay with Repeated Exposure
# =============================================================================

class TestNoveltyDecayWithRepeatedExposure:
    """
    Tests for Property 8: Novelty Decay with Repeated Exposure
    
    *For any* stimulus encountered multiple times, the novelty_score SHALL 
    monotonically decrease with each encounter, and curiosity rewards SHALL 
    decay accordingly.
    
    **Validates: Requirements 2.3, 2.5**
    """

    @given(
        stimulus=stimulus_strategy,
        num_encounters=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=100)
    def test_novelty_decreases_with_repeated_exposure(
        self, stimulus: Any, num_encounters: int
    ):
        """
        Feature: consciousness-system-deepening, Property 8: Novelty Decay with Repeated Exposure
        
        For any stimulus encountered multiple times, the novelty_score SHALL 
        monotonically decrease with each encounter.
        
        **Validates: Requirements 2.3, 2.5**
        """
        engine = IntrinsicMotivationEngine()
        
        novelty_scores = []
        
        # Record novelty before first encounter
        initial_novelty = engine.get_novelty_score(stimulus)
        novelty_scores.append(initial_novelty)
        
        # Encounter the stimulus multiple times
        for i in range(num_encounters):
            engine.update_familiarity(stimulus)
            engine.decay_curiosity(stimulus)
            novelty = engine.get_novelty_score(stimulus)
            novelty_scores.append(novelty)
        
        # Verify monotonic decrease
        for i in range(1, len(novelty_scores)):
            assert novelty_scores[i] <= novelty_scores[i-1], \
                f"Novelty should monotonically decrease: " \
                f"encounter {i-1}={novelty_scores[i-1]} -> encounter {i}={novelty_scores[i]}"

    @given(
        stimulus=stimulus_strategy,
        num_encounters=st.integers(min_value=2, max_value=15)
    )
    @settings(max_examples=100)
    def test_curiosity_rewards_decay_with_repeated_exposure(
        self, stimulus: Any, num_encounters: int
    ):
        """
        Feature: consciousness-system-deepening, Property 8: Novelty Decay with Repeated Exposure
        
        For any stimulus encountered multiple times, the curiosity rewards SHALL 
        decay accordingly (later encounters produce lower rewards).
        
        **Validates: Requirements 2.3, 2.5**
        """
        engine = IntrinsicMotivationEngine()
        
        rewards = []
        
        # Get rewards for multiple encounters
        for _ in range(num_encounters):
            reward = engine.get_intrinsic_reward(stimulus)
            rewards.append(reward)
        
        # First reward should be highest (or equal to subsequent)
        assert rewards[0] >= rewards[-1], \
            f"First encounter reward ({rewards[0]}) should be >= " \
            f"last encounter reward ({rewards[-1]})"
        
        # Overall trend should be decreasing
        # Check that average of first half >= average of second half
        mid = len(rewards) // 2
        first_half_avg = sum(rewards[:mid]) / mid if mid > 0 else rewards[0]
        second_half_avg = sum(rewards[mid:]) / (len(rewards) - mid)
        
        assert first_half_avg >= second_half_avg - 0.01, \
            f"Rewards should generally decrease: first_half_avg={first_half_avg}, " \
            f"second_half_avg={second_half_avg}"

    @given(stimulus=stimulus_strategy)
    @settings(max_examples=100)
    def test_familiarity_increases_with_encounters(self, stimulus: Any):
        """
        Feature: consciousness-system-deepening, Property 8: Novelty Decay with Repeated Exposure
        
        For any stimulus, the familiarity_score SHALL increase with each encounter.
        
        **Validates: Requirements 2.3**
        """
        engine = IntrinsicMotivationEngine()
        
        # First encounter
        engine.update_familiarity(stimulus)
        entry1 = engine.get_familiarity_entry(stimulus)
        assert entry1 is not None
        familiarity1 = entry1.familiarity_score
        
        # Second encounter
        engine.update_familiarity(stimulus)
        entry2 = engine.get_familiarity_entry(stimulus)
        familiarity2 = entry2.familiarity_score
        
        # Third encounter
        engine.update_familiarity(stimulus)
        entry3 = engine.get_familiarity_entry(stimulus)
        familiarity3 = entry3.familiarity_score
        
        assert familiarity2 >= familiarity1, \
            f"Familiarity should increase: {familiarity1} -> {familiarity2}"
        assert familiarity3 >= familiarity2, \
            f"Familiarity should increase: {familiarity2} -> {familiarity3}"

    @given(
        stimulus=stimulus_strategy,
        config=intrinsic_motivation_config_strategy()
    )
    @settings(max_examples=100)
    def test_curiosity_level_decreases_with_decay(
        self, stimulus: Any, config: IntrinsicMotivationConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 8: Novelty Decay with Repeated Exposure
        
        For any stimulus, applying decay_curiosity SHALL decrease the curiosity_level.
        
        **Validates: Requirements 2.5**
        """
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # First encounter to create entry
        engine.update_familiarity(stimulus)
        entry_before = engine.get_familiarity_entry(stimulus)
        assert entry_before is not None
        curiosity_before = entry_before.curiosity_level
        
        # Apply decay
        engine.decay_curiosity(stimulus)
        entry_after = engine.get_familiarity_entry(stimulus)
        curiosity_after = entry_after.curiosity_level
        
        assert curiosity_after <= curiosity_before, \
            f"Curiosity level should decrease after decay: {curiosity_before} -> {curiosity_after}"

    @given(stimulus=stimulus_strategy)
    @settings(max_examples=100)
    def test_encounter_count_increases(self, stimulus: Any):
        """
        Feature: consciousness-system-deepening, Property 8: Novelty Decay with Repeated Exposure
        
        For any stimulus, the encounter_count SHALL increase with each update_familiarity call.
        
        **Validates: Requirements 2.3**
        """
        engine = IntrinsicMotivationEngine()
        
        # Multiple encounters
        for expected_count in range(1, 6):
            engine.update_familiarity(stimulus)
            entry = engine.get_familiarity_entry(stimulus)
            assert entry is not None
            assert entry.encounter_count == expected_count, \
                f"Encounter count should be {expected_count}, got {entry.encounter_count}"


# =============================================================================
# Property 9: Exploration-Exploitation Balance
# =============================================================================

class TestExplorationExploitationBalance:
    """
    Tests for Property 9: Exploration-Exploitation Balance
    
    *For any* action selection with multiple candidates, the selection probability 
    SHALL consider both expected information gain and known rewards, with the 
    balance controlled by a configurable exploration parameter.
    
    **Validates: Requirements 2.4**
    """

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=10, unique=True),
        context=context_strategy
    )
    @settings(max_examples=100)
    def test_action_selection_returns_valid_action(
        self, actions: List[str], context: Dict[str, Any]
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action selection, the selected action SHALL be one of the candidates.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        selected_action, score = engine.select_action_with_exploration(actions, context)
        
        assert selected_action in actions, \
            f"Selected action '{selected_action}' must be in candidates {actions}"
        assert 0.0 <= score <= 1.0, \
            f"Selection score {score} must be in [0.0, 1.0]"

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=5, unique=True),
        context=context_strategy
    )
    @settings(max_examples=100)
    def test_exploration_bonus_affects_selection(
        self, actions: List[str], context: Dict[str, Any]
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action selection, unexplored actions SHALL have higher exploration bonus.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        # All actions should have high exploration bonus initially (unexplored)
        bonuses = [engine.get_exploration_bonus(action, context) for action in actions]
        
        # All unexplored actions should have bonus of 1.0
        for action, bonus in zip(actions, bonuses):
            assert bonus == 1.0, \
                f"Unexplored action '{action}' should have bonus 1.0, got {bonus}"

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=5, unique=True),
        context=context_strategy,
        num_attempts=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_exploration_bonus_decreases_with_attempts(
        self, actions: List[str], context: Dict[str, Any], num_attempts: int
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action attempted multiple times, the exploration bonus SHALL decrease.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        action = actions[0]
        
        # Initial bonus
        initial_bonus = engine.get_exploration_bonus(action, context)
        
        # Attempt the action multiple times
        for _ in range(num_attempts):
            engine.update_action_exploration(action, context, information_gain=0.5)
        
        # Bonus should decrease
        final_bonus = engine.get_exploration_bonus(action, context)
        
        assert final_bonus <= initial_bonus, \
            f"Exploration bonus should decrease with attempts: {initial_bonus} -> {final_bonus}"

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=5, unique=True),
        context=context_strategy
    )
    @settings(max_examples=100)
    def test_known_rewards_affect_selection(
        self, actions: List[str], context: Dict[str, Any]
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action selection with known rewards, actions with higher known 
        rewards SHALL be more likely to be selected (when exploration is low).
        
        **Validates: Requirements 2.4**
        """
        # Use low exploration weight to emphasize exploitation
        config = IntrinsicMotivationConfig(exploration_weight=0.1)
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # Make all actions equally explored
        for action in actions:
            for _ in range(5):
                engine.update_action_exploration(action, context, information_gain=0.5)
        
        # Set very different known rewards
        known_rewards = {action: 0.1 for action in actions}
        best_action = actions[0]
        known_rewards[best_action] = 0.99
        
        selected_action, _ = engine.select_action_with_exploration(
            actions, context, known_rewards=known_rewards
        )
        
        # With low exploration weight, should select highest reward action
        assert selected_action == best_action, \
            f"With low exploration, should select highest reward action '{best_action}', " \
            f"got '{selected_action}'"

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=5, unique=True),
        context=context_strategy,
        exploration_weight=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_exploration_weight_controls_balance(
        self, actions: List[str], context: Dict[str, Any], exploration_weight: float
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any exploration_weight configuration, the selection SHALL balance 
        exploration and exploitation according to the weight.
        
        **Validates: Requirements 2.4**
        """
        config = IntrinsicMotivationConfig(exploration_weight=exploration_weight)
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # Set up scenario: one action is explored (low bonus), one is unexplored (high bonus)
        explored_action = actions[0]
        unexplored_action = actions[1]
        
        # Make explored_action familiar
        for _ in range(10):
            engine.update_action_exploration(explored_action, context, information_gain=0.3)
        
        # Set known rewards: explored action has higher known reward
        known_rewards = {
            explored_action: 0.9,
            unexplored_action: 0.1
        }
        
        # Get exploration bonuses
        explored_bonus = engine.get_exploration_bonus(explored_action, context)
        unexplored_bonus = engine.get_exploration_bonus(unexplored_action, context)
        
        # Calculate expected scores
        explored_score = (
            exploration_weight * explored_bonus +
            (1 - exploration_weight) * known_rewards[explored_action]
        )
        unexplored_score = (
            exploration_weight * unexplored_bonus +
            (1 - exploration_weight) * known_rewards[unexplored_action]
        )
        
        selected_action, actual_score = engine.select_action_with_exploration(
            [explored_action, unexplored_action], context, known_rewards=known_rewards
        )
        
        # Verify the selection matches expected behavior
        if explored_score > unexplored_score:
            expected_action = explored_action
        else:
            expected_action = unexplored_action
        
        assert selected_action == expected_action, \
            f"Selection should match score calculation: " \
            f"explored_score={explored_score:.3f}, unexplored_score={unexplored_score:.3f}, " \
            f"expected={expected_action}, got={selected_action}"

    @given(context=context_strategy)
    @settings(max_examples=100)
    def test_empty_actions_raises_error(self, context: Dict[str, Any]):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action selection with empty candidates, a ValueError SHALL be raised.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        with pytest.raises(ValueError):
            engine.select_action_with_exploration([], context)

    @given(
        action=action_strategy,
        context=context_strategy,
        info_gain=unit_float_strategy
    )
    @settings(max_examples=100)
    def test_action_exploration_info_updated(
        self, action: str, context: Dict[str, Any], info_gain: float
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action exploration update, the action's exploration info SHALL be updated.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        # Initially no info
        info_before = engine.get_action_exploration_info(action)
        assert info_before is None
        
        # Update exploration
        engine.update_action_exploration(action, context, information_gain=info_gain)
        
        # Now should have info
        info_after = engine.get_action_exploration_info(action)
        assert info_after is not None
        assert info_after.action == action
        assert info_after.attempt_count == 1

    @given(
        actions=st.lists(action_strategy, min_size=3, max_size=5, unique=True),
        context=context_strategy
    )
    @settings(max_examples=100)
    def test_selection_is_deterministic(
        self, actions: List[str], context: Dict[str, Any]
    ):
        """
        Feature: consciousness-system-deepening, Property 9: Exploration-Exploitation Balance
        
        For any action selection with the same state, the selection SHALL be deterministic.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine()
        
        # Same inputs should produce same outputs
        selected1, score1 = engine.select_action_with_exploration(actions, context)
        selected2, score2 = engine.select_action_with_exploration(actions, context)
        
        assert selected1 == selected2, \
            f"Selection should be deterministic: {selected1} != {selected2}"
        assert score1 == score2, \
            f"Score should be deterministic: {score1} != {score2}"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestIntrinsicMotivationIntegration:
    """
    Integration tests for IntrinsicMotivationEngine combining multiple properties.
    """

    @given(
        stimulus=stimulus_strategy,
        config=intrinsic_motivation_config_strategy()
    )
    @settings(max_examples=100)
    def test_full_stimulus_lifecycle(
        self, stimulus: Any, config: IntrinsicMotivationConfig
    ):
        """
        Test the full lifecycle of a stimulus: novel -> familiar -> decayed.
        
        **Validates: Requirements 2.1-2.5**
        """
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # Phase 1: Novel stimulus
        initial_novelty = engine.get_novelty_score(stimulus)
        assert initial_novelty == 1.0, "New stimulus should be maximally novel"
        
        # Phase 2: First encounter
        first_reward = engine.get_intrinsic_reward(stimulus)
        assert 0.0 <= first_reward <= 1.0
        
        # Phase 3: Multiple encounters
        rewards = [first_reward]
        for _ in range(5):
            reward = engine.get_intrinsic_reward(stimulus)
            rewards.append(reward)
        
        # Rewards should generally decrease
        assert rewards[-1] <= rewards[0], \
            f"Rewards should decrease over time: {rewards[0]} -> {rewards[-1]}"
        
        # Phase 4: Check final state
        final_novelty = engine.get_novelty_score(stimulus)
        assert final_novelty < initial_novelty, \
            f"Novelty should decrease: {initial_novelty} -> {final_novelty}"

    @given(
        actions=st.lists(action_strategy, min_size=2, max_size=5, unique=True),
        context=context_strategy,
        config=intrinsic_motivation_config_strategy()
    )
    @settings(max_examples=100)
    def test_exploration_exploitation_tradeoff(
        self, actions: List[str], context: Dict[str, Any], config: IntrinsicMotivationConfig
    ):
        """
        Test that exploration-exploitation tradeoff works correctly.
        
        **Validates: Requirements 2.4**
        """
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # Make first action very explored
        explored_action = actions[0]
        for _ in range(20):
            engine.update_action_exploration(explored_action, context, information_gain=0.3)
        
        # Keep second action unexplored
        unexplored_action = actions[1]
        
        # Get bonuses
        explored_bonus = engine.get_exploration_bonus(explored_action, context)
        unexplored_bonus = engine.get_exploration_bonus(unexplored_action, context)
        
        # Unexplored should have higher bonus
        assert unexplored_bonus > explored_bonus, \
            f"Unexplored action should have higher bonus: {unexplored_bonus} vs {explored_bonus}"

    @given(config=intrinsic_motivation_config_strategy())
    @settings(max_examples=100)
    def test_state_serialization_roundtrip(self, config: IntrinsicMotivationConfig):
        """
        Test that engine state can be serialized and restored.
        
        **Validates: Requirements 2.1-2.5**
        """
        engine = IntrinsicMotivationEngine(config=config.to_dict())
        
        # Add some state
        engine.update_familiarity("test_stimulus_1")
        engine.update_familiarity("test_stimulus_2")
        engine.update_action_exploration("action1", {"key": "value"}, 0.5)
        engine.get_intrinsic_reward("test_stimulus_1")
        
        # Serialize
        state_dict = engine.to_dict()
        
        # Create new engine and restore
        new_engine = IntrinsicMotivationEngine()
        new_engine.from_dict(state_dict)
        
        # Verify state matches
        assert new_engine._total_stimuli_encountered == engine._total_stimuli_encountered
        assert len(new_engine._familiarity_tracker) == len(engine._familiarity_tracker)
        assert len(new_engine._action_exploration) == len(engine._action_exploration)

    @given(
        predicted=numeric_outcome_strategy,
        actual=numeric_outcome_strategy,
        stimulus=stimulus_strategy
    )
    @settings(max_examples=100)
    def test_curiosity_and_novelty_independence(
        self, predicted: float, actual: float, stimulus: Any
    ):
        """
        Test that curiosity reward (prediction error) and novelty reward are independent.
        
        **Validates: Requirements 2.1, 2.2**
        """
        engine = IntrinsicMotivationEngine()
        
        # Curiosity reward is based on prediction error
        curiosity_reward = engine.calculate_curiosity_reward(predicted, actual)
        
        # Novelty/intrinsic reward is based on stimulus familiarity
        intrinsic_reward = engine.get_intrinsic_reward(stimulus)
        
        # Both should be valid
        assert 0.0 <= curiosity_reward <= 1.0
        assert 0.0 <= intrinsic_reward <= 1.0
        
        # They measure different things and can vary independently
        # (no specific relationship required, just both valid)
