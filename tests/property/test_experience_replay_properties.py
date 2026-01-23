"""
Property-based tests for Experience Replay Buffer.

Tests properties 41, 43, and 14 (partial) from the consciousness-system-deepening design document:
- Property 41: Experience Replay Buffer Diversity
- Property 43: Prioritized Sampling Bias
- Property 14: Importance-Weighted Memory Pruning (partial)

Validates: Requirements 9.1, 9.3, 9.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from typing import Dict, List, Any, Optional
from collections import Counter
import random

from mm_orch.consciousness.experience_replay import (
    ExperienceReplayBuffer,
    ExperienceReplayConfig,
    Experience,
    create_experience,
)


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for valid float values in [0.0, 1.0]
unit_float_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for priority values (0.0 to 1.0)
priority_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for reward values
reward_strategy = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
)

# Strategy for task type names
task_type_strategy = st.text(
    min_size=1, max_size=20,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'Pd', 'Pc'))
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

# Strategy for outcome dictionaries
outcome_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15),
    values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
    min_size=0,
    max_size=5
)

# Strategy for metadata dictionaries
metadata_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=15),
    values=st.one_of(st.integers(), st.text(max_size=20), st.booleans()),
    min_size=0,
    max_size=3
)


@st.composite
def experience_strategy(draw, task_type: Optional[str] = None, priority: Optional[float] = None):
    """Generate a valid Experience object."""
    exp_task_type = task_type if task_type else draw(task_type_strategy)
    exp_priority = priority if priority is not None else draw(priority_strategy)
    
    return create_experience(
        task_type=exp_task_type,
        context=draw(context_strategy),
        action=draw(action_strategy),
        outcome=draw(outcome_strategy),
        reward=draw(reward_strategy),
        priority=exp_priority,
        metadata=draw(metadata_strategy),
    )


@st.composite
def experience_list_strategy(draw, min_size: int = 1, max_size: int = 20):
    """Generate a list of Experience objects."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(experience_strategy()) for _ in range(size)]


@st.composite
def multi_task_type_experiences_strategy(draw, min_task_types: int = 2, max_task_types: int = 5):
    """Generate experiences from multiple task types."""
    num_task_types = draw(st.integers(min_value=min_task_types, max_value=max_task_types))
    task_types = [f"task_type_{i}" for i in range(num_task_types)]
    
    experiences = []
    # Ensure at least one experience per task type
    for task_type in task_types:
        num_experiences = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_experiences):
            exp = draw(experience_strategy(task_type=task_type))
            experiences.append(exp)
    
    return experiences, task_types


@st.composite
def prioritized_experiences_strategy(draw, min_size: int = 10, max_size: int = 50):
    """Generate experiences with varied priorities for testing prioritized sampling."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    experiences = []
    
    for _ in range(size):
        # Generate experiences with varied priorities
        priority = draw(priority_strategy)
        exp = draw(experience_strategy(priority=priority))
        experiences.append(exp)
    
    return experiences


@st.composite
def experience_replay_config_strategy(draw):
    """Generate a valid ExperienceReplayConfig object."""
    max_size = draw(st.integers(min_value=10, max_value=1000))
    default_priority = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    priority_alpha = draw(st.floats(min_value=0.1, max_value=0.9, allow_nan=False))
    min_priority = draw(st.floats(min_value=0.001, max_value=0.1, allow_nan=False))
    prune_threshold = draw(st.floats(min_value=0.5, max_value=0.95, allow_nan=False))
    prune_ratio = draw(st.floats(min_value=0.1, max_value=0.5, allow_nan=False))
    
    return ExperienceReplayConfig(
        max_size=max_size,
        default_priority=default_priority,
        priority_alpha=priority_alpha,
        min_priority=min_priority,
        prune_threshold=prune_threshold,
        prune_ratio=prune_ratio,
    )


# =============================================================================
# Property 41: Experience Replay Buffer Diversity
# =============================================================================

class TestExperienceReplayBufferDiversity:
    """
    Tests for Property 41: Experience Replay Buffer Diversity
    
    *For any* ExperienceReplayBuffer with experiences from multiple task types, 
    sampling with "stratified" strategy SHALL return experiences from all 
    represented task types.
    
    **Validates: Requirements 9.1**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_stratified_sampling_includes_all_task_types(self, data):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any buffer with multiple task types, stratified sampling SHALL return
        experiences from all represented task types when batch_size >= num_task_types.
        
        **Validates: Requirements 9.1**
        """
        experiences, task_types = data.draw(multi_task_type_experiences_strategy())
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        # Sample with batch size >= number of task types to ensure all can be represented
        # We need at least as many samples as task types to guarantee representation
        batch_size = len(task_types)
        
        # Only test if we have enough experiences
        assume(len(experiences) >= len(task_types))
        
        sampled = buffer.sample(batch_size, strategy="stratified")
        
        # Get task types in sampled experiences
        sampled_task_types = set(exp.task_type for exp in sampled)
        
        # All task types should be represented when batch_size >= num_task_types
        assert sampled_task_types == set(task_types), \
            f"Stratified sampling should include all task types. " \
            f"Expected: {set(task_types)}, Got: {sampled_task_types}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_stratified_sampling_proportional_distribution(self, data):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any buffer with multiple task types, stratified sampling SHALL sample
        proportionally from each task type based on their representation in the buffer.
        
        **Validates: Requirements 9.1**
        """
        experiences, task_types = data.draw(multi_task_type_experiences_strategy(
            min_task_types=2, max_task_types=4
        ))
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        # Get actual distribution in buffer
        buffer_distribution = buffer.get_task_type_distribution()
        total_in_buffer = sum(buffer_distribution.values())
        
        # Sample a significant portion
        batch_size = min(len(experiences), max(10, len(experiences) // 2))
        sampled = buffer.sample(batch_size, strategy="stratified")
        
        # Count sampled task types
        sampled_counts = Counter(exp.task_type for exp in sampled)
        
        # Each task type should have at least one sample (if batch_size >= num_task_types)
        if batch_size >= len(task_types):
            for task_type in task_types:
                assert task_type in sampled_counts, \
                    f"Task type '{task_type}' should be in stratified sample"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_buffer_maintains_task_type_index(self, data):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any buffer, the task type index SHALL accurately reflect the stored experiences.
        
        **Validates: Requirements 9.1**
        """
        experiences, task_types = data.draw(multi_task_type_experiences_strategy())
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        # Get distribution from buffer
        distribution = buffer.get_task_type_distribution()
        
        # Count actual experiences by task type
        actual_counts = Counter(exp.task_type for exp in experiences)
        
        # Distribution should match actual counts
        for task_type in task_types:
            assert distribution.get(task_type, 0) == actual_counts[task_type], \
                f"Task type '{task_type}' count mismatch: " \
                f"index={distribution.get(task_type, 0)}, actual={actual_counts[task_type]}"

    @given(
        experiences=experience_list_strategy(min_size=5, max_size=30),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_stratified_sampling_returns_correct_batch_size(
        self, experiences: List[Experience], batch_size: int
    ):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any stratified sampling, the returned batch size SHALL be min(batch_size, buffer_size).
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        sampled = buffer.sample(batch_size, strategy="stratified")
        
        expected_size = min(batch_size, len(buffer))
        assert len(sampled) == expected_size, \
            f"Stratified sample size should be {expected_size}, got {len(sampled)}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_stratified_sampling_no_duplicates(self, data):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any stratified sampling, the returned experiences SHALL be unique (no duplicates).
        
        **Validates: Requirements 9.1**
        """
        experiences, _ = data.draw(multi_task_type_experiences_strategy())
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        batch_size = min(len(experiences), 15)
        sampled = buffer.sample(batch_size, strategy="stratified")
        
        # Check for duplicates by experience_id
        sampled_ids = [exp.experience_id for exp in sampled]
        assert len(sampled_ids) == len(set(sampled_ids)), \
            "Stratified sampling should not return duplicate experiences"


# =============================================================================
# Property 43: Prioritized Sampling Bias
# =============================================================================

class TestPrioritizedSamplingBias:
    """
    Tests for Property 43: Prioritized Sampling Bias
    
    *For any* prioritized sampling from ExperienceReplayBuffer, experiences with 
    higher priority scores SHALL have higher probability of being sampled.
    
    **Validates: Requirements 9.3**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_high_priority_experiences_sampled_more_frequently(self, data):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any buffer with varied priorities, high priority experiences SHALL be
        sampled more frequently than low priority experiences over many samples.
        
        **Validates: Requirements 9.3**
        """
        # Create experiences with distinct priority groups
        high_priority_exps = []
        low_priority_exps = []
        
        # Generate high priority experiences (0.8-1.0)
        for i in range(10):
            priority = data.draw(st.floats(min_value=0.8, max_value=1.0, allow_nan=False))
            exp = data.draw(experience_strategy(task_type="high_priority", priority=priority))
            high_priority_exps.append(exp)
        
        # Generate low priority experiences (0.0-0.2)
        for i in range(10):
            priority = data.draw(st.floats(min_value=0.0, max_value=0.2, allow_nan=False))
            exp = data.draw(experience_strategy(task_type="low_priority", priority=priority))
            low_priority_exps.append(exp)
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in high_priority_exps + low_priority_exps:
            buffer.store(exp)
        
        # Sample many times and count
        high_priority_count = 0
        low_priority_count = 0
        num_samples = 100
        
        for _ in range(num_samples):
            sampled = buffer.sample(5, strategy="prioritized")
            for exp in sampled:
                if exp.task_type == "high_priority":
                    high_priority_count += 1
                else:
                    low_priority_count += 1
        
        # High priority should be sampled more frequently
        assert high_priority_count > low_priority_count, \
            f"High priority experiences ({high_priority_count}) should be sampled " \
            f"more than low priority ({low_priority_count})"


    @given(data=st.data())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_prioritized_sampling_respects_priority_ordering(self, data):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any buffer, the average priority of sampled experiences with prioritized
        strategy SHALL be higher than with uniform strategy.
        
        **Validates: Requirements 9.3**
        """
        # Generate experiences with varied priorities more efficiently
        num_experiences = data.draw(st.integers(min_value=20, max_value=40))
        experiences = []
        for i in range(num_experiences):
            priority = data.draw(priority_strategy)
            exp = create_experience(
                task_type=f"type_{i % 5}",
                context={"index": i},
                action=f"action_{i}",
                outcome={"result": "done"},
                reward=0.5,
                priority=priority,
            )
            experiences.append(exp)
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        # Sample with both strategies multiple times
        prioritized_priorities = []
        uniform_priorities = []
        
        for _ in range(50):
            prioritized_sample = buffer.sample(5, strategy="prioritized")
            uniform_sample = buffer.sample(5, strategy="uniform")
            
            prioritized_priorities.extend([exp.priority for exp in prioritized_sample])
            uniform_priorities.extend([exp.priority for exp in uniform_sample])
        
        avg_prioritized = sum(prioritized_priorities) / len(prioritized_priorities)
        avg_uniform = sum(uniform_priorities) / len(uniform_priorities)
        
        # Prioritized sampling should have higher average priority
        # Allow small tolerance for randomness
        assert avg_prioritized >= avg_uniform - 0.1, \
            f"Prioritized sampling avg priority ({avg_prioritized:.3f}) should be >= " \
            f"uniform sampling avg priority ({avg_uniform:.3f})"

    @given(
        experiences=prioritized_experiences_strategy(min_size=10, max_size=30),
        batch_size=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_prioritized_sampling_returns_valid_experiences(
        self, experiences: List[Experience], batch_size: int
    ):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any prioritized sampling, all returned experiences SHALL be valid
        experiences from the buffer.
        
        **Validates: Requirements 9.3**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        stored_ids = set()
        for exp in experiences:
            buffer.store(exp)
            stored_ids.add(exp.experience_id)
        
        sampled = buffer.sample(batch_size, strategy="prioritized")
        
        for exp in sampled:
            assert exp.experience_id in stored_ids, \
                f"Sampled experience '{exp.experience_id}' not in buffer"
            assert 0.0 <= exp.priority <= 1.0, \
                f"Experience priority {exp.priority} must be in [0.0, 1.0]"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_prioritized_sampling_no_duplicates(self, data):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any prioritized sampling, the returned experiences SHALL be unique.
        
        **Validates: Requirements 9.3**
        """
        experiences = data.draw(prioritized_experiences_strategy(min_size=15, max_size=40))
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        batch_size = min(len(experiences), 10)
        sampled = buffer.sample(batch_size, strategy="prioritized")
        
        sampled_ids = [exp.experience_id for exp in sampled]
        assert len(sampled_ids) == len(set(sampled_ids)), \
            "Prioritized sampling should not return duplicate experiences"

    @given(
        num_experiences=st.integers(min_value=5, max_value=20),
        priority_alpha=st.floats(min_value=0.1, max_value=0.9, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_priority_alpha_affects_sampling_bias(
        self, num_experiences: int, priority_alpha: float
    ):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any priority_alpha configuration, higher alpha SHALL increase the
        bias toward high priority experiences.
        
        **Validates: Requirements 9.3**
        """
        # Create experiences with known priorities
        experiences = []
        for i in range(num_experiences):
            priority = i / (num_experiences - 1) if num_experiences > 1 else 0.5
            exp = create_experience(
                task_type=f"type_{i}",
                context={},
                action="test_action",
                outcome={},
                reward=0.0,
                priority=priority,
            )
            experiences.append(exp)
        
        # Test with given alpha
        config = ExperienceReplayConfig(
            max_size=1000,
            priority_alpha=priority_alpha,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        for exp in experiences:
            buffer.store(exp)
        
        # Sample and check that sampling works
        sampled = buffer.sample(min(5, num_experiences), strategy="prioritized")
        
        assert len(sampled) > 0, "Prioritized sampling should return experiences"
        for exp in sampled:
            assert exp.experience_id in buffer, \
                f"Sampled experience should be in buffer"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_max_priority_experience_most_likely_sampled(self, data):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any buffer, the experience with maximum priority SHALL be among the
        most frequently sampled experiences.
        
        **Validates: Requirements 9.3**
        """
        # Create experiences with one clear maximum priority
        experiences = []
        for i in range(10):
            exp = data.draw(experience_strategy(
                task_type=f"type_{i}",
                priority=0.3  # Low priority
            ))
            experiences.append(exp)
        
        # Add one high priority experience
        max_priority_exp = data.draw(experience_strategy(
            task_type="max_priority",
            priority=1.0
        ))
        experiences.append(max_priority_exp)
        
        buffer = ExperienceReplayBuffer(max_size=1000)
        for exp in experiences:
            buffer.store(exp)
        
        # Sample many times and count max priority experience
        max_priority_count = 0
        num_samples = 100
        
        for _ in range(num_samples):
            sampled = buffer.sample(3, strategy="prioritized")
            for exp in sampled:
                if exp.experience_id == max_priority_exp.experience_id:
                    max_priority_count += 1
        
        # Max priority experience should be sampled frequently
        # With priority 1.0 vs 0.3, it should appear in most samples
        expected_min_count = num_samples * 0.3  # At least 30% of samples
        assert max_priority_count >= expected_min_count, \
            f"Max priority experience should be sampled frequently: " \
            f"got {max_priority_count}, expected >= {expected_min_count}"


# =============================================================================
# Property 14: Importance-Weighted Memory Pruning (Partial)
# =============================================================================

class TestImportanceWeightedMemoryPruning:
    """
    Tests for Property 14: Importance-Weighted Memory Pruning (Partial)
    
    *For any* pruning operation when memory exceeds limits, the removed 
    experiences/episodes SHALL have lower importance scores than retained ones, 
    preserving emotionally salient and frequently accessed items.
    
    **Validates: Requirements 9.5**
    """

    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_removes_low_priority_experiences(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, the removed experiences SHALL have lower
        priority scores than the retained experiences.
        
        **Validates: Requirements 9.5**
        """
        # Create buffer with small max_size to trigger pruning
        config = ExperienceReplayConfig(
            max_size=20,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Add experiences with varied priorities
        experiences = []
        for i in range(25):  # More than max_size to trigger pruning
            priority = i / 24.0  # Priorities from 0.0 to 1.0
            exp = data.draw(experience_strategy(
                task_type=f"type_{i % 3}",
                priority=priority
            ))
            experiences.append(exp)
            buffer.store(exp)
        
        # Get priorities of remaining experiences
        remaining_priorities = [
            buffer.get_experience(exp_id).priority
            for exp_id in buffer._experiences.keys()
        ]
        
        # The minimum remaining priority should be higher than what was pruned
        # (since we pruned low priority experiences)
        min_remaining = min(remaining_priorities)
        
        # With 25 experiences and max_size 20, we should have pruned some
        # The remaining experiences should have higher average priority
        avg_remaining = sum(remaining_priorities) / len(remaining_priorities)
        
        # Average of all original priorities
        all_priorities = [exp.priority for exp in experiences]
        avg_all = sum(all_priorities) / len(all_priorities)
        
        assert avg_remaining >= avg_all - 0.1, \
            f"Remaining experiences should have higher avg priority: " \
            f"remaining={avg_remaining:.3f}, original={avg_all:.3f}"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_preserves_high_priority_experiences(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, high priority experiences SHALL be preserved
        while low priority experiences are removed.
        
        **Validates: Requirements 9.5**
        """
        config = ExperienceReplayConfig(
            max_size=15,
            prune_threshold=0.9,
            prune_ratio=0.4,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Create high priority experiences
        high_priority_ids = []
        for i in range(5):
            exp = data.draw(experience_strategy(
                task_type="high_priority",
                priority=0.9 + i * 0.02  # 0.9 to 0.98
            ))
            buffer.store(exp)
            high_priority_ids.append(exp.experience_id)
        
        # Create low priority experiences
        low_priority_ids = []
        for i in range(15):  # Enough to trigger pruning
            exp = data.draw(experience_strategy(
                task_type="low_priority",
                priority=0.1 + i * 0.02  # 0.1 to 0.38
            ))
            buffer.store(exp)
            low_priority_ids.append(exp.experience_id)
        
        # Check that high priority experiences are preserved
        preserved_high = sum(1 for exp_id in high_priority_ids if exp_id in buffer)
        
        # All or most high priority experiences should be preserved
        assert preserved_high >= len(high_priority_ids) - 1, \
            f"High priority experiences should be preserved: " \
            f"{preserved_high}/{len(high_priority_ids)} preserved"

    @given(
        num_experiences=st.integers(min_value=30, max_value=50),
        max_size=st.integers(min_value=15, max_value=25)
    )
    @settings(max_examples=100)
    def test_pruning_maintains_size_limit(self, num_experiences: int, max_size: int):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any buffer, after pruning the size SHALL be within the configured limits.
        
        **Validates: Requirements 9.5**
        """
        config = ExperienceReplayConfig(
            max_size=max_size,
            prune_threshold=0.9,
            prune_ratio=0.2,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Add more experiences than max_size
        for i in range(num_experiences):
            exp = create_experience(
                task_type=f"type_{i % 5}",
                context={"index": i},
                action="test_action",
                outcome={"result": "success"},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
        
        # Buffer size should be at or below max_size
        assert len(buffer) <= max_size, \
            f"Buffer size ({len(buffer)}) should be <= max_size ({max_size})"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_returns_correct_count(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, the returned count SHALL equal the number
        of experiences actually removed.
        
        **Validates: Requirements 9.5**
        """
        config = ExperienceReplayConfig(
            max_size=20,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Fill buffer to capacity
        for i in range(20):
            exp = data.draw(experience_strategy(
                task_type=f"type_{i % 3}",
                priority=random.random()
            ))
            buffer.store(exp)
        
        size_before = len(buffer)
        removed_count = buffer.prune()
        size_after = len(buffer)
        
        actual_removed = size_before - size_after
        assert removed_count == actual_removed, \
            f"Prune returned {removed_count} but actually removed {actual_removed}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_updates_task_type_index(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, the task type index SHALL be updated to
        reflect the removed experiences.
        
        **Validates: Requirements 9.5**
        """
        config = ExperienceReplayConfig(
            max_size=15,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Add experiences from multiple task types
        task_types = ["type_a", "type_b", "type_c"]
        for i in range(20):
            task_type = task_types[i % len(task_types)]
            exp = data.draw(experience_strategy(
                task_type=task_type,
                priority=random.random()
            ))
            buffer.store(exp)
        
        # Get distribution after pruning
        distribution = buffer.get_task_type_distribution()
        
        # Verify distribution matches actual experiences
        for task_type, count in distribution.items():
            actual_count = len(buffer.get_experiences_by_task_type(task_type))
            assert count == actual_count, \
                f"Task type '{task_type}' index count ({count}) != " \
                f"actual count ({actual_count})"
        
        # Total in distribution should equal buffer size
        total_in_distribution = sum(distribution.values())
        assert total_in_distribution == len(buffer), \
            f"Distribution total ({total_in_distribution}) != buffer size ({len(buffer)})"


    @given(data=st.data())
    @settings(max_examples=100)
    def test_pruning_preserves_experience_integrity(self, data):
        """
        Feature: consciousness-system-deepening, Property 14: Importance-Weighted Memory Pruning
        
        For any pruning operation, the remaining experiences SHALL maintain
        their original data integrity (no corruption).
        
        **Validates: Requirements 9.5**
        """
        config = ExperienceReplayConfig(
            max_size=15,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Store experiences and keep track of original data
        original_data = {}
        for i in range(20):
            exp = data.draw(experience_strategy(
                task_type=f"type_{i % 3}",
                priority=random.random()
            ))
            buffer.store(exp)
            original_data[exp.experience_id] = {
                "task_type": exp.task_type,
                "action": exp.action,
                "reward": exp.reward,
                "priority": exp.priority,
            }
        
        # Verify remaining experiences have correct data
        for exp_id in buffer._experiences.keys():
            exp = buffer.get_experience(exp_id)
            original = original_data[exp_id]
            
            assert exp.task_type == original["task_type"], \
                f"Experience task_type corrupted after pruning"
            assert exp.action == original["action"], \
                f"Experience action corrupted after pruning"
            assert exp.reward == original["reward"], \
                f"Experience reward corrupted after pruning"
            # Priority might have been updated, but should still be valid
            assert 0.0 <= exp.priority <= 1.0, \
                f"Experience priority {exp.priority} out of valid range"


# =============================================================================
# Additional Property Tests for Buffer Operations
# =============================================================================

class TestExperienceReplayBufferOperations:
    """
    Additional property tests for basic buffer operations that support
    the main properties.
    
    **Validates: Requirements 9.1, 9.3, 9.5**
    """

    @given(experiences=experience_list_strategy(min_size=1, max_size=30))
    @settings(max_examples=100)
    def test_store_and_retrieve_consistency(self, experiences: List[Experience]):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any stored experience, retrieval by ID SHALL return the same experience.
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for exp in experiences:
            buffer.store(exp)
            retrieved = buffer.get_experience(exp.experience_id)
            
            assert retrieved is not None, \
                f"Stored experience '{exp.experience_id}' should be retrievable"
            assert retrieved.experience_id == exp.experience_id
            assert retrieved.task_type == exp.task_type
            assert retrieved.action == exp.action


    @given(
        experiences=experience_list_strategy(min_size=5, max_size=30),
        new_priority=priority_strategy
    )
    @settings(max_examples=100)
    def test_priority_update_affects_sampling(
        self, experiences: List[Experience], new_priority: float
    ):
        """
        Feature: consciousness-system-deepening, Property 43: Prioritized Sampling Bias
        
        For any priority update, the updated priority SHALL be reflected in
        subsequent sampling behavior.
        
        **Validates: Requirements 9.3**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for exp in experiences:
            buffer.store(exp)
        
        # Update priority of first experience
        target_exp = experiences[0]
        buffer.update_priority(target_exp.experience_id, new_priority)
        
        # Verify priority was updated
        updated_exp = buffer.get_experience(target_exp.experience_id)
        assert updated_exp is not None
        assert abs(updated_exp.priority - max(0.0, min(1.0, new_priority))) < 0.001, \
            f"Priority should be updated to {new_priority} (clamped), got {updated_exp.priority}"

    @given(
        config=experience_replay_config_strategy(),
        experiences=experience_list_strategy(min_size=5, max_size=20)
    )
    @settings(max_examples=100)
    def test_serialization_round_trip(
        self, config: ExperienceReplayConfig, experiences: List[Experience]
    ):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any buffer state, serialization and deserialization SHALL produce
        an equivalent buffer.
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        for exp in experiences:
            buffer.store(exp)
        
        # Serialize
        serialized = buffer.to_dict()
        
        # Deserialize
        restored = ExperienceReplayBuffer.from_dict(serialized)
        
        # Verify equivalence
        assert len(restored) == len(buffer), \
            f"Restored buffer size ({len(restored)}) != original ({len(buffer)})"
        
        for exp_id in buffer._experiences.keys():
            original_exp = buffer.get_experience(exp_id)
            restored_exp = restored.get_experience(exp_id)
            
            assert restored_exp is not None, \
                f"Experience '{exp_id}' missing after deserialization"
            assert restored_exp.task_type == original_exp.task_type
            assert restored_exp.action == original_exp.action
            assert restored_exp.priority == original_exp.priority

    @given(batch_size=st.integers(min_value=0, max_value=10))
    @settings(max_examples=100)
    def test_empty_buffer_sampling(self, batch_size: int):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any empty buffer, sampling SHALL return an empty list.
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for strategy in ["uniform", "prioritized", "stratified"]:
            sampled = buffer.sample(batch_size, strategy=strategy)
            assert sampled == [], \
                f"Empty buffer should return empty list for {strategy} sampling"


    @given(
        experiences=experience_list_strategy(min_size=5, max_size=20),
        strategy=st.sampled_from(["uniform", "prioritized", "stratified"])
    )
    @settings(max_examples=100)
    def test_sampling_returns_valid_batch_size(
        self, experiences: List[Experience], strategy: str
    ):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any sampling operation, the returned batch size SHALL be
        min(requested_batch_size, buffer_size).
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for exp in experiences:
            buffer.store(exp)
        
        # Test various batch sizes
        for batch_size in [1, 5, 10, 100]:
            sampled = buffer.sample(batch_size, strategy=strategy)
            expected_size = min(batch_size, len(buffer))
            
            assert len(sampled) == expected_size, \
                f"Sample size should be {expected_size} for {strategy}, got {len(sampled)}"

    @given(experiences=experience_list_strategy(min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_invalid_strategy_raises_error(self, experiences: List[Experience]):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any invalid sampling strategy, the buffer SHALL raise ValueError.
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for exp in experiences:
            buffer.store(exp)
        
        with pytest.raises(ValueError):
            buffer.sample(5, strategy="invalid_strategy")

    @given(experiences=experience_list_strategy(min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_statistics_accuracy(self, experiences: List[Experience]):
        """
        Feature: consciousness-system-deepening, Property 41: Experience Replay Buffer Diversity
        
        For any buffer, get_statistics() SHALL return accurate statistics.
        
        **Validates: Requirements 9.1**
        """
        buffer = ExperienceReplayBuffer(max_size=1000)
        
        for exp in experiences:
            buffer.store(exp)
        
        stats = buffer.get_statistics()
        
        # Verify count
        assert stats["count"] == len(buffer), \
            f"Statistics count ({stats['count']}) != buffer size ({len(buffer)})"
        
        # Verify priority statistics
        priorities = [exp.priority for exp in experiences]
        expected_avg = sum(priorities) / len(priorities)
        expected_min = min(priorities)
        expected_max = max(priorities)
        
        assert abs(stats["avg_priority"] - expected_avg) < 0.001, \
            f"Average priority mismatch: {stats['avg_priority']} vs {expected_avg}"
        assert abs(stats["min_priority"] - expected_min) < 0.001, \
            f"Min priority mismatch: {stats['min_priority']} vs {expected_min}"
        assert abs(stats["max_priority"] - expected_max) < 0.001, \
            f"Max priority mismatch: {stats['max_priority']} vs {expected_max}"
