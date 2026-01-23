"""
Property-based tests for Curriculum Learning System.

Tests properties 1-5 from the consciousness-system-deepening design document:
- Property 1: Task Difficulty Estimation Validity
- Property 2: Zone of Proximal Development Assessment Consistency
- Property 3: Scaffolding Recommendation Trigger
- Property 4: Capability Evolution Based on Task Outcomes
- Property 5: Consecutive Failure Handling

Validates: Requirements 1.1-1.6
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List

from mm_orch.consciousness.curriculum import (
    CurriculumLearningSystem,
    CurriculumConfig,
    Task,
    TaskDifficulty,
    ZPDAssessment,
    CapabilityDimension,
    TASK_TYPE_CAPABILITY_WEIGHTS,
)


# =============================================================================
# Hypothesis Strategies for Test Data Generation
# =============================================================================

# Strategy for valid complexity/capability values (0.0 to 1.0)
unit_float_strategy = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# Strategy for valid task types
task_type_strategy = st.sampled_from(list(TASK_TYPE_CAPABILITY_WEIGHTS.keys()))

# Strategy for capability dimension names
capability_dimension_strategy = st.sampled_from([d.value for d in CapabilityDimension])


# Strategy for generating valid required capabilities dictionaries
@st.composite
def required_capabilities_strategy(draw):
    """Generate a dictionary of required capabilities."""
    num_dims = draw(st.integers(min_value=0, max_value=len(CapabilityDimension)))
    dims = draw(st.lists(
        capability_dimension_strategy,
        min_size=num_dims,
        max_size=num_dims,
        unique=True
    ))
    levels = draw(st.lists(
        unit_float_strategy,
        min_size=num_dims,
        max_size=num_dims
    ))
    return dict(zip(dims, levels))


# Strategy for generating valid Task objects
@st.composite
def task_strategy(draw):
    """Generate a valid Task object."""
    task_id = draw(st.text(
        min_size=1, max_size=20,
        alphabet=st.characters(whitelist_categories=('L', 'N', 'Pd'))
    ))
    task_type = draw(task_type_strategy)
    complexity = draw(unit_float_strategy)
    required_caps = draw(required_capabilities_strategy())
    
    return Task(
        task_id=task_id,
        task_type=task_type,
        complexity=complexity,
        required_capabilities=required_caps,
        metadata={}
    )


# Strategy for generating valid CurriculumConfig objects
@st.composite
def curriculum_config_strategy(draw):
    """Generate a valid CurriculumConfig object."""
    zpd_lower = draw(st.floats(min_value=0.05, max_value=0.3, allow_nan=False))
    zpd_upper = draw(st.floats(min_value=zpd_lower + 0.05, max_value=0.6, allow_nan=False))
    growth_rate = draw(st.floats(min_value=0.01, max_value=0.2, allow_nan=False))
    decay_rate = draw(st.floats(min_value=0.01, max_value=0.1, allow_nan=False))
    failure_threshold = draw(st.integers(min_value=1, max_value=10))
    
    return CurriculumConfig(
        zpd_lower_threshold=zpd_lower,
        zpd_upper_threshold=zpd_upper,
        capability_growth_rate=growth_rate,
        capability_decay_rate=decay_rate,
        consecutive_failure_threshold=failure_threshold,
    )


# =============================================================================
# Property 1: Task Difficulty Estimation Validity
# =============================================================================

class TestTaskDifficultyEstimationValidity:
    """
    Tests for Property 1: Task Difficulty Estimation Validity
    
    *For any* task provided to the CurriculumLearningSystem, the estimated 
    difficulty SHALL be a valid TaskDifficulty object with overall_difficulty 
    in range [0.0, 1.0], and the difficulty score SHALL be deterministic for 
    the same task parameters.
    
    **Validates: Requirements 1.1**
    """
    
    @given(task=task_strategy())
    @settings(max_examples=100)
    def test_difficulty_estimation_returns_valid_object(self, task: Task):
        """
        Feature: consciousness-system-deepening, Property 1: Task Difficulty Estimation Validity
        
        For any task, estimate_task_difficulty SHALL return a valid TaskDifficulty
        object with all fields properly populated.
        
        **Validates: Requirements 1.1**
        """
        system = CurriculumLearningSystem()
        
        difficulty = system.estimate_task_difficulty(task)
        
        # Verify return type
        assert isinstance(difficulty, TaskDifficulty), \
            "estimate_task_difficulty must return a TaskDifficulty object"
        
        # Verify all fields are present and valid
        assert isinstance(difficulty.complexity, float)
        assert isinstance(difficulty.cognitive_load, float)
        assert isinstance(difficulty.overall_difficulty, float)
        assert isinstance(difficulty.required_capabilities, dict)
    
    @given(task=task_strategy())
    @settings(max_examples=100)
    def test_difficulty_overall_in_valid_range(self, task: Task):
        """
        Feature: consciousness-system-deepening, Property 1: Task Difficulty Estimation Validity
        
        For any task, the overall_difficulty SHALL be in range [0.0, 1.0].
        
        **Validates: Requirements 1.1**
        """
        system = CurriculumLearningSystem()
        
        difficulty = system.estimate_task_difficulty(task)
        
        assert 0.0 <= difficulty.overall_difficulty <= 1.0, \
            f"overall_difficulty {difficulty.overall_difficulty} must be in [0.0, 1.0]"
        assert 0.0 <= difficulty.complexity <= 1.0, \
            f"complexity {difficulty.complexity} must be in [0.0, 1.0]"
        assert 0.0 <= difficulty.cognitive_load <= 1.0, \
            f"cognitive_load {difficulty.cognitive_load} must be in [0.0, 1.0]"
        
        # All required capability levels must also be in valid range
        for dim, level in difficulty.required_capabilities.items():
            assert 0.0 <= level <= 1.0, \
                f"required_capabilities[{dim}] = {level} must be in [0.0, 1.0]"

    
    @given(task=task_strategy())
    @settings(max_examples=100)
    def test_difficulty_estimation_is_deterministic(self, task: Task):
        """
        Feature: consciousness-system-deepening, Property 1: Task Difficulty Estimation Validity
        
        For the same task parameters, the difficulty score SHALL be deterministic
        (calling estimate_task_difficulty multiple times returns the same result).
        
        **Validates: Requirements 1.1**
        """
        system = CurriculumLearningSystem()
        
        # Call multiple times with the same task
        difficulty1 = system.estimate_task_difficulty(task)
        difficulty2 = system.estimate_task_difficulty(task)
        difficulty3 = system.estimate_task_difficulty(task)
        
        # All calls should return identical results
        assert difficulty1.overall_difficulty == difficulty2.overall_difficulty, \
            "Difficulty estimation must be deterministic"
        assert difficulty2.overall_difficulty == difficulty3.overall_difficulty, \
            "Difficulty estimation must be deterministic"
        assert difficulty1.complexity == difficulty2.complexity
        assert difficulty1.cognitive_load == difficulty2.cognitive_load
        assert difficulty1.required_capabilities == difficulty2.required_capabilities


# =============================================================================
# Property 2: Zone of Proximal Development Assessment Consistency
# =============================================================================

class TestZPDAssessmentConsistency:
    """
    Tests for Property 2: Zone of Proximal Development Assessment Consistency
    
    *For any* task and capability level, the ZPD assessment SHALL correctly 
    identify whether the task is within ZPD based on the configured thresholds, 
    and the difficulty_gap SHALL equal (task_difficulty - capability_level).
    
    **Validates: Requirements 1.2**
    """
    
    @given(task=task_strategy(), config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_zpd_assessment_returns_valid_object(self, task: Task, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 2: ZPD Assessment Consistency
        
        For any task, is_in_zpd SHALL return a valid ZPDAssessment object.
        
        **Validates: Requirements 1.2**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        assessment = system.is_in_zpd(task)
        
        # Verify return type
        assert isinstance(assessment, ZPDAssessment), \
            "is_in_zpd must return a ZPDAssessment object"
        
        # Verify all fields are present
        assert isinstance(assessment.in_zpd, bool)
        assert isinstance(assessment.difficulty_gap, float)
        assert isinstance(assessment.recommendations, list)
        assert assessment.suggested_scaffolding is None or isinstance(assessment.suggested_scaffolding, list)

    
    @given(task=task_strategy(), config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_zpd_threshold_logic(self, task: Task, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 2: ZPD Assessment Consistency
        
        For any task, in_zpd SHALL be True if and only if the difficulty_gap
        is within the configured ZPD thresholds [zpd_lower_threshold, zpd_upper_threshold].
        
        **Validates: Requirements 1.2**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        assessment = system.is_in_zpd(task)
        
        # Verify ZPD logic is consistent with thresholds
        expected_in_zpd = (
            config.zpd_lower_threshold <= assessment.difficulty_gap <= config.zpd_upper_threshold
        )
        
        assert assessment.in_zpd == expected_in_zpd, \
            f"in_zpd={assessment.in_zpd} but difficulty_gap={assessment.difficulty_gap} " \
            f"with thresholds [{config.zpd_lower_threshold}, {config.zpd_upper_threshold}]"
    
    @given(
        task_type=task_type_strategy,
        complexity=unit_float_strategy,
        dimension=capability_dimension_strategy,
        capability_level=unit_float_strategy
    )
    @settings(max_examples=100)
    def test_difficulty_gap_calculation(
        self, 
        task_type: str, 
        complexity: float, 
        dimension: str,
        capability_level: float
    ):
        """
        Feature: consciousness-system-deepening, Property 2: ZPD Assessment Consistency
        
        For any task with a single required capability, the difficulty_gap SHALL
        equal (required_level - capability_level).
        
        **Validates: Requirements 1.2**
        """
        # Create a task with a single explicit required capability
        task = Task(
            task_id="test_task",
            task_type=task_type,
            complexity=complexity,
            required_capabilities={dimension: complexity}  # Use complexity as required level
        )
        
        system = CurriculumLearningSystem()
        
        # Set the capability level for the dimension
        system._capability_levels[dimension] = capability_level
        
        assessment = system.is_in_zpd(task)
        difficulty = system.estimate_task_difficulty(task)
        
        # The difficulty gap should be based on the required capability vs current level
        # For a single-dimension task, gap = required_level - capability_level
        expected_gap = difficulty.required_capabilities.get(dimension, complexity) - capability_level
        
        # Allow small floating point tolerance
        assert abs(assessment.difficulty_gap - expected_gap) < 0.01, \
            f"difficulty_gap={assessment.difficulty_gap} but expected {expected_gap}"


# =============================================================================
# Property 3: Scaffolding Recommendation Trigger
# =============================================================================

class TestScaffoldingRecommendationTrigger:
    """
    Tests for Property 3: Scaffolding Recommendation Trigger
    
    *For any* task where difficulty exceeds capability by more than the 
    configured threshold, the CurriculumLearningSystem SHALL return non-empty 
    scaffolding recommendations.
    
    **Validates: Requirements 1.3**
    """
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_scaffolding_triggered_when_too_difficult(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 3: Scaffolding Recommendation Trigger
        
        For any task where difficulty exceeds capability by more than zpd_upper_threshold,
        suggest_scaffolding SHALL return a non-empty list of sub-tasks.
        
        **Validates: Requirements 1.3**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set low capability levels to ensure task is too difficult
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.1
        
        # Create a high-difficulty task
        task = Task(
            task_id="difficult_task",
            task_type="reasoning",
            complexity=0.9,
            required_capabilities={"reasoning": 0.9, "memory": 0.8}
        )
        
        # Verify the task is indeed too difficult (outside ZPD)
        assessment = system.is_in_zpd(task)
        assume(assessment.difficulty_gap > config.zpd_upper_threshold)
        
        # Scaffolding should be recommended
        scaffolding = system.suggest_scaffolding(task)
        
        assert isinstance(scaffolding, list), \
            "suggest_scaffolding must return a list"
        assert len(scaffolding) > 0, \
            f"When difficulty_gap ({assessment.difficulty_gap}) > zpd_upper_threshold " \
            f"({config.zpd_upper_threshold}), scaffolding must be non-empty"
        
        # All scaffolded tasks should be valid Task objects
        for sub_task in scaffolding:
            assert isinstance(sub_task, Task)
            assert sub_task.task_id.startswith(task.task_id)

    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_no_scaffolding_when_in_zpd(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 3: Scaffolding Recommendation Trigger
        
        For any task within ZPD, suggest_scaffolding SHALL return an empty list.
        
        **Validates: Requirements 1.3**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set capability levels to match task difficulty (within ZPD)
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        # Create a task that should be within ZPD
        # Target difficulty gap in the middle of ZPD range
        target_gap = (config.zpd_lower_threshold + config.zpd_upper_threshold) / 2
        task_difficulty = 0.5 + target_gap
        
        task = Task(
            task_id="zpd_task",
            task_type="reasoning",
            complexity=min(0.95, task_difficulty),
            required_capabilities={"reasoning": min(0.95, task_difficulty)}
        )
        
        assessment = system.is_in_zpd(task)
        
        # Only test if task is actually in ZPD
        assume(assessment.in_zpd)
        
        # No scaffolding should be needed
        scaffolding = system.suggest_scaffolding(task)
        
        assert scaffolding == [], \
            f"When task is in ZPD (gap={assessment.difficulty_gap}), " \
            f"scaffolding should be empty but got {len(scaffolding)} tasks"
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_scaffolded_tasks_are_easier(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 3: Scaffolding Recommendation Trigger
        
        For any scaffolded sub-tasks, each sub-task SHALL have lower complexity
        than the original task.
        
        **Validates: Requirements 1.3**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set low capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.1
        
        # Create a difficult task
        task = Task(
            task_id="hard_task",
            task_type="reasoning",
            complexity=0.9,
            required_capabilities={"reasoning": 0.9}
        )
        
        assessment = system.is_in_zpd(task)
        assume(assessment.difficulty_gap > config.zpd_upper_threshold)
        
        scaffolding = system.suggest_scaffolding(task)
        assume(len(scaffolding) > 0)
        
        # Each scaffolded task should be easier
        for sub_task in scaffolding:
            assert sub_task.complexity <= task.complexity, \
                f"Scaffolded task complexity ({sub_task.complexity}) " \
                f"should be <= original ({task.complexity})"


# =============================================================================
# Property 4: Capability Evolution Based on Task Outcomes
# =============================================================================

class TestCapabilityEvolution:
    """
    Tests for Property 4: Capability Evolution Based on Task Outcomes
    
    *For any* sequence of task completions, capability levels SHALL increase 
    after successful tasks and decrease after failed tasks, with the magnitude 
    proportional to the configured growth/decay rates. Consecutive successes 
    at a difficulty level SHALL increase the difficulty threshold.
    
    **Validates: Requirements 1.4, 1.5**
    """
    
    @given(
        task_type=task_type_strategy,
        score=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=100)
    def test_capability_increases_on_success(
        self, task_type: str, score: float, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 4: Capability Evolution
        
        For any successful task completion, capability levels SHALL increase.
        
        **Validates: Requirements 1.4**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Get initial capability levels for affected dimensions
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
            task_type, {"reasoning": 0.5, "language": 0.5}
        )
        initial_levels = {
            dim: system.get_capability_level(dim) 
            for dim in weights.keys()
        }
        
        # Complete task successfully
        updates = system.update_capabilities(task_type, success=True, score=score)
        
        # Verify capabilities increased for affected dimensions
        for dim, weight in weights.items():
            if dim in updates:
                initial = initial_levels[dim]
                new_level = updates[dim]
                
                # Capability should increase (or stay at max 1.0)
                assert new_level >= initial or initial >= 0.99, \
                    f"Capability for {dim} should increase on success: " \
                    f"{initial} -> {new_level}"

    
    @given(
        task_type=task_type_strategy,
        score=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=100)
    def test_capability_decreases_on_failure(
        self, task_type: str, score: float, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 4: Capability Evolution
        
        For any failed task completion, capability levels SHALL decrease.
        
        **Validates: Requirements 1.4**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels above minimum to allow decrease
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        # Get initial capability levels for affected dimensions
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
            task_type, {"reasoning": 0.5, "language": 0.5}
        )
        initial_levels = {
            dim: system.get_capability_level(dim) 
            for dim in weights.keys()
        }
        
        # Complete task with failure
        updates = system.update_capabilities(task_type, success=False, score=score)
        
        # Verify capabilities decreased for affected dimensions
        for dim, weight in weights.items():
            if dim in updates:
                initial = initial_levels[dim]
                new_level = updates[dim]
                
                # Capability should decrease (or stay at min 0.0)
                assert new_level <= initial or initial <= 0.01, \
                    f"Capability for {dim} should decrease on failure: " \
                    f"{initial} -> {new_level}"
    
    @given(
        task_type=task_type_strategy,
        score1=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        score2=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=100)
    def test_growth_proportional_to_score(
        self, task_type: str, score1: float, score2: float, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 4: Capability Evolution
        
        For any two successful tasks, the capability growth SHALL be proportional
        to the score (higher score = more growth).
        
        **Validates: Requirements 1.4**
        """
        # Ensure scores are different enough to compare
        assume(abs(score1 - score2) > 0.1)
        
        # Create two separate systems with same initial state
        system1 = CurriculumLearningSystem(config=config.to_dict())
        system2 = CurriculumLearningSystem(config=config.to_dict())
        
        # Get a dimension affected by this task type
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
            task_type, {"reasoning": 0.5, "language": 0.5}
        )
        dim = list(weights.keys())[0]
        
        initial_level = system1.get_capability_level(dim)
        
        # Complete tasks with different scores
        updates1 = system1.update_capabilities(task_type, success=True, score=score1)
        updates2 = system2.update_capabilities(task_type, success=True, score=score2)
        
        if dim in updates1 and dim in updates2:
            growth1 = updates1[dim] - initial_level
            growth2 = updates2[dim] - initial_level
            
            # Higher score should produce more growth (unless at ceiling)
            if initial_level < 0.95:
                if score1 > score2:
                    assert growth1 >= growth2, \
                        f"Higher score ({score1}) should produce more growth than ({score2})"
                else:
                    assert growth2 >= growth1, \
                        f"Higher score ({score2}) should produce more growth than ({score1})"

    
    @given(
        task_type=task_type_strategy,
        num_successes=st.integers(min_value=2, max_value=5),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=100)
    def test_consecutive_successes_increase_difficulty_threshold(
        self, task_type: str, num_successes: int, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 4: Capability Evolution
        
        Consecutive successes at a difficulty level SHALL increase the 
        difficulty threshold.
        
        **Validates: Requirements 1.5**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Get a dimension affected by this task type
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
            task_type, {"reasoning": 0.5, "language": 0.5}
        )
        dim = list(weights.keys())[0]
        
        initial_threshold = system.get_difficulty_threshold(dim)
        
        # Complete multiple successful tasks with high scores
        for _ in range(num_successes):
            system.update_capabilities(task_type, success=True, score=0.9)
        
        new_threshold = system.get_difficulty_threshold(dim)
        
        # Threshold should increase (or stay at max 1.0)
        assert new_threshold >= initial_threshold or initial_threshold >= 0.99, \
            f"Difficulty threshold should increase after {num_successes} successes: " \
            f"{initial_threshold} -> {new_threshold}"
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_success_resets_consecutive_failures(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 4: Capability Evolution
        
        A successful task completion SHALL reset the consecutive failure counter.
        
        **Validates: Requirements 1.4**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task_type = "reasoning"
        dim = "reasoning"
        
        # Accumulate some failures
        for _ in range(2):
            system.update_capabilities(task_type, success=False, score=0.2)
        
        failures_before = system.get_consecutive_failures(dim)
        assert failures_before > 0, "Should have accumulated failures"
        
        # Complete a successful task
        system.update_capabilities(task_type, success=True, score=0.8)
        
        failures_after = system.get_consecutive_failures(dim)
        assert failures_after == 0, \
            f"Success should reset consecutive failures: {failures_before} -> {failures_after}"


# =============================================================================
# Property 5: Consecutive Failure Handling
# =============================================================================

class TestConsecutiveFailureHandling:
    """
    Tests for Property 5: Consecutive Failure Handling
    
    *For any* sequence of N consecutive failures (where N >= configured threshold), 
    the CurriculumLearningSystem SHALL reduce the difficulty threshold and the 
    next ZPD assessment SHALL recommend easier tasks.
    
    **Validates: Requirements 1.6**
    """
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_consecutive_failures_detected(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 5: Consecutive Failure Handling
        
        For any sequence of N consecutive failures where N >= threshold,
        check_consecutive_failures SHALL return True.
        
        **Validates: Requirements 1.6**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task_type = "reasoning"
        dim = "reasoning"
        
        # Initially should not exceed threshold
        assert not system.check_consecutive_failures(dim), \
            "Should not exceed threshold initially"
        
        # Accumulate failures up to threshold
        for i in range(config.consecutive_failure_threshold):
            system.update_capabilities(task_type, success=False, score=0.1)
        
        # Now should exceed threshold
        assert system.check_consecutive_failures(dim), \
            f"Should exceed threshold after {config.consecutive_failure_threshold} failures"
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_difficulty_threshold_reduced_after_consecutive_failures(
        self, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 5: Consecutive Failure Handling
        
        For any sequence of N consecutive failures where N >= threshold,
        reduce_difficulty_threshold SHALL reduce the threshold.
        
        **Validates: Requirements 1.6**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task_type = "reasoning"
        dim = "reasoning"
        
        initial_threshold = system.get_difficulty_threshold(dim)
        
        # Accumulate failures to exceed threshold
        for _ in range(config.consecutive_failure_threshold):
            system.update_capabilities(task_type, success=False, score=0.1)
        
        # Reduce difficulty threshold
        new_threshold = system.reduce_difficulty_threshold(dim)
        
        # Threshold should be reduced (unless already at minimum)
        assert new_threshold <= initial_threshold or initial_threshold <= 0.11, \
            f"Difficulty threshold should decrease: {initial_threshold} -> {new_threshold}"

    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_remedial_tasks_suggested_after_consecutive_failures(
        self, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 5: Consecutive Failure Handling
        
        For any sequence of N consecutive failures where N >= threshold,
        suggest_remedial_tasks SHALL return non-empty list of easier tasks.
        
        **Validates: Requirements 1.6**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task_type = "reasoning"
        dim = "reasoning"
        
        # Accumulate failures to exceed threshold
        for _ in range(config.consecutive_failure_threshold):
            system.update_capabilities(task_type, success=False, score=0.1)
        
        # Should now recommend remedial tasks
        assert system.check_consecutive_failures(dim)
        
        remedial_tasks = system.suggest_remedial_tasks(dim)
        
        assert isinstance(remedial_tasks, list), \
            "suggest_remedial_tasks must return a list"
        assert len(remedial_tasks) > 0, \
            "Should suggest remedial tasks after consecutive failures"
        
        # All remedial tasks should be valid and easier
        current_level = system.get_capability_level(dim)
        for task in remedial_tasks:
            assert isinstance(task, Task)
            assert task.complexity <= current_level, \
                f"Remedial task complexity ({task.complexity}) should be <= " \
                f"current capability ({current_level})"
    
    @given(config=curriculum_config_strategy())
    @settings(max_examples=100)
    def test_handle_task_failure_workflow(self, config: CurriculumConfig):
        """
        Feature: consciousness-system-deepening, Property 5: Consecutive Failure Handling
        
        The handle_task_failure method SHALL combine failure tracking, threshold
        reduction, and remedial task suggestion into a single workflow.
        
        **Validates: Requirements 1.6**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task = Task(
            task_id="test_task",
            task_type="reasoning",
            complexity=0.8,
            required_capabilities={"reasoning": 0.8}
        )
        
        # Accumulate failures just below threshold
        for _ in range(config.consecutive_failure_threshold - 1):
            system.update_capabilities(task.task_type, success=False, score=0.1)
        
        # This failure should trigger remediation
        result = system.handle_task_failure(task, score=0.1)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "capability_updates" in result
        assert "dimensions_needing_remediation" in result
        assert "reduced_thresholds" in result
        assert "remedial_tasks" in result
        
        # Should have triggered remediation for reasoning dimension
        assert "reasoning" in result["dimensions_needing_remediation"], \
            "Should identify reasoning as needing remediation"
        assert "reasoning" in result["reduced_thresholds"], \
            "Should have reduced threshold for reasoning"
        assert len(result["remedial_tasks"]) > 0, \
            "Should suggest remedial tasks"

    
    @given(
        num_failures=st.integers(min_value=1, max_value=10),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=100)
    def test_zpd_recommends_easier_tasks_after_threshold_reduction(
        self, num_failures: int, config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 5: Consecutive Failure Handling
        
        After difficulty threshold reduction, the next ZPD assessment SHALL
        recommend easier tasks (tasks that were previously outside ZPD may
        now be within ZPD).
        
        **Validates: Requirements 1.6**
        """
        # Ensure we exceed the failure threshold
        assume(num_failures >= config.consecutive_failure_threshold)
        
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        task_type = "reasoning"
        dim = "reasoning"
        
        # Get initial recommended difficulty
        initial_recommended = system.get_recommended_difficulty(dim)
        
        # Accumulate failures
        for _ in range(num_failures):
            system.update_capabilities(task_type, success=False, score=0.1)
        
        # Reduce threshold if failures exceeded threshold
        if system.check_consecutive_failures(dim):
            system.reduce_difficulty_threshold(dim)
        
        # Get new recommended difficulty
        new_recommended = system.get_recommended_difficulty(dim)
        
        # Recommended difficulty should be lower or equal
        # (capability decreased due to failures, so recommended difficulty decreases)
        assert new_recommended <= initial_recommended + 0.01, \
            f"Recommended difficulty should not increase after failures: " \
            f"{initial_recommended} -> {new_recommended}"


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestCurriculumLearningIntegration:
    """Integration tests for the curriculum learning system."""
    
    @given(
        task_sequence=st.lists(
            st.tuples(task_type_strategy, st.booleans(), unit_float_strategy),
            min_size=5,
            max_size=20
        ),
        config=curriculum_config_strategy()
    )
    @settings(max_examples=50)
    def test_capability_evolution_over_task_sequence(
        self, 
        task_sequence: List[tuple],
        config: CurriculumConfig
    ):
        """
        Feature: consciousness-system-deepening, Property 4 & 5: Capability Evolution
        
        For any sequence of tasks, the system SHALL correctly track capability
        evolution, consecutive failures, and difficulty thresholds.
        
        **Validates: Requirements 1.4, 1.5, 1.6**
        """
        system = CurriculumLearningSystem(config=config.to_dict())
        
        # Set initial capability levels
        for dim in CapabilityDimension:
            system._capability_levels[dim.value] = 0.5
        
        for task_type, success, score in task_sequence:
            # Record state before update
            weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
                task_type, {"reasoning": 0.5, "language": 0.5}
            )
            dim = list(weights.keys())[0]
            
            level_before = system.get_capability_level(dim)
            failures_before = system.get_consecutive_failures(dim)
            
            # Update capabilities
            system.update_capabilities(task_type, success, score)
            
            level_after = system.get_capability_level(dim)
            failures_after = system.get_consecutive_failures(dim)
            
            # Verify invariants
            assert 0.0 <= level_after <= 1.0, \
                f"Capability level must be in [0, 1]: {level_after}"
            
            if success:
                # Success should increase capability (or stay at max)
                assert level_after >= level_before or level_before >= 0.99
                # Success should reset failures
                assert failures_after == 0
            else:
                # Failure should decrease capability (or stay at min)
                assert level_after <= level_before or level_before <= 0.01
                # Failure should increment counter
                assert failures_after == failures_before + 1
