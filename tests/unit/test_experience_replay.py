"""
Unit tests for Experience Replay Buffer.

Tests specific examples and edge cases for:
- Experience dataclass creation and validation
- ExperienceReplayConfig creation and validation
- ExperienceReplayBuffer initialization
- Storage and retrieval operations (store, get, contains)
- Sampling strategies (uniform, prioritized, stratified)
- Priority updates
- Pruning behavior when buffer exceeds max_size
- Statistics and distribution methods
- Serialization/deserialization (to_dict/from_dict)
- Edge cases (empty buffer, single experience, etc.)

Requirements: 9.1, 9.3, 9.5
"""

import pytest
import time
import random
from typing import Dict, Any, List

from mm_orch.consciousness.experience_replay import (
    ExperienceReplayBuffer,
    ExperienceReplayConfig,
    Experience,
    create_experience,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_buffer() -> ExperienceReplayBuffer:
    """Create an ExperienceReplayBuffer with default configuration."""
    return ExperienceReplayBuffer()


@pytest.fixture
def small_buffer() -> ExperienceReplayBuffer:
    """Create a small ExperienceReplayBuffer for testing pruning."""
    return ExperienceReplayBuffer(max_size=10)


@pytest.fixture
def custom_config() -> ExperienceReplayConfig:
    """Create a custom ExperienceReplayConfig for testing."""
    return ExperienceReplayConfig(
        max_size=100,
        default_priority=0.6,
        priority_alpha=0.7,
        min_priority=0.05,
        prune_threshold=0.8,
        prune_ratio=0.3,
    )


@pytest.fixture
def custom_buffer(custom_config: ExperienceReplayConfig) -> ExperienceReplayBuffer:
    """Create an ExperienceReplayBuffer with custom configuration."""
    return ExperienceReplayBuffer(config=custom_config.to_dict())


@pytest.fixture
def sample_experience() -> Experience:
    """Create a sample Experience for testing."""
    return create_experience(
        task_type="search_qa",
        context={"query": "test query", "user_id": "user123"},
        action="search",
        outcome={"success": True, "results": 5},
        reward=0.8,
        priority=0.7,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_experiences() -> List[Experience]:
    """Create a list of sample experiences for testing."""
    experiences = []
    task_types = ["search_qa", "chat_generate", "lesson_pack", "rag_qa"]
    
    for i in range(20):
        exp = create_experience(
            task_type=task_types[i % len(task_types)],
            context={"index": i, "query": f"query_{i}"},
            action=f"action_{i}",
            outcome={"success": i % 2 == 0, "score": i * 0.1},
            reward=(i % 10) * 0.1,
            priority=(i % 10) * 0.1,
            metadata={"batch": i // 5},
        )
        experiences.append(exp)
    
    return experiences


# =============================================================================
# Tests for Experience Dataclass
# =============================================================================

class TestExperienceDataclass:
    """Unit tests for Experience dataclass creation and validation."""
    
    def test_valid_experience_creation(self, sample_experience: Experience):
        """Test creating a valid Experience object."""
        assert sample_experience.task_type == "search_qa"
        assert sample_experience.context == {"query": "test query", "user_id": "user123"}
        assert sample_experience.action == "search"
        assert sample_experience.outcome == {"success": True, "results": 5}
        assert sample_experience.reward == 0.8
        assert sample_experience.priority == 0.7
        assert sample_experience.metadata == {"source": "test"}
        assert sample_experience.experience_id is not None
        assert sample_experience.timestamp > 0
    
    def test_experience_with_default_values(self):
        """Test Experience creation with minimal required fields."""
        exp = Experience(
            experience_id="test_id",
            task_type="test_task",
            context={},
            action="test_action",
            outcome={},
            reward=0.5,
            priority=0.5,
        )
        
        assert exp.experience_id == "test_id"
        assert exp.task_type == "test_task"
        assert exp.metadata == {}
        assert exp.timestamp > 0

    def test_experience_priority_clamping(self):
        """Test that priority is clamped to [0.0, 1.0] range."""
        # Priority above 1.0 should be clamped
        exp_high = Experience(
            experience_id="high_priority",
            task_type="test",
            context={},
            action="test",
            outcome={},
            reward=0.5,
            priority=1.5,
        )
        assert exp_high.priority == 1.0
        
        # Priority below 0.0 should be clamped
        exp_low = Experience(
            experience_id="low_priority",
            task_type="test",
            context={},
            action="test",
            outcome={},
            reward=0.5,
            priority=-0.5,
        )
        assert exp_low.priority == 0.0
    
    def test_experience_invalid_empty_id_raises_error(self):
        """Test that empty experience_id raises ValueError."""
        with pytest.raises(ValueError, match="experience_id cannot be empty"):
            Experience(
                experience_id="",
                task_type="test",
                context={},
                action="test",
                outcome={},
                reward=0.5,
                priority=0.5,
            )
    
    def test_experience_invalid_empty_task_type_raises_error(self):
        """Test that empty task_type raises ValueError."""
        with pytest.raises(ValueError, match="task_type cannot be empty"):
            Experience(
                experience_id="test_id",
                task_type="",
                context={},
                action="test",
                outcome={},
                reward=0.5,
                priority=0.5,
            )
    
    def test_experience_invalid_context_type_raises_error(self):
        """Test that non-dict context raises ValueError."""
        with pytest.raises(ValueError, match="context must be a dictionary"):
            Experience(
                experience_id="test_id",
                task_type="test",
                context="not a dict",  # type: ignore
                action="test",
                outcome={},
                reward=0.5,
                priority=0.5,
            )
    
    def test_experience_invalid_action_type_raises_error(self):
        """Test that non-string action raises ValueError."""
        with pytest.raises(ValueError, match="action must be a string"):
            Experience(
                experience_id="test_id",
                task_type="test",
                context={},
                action=123,  # type: ignore
                outcome={},
                reward=0.5,
                priority=0.5,
            )
    
    def test_experience_invalid_outcome_type_raises_error(self):
        """Test that non-dict outcome raises ValueError."""
        with pytest.raises(ValueError, match="outcome must be a dictionary"):
            Experience(
                experience_id="test_id",
                task_type="test",
                context={},
                action="test",
                outcome="not a dict",  # type: ignore
                reward=0.5,
                priority=0.5,
            )

    def test_experience_to_dict(self, sample_experience: Experience):
        """Test Experience serialization to dictionary."""
        data = sample_experience.to_dict()
        
        assert data["experience_id"] == sample_experience.experience_id
        assert data["task_type"] == "search_qa"
        assert data["context"] == {"query": "test query", "user_id": "user123"}
        assert data["action"] == "search"
        assert data["outcome"] == {"success": True, "results": 5}
        assert data["reward"] == 0.8
        assert data["priority"] == 0.7
        assert data["metadata"] == {"source": "test"}
        assert "timestamp" in data
    
    def test_experience_from_dict(self):
        """Test Experience deserialization from dictionary."""
        data = {
            "experience_id": "restored_exp",
            "task_type": "chat_generate",
            "context": {"session": "abc"},
            "action": "generate",
            "outcome": {"text": "response"},
            "reward": 0.9,
            "priority": 0.8,
            "timestamp": 1000.0,
            "metadata": {"model": "gpt2"},
        }
        
        exp = Experience.from_dict(data)
        
        assert exp.experience_id == "restored_exp"
        assert exp.task_type == "chat_generate"
        assert exp.context == {"session": "abc"}
        assert exp.action == "generate"
        assert exp.outcome == {"text": "response"}
        assert exp.reward == 0.9
        assert exp.priority == 0.8
        assert exp.timestamp == 1000.0
        assert exp.metadata == {"model": "gpt2"}
    
    def test_experience_from_dict_with_defaults(self):
        """Test Experience deserialization with missing optional keys uses defaults."""
        data = {
            "experience_id": "minimal_exp",
            "task_type": "test_task",
            "action": "test_action",
        }
        
        exp = Experience.from_dict(data)
        
        assert exp.experience_id == "minimal_exp"
        assert exp.task_type == "test_task"
        assert exp.context == {}  # Default
        assert exp.action == "test_action"
        assert exp.outcome == {}  # Default
        assert exp.reward == 0.0  # Default
        assert exp.priority == 0.5  # Default
        assert exp.metadata == {}  # Default


# =============================================================================
# Tests for ExperienceReplayConfig Dataclass
# =============================================================================

class TestExperienceReplayConfigDataclass:
    """Unit tests for ExperienceReplayConfig dataclass creation and validation."""
    
    def test_default_config_creation(self):
        """Test creating ExperienceReplayConfig with default values."""
        config = ExperienceReplayConfig()
        
        assert config.max_size == 10000
        assert config.default_priority == 0.5
        assert config.priority_alpha == 0.6
        assert config.min_priority == 0.01
        assert config.prune_threshold == 0.9
        assert config.prune_ratio == 0.2

    def test_custom_config_creation(self, custom_config: ExperienceReplayConfig):
        """Test creating ExperienceReplayConfig with custom values."""
        assert custom_config.max_size == 100
        assert custom_config.default_priority == 0.6
        assert custom_config.priority_alpha == 0.7
        assert custom_config.min_priority == 0.05
        assert custom_config.prune_threshold == 0.8
        assert custom_config.prune_ratio == 0.3
    
    def test_config_boundary_values(self):
        """Test ExperienceReplayConfig with boundary values."""
        # Minimum valid values
        min_config = ExperienceReplayConfig(
            max_size=1,
            default_priority=0.0,
            priority_alpha=0.0,
            min_priority=0.001,
            prune_threshold=0.1,
            prune_ratio=0.01,
        )
        assert min_config.max_size == 1
        assert min_config.default_priority == 0.0
        
        # Maximum valid values
        max_config = ExperienceReplayConfig(
            max_size=1000000,
            default_priority=1.0,
            priority_alpha=1.0,
            min_priority=1.0,
            prune_threshold=1.0,
            prune_ratio=0.99,
        )
        assert max_config.default_priority == 1.0
    
    def test_config_invalid_max_size_raises_error(self):
        """Test that invalid max_size values raise ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ExperienceReplayConfig(max_size=0)
        
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ExperienceReplayConfig(max_size=-10)
    
    def test_config_invalid_default_priority_raises_error(self):
        """Test that invalid default_priority values raise ValueError."""
        with pytest.raises(ValueError, match="default_priority must be between"):
            ExperienceReplayConfig(default_priority=1.5)
        
        with pytest.raises(ValueError, match="default_priority must be between"):
            ExperienceReplayConfig(default_priority=-0.1)
    
    def test_config_invalid_priority_alpha_raises_error(self):
        """Test that invalid priority_alpha values raise ValueError."""
        with pytest.raises(ValueError, match="priority_alpha must be between"):
            ExperienceReplayConfig(priority_alpha=1.5)
        
        with pytest.raises(ValueError, match="priority_alpha must be between"):
            ExperienceReplayConfig(priority_alpha=-0.1)
    
    def test_config_invalid_min_priority_raises_error(self):
        """Test that invalid min_priority values raise ValueError."""
        with pytest.raises(ValueError, match="min_priority must be between"):
            ExperienceReplayConfig(min_priority=0.0)
        
        with pytest.raises(ValueError, match="min_priority must be between"):
            ExperienceReplayConfig(min_priority=-0.1)
    
    def test_config_invalid_prune_threshold_raises_error(self):
        """Test that invalid prune_threshold values raise ValueError."""
        with pytest.raises(ValueError, match="prune_threshold must be between"):
            ExperienceReplayConfig(prune_threshold=0.0)
        
        with pytest.raises(ValueError, match="prune_threshold must be between"):
            ExperienceReplayConfig(prune_threshold=1.5)
    
    def test_config_invalid_prune_ratio_raises_error(self):
        """Test that invalid prune_ratio values raise ValueError."""
        with pytest.raises(ValueError, match="prune_ratio must be between"):
            ExperienceReplayConfig(prune_ratio=0.0)
        
        with pytest.raises(ValueError, match="prune_ratio must be between"):
            ExperienceReplayConfig(prune_ratio=1.0)

    def test_config_to_dict(self, custom_config: ExperienceReplayConfig):
        """Test ExperienceReplayConfig serialization to dictionary."""
        data = custom_config.to_dict()
        
        assert data["max_size"] == 100
        assert data["default_priority"] == 0.6
        assert data["priority_alpha"] == 0.7
        assert data["min_priority"] == 0.05
        assert data["prune_threshold"] == 0.8
        assert data["prune_ratio"] == 0.3
    
    def test_config_from_dict(self):
        """Test ExperienceReplayConfig deserialization from dictionary."""
        data = {
            "max_size": 500,
            "default_priority": 0.4,
            "priority_alpha": 0.5,
            "min_priority": 0.02,
            "prune_threshold": 0.85,
            "prune_ratio": 0.25,
        }
        
        config = ExperienceReplayConfig.from_dict(data)
        
        assert config.max_size == 500
        assert config.default_priority == 0.4
        assert config.priority_alpha == 0.5
        assert config.min_priority == 0.02
        assert config.prune_threshold == 0.85
        assert config.prune_ratio == 0.25
    
    def test_config_from_dict_with_defaults(self):
        """Test ExperienceReplayConfig deserialization with missing keys uses defaults."""
        data = {"max_size": 200}
        
        config = ExperienceReplayConfig.from_dict(data)
        
        assert config.max_size == 200
        assert config.default_priority == 0.5  # Default
        assert config.priority_alpha == 0.6  # Default
        assert config.min_priority == 0.01  # Default


# =============================================================================
# Tests for ExperienceReplayBuffer Initialization
# =============================================================================

class TestExperienceReplayBufferInitialization:
    """Unit tests for ExperienceReplayBuffer initialization."""
    
    def test_default_initialization(self, default_buffer: ExperienceReplayBuffer):
        """Test initialization with default configuration."""
        assert default_buffer is not None
        assert len(default_buffer) == 0
        
        state = default_buffer.get_state()
        assert state["size"] == 0
        assert state["max_size"] == 10000
        assert state["total_stored"] == 0
        assert state["total_sampled"] == 0
        assert state["total_pruned"] == 0
    
    def test_initialization_with_max_size(self, small_buffer: ExperienceReplayBuffer):
        """Test initialization with custom max_size."""
        state = small_buffer.get_state()
        assert state["max_size"] == 10
    
    def test_initialization_with_custom_config(self, custom_buffer: ExperienceReplayBuffer):
        """Test initialization with custom configuration."""
        state = custom_buffer.get_state()
        config = state["config"]
        
        assert config["max_size"] == 100
        assert config["default_priority"] == 0.6
        assert config["priority_alpha"] == 0.7
    
    def test_initialization_with_config_dict(self):
        """Test initialization with config as dictionary."""
        config_dict = {
            "max_size": 50,
            "default_priority": 0.7,
        }
        
        buffer = ExperienceReplayBuffer(config=config_dict)
        state = buffer.get_state()
        
        assert state["config"]["max_size"] == 50
        assert state["config"]["default_priority"] == 0.7
    
    def test_initial_state_is_empty(self, default_buffer: ExperienceReplayBuffer):
        """Test that initial state has no experiences."""
        assert len(default_buffer) == 0
        assert default_buffer.get_task_type_distribution() == {}


# =============================================================================
# Tests for Storage and Retrieval Operations
# =============================================================================

class TestStorageAndRetrieval:
    """Unit tests for store, get, and contains operations."""
    
    def test_store_single_experience(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test storing a single experience."""
        exp_id = default_buffer.store(sample_experience)
        
        assert exp_id == sample_experience.experience_id
        assert len(default_buffer) == 1
        assert exp_id in default_buffer
    
    def test_store_multiple_experiences(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test storing multiple experiences."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        assert len(default_buffer) == len(sample_experiences)

    def test_get_experience_by_id(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test retrieving an experience by ID."""
        default_buffer.store(sample_experience)
        
        retrieved = default_buffer.get_experience(sample_experience.experience_id)
        
        assert retrieved is not None
        assert retrieved.experience_id == sample_experience.experience_id
        assert retrieved.task_type == sample_experience.task_type
        assert retrieved.action == sample_experience.action
    
    def test_get_experience_not_found_returns_none(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test that getting non-existent experience returns None."""
        result = default_buffer.get_experience("non_existent_id")
        assert result is None
    
    def test_contains_operator(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test the __contains__ operator."""
        assert sample_experience.experience_id not in default_buffer
        
        default_buffer.store(sample_experience)
        
        assert sample_experience.experience_id in default_buffer
    
    def test_get_experiences_by_task_type(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test retrieving experiences by task type."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        search_qa_exps = default_buffer.get_experiences_by_task_type("search_qa")
        
        assert len(search_qa_exps) > 0
        for exp in search_qa_exps:
            assert exp.task_type == "search_qa"
    
    def test_get_experiences_by_task_type_not_found(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test retrieving experiences for non-existent task type returns empty list."""
        default_buffer.store(sample_experience)
        
        result = default_buffer.get_experiences_by_task_type("non_existent_type")
        
        assert result == []
    
    def test_clear_buffer(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test clearing all experiences from buffer."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        assert len(default_buffer) > 0
        
        default_buffer.clear()
        
        assert len(default_buffer) == 0
        assert default_buffer.get_task_type_distribution() == {}


# =============================================================================
# Tests for Sampling Strategies
# =============================================================================

class TestSamplingStrategies:
    """Unit tests for uniform, prioritized, and stratified sampling."""
    
    def test_sample_uniform_returns_correct_batch_size(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that uniform sampling returns the requested batch size."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        batch_size = 5
        sampled = default_buffer.sample(batch_size, strategy="uniform")
        
        assert len(sampled) == batch_size
    
    def test_sample_uniform_returns_valid_experiences(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that uniform sampling returns valid experiences from buffer."""
        stored_ids = set()
        for exp in sample_experiences:
            default_buffer.store(exp)
            stored_ids.add(exp.experience_id)
        
        sampled = default_buffer.sample(5, strategy="uniform")
        
        for exp in sampled:
            assert exp.experience_id in stored_ids
    
    def test_sample_uniform_no_duplicates(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that uniform sampling returns unique experiences."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        sampled = default_buffer.sample(10, strategy="uniform")
        sampled_ids = [exp.experience_id for exp in sampled]
        
        assert len(sampled_ids) == len(set(sampled_ids))

    def test_sample_prioritized_returns_correct_batch_size(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that prioritized sampling returns the requested batch size."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        batch_size = 5
        sampled = default_buffer.sample(batch_size, strategy="prioritized")
        
        assert len(sampled) == batch_size
    
    def test_sample_prioritized_favors_high_priority(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test that prioritized sampling favors high priority experiences."""
        # Create experiences with distinct priorities
        high_priority_exp = create_experience(
            task_type="high",
            context={},
            action="high_action",
            outcome={},
            reward=1.0,
            priority=1.0,
        )
        
        low_priority_exps = []
        for i in range(10):
            exp = create_experience(
                task_type="low",
                context={},
                action=f"low_action_{i}",
                outcome={},
                reward=0.1,
                priority=0.1,
            )
            low_priority_exps.append(exp)
        
        default_buffer.store(high_priority_exp)
        for exp in low_priority_exps:
            default_buffer.store(exp)
        
        # Sample many times and count high priority occurrences
        high_priority_count = 0
        num_trials = 100
        
        for _ in range(num_trials):
            sampled = default_buffer.sample(3, strategy="prioritized")
            for exp in sampled:
                if exp.experience_id == high_priority_exp.experience_id:
                    high_priority_count += 1
        
        # High priority should appear frequently
        assert high_priority_count > num_trials * 0.3, \
            f"High priority experience should be sampled frequently: {high_priority_count}/{num_trials * 3}"
    
    def test_sample_stratified_returns_correct_batch_size(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that stratified sampling returns the requested batch size."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        batch_size = 8
        sampled = default_buffer.sample(batch_size, strategy="stratified")
        
        assert len(sampled) == batch_size
    
    def test_sample_stratified_includes_all_task_types(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that stratified sampling includes all task types when possible."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        task_types = set(exp.task_type for exp in sample_experiences)
        
        # Sample with batch size >= number of task types
        batch_size = len(task_types)
        sampled = default_buffer.sample(batch_size, strategy="stratified")
        
        sampled_task_types = set(exp.task_type for exp in sampled)
        
        assert sampled_task_types == task_types
    
    def test_sample_invalid_strategy_raises_error(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that invalid sampling strategy raises ValueError."""
        default_buffer.store(sample_experience)
        
        with pytest.raises(ValueError, match="Invalid strategy"):
            default_buffer.sample(1, strategy="invalid_strategy")
    
    def test_sample_negative_batch_size_raises_error(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that negative batch size raises ValueError."""
        default_buffer.store(sample_experience)
        
        with pytest.raises(ValueError, match="batch_size must be non-negative"):
            default_buffer.sample(-1, strategy="uniform")
    
    def test_sample_zero_batch_size_returns_empty(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that zero batch size returns empty list."""
        default_buffer.store(sample_experience)
        
        sampled = default_buffer.sample(0, strategy="uniform")
        
        assert sampled == []
    
    def test_sample_from_empty_buffer_returns_empty(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test that sampling from empty buffer returns empty list."""
        sampled = default_buffer.sample(5, strategy="uniform")
        assert sampled == []
        
        sampled = default_buffer.sample(5, strategy="prioritized")
        assert sampled == []
        
        sampled = default_buffer.sample(5, strategy="stratified")
        assert sampled == []

    def test_sample_batch_size_larger_than_buffer(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that sampling more than buffer size returns all experiences."""
        default_buffer.store(sample_experience)
        
        sampled = default_buffer.sample(100, strategy="uniform")
        
        assert len(sampled) == 1
    
    def test_sample_updates_total_sampled_count(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that sampling updates the total_sampled statistic."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        initial_state = default_buffer.get_state()
        initial_sampled = initial_state["total_sampled"]
        
        default_buffer.sample(5, strategy="uniform")
        default_buffer.sample(3, strategy="prioritized")
        
        final_state = default_buffer.get_state()
        
        assert final_state["total_sampled"] == initial_sampled + 8


# =============================================================================
# Tests for Priority Updates
# =============================================================================

class TestPriorityUpdates:
    """Unit tests for priority update operations."""
    
    def test_update_priority_success(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test successfully updating an experience's priority."""
        default_buffer.store(sample_experience)
        
        new_priority = 0.9
        default_buffer.update_priority(sample_experience.experience_id, new_priority)
        
        retrieved = default_buffer.get_experience(sample_experience.experience_id)
        assert retrieved.priority == new_priority
    
    def test_update_priority_clamping_high(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that priority above 1.0 is clamped."""
        default_buffer.store(sample_experience)
        
        default_buffer.update_priority(sample_experience.experience_id, 1.5)
        
        retrieved = default_buffer.get_experience(sample_experience.experience_id)
        assert retrieved.priority == 1.0
    
    def test_update_priority_clamping_low(
        self, default_buffer: ExperienceReplayBuffer, sample_experience: Experience
    ):
        """Test that priority below 0.0 is clamped."""
        default_buffer.store(sample_experience)
        
        default_buffer.update_priority(sample_experience.experience_id, -0.5)
        
        retrieved = default_buffer.get_experience(sample_experience.experience_id)
        assert retrieved.priority == 0.0
    
    def test_update_priority_not_found_raises_error(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test that updating non-existent experience raises KeyError."""
        with pytest.raises(KeyError, match="not found in buffer"):
            default_buffer.update_priority("non_existent_id", 0.5)


# =============================================================================
# Tests for Pruning Behavior
# =============================================================================

class TestPruningBehavior:
    """Unit tests for pruning when buffer exceeds max_size."""
    
    def test_pruning_triggered_when_buffer_full(self):
        """Test that pruning is triggered when buffer reaches max_size."""
        buffer = ExperienceReplayBuffer(max_size=10)
        
        # Add more experiences than max_size
        for i in range(15):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=i * 0.05,  # Varied priorities
            )
            buffer.store(exp)
        
        # Buffer should not exceed max_size
        assert len(buffer) <= 10
    
    def test_pruning_removes_low_priority_experiences(self):
        """Test that pruning removes low priority experiences first."""
        config = ExperienceReplayConfig(
            max_size=10,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Add high priority experiences
        high_priority_ids = []
        for i in range(5):
            exp = create_experience(
                task_type="high",
                context={},
                action=f"high_{i}",
                outcome={},
                reward=1.0,
                priority=0.9,
            )
            buffer.store(exp)
            high_priority_ids.append(exp.experience_id)
        
        # Add low priority experiences to trigger pruning
        for i in range(10):
            exp = create_experience(
                task_type="low",
                context={},
                action=f"low_{i}",
                outcome={},
                reward=0.1,
                priority=0.1,
            )
            buffer.store(exp)
        
        # High priority experiences should be preserved
        preserved_high = sum(1 for exp_id in high_priority_ids if exp_id in buffer)
        assert preserved_high >= 4, f"Most high priority experiences should be preserved: {preserved_high}/5"

    def test_prune_returns_removed_count(self):
        """Test that prune() returns the number of removed experiences."""
        config = ExperienceReplayConfig(
            max_size=20,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Fill buffer
        for i in range(20):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
        
        initial_size = len(buffer)
        removed = buffer.prune()
        
        assert removed > 0
        assert len(buffer) == initial_size - removed
    
    def test_prune_empty_buffer_returns_zero(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test that pruning empty buffer returns 0."""
        removed = default_buffer.prune()
        assert removed == 0
    
    def test_prune_updates_total_pruned_count(self):
        """Test that pruning updates the total_pruned statistic."""
        config = ExperienceReplayConfig(
            max_size=10,
            prune_threshold=0.9,
            prune_ratio=0.3,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        # Fill buffer
        for i in range(10):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
        
        initial_state = buffer.get_state()
        initial_pruned = initial_state["total_pruned"]
        
        removed = buffer.prune()
        
        final_state = buffer.get_state()
        assert final_state["total_pruned"] == initial_pruned + removed
    
    def test_prune_maintains_task_type_index_consistency(self):
        """Test that pruning maintains task type index consistency."""
        config = ExperienceReplayConfig(
            max_size=15,
            prune_threshold=0.9,
            prune_ratio=0.4,
        )
        buffer = ExperienceReplayBuffer(config=config.to_dict())
        
        task_types = ["type_a", "type_b", "type_c"]
        
        # Add experiences of different types
        for i in range(20):
            exp = create_experience(
                task_type=task_types[i % len(task_types)],
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
        
        # Verify distribution matches actual experiences
        distribution = buffer.get_task_type_distribution()
        
        for task_type, count in distribution.items():
            actual_exps = buffer.get_experiences_by_task_type(task_type)
            assert len(actual_exps) == count, \
                f"Distribution mismatch for {task_type}: index={count}, actual={len(actual_exps)}"


# =============================================================================
# Tests for Statistics and Distribution Methods
# =============================================================================

class TestStatisticsAndDistribution:
    """Unit tests for statistics and distribution methods."""
    
    def test_get_task_type_distribution(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test getting task type distribution."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        distribution = default_buffer.get_task_type_distribution()
        
        # Verify distribution is correct
        expected_counts = {}
        for exp in sample_experiences:
            expected_counts[exp.task_type] = expected_counts.get(exp.task_type, 0) + 1
        
        assert distribution == expected_counts
    
    def test_get_task_type_distribution_empty_buffer(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test getting task type distribution from empty buffer."""
        distribution = default_buffer.get_task_type_distribution()
        assert distribution == {}
    
    def test_get_statistics(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test getting detailed statistics."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        stats = default_buffer.get_statistics()
        
        assert stats["count"] == len(sample_experiences)
        assert 0.0 <= stats["avg_priority"] <= 1.0
        assert 0.0 <= stats["min_priority"] <= 1.0
        assert 0.0 <= stats["max_priority"] <= 1.0
        assert stats["min_priority"] <= stats["avg_priority"] <= stats["max_priority"]
        assert stats["task_type_count"] > 0

    def test_get_statistics_empty_buffer(
        self, default_buffer: ExperienceReplayBuffer
    ):
        """Test getting statistics from empty buffer."""
        stats = default_buffer.get_statistics()
        
        assert stats["count"] == 0
        assert stats["avg_priority"] == 0.0
        assert stats["min_priority"] == 0.0
        assert stats["max_priority"] == 0.0
        assert stats["avg_reward"] == 0.0
        assert stats["task_type_count"] == 0
    
    def test_get_state(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test getting buffer state."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        default_buffer.sample(5, strategy="uniform")
        
        state = default_buffer.get_state()
        
        assert state["size"] == len(sample_experiences)
        assert state["max_size"] == 10000
        assert state["total_stored"] == len(sample_experiences)
        assert state["total_sampled"] == 5
        assert state["total_pruned"] == 0
        assert state["uptime"] >= 0
        assert "config" in state
        assert "task_types" in state
        assert "task_type_distribution" in state
    
    def test_len_operator(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test the __len__ operator."""
        assert len(default_buffer) == 0
        
        for i, exp in enumerate(sample_experiences):
            default_buffer.store(exp)
            assert len(default_buffer) == i + 1


# =============================================================================
# Tests for Serialization (to_dict/from_dict)
# =============================================================================

class TestSerialization:
    """Unit tests for serialization and deserialization."""
    
    def test_to_dict_empty_buffer(self, default_buffer: ExperienceReplayBuffer):
        """Test serializing empty buffer to dictionary."""
        data = default_buffer.to_dict()
        
        assert "config" in data
        assert "experiences" in data
        assert data["experiences"] == {}
        assert "total_stored" in data
        assert "total_sampled" in data
        assert "total_pruned" in data
    
    def test_to_dict_with_experiences(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test serializing buffer with experiences to dictionary."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        default_buffer.sample(3, strategy="uniform")
        
        data = default_buffer.to_dict()
        
        assert len(data["experiences"]) == len(sample_experiences)
        assert data["total_stored"] == len(sample_experiences)
        assert data["total_sampled"] == 3
        
        # Verify each experience is serialized correctly
        for exp_id, exp_data in data["experiences"].items():
            assert "task_type" in exp_data
            assert "action" in exp_data
            assert "priority" in exp_data
    
    def test_from_dict_empty_buffer(self):
        """Test deserializing empty buffer from dictionary."""
        data = {
            "config": {"max_size": 500},
            "experiences": {},
            "total_stored": 0,
            "total_sampled": 0,
            "total_pruned": 0,
        }
        
        buffer = ExperienceReplayBuffer.from_dict(data)
        
        assert len(buffer) == 0
        state = buffer.get_state()
        assert state["config"]["max_size"] == 500
    
    def test_from_dict_with_experiences(self):
        """Test deserializing buffer with experiences from dictionary."""
        data = {
            "config": {"max_size": 100},
            "experiences": {
                "exp_1": {
                    "experience_id": "exp_1",
                    "task_type": "search_qa",
                    "context": {"query": "test"},
                    "action": "search",
                    "outcome": {"success": True},
                    "reward": 0.8,
                    "priority": 0.7,
                    "timestamp": 1000.0,
                    "metadata": {},
                },
                "exp_2": {
                    "experience_id": "exp_2",
                    "task_type": "chat_generate",
                    "context": {},
                    "action": "generate",
                    "outcome": {},
                    "reward": 0.5,
                    "priority": 0.5,
                    "timestamp": 2000.0,
                    "metadata": {},
                },
            },
            "total_stored": 10,
            "total_sampled": 5,
            "total_pruned": 3,
        }
        
        buffer = ExperienceReplayBuffer.from_dict(data)
        
        assert len(buffer) == 2
        assert "exp_1" in buffer
        assert "exp_2" in buffer
        
        state = buffer.get_state()
        assert state["total_stored"] == 10
        assert state["total_sampled"] == 5
        assert state["total_pruned"] == 3

    def test_serialization_round_trip(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that serialization and deserialization preserves all data."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        default_buffer.sample(5, strategy="uniform")
        
        # Serialize
        data = default_buffer.to_dict()
        
        # Deserialize
        restored_buffer = ExperienceReplayBuffer.from_dict(data)
        
        # Verify state matches
        original_state = default_buffer.get_state()
        restored_state = restored_buffer.get_state()
        
        assert restored_state["size"] == original_state["size"]
        assert restored_state["total_stored"] == original_state["total_stored"]
        assert restored_state["total_sampled"] == original_state["total_sampled"]
        assert restored_state["total_pruned"] == original_state["total_pruned"]
        
        # Verify all experiences are preserved
        for exp in sample_experiences:
            assert exp.experience_id in restored_buffer
            restored_exp = restored_buffer.get_experience(exp.experience_id)
            assert restored_exp.task_type == exp.task_type
            assert restored_exp.action == exp.action
            assert restored_exp.priority == exp.priority
    
    def test_serialization_preserves_task_type_index(
        self, default_buffer: ExperienceReplayBuffer, sample_experiences: List[Experience]
    ):
        """Test that serialization preserves task type index."""
        for exp in sample_experiences:
            default_buffer.store(exp)
        
        original_distribution = default_buffer.get_task_type_distribution()
        
        # Serialize and deserialize
        data = default_buffer.to_dict()
        restored_buffer = ExperienceReplayBuffer.from_dict(data)
        
        restored_distribution = restored_buffer.get_task_type_distribution()
        
        assert restored_distribution == original_distribution


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Unit tests for edge cases."""
    
    def test_single_experience_buffer(self):
        """Test buffer with single experience."""
        buffer = ExperienceReplayBuffer(max_size=1)
        
        exp = create_experience(
            task_type="test",
            context={},
            action="test_action",
            outcome={},
            reward=0.5,
            priority=0.5,
        )
        
        buffer.store(exp)
        
        assert len(buffer) == 1
        
        # Sampling should work
        sampled = buffer.sample(1, strategy="uniform")
        assert len(sampled) == 1
        assert sampled[0].experience_id == exp.experience_id
        
        # Stratified sampling should work
        sampled = buffer.sample(1, strategy="stratified")
        assert len(sampled) == 1
        
        # Prioritized sampling should work
        sampled = buffer.sample(1, strategy="prioritized")
        assert len(sampled) == 1
    
    def test_buffer_with_all_same_priority(self):
        """Test buffer where all experiences have the same priority."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        for i in range(20):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=0.5,  # All same priority
            )
            buffer.store(exp)
        
        # Prioritized sampling should still work
        sampled = buffer.sample(5, strategy="prioritized")
        assert len(sampled) == 5
    
    def test_buffer_with_all_zero_priority(self):
        """Test buffer where all experiences have zero priority."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        for i in range(10):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.0,
                priority=0.0,
            )
            buffer.store(exp)
        
        # Prioritized sampling should fall back to uniform
        sampled = buffer.sample(5, strategy="prioritized")
        assert len(sampled) == 5

    def test_buffer_with_single_task_type(self):
        """Test buffer where all experiences have the same task type."""
        buffer = ExperienceReplayBuffer(max_size=100)
        
        for i in range(15):
            exp = create_experience(
                task_type="single_type",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
        
        # Stratified sampling should work with single task type
        sampled = buffer.sample(5, strategy="stratified")
        assert len(sampled) == 5
        
        for exp in sampled:
            assert exp.task_type == "single_type"
    
    def test_very_small_max_size(self):
        """Test buffer with very small max_size."""
        buffer = ExperienceReplayBuffer(max_size=2)
        
        # Add more experiences than max_size
        for i in range(5):
            exp = create_experience(
                task_type="test",
                context={},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=i * 0.2,
            )
            buffer.store(exp)
        
        # Buffer should not exceed max_size
        assert len(buffer) <= 2
    
    def test_experience_with_complex_context(self):
        """Test experience with complex nested context."""
        complex_context = {
            "query": "test query",
            "nested": {
                "level1": {
                    "level2": ["a", "b", "c"]
                }
            },
            "list_field": [1, 2, 3, {"inner": "value"}],
            "number": 42,
            "boolean": True,
        }
        
        exp = create_experience(
            task_type="complex",
            context=complex_context,
            action="test",
            outcome={"result": "success"},
            reward=0.5,
            priority=0.5,
        )
        
        buffer = ExperienceReplayBuffer()
        buffer.store(exp)
        
        retrieved = buffer.get_experience(exp.experience_id)
        assert retrieved.context == complex_context
    
    def test_experience_with_empty_fields(self):
        """Test experience with empty context and outcome."""
        exp = create_experience(
            task_type="minimal",
            context={},
            action="minimal_action",
            outcome={},
            reward=0.0,
            priority=0.0,
        )
        
        buffer = ExperienceReplayBuffer()
        buffer.store(exp)
        
        retrieved = buffer.get_experience(exp.experience_id)
        assert retrieved.context == {}
        assert retrieved.outcome == {}
    
    def test_concurrent_store_and_sample(self):
        """Test storing and sampling in interleaved manner."""
        buffer = ExperienceReplayBuffer(max_size=50)
        
        for i in range(30):
            # Store an experience
            exp = create_experience(
                task_type=f"type_{i % 3}",
                context={"index": i},
                action=f"action_{i}",
                outcome={},
                reward=0.5,
                priority=random.random(),
            )
            buffer.store(exp)
            
            # Sample after every 5 stores
            if (i + 1) % 5 == 0:
                sampled = buffer.sample(min(3, len(buffer)), strategy="uniform")
                assert len(sampled) <= 3
        
        assert len(buffer) > 0


# =============================================================================
# Tests for create_experience Factory Function
# =============================================================================

class TestCreateExperienceFactory:
    """Unit tests for create_experience factory function."""
    
    def test_create_experience_generates_unique_ids(self):
        """Test that create_experience generates unique IDs."""
        experiences = []
        for i in range(100):
            exp = create_experience(
                task_type="test",
                context={},
                action="test",
                outcome={},
                reward=0.5,
            )
            experiences.append(exp)
        
        ids = [exp.experience_id for exp in experiences]
        assert len(ids) == len(set(ids)), "All experience IDs should be unique"
    
    def test_create_experience_with_all_parameters(self):
        """Test create_experience with all parameters specified."""
        exp = create_experience(
            task_type="full_test",
            context={"key": "value"},
            action="full_action",
            outcome={"result": "success"},
            reward=0.9,
            priority=0.8,
            metadata={"source": "test", "version": 1},
        )
        
        assert exp.task_type == "full_test"
        assert exp.context == {"key": "value"}
        assert exp.action == "full_action"
        assert exp.outcome == {"result": "success"}
        assert exp.reward == 0.9
        assert exp.priority == 0.8
        assert exp.metadata == {"source": "test", "version": 1}
        assert exp.experience_id is not None
        assert exp.timestamp > 0
    
    def test_create_experience_with_default_priority(self):
        """Test create_experience uses default priority when not specified."""
        exp = create_experience(
            task_type="test",
            context={},
            action="test",
            outcome={},
            reward=0.5,
        )
        
        assert exp.priority == 0.5  # Default priority
    
    def test_create_experience_with_default_metadata(self):
        """Test create_experience uses empty metadata when not specified."""
        exp = create_experience(
            task_type="test",
            context={},
            action="test",
            outcome={},
            reward=0.5,
        )
        
        assert exp.metadata == {}
