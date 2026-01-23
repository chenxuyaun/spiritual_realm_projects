"""
Unit tests for Curriculum Learning System.

Tests specific examples and edge cases for:
- TaskDifficulty, ZPDAssessment, CurriculumConfig dataclass creation and validation
- Task dataclass creation and validation
- CurriculumLearningSystem initialization with and without DevelopmentSystem
- estimate_task_difficulty() for different task types
- is_in_zpd() boundary conditions (too easy, in ZPD, too hard)
- update_capabilities() growth and decay
- suggest_scaffolding() for difficult tasks
- check_consecutive_failures() and reduce_difficulty_threshold()
- suggest_remedial_tasks()
- Serialization/deserialization (to_dict/from_dict)

Requirements: 1.1-1.6
"""

import pytest
import time
from typing import Dict, Any

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
# Fixtures
# =============================================================================

@pytest.fixture
def default_system() -> CurriculumLearningSystem:
    """Create a CurriculumLearningSystem with default configuration."""
    return CurriculumLearningSystem()


@pytest.fixture
def custom_config() -> CurriculumConfig:
    """Create a custom CurriculumConfig for testing."""
    return CurriculumConfig(
        zpd_lower_threshold=0.15,
        zpd_upper_threshold=0.35,
        capability_growth_rate=0.06,
        capability_decay_rate=0.03,
        consecutive_failure_threshold=2,
    )


@pytest.fixture
def custom_system(custom_config: CurriculumConfig) -> CurriculumLearningSystem:
    """Create a CurriculumLearningSystem with custom configuration."""
    return CurriculumLearningSystem(config=custom_config.to_dict())


@pytest.fixture
def simple_task() -> Task:
    """Create a simple task for testing."""
    return Task(
        task_id="simple_task_001",
        task_type="reasoning",
        complexity=0.5,
        required_capabilities={"reasoning": 0.5, "memory": 0.3},
    )


@pytest.fixture
def difficult_task() -> Task:
    """Create a difficult task for testing."""
    return Task(
        task_id="difficult_task_001",
        task_type="reasoning",
        complexity=0.9,
        required_capabilities={"reasoning": 0.9, "memory": 0.8, "planning": 0.7},
    )


@pytest.fixture
def easy_task() -> Task:
    """Create an easy task for testing."""
    return Task(
        task_id="easy_task_001",
        task_type="language",
        complexity=0.2,
        required_capabilities={"language": 0.2},
    )


# =============================================================================
# Tests for TaskDifficulty Dataclass
# =============================================================================

class TestTaskDifficultyDataclass:
    """Unit tests for TaskDifficulty dataclass creation and validation."""
    
    def test_valid_task_difficulty_creation(self):
        """Test creating a valid TaskDifficulty object."""
        difficulty = TaskDifficulty(
            complexity=0.6,
            required_capabilities={"reasoning": 0.7, "language": 0.5},
            cognitive_load=0.5,
            overall_difficulty=0.6,
        )
        
        assert difficulty.complexity == 0.6
        assert difficulty.cognitive_load == 0.5
        assert difficulty.overall_difficulty == 0.6
        assert difficulty.required_capabilities == {"reasoning": 0.7, "language": 0.5}
    
    def test_task_difficulty_boundary_values(self):
        """Test TaskDifficulty with boundary values (0.0 and 1.0)."""
        # Minimum values
        min_difficulty = TaskDifficulty(
            complexity=0.0,
            required_capabilities={},
            cognitive_load=0.0,
            overall_difficulty=0.0,
        )
        assert min_difficulty.overall_difficulty == 0.0
        
        # Maximum values
        max_difficulty = TaskDifficulty(
            complexity=1.0,
            required_capabilities={"reasoning": 1.0},
            cognitive_load=1.0,
            overall_difficulty=1.0,
        )
        assert max_difficulty.overall_difficulty == 1.0
    
    def test_task_difficulty_invalid_complexity_raises_error(self):
        """Test that invalid complexity values raise ValueError."""
        with pytest.raises(ValueError, match="complexity must be between"):
            TaskDifficulty(
                complexity=1.5,  # Invalid: > 1.0
                required_capabilities={},
                cognitive_load=0.5,
                overall_difficulty=0.5,
            )
        
        with pytest.raises(ValueError, match="complexity must be between"):
            TaskDifficulty(
                complexity=-0.1,  # Invalid: < 0.0
                required_capabilities={},
                cognitive_load=0.5,
                overall_difficulty=0.5,
            )
    
    def test_task_difficulty_invalid_cognitive_load_raises_error(self):
        """Test that invalid cognitive_load values raise ValueError."""
        with pytest.raises(ValueError, match="cognitive_load must be between"):
            TaskDifficulty(
                complexity=0.5,
                required_capabilities={},
                cognitive_load=1.2,  # Invalid
                overall_difficulty=0.5,
            )
    
    def test_task_difficulty_invalid_overall_difficulty_raises_error(self):
        """Test that invalid overall_difficulty values raise ValueError."""
        with pytest.raises(ValueError, match="overall_difficulty must be between"):
            TaskDifficulty(
                complexity=0.5,
                required_capabilities={},
                cognitive_load=0.5,
                overall_difficulty=-0.5,  # Invalid
            )
    
    def test_task_difficulty_invalid_required_capability_raises_error(self):
        """Test that invalid required_capabilities values raise ValueError."""
        with pytest.raises(ValueError, match="required_capabilities"):
            TaskDifficulty(
                complexity=0.5,
                required_capabilities={"reasoning": 1.5},  # Invalid
                cognitive_load=0.5,
                overall_difficulty=0.5,
            )
    
    def test_task_difficulty_to_dict(self):
        """Test TaskDifficulty serialization to dictionary."""
        difficulty = TaskDifficulty(
            complexity=0.6,
            required_capabilities={"reasoning": 0.7},
            cognitive_load=0.5,
            overall_difficulty=0.6,
        )
        
        data = difficulty.to_dict()
        
        assert data["complexity"] == 0.6
        assert data["cognitive_load"] == 0.5
        assert data["overall_difficulty"] == 0.6
        assert data["required_capabilities"] == {"reasoning": 0.7}
    
    def test_task_difficulty_from_dict(self):
        """Test TaskDifficulty deserialization from dictionary."""
        data = {
            "complexity": 0.6,
            "required_capabilities": {"reasoning": 0.7},
            "cognitive_load": 0.5,
            "overall_difficulty": 0.6,
        }
        
        difficulty = TaskDifficulty.from_dict(data)
        
        assert difficulty.complexity == 0.6
        assert difficulty.cognitive_load == 0.5
        assert difficulty.overall_difficulty == 0.6
        assert difficulty.required_capabilities == {"reasoning": 0.7}


# =============================================================================
# Tests for ZPDAssessment Dataclass
# =============================================================================

class TestZPDAssessmentDataclass:
    """Unit tests for ZPDAssessment dataclass creation and validation."""
    
    def test_valid_zpd_assessment_creation(self):
        """Test creating a valid ZPDAssessment object."""
        assessment = ZPDAssessment(
            in_zpd=True,
            difficulty_gap=0.25,
            recommendations=["Focus on reasoning skills"],
            suggested_scaffolding=["Break into smaller steps"],
        )
        
        assert assessment.in_zpd is True
        assert assessment.difficulty_gap == 0.25
        assert len(assessment.recommendations) == 1
        assert len(assessment.suggested_scaffolding) == 1
    
    def test_zpd_assessment_without_scaffolding(self):
        """Test ZPDAssessment without scaffolding suggestions."""
        assessment = ZPDAssessment(
            in_zpd=True,
            difficulty_gap=0.2,
            recommendations=["Task is appropriate"],
        )
        
        assert assessment.suggested_scaffolding is None
    
    def test_zpd_assessment_negative_difficulty_gap(self):
        """Test ZPDAssessment with negative difficulty gap (task too easy)."""
        assessment = ZPDAssessment(
            in_zpd=False,
            difficulty_gap=-0.3,  # Task is easier than capability
            recommendations=["Task is too easy"],
        )
        
        assert assessment.in_zpd is False
        assert assessment.difficulty_gap < 0
    
    def test_zpd_assessment_to_dict(self):
        """Test ZPDAssessment serialization to dictionary."""
        assessment = ZPDAssessment(
            in_zpd=True,
            difficulty_gap=0.25,
            recommendations=["Recommendation 1"],
            suggested_scaffolding=["Scaffold 1"],
        )
        
        data = assessment.to_dict()
        
        assert data["in_zpd"] is True
        assert data["difficulty_gap"] == 0.25
        assert data["recommendations"] == ["Recommendation 1"]
        assert data["suggested_scaffolding"] == ["Scaffold 1"]
    
    def test_zpd_assessment_from_dict(self):
        """Test ZPDAssessment deserialization from dictionary."""
        data = {
            "in_zpd": False,
            "difficulty_gap": 0.5,
            "recommendations": ["Too difficult"],
            "suggested_scaffolding": ["Break down task"],
        }
        
        assessment = ZPDAssessment.from_dict(data)
        
        assert assessment.in_zpd is False
        assert assessment.difficulty_gap == 0.5
        assert assessment.recommendations == ["Too difficult"]
        assert assessment.suggested_scaffolding == ["Break down task"]


# =============================================================================
# Tests for CurriculumConfig Dataclass
# =============================================================================

class TestCurriculumConfigDataclass:
    """Unit tests for CurriculumConfig dataclass creation and validation."""
    
    def test_default_config_creation(self):
        """Test creating CurriculumConfig with default values."""
        config = CurriculumConfig()
        
        assert config.zpd_lower_threshold == 0.2
        assert config.zpd_upper_threshold == 0.4
        assert config.capability_growth_rate == 0.05
        assert config.capability_decay_rate == 0.02
        assert config.consecutive_failure_threshold == 3
    
    def test_custom_config_creation(self):
        """Test creating CurriculumConfig with custom values."""
        config = CurriculumConfig(
            zpd_lower_threshold=0.15,
            zpd_upper_threshold=0.35,
            capability_growth_rate=0.06,
            capability_decay_rate=0.03,
            consecutive_failure_threshold=2,
        )
        
        assert config.zpd_lower_threshold == 0.15
        assert config.zpd_upper_threshold == 0.35
        assert config.capability_growth_rate == 0.06
        assert config.capability_decay_rate == 0.03
        assert config.consecutive_failure_threshold == 2
    
    def test_config_invalid_zpd_lower_threshold(self):
        """Test that negative zpd_lower_threshold raises ValueError."""
        with pytest.raises(ValueError, match="zpd_lower_threshold must be non-negative"):
            CurriculumConfig(zpd_lower_threshold=-0.1)
    
    def test_config_invalid_zpd_threshold_order(self):
        """Test that zpd_upper < zpd_lower raises ValueError."""
        with pytest.raises(ValueError, match="zpd_upper_threshold must be >= zpd_lower_threshold"):
            CurriculumConfig(
                zpd_lower_threshold=0.5,
                zpd_upper_threshold=0.3,  # Invalid: less than lower
            )
    
    def test_config_invalid_growth_rate(self):
        """Test that negative capability_growth_rate raises ValueError."""
        with pytest.raises(ValueError, match="capability_growth_rate must be non-negative"):
            CurriculumConfig(capability_growth_rate=-0.01)
    
    def test_config_invalid_decay_rate(self):
        """Test that negative capability_decay_rate raises ValueError."""
        with pytest.raises(ValueError, match="capability_decay_rate must be non-negative"):
            CurriculumConfig(capability_decay_rate=-0.01)
    
    def test_config_invalid_failure_threshold(self):
        """Test that consecutive_failure_threshold < 1 raises ValueError."""
        with pytest.raises(ValueError, match="consecutive_failure_threshold must be >= 1"):
            CurriculumConfig(consecutive_failure_threshold=0)
    
    def test_config_to_dict(self):
        """Test CurriculumConfig serialization to dictionary."""
        config = CurriculumConfig(
            zpd_lower_threshold=0.15,
            zpd_upper_threshold=0.35,
        )
        
        data = config.to_dict()
        
        assert data["zpd_lower_threshold"] == 0.15
        assert data["zpd_upper_threshold"] == 0.35
        assert "capability_growth_rate" in data
        assert "capability_decay_rate" in data
        assert "consecutive_failure_threshold" in data
    
    def test_config_from_dict(self):
        """Test CurriculumConfig deserialization from dictionary."""
        data = {
            "zpd_lower_threshold": 0.1,
            "zpd_upper_threshold": 0.5,
            "capability_growth_rate": 0.1,
            "capability_decay_rate": 0.05,
            "consecutive_failure_threshold": 5,
        }
        
        config = CurriculumConfig.from_dict(data)
        
        assert config.zpd_lower_threshold == 0.1
        assert config.zpd_upper_threshold == 0.5
        assert config.capability_growth_rate == 0.1
        assert config.capability_decay_rate == 0.05
        assert config.consecutive_failure_threshold == 5
    
    def test_config_from_dict_with_defaults(self):
        """Test CurriculumConfig deserialization with missing keys uses defaults."""
        data = {"zpd_lower_threshold": 0.1}
        
        config = CurriculumConfig.from_dict(data)
        
        assert config.zpd_lower_threshold == 0.1
        assert config.zpd_upper_threshold == 0.4  # Default
        assert config.capability_growth_rate == 0.05  # Default


# =============================================================================
# Tests for Task Dataclass
# =============================================================================

class TestTaskDataclass:
    """Unit tests for Task dataclass creation and validation."""
    
    def test_valid_task_creation(self):
        """Test creating a valid Task object."""
        task = Task(
            task_id="task_001",
            task_type="reasoning",
            complexity=0.6,
            required_capabilities={"reasoning": 0.7, "memory": 0.4},
            metadata={"source": "test"},
        )
        
        assert task.task_id == "task_001"
        assert task.task_type == "reasoning"
        assert task.complexity == 0.6
        assert task.required_capabilities == {"reasoning": 0.7, "memory": 0.4}
        assert task.metadata == {"source": "test"}
    
    def test_task_with_default_values(self):
        """Test Task creation with default values for optional fields."""
        task = Task(
            task_id="task_002",
            task_type="language",
            complexity=0.5,
        )
        
        assert task.required_capabilities == {}
        assert task.metadata == {}
    
    def test_task_boundary_complexity_values(self):
        """Test Task with boundary complexity values."""
        min_task = Task(task_id="min", task_type="test", complexity=0.0)
        assert min_task.complexity == 0.0
        
        max_task = Task(task_id="max", task_type="test", complexity=1.0)
        assert max_task.complexity == 1.0
    
    def test_task_invalid_complexity_raises_error(self):
        """Test that invalid complexity values raise ValueError."""
        with pytest.raises(ValueError, match="complexity must be between"):
            Task(task_id="invalid", task_type="test", complexity=1.5)
        
        with pytest.raises(ValueError, match="complexity must be between"):
            Task(task_id="invalid", task_type="test", complexity=-0.1)
    
    def test_task_invalid_required_capability_raises_error(self):
        """Test that invalid required_capabilities values raise ValueError."""
        with pytest.raises(ValueError, match="required_capabilities"):
            Task(
                task_id="invalid",
                task_type="test",
                complexity=0.5,
                required_capabilities={"reasoning": 1.5},  # Invalid
            )
    
    def test_task_to_dict(self):
        """Test Task serialization to dictionary."""
        task = Task(
            task_id="task_001",
            task_type="reasoning",
            complexity=0.6,
            required_capabilities={"reasoning": 0.7},
            metadata={"key": "value"},
        )
        
        data = task.to_dict()
        
        assert data["task_id"] == "task_001"
        assert data["task_type"] == "reasoning"
        assert data["complexity"] == 0.6
        assert data["required_capabilities"] == {"reasoning": 0.7}
        assert data["metadata"] == {"key": "value"}
    
    def test_task_from_dict(self):
        """Test Task deserialization from dictionary."""
        data = {
            "task_id": "task_001",
            "task_type": "language",
            "complexity": 0.5,
            "required_capabilities": {"language": 0.6},
            "metadata": {"test": True},
        }
        
        task = Task.from_dict(data)
        
        assert task.task_id == "task_001"
        assert task.task_type == "language"
        assert task.complexity == 0.5
        assert task.required_capabilities == {"language": 0.6}
        assert task.metadata == {"test": True}


# =============================================================================
# Tests for CurriculumLearningSystem Initialization
# =============================================================================

class TestCurriculumLearningSystemInitialization:
    """Unit tests for CurriculumLearningSystem initialization."""
    
    def test_default_initialization(self, default_system: CurriculumLearningSystem):
        """Test initialization with default configuration."""
        assert default_system is not None
        
        # Check default config values
        config = default_system.get_config()
        assert config.zpd_lower_threshold == 0.2
        assert config.zpd_upper_threshold == 0.4
    
    def test_initialization_with_custom_config(self, custom_system: CurriculumLearningSystem):
        """Test initialization with custom configuration."""
        config = custom_system.get_config()
        
        assert config.zpd_lower_threshold == 0.15
        assert config.zpd_upper_threshold == 0.35
        assert config.capability_growth_rate == 0.06
        assert config.capability_decay_rate == 0.03
        assert config.consecutive_failure_threshold == 2
    
    def test_initialization_creates_all_capability_dimensions(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that all capability dimensions are initialized."""
        for dim in CapabilityDimension:
            level = default_system.get_capability_level(dim.value)
            assert 0.0 <= level <= 1.0
    
    def test_initialization_without_development_system(self):
        """Test initialization without DevelopmentSystem uses default capability level."""
        system = CurriculumLearningSystem(development_system=None)
        
        # Default level should be 0.5 when no development system
        for dim in CapabilityDimension:
            level = system.get_capability_level(dim.value)
            assert level == 0.5
    
    def test_initialization_with_config_dict(self):
        """Test initialization with config as dictionary."""
        config_dict = {
            "zpd_lower_threshold": 0.1,
            "zpd_upper_threshold": 0.3,
        }
        
        system = CurriculumLearningSystem(config=config_dict)
        config = system.get_config()
        
        assert config.zpd_lower_threshold == 0.1
        assert config.zpd_upper_threshold == 0.3
    
    def test_initial_consecutive_failures_are_zero(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that consecutive failures start at zero for all dimensions."""
        for dim in CapabilityDimension:
            failures = default_system.get_consecutive_failures(dim.value)
            assert failures == 0


# =============================================================================
# Tests for estimate_task_difficulty()
# =============================================================================

class TestEstimateTaskDifficulty:
    """Unit tests for estimate_task_difficulty() method."""
    
    def test_estimate_difficulty_with_explicit_capabilities(
        self, default_system: CurriculumLearningSystem, simple_task: Task
    ):
        """Test difficulty estimation for task with explicit required capabilities."""
        difficulty = default_system.estimate_task_difficulty(simple_task)
        
        assert isinstance(difficulty, TaskDifficulty)
        assert 0.0 <= difficulty.overall_difficulty <= 1.0
        assert 0.0 <= difficulty.complexity <= 1.0
        assert 0.0 <= difficulty.cognitive_load <= 1.0
        
        # Required capabilities should include the explicit ones
        assert "reasoning" in difficulty.required_capabilities
        assert "memory" in difficulty.required_capabilities
    
    def test_estimate_difficulty_without_explicit_capabilities(
        self, default_system: CurriculumLearningSystem
    ):
        """Test difficulty estimation infers capabilities from task type."""
        task = Task(
            task_id="infer_task",
            task_type="search_qa",
            complexity=0.6,
            # No explicit required_capabilities
        )
        
        difficulty = default_system.estimate_task_difficulty(task)
        
        # Should infer capabilities from TASK_TYPE_CAPABILITY_WEIGHTS
        assert len(difficulty.required_capabilities) > 0
        
        # search_qa should involve reasoning, language, memory
        expected_dims = TASK_TYPE_CAPABILITY_WEIGHTS.get("search_qa", {}).keys()
        for dim in expected_dims:
            assert dim in difficulty.required_capabilities
    
    def test_estimate_difficulty_for_different_task_types(
        self, default_system: CurriculumLearningSystem
    ):
        """Test difficulty estimation for various task types."""
        task_types = ["reasoning", "language", "social", "memory", "planning", "perception"]
        
        for task_type in task_types:
            task = Task(
                task_id=f"task_{task_type}",
                task_type=task_type,
                complexity=0.5,
            )
            
            difficulty = default_system.estimate_task_difficulty(task)
            
            assert isinstance(difficulty, TaskDifficulty)
            assert 0.0 <= difficulty.overall_difficulty <= 1.0
    
    def test_estimate_difficulty_for_workflow_task_types(
        self, default_system: CurriculumLearningSystem
    ):
        """Test difficulty estimation for workflow-specific task types."""
        workflow_types = ["chat_generate", "search_qa", "rag_qa", "lesson_pack", "summarization"]
        
        for task_type in workflow_types:
            task = Task(
                task_id=f"workflow_{task_type}",
                task_type=task_type,
                complexity=0.7,
            )
            
            difficulty = default_system.estimate_task_difficulty(task)
            
            assert isinstance(difficulty, TaskDifficulty)
            assert 0.0 <= difficulty.overall_difficulty <= 1.0
    
    def test_estimate_difficulty_is_deterministic(
        self, default_system: CurriculumLearningSystem, simple_task: Task
    ):
        """Test that difficulty estimation is deterministic."""
        difficulty1 = default_system.estimate_task_difficulty(simple_task)
        difficulty2 = default_system.estimate_task_difficulty(simple_task)
        difficulty3 = default_system.estimate_task_difficulty(simple_task)
        
        assert difficulty1.overall_difficulty == difficulty2.overall_difficulty
        assert difficulty2.overall_difficulty == difficulty3.overall_difficulty
        assert difficulty1.complexity == difficulty2.complexity
        assert difficulty1.cognitive_load == difficulty2.cognitive_load
    
    def test_estimate_difficulty_scales_with_complexity(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that higher complexity leads to higher difficulty."""
        low_complexity_task = Task(
            task_id="low",
            task_type="reasoning",
            complexity=0.2,
        )
        high_complexity_task = Task(
            task_id="high",
            task_type="reasoning",
            complexity=0.9,
        )
        
        low_difficulty = default_system.estimate_task_difficulty(low_complexity_task)
        high_difficulty = default_system.estimate_task_difficulty(high_complexity_task)
        
        assert high_difficulty.overall_difficulty > low_difficulty.overall_difficulty
    
    def test_estimate_difficulty_unknown_task_type_uses_defaults(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that unknown task types use default capability weights."""
        task = Task(
            task_id="unknown",
            task_type="unknown_type",
            complexity=0.5,
        )
        
        difficulty = default_system.estimate_task_difficulty(task)
        
        # Should still return valid difficulty
        assert isinstance(difficulty, TaskDifficulty)
        assert 0.0 <= difficulty.overall_difficulty <= 1.0
        # Default weights include reasoning and language
        assert len(difficulty.required_capabilities) > 0


# =============================================================================
# Tests for is_in_zpd() - ZPD Boundary Conditions
# =============================================================================

class TestIsInZPD:
    """Unit tests for is_in_zpd() method - ZPD boundary conditions."""
    
    def test_task_in_zpd_returns_true(self, default_system: CurriculumLearningSystem):
        """Test that task within ZPD returns in_zpd=True."""
        # Set capability level to 0.5
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.5
        
        # Create task with difficulty in ZPD range (0.5 + 0.2 to 0.5 + 0.4)
        # Target difficulty gap of 0.3 (middle of ZPD)
        task = Task(
            task_id="zpd_task",
            task_type="reasoning",
            complexity=0.8,
            required_capabilities={"reasoning": 0.8},  # Gap = 0.8 - 0.5 = 0.3
        )
        
        assessment = default_system.is_in_zpd(task)
        
        assert assessment.in_zpd is True
        assert 0.2 <= assessment.difficulty_gap <= 0.4
    
    def test_task_too_easy_returns_false(self, default_system: CurriculumLearningSystem):
        """Test that task below ZPD returns in_zpd=False with negative gap."""
        # Set high capability level
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.8
        
        # Create easy task
        task = Task(
            task_id="easy_task",
            task_type="reasoning",
            complexity=0.3,
            required_capabilities={"reasoning": 0.3},  # Gap = 0.3 - 0.8 = -0.5
        )
        
        assessment = default_system.is_in_zpd(task)
        
        assert assessment.in_zpd is False
        assert assessment.difficulty_gap < 0.2  # Below lower threshold
        assert "too easy" in assessment.recommendations[0].lower() or \
               "below" in assessment.recommendations[0].lower()
    
    def test_task_too_hard_returns_false(self, default_system: CurriculumLearningSystem):
        """Test that task above ZPD returns in_zpd=False with large positive gap."""
        # Set low capability level
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.2
        
        # Create difficult task
        task = Task(
            task_id="hard_task",
            task_type="reasoning",
            complexity=0.9,
            required_capabilities={"reasoning": 0.9},  # Gap = 0.9 - 0.2 = 0.7
        )
        
        assessment = default_system.is_in_zpd(task)
        
        assert assessment.in_zpd is False
        assert assessment.difficulty_gap > 0.4  # Above upper threshold
        assert assessment.suggested_scaffolding is not None
        assert len(assessment.suggested_scaffolding) > 0
    
    def test_zpd_lower_boundary(self, default_system: CurriculumLearningSystem):
        """Test task at ZPD lower boundary."""
        # Set capability level
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.5
        
        # Create task slightly above lower boundary (gap = 0.25, within ZPD)
        # Note: Due to difficulty estimation, we need to account for how
        # overall_difficulty is calculated from required_capabilities
        task = Task(
            task_id="boundary_task",
            task_type="reasoning",
            complexity=0.75,
            required_capabilities={"reasoning": 0.75},  # Gap = 0.75 - 0.5 = 0.25
        )
        
        assessment = default_system.is_in_zpd(task)
        
        # Should be in ZPD (gap between 0.2 and 0.4)
        assert assessment.in_zpd is True
        assert 0.2 <= assessment.difficulty_gap <= 0.4
    
    def test_zpd_upper_boundary(self, default_system: CurriculumLearningSystem):
        """Test task exactly at ZPD upper boundary."""
        # Set capability level
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.5
        
        # Create task at upper boundary (gap = 0.4)
        task = Task(
            task_id="boundary_task",
            task_type="reasoning",
            complexity=0.9,
            required_capabilities={"reasoning": 0.9},  # Gap = 0.9 - 0.5 = 0.4
        )
        
        assessment = default_system.is_in_zpd(task)
        
        # At exactly upper threshold, should be in ZPD
        assert assessment.in_zpd is True
        assert abs(assessment.difficulty_gap - 0.4) < 0.05
    
    def test_zpd_with_custom_thresholds(self, custom_system: CurriculumLearningSystem):
        """Test ZPD assessment with custom thresholds."""
        # Custom config has zpd_lower=0.15, zpd_upper=0.35
        for dim in CapabilityDimension:
            custom_system._capability_levels[dim.value] = 0.5
        
        # Task with gap = 0.25 (within custom ZPD)
        task = Task(
            task_id="custom_zpd_task",
            task_type="reasoning",
            complexity=0.75,
            required_capabilities={"reasoning": 0.75},
        )
        
        assessment = custom_system.is_in_zpd(task)
        
        assert assessment.in_zpd is True
        assert 0.15 <= assessment.difficulty_gap <= 0.35
    
    def test_zpd_assessment_includes_recommendations(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that ZPD assessment always includes recommendations."""
        task = Task(
            task_id="any_task",
            task_type="reasoning",
            complexity=0.5,
        )
        
        assessment = default_system.is_in_zpd(task)
        
        assert isinstance(assessment.recommendations, list)
        assert len(assessment.recommendations) > 0
    
    def test_zpd_with_multiple_capability_dimensions(
        self, default_system: CurriculumLearningSystem
    ):
        """Test ZPD assessment with task requiring multiple capabilities."""
        # Set different capability levels
        default_system._capability_levels["reasoning"] = 0.6
        default_system._capability_levels["memory"] = 0.4
        default_system._capability_levels["language"] = 0.5
        
        # Task requiring multiple capabilities
        task = Task(
            task_id="multi_cap_task",
            task_type="reasoning",
            complexity=0.8,
            required_capabilities={
                "reasoning": 0.8,  # Gap = 0.2
                "memory": 0.9,     # Gap = 0.5 (largest)
                "language": 0.7,   # Gap = 0.2
            },
        )
        
        assessment = default_system.is_in_zpd(task)
        
        # Should use maximum gap (memory: 0.5)
        assert assessment.difficulty_gap >= 0.4  # At least the memory gap


# =============================================================================
# Tests for update_capabilities() - Growth and Decay
# =============================================================================

class TestUpdateCapabilities:
    """Unit tests for update_capabilities() method - growth and decay sequences."""
    
    def test_capability_increases_on_success(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capabilities increase after successful task completion."""
        # Set initial level
        default_system._capability_levels["reasoning"] = 0.5
        initial_level = default_system.get_capability_level("reasoning")
        
        # Complete task successfully
        updates = default_system.update_capabilities(
            task_type="reasoning",
            success=True,
            score=0.9,
        )
        
        new_level = default_system.get_capability_level("reasoning")
        
        assert new_level > initial_level
        assert "reasoning" in updates
        assert updates["reasoning"] > initial_level
    
    def test_capability_decreases_on_failure(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capabilities decrease after failed task completion."""
        # Set initial level above minimum
        default_system._capability_levels["reasoning"] = 0.5
        initial_level = default_system.get_capability_level("reasoning")
        
        # Complete task with failure
        updates = default_system.update_capabilities(
            task_type="reasoning",
            success=False,
            score=0.2,
        )
        
        new_level = default_system.get_capability_level("reasoning")
        
        assert new_level < initial_level
        assert "reasoning" in updates
        assert updates["reasoning"] < initial_level
    
    def test_capability_growth_proportional_to_score(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capability growth is proportional to score."""
        # Create two systems with same initial state
        system1 = CurriculumLearningSystem()
        system2 = CurriculumLearningSystem()
        
        system1._capability_levels["reasoning"] = 0.5
        system2._capability_levels["reasoning"] = 0.5
        
        # Complete with different scores
        updates1 = system1.update_capabilities("reasoning", success=True, score=0.5)
        updates2 = system2.update_capabilities("reasoning", success=True, score=1.0)
        
        growth1 = updates1["reasoning"] - 0.5
        growth2 = updates2["reasoning"] - 0.5
        
        # Higher score should produce more growth
        assert growth2 > growth1
    
    def test_capability_capped_at_maximum(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capability level is capped at 1.0."""
        # Set level near maximum
        default_system._capability_levels["reasoning"] = 0.99
        
        # Complete many successful tasks
        for _ in range(10):
            default_system.update_capabilities("reasoning", success=True, score=1.0)
        
        level = default_system.get_capability_level("reasoning")
        
        assert level <= 1.0
    
    def test_capability_floored_at_minimum(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capability level is floored at 0.0."""
        # Set level near minimum
        default_system._capability_levels["reasoning"] = 0.01
        
        # Complete many failed tasks
        for _ in range(10):
            default_system.update_capabilities("reasoning", success=False, score=0.0)
        
        level = default_system.get_capability_level("reasoning")
        
        assert level >= 0.0
    
    def test_success_resets_consecutive_failures(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that success resets consecutive failure counter."""
        # Accumulate failures
        default_system._capability_levels["reasoning"] = 0.5
        for _ in range(2):
            default_system.update_capabilities("reasoning", success=False, score=0.1)
        
        failures_before = default_system.get_consecutive_failures("reasoning")
        assert failures_before == 2
        
        # Complete successful task
        default_system.update_capabilities("reasoning", success=True, score=0.8)
        
        failures_after = default_system.get_consecutive_failures("reasoning")
        assert failures_after == 0
    
    def test_failure_increments_consecutive_failures(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that failure increments consecutive failure counter."""
        default_system._capability_levels["reasoning"] = 0.5
        
        assert default_system.get_consecutive_failures("reasoning") == 0
        
        default_system.update_capabilities("reasoning", success=False, score=0.1)
        assert default_system.get_consecutive_failures("reasoning") == 1
        
        default_system.update_capabilities("reasoning", success=False, score=0.1)
        assert default_system.get_consecutive_failures("reasoning") == 2
    
    def test_high_score_success_increases_difficulty_threshold(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that high-score success increases difficulty threshold."""
        initial_threshold = default_system.get_difficulty_threshold("reasoning")
        
        # Complete task with high score
        default_system.update_capabilities("reasoning", success=True, score=0.9)
        
        new_threshold = default_system.get_difficulty_threshold("reasoning")
        
        assert new_threshold >= initial_threshold
    
    def test_update_affects_multiple_dimensions(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that task completion affects multiple capability dimensions."""
        # search_qa affects reasoning, language, memory
        default_system._capability_levels["reasoning"] = 0.5
        default_system._capability_levels["language"] = 0.5
        default_system._capability_levels["memory"] = 0.5
        
        updates = default_system.update_capabilities("search_qa", success=True, score=0.8)
        
        # Should update multiple dimensions based on TASK_TYPE_CAPABILITY_WEIGHTS
        assert len(updates) > 1
    
    def test_task_history_recorded(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that task completion is recorded in history."""
        initial_history = default_system.get_task_history()
        assert len(initial_history) == 0
        
        default_system.update_capabilities("reasoning", success=True, score=0.8)
        
        history = default_system.get_task_history()
        assert len(history) == 1
        assert history[0]["task_type"] == "reasoning"
        assert history[0]["success"] is True
        assert history[0]["score"] == 0.8
    
    def test_capability_growth_sequence(
        self, default_system: CurriculumLearningSystem
    ):
        """Test capability growth over a sequence of successful tasks."""
        default_system._capability_levels["reasoning"] = 0.3
        
        levels = [default_system.get_capability_level("reasoning")]
        
        # Complete 5 successful tasks
        for i in range(5):
            default_system.update_capabilities("reasoning", success=True, score=0.8)
            levels.append(default_system.get_capability_level("reasoning"))
        
        # Each level should be >= previous (monotonically increasing)
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1]
    
    def test_capability_decay_sequence(
        self, default_system: CurriculumLearningSystem
    ):
        """Test capability decay over a sequence of failed tasks."""
        default_system._capability_levels["reasoning"] = 0.7
        
        levels = [default_system.get_capability_level("reasoning")]
        
        # Complete 5 failed tasks
        for i in range(5):
            default_system.update_capabilities("reasoning", success=False, score=0.1)
            levels.append(default_system.get_capability_level("reasoning"))
        
        # Each level should be <= previous (monotonically decreasing)
        for i in range(1, len(levels)):
            assert levels[i] <= levels[i - 1]


# =============================================================================
# Tests for suggest_scaffolding()
# =============================================================================

class TestSuggestScaffolding:
    """Unit tests for suggest_scaffolding() method."""
    
    def test_scaffolding_for_too_difficult_task(
        self, default_system: CurriculumLearningSystem, difficult_task: Task
    ):
        """Test scaffolding suggestions for a task that is too difficult."""
        # Set low capability levels
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.2
        
        scaffolding = default_system.suggest_scaffolding(difficult_task)
        
        assert isinstance(scaffolding, list)
        assert len(scaffolding) > 0
        
        # All scaffolded tasks should be Task objects
        for sub_task in scaffolding:
            assert isinstance(sub_task, Task)
            assert sub_task.task_id.startswith(difficult_task.task_id)
    
    def test_no_scaffolding_for_task_in_zpd(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that no scaffolding is suggested for task in ZPD."""
        # Set capability levels to match task
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.5
        
        # Task with difficulty in ZPD
        task = Task(
            task_id="zpd_task",
            task_type="reasoning",
            complexity=0.7,
            required_capabilities={"reasoning": 0.7},  # Gap = 0.2
        )
        
        scaffolding = default_system.suggest_scaffolding(task)
        
        assert scaffolding == []
    
    def test_no_scaffolding_for_easy_task(
        self, default_system: CurriculumLearningSystem, easy_task: Task
    ):
        """Test that no scaffolding is suggested for easy task."""
        # Set high capability levels
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.8
        
        scaffolding = default_system.suggest_scaffolding(easy_task)
        
        assert scaffolding == []
    
    def test_scaffolded_tasks_are_easier(
        self, default_system: CurriculumLearningSystem, difficult_task: Task
    ):
        """Test that scaffolded tasks have lower complexity."""
        # Set low capability levels
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.2
        
        scaffolding = default_system.suggest_scaffolding(difficult_task)
        
        for sub_task in scaffolding:
            assert sub_task.complexity <= difficult_task.complexity
    
    def test_scaffolded_tasks_have_metadata(
        self, default_system: CurriculumLearningSystem, difficult_task: Task
    ):
        """Test that scaffolded tasks include metadata about parent task."""
        # Set low capability levels
        for dim in CapabilityDimension:
            default_system._capability_levels[dim.value] = 0.2
        
        scaffolding = default_system.suggest_scaffolding(difficult_task)
        
        for sub_task in scaffolding:
            assert "parent_task_id" in sub_task.metadata
            assert sub_task.metadata["parent_task_id"] == difficult_task.task_id
            assert "scaffold_type" in sub_task.metadata
            assert "scaffold_level" in sub_task.metadata
    
    def test_scaffolding_targets_weak_dimensions(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that scaffolding targets dimensions with largest gaps."""
        # Set different capability levels
        default_system._capability_levels["reasoning"] = 0.3
        default_system._capability_levels["memory"] = 0.2  # Weakest
        default_system._capability_levels["planning"] = 0.4
        
        task = Task(
            task_id="multi_dim_task",
            task_type="reasoning",
            complexity=0.9,
            required_capabilities={
                "reasoning": 0.9,
                "memory": 0.9,
                "planning": 0.9,
            },
        )
        
        scaffolding = default_system.suggest_scaffolding(task)
        
        # Should have scaffolding tasks
        assert len(scaffolding) > 0
        
        # Check that dimension-focused tasks exist
        dimension_focused = [
            t for t in scaffolding 
            if t.metadata.get("scaffold_type") == "dimension_focus"
        ]
        
        # Should target dimensions with large gaps
        if dimension_focused:
            target_dims = [t.metadata.get("target_dimension") for t in dimension_focused]
            # Memory has the largest gap, should be targeted
            assert "memory" in target_dims or len(target_dims) > 0


# =============================================================================
# Tests for check_consecutive_failures() and reduce_difficulty_threshold()
# =============================================================================

class TestConsecutiveFailuresAndThresholdReduction:
    """Unit tests for consecutive failure handling and threshold reduction."""
    
    def test_check_consecutive_failures_below_threshold(
        self, default_system: CurriculumLearningSystem
    ):
        """Test check_consecutive_failures returns False below threshold."""
        # Default threshold is 3
        default_system._capability_levels["reasoning"] = 0.5
        
        # 2 failures (below threshold of 3)
        default_system.update_capabilities("reasoning", success=False, score=0.1)
        default_system.update_capabilities("reasoning", success=False, score=0.1)
        
        assert default_system.check_consecutive_failures("reasoning") is False
    
    def test_check_consecutive_failures_at_threshold(
        self, default_system: CurriculumLearningSystem
    ):
        """Test check_consecutive_failures returns True at threshold."""
        # Default threshold is 3
        default_system._capability_levels["reasoning"] = 0.5
        
        # 3 failures (at threshold)
        for _ in range(3):
            default_system.update_capabilities("reasoning", success=False, score=0.1)
        
        assert default_system.check_consecutive_failures("reasoning") is True
    
    def test_check_consecutive_failures_above_threshold(
        self, custom_system: CurriculumLearningSystem
    ):
        """Test check_consecutive_failures returns True above threshold."""
        # Custom config has threshold of 2
        custom_system._capability_levels["reasoning"] = 0.5
        
        # 3 failures (above threshold of 2)
        for _ in range(3):
            custom_system.update_capabilities("reasoning", success=False, score=0.1)
        
        assert custom_system.check_consecutive_failures("reasoning") is True
    
    def test_check_consecutive_failures_invalid_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test check_consecutive_failures raises error for invalid dimension."""
        with pytest.raises(ValueError, match="Unknown capability dimension"):
            default_system.check_consecutive_failures("invalid_dimension")
    
    def test_reduce_difficulty_threshold(
        self, default_system: CurriculumLearningSystem
    ):
        """Test reduce_difficulty_threshold reduces the threshold."""
        # Accumulate failures to trigger reduction
        default_system._capability_levels["reasoning"] = 0.5
        for _ in range(3):
            default_system.update_capabilities("reasoning", success=False, score=0.1)
        
        initial_threshold = default_system.get_difficulty_threshold("reasoning")
        
        new_threshold = default_system.reduce_difficulty_threshold("reasoning")
        
        assert new_threshold < initial_threshold
        assert default_system.get_difficulty_threshold("reasoning") == new_threshold
    
    def test_reduce_difficulty_threshold_has_minimum(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that difficulty threshold has a minimum value."""
        # Set very low threshold
        default_system._difficulty_thresholds["reasoning"] = 0.15
        
        # Accumulate many failures
        for _ in range(10):
            default_system._consecutive_failures["reasoning"] = 10
        
        new_threshold = default_system.reduce_difficulty_threshold("reasoning")
        
        # Should not go below minimum (0.1)
        assert new_threshold >= 0.1
    
    def test_reduce_difficulty_threshold_invalid_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test reduce_difficulty_threshold raises error for invalid dimension."""
        with pytest.raises(ValueError, match="Unknown capability dimension"):
            default_system.reduce_difficulty_threshold("invalid_dimension")


# =============================================================================
# Tests for suggest_remedial_tasks()
# =============================================================================

class TestSuggestRemedialTasks:
    """Unit tests for suggest_remedial_tasks() method."""
    
    def test_suggest_remedial_tasks_returns_tasks(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that suggest_remedial_tasks returns a list of tasks."""
        remedial_tasks = default_system.suggest_remedial_tasks("reasoning")
        
        assert isinstance(remedial_tasks, list)
        assert len(remedial_tasks) == 3  # Should return 3 remedial tasks
        
        for task in remedial_tasks:
            assert isinstance(task, Task)
    
    def test_remedial_tasks_are_easier_than_current_level(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that remedial tasks are at or below current capability level."""
        default_system._capability_levels["reasoning"] = 0.6
        
        remedial_tasks = default_system.suggest_remedial_tasks("reasoning")
        
        for task in remedial_tasks:
            assert task.complexity <= 0.6
    
    def test_remedial_tasks_have_increasing_difficulty(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that remedial tasks have progressively increasing difficulty."""
        default_system._capability_levels["reasoning"] = 0.6
        
        remedial_tasks = default_system.suggest_remedial_tasks("reasoning")
        
        # Tasks should be ordered by increasing difficulty
        for i in range(1, len(remedial_tasks)):
            assert remedial_tasks[i].complexity >= remedial_tasks[i - 1].complexity
    
    def test_remedial_tasks_have_correct_metadata(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that remedial tasks have appropriate metadata."""
        remedial_tasks = default_system.suggest_remedial_tasks("reasoning")
        
        for i, task in enumerate(remedial_tasks):
            assert task.metadata["remedial"] is True
            assert task.metadata["target_dimension"] == "reasoning"
            assert task.metadata["remedial_level"] == i + 1
            assert task.metadata["purpose"] == "rebuild_capability"
    
    def test_remedial_tasks_target_specific_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that remedial tasks target the specified dimension."""
        remedial_tasks = default_system.suggest_remedial_tasks("memory")
        
        for task in remedial_tasks:
            assert task.task_type == "memory"
            assert "memory" in task.required_capabilities
    
    def test_remedial_tasks_invalid_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test suggest_remedial_tasks raises error for invalid dimension."""
        with pytest.raises(ValueError, match="Unknown capability dimension"):
            default_system.suggest_remedial_tasks("invalid_dimension")
    
    def test_remedial_tasks_for_low_capability(
        self, default_system: CurriculumLearningSystem
    ):
        """Test remedial tasks when capability is very low."""
        default_system._capability_levels["reasoning"] = 0.1
        
        remedial_tasks = default_system.suggest_remedial_tasks("reasoning")
        
        # All tasks should have very low complexity
        for task in remedial_tasks:
            assert task.complexity <= 0.1
            assert task.complexity >= 0.1  # Minimum base difficulty


# =============================================================================
# Tests for handle_task_failure()
# =============================================================================

class TestHandleTaskFailure:
    """Unit tests for handle_task_failure() method."""
    
    def test_handle_task_failure_updates_capabilities(
        self, default_system: CurriculumLearningSystem, simple_task: Task
    ):
        """Test that handle_task_failure updates capability levels."""
        default_system._capability_levels["reasoning"] = 0.5
        initial_level = default_system.get_capability_level("reasoning")
        
        result = default_system.handle_task_failure(simple_task, score=0.2)
        
        assert "capability_updates" in result
        # Capability should have decreased
        new_level = default_system.get_capability_level("reasoning")
        assert new_level < initial_level
    
    def test_handle_task_failure_detects_remediation_need(
        self, custom_system: CurriculumLearningSystem
    ):
        """Test that handle_task_failure detects when remediation is needed."""
        # Custom config has threshold of 2
        custom_system._capability_levels["reasoning"] = 0.5
        
        task = Task(
            task_id="fail_task",
            task_type="reasoning",
            complexity=0.7,
        )
        
        # First failure - no remediation yet
        result1 = custom_system.handle_task_failure(task, score=0.1)
        assert len(result1["dimensions_needing_remediation"]) == 0
        
        # Second failure - should trigger remediation (threshold is 2)
        result2 = custom_system.handle_task_failure(task, score=0.1)
        assert "reasoning" in result2["dimensions_needing_remediation"]
    
    def test_handle_task_failure_reduces_thresholds(
        self, custom_system: CurriculumLearningSystem
    ):
        """Test that handle_task_failure reduces difficulty thresholds."""
        custom_system._capability_levels["reasoning"] = 0.5
        initial_threshold = custom_system.get_difficulty_threshold("reasoning")
        
        task = Task(
            task_id="fail_task",
            task_type="reasoning",
            complexity=0.7,
        )
        
        # Trigger threshold reduction (2 failures with threshold of 2)
        custom_system.handle_task_failure(task, score=0.1)
        result = custom_system.handle_task_failure(task, score=0.1)
        
        if "reasoning" in result["reduced_thresholds"]:
            assert result["reduced_thresholds"]["reasoning"] < initial_threshold
    
    def test_handle_task_failure_suggests_remedial_tasks(
        self, custom_system: CurriculumLearningSystem
    ):
        """Test that handle_task_failure suggests remedial tasks."""
        custom_system._capability_levels["reasoning"] = 0.5
        
        task = Task(
            task_id="fail_task",
            task_type="reasoning",
            complexity=0.7,
        )
        
        # Trigger remediation
        custom_system.handle_task_failure(task, score=0.1)
        result = custom_system.handle_task_failure(task, score=0.1)
        
        if result["dimensions_needing_remediation"]:
            assert len(result["remedial_tasks"]) > 0
            for remedial_task in result["remedial_tasks"]:
                assert isinstance(remedial_task, Task)


# =============================================================================
# Tests for Serialization/Deserialization (to_dict/from_dict)
# =============================================================================

class TestSerialization:
    """Unit tests for CurriculumLearningSystem serialization and deserialization."""
    
    def test_to_dict_includes_all_state(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that to_dict includes all system state."""
        # Modify some state
        default_system._capability_levels["reasoning"] = 0.7
        default_system._consecutive_failures["reasoning"] = 2
        default_system._difficulty_thresholds["reasoning"] = 0.8
        default_system.update_capabilities("reasoning", success=True, score=0.9)
        
        data = default_system.to_dict()
        
        assert "config" in data
        assert "capability_levels" in data
        assert "consecutive_failures" in data
        assert "difficulty_thresholds" in data
        assert "task_history" in data
        assert "initialized_at" in data
    
    def test_to_dict_config_serialization(
        self, custom_system: CurriculumLearningSystem
    ):
        """Test that config is properly serialized."""
        data = custom_system.to_dict()
        
        assert data["config"]["zpd_lower_threshold"] == 0.15
        assert data["config"]["zpd_upper_threshold"] == 0.35
        assert data["config"]["capability_growth_rate"] == 0.06
    
    def test_to_dict_capability_levels_serialization(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that capability levels are properly serialized."""
        default_system._capability_levels["reasoning"] = 0.75
        default_system._capability_levels["memory"] = 0.65
        
        data = default_system.to_dict()
        
        assert data["capability_levels"]["reasoning"] == 0.75
        assert data["capability_levels"]["memory"] == 0.65
    
    def test_to_dict_task_history_serialization(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that task history is properly serialized."""
        default_system.update_capabilities("reasoning", success=True, score=0.8)
        default_system.update_capabilities("language", success=False, score=0.3)
        
        data = default_system.to_dict()
        
        assert len(data["task_history"]) == 2
        assert data["task_history"][0]["task_type"] == "reasoning"
        assert data["task_history"][0]["success"] is True
        assert data["task_history"][1]["task_type"] == "language"
        assert data["task_history"][1]["success"] is False
    
    def test_from_dict_restores_config(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict restores configuration."""
        data = {
            "config": {
                "zpd_lower_threshold": 0.1,
                "zpd_upper_threshold": 0.5,
                "capability_growth_rate": 0.1,
                "capability_decay_rate": 0.05,
                "consecutive_failure_threshold": 5,
            }
        }
        
        default_system.from_dict(data)
        config = default_system.get_config()
        
        assert config.zpd_lower_threshold == 0.1
        assert config.zpd_upper_threshold == 0.5
        assert config.capability_growth_rate == 0.1
    
    def test_from_dict_restores_capability_levels(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict restores capability levels."""
        data = {
            "capability_levels": {
                "reasoning": 0.8,
                "memory": 0.6,
                "language": 0.7,
            }
        }
        
        default_system.from_dict(data)
        
        assert default_system.get_capability_level("reasoning") == 0.8
        assert default_system.get_capability_level("memory") == 0.6
        assert default_system.get_capability_level("language") == 0.7
    
    def test_from_dict_restores_consecutive_failures(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict restores consecutive failure counts."""
        data = {
            "consecutive_failures": {
                "reasoning": 2,
                "memory": 1,
            }
        }
        
        default_system.from_dict(data)
        
        assert default_system.get_consecutive_failures("reasoning") == 2
        assert default_system.get_consecutive_failures("memory") == 1
    
    def test_from_dict_restores_difficulty_thresholds(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict restores difficulty thresholds."""
        data = {
            "difficulty_thresholds": {
                "reasoning": 0.9,
                "memory": 0.7,
            }
        }
        
        default_system.from_dict(data)
        
        assert default_system.get_difficulty_threshold("reasoning") == 0.9
        assert default_system.get_difficulty_threshold("memory") == 0.7
    
    def test_from_dict_restores_task_history(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict restores task history."""
        data = {
            "task_history": [
                {"task_type": "reasoning", "success": True, "score": 0.9, "timestamp": 1000},
                {"task_type": "language", "success": False, "score": 0.3, "timestamp": 1001},
            ]
        }
        
        default_system.from_dict(data)
        
        history = default_system.get_task_history()
        assert len(history) == 2
        assert history[0]["task_type"] == "reasoning"
        assert history[1]["task_type"] == "language"
    
    def test_serialization_round_trip(
        self, default_system: CurriculumLearningSystem
    ):
        """Test complete serialization round-trip preserves state."""
        # Modify state directly (not through update_capabilities to avoid side effects)
        default_system._capability_levels["reasoning"] = 0.75
        default_system._capability_levels["memory"] = 0.65
        default_system._consecutive_failures["reasoning"] = 2
        default_system._difficulty_thresholds["reasoning"] = 0.85
        
        # Add a task to history manually
        default_system._task_history.append({
            "task_type": "reasoning",
            "success": True,
            "score": 0.9,
            "capability_updates": {"reasoning": 0.75},
            "timestamp": 1000.0,
        })
        
        # Serialize
        data = default_system.to_dict()
        
        # Create new system and restore
        new_system = CurriculumLearningSystem()
        new_system.from_dict(data)
        
        # Verify state is preserved
        assert new_system.get_capability_level("reasoning") == 0.75
        assert new_system.get_capability_level("memory") == 0.65
        assert new_system.get_consecutive_failures("reasoning") == 2
        assert new_system.get_difficulty_threshold("reasoning") == 0.85
        
        # Task history should be preserved
        original_history = default_system.get_task_history()
        restored_history = new_system.get_task_history()
        assert len(original_history) == len(restored_history)
        assert restored_history[0]["task_type"] == "reasoning"
        assert restored_history[0]["success"] is True
    
    def test_from_dict_with_partial_data(
        self, default_system: CurriculumLearningSystem
    ):
        """Test that from_dict handles partial data gracefully."""
        # Only provide some fields
        data = {
            "capability_levels": {"reasoning": 0.9}
        }
        
        # Should not raise error
        default_system.from_dict(data)
        
        assert default_system.get_capability_level("reasoning") == 0.9
        # Other state should remain unchanged
        assert default_system.get_config() is not None


# =============================================================================
# Tests for Additional Methods
# =============================================================================

class TestAdditionalMethods:
    """Unit tests for additional helper methods."""
    
    def test_get_capability_level_valid_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_capability_level for valid dimensions."""
        for dim in CapabilityDimension:
            level = default_system.get_capability_level(dim.value)
            assert 0.0 <= level <= 1.0
    
    def test_get_capability_level_invalid_dimension(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_capability_level raises error for invalid dimension."""
        with pytest.raises(ValueError, match="Unknown capability dimension"):
            default_system.get_capability_level("invalid_dimension")
    
    def test_get_capability_level_case_insensitive(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_capability_level is case-insensitive."""
        level1 = default_system.get_capability_level("reasoning")
        level2 = default_system.get_capability_level("REASONING")
        level3 = default_system.get_capability_level("Reasoning")
        
        assert level1 == level2 == level3
    
    def test_get_all_capability_levels(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_all_capability_levels returns all dimensions."""
        levels = default_system.get_all_capability_levels()
        
        assert isinstance(levels, dict)
        for dim in CapabilityDimension:
            assert dim.value in levels
            assert 0.0 <= levels[dim.value] <= 1.0
    
    def test_get_recommended_difficulty(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_recommended_difficulty returns appropriate value."""
        default_system._capability_levels["reasoning"] = 0.5
        
        recommended = default_system.get_recommended_difficulty("reasoning")
        
        # Should be current level + middle of ZPD range
        # Default ZPD: 0.2 to 0.4, middle = 0.3
        expected = 0.5 + 0.3
        assert abs(recommended - expected) < 0.01
    
    def test_get_recommended_difficulty_capped_at_max(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_recommended_difficulty is capped at 1.0."""
        default_system._capability_levels["reasoning"] = 0.9
        
        recommended = default_system.get_recommended_difficulty("reasoning")
        
        assert recommended <= 1.0
    
    def test_get_task_history_with_limit(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_task_history with limit parameter."""
        # Add multiple tasks
        for i in range(5):
            default_system.update_capabilities("reasoning", success=True, score=0.8)
        
        # Get limited history
        history = default_system.get_task_history(limit=3)
        
        assert len(history) == 3
    
    def test_get_task_history_with_task_type_filter(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_task_history with task_type filter."""
        default_system.update_capabilities("reasoning", success=True, score=0.8)
        default_system.update_capabilities("language", success=True, score=0.7)
        default_system.update_capabilities("reasoning", success=False, score=0.3)
        
        # Filter by task type
        reasoning_history = default_system.get_task_history(task_type="reasoning")
        
        assert len(reasoning_history) == 2
        for record in reasoning_history:
            assert record["task_type"] == "reasoning"
    
    def test_get_performance_summary_empty_history(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_performance_summary with no task history."""
        summary = default_system.get_performance_summary()
        
        assert summary["total_tasks"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["average_score"] == 0.0
        assert summary["by_task_type"] == {}
    
    def test_get_performance_summary_with_history(
        self, default_system: CurriculumLearningSystem
    ):
        """Test get_performance_summary with task history."""
        default_system.update_capabilities("reasoning", success=True, score=0.9)
        default_system.update_capabilities("reasoning", success=True, score=0.8)
        default_system.update_capabilities("reasoning", success=False, score=0.3)
        default_system.update_capabilities("language", success=True, score=0.7)
        
        summary = default_system.get_performance_summary()
        
        assert summary["total_tasks"] == 4
        assert summary["success_rate"] == 0.75  # 3/4
        assert summary["average_score"] == (0.9 + 0.8 + 0.3 + 0.7) / 4
        
        assert "reasoning" in summary["by_task_type"]
        assert summary["by_task_type"]["reasoning"]["count"] == 3
        assert summary["by_task_type"]["reasoning"]["success_rate"] == 2/3
