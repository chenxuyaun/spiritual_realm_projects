"""
Curriculum Learning System.

This module defines the core data structures and the CurriculumLearningSystem
for curriculum-based developmental training. It includes models for task
difficulty assessment, Zone of Proximal Development (ZPD) evaluation, and
capability dimension tracking.

Requirements: 1.1, 1.2, 1.4, 1.5
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import random
import time

if TYPE_CHECKING:
    from mm_orch.consciousness.development import DevelopmentSystem


class CapabilityDimension(Enum):
    """
    Dimensions of cognitive capability.

    These dimensions represent different aspects of cognitive development
    that the curriculum learning system tracks and develops progressively.

    Attributes:
        PERCEPTION: Ability to perceive and process sensory information
        REASONING: Logical and analytical thinking capabilities
        LANGUAGE: Natural language understanding and generation
        SOCIAL: Social interaction and understanding capabilities
        MEMORY: Information storage and retrieval capabilities
        PLANNING: Goal-setting and action planning capabilities
    """

    PERCEPTION = "perception"
    REASONING = "reasoning"
    LANGUAGE = "language"
    SOCIAL = "social"
    MEMORY = "memory"
    PLANNING = "planning"


@dataclass
class TaskDifficulty:
    """
    Task difficulty assessment.

    Represents a comprehensive assessment of a task's difficulty across
    multiple dimensions. Used by the CurriculumLearningSystem to determine
    whether a task is appropriate for the agent's current capability level.

    Attributes:
        complexity: Overall task complexity score (0.0 to 1.0)
        required_capabilities: Mapping of capability dimension names to
            required proficiency levels (0.0 to 1.0)
        cognitive_load: Estimated cognitive load for the task (0.0 to 1.0)
        overall_difficulty: Aggregate difficulty score (0.0 to 1.0)

    Example:
        >>> difficulty = TaskDifficulty(
        ...     complexity=0.6,
        ...     required_capabilities={"reasoning": 0.7, "language": 0.5},
        ...     cognitive_load=0.5,
        ...     overall_difficulty=0.6
        ... )
    """

    complexity: float  # 0.0 to 1.0
    required_capabilities: Dict[str, float]  # dimension -> required level
    cognitive_load: float  # 0.0 to 1.0
    overall_difficulty: float  # 0.0 to 1.0

    def __post_init__(self) -> None:
        """Validate that all values are within expected bounds."""
        self._validate_bounds("complexity", self.complexity, 0.0, 1.0)
        self._validate_bounds("cognitive_load", self.cognitive_load, 0.0, 1.0)
        self._validate_bounds("overall_difficulty", self.overall_difficulty, 0.0, 1.0)

        for dim, level in self.required_capabilities.items():
            self._validate_bounds(f"required_capabilities[{dim}]", level, 0.0, 1.0)

    def _validate_bounds(
        self, field_name: str, value: float, min_val: float, max_val: float
    ) -> None:
        """Validate that a value is within the specified bounds."""
        if not (min_val <= value <= max_val):
            raise ValueError(f"{field_name} must be between {min_val} and {max_val}, got {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "complexity": self.complexity,
            "required_capabilities": self.required_capabilities.copy(),
            "cognitive_load": self.cognitive_load,
            "overall_difficulty": self.overall_difficulty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskDifficulty":
        """Create TaskDifficulty from dictionary representation."""
        return cls(
            complexity=data["complexity"],
            required_capabilities=data.get("required_capabilities", {}),
            cognitive_load=data["cognitive_load"],
            overall_difficulty=data["overall_difficulty"],
        )


@dataclass
class ZPDAssessment:
    """
    Zone of Proximal Development assessment.

    Represents the result of evaluating whether a task falls within the
    agent's Zone of Proximal Development - the range of tasks that are
    challenging enough to promote learning but not so difficult as to
    be impossible.

    Attributes:
        in_zpd: Whether the task is within the Zone of Proximal Development
        difficulty_gap: The gap between task difficulty and agent capability.
            Positive values indicate the task is too hard, negative values
            indicate it's too easy.
        recommendations: List of recommendations for the agent or system
        suggested_scaffolding: Optional list of scaffolding strategies or
            sub-tasks that could help the agent complete the task

    Example:
        >>> assessment = ZPDAssessment(
        ...     in_zpd=True,
        ...     difficulty_gap=0.25,
        ...     recommendations=["Focus on reasoning skills"],
        ...     suggested_scaffolding=["Break into smaller steps"]
        ... )
    """

    in_zpd: bool
    difficulty_gap: float  # positive = too hard, negative = too easy
    recommendations: List[str]
    suggested_scaffolding: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "in_zpd": self.in_zpd,
            "difficulty_gap": self.difficulty_gap,
            "recommendations": self.recommendations.copy(),
            "suggested_scaffolding": (
                self.suggested_scaffolding.copy() if self.suggested_scaffolding else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZPDAssessment":
        """Create ZPDAssessment from dictionary representation."""
        return cls(
            in_zpd=data["in_zpd"],
            difficulty_gap=data["difficulty_gap"],
            recommendations=data.get("recommendations", []),
            suggested_scaffolding=data.get("suggested_scaffolding"),
        )


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning system.

    Contains all configurable parameters for the curriculum learning system,
    including ZPD thresholds, capability growth/decay rates, and failure
    handling settings.

    Attributes:
        zpd_lower_threshold: Minimum difficulty gap for a task to be
            considered within ZPD (default: 0.2)
        zpd_upper_threshold: Maximum difficulty gap for a task to be
            considered within ZPD (default: 0.4)
        capability_growth_rate: Rate at which capabilities grow after
            successful task completion (default: 0.05)
        capability_decay_rate: Rate at which capabilities decay after
            failed task completion (default: 0.02)
        consecutive_failure_threshold: Number of consecutive failures
            before triggering difficulty reduction (default: 3)

    Example:
        >>> config = CurriculumConfig(
        ...     zpd_lower_threshold=0.15,
        ...     zpd_upper_threshold=0.35,
        ...     capability_growth_rate=0.06
        ... )
    """

    zpd_lower_threshold: float = 0.2  # Min difficulty gap for ZPD
    zpd_upper_threshold: float = 0.4  # Max difficulty gap for ZPD
    capability_growth_rate: float = 0.05  # Per successful task
    capability_decay_rate: float = 0.02  # Per failed task
    consecutive_failure_threshold: int = 3  # Failures before difficulty reduction

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.zpd_lower_threshold < 0:
            raise ValueError("zpd_lower_threshold must be non-negative")
        if self.zpd_upper_threshold < self.zpd_lower_threshold:
            raise ValueError("zpd_upper_threshold must be >= zpd_lower_threshold")
        if self.capability_growth_rate < 0:
            raise ValueError("capability_growth_rate must be non-negative")
        if self.capability_decay_rate < 0:
            raise ValueError("capability_decay_rate must be non-negative")
        if self.consecutive_failure_threshold < 1:
            raise ValueError("consecutive_failure_threshold must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "zpd_lower_threshold": self.zpd_lower_threshold,
            "zpd_upper_threshold": self.zpd_upper_threshold,
            "capability_growth_rate": self.capability_growth_rate,
            "capability_decay_rate": self.capability_decay_rate,
            "consecutive_failure_threshold": self.consecutive_failure_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumConfig":
        """Create CurriculumConfig from dictionary representation."""
        return cls(
            zpd_lower_threshold=data.get("zpd_lower_threshold", 0.2),
            zpd_upper_threshold=data.get("zpd_upper_threshold", 0.4),
            capability_growth_rate=data.get("capability_growth_rate", 0.05),
            capability_decay_rate=data.get("capability_decay_rate", 0.02),
            consecutive_failure_threshold=data.get("consecutive_failure_threshold", 3),
        )


@dataclass
class Task:
    """
    Represents a task for curriculum learning.

    A task is a unit of work that the agent can attempt. Tasks have
    associated complexity, required capabilities, and metadata.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type/category of the task (e.g., "reasoning", "language")
        complexity: Base complexity score (0.0 to 1.0)
        required_capabilities: Mapping of capability dimensions to required levels
        metadata: Additional task-specific information

    Example:
        >>> task = Task(
        ...     task_id="task_001",
        ...     task_type="reasoning",
        ...     complexity=0.6,
        ...     required_capabilities={"reasoning": 0.7, "memory": 0.4}
        ... )
    """

    task_id: str
    task_type: str
    complexity: float  # 0.0 to 1.0
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate task parameters."""
        if not (0.0 <= self.complexity <= 1.0):
            raise ValueError(f"complexity must be between 0.0 and 1.0, got {self.complexity}")
        for dim, level in self.required_capabilities.items():
            if not (0.0 <= level <= 1.0):
                raise ValueError(
                    f"required_capabilities[{dim}] must be between 0.0 and 1.0, " f"got {level}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "complexity": self.complexity,
            "required_capabilities": self.required_capabilities.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dictionary representation."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            complexity=data["complexity"],
            required_capabilities=data.get("required_capabilities", {}),
            metadata=data.get("metadata", {}),
        )


# Default capability weights for task type to dimension mapping
TASK_TYPE_CAPABILITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "perception": {"perception": 0.8, "memory": 0.2},
    "reasoning": {"reasoning": 0.7, "memory": 0.2, "planning": 0.1},
    "language": {"language": 0.8, "memory": 0.1, "social": 0.1},
    "social": {"social": 0.6, "language": 0.3, "perception": 0.1},
    "memory": {"memory": 0.8, "perception": 0.1, "reasoning": 0.1},
    "planning": {"planning": 0.6, "reasoning": 0.3, "memory": 0.1},
    "chat_generate": {"language": 0.6, "social": 0.3, "memory": 0.1},
    "search_qa": {"reasoning": 0.4, "language": 0.4, "memory": 0.2},
    "rag_qa": {"memory": 0.4, "reasoning": 0.3, "language": 0.3},
    "lesson_pack": {"language": 0.4, "reasoning": 0.3, "planning": 0.3},
    "summarization": {"language": 0.5, "reasoning": 0.3, "memory": 0.2},
}


class CurriculumLearningSystem:
    """
    Manages curriculum-based developmental training.

    The CurriculumLearningSystem tracks the agent's capability levels across
    multiple dimensions and determines whether tasks are appropriate for the
    agent's current level of development. It implements the Zone of Proximal
    Development (ZPD) concept to select tasks that are challenging but achievable.

    Attributes:
        development_system: Reference to the DevelopmentSystem for stage info
        config: Configuration parameters for the curriculum system
        capability_levels: Current capability levels per dimension
        task_history: History of task attempts for tracking performance

    Requirements: 1.1, 1.2, 1.4, 1.5
    """

    # Default initial capability levels by development stage
    STAGE_INITIAL_CAPABILITIES: Dict[str, float] = {
        "infant": 0.2,
        "child": 0.4,
        "adolescent": 0.6,
        "adult": 0.8,
    }

    def __init__(
        self,
        development_system: Optional["DevelopmentSystem"] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CurriculumLearningSystem.

        Args:
            development_system: Reference to the existing DevelopmentSystem.
                If None, operates in standalone mode with default capabilities.
            config: Optional configuration dictionary. If provided, creates
                a CurriculumConfig from it.
        """
        self._development_system = development_system

        # Initialize configuration
        if config is not None:
            self._config = CurriculumConfig.from_dict(config)
        else:
            self._config = CurriculumConfig()

        # Initialize capability levels for all dimensions
        initial_level = self._get_initial_capability_level()
        self._capability_levels: Dict[str, float] = {
            dim.value: initial_level for dim in CapabilityDimension
        }

        # Track task history for performance analysis
        self._task_history: List[Dict[str, Any]] = []
        self._max_history: int = 1000

        # Track consecutive failures per dimension for difficulty adjustment
        self._consecutive_failures: Dict[str, int] = {dim.value: 0 for dim in CapabilityDimension}

        # Track difficulty thresholds per dimension (can increase with success)
        self._difficulty_thresholds: Dict[str, float] = {
            dim.value: initial_level + self._config.zpd_upper_threshold
            for dim in CapabilityDimension
        }

        self._initialized_at: float = time.time()

    def _get_initial_capability_level(self) -> float:
        """Get initial capability level based on development stage."""
        if self._development_system is not None:
            stage = self._development_system.get_current_stage().value
            return self.STAGE_INITIAL_CAPABILITIES.get(stage, 0.5)
        return 0.5  # Default to middle level

    def estimate_task_difficulty(self, task: Task) -> TaskDifficulty:
        """
        Estimate difficulty of a task across multiple dimensions.

        Calculates a comprehensive difficulty assessment based on:
        - Base complexity of the task
        - Required capabilities and their levels
        - Cognitive load estimation

        Args:
            task: The task to estimate difficulty for.

        Returns:
            TaskDifficulty with scores for complexity, required_capabilities,
            cognitive_load, and overall_difficulty (0.0 to 1.0).

        Validates: Requirements 1.1
        """
        # Handle both curriculum.Task and schemas.Task
        if hasattr(task, "required_capabilities"):
            required_caps = task.required_capabilities.copy()
            complexity = task.complexity
        else:
            # For schemas.Task, infer from task type
            required_caps = {}
            complexity = 0.5  # Default medium complexity

        # If no explicit capabilities, infer from task type
        if not required_caps:
            weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
                task.task_type, {"reasoning": 0.5, "language": 0.5}  # Default weights
            )
            # Scale weights by task complexity
            required_caps = {dim: weight * complexity for dim, weight in weights.items()}

        # Calculate cognitive load based on number of capabilities required
        # and their combined difficulty
        num_capabilities = len(required_caps)
        avg_requirement = (
            sum(required_caps.values()) / num_capabilities if num_capabilities > 0 else complexity
        )

        # Cognitive load increases with more capabilities and higher requirements
        cognitive_load = min(
            1.0, (0.3 * (num_capabilities / len(CapabilityDimension)) + 0.7 * avg_requirement)
        )

        # Overall difficulty is weighted combination
        overall_difficulty = min(
            1.0, max(0.0, (0.4 * complexity + 0.4 * avg_requirement + 0.2 * cognitive_load))
        )

        return TaskDifficulty(
            complexity=complexity,
            required_capabilities=required_caps,
            cognitive_load=cognitive_load,
            overall_difficulty=overall_difficulty,
        )

    def get_capability_level(self, dimension: str) -> float:
        """
        Get current capability level for a dimension.

        Args:
            dimension: The capability dimension name (e.g., "reasoning").

        Returns:
            Current capability level (0.0 to 1.0).

        Raises:
            ValueError: If the dimension is not recognized.

        Validates: Requirements 1.4
        """
        # Normalize dimension name
        dim_lower = dimension.lower()

        # Check if it's a valid dimension
        valid_dimensions = {d.value for d in CapabilityDimension}
        if dim_lower not in valid_dimensions:
            raise ValueError(
                f"Unknown capability dimension: {dimension}. "
                f"Valid dimensions are: {valid_dimensions}"
            )

        return self._capability_levels.get(dim_lower, 0.5)

    def get_all_capability_levels(self) -> Dict[str, float]:
        """
        Get all current capability levels.

        Returns:
            Dictionary mapping dimension names to capability levels.
        """
        return self._capability_levels.copy()

    def is_in_zpd(self, task: Task) -> ZPDAssessment:
        """
        Check if task is in Zone of Proximal Development.

        The ZPD is the range of tasks that are challenging enough to promote
        learning but not so difficult as to be impossible. A task is in the
        ZPD if its difficulty exceeds the agent's capability by an amount
        within the configured thresholds.

        Args:
            task: The task to assess.

        Returns:
            ZPDAssessment with in_zpd boolean, difficulty_gap, and recommendations.

        Validates: Requirements 1.2
        """
        difficulty = self.estimate_task_difficulty(task)

        # Calculate the gap between task difficulty and agent capability
        # For each required capability, find the gap
        gaps: List[float] = []
        dimension_gaps: Dict[str, float] = {}

        for dim, required_level in difficulty.required_capabilities.items():
            current_level = self._capability_levels.get(dim, 0.5)
            gap = required_level - current_level
            gaps.append(gap)
            dimension_gaps[dim] = gap

        # If no specific capabilities, use overall difficulty vs average capability
        if not gaps:
            avg_capability = sum(self._capability_levels.values()) / len(self._capability_levels)
            difficulty_gap = difficulty.overall_difficulty - avg_capability
        else:
            # Use the maximum gap (most challenging dimension)
            difficulty_gap = max(gaps) if gaps else 0.0

        # Determine if in ZPD based on thresholds
        # Task is in ZPD if gap is positive (challenging) but not too large
        in_zpd = (
            self._config.zpd_lower_threshold <= difficulty_gap <= self._config.zpd_upper_threshold
        )

        # Generate recommendations
        recommendations: List[str] = []
        suggested_scaffolding: Optional[List[str]] = None

        if difficulty_gap < self._config.zpd_lower_threshold:
            # Task is too easy
            recommendations.append(
                f"Task difficulty ({difficulty.overall_difficulty:.2f}) is below "
                f"optimal learning zone. Consider a more challenging task."
            )
            if difficulty_gap < 0:
                recommendations.append(
                    "This task is well within current capabilities and may not "
                    "promote significant learning."
                )
        elif difficulty_gap > self._config.zpd_upper_threshold:
            # Task is too hard
            recommendations.append(
                f"Task difficulty exceeds current capabilities by {difficulty_gap:.2f}. "
                f"Consider scaffolding or prerequisite tasks."
            )

            # Identify which dimensions need improvement
            weak_dimensions = [
                dim for dim, gap in dimension_gaps.items() if gap > self._config.zpd_upper_threshold
            ]
            if weak_dimensions:
                recommendations.append(f"Focus on improving: {', '.join(weak_dimensions)}")

            # Suggest scaffolding
            suggested_scaffolding = [
                f"Break task into smaller sub-tasks",
                (
                    f"Practice prerequisite skills in: {', '.join(weak_dimensions)}"
                    if weak_dimensions
                    else "Practice prerequisite skills"
                ),
                "Reduce task complexity before attempting full task",
            ]
        else:
            # Task is in ZPD - optimal for learning
            recommendations.append(
                "Task is in the optimal learning zone (ZPD). "
                "Challenging but achievable with effort."
            )

        return ZPDAssessment(
            in_zpd=in_zpd,
            difficulty_gap=difficulty_gap,
            recommendations=recommendations,
            suggested_scaffolding=suggested_scaffolding,
        )

    def update_capabilities(self, task_type: str, success: bool, score: float) -> Dict[str, float]:
        """
        Update capability estimates after task completion.

        Capabilities grow after successful task completion and decay after
        failures. The magnitude of change is proportional to the configured
        growth/decay rates and the task score.

        Args:
            task_type: The type of task completed.
            success: Whether the task was completed successfully.
            score: Performance score (0.0 to 1.0).

        Returns:
            Dictionary of updated capability levels.

        Validates: Requirements 1.4, 1.5
        """
        # Get the capability dimensions affected by this task type
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(task_type, {"reasoning": 0.5, "language": 0.5})

        updates: Dict[str, float] = {}

        for dim, weight in weights.items():
            if dim not in self._capability_levels:
                continue

            current_level = self._capability_levels[dim]

            if success:
                # Growth on success - proportional to score and weight
                growth = self._config.capability_growth_rate * score * weight
                new_level = min(1.0, current_level + growth)

                # Reset consecutive failures
                self._consecutive_failures[dim] = 0

                # Requirement 1.5: Increase difficulty threshold on success
                # If performing well, raise the bar
                if score >= 0.8:
                    self._difficulty_thresholds[dim] = min(
                        1.0,
                        self._difficulty_thresholds[dim]
                        + self._config.capability_growth_rate * 0.5,
                    )
            else:
                # Decay on failure - proportional to weight
                decay = self._config.capability_decay_rate * weight
                new_level = max(0.0, current_level - decay)

                # Track consecutive failures
                self._consecutive_failures[dim] = self._consecutive_failures.get(dim, 0) + 1

            self._capability_levels[dim] = new_level
            updates[dim] = new_level

        # Record in task history
        self._record_task(task_type, success, score, updates)

        return updates

    def _record_task(
        self, task_type: str, success: bool, score: float, capability_updates: Dict[str, float]
    ) -> None:
        """Record a task attempt in history."""
        record = {
            "task_type": task_type,
            "success": success,
            "score": score,
            "capability_updates": capability_updates.copy(),
            "timestamp": time.time(),
        }

        self._task_history.append(record)

        # Maintain history size limit
        if len(self._task_history) > self._max_history:
            self._task_history = self._task_history[-self._max_history :]

    def get_recommended_difficulty(self, capability_dimension: str) -> float:
        """
        Get recommended task difficulty for optimal learning.

        Returns a difficulty level that falls within the ZPD for the
        specified capability dimension.

        Args:
            capability_dimension: The capability dimension to get recommendation for.

        Returns:
            Recommended difficulty level (0.0 to 1.0).

        Validates: Requirements 1.2
        """
        dim_lower = capability_dimension.lower()
        current_level = self._capability_levels.get(dim_lower, 0.5)

        # Optimal difficulty is current level plus middle of ZPD range
        zpd_middle = (self._config.zpd_lower_threshold + self._config.zpd_upper_threshold) / 2

        recommended = current_level + zpd_middle

        # Clamp to valid range
        return min(1.0, max(0.0, recommended))

    def get_consecutive_failures(self, dimension: str) -> int:
        """
        Get the number of consecutive failures for a dimension.

        Args:
            dimension: The capability dimension.

        Returns:
            Number of consecutive failures.
        """
        return self._consecutive_failures.get(dimension.lower(), 0)

    def get_difficulty_threshold(self, dimension: str) -> float:
        """
        Get the current difficulty threshold for a dimension.

        Args:
            dimension: The capability dimension.

        Returns:
            Current difficulty threshold.
        """
        return self._difficulty_thresholds.get(
            dimension.lower(), 0.5 + self._config.zpd_upper_threshold
        )

    def get_config(self) -> CurriculumConfig:
        """Get the current configuration."""
        return self._config

    def get_task_history(
        self, limit: Optional[int] = None, task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get task history records.

        Args:
            limit: Maximum number of records to return.
            task_type: Optional filter by task type.

        Returns:
            List of task history records.
        """
        records = self._task_history

        if task_type:
            records = [r for r in records if r["task_type"] == task_type]

        if limit:
            records = records[-limit:]

        return [r.copy() for r in records]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance across all task types.

        Returns:
            Dictionary with performance statistics.
        """
        if not self._task_history:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "by_task_type": {},
            }

        total = len(self._task_history)
        successes = sum(1 for r in self._task_history if r["success"])
        total_score = sum(r["score"] for r in self._task_history)

        # Group by task type
        by_type: Dict[str, Dict[str, Any]] = {}
        for record in self._task_history:
            tt = record["task_type"]
            if tt not in by_type:
                by_type[tt] = {"count": 0, "successes": 0, "total_score": 0.0}
            by_type[tt]["count"] += 1
            if record["success"]:
                by_type[tt]["successes"] += 1
            by_type[tt]["total_score"] += record["score"]

        # Calculate rates
        for tt, stats in by_type.items():
            stats["success_rate"] = stats["successes"] / stats["count"]
            stats["average_score"] = stats["total_score"] / stats["count"]

        return {
            "total_tasks": total,
            "success_rate": successes / total,
            "average_score": total_score / total,
            "by_task_type": by_type,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the curriculum learning system to a dictionary.

        Returns:
            Dictionary representation of the system state.
        """
        return {
            "config": self._config.to_dict(),
            "capability_levels": self._capability_levels.copy(),
            "consecutive_failures": self._consecutive_failures.copy(),
            "difficulty_thresholds": self._difficulty_thresholds.copy(),
            "task_history": [r.copy() for r in self._task_history],
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the curriculum learning system from a dictionary.

        Args:
            data: Dictionary representation of the system state.
        """
        if "config" in data:
            self._config = CurriculumConfig.from_dict(data["config"])

        if "capability_levels" in data:
            self._capability_levels = data["capability_levels"].copy()

        if "consecutive_failures" in data:
            self._consecutive_failures = data["consecutive_failures"].copy()

        if "difficulty_thresholds" in data:
            self._difficulty_thresholds = data["difficulty_thresholds"].copy()

        if "task_history" in data:
            self._task_history = [r.copy() for r in data["task_history"]]

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]

    def suggest_scaffolding(self, task: Task) -> List[Task]:
        """
        Suggest decomposed sub-tasks for a too-difficult task.

        When a task's difficulty exceeds the agent's capability by more than
        the configured threshold, this method decomposes it into simpler
        sub-tasks that target specific capability dimensions at reduced
        complexity levels.

        Args:
            task: The task that is too difficult.

        Returns:
            List of simpler sub-tasks that scaffold toward the original task.
            Returns an empty list if the task is not too difficult.

        Validates: Requirements 1.3
        """
        # First assess if scaffolding is needed
        zpd_assessment = self.is_in_zpd(task)

        # If task is in ZPD or too easy, no scaffolding needed
        if (
            zpd_assessment.in_zpd
            or zpd_assessment.difficulty_gap < self._config.zpd_lower_threshold
        ):
            return []

        # Task is too difficult - create scaffolded sub-tasks
        difficulty = self.estimate_task_difficulty(task)
        sub_tasks: List[Task] = []

        # Identify dimensions where the gap is too large
        dimension_gaps: Dict[str, float] = {}
        for dim, required_level in difficulty.required_capabilities.items():
            current_level = self._capability_levels.get(dim, 0.5)
            gap = required_level - current_level
            if gap > self._config.zpd_upper_threshold:
                dimension_gaps[dim] = gap

        # If no specific dimension gaps, use overall difficulty
        if not dimension_gaps:
            # Create a single simplified version of the task
            reduced_complexity = max(0.1, task.complexity * 0.6)
            sub_tasks.append(
                Task(
                    task_id=f"{task.task_id}_scaffold_1",
                    task_type=task.task_type,
                    complexity=reduced_complexity,
                    required_capabilities={
                        dim: max(0.1, level * 0.6)
                        for dim, level in task.required_capabilities.items()
                    },
                    metadata={
                        "parent_task_id": task.task_id,
                        "scaffold_type": "complexity_reduction",
                        "scaffold_level": 1,
                    },
                )
            )
            return sub_tasks

        # Create sub-tasks for each dimension that needs improvement
        scaffold_level = 1
        for dim, gap in sorted(dimension_gaps.items(), key=lambda x: x[1], reverse=True):
            current_level = self._capability_levels.get(dim, 0.5)

            # Target difficulty within ZPD for this dimension
            target_level = (
                current_level
                + (self._config.zpd_lower_threshold + self._config.zpd_upper_threshold) / 2
            )

            # Create a focused sub-task for this dimension
            sub_task = Task(
                task_id=f"{task.task_id}_scaffold_{scaffold_level}",
                task_type=dim,  # Focus on the specific dimension
                complexity=min(0.9, target_level),
                required_capabilities={
                    dim: target_level,
                },
                metadata={
                    "parent_task_id": task.task_id,
                    "scaffold_type": "dimension_focus",
                    "target_dimension": dim,
                    "scaffold_level": scaffold_level,
                    "original_gap": gap,
                },
            )
            sub_tasks.append(sub_task)
            scaffold_level += 1

        # Add a final integration sub-task at reduced complexity
        if len(sub_tasks) > 1:
            integration_complexity = max(0.2, task.complexity * 0.7)
            integration_task = Task(
                task_id=f"{task.task_id}_scaffold_{scaffold_level}",
                task_type=task.task_type,
                complexity=integration_complexity,
                required_capabilities={
                    dim: max(0.1, level * 0.7) for dim, level in task.required_capabilities.items()
                },
                metadata={
                    "parent_task_id": task.task_id,
                    "scaffold_type": "integration",
                    "scaffold_level": scaffold_level,
                },
            )
            sub_tasks.append(integration_task)

        return sub_tasks

    def check_consecutive_failures(self, dimension: str) -> bool:
        """
        Check if consecutive failures exceed the configured threshold.

        This method determines whether the agent has failed enough consecutive
        tasks in a dimension to trigger difficulty reduction and remedial
        task suggestions.

        Args:
            dimension: The capability dimension to check.

        Returns:
            True if consecutive failures exceed the threshold, False otherwise.

        Validates: Requirements 1.6
        """
        dim_lower = dimension.lower()

        # Validate dimension
        valid_dimensions = {d.value for d in CapabilityDimension}
        if dim_lower not in valid_dimensions:
            raise ValueError(
                f"Unknown capability dimension: {dimension}. "
                f"Valid dimensions are: {valid_dimensions}"
            )

        failures = self._consecutive_failures.get(dim_lower, 0)
        return failures >= self._config.consecutive_failure_threshold

    def reduce_difficulty_threshold(self, dimension: str) -> float:
        """
        Reduce difficulty threshold after consecutive failures.

        When the agent fails multiple consecutive tasks at a difficulty level,
        this method reduces the difficulty threshold for that dimension to
        allow easier tasks to be selected.

        Args:
            dimension: The capability dimension to reduce threshold for.

        Returns:
            The new (reduced) difficulty threshold.

        Validates: Requirements 1.6
        """
        dim_lower = dimension.lower()

        # Validate dimension
        valid_dimensions = {d.value for d in CapabilityDimension}
        if dim_lower not in valid_dimensions:
            raise ValueError(
                f"Unknown capability dimension: {dimension}. "
                f"Valid dimensions are: {valid_dimensions}"
            )

        current_threshold = self._difficulty_thresholds.get(
            dim_lower, 0.5 + self._config.zpd_upper_threshold
        )

        # Reduce threshold by the decay rate
        # The reduction is proportional to the number of consecutive failures
        failures = self._consecutive_failures.get(dim_lower, 0)
        reduction_factor = min(0.3, self._config.capability_decay_rate * failures)

        new_threshold = max(0.1, current_threshold - reduction_factor)  # Minimum threshold

        self._difficulty_thresholds[dim_lower] = new_threshold

        return new_threshold

    def suggest_remedial_tasks(self, dimension: str) -> List[Task]:
        """
        Suggest easier tasks for remediation after consecutive failures.

        When the agent has failed multiple consecutive tasks in a dimension,
        this method suggests remedial tasks at a lower difficulty level to
        help rebuild capability and confidence.

        Args:
            dimension: The capability dimension needing remediation.

        Returns:
            List of remedial tasks at reduced difficulty.

        Validates: Requirements 1.6
        """
        dim_lower = dimension.lower()

        # Validate dimension
        valid_dimensions = {d.value for d in CapabilityDimension}
        if dim_lower not in valid_dimensions:
            raise ValueError(
                f"Unknown capability dimension: {dimension}. "
                f"Valid dimensions are: {valid_dimensions}"
            )

        remedial_tasks: List[Task] = []
        current_level = self._capability_levels.get(dim_lower, 0.5)

        # Calculate remedial difficulty levels
        # Start well below current level and gradually increase
        base_difficulty = max(0.1, current_level - 0.2)

        # Create 3 remedial tasks at increasing difficulty
        for i in range(3):
            # Each task is slightly harder than the previous
            task_difficulty = min(
                current_level, base_difficulty + (i * 0.1)  # Don't exceed current level
            )

            remedial_task = Task(
                task_id=f"remedial_{dim_lower}_{i + 1}",
                task_type=dim_lower,
                complexity=task_difficulty,
                required_capabilities={
                    dim_lower: task_difficulty,
                },
                metadata={
                    "remedial": True,
                    "target_dimension": dim_lower,
                    "remedial_level": i + 1,
                    "purpose": "rebuild_capability",
                },
            )
            remedial_tasks.append(remedial_task)

        return remedial_tasks

    def handle_task_failure(self, task: Task, score: float = 0.0) -> Dict[str, Any]:
        """
        Handle a task failure with automatic difficulty adjustment.

        This method combines consecutive failure checking, difficulty threshold
        reduction, and remedial task suggestion into a single workflow.

        Args:
            task: The task that was failed.
            score: The score achieved (0.0 to 1.0).

        Returns:
            Dictionary containing:
            - capability_updates: Updated capability levels
            - dimensions_needing_remediation: Dimensions that exceeded failure threshold
            - reduced_thresholds: New thresholds for affected dimensions
            - remedial_tasks: Suggested remedial tasks

        Validates: Requirements 1.6
        """
        # Update capabilities (this also tracks consecutive failures)
        capability_updates = self.update_capabilities(task.task_type, success=False, score=score)

        # Check which dimensions need remediation
        dimensions_needing_remediation: List[str] = []
        reduced_thresholds: Dict[str, float] = {}
        all_remedial_tasks: List[Task] = []

        # Get dimensions affected by this task type
        weights = TASK_TYPE_CAPABILITY_WEIGHTS.get(
            task.task_type, {"reasoning": 0.5, "language": 0.5}
        )

        for dim in weights.keys():
            if self.check_consecutive_failures(dim):
                dimensions_needing_remediation.append(dim)

                # Reduce difficulty threshold
                new_threshold = self.reduce_difficulty_threshold(dim)
                reduced_thresholds[dim] = new_threshold

                # Get remedial tasks
                remedial_tasks = self.suggest_remedial_tasks(dim)
                all_remedial_tasks.extend(remedial_tasks)

        return {
            "capability_updates": capability_updates,
            "dimensions_needing_remediation": dimensions_needing_remediation,
            "reduced_thresholds": reduced_thresholds,
            "remedial_tasks": all_remedial_tasks,
        }

    def compose_learning_batch(
        self,
        new_experiences: List[Any],
        replay_buffer: Optional[Any] = None,
        replay_ratio: float = 0.3,
        replay_strategy: str = "prioritized",
    ) -> List[Any]:
        """
        Compose a learning batch mixing new and replayed experiences.

        For continuous learning without catastrophic forgetting, this method
        creates batches that contain both new experiences and replayed past
        experiences. The replay ratio controls the proportion of replayed
        experiences in the batch.

        Args:
            new_experiences: List of new experiences to learn from.
            replay_buffer: Optional ExperienceReplayBuffer to sample from.
                If None or empty, returns only new experiences.
            replay_ratio: Proportion of batch that should be replayed experiences
                (0.0 to 1.0). Default is 0.3 (30% replay).
            replay_strategy: Strategy for sampling replayed experiences
                ("uniform", "prioritized", or "stratified").

        Returns:
            Combined list of new and replayed experiences.

        Validates: Requirements 9.2
        """
        # Validate replay_ratio
        if not (0.0 <= replay_ratio <= 1.0):
            raise ValueError(f"replay_ratio must be between 0.0 and 1.0, got {replay_ratio}")

        # If no replay buffer or it's empty, return only new experiences
        if replay_buffer is None or len(replay_buffer) == 0:
            return new_experiences.copy()

        # If no new experiences, return empty (nothing to learn)
        if not new_experiences:
            return []

        # Calculate batch composition
        total_new = len(new_experiences)

        # Calculate number of replayed experiences to include
        # replay_ratio is the proportion of the TOTAL batch that should be replay
        # So if we have N new experiences and want R% replay:
        # total_batch = new + replay
        # replay / total_batch = R
        # replay = R * total_batch = R * (new + replay)
        # replay = R * new + R * replay
        # replay * (1 - R) = R * new
        # replay = (R * new) / (1 - R)

        if replay_ratio >= 1.0:
            # Edge case: 100% replay means no new experiences (shouldn't happen)
            num_replay = total_new
        elif replay_ratio == 0.0:
            # No replay requested
            num_replay = 0
        else:
            num_replay = max(1, int((replay_ratio * total_new) / (1.0 - replay_ratio)))

        # Ensure we don't request more than available in buffer
        num_replay = min(num_replay, len(replay_buffer))

        # Sample replayed experiences
        if num_replay > 0:
            replayed_experiences = replay_buffer.sample(
                batch_size=num_replay, strategy=replay_strategy
            )
        else:
            replayed_experiences = []

        # Combine new and replayed experiences
        # Shuffle to mix them together
        combined = new_experiences.copy() + replayed_experiences
        random.shuffle(combined)

        return combined

    def detect_performance_degradation(
        self, task_type: str, window_size: int = 20, degradation_threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Detect performance degradation on a previously mastered task type.

        Compares recent performance on a task type to historical baseline
        performance to detect if the agent has "forgotten" how to perform
        the task well (catastrophic forgetting).

        Args:
            task_type: The task type to check for degradation.
            window_size: Number of recent tasks to compare (default: 20).
            degradation_threshold: Minimum success rate drop to trigger
                degradation detection (default: 0.15 = 15% drop).

        Returns:
            Dictionary containing:
            - degraded: Boolean indicating if degradation detected
            - baseline_success_rate: Historical success rate
            - recent_success_rate: Recent success rate
            - success_rate_drop: Absolute drop in success rate
            - baseline_avg_score: Historical average score
            - recent_avg_score: Recent average score
            - score_drop: Absolute drop in average score
            - recommendation: Suggested action if degraded

        Validates: Requirements 9.4
        """
        # Get task history for this task type
        task_records = [r for r in self._task_history if r["task_type"] == task_type]

        if len(task_records) < window_size * 2:
            # Not enough history to detect degradation
            return {
                "degraded": False,
                "baseline_success_rate": 0.0,
                "recent_success_rate": 0.0,
                "success_rate_drop": 0.0,
                "baseline_avg_score": 0.0,
                "recent_avg_score": 0.0,
                "score_drop": 0.0,
                "recommendation": "Insufficient history for degradation detection",
            }

        # Split into baseline (older) and recent windows
        # Baseline: everything except the most recent window
        # Recent: the most recent window_size tasks
        recent_records = task_records[-window_size:]
        baseline_records = task_records[:-window_size]

        # Calculate baseline performance
        baseline_successes = sum(1 for r in baseline_records if r["success"])
        baseline_success_rate = baseline_successes / len(baseline_records)
        baseline_total_score = sum(r["score"] for r in baseline_records)
        baseline_avg_score = baseline_total_score / len(baseline_records)

        # Calculate recent performance
        recent_successes = sum(1 for r in recent_records if r["success"])
        recent_success_rate = recent_successes / len(recent_records)
        recent_total_score = sum(r["score"] for r in recent_records)
        recent_avg_score = recent_total_score / len(recent_records)

        # Calculate drops
        success_rate_drop = baseline_success_rate - recent_success_rate
        score_drop = baseline_avg_score - recent_avg_score

        # Detect degradation
        # Degradation is detected if:
        # 1. Success rate has dropped by more than threshold
        # OR
        # 2. Average score has dropped significantly (more lenient check)
        degraded = success_rate_drop >= degradation_threshold or (
            success_rate_drop >= degradation_threshold * 0.7 and score_drop > 0.03
        )

        recommendation = ""
        if degraded:
            recommendation = (
                f"Performance degradation detected for {task_type}. "
                f"Trigger remedial replay with prioritized sampling of "
                f"{task_type} experiences."
            )
        else:
            recommendation = "No significant performance degradation detected."

        return {
            "degraded": degraded,
            "baseline_success_rate": baseline_success_rate,
            "recent_success_rate": recent_success_rate,
            "success_rate_drop": success_rate_drop,
            "baseline_avg_score": baseline_avg_score,
            "recent_avg_score": recent_avg_score,
            "score_drop": score_drop,
            "recommendation": recommendation,
        }

    def trigger_remedial_replay(
        self, task_type: str, replay_buffer: Any, batch_size: int = 10
    ) -> List[Any]:
        """
        Trigger remedial replay for a task type showing degradation.

        When performance degradation is detected, this method samples
        experiences of the degraded task type from the replay buffer
        for remedial learning.

        Args:
            task_type: The task type showing degradation.
            replay_buffer: ExperienceReplayBuffer to sample from.
            batch_size: Number of experiences to sample for remediation.

        Returns:
            List of experiences for remedial learning.

        Validates: Requirements 9.4
        """
        if replay_buffer is None or len(replay_buffer) == 0:
            return []

        # Get experiences of the specified task type
        task_experiences = replay_buffer.get_experiences_by_task_type(task_type)

        if not task_experiences:
            # No experiences of this type in buffer
            return []

        # Sample from task-specific experiences
        # Prioritize high-value experiences (errors, edge cases)
        actual_batch_size = min(batch_size, len(task_experiences))

        # Sort by priority and sample top experiences
        sorted_experiences = sorted(task_experiences, key=lambda e: e.priority, reverse=True)

        return sorted_experiences[:actual_batch_size]

    def monitor_continuous_learning(
        self, replay_buffer: Any, check_interval: int = 50
    ) -> Dict[str, Any]:
        """
        Monitor all task types for performance degradation.

        Periodically checks all task types that have sufficient history
        for signs of catastrophic forgetting and recommends remedial
        replay when needed.

        Args:
            replay_buffer: ExperienceReplayBuffer for remedial replay.
            check_interval: Check degradation every N tasks (default: 50).

        Returns:
            Dictionary containing:
            - total_tasks_checked: Number of task types checked
            - degraded_tasks: List of task types showing degradation
            - degradation_details: Detailed degradation info per task type
            - remedial_batches: Recommended remedial replay batches

        Validates: Requirements 9.4
        """
        # Check if it's time to monitor (based on total task count)
        total_tasks = len(self._task_history)
        if total_tasks % check_interval != 0:
            # Not time to check yet
            return {
                "total_tasks_checked": 0,
                "degraded_tasks": [],
                "degradation_details": {},
                "remedial_batches": {},
            }

        # Get all unique task types from history
        task_types = set(r["task_type"] for r in self._task_history)

        degraded_tasks: List[str] = []
        degradation_details: Dict[str, Dict[str, Any]] = {}
        remedial_batches: Dict[str, List[Any]] = {}

        # Check each task type for degradation
        for task_type in task_types:
            degradation_info = self.detect_performance_degradation(task_type)

            if degradation_info["degraded"]:
                degraded_tasks.append(task_type)
                degradation_details[task_type] = degradation_info

                # Trigger remedial replay
                if replay_buffer is not None:
                    remedial_batch = self.trigger_remedial_replay(
                        task_type, replay_buffer, batch_size=10
                    )
                    if remedial_batch:
                        remedial_batches[task_type] = remedial_batch

        return {
            "total_tasks_checked": len(task_types),
            "degraded_tasks": degraded_tasks,
            "degradation_details": degradation_details,
            "remedial_batches": remedial_batches,
        }
