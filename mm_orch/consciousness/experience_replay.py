"""
Experience Replay Buffer module for consciousness system.

This module implements the ExperienceReplayBuffer which manages experience storage
and replay for continuous learning. It supports multiple sampling strategies
(uniform, prioritized, stratified) and importance-weighted pruning to prevent
catastrophic forgetting.

Requirements: 9.1, 9.3, 9.5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random
import time
import uuid


@dataclass
class Experience:
    """
    Experience for replay buffer.

    Represents a single experience that can be stored and replayed for
    continuous learning. Each experience captures the context, action taken,
    outcome, and associated reward/priority information.

    Attributes:
        experience_id: Unique identifier for the experience.
        task_type: Type of task this experience relates to.
        context: Contextual information at the time of the experience.
        action: The action that was taken.
        outcome: The outcome/result of the action.
        reward: The reward received (can be positive or negative).
        priority: Priority score for sampling (higher = more likely to sample).
        timestamp: When the experience occurred.
        metadata: Additional metadata about the experience.
    """

    experience_id: str
    task_type: str
    context: Dict[str, Any]
    action: str
    outcome: Dict[str, Any]
    reward: float
    priority: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate experience fields."""
        if not self.experience_id:
            raise ValueError("experience_id cannot be empty")
        if not self.task_type:
            raise ValueError("task_type cannot be empty")
        if not isinstance(self.context, dict):
            raise ValueError("context must be a dictionary")
        if not isinstance(self.action, str):
            raise ValueError("action must be a string")
        if not isinstance(self.outcome, dict):
            raise ValueError("outcome must be a dictionary")
        if not isinstance(self.reward, (int, float)):
            raise ValueError("reward must be a number")
        if not isinstance(self.priority, (int, float)):
            raise ValueError("priority must be a number")
        # Clamp priority to valid range
        self.priority = max(0.0, min(1.0, float(self.priority)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert experience to dictionary representation."""
        return {
            "experience_id": self.experience_id,
            "task_type": self.task_type,
            "context": self.context.copy(),
            "action": self.action,
            "outcome": self.outcome.copy(),
            "reward": self.reward,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        """Create experience from dictionary representation."""
        return cls(
            experience_id=data["experience_id"],
            task_type=data["task_type"],
            context=data.get("context", {}),
            action=data["action"],
            outcome=data.get("outcome", {}),
            reward=data.get("reward", 0.0),
            priority=data.get("priority", 0.5),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperienceReplayConfig:
    """Configuration for the experience replay buffer."""

    max_size: int = 10000
    default_priority: float = 0.5
    priority_alpha: float = 0.6  # How much prioritization to use (0 = uniform, 1 = full priority)
    min_priority: float = 0.01  # Minimum priority to prevent zero probability
    prune_threshold: float = 0.9  # Prune when buffer reaches this fraction of max_size
    prune_ratio: float = 0.2  # Remove this fraction of experiences when pruning

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_size < 1:
            raise ValueError("max_size must be at least 1")
        if not (0.0 <= self.default_priority <= 1.0):
            raise ValueError("default_priority must be between 0.0 and 1.0")
        if not (0.0 <= self.priority_alpha <= 1.0):
            raise ValueError("priority_alpha must be between 0.0 and 1.0")
        if not (0.0 < self.min_priority <= 1.0):
            raise ValueError("min_priority must be between 0.0 (exclusive) and 1.0")
        if not (0.0 < self.prune_threshold <= 1.0):
            raise ValueError("prune_threshold must be between 0.0 (exclusive) and 1.0")
        if not (0.0 < self.prune_ratio < 1.0):
            raise ValueError("prune_ratio must be between 0.0 and 1.0 (exclusive)")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "max_size": self.max_size,
            "default_priority": self.default_priority,
            "priority_alpha": self.priority_alpha,
            "min_priority": self.min_priority,
            "prune_threshold": self.prune_threshold,
            "prune_ratio": self.prune_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceReplayConfig":
        """Create config from dictionary representation."""
        return cls(
            max_size=data.get("max_size", 10000),
            default_priority=data.get("default_priority", 0.5),
            priority_alpha=data.get("priority_alpha", 0.6),
            min_priority=data.get("min_priority", 0.01),
            prune_threshold=data.get("prune_threshold", 0.9),
            prune_ratio=data.get("prune_ratio", 0.2),
        )


class ExperienceReplayBuffer:
    """
    Manages experience storage and replay for continuous learning.

    This buffer stores experiences and supports multiple sampling strategies:
    - uniform: All experiences have equal probability of being sampled
    - prioritized: Higher priority experiences are more likely to be sampled
    - stratified: Samples are drawn proportionally from each task type

    The buffer also implements importance-weighted pruning to maintain size
    limits while retaining the most valuable experiences.

    Requirements: 9.1, 9.3, 9.5
    """

    def __init__(self, max_size: int = 10000, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experience replay buffer.

        Args:
            max_size: Maximum number of experiences to store.
            config: Optional configuration dictionary.
        """
        if config is not None:
            self._config = ExperienceReplayConfig.from_dict(config)
            # Override max_size if provided in config
            if "max_size" not in config:
                self._config.max_size = max_size
        else:
            self._config = ExperienceReplayConfig(max_size=max_size)

        # Main storage: experience_id -> Experience
        self._experiences: Dict[str, Experience] = {}

        # Index by task type for stratified sampling
        self._task_type_index: Dict[str, List[str]] = {}

        # Statistics
        self._total_stored: int = 0
        self._total_sampled: int = 0
        self._total_pruned: int = 0
        self._initialized_at: float = time.time()

    def _generate_experience_id(self) -> str:
        """Generate a unique experience ID."""
        return str(uuid.uuid4())

    def store(self, experience: Experience) -> str:
        """
        Store an experience and return its ID.

        If the experience already has an ID, it will be used. Otherwise,
        a new ID will be generated. If the buffer is at capacity, pruning
        will be triggered automatically.

        Args:
            experience: The experience to store.

        Returns:
            The experience ID.

        Validates: Requirements 9.1
        """
        # Check if we need to prune before adding
        if len(self._experiences) >= self._config.max_size:
            self.prune()

        # Store the experience
        exp_id = experience.experience_id
        self._experiences[exp_id] = experience

        # Update task type index
        task_type = experience.task_type
        if task_type not in self._task_type_index:
            self._task_type_index[task_type] = []
        if exp_id not in self._task_type_index[task_type]:
            self._task_type_index[task_type].append(exp_id)

        self._total_stored += 1
        return exp_id

    def sample(self, batch_size: int, strategy: str = "prioritized") -> List[Experience]:
        """
        Sample experiences for replay.

        Args:
            batch_size: Number of experiences to sample.
            strategy: Sampling strategy - "uniform", "prioritized", or "stratified".

        Returns:
            List of sampled experiences.

        Raises:
            ValueError: If strategy is invalid or batch_size is invalid.

        Validates: Requirements 9.1, 9.3
        """
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")

        if batch_size == 0 or len(self._experiences) == 0:
            return []

        valid_strategies = {"uniform", "prioritized", "stratified"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")

        # Limit batch size to available experiences
        actual_batch_size = min(batch_size, len(self._experiences))

        if strategy == "uniform":
            sampled = self._sample_uniform(actual_batch_size)
        elif strategy == "prioritized":
            sampled = self._sample_prioritized(actual_batch_size)
        else:  # stratified
            sampled = self._sample_stratified(actual_batch_size)

        self._total_sampled += len(sampled)
        return sampled

    def _sample_uniform(self, batch_size: int) -> List[Experience]:
        """Sample experiences uniformly at random."""
        exp_ids = list(self._experiences.keys())
        sampled_ids = random.sample(exp_ids, batch_size)
        return [self._experiences[exp_id] for exp_id in sampled_ids]

    def _sample_prioritized(self, batch_size: int) -> List[Experience]:
        """
        Sample experiences with probability proportional to priority.

        Uses priority^alpha as the sampling weight, where alpha controls
        how much prioritization is used (0 = uniform, 1 = full priority).

        Validates: Requirements 9.3
        """
        exp_list = list(self._experiences.values())

        # Calculate sampling weights based on priority
        weights = []
        for exp in exp_list:
            # Ensure minimum priority to prevent zero probability
            priority = max(self._config.min_priority, exp.priority)
            # Apply alpha exponent for prioritization control
            weight = priority**self._config.priority_alpha
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to uniform if all weights are zero
            return self._sample_uniform(batch_size)

        probabilities = [w / total_weight for w in weights]

        # Sample without replacement using weighted selection
        sampled = []
        available_indices = list(range(len(exp_list)))
        available_probs = probabilities.copy()

        for _ in range(batch_size):
            if not available_indices:
                break

            # Normalize remaining probabilities
            total_prob = sum(available_probs)
            if total_prob == 0:
                # Fallback to uniform for remaining samples
                idx = random.choice(available_indices)
            else:
                normalized_probs = [p / total_prob for p in available_probs]
                idx = random.choices(range(len(available_indices)), weights=normalized_probs, k=1)[
                    0
                ]

            sampled.append(exp_list[available_indices[idx]])

            # Remove selected index
            available_indices.pop(idx)
            available_probs.pop(idx)

        return sampled

    def _sample_stratified(self, batch_size: int) -> List[Experience]:
        """
        Sample experiences proportionally from each task type.

        Ensures representation from all task types in the buffer.

        Validates: Requirements 9.1
        """
        if not self._task_type_index:
            return []

        # Calculate how many samples per task type
        task_types = list(self._task_type_index.keys())
        num_task_types = len(task_types)

        # Calculate proportional allocation based on task type counts
        total_experiences = len(self._experiences)
        samples_per_type: Dict[str, int] = {}

        # First pass: ensure at least 1 sample from each type if batch_size >= num_task_types
        if batch_size >= num_task_types:
            # Guarantee at least 1 from each type
            for task_type in task_types:
                samples_per_type[task_type] = 1
            remaining = batch_size - num_task_types

            # Distribute remaining samples proportionally
            if remaining > 0:
                for task_type in task_types:
                    type_count = len(self._task_type_index[task_type])
                    proportion = type_count / total_experiences
                    additional = int(remaining * proportion)
                    # Don't exceed available experiences for this type
                    max_additional = type_count - samples_per_type[task_type]
                    samples_per_type[task_type] += min(additional, max_additional)

                # Distribute any leftover samples
                current_total = sum(samples_per_type.values())
                leftover = batch_size - current_total
                while leftover > 0:
                    for task_type in task_types:
                        if leftover <= 0:
                            break
                        type_count = len(self._task_type_index[task_type])
                        if samples_per_type[task_type] < type_count:
                            samples_per_type[task_type] += 1
                            leftover -= 1
        else:
            # batch_size < num_task_types: can't guarantee all types
            # Allocate proportionally
            remaining = batch_size
            for task_type in task_types:
                if remaining <= 0:
                    samples_per_type[task_type] = 0
                    continue
                type_count = len(self._task_type_index[task_type])
                proportion = type_count / total_experiences
                allocation = max(1, int(batch_size * proportion))
                allocation = min(allocation, type_count, remaining)
                samples_per_type[task_type] = allocation
                remaining -= allocation

        # Sample from each task type
        sampled = []
        for task_type, num_samples in samples_per_type.items():
            if num_samples == 0:
                continue
            exp_ids = self._task_type_index[task_type]
            sampled_ids = random.sample(exp_ids, min(num_samples, len(exp_ids)))
            sampled.extend([self._experiences[exp_id] for exp_id in sampled_ids])

        return sampled

    def update_priority(self, experience_id: str, priority: float) -> None:
        """
        Update priority of an experience.

        Args:
            experience_id: ID of the experience to update.
            priority: New priority value (will be clamped to [0.0, 1.0]).

        Raises:
            KeyError: If experience_id is not found.
        """
        if experience_id not in self._experiences:
            raise KeyError(f"Experience '{experience_id}' not found in buffer")

        # Clamp priority to valid range
        clamped_priority = max(0.0, min(1.0, float(priority)))
        self._experiences[experience_id].priority = clamped_priority

    def get_task_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of experiences by task type.

        Returns:
            Dictionary mapping task type to count of experiences.

        Validates: Requirements 9.1
        """
        return {task_type: len(exp_ids) for task_type, exp_ids in self._task_type_index.items()}

    def prune(self) -> int:
        """
        Remove low-priority experiences to maintain size limit.

        Uses importance-weighted pruning to retain the most valuable
        experiences. Experiences with lower priority are more likely
        to be removed.

        Returns:
            Number of experiences removed.

        Validates: Requirements 9.5
        """
        current_size = len(self._experiences)
        if current_size == 0:
            return 0

        # Calculate target size after pruning
        target_size = int(self._config.max_size * (1 - self._config.prune_ratio))
        num_to_remove = max(0, current_size - target_size)

        if num_to_remove == 0:
            return 0

        # Sort experiences by priority (ascending) - lowest priority first
        sorted_experiences = sorted(
            self._experiences.values(),
            key=lambda e: (e.priority, e.timestamp),  # Secondary sort by timestamp (older first)
        )

        # Remove lowest priority experiences
        removed_count = 0
        for exp in sorted_experiences[:num_to_remove]:
            self._remove_experience(exp.experience_id)
            removed_count += 1

        self._total_pruned += removed_count
        return removed_count

    def _remove_experience(self, experience_id: str) -> None:
        """Remove an experience from the buffer and all indices."""
        if experience_id not in self._experiences:
            return

        exp = self._experiences[experience_id]
        task_type = exp.task_type

        # Remove from main storage
        del self._experiences[experience_id]

        # Remove from task type index
        if task_type in self._task_type_index:
            if experience_id in self._task_type_index[task_type]:
                self._task_type_index[task_type].remove(experience_id)
            # Clean up empty task type entries
            if not self._task_type_index[task_type]:
                del self._task_type_index[task_type]

    def get_experience(self, experience_id: str) -> Optional[Experience]:
        """
        Get an experience by ID.

        Args:
            experience_id: ID of the experience to retrieve.

        Returns:
            The experience if found, None otherwise.
        """
        return self._experiences.get(experience_id)

    def get_experiences_by_task_type(self, task_type: str) -> List[Experience]:
        """
        Get all experiences of a specific task type.

        Args:
            task_type: The task type to filter by.

        Returns:
            List of experiences with the specified task type.
        """
        if task_type not in self._task_type_index:
            return []
        return [
            self._experiences[exp_id]
            for exp_id in self._task_type_index[task_type]
            if exp_id in self._experiences
        ]

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self._experiences.clear()
        self._task_type_index.clear()

    def __len__(self) -> int:
        """Return the number of experiences in the buffer."""
        return len(self._experiences)

    def __contains__(self, experience_id: str) -> bool:
        """Check if an experience ID is in the buffer."""
        return experience_id in self._experiences

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the experience replay buffer."""
        return {
            "size": len(self._experiences),
            "max_size": self._config.max_size,
            "task_types": list(self._task_type_index.keys()),
            "task_type_distribution": self.get_task_type_distribution(),
            "total_stored": self._total_stored,
            "total_sampled": self._total_sampled,
            "total_pruned": self._total_pruned,
            "uptime": time.time() - self._initialized_at,
            "config": self._config.to_dict(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the buffer."""
        if not self._experiences:
            return {
                "count": 0,
                "avg_priority": 0.0,
                "min_priority": 0.0,
                "max_priority": 0.0,
                "avg_reward": 0.0,
                "task_type_count": 0,
            }

        priorities = [exp.priority for exp in self._experiences.values()]
        rewards = [exp.reward for exp in self._experiences.values()]

        return {
            "count": len(self._experiences),
            "avg_priority": sum(priorities) / len(priorities),
            "min_priority": min(priorities),
            "max_priority": max(priorities),
            "avg_reward": sum(rewards) / len(rewards),
            "task_type_count": len(self._task_type_index),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the experience replay buffer to a dictionary."""
        return {
            "config": self._config.to_dict(),
            "experiences": {exp_id: exp.to_dict() for exp_id, exp in self._experiences.items()},
            "total_stored": self._total_stored,
            "total_sampled": self._total_sampled,
            "total_pruned": self._total_pruned,
            "initialized_at": self._initialized_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceReplayBuffer":
        """Create an experience replay buffer from a dictionary."""
        config = data.get("config", {})
        buffer = cls(config=config)

        # Restore experiences
        experiences_data = data.get("experiences", {})
        for exp_id, exp_data in experiences_data.items():
            exp = Experience.from_dict(exp_data)
            buffer._experiences[exp_id] = exp

            # Rebuild task type index
            task_type = exp.task_type
            if task_type not in buffer._task_type_index:
                buffer._task_type_index[task_type] = []
            buffer._task_type_index[task_type].append(exp_id)

        # Restore statistics
        buffer._total_stored = data.get("total_stored", len(buffer._experiences))
        buffer._total_sampled = data.get("total_sampled", 0)
        buffer._total_pruned = data.get("total_pruned", 0)
        buffer._initialized_at = data.get("initialized_at", time.time())

        return buffer


def create_experience(
    task_type: str,
    context: Dict[str, Any],
    action: str,
    outcome: Dict[str, Any],
    reward: float,
    priority: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Experience:
    """
    Factory function to create an Experience with auto-generated ID.

    Args:
        task_type: Type of task this experience relates to.
        context: Contextual information at the time of the experience.
        action: The action that was taken.
        outcome: The outcome/result of the action.
        reward: The reward received.
        priority: Priority score (defaults to 0.5).
        metadata: Additional metadata.

    Returns:
        A new Experience instance.
    """
    return Experience(
        experience_id=str(uuid.uuid4()),
        task_type=task_type,
        context=context,
        action=action,
        outcome=outcome,
        reward=reward,
        priority=priority if priority is not None else 0.5,
        timestamp=time.time(),
        metadata=metadata if metadata is not None else {},
    )
