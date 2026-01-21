"""
Motivation System module for consciousness system.

The MotivationSystem manages the system's goals and drives:
- Goal hierarchy (short-term and long-term goals)
- Goal priority calculation
- Goal completion tracking
- Behavior selection based on motivation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
from enum import Enum


class GoalType(Enum):
    """Types of goals in the motivation system."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    IMMEDIATE = "immediate"


class GoalStatus(Enum):
    """Status of a goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Represents a goal in the motivation system."""

    goal_id: str
    name: str
    description: str
    goal_type: GoalType
    priority: float = 0.5  # 0.0 to 1.0
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    parent_goal_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MotivationSystem:
    """
    Motivation System manages the system's goals and drives.

    Maintains goal hierarchy, calculates priorities, and tracks completion.
    Implements requirements 7.1, 7.2: maintain goal hierarchy and update goal status.
    """

    def __init__(self):
        """Initialize the motivation system."""
        self._goals: Dict[str, Goal] = {}
        self._goal_counter: int = 0
        self._drive_levels: Dict[str, float] = {
            "curiosity": 0.7,  # Drive to learn and explore
            "helpfulness": 0.8,  # Drive to assist users
            "accuracy": 0.9,  # Drive for correct responses
            "efficiency": 0.6,  # Drive for quick responses
            "creativity": 0.5,  # Drive for novel solutions
        }
        self._initialized_at: float = time.time()

        # Initialize default long-term goals
        self._initialize_default_goals()

    def _initialize_default_goals(self) -> None:
        """Initialize default system goals."""
        default_goals = [
            (
                "provide_accurate_answers",
                "Provide Accurate Answers",
                "Ensure all responses are factually correct and helpful",
                GoalType.LONG_TERM,
                0.9,
            ),
            (
                "improve_performance",
                "Improve Performance",
                "Continuously improve response quality and speed",
                GoalType.LONG_TERM,
                0.7,
            ),
            (
                "learn_from_interactions",
                "Learn from Interactions",
                "Adapt and improve based on user feedback",
                GoalType.LONG_TERM,
                0.6,
            ),
        ]
        for goal_id, name, desc, goal_type, priority in default_goals:
            self._goals[goal_id] = Goal(
                goal_id=goal_id,
                name=name,
                description=desc,
                goal_type=goal_type,
                priority=priority,
            )

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current motivation system state.

        Returns:
            Dictionary containing current state information.
        """
        return {
            "goal_count": len(self._goals),
            "active_goals": sum(
                1
                for g in self._goals.values()
                if g.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]
            ),
            "completed_goals": sum(
                1 for g in self._goals.values() if g.status == GoalStatus.COMPLETED
            ),
            "drive_levels": self._drive_levels.copy(),
            "uptime": time.time() - self._initialized_at,
        }

    # Goal management
    def create_goal(
        self,
        name: str,
        description: str,
        goal_type: GoalType = GoalType.SHORT_TERM,
        priority: float = 0.5,
        parent_goal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Goal:
        """
        Create a new goal.

        Args:
            name: Goal name.
            description: Goal description.
            goal_type: Type of goal.
            priority: Goal priority (0.0 to 1.0).
            parent_goal_id: Optional parent goal ID.
            metadata: Optional metadata.

        Returns:
            The created Goal object.
        """
        self._goal_counter += 1
        goal_id = f"goal_{self._goal_counter}_{int(time.time())}"

        goal = Goal(
            goal_id=goal_id,
            name=name,
            description=description,
            goal_type=goal_type,
            priority=min(1.0, max(0.0, priority)),
            parent_goal_id=parent_goal_id,
            metadata=metadata or {},
        )

        self._goals[goal_id] = goal

        # Link to parent if specified
        if parent_goal_id and parent_goal_id in self._goals:
            self._goals[parent_goal_id].sub_goal_ids.append(goal_id)
            self._goals[parent_goal_id].updated_at = time.time()

        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """
        Get a goal by ID.

        Args:
            goal_id: The goal identifier.

        Returns:
            The Goal object or None if not found.
        """
        return self._goals.get(goal_id)

    def get_goals_by_type(self, goal_type: GoalType) -> List[Goal]:
        """
        Get all goals of a specific type.

        Args:
            goal_type: The goal type to filter by.

        Returns:
            List of Goal objects of the specified type.
        """
        return [g for g in self._goals.values() if g.goal_type == goal_type]

    def get_active_goals(self) -> List[Goal]:
        """
        Get all active (pending or in-progress) goals.

        Returns:
            List of active Goal objects.
        """
        return [
            g
            for g in self._goals.values()
            if g.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]
        ]

    def get_prioritized_goals(self, limit: Optional[int] = None) -> List[Goal]:
        """
        Get goals sorted by priority.

        Args:
            limit: Maximum number of goals to return.

        Returns:
            List of Goal objects sorted by priority (highest first).
        """
        active = self.get_active_goals()
        sorted_goals = sorted(active, key=lambda g: g.priority, reverse=True)
        if limit:
            return sorted_goals[:limit]
        return sorted_goals

    # Goal status updates
    def update_goal_status(self, goal_id: str, status: GoalStatus) -> bool:
        """
        Update a goal's status.

        Args:
            goal_id: The goal identifier.
            status: The new status.

        Returns:
            True if the goal was updated, False if not found.
        """
        if goal_id in self._goals:
            goal = self._goals[goal_id]
            goal.status = status
            goal.updated_at = time.time()

            if status == GoalStatus.COMPLETED:
                goal.completed_at = time.time()
                goal.progress = 1.0
                # Update parent goal progress
                self._update_parent_progress(goal)

            return True
        return False

    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """
        Update a goal's progress.

        Args:
            goal_id: The goal identifier.
            progress: The new progress value (0.0 to 1.0).

        Returns:
            True if the goal was updated, False if not found.
        """
        if goal_id in self._goals:
            goal = self._goals[goal_id]
            goal.progress = min(1.0, max(0.0, progress))
            goal.updated_at = time.time()

            # Auto-update status based on progress
            if goal.progress > 0 and goal.status == GoalStatus.PENDING:
                goal.status = GoalStatus.IN_PROGRESS
            elif goal.progress >= 1.0:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = time.time()

            # Update parent goal progress
            self._update_parent_progress(goal)

            return True
        return False

    def _update_parent_progress(self, goal: Goal) -> None:
        """Update parent goal progress based on sub-goals."""
        if goal.parent_goal_id and goal.parent_goal_id in self._goals:
            parent = self._goals[goal.parent_goal_id]
            if parent.sub_goal_ids:
                # Calculate average progress of sub-goals
                sub_progress = []
                for sub_id in parent.sub_goal_ids:
                    if sub_id in self._goals:
                        sub_progress.append(self._goals[sub_id].progress)
                if sub_progress:
                    parent.progress = sum(sub_progress) / len(sub_progress)
                    parent.updated_at = time.time()

    def update_goal_priority(self, goal_id: str, priority: float) -> bool:
        """
        Update a goal's priority.

        Args:
            goal_id: The goal identifier.
            priority: The new priority value (0.0 to 1.0).

        Returns:
            True if the goal was updated, False if not found.
        """
        if goal_id in self._goals:
            self._goals[goal_id].priority = min(1.0, max(0.0, priority))
            self._goals[goal_id].updated_at = time.time()
            return True
        return False

    # Drive management
    def get_drive_level(self, drive_name: str) -> float:
        """
        Get the level of a specific drive.

        Args:
            drive_name: The drive name.

        Returns:
            The drive level (0.0 to 1.0) or 0.0 if not found.
        """
        return self._drive_levels.get(drive_name, 0.0)

    def update_drive_level(self, drive_name: str, delta: float) -> float:
        """
        Update a drive level.

        Args:
            drive_name: The drive name.
            delta: The change in drive level.

        Returns:
            The new drive level.
        """
        if drive_name in self._drive_levels:
            new_level = self._drive_levels[drive_name] + delta
            self._drive_levels[drive_name] = min(1.0, max(0.0, new_level))
        return self._drive_levels.get(drive_name, 0.0)

    def set_drive_level(self, drive_name: str, level: float) -> None:
        """
        Set a drive level directly.

        Args:
            drive_name: The drive name.
            level: The new drive level (0.0 to 1.0).
        """
        self._drive_levels[drive_name] = min(1.0, max(0.0, level))

    # Task completion handling
    def on_task_completed(self, task_type: str, success: bool, score: float) -> Dict[str, Any]:
        """
        Handle task completion and update motivation state.

        Args:
            task_type: The type of task completed.
            success: Whether the task was successful.
            score: Performance score (0.0 to 1.0).

        Returns:
            Dictionary of motivation state changes.
        """
        changes = {}

        # Update drives based on task outcome
        if success:
            # Successful task increases helpfulness and accuracy drives
            changes["helpfulness"] = self.update_drive_level("helpfulness", 0.05)
            if score > 0.8:
                changes["accuracy"] = self.update_drive_level("accuracy", 0.03)
        else:
            # Failed task decreases drives slightly
            changes["helpfulness"] = self.update_drive_level("helpfulness", -0.02)

        # Update relevant goals
        for goal in self.get_active_goals():
            if self._is_goal_related_to_task(goal, task_type):
                if success:
                    new_progress = min(1.0, goal.progress + 0.1 * score)
                    self.update_goal_progress(goal.goal_id, new_progress)
                    changes[f"goal_{goal.goal_id}_progress"] = new_progress

        return changes

    def _is_goal_related_to_task(self, goal: Goal, task_type: str) -> bool:
        """Check if a goal is related to a task type."""
        # Simple mapping of task types to goal keywords
        task_goal_mapping = {
            "search_qa": ["accurate", "answer"],
            "lesson_pack": ["learn", "improve"],
            "chat_generate": ["help", "assist"],
            "rag_qa": ["accurate", "knowledge"],
        }
        keywords = task_goal_mapping.get(task_type, [])
        goal_text = f"{goal.name} {goal.description}".lower()
        return any(kw in goal_text for kw in keywords)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the motivation system to a dictionary.

        Returns:
            Dictionary representation of the motivation system.
        """
        return {
            "goals": {
                gid: {
                    "goal_id": g.goal_id,
                    "name": g.name,
                    "description": g.description,
                    "goal_type": g.goal_type.value,
                    "priority": g.priority,
                    "status": g.status.value,
                    "progress": g.progress,
                    "parent_goal_id": g.parent_goal_id,
                    "sub_goal_ids": g.sub_goal_ids,
                    "created_at": g.created_at,
                    "updated_at": g.updated_at,
                    "completed_at": g.completed_at,
                    "metadata": g.metadata,
                }
                for gid, g in self._goals.items()
            },
            "drive_levels": self._drive_levels.copy(),
            "goal_counter": self._goal_counter,
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the motivation system from a dictionary.

        Args:
            data: Dictionary representation of the motivation system.
        """
        if "goals" in data:
            self._goals = {}
            for gid, g_data in data["goals"].items():
                self._goals[gid] = Goal(
                    goal_id=g_data["goal_id"],
                    name=g_data["name"],
                    description=g_data["description"],
                    goal_type=GoalType(g_data["goal_type"]),
                    priority=g_data.get("priority", 0.5),
                    status=GoalStatus(g_data.get("status", "pending")),
                    progress=g_data.get("progress", 0.0),
                    parent_goal_id=g_data.get("parent_goal_id"),
                    sub_goal_ids=g_data.get("sub_goal_ids", []),
                    created_at=g_data.get("created_at", time.time()),
                    updated_at=g_data.get("updated_at", time.time()),
                    completed_at=g_data.get("completed_at"),
                    metadata=g_data.get("metadata", {}),
                )

        if "drive_levels" in data:
            self._drive_levels = data["drive_levels"]

        if "goal_counter" in data:
            self._goal_counter = data["goal_counter"]

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]
