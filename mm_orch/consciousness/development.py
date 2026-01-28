"""
Development System module for consciousness system.

The DevelopmentSystem manages the system's growth stages:
- Development stage definitions (infant, child, adolescent, adult)
- Stage-based feature restrictions
- Stage promotion conditions
- Learning data recording
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import time
from enum import Enum


class DevelopmentStage(Enum):
    """Development stages for the system."""

    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"


@dataclass
class StageConfig:
    """Configuration for a development stage."""

    stage: DevelopmentStage
    enabled_features: Set[str]
    max_complexity: float  # 0.0 to 1.0
    promotion_requirements: Dict[str, Any]
    description: str


@dataclass
class LearningRecord:
    """Records learning progress data."""

    record_id: str
    stage: str
    metric_type: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DevelopmentSystem:
    """
    Development System manages the system's growth stages.

    Defines stages, manages feature restrictions, and tracks promotion.
    Implements requirements 8.1-8.5: development stages and feature management.
    """

    # Stage configurations
    STAGE_CONFIGS = {
        DevelopmentStage.INFANT: StageConfig(
            stage=DevelopmentStage.INFANT,
            enabled_features={"chat_generate", "simple_qa"},
            max_complexity=0.3,
            promotion_requirements={
                "task_count": 100,
                "success_rate": 0.7,
                "time_active_hours": 24,
            },
            description="Basic conversational abilities only",
        ),
        DevelopmentStage.CHILD: StageConfig(
            stage=DevelopmentStage.CHILD,
            enabled_features={"chat_generate", "simple_qa", "search_qa", "summarization"},
            max_complexity=0.5,
            promotion_requirements={
                "task_count": 500,
                "success_rate": 0.75,
                "time_active_hours": 168,  # 1 week
            },
            description="Search and summarization capabilities added",
        ),
        DevelopmentStage.ADOLESCENT: StageConfig(
            stage=DevelopmentStage.ADOLESCENT,
            enabled_features={
                "chat_generate",
                "simple_qa",
                "search_qa",
                "summarization",
                "rag_qa",
                "lesson_pack",
                "embedding",
            },
            max_complexity=0.8,
            promotion_requirements={
                "task_count": 2000,
                "success_rate": 0.8,
                "time_active_hours": 720,  # 1 month
            },
            description="RAG and teaching capabilities added",
        ),
        DevelopmentStage.ADULT: StageConfig(
            stage=DevelopmentStage.ADULT,
            enabled_features={
                "chat_generate",
                "simple_qa",
                "search_qa",
                "summarization",
                "rag_qa",
                "lesson_pack",
                "embedding",
                "self_ask_search_qa",
                "complex_reasoning",
                "multi_step",
            },
            max_complexity=1.0,
            promotion_requirements={},  # No further promotion
            description="Full capabilities unlocked",
        ),
    }

    # Stage order for promotion
    STAGE_ORDER = [
        DevelopmentStage.INFANT,
        DevelopmentStage.CHILD,
        DevelopmentStage.ADOLESCENT,
        DevelopmentStage.ADULT,
    ]

    def __init__(self, initial_stage: str = "adult"):
        """
        Initialize the development system.

        Args:
            initial_stage: Initial development stage name.
        """
        self._current_stage = DevelopmentStage(initial_stage)
        self._learning_records: List[LearningRecord] = []
        self._max_records: int = 1000
        self._record_counter: int = 0

        # Performance metrics for promotion evaluation
        self._metrics: Dict[str, Any] = {
            "task_count": 0,
            "success_count": 0,
            "total_score": 0.0,
            "time_active_hours": 0.0,
            "stage_start_time": time.time(),
        }

        self._initialized_at: float = time.time()

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current development system state.

        Returns:
            Dictionary containing current state information.
        """
        config = self.STAGE_CONFIGS[self._current_stage]
        return {
            "current_stage": self._current_stage.value,
            "stage_description": config.description,
            "enabled_feature_count": len(config.enabled_features),
            "max_complexity": config.max_complexity,
            "learning_record_count": len(self._learning_records),
            "metrics": self._metrics.copy(),
            "uptime": time.time() - self._initialized_at,
        }

    def get_current_stage(self) -> DevelopmentStage:
        """
        Get the current development stage.

        Returns:
            The current DevelopmentStage.
        """
        return self._current_stage

    def get_stage_config(self, stage: Optional[DevelopmentStage] = None) -> StageConfig:
        """
        Get configuration for a stage.

        Args:
            stage: The stage to get config for (default: current stage).

        Returns:
            The StageConfig for the stage.
        """
        stage_to_use = stage or self._current_stage
        return self.STAGE_CONFIGS[stage_to_use]

    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled in the current stage.

        Args:
            feature: The feature name to check.

        Returns:
            True if the feature is enabled.
        """
        config = self.STAGE_CONFIGS[self._current_stage]
        return feature in config.enabled_features

    def get_enabled_features(self) -> Set[str]:
        """
        Get all enabled features for the current stage.

        Returns:
            Set of enabled feature names.
        """
        return self.STAGE_CONFIGS[self._current_stage].enabled_features.copy()

    def get_max_complexity(self) -> float:
        """
        Get the maximum complexity allowed in the current stage.

        Returns:
            Maximum complexity value (0.0 to 1.0).
        """
        return self.STAGE_CONFIGS[self._current_stage].max_complexity

    def check_feature_access(self, feature: str) -> Dict[str, Any]:
        """
        Check if a feature can be accessed and return details.

        Args:
            feature: The feature name to check.

        Returns:
            Dictionary with access status and details.
        """
        enabled = self.is_feature_enabled(feature)

        result = {
            "feature": feature,
            "enabled": enabled,
            "current_stage": self._current_stage.value,
        }

        if not enabled:
            # Find which stage enables this feature
            for stage in self.STAGE_ORDER:
                if feature in self.STAGE_CONFIGS[stage].enabled_features:
                    result["required_stage"] = stage.value
                    result["message"] = (
                        f"Feature '{feature}' requires stage '{stage.value}' or higher"
                    )
                    break
            else:
                result["message"] = f"Feature '{feature}' is not recognized"

        return result

    # Stage promotion
    def check_promotion_eligibility(self) -> Dict[str, Any]:
        """
        Check if the system is eligible for stage promotion.

        Returns:
            Dictionary with eligibility status and details.
        """
        if self._current_stage == DevelopmentStage.ADULT:
            return {
                "eligible": False,
                "reason": "Already at maximum stage",
                "current_stage": self._current_stage.value,
            }

        config = self.STAGE_CONFIGS[self._current_stage]
        requirements = config.promotion_requirements

        # Update time active
        self._metrics["time_active_hours"] = (
            time.time() - self._metrics["stage_start_time"]
        ) / 3600

        # Check each requirement
        met_requirements = {}
        unmet_requirements = {}

        for req_name, req_value in requirements.items():
            current_value = self._metrics.get(req_name, 0)
            if req_name == "success_rate":
                current_value = (
                    self._metrics["success_count"] / self._metrics["task_count"]
                    if self._metrics["task_count"] > 0
                    else 0
                )

            if current_value >= req_value:
                met_requirements[req_name] = {"current": current_value, "required": req_value}
            else:
                unmet_requirements[req_name] = {"current": current_value, "required": req_value}

        eligible = len(unmet_requirements) == 0

        return {
            "eligible": eligible,
            "current_stage": self._current_stage.value,
            "next_stage": self._get_next_stage().value if self._get_next_stage() else None,
            "met_requirements": met_requirements,
            "unmet_requirements": unmet_requirements,
        }

    def _get_next_stage(self) -> Optional[DevelopmentStage]:
        """Get the next stage in the progression."""
        current_idx = self.STAGE_ORDER.index(self._current_stage)
        if current_idx < len(self.STAGE_ORDER) - 1:
            return self.STAGE_ORDER[current_idx + 1]
        return None

    def promote(self, force: bool = False) -> Dict[str, Any]:
        """
        Attempt to promote to the next stage.

        Args:
            force: If True, promote regardless of requirements.

        Returns:
            Dictionary with promotion result.
        """
        eligibility = self.check_promotion_eligibility()

        if not eligibility["eligible"] and not force:
            return {
                "success": False,
                "reason": "Requirements not met",
                "details": eligibility,
            }

        next_stage = self._get_next_stage()
        if not next_stage:
            return {
                "success": False,
                "reason": "Already at maximum stage",
            }

        old_stage = self._current_stage
        self._current_stage = next_stage

        # Record promotion
        self._record_learning(
            metric_type="stage_promotion",
            value=1.0,
            metadata={
                "from_stage": old_stage.value,
                "to_stage": next_stage.value,
                "forced": force,
            },
        )

        # Reset stage metrics
        self._metrics["stage_start_time"] = time.time()

        return {
            "success": True,
            "from_stage": old_stage.value,
            "to_stage": next_stage.value,
            "new_features": list(
                self.STAGE_CONFIGS[next_stage].enabled_features
                - self.STAGE_CONFIGS[old_stage].enabled_features
            ),
        }

    def set_stage(self, stage: str) -> bool:
        """
        Directly set the development stage.

        Args:
            stage: The stage name to set.

        Returns:
            True if the stage was set successfully.
        """
        try:
            self._current_stage = DevelopmentStage(stage)
            self._metrics["stage_start_time"] = time.time()
            return True
        except ValueError:
            return False

    # Learning data recording
    def record_task_result(
        self, success: bool, score: float, task_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a task result for learning tracking.

        Args:
            success: Whether the task was successful.
            score: Performance score (0.0 to 1.0).
            task_type: Type of the task.
            metadata: Optional additional metadata.
        """
        self._metrics["task_count"] += 1
        if success:
            self._metrics["success_count"] += 1
        self._metrics["total_score"] += score

        self._record_learning(
            metric_type="task_result",
            value=score,
            metadata={
                "success": success,
                "task_type": task_type,
                **(metadata or {}),
            },
        )

    def _record_learning(
        self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a learning data point."""
        self._record_counter += 1
        record = LearningRecord(
            record_id=f"lr_{self._record_counter}_{int(time.time())}",
            stage=self._current_stage.value,
            metric_type=metric_type,
            value=value,
            metadata=metadata or {},
        )

        self._learning_records.append(record)
        if len(self._learning_records) > self._max_records:
            self._learning_records = self._learning_records[-self._max_records :]

    def get_learning_records(
        self, limit: Optional[int] = None, metric_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get learning records.

        Args:
            limit: Maximum number of records to return.
            metric_type: Optional filter by metric type.

        Returns:
            List of learning record dictionaries.
        """
        records = self._learning_records
        if metric_type:
            records = [r for r in records if r.metric_type == metric_type]
        if limit:
            records = records[-limit:]

        return [
            {
                "record_id": r.record_id,
                "stage": r.stage,
                "metric_type": r.metric_type,
                "value": r.value,
                "timestamp": r.timestamp,
                "metadata": r.metadata,
            }
            for r in records
        ]

    def get_stage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the current stage.

        Returns:
            Dictionary of stage statistics.
        """
        task_count = self._metrics["task_count"]
        success_count = self._metrics["success_count"]

        return {
            "stage": self._current_stage.value,
            "task_count": task_count,
            "success_count": success_count,
            "success_rate": success_count / task_count if task_count > 0 else 0.0,
            "average_score": self._metrics["total_score"] / task_count if task_count > 0 else 0.0,
            "time_in_stage_hours": (time.time() - self._metrics["stage_start_time"]) / 3600,
            "learning_record_count": len(self._learning_records),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the development system to a dictionary.

        Returns:
            Dictionary representation of the development system.
        """
        return {
            "current_stage": self._current_stage.value,
            "metrics": self._metrics.copy(),
            "learning_records": [
                {
                    "record_id": r.record_id,
                    "stage": r.stage,
                    "metric_type": r.metric_type,
                    "value": r.value,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata,
                }
                for r in self._learning_records
            ],
            "record_counter": self._record_counter,
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the development system from a dictionary.

        Args:
            data: Dictionary representation of the development system.
        """
        if "current_stage" in data:
            self._current_stage = DevelopmentStage(data["current_stage"])

        if "metrics" in data:
            self._metrics = data["metrics"]

        if "learning_records" in data:
            self._learning_records = [
                LearningRecord(
                    record_id=r["record_id"],
                    stage=r["stage"],
                    metric_type=r["metric_type"],
                    value=r["value"],
                    timestamp=r.get("timestamp", time.time()),
                    metadata=r.get("metadata", {}),
                )
                for r in data["learning_records"]
            ]

        if "record_counter" in data:
            self._record_counter = data["record_counter"]

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]
