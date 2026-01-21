"""
Metacognition module for consciousness system.

The Metacognition module monitors and regulates cognitive processes:
- Task execution monitoring
- Strategy evaluation and adjustment
- Performance assessment
- Strategy suggestion generation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

from mm_orch.schemas import StrategySuggestion, Task


@dataclass
class TaskMonitor:
    """Monitors a task's execution."""

    task_id: str
    task_type: str
    started_at: float
    status: str = "running"  # running, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyRecord:
    """Records a strategy's usage and effectiveness."""

    strategy_name: str
    usage_count: int = 0
    success_count: int = 0
    total_time: float = 0.0
    average_score: float = 0.5


class Metacognition:
    """
    Metacognition module monitors and regulates cognitive processes.

    Provides task monitoring, strategy evaluation, and adjustment suggestions.
    Implements requirement 6.5: monitor task execution and provide strategy suggestions.
    """

    # Strategy definitions with their applicability conditions
    STRATEGIES = {
        "direct_answer": {
            "description": "Directly answer using available knowledge",
            "applicable_to": ["simple_qa", "chat"],
            "complexity_threshold": 0.3,
        },
        "search_and_answer": {
            "description": "Search for information then answer",
            "applicable_to": ["search_qa", "complex_qa"],
            "complexity_threshold": 0.5,
        },
        "decompose_and_solve": {
            "description": "Break down complex problem into sub-problems",
            "applicable_to": ["complex_reasoning", "self_ask"],
            "complexity_threshold": 0.7,
        },
        "retrieve_and_generate": {
            "description": "Retrieve from knowledge base then generate",
            "applicable_to": ["rag_qa", "knowledge_query"],
            "complexity_threshold": 0.4,
        },
        "structured_generation": {
            "description": "Generate structured content step by step",
            "applicable_to": ["lesson_pack", "teaching"],
            "complexity_threshold": 0.6,
        },
    }

    def __init__(self):
        """Initialize the metacognition module."""
        self._active_tasks: Dict[str, TaskMonitor] = {}
        self._completed_tasks: List[TaskMonitor] = []
        self._strategy_records: Dict[str, StrategyRecord] = {
            name: StrategyRecord(strategy_name=name) for name in self.STRATEGIES
        }
        self._max_completed_history: int = 100
        self._initialized_at: float = time.time()

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current metacognition state.

        Returns:
            Dictionary containing current state information.
        """
        return {
            "active_task_count": len(self._active_tasks),
            "completed_task_count": len(self._completed_tasks),
            "strategy_count": len(self._strategy_records),
            "uptime": time.time() - self._initialized_at,
        }

    # Task monitoring
    def start_monitoring(self, task: Task) -> TaskMonitor:
        """
        Start monitoring a task.

        Args:
            task: The Task object to monitor.

        Returns:
            The TaskMonitor object for the task.
        """
        monitor = TaskMonitor(
            task_id=task.task_id,
            task_type=task.task_type,
            started_at=time.time(),
        )
        self._active_tasks[task.task_id] = monitor
        return monitor

    def update_progress(
        self, task_id: str, progress: float, checkpoint: Optional[str] = None
    ) -> bool:
        """
        Update task progress.

        Args:
            task_id: The task identifier.
            progress: Progress value (0.0 to 1.0).
            checkpoint: Optional checkpoint description.

        Returns:
            True if the task was updated, False if not found.
        """
        if task_id in self._active_tasks:
            monitor = self._active_tasks[task_id]
            monitor.progress = min(1.0, max(0.0, progress))
            if checkpoint:
                monitor.checkpoints.append(
                    {
                        "description": checkpoint,
                        "progress": progress,
                        "timestamp": time.time(),
                    }
                )
            return True
        return False

    def complete_task(
        self, task_id: str, success: bool, metrics: Optional[Dict[str, float]] = None
    ) -> Optional[TaskMonitor]:
        """
        Mark a task as completed.

        Args:
            task_id: The task identifier.
            success: Whether the task was successful.
            metrics: Optional performance metrics.

        Returns:
            The completed TaskMonitor or None if not found.
        """
        if task_id in self._active_tasks:
            monitor = self._active_tasks.pop(task_id)
            monitor.status = "completed" if success else "failed"
            monitor.progress = 1.0 if success else monitor.progress
            if metrics:
                monitor.metrics = metrics

            # Add to completed history
            self._completed_tasks.append(monitor)
            if len(self._completed_tasks) > self._max_completed_history:
                self._completed_tasks = self._completed_tasks[-self._max_completed_history :]

            return monitor
        return None

    def get_active_tasks(self) -> List[TaskMonitor]:
        """
        Get all active task monitors.

        Returns:
            List of active TaskMonitor objects.
        """
        return list(self._active_tasks.values())

    def get_task_monitor(self, task_id: str) -> Optional[TaskMonitor]:
        """
        Get a task monitor by ID.

        Args:
            task_id: The task identifier.

        Returns:
            The TaskMonitor or None if not found.
        """
        return self._active_tasks.get(task_id)

    # Strategy suggestion
    def get_strategy_suggestion(self, task: Task) -> StrategySuggestion:
        """
        Get a strategy suggestion for a task.

        Args:
            task: The Task object to get suggestion for.

        Returns:
            A StrategySuggestion object with the recommended strategy.
        """
        task_type = task.task_type
        complexity = self._estimate_complexity(task)

        # Find applicable strategies
        applicable = []
        for name, config in self.STRATEGIES.items():
            if task_type in config["applicable_to"] or self._is_type_compatible(
                task_type, config["applicable_to"]
            ):
                score = self._calculate_strategy_score(name, complexity, config)
                applicable.append((name, score, config))

        # Sort by score and select best
        if applicable:
            applicable.sort(key=lambda x: x[1], reverse=True)
            best_name, best_score, best_config = applicable[0]

            return StrategySuggestion(
                strategy=best_name,
                confidence=min(1.0, best_score),
                reasoning=f"Selected '{best_name}' for task type '{task_type}' with complexity {complexity:.2f}. {best_config['description']}",
                parameters={
                    "complexity": complexity,
                    "alternatives": [a[0] for a in applicable[1:3]],
                },
            )

        # Default strategy
        return StrategySuggestion(
            strategy="direct_answer",
            confidence=0.5,
            reasoning=f"No specific strategy found for task type '{task_type}', using default direct answer approach.",
            parameters={"complexity": complexity},
        )

    def _estimate_complexity(self, task: Task) -> float:
        """
        Estimate task complexity.

        Args:
            task: The Task object.

        Returns:
            Complexity score (0.0 to 1.0).
        """
        complexity = 0.3  # Base complexity

        params = task.parameters

        # Adjust based on query length
        if "query" in params:
            query_len = len(params["query"])
            if query_len > 200:
                complexity += 0.2
            elif query_len > 100:
                complexity += 0.1

        # Adjust based on task type
        complex_types = {"self_ask_search_qa", "complex_reasoning", "lesson_pack"}
        if task.task_type in complex_types:
            complexity += 0.3

        # Adjust based on context
        if params.get("context"):
            complexity += 0.1

        return min(1.0, complexity)

    def _is_type_compatible(self, task_type: str, applicable_types: List[str]) -> bool:
        """Check if task type is compatible with applicable types."""
        type_mappings = {
            "search_qa": ["search_qa", "complex_qa"],
            "lesson_pack": ["lesson_pack", "teaching"],
            "chat_generate": ["chat", "simple_qa"],
            "rag_qa": ["rag_qa", "knowledge_query"],
            "self_ask_search_qa": ["self_ask", "complex_reasoning"],
        }
        mapped_types = type_mappings.get(task_type, [task_type])
        return any(t in applicable_types for t in mapped_types)

    def _calculate_strategy_score(
        self, strategy_name: str, complexity: float, config: Dict
    ) -> float:
        """Calculate strategy score based on complexity and historical performance."""
        # Base score from complexity match
        threshold = config["complexity_threshold"]
        complexity_match = 1.0 - abs(complexity - threshold)

        # Historical performance
        record = self._strategy_records.get(strategy_name)
        if record and record.usage_count > 0:
            success_rate = record.success_count / record.usage_count
            historical_score = (success_rate + record.average_score) / 2
        else:
            historical_score = 0.5

        # Combined score
        return 0.6 * complexity_match + 0.4 * historical_score

    def record_strategy_result(
        self, strategy_name: str, success: bool, score: float, execution_time: float
    ) -> None:
        """
        Record the result of using a strategy.

        Args:
            strategy_name: The strategy name.
            success: Whether the strategy was successful.
            score: Performance score (0.0 to 1.0).
            execution_time: Time taken to execute.
        """
        if strategy_name in self._strategy_records:
            record = self._strategy_records[strategy_name]
            record.usage_count += 1
            if success:
                record.success_count += 1
            record.total_time += execution_time
            # Update average score with exponential moving average
            alpha = 0.3
            record.average_score = alpha * score + (1 - alpha) * record.average_score

    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all strategies.

        Returns:
            Dictionary of strategy statistics.
        """
        return {
            name: {
                "usage_count": record.usage_count,
                "success_rate": (
                    record.success_count / record.usage_count if record.usage_count > 0 else 0.0
                ),
                "average_score": record.average_score,
                "average_time": (
                    record.total_time / record.usage_count if record.usage_count > 0 else 0.0
                ),
            }
            for name, record in self._strategy_records.items()
        }

    # Performance assessment
    def assess_performance(self, task_id: str) -> Dict[str, Any]:
        """
        Assess the performance of a completed task.

        Args:
            task_id: The task identifier.

        Returns:
            Dictionary containing performance assessment.
        """
        # Find in completed tasks
        monitor = None
        for m in self._completed_tasks:
            if m.task_id == task_id:
                monitor = m
                break

        if not monitor:
            return {"error": "Task not found in completed tasks"}

        execution_time = time.time() - monitor.started_at
        checkpoint_count = len(monitor.checkpoints)

        return {
            "task_id": task_id,
            "task_type": monitor.task_type,
            "status": monitor.status,
            "execution_time": execution_time,
            "checkpoint_count": checkpoint_count,
            "metrics": monitor.metrics,
            "assessment": self._generate_assessment(monitor, execution_time),
        }

    def _generate_assessment(self, monitor: TaskMonitor, execution_time: float) -> str:
        """Generate a text assessment of task performance."""
        if monitor.status == "completed":
            if execution_time < 5.0:
                return "Excellent performance - task completed quickly"
            elif execution_time < 30.0:
                return "Good performance - task completed in reasonable time"
            else:
                return "Acceptable performance - task completed but took longer than expected"
        else:
            return "Task failed - review checkpoints for debugging"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the metacognition module to a dictionary.

        Returns:
            Dictionary representation of the metacognition module.
        """
        return {
            "active_tasks": {
                tid: {
                    "task_id": m.task_id,
                    "task_type": m.task_type,
                    "started_at": m.started_at,
                    "status": m.status,
                    "progress": m.progress,
                    "checkpoints": m.checkpoints,
                    "metrics": m.metrics,
                }
                for tid, m in self._active_tasks.items()
            },
            "completed_tasks": [
                {
                    "task_id": m.task_id,
                    "task_type": m.task_type,
                    "started_at": m.started_at,
                    "status": m.status,
                    "progress": m.progress,
                    "checkpoints": m.checkpoints,
                    "metrics": m.metrics,
                }
                for m in self._completed_tasks
            ],
            "strategy_records": {
                name: {
                    "strategy_name": r.strategy_name,
                    "usage_count": r.usage_count,
                    "success_count": r.success_count,
                    "total_time": r.total_time,
                    "average_score": r.average_score,
                }
                for name, r in self._strategy_records.items()
            },
            "initialized_at": self._initialized_at,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore the metacognition module from a dictionary.

        Args:
            data: Dictionary representation of the metacognition module.
        """
        if "active_tasks" in data:
            self._active_tasks = {}
            for tid, m_data in data["active_tasks"].items():
                self._active_tasks[tid] = TaskMonitor(
                    task_id=m_data["task_id"],
                    task_type=m_data["task_type"],
                    started_at=m_data["started_at"],
                    status=m_data.get("status", "running"),
                    progress=m_data.get("progress", 0.0),
                    checkpoints=m_data.get("checkpoints", []),
                    metrics=m_data.get("metrics", {}),
                )

        if "completed_tasks" in data:
            self._completed_tasks = [
                TaskMonitor(
                    task_id=m_data["task_id"],
                    task_type=m_data["task_type"],
                    started_at=m_data["started_at"],
                    status=m_data.get("status", "completed"),
                    progress=m_data.get("progress", 1.0),
                    checkpoints=m_data.get("checkpoints", []),
                    metrics=m_data.get("metrics", {}),
                )
                for m_data in data["completed_tasks"]
            ]

        if "strategy_records" in data:
            for name, r_data in data["strategy_records"].items():
                if name in self._strategy_records:
                    self._strategy_records[name] = StrategyRecord(
                        strategy_name=r_data["strategy_name"],
                        usage_count=r_data.get("usage_count", 0),
                        success_count=r_data.get("success_count", 0),
                        total_time=r_data.get("total_time", 0.0),
                        average_score=r_data.get("average_score", 0.5),
                    )

        if "initialized_at" in data:
            self._initialized_at = data["initialized_at"]
