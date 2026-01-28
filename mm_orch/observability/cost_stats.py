"""Cost statistics aggregation for workflow executions."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from mm_orch.observability.tracer import WorkflowTrace


@dataclass
class WorkflowCostStats:
    """Aggregated cost statistics for a workflow."""

    workflow_name: str
    execution_count: int = 0
    avg_latency_ms: float = 0.0
    avg_vram_mb: float = 0.0
    avg_model_loads: float = 0.0
    success_rate: float = 1.0

    def update(self, trace: "WorkflowTrace"):
        """
        Update statistics with new trace using incremental averaging.

        Args:
            trace: Workflow trace to incorporate into statistics
        """
        # Import here to avoid circular dependency
        from mm_orch.observability.tracer import WorkflowTrace

        n = self.execution_count

        # Calculate totals from trace
        total_latency = sum(s.latency_ms for s in trace.steps)
        max_vram = max((s.vram_peak_mb for s in trace.steps), default=0)
        total_loads = sum(s.model_loads for s in trace.steps)

        # Incremental average formula: new_avg = (old_avg * n + new_value) / (n + 1)
        self.avg_latency_ms = (self.avg_latency_ms * n + total_latency) / (n + 1)
        self.avg_vram_mb = (self.avg_vram_mb * n + max_vram) / (n + 1)
        self.avg_model_loads = (self.avg_model_loads * n + total_loads) / (n + 1)

        # Update success rate
        success = all(s.success for s in trace.steps)
        self.success_rate = (self.success_rate * n + (1 if success else 0)) / (n + 1)

        self.execution_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCostStats":
        """Create from dictionary."""
        return cls(**data)


class CostStatsManager:
    """Manages cost statistics persistence and retrieval."""

    def __init__(self, stats_path: str):
        """
        Initialize cost stats manager.

        Args:
            stats_path: Path to JSON file for statistics storage
        """
        self.stats_path = Path(stats_path)
        self.stats: Dict[str, WorkflowCostStats] = {}

        # Ensure directory exists
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing stats if available
        self.load()

    def update(self, trace: "WorkflowTrace"):
        """
        Update statistics for a workflow.

        Args:
            trace: Workflow trace to incorporate
        """
        workflow_name = trace.chosen_workflow

        if workflow_name not in self.stats:
            self.stats[workflow_name] = WorkflowCostStats(workflow_name=workflow_name)

        self.stats[workflow_name].update(trace)

    def get(self, workflow_name: str) -> WorkflowCostStats:
        """
        Get statistics for a workflow.

        Args:
            workflow_name: Name of the workflow

        Returns:
            WorkflowCostStats for the workflow, or default if not found
        """
        return self.stats.get(workflow_name, WorkflowCostStats(workflow_name=workflow_name))

    def save(self):
        """Persist statistics to JSON file."""
        try:
            data = {name: stats.to_dict() for name, stats in self.stats.items()}
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import sys

            print(f"Failed to save cost stats: {e}", file=sys.stderr)

    def load(self):
        """Load statistics from JSON file."""
        if not self.stats_path.exists():
            return

        try:
            with open(self.stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.stats = {
                    name: WorkflowCostStats.from_dict(stats_dict)
                    for name, stats_dict in data.items()
                }
        except Exception as e:
            import sys

            print(f"Failed to load cost stats: {e}", file=sys.stderr)

    def get_all_stats(self) -> Dict[str, WorkflowCostStats]:
        """Get all workflow statistics."""
        return self.stats.copy()
