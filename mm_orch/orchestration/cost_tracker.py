"""
Cost Tracker - Track and calculate execution costs for workflow steps.

This module provides cost tracking functionality for workflow steps, including:
- Latency measurement in milliseconds
- VRAM peak usage tracking in megabytes
- Model load count tracking
- Normalized cost calculation with weighted formula
"""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class StepCost:
    """
    Cost metrics for a single step execution.

    Attributes:
        step_name: Name of the step
        latency_ms: Execution time in milliseconds
        vram_peak_mb: Peak VRAM usage in megabytes
        model_loads: Number of model loads performed
        normalized_cost: Weighted combination of metrics
        start_time: Timestamp when step started
        end_time: Timestamp when step ended
    """

    step_name: str
    latency_ms: float = 0.0
    vram_peak_mb: int = 0
    model_loads: int = 0
    normalized_cost: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "latency_ms": self.latency_ms,
            "vram_peak_mb": self.vram_peak_mb,
            "model_loads": self.model_loads,
            "normalized_cost": self.normalized_cost,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class CostTracker:
    """
    Tracks execution costs for workflow steps.

    The cost tracker:
    1. Measures latency for each step execution
    2. Tracks peak VRAM usage during execution
    3. Counts model loads performed
    4. Calculates normalized cost using weighted formula

    Cost Formula:
        normalized_cost = w1 * latency_ms + w2 * vram_peak_mb + w3 * model_loads

    Default weights:
        w1 = 0.001 (latency weight - 1ms = 0.001 cost units)
        w2 = 0.01  (VRAM weight - 1MB = 0.01 cost units)
        w3 = 10.0  (model load weight - 1 load = 10 cost units)

    Example:
        tracker = CostTracker()

        # Start tracking
        cost_id = tracker.start_step("web_search")

        # ... execute step ...

        # End tracking
        cost = tracker.end_step(cost_id, model_loads=0)
        print(f"Cost: {cost.normalized_cost}")
    """

    # Default cost weights
    DEFAULT_LATENCY_WEIGHT = 0.001  # 1ms = 0.001 cost units
    DEFAULT_VRAM_WEIGHT = 0.01  # 1MB = 0.01 cost units
    DEFAULT_MODEL_LOAD_WEIGHT = 10.0  # 1 load = 10 cost units

    def __init__(
        self,
        latency_weight: float = DEFAULT_LATENCY_WEIGHT,
        vram_weight: float = DEFAULT_VRAM_WEIGHT,
        model_load_weight: float = DEFAULT_MODEL_LOAD_WEIGHT,
    ):
        """
        Initialize cost tracker with custom weights.

        Args:
            latency_weight: Weight for latency in cost calculation
            vram_weight: Weight for VRAM in cost calculation
            model_load_weight: Weight for model loads in cost calculation

        Raises:
            ValueError: If any weight is negative
        """
        if latency_weight < 0 or vram_weight < 0 or model_load_weight < 0:
            raise ValueError("All cost weights must be non-negative")

        self.latency_weight = latency_weight
        self.vram_weight = vram_weight
        self.model_load_weight = model_load_weight

        self._active_costs: Dict[str, StepCost] = {}
        self._completed_costs: Dict[str, StepCost] = {}
        self._cuda_available = torch.cuda.is_available()

    def start_step(self, step_name: str) -> str:
        """
        Begin tracking costs for a step execution.

        Args:
            step_name: Name of the step

        Returns:
            Cost tracking ID for this execution
        """
        cost_id = f"{step_name}_{time.time()}"

        # Record initial VRAM if available
        vram_start = 0
        if self._cuda_available:
            try:
                vram_start = torch.cuda.memory_allocated() // (1024 * 1024)  # Convert to MB
            except Exception:
                vram_start = 0

        cost = StepCost(step_name=step_name, start_time=time.time(), vram_peak_mb=vram_start)

        self._active_costs[cost_id] = cost
        return cost_id

    def end_step(self, cost_id: str, model_loads: int = 0) -> StepCost:
        """
        Complete cost tracking for a step execution.

        Args:
            cost_id: Cost tracking ID from start_step
            model_loads: Number of model loads performed during execution

        Returns:
            StepCost object with all metrics calculated

        Raises:
            KeyError: If cost_id is not found in active costs
        """
        if cost_id not in self._active_costs:
            raise KeyError(f"Cost tracking ID '{cost_id}' not found in active costs")

        cost = self._active_costs[cost_id]
        cost.end_time = time.time()

        # Calculate latency
        cost.latency_ms = (cost.end_time - cost.start_time) * 1000

        # Update VRAM peak
        if self._cuda_available:
            try:
                vram_current = torch.cuda.memory_allocated() // (1024 * 1024)  # Convert to MB
                cost.vram_peak_mb = max(cost.vram_peak_mb, vram_current)
            except Exception:
                pass

        # Record model loads
        cost.model_loads = model_loads

        # Calculate normalized cost
        cost.normalized_cost = self._calculate_cost(
            cost.latency_ms, cost.vram_peak_mb, cost.model_loads
        )

        # Move to completed costs
        self._completed_costs[cost_id] = cost
        del self._active_costs[cost_id]

        return cost

    def _calculate_cost(self, latency_ms: float, vram_peak_mb: int, model_loads: int) -> float:
        """
        Calculate normalized cost using weighted formula.

        Formula:
            cost = w1 * latency_ms + w2 * vram_peak_mb + w3 * model_loads

        Args:
            latency_ms: Latency in milliseconds
            vram_peak_mb: Peak VRAM in megabytes
            model_loads: Number of model loads

        Returns:
            Normalized cost value
        """
        cost = (
            self.latency_weight * latency_ms
            + self.vram_weight * vram_peak_mb
            + self.model_load_weight * model_loads
        )
        return cost

    def get_cost(self, cost_id: str) -> Optional[StepCost]:
        """
        Retrieve cost metrics for a completed step.

        Args:
            cost_id: Cost tracking ID

        Returns:
            StepCost object or None if not found
        """
        return self._completed_costs.get(cost_id)

    def get_all_costs(self) -> Dict[str, StepCost]:
        """
        Get all completed cost records.

        Returns:
            Dictionary mapping cost IDs to StepCost objects
        """
        return dict(self._completed_costs)

    def clear(self) -> None:
        """Clear all cost records."""
        self._active_costs.clear()
        self._completed_costs.clear()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of tracked costs.

        Returns:
            Dictionary with summary statistics
        """
        if not self._completed_costs:
            return {
                "total_steps": 0,
                "total_latency_ms": 0.0,
                "total_vram_mb": 0,
                "total_model_loads": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
                "avg_vram_mb": 0.0,
                "avg_model_loads": 0.0,
                "avg_cost": 0.0,
            }

        costs = list(self._completed_costs.values())
        total_steps = len(costs)

        total_latency = sum(c.latency_ms for c in costs)
        total_vram = sum(c.vram_peak_mb for c in costs)
        total_loads = sum(c.model_loads for c in costs)
        total_cost = sum(c.normalized_cost for c in costs)

        return {
            "total_steps": total_steps,
            "total_latency_ms": total_latency,
            "total_vram_mb": total_vram,
            "total_model_loads": total_loads,
            "total_cost": total_cost,
            "avg_latency_ms": total_latency / total_steps,
            "avg_vram_mb": total_vram / total_steps,
            "avg_model_loads": total_loads / total_steps,
            "avg_cost": total_cost / total_steps,
        }
