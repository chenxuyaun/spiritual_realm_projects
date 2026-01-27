"""
Property-based tests for cost tracking functionality.

This module validates:
- Property 13: Cost Calculation Invariant
- Property 14: Cost Statistics Convergence

Validates Requirements: 7.1, 7.2, 7.3, 7.4, 12.1, 12.2, 12.3
"""

import time
from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
import pytest
from mm_orch.orchestration.cost_tracker import CostTracker, StepCost


# ============================================================================
# Property 13: Cost Calculation Invariant
# ============================================================================
# For any completed Step execution, the recorded cost metrics (latency_ms,
# vram_peak_mb, model_loads) must all be non-negative, and the normalized
# cost must equal the weighted sum: w1×latency + w2×vram + w3×loads,
# where weights are positive constants.
# ============================================================================

@given(
    latency_ms=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    vram_peak_mb=st.integers(min_value=0, max_value=100000),
    model_loads=st.integers(min_value=0, max_value=100),
    latency_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    vram_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    model_load_weight=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=200, deadline=None)
def test_property_13_cost_calculation_invariant(
    latency_ms, vram_peak_mb, model_loads,
    latency_weight, vram_weight, model_load_weight
):
    """
    Property 13: Cost Calculation Invariant
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
    
    For any completed Step execution, the recorded cost metrics must all be
    non-negative, and the normalized cost must equal the weighted sum formula.
    """
    # Create tracker with given weights
    tracker = CostTracker(
        latency_weight=latency_weight,
        vram_weight=vram_weight,
        model_load_weight=model_load_weight
    )
    
    # Calculate cost using internal method
    calculated_cost = tracker._calculate_cost(latency_ms, vram_peak_mb, model_loads)
    
    # Property 1: All metrics must be non-negative
    assert latency_ms >= 0, "Latency must be non-negative"
    assert vram_peak_mb >= 0, "VRAM peak must be non-negative"
    assert model_loads >= 0, "Model loads must be non-negative"
    
    # Property 2: Normalized cost must be non-negative (since all inputs are non-negative)
    assert calculated_cost >= 0, "Normalized cost must be non-negative"
    
    # Property 3: Cost must equal weighted sum formula
    expected_cost = (
        latency_weight * latency_ms +
        vram_weight * vram_peak_mb +
        model_load_weight * model_loads
    )
    
    # Use relative tolerance for floating point comparison
    tolerance = 1e-6
    assert abs(calculated_cost - expected_cost) <= tolerance, (
        f"Cost calculation invariant violated: "
        f"calculated={calculated_cost}, expected={expected_cost}"
    )
    
    # Property 4: If all metrics are zero, cost must be zero
    if latency_ms == 0 and vram_peak_mb == 0 and model_loads == 0:
        assert calculated_cost == 0, "Cost must be zero when all metrics are zero"
    
    # Property 5: Cost is monotonic in each metric (holding others constant)
    # If we increase any metric, cost should not decrease
    if latency_ms > 0:
        higher_latency_cost = tracker._calculate_cost(
            latency_ms * 1.1, vram_peak_mb, model_loads
        )
        assert higher_latency_cost >= calculated_cost, (
            "Cost must be monotonic in latency"
        )
    
    if vram_peak_mb > 0:
        higher_vram_cost = tracker._calculate_cost(
            latency_ms, int(vram_peak_mb * 1.1) + 1, model_loads
        )
        assert higher_vram_cost >= calculated_cost, (
            "Cost must be monotonic in VRAM"
        )
    
    if model_loads > 0:
        higher_loads_cost = tracker._calculate_cost(
            latency_ms, vram_peak_mb, model_loads + 1
        )
        assert higher_loads_cost >= calculated_cost, (
            "Cost must be monotonic in model loads"
        )


@given(
    step_name=st.text(min_size=1, max_size=50),
    model_loads=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100, deadline=None)
def test_property_13_step_cost_metrics_non_negative(step_name, model_loads):
    """
    Property 13: Step Cost Metrics Non-Negative
    
    **Validates: Requirements 7.1, 7.2, 7.3**
    
    For any step execution, all recorded metrics must be non-negative.
    """
    tracker = CostTracker()
    
    # Start and end step tracking
    cost_id = tracker.start_step(step_name)
    
    # Simulate some work
    time.sleep(0.001)
    
    cost = tracker.end_step(cost_id, model_loads=model_loads)
    
    # All metrics must be non-negative
    assert cost.latency_ms >= 0, "Latency must be non-negative"
    assert cost.vram_peak_mb >= 0, "VRAM peak must be non-negative"
    assert cost.model_loads >= 0, "Model loads must be non-negative"
    assert cost.normalized_cost >= 0, "Normalized cost must be non-negative"
    
    # Timestamps must be valid
    assert cost.start_time > 0, "Start time must be positive"
    assert cost.end_time > 0, "End time must be positive"
    assert cost.end_time >= cost.start_time, "End time must be >= start time"
    
    # Model loads must match what was provided
    assert cost.model_loads == model_loads, "Model loads must match input"


# ============================================================================
# Property 14: Cost Statistics Convergence
# ============================================================================
# For any workflow with multiple executions, the running average statistics
# (avg_latency_ms, avg_vram_mb, avg_model_loads) must converge toward the
# true mean as execution_count increases, following the incremental average
# formula.
# ============================================================================

@given(
    execution_data=st.lists(
        st.tuples(
            st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),  # latency
            st.integers(min_value=0, max_value=10000),  # vram
            st.integers(min_value=0, max_value=10)  # model_loads
        ),
        min_size=1,
        max_size=50
    )
)
@settings(max_examples=100, deadline=None)
def test_property_14_cost_statistics_convergence(execution_data):
    """
    Property 14: Cost Statistics Convergence
    
    **Validates: Requirements 12.1, 12.2, 12.3**
    
    For any workflow with multiple executions, the running average statistics
    must converge toward the true mean as execution_count increases.
    """
    num_executions = len(execution_data)
    latencies = [data[0] for data in execution_data]
    vram_values = [data[1] for data in execution_data]
    model_loads_list = [data[2] for data in execution_data]
    
    tracker = CostTracker()
    
    # Execute steps and track costs
    for i in range(num_executions):
        cost_id = tracker.start_step(f"step_{i}")
        
        # Simulate execution with known metrics
        # We'll manually set the metrics for testing
        cost = tracker._active_costs[cost_id]
        cost.end_time = cost.start_time + (latencies[i] / 1000.0)
        cost.latency_ms = latencies[i]
        cost.vram_peak_mb = vram_values[i]
        cost.model_loads = model_loads_list[i]
        cost.normalized_cost = tracker._calculate_cost(
            cost.latency_ms, cost.vram_peak_mb, cost.model_loads
        )
        
        # Move to completed
        tracker._completed_costs[cost_id] = cost
        del tracker._active_costs[cost_id]
    
    # Get summary statistics
    summary = tracker.get_summary()
    
    # Calculate true means
    true_avg_latency = sum(latencies) / num_executions
    true_avg_vram = sum(vram_values) / num_executions
    true_avg_loads = sum(model_loads_list) / num_executions
    
    # Property 1: Execution count must match
    assert summary["total_steps"] == num_executions, (
        "Total steps must match number of executions"
    )
    
    # Property 2: Average latency must equal true mean
    tolerance = 1e-6
    assert abs(summary["avg_latency_ms"] - true_avg_latency) <= tolerance, (
        f"Average latency must converge to true mean: "
        f"got={summary['avg_latency_ms']}, expected={true_avg_latency}"
    )
    
    # Property 3: Average VRAM must equal true mean
    assert abs(summary["avg_vram_mb"] - true_avg_vram) <= tolerance, (
        f"Average VRAM must converge to true mean: "
        f"got={summary['avg_vram_mb']}, expected={true_avg_vram}"
    )
    
    # Property 4: Average model loads must equal true mean
    assert abs(summary["avg_model_loads"] - true_avg_loads) <= tolerance, (
        f"Average model loads must converge to true mean: "
        f"got={summary['avg_model_loads']}, expected={true_avg_loads}"
    )
    
    # Property 5: Total metrics must equal sum of individual metrics
    assert abs(summary["total_latency_ms"] - sum(latencies)) <= tolerance, (
        "Total latency must equal sum of individual latencies"
    )
    assert summary["total_vram_mb"] == sum(vram_values), (
        "Total VRAM must equal sum of individual VRAM values"
    )
    assert summary["total_model_loads"] == sum(model_loads_list), (
        "Total model loads must equal sum of individual loads"
    )
    
    # Property 6: Average cost must be consistent with individual costs
    total_cost = sum(
        tracker._calculate_cost(latencies[i], vram_values[i], model_loads_list[i])
        for i in range(num_executions)
    )
    expected_avg_cost = total_cost / num_executions
    
    assert abs(summary["avg_cost"] - expected_avg_cost) <= tolerance, (
        f"Average cost must be consistent: "
        f"got={summary['avg_cost']}, expected={expected_avg_cost}"
    )


class CostStatisticsStateMachine(RuleBasedStateMachine):
    """
    Stateful property testing for cost statistics convergence.
    
    This tests that statistics remain consistent as we add more executions.
    """
    
    def __init__(self):
        super().__init__()
        self.tracker = CostTracker()
        self.all_latencies = []
        self.all_vram = []
        self.all_loads = []
        self.execution_count = 0
    
    @rule(
        latency=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        vram=st.integers(min_value=0, max_value=10000),
        loads=st.integers(min_value=0, max_value=10)
    )
    def add_execution(self, latency, vram, loads):
        """Add a new execution and verify statistics."""
        cost_id = self.tracker.start_step(f"step_{self.execution_count}")
        
        # Set metrics manually for testing
        cost = self.tracker._active_costs[cost_id]
        cost.end_time = cost.start_time + (latency / 1000.0)
        cost.latency_ms = latency
        cost.vram_peak_mb = vram
        cost.model_loads = loads
        cost.normalized_cost = self.tracker._calculate_cost(
            cost.latency_ms, cost.vram_peak_mb, cost.model_loads
        )
        
        # Move to completed
        self.tracker._completed_costs[cost_id] = cost
        del self.tracker._active_costs[cost_id]
        
        # Track for verification
        self.all_latencies.append(latency)
        self.all_vram.append(vram)
        self.all_loads.append(loads)
        self.execution_count += 1
    
    @invariant()
    def statistics_are_consistent(self):
        """Verify that statistics are always consistent with tracked data."""
        if self.execution_count == 0:
            return
        
        summary = self.tracker.get_summary()
        
        # Verify execution count
        assert summary["total_steps"] == self.execution_count
        
        # Verify averages
        tolerance = 1e-6
        expected_avg_latency = sum(self.all_latencies) / self.execution_count
        expected_avg_vram = sum(self.all_vram) / self.execution_count
        expected_avg_loads = sum(self.all_loads) / self.execution_count
        
        assert abs(summary["avg_latency_ms"] - expected_avg_latency) <= tolerance
        assert abs(summary["avg_vram_mb"] - expected_avg_vram) <= tolerance
        assert abs(summary["avg_model_loads"] - expected_avg_loads) <= tolerance
        
        # Verify totals
        assert abs(summary["total_latency_ms"] - sum(self.all_latencies)) <= tolerance
        assert summary["total_vram_mb"] == sum(self.all_vram)
        assert summary["total_model_loads"] == sum(self.all_loads)


# Run the stateful test
TestCostStatistics = CostStatisticsStateMachine.TestCase


# ============================================================================
# Additional Property Tests
# ============================================================================

@given(
    num_steps=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=50, deadline=None)
def test_property_cost_tracking_isolation(num_steps):
    """
    Property: Cost Tracking Isolation
    
    Each step's cost tracking should be independent and not affect others.
    """
    tracker = CostTracker()
    cost_ids = []
    step_names = [f"step_{i}" for i in range(num_steps)]
    
    # Start tracking all steps
    for step_name in step_names:
        cost_id = tracker.start_step(step_name)
        cost_ids.append(cost_id)
    
    # All should be in active costs
    assert len(tracker._active_costs) == num_steps
    
    # End tracking in reverse order
    for i in range(num_steps - 1, -1, -1):
        cost = tracker.end_step(cost_ids[i], model_loads=i)
        assert cost.step_name == step_names[i]
        assert cost.model_loads == i
    
    # All should be in completed costs
    assert len(tracker._active_costs) == 0
    assert len(tracker._completed_costs) == num_steps
    
    # Each cost should be retrievable and correct
    for i, cost_id in enumerate(cost_ids):
        cost = tracker.get_cost(cost_id)
        assert cost is not None
        assert cost.step_name == step_names[i]
        assert cost.model_loads == i


@given(
    latency_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    vram_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    model_load_weight=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_property_cost_weights_linearity(latency_weight, vram_weight, model_load_weight):
    """
    Property: Cost Weights Linearity
    
    Doubling all weights should double the cost.
    """
    tracker1 = CostTracker(latency_weight, vram_weight, model_load_weight)
    tracker2 = CostTracker(
        latency_weight * 2,
        vram_weight * 2,
        model_load_weight * 2
    )
    
    # Use fixed metrics
    latency = 100.0
    vram = 500
    loads = 2
    
    cost1 = tracker1._calculate_cost(latency, vram, loads)
    cost2 = tracker2._calculate_cost(latency, vram, loads)
    
    # Cost should double (within floating point tolerance)
    tolerance = 1e-6
    expected_cost2 = cost1 * 2
    
    assert abs(cost2 - expected_cost2) <= tolerance, (
        f"Doubling weights should double cost: "
        f"cost1={cost1}, cost2={cost2}, expected={expected_cost2}"
    )


@given(
    num_clears=st.integers(min_value=1, max_value=5),
    steps_per_clear=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=50, deadline=None)
def test_property_cost_tracker_clear_resets_state(num_clears, steps_per_clear):
    """
    Property: Cost Tracker Clear Resets State
    
    After clearing, the tracker should behave as if newly initialized.
    """
    tracker = CostTracker()
    
    for clear_iteration in range(num_clears):
        # Add some costs
        for i in range(steps_per_clear):
            cost_id = tracker.start_step(f"step_{i}")
            tracker.end_step(cost_id, model_loads=i)
        
        # Verify costs exist
        summary_before = tracker.get_summary()
        assert summary_before["total_steps"] == steps_per_clear
        
        # Clear
        tracker.clear()
        
        # Verify empty state
        summary_after = tracker.get_summary()
        assert summary_after["total_steps"] == 0
        assert summary_after["total_latency_ms"] == 0.0
        assert summary_after["total_vram_mb"] == 0
        assert summary_after["total_model_loads"] == 0
        assert summary_after["total_cost"] == 0.0
        
        assert len(tracker._active_costs) == 0
        assert len(tracker._completed_costs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
