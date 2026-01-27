"""
Unit tests for cost tracking functionality.
"""

import time
import pytest
from unittest.mock import Mock, patch
from mm_orch.orchestration.cost_tracker import CostTracker, StepCost


class TestStepCost:
    """Test StepCost dataclass."""
    
    def test_step_cost_creation(self):
        """Test creating a StepCost instance."""
        cost = StepCost(
            step_name="test_step",
            latency_ms=100.0,
            vram_peak_mb=500,
            model_loads=2,
            normalized_cost=25.1
        )
        
        assert cost.step_name == "test_step"
        assert cost.latency_ms == 100.0
        assert cost.vram_peak_mb == 500
        assert cost.model_loads == 2
        assert cost.normalized_cost == 25.1
    
    def test_step_cost_to_dict(self):
        """Test converting StepCost to dictionary."""
        cost = StepCost(
            step_name="test_step",
            latency_ms=100.0,
            vram_peak_mb=500,
            model_loads=2,
            normalized_cost=25.1,
            start_time=1000.0,
            end_time=1000.1
        )
        
        result = cost.to_dict()
        
        assert result["step_name"] == "test_step"
        assert result["latency_ms"] == 100.0
        assert result["vram_peak_mb"] == 500
        assert result["model_loads"] == 2
        assert result["normalized_cost"] == 25.1
        assert result["start_time"] == 1000.0
        assert result["end_time"] == 1000.1


class TestCostTracker:
    """Test CostTracker functionality."""
    
    def test_initialization_default_weights(self):
        """Test CostTracker initialization with default weights."""
        tracker = CostTracker()
        
        assert tracker.latency_weight == CostTracker.DEFAULT_LATENCY_WEIGHT
        assert tracker.vram_weight == CostTracker.DEFAULT_VRAM_WEIGHT
        assert tracker.model_load_weight == CostTracker.DEFAULT_MODEL_LOAD_WEIGHT
    
    def test_initialization_custom_weights(self):
        """Test CostTracker initialization with custom weights."""
        tracker = CostTracker(
            latency_weight=0.002,
            vram_weight=0.02,
            model_load_weight=20.0
        )
        
        assert tracker.latency_weight == 0.002
        assert tracker.vram_weight == 0.02
        assert tracker.model_load_weight == 20.0
    
    def test_initialization_negative_weights(self):
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            CostTracker(latency_weight=-0.1)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            CostTracker(vram_weight=-0.1)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            CostTracker(model_load_weight=-1.0)
    
    def test_start_step(self):
        """Test starting cost tracking for a step."""
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        
        assert cost_id.startswith("test_step_")
        assert cost_id in tracker._active_costs
        
        cost = tracker._active_costs[cost_id]
        assert cost.step_name == "test_step"
        assert cost.start_time > 0
    
    def test_end_step_basic(self):
        """Test ending cost tracking for a step."""
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        time.sleep(0.01)  # Small delay to ensure measurable latency
        
        cost = tracker.end_step(cost_id, model_loads=1)
        
        assert cost.step_name == "test_step"
        assert cost.latency_ms > 0
        assert cost.model_loads == 1
        assert cost.normalized_cost > 0
        assert cost.end_time > cost.start_time
        
        # Should be moved to completed costs
        assert cost_id not in tracker._active_costs
        assert cost_id in tracker._completed_costs
    
    def test_end_step_nonexistent_id(self):
        """Test ending cost tracking with invalid ID."""
        tracker = CostTracker()
        
        with pytest.raises(KeyError, match="not found in active costs"):
            tracker.end_step("nonexistent_id")
    
    def test_cost_calculation(self):
        """Test normalized cost calculation."""
        tracker = CostTracker(
            latency_weight=0.001,
            vram_weight=0.01,
            model_load_weight=10.0
        )
        
        # Test calculation: 0.001*100 + 0.01*500 + 10.0*2 = 0.1 + 5.0 + 20.0 = 25.1
        cost = tracker._calculate_cost(
            latency_ms=100.0,
            vram_peak_mb=500,
            model_loads=2
        )
        
        assert abs(cost - 25.1) < 0.001
    
    def test_cost_calculation_zero_values(self):
        """Test cost calculation with zero values."""
        tracker = CostTracker()
        
        cost = tracker._calculate_cost(
            latency_ms=0.0,
            vram_peak_mb=0,
            model_loads=0
        )
        
        assert cost == 0.0
    
    def test_get_cost(self):
        """Test retrieving completed cost."""
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        tracker.end_step(cost_id, model_loads=0)
        
        retrieved_cost = tracker.get_cost(cost_id)
        
        assert retrieved_cost is not None
        assert retrieved_cost.step_name == "test_step"
    
    def test_get_cost_nonexistent(self):
        """Test retrieving nonexistent cost returns None."""
        tracker = CostTracker()
        
        cost = tracker.get_cost("nonexistent_id")
        
        assert cost is None
    
    def test_get_all_costs(self):
        """Test retrieving all completed costs."""
        tracker = CostTracker()
        
        cost_id1 = tracker.start_step("step1")
        cost_id2 = tracker.start_step("step2")
        
        tracker.end_step(cost_id1, model_loads=0)
        tracker.end_step(cost_id2, model_loads=1)
        
        all_costs = tracker.get_all_costs()
        
        assert len(all_costs) == 2
        assert cost_id1 in all_costs
        assert cost_id2 in all_costs
    
    def test_clear(self):
        """Test clearing all cost records."""
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        tracker.end_step(cost_id, model_loads=0)
        
        tracker.clear()
        
        assert len(tracker._active_costs) == 0
        assert len(tracker._completed_costs) == 0
    
    def test_get_summary_empty(self):
        """Test getting summary with no costs."""
        tracker = CostTracker()
        
        summary = tracker.get_summary()
        
        assert summary["total_steps"] == 0
        assert summary["total_latency_ms"] == 0.0
        assert summary["total_vram_mb"] == 0
        assert summary["total_model_loads"] == 0
        assert summary["total_cost"] == 0.0
        assert summary["avg_latency_ms"] == 0.0
        assert summary["avg_vram_mb"] == 0.0
        assert summary["avg_model_loads"] == 0.0
        assert summary["avg_cost"] == 0.0
    
    def test_get_summary_with_costs(self):
        """Test getting summary with multiple costs."""
        tracker = CostTracker()
        
        # Add multiple steps
        for i in range(3):
            cost_id = tracker.start_step(f"step{i}")
            time.sleep(0.01)
            tracker.end_step(cost_id, model_loads=i)
        
        summary = tracker.get_summary()
        
        assert summary["total_steps"] == 3
        assert summary["total_latency_ms"] > 0
        assert summary["total_model_loads"] == 3  # 0 + 1 + 2
        assert summary["avg_latency_ms"] > 0
        assert summary["avg_model_loads"] == 1.0
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated")
    def test_vram_tracking_with_cuda(self, mock_memory, mock_cuda):
        """Test VRAM tracking when CUDA is available."""
        # Simulate VRAM usage: 100MB at start, 200MB at end
        mock_memory.side_effect = [
            100 * 1024 * 1024,  # start_step
            200 * 1024 * 1024   # end_step
        ]
        
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        cost = tracker.end_step(cost_id, model_loads=0)
        
        assert cost.vram_peak_mb == 200
    
    @patch("torch.cuda.is_available", return_value=False)
    def test_vram_tracking_without_cuda(self, mock_cuda):
        """Test VRAM tracking when CUDA is not available."""
        tracker = CostTracker()
        
        cost_id = tracker.start_step("test_step")
        cost = tracker.end_step(cost_id, model_loads=0)
        
        assert cost.vram_peak_mb == 0
    
    def test_multiple_concurrent_steps(self):
        """Test tracking multiple steps concurrently."""
        tracker = CostTracker()
        
        # Start multiple steps
        cost_id1 = tracker.start_step("step1")
        cost_id2 = tracker.start_step("step2")
        cost_id3 = tracker.start_step("step3")
        
        assert len(tracker._active_costs) == 3
        
        # End them in different order
        tracker.end_step(cost_id2, model_loads=0)
        tracker.end_step(cost_id1, model_loads=1)
        tracker.end_step(cost_id3, model_loads=2)
        
        assert len(tracker._active_costs) == 0
        assert len(tracker._completed_costs) == 3
        
        # Verify each cost
        cost1 = tracker.get_cost(cost_id1)
        cost2 = tracker.get_cost(cost_id2)
        cost3 = tracker.get_cost(cost_id3)
        
        assert cost1.step_name == "step1"
        assert cost2.step_name == "step2"
        assert cost3.step_name == "step3"
        
        assert cost1.model_loads == 1
        assert cost2.model_loads == 0
        assert cost3.model_loads == 2


class TestCostTrackerIntegration:
    """Integration tests for cost tracking."""
    
    def test_realistic_workflow_costs(self):
        """Test cost tracking for a realistic workflow."""
        tracker = CostTracker()
        
        # Simulate a search workflow
        steps = [
            ("web_search", 0),      # No model loads
            ("fetch_urls", 0),      # No model loads
            ("summarize", 2),       # 2 model loads
            ("generate_answer", 1)  # 1 model load
        ]
        
        for step_name, model_loads in steps:
            cost_id = tracker.start_step(step_name)
            time.sleep(0.01)  # Simulate work
            tracker.end_step(cost_id, model_loads=model_loads)
        
        summary = tracker.get_summary()
        
        assert summary["total_steps"] == 4
        assert summary["total_model_loads"] == 3
        assert summary["avg_model_loads"] == 0.75
        assert summary["total_cost"] > 0
    
    def test_cost_weights_impact(self):
        """Test that different weights produce different costs."""
        # High latency weight
        tracker1 = CostTracker(latency_weight=1.0, vram_weight=0.0, model_load_weight=0.0)
        cost1 = tracker1._calculate_cost(100.0, 500, 2)
        
        # High VRAM weight
        tracker2 = CostTracker(latency_weight=0.0, vram_weight=1.0, model_load_weight=0.0)
        cost2 = tracker2._calculate_cost(100.0, 500, 2)
        
        # High model load weight
        tracker3 = CostTracker(latency_weight=0.0, vram_weight=0.0, model_load_weight=1.0)
        cost3 = tracker3._calculate_cost(100.0, 500, 2)
        
        # All should be different
        assert cost1 == 100.0
        assert cost2 == 500.0
        assert cost3 == 2.0
