"""
Property-based tests for multi-GPU support.

Tests Properties 40-44 from the design document:
- Property 40: Tensor parallelism is supported on multi-GPU
- Property 41: Pipeline parallelism is supported on multi-GPU
- Property 42: Load is balanced across GPUs
- Property 43: GPU failures are detected
- Property 44: Per-GPU metrics are exposed

Feature: advanced-optimization-monitoring
Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import torch

from mm_orch.optimization.gpu_utils import GPUManager, GPUInfo
from mm_orch.optimization.config import VLLMConfig, DeepSpeedConfig
from mm_orch.optimization.vllm_engine import VLLMEngine
from mm_orch.optimization.deepspeed_engine import DeepSpeedEngine
from mm_orch.monitoring.prometheus_exporter import PrometheusExporter


# Feature: advanced-optimization-monitoring, Property 40: Tensor parallelism is supported on multi-GPU
@given(
    num_gpus=st.integers(min_value=2, max_value=8),
    tensor_parallel=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_tensor_parallelism_supported(num_gpus, tensor_parallel):
    """
    Property 40: Tensor parallelism is supported on multi-GPU.
    
    For any multi-GPU system, tensor parallelism should be available
    for distributing model layers across GPUs.
    
    Validates: Requirements 11.1
    """
    # Assume we have enough GPUs for the requested parallelism
    assume(num_gpus >= tensor_parallel)
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=num_gpus), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.memory_allocated', return_value=0), \
         patch('torch.cuda.memory_reserved', return_value=0):
        
        # Setup mock GPU properties
        mock_prop = Mock()
        mock_prop.name = "Mock GPU"
        mock_prop.total_memory = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = mock_prop
        
        # Create GPU manager
        manager = GPUManager()
        
        # Allocate GPUs for tensor parallelism
        try:
            gpu_ids, strategy = manager.allocate_gpus(
                tensor_parallel=tensor_parallel,
                pipeline_parallel=1
            )
            
            # Verify allocation
            assert len(gpu_ids) == tensor_parallel, \
                f"Expected {tensor_parallel} GPUs, got {len(gpu_ids)}"
            assert "tensor_parallel" in strategy.lower(), \
                f"Strategy should mention tensor parallelism: {strategy}"
            assert all(0 <= gpu_id < num_gpus for gpu_id in gpu_ids), \
                f"GPU IDs should be valid: {gpu_ids}"
            
        except RuntimeError as e:
            # Should not fail if we have enough GPUs
            pytest.fail(f"Tensor parallelism allocation failed: {e}")


# Feature: advanced-optimization-monitoring, Property 41: Pipeline parallelism is supported on multi-GPU
@given(
    num_gpus=st.integers(min_value=2, max_value=8),
    pipeline_parallel=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_pipeline_parallelism_supported(num_gpus, pipeline_parallel):
    """
    Property 41: Pipeline parallelism is supported on multi-GPU.
    
    For any multi-GPU system, pipeline parallelism should be available
    for distributing model stages across GPUs.
    
    Validates: Requirements 11.2
    """
    # Assume we have enough GPUs for the requested parallelism
    assume(num_gpus >= pipeline_parallel)
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=num_gpus), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.memory_allocated', return_value=0), \
         patch('torch.cuda.memory_reserved', return_value=0):
        
        # Setup mock GPU properties
        mock_prop = Mock()
        mock_prop.name = "Mock GPU"
        mock_prop.total_memory = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = mock_prop
        
        # Create GPU manager
        manager = GPUManager()
        
        # Allocate GPUs for pipeline parallelism
        try:
            gpu_ids, strategy = manager.allocate_gpus(
                tensor_parallel=1,
                pipeline_parallel=pipeline_parallel
            )
            
            # Verify allocation
            assert len(gpu_ids) == pipeline_parallel, \
                f"Expected {pipeline_parallel} GPUs, got {len(gpu_ids)}"
            assert "pipeline" in strategy.lower(), \
                f"Strategy should mention pipeline parallelism: {strategy}"
            assert all(0 <= gpu_id < num_gpus for gpu_id in gpu_ids), \
                f"GPU IDs should be valid: {gpu_ids}"
            
        except RuntimeError as e:
            # Should not fail if we have enough GPUs
            pytest.fail(f"Pipeline parallelism allocation failed: {e}")


# Feature: advanced-optimization-monitoring, Property 42: Load is balanced across GPUs
@given(
    num_gpus=st.integers(min_value=4, max_value=8),
    tensor_parallel=st.integers(min_value=2, max_value=2),
    pipeline_parallel=st.integers(min_value=2, max_value=4)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_load_balanced_across_gpus(num_gpus, tensor_parallel, pipeline_parallel):
    """
    Property 42: Load is balanced across GPUs.
    
    For any model distributed across multiple GPUs, computational load
    should be balanced to maximize utilization.
    
    Validates: Requirements 11.3
    """
    total_gpus_needed = tensor_parallel * pipeline_parallel
    assume(num_gpus >= total_gpus_needed)
    
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=num_gpus), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.memory_allocated', return_value=0), \
         patch('torch.cuda.memory_reserved', return_value=0):
        
        # Setup mock GPU properties
        mock_prop = Mock()
        mock_prop.name = "Mock GPU"
        mock_prop.total_memory = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = mock_prop
        
        # Create GPU manager
        manager = GPUManager()
        
        # Get balanced load allocation
        allocation = manager.balance_load(
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel
        )
        
        # Verify balanced allocation
        assert len(allocation) == pipeline_parallel, \
            f"Expected {pipeline_parallel} pipeline stages, got {len(allocation)}"
        
        # Each stage should have tensor_parallel GPUs
        for stage_idx, stage_gpus in enumerate(allocation):
            assert len(stage_gpus) == tensor_parallel, \
                f"Stage {stage_idx} should have {tensor_parallel} GPUs, got {len(stage_gpus)}"
        
        # All GPUs should be unique
        all_gpus = [gpu for stage in allocation for gpu in stage]
        assert len(all_gpus) == len(set(all_gpus)), \
            "GPUs should not be duplicated across stages"
        
        # GPUs should be distributed evenly (no stage should have significantly more load)
        # In this case, all stages have the same number of GPUs, so load is balanced


# Feature: advanced-optimization-monitoring, Property 43: GPU failures are detected
@given(
    num_gpus=st.integers(min_value=2, max_value=4),
    failed_gpu_id=st.integers(min_value=0, max_value=3)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_gpu_failures_detected(num_gpus, failed_gpu_id):
    """
    Property 43: GPU failures are detected.
    
    For any GPU failure during inference, the failure should be
    detected and logged.
    
    Validates: Requirements 11.4
    """
    assume(failed_gpu_id < num_gpus)
    
    # Mock GPU detection with one failed GPU
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=num_gpus), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.set_device') as mock_set_device, \
         patch('torch.cuda.memory_allocated', return_value=0), \
         patch('torch.cuda.memory_reserved', return_value=0), \
         patch('torch.zeros') as mock_zeros:
        
        # Setup mock GPU properties
        mock_prop = Mock()
        mock_prop.name = "Mock GPU"
        mock_prop.total_memory = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = mock_prop
        
        # Make the failed GPU raise an error
        def set_device_side_effect(device_id):
            if device_id == failed_gpu_id:
                raise RuntimeError(f"GPU {device_id} failed")
        
        mock_set_device.side_effect = set_device_side_effect
        
        # Create GPU manager
        manager = GPUManager()
        
        # Detect GPU failure
        has_failed, failure_reason = manager.detect_gpu_failure(failed_gpu_id)
        
        # Verify failure was detected
        assert has_failed, f"GPU {failed_gpu_id} failure should be detected"
        assert failure_reason is not None, "Failure reason should be provided"
        assert "failed" in failure_reason.lower() or "error" in failure_reason.lower(), \
            f"Failure reason should indicate error: {failure_reason}"
        
        # Verify failed GPU is tracked
        failed_gpus = manager.get_failed_gpus()
        assert failed_gpu_id in failed_gpus, \
            f"Failed GPU {failed_gpu_id} should be in failed list: {failed_gpus}"


# Feature: advanced-optimization-monitoring, Property 44: Per-GPU metrics are exposed
@given(
    num_gpus=st.integers(min_value=1, max_value=4),
    memory_used_mb=st.floats(min_value=100, max_value=15000),
    utilization_percent=st.floats(min_value=0, max_value=100)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_per_gpu_metrics_exposed(num_gpus, memory_used_mb, utilization_percent):
    """
    Property 44: Per-GPU metrics are exposed.
    
    For any GPU in the system, individual metrics (memory, utilization)
    should be exposed for monitoring.
    
    Validates: Requirements 11.5
    """
    # Mock GPU detection
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.device_count', return_value=num_gpus), \
         patch('torch.cuda.get_device_properties') as mock_props, \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.memory_allocated') as mock_allocated, \
         patch('torch.cuda.memory_reserved') as mock_reserved:
        
        # Setup mock GPU properties
        mock_prop = Mock()
        mock_prop.name = "Mock GPU"
        mock_prop.total_memory = 16 * 1024 ** 3  # 16GB
        mock_props.return_value = mock_prop
        
        # Setup memory usage
        mock_allocated.return_value = memory_used_mb * 1024 ** 2
        mock_reserved.return_value = memory_used_mb * 1024 ** 2
        
        # Create GPU manager
        manager = GPUManager()
        
        # Get metrics for all GPUs
        all_metrics = manager.get_all_gpu_metrics()
        
        # Verify metrics are exposed for all GPUs
        assert len(all_metrics) == num_gpus, \
            f"Expected metrics for {num_gpus} GPUs, got {len(all_metrics)}"
        
        # Verify each GPU has required metrics
        for gpu_id in range(num_gpus):
            assert gpu_id in all_metrics, f"Metrics missing for GPU {gpu_id}"
            
            metrics = all_metrics[gpu_id]
            
            # Check required metric fields
            assert 'memory_used_mb' in metrics, \
                f"GPU {gpu_id} metrics missing memory_used_mb"
            assert 'memory_available_mb' in metrics, \
                f"GPU {gpu_id} metrics missing memory_available_mb"
            assert 'memory_total_mb' in metrics, \
                f"GPU {gpu_id} metrics missing memory_total_mb"
            assert 'utilization_percent' in metrics, \
                f"GPU {gpu_id} metrics missing utilization_percent"
            
            # Verify metric values are reasonable
            assert metrics['memory_used_mb'] >= 0, \
                f"Memory used should be non-negative: {metrics['memory_used_mb']}"
            assert metrics['memory_available_mb'] >= 0, \
                f"Memory available should be non-negative: {metrics['memory_available_mb']}"
            assert metrics['memory_total_mb'] > 0, \
                f"Total memory should be positive: {metrics['memory_total_mb']}"
            assert 0 <= metrics['utilization_percent'] <= 100, \
                f"Utilization should be 0-100: {metrics['utilization_percent']}"


# Additional test: Verify Prometheus integration for per-GPU metrics
@given(
    num_gpus=st.integers(min_value=1, max_value=4)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_per_gpu_metrics_prometheus_integration(num_gpus):
    """
    Verify that per-GPU metrics can be recorded to Prometheus.
    
    This test ensures the integration between GPU monitoring and
    Prometheus metrics export works correctly.
    
    Validates: Requirements 11.5
    """
    # Mock Prometheus client completely to avoid registry conflicts
    with patch('mm_orch.monitoring.prometheus_exporter.PROMETHEUS_AVAILABLE', True), \
         patch('mm_orch.monitoring.prometheus_exporter.Histogram') as mock_histogram, \
         patch('mm_orch.monitoring.prometheus_exporter.Counter') as mock_counter, \
         patch('mm_orch.monitoring.prometheus_exporter.Gauge') as mock_gauge:
        
        # Create mock metric instances
        mock_histogram_instance = Mock()
        mock_counter_instance = Mock()
        mock_gauge_instance = Mock()
        
        mock_histogram.return_value = mock_histogram_instance
        mock_counter.return_value = mock_counter_instance
        mock_gauge.return_value = mock_gauge_instance
        
        # Setup labels mock
        mock_labels = Mock()
        mock_gauge_instance.labels.return_value = mock_labels
        
        # Create Prometheus exporter
        exporter = PrometheusExporter(port=9090, enabled=True)
        
        # Record per-GPU metrics
        for gpu_id in range(num_gpus):
            exporter.record_per_gpu_metrics(
                gpu_id=gpu_id,
                memory_used_mb=8192.0,
                memory_available_mb=7808.0,
                utilization_percent=75.5,
                temperature_celsius=65.0,
                is_healthy=True
            )
        
        # Verify metrics were recorded (labels method should be called)
        # At least once per GPU for each metric type (5 metrics per GPU)
        # memory_used, memory_available, utilization, temperature, health_status
        expected_min_calls = num_gpus * 5
        assert mock_gauge_instance.labels.call_count >= expected_min_calls, \
            f"Expected at least {expected_min_calls} label calls for {num_gpus} GPUs, got {mock_gauge_instance.labels.call_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
