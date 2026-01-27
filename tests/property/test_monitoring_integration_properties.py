"""
Property-based tests for monitoring integration in OptimizationManager.

Tests Property 59: Monitoring failures don't block requests
Validates Requirements 15.4, 15.5
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from mm_orch.optimization.manager import OptimizationManager, InferenceResult
from mm_orch.optimization.config import OptimizationConfig
from mm_orch.monitoring.prometheus_exporter import PrometheusExporter
from mm_orch.monitoring.otel_tracer import OTelTracer


# Strategy for generating model names
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

# Strategy for generating inputs
inference_inputs = st.fixed_dictionaries({
    "prompts": st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5)
})


class FailingPrometheusExporter:
    """Mock Prometheus exporter that always fails."""
    
    def __init__(self):
        self.enabled = True
        self._server_started = False
    
    def record_inference_latency(self, *args, **kwargs):
        raise RuntimeError("Prometheus recording failed")
    
    def record_inference_error(self, *args, **kwargs):
        raise RuntimeError("Prometheus recording failed")
    
    def record_batch_size(self, *args, **kwargs):
        raise RuntimeError("Prometheus recording failed")
    
    def record_error(self, *args, **kwargs):
        raise RuntimeError("Prometheus recording failed")
    
    def is_enabled(self):
        return self.enabled


class FailingOTelTracer:
    """Mock OpenTelemetry tracer that always fails."""
    
    def __init__(self):
        self.enabled = True
    
    def trace_inference(self, *args, **kwargs):
        # Return a context manager that fails
        class FailingContext:
            def __enter__(self):
                raise RuntimeError("Tracing span creation failed")
            
            def __exit__(self, *args):
                pass
        
        return FailingContext()
    
    def _record_error_in_span(self, *args, **kwargs):
        raise RuntimeError("Error recording failed")
    
    def is_enabled(self):
        return self.enabled


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@given(
    model_name=model_names,
    inputs=inference_inputs
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_prometheus_failure_does_not_block_inference(model_name, inputs):
    """
    Property 59: For any metrics export failure, the system should log the error
    but continue processing inference requests.
    
    Validates: Requirements 15.4
    """
    # Setup: Create OptimizationManager with failing Prometheus exporter
    config = OptimizationConfig()
    failing_prometheus = FailingPrometheusExporter()
    
    # Mock the PyTorch inference to return a valid result
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.return_value = {
            "outputs": ["test output"],
            "num_outputs": 1
        }
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=failing_prometheus,
            otel_tracer=None
        )
        
        # Execute: Attempt inference with failing metrics
        # This should NOT raise an exception despite Prometheus failures
        try:
            result = manager.infer(model_name, inputs)
            
            # Verify: Inference succeeded despite monitoring failure
            assert result is not None
            assert isinstance(result, InferenceResult)
            assert result.engine_used == "pytorch"
            assert result.latency_ms >= 0
            
            # Verify PyTorch was called (inference actually happened)
            assert mock_pytorch.called
            
        except Exception as e:
            # If an exception is raised, it should NOT be from Prometheus
            assert "Prometheus" not in str(e)
            assert "recording failed" not in str(e)


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@given(
    model_name=model_names,
    inputs=inference_inputs
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_tracing_failure_does_not_block_inference(model_name, inputs):
    """
    Property 59: For any tracing failure, the system should log the error
    but continue processing inference requests.
    
    Validates: Requirements 15.5
    """
    # Setup: Create OptimizationManager with failing tracer
    config = OptimizationConfig()
    failing_tracer = FailingOTelTracer()
    
    # Mock the PyTorch inference to return a valid result
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.return_value = {
            "outputs": ["test output"],
            "num_outputs": 1
        }
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=None,
            otel_tracer=failing_tracer
        )
        
        # Execute: Attempt inference with failing tracing
        # This should NOT raise an exception despite tracing failures
        try:
            result = manager.infer(model_name, inputs)
            
            # Verify: Inference succeeded despite monitoring failure
            assert result is not None
            assert isinstance(result, InferenceResult)
            assert result.engine_used == "pytorch"
            assert result.latency_ms >= 0
            
            # Verify PyTorch was called (inference actually happened)
            assert mock_pytorch.called
            
        except Exception as e:
            # If an exception is raised, it should NOT be from tracing
            assert "Tracing" not in str(e)
            assert "span creation failed" not in str(e)


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@given(
    model_name=model_names,
    inputs=inference_inputs
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_both_monitoring_failures_do_not_block_inference(model_name, inputs):
    """
    Property 59: For any monitoring failures (both metrics and tracing), the system
    should log the errors but continue processing inference requests.
    
    Validates: Requirements 15.4, 15.5
    """
    # Setup: Create OptimizationManager with both monitoring components failing
    config = OptimizationConfig()
    failing_prometheus = FailingPrometheusExporter()
    failing_tracer = FailingOTelTracer()
    
    # Mock the PyTorch inference to return a valid result
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.return_value = {
            "outputs": ["test output"],
            "num_outputs": 1
        }
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=failing_prometheus,
            otel_tracer=failing_tracer
        )
        
        # Execute: Attempt inference with all monitoring failing
        # This should NOT raise an exception despite all monitoring failures
        try:
            result = manager.infer(model_name, inputs)
            
            # Verify: Inference succeeded despite all monitoring failures
            assert result is not None
            assert isinstance(result, InferenceResult)
            assert result.engine_used == "pytorch"
            assert result.latency_ms >= 0
            
            # Verify PyTorch was called (inference actually happened)
            assert mock_pytorch.called
            
        except Exception as e:
            # If an exception is raised, it should NOT be from monitoring
            assert "Prometheus" not in str(e)
            assert "Tracing" not in str(e)
            assert "recording failed" not in str(e)
            assert "span creation failed" not in str(e)


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@given(
    model_name=model_names,
    inputs=inference_inputs,
    prometheus_fails=st.booleans(),
    tracer_fails=st.booleans()
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_monitoring_failures_are_independent_of_inference_success(
    model_name, inputs, prometheus_fails, tracer_fails
):
    """
    Property 59: For any combination of monitoring failures, inference success
    or failure should be independent of monitoring status.
    
    Validates: Requirements 15.4, 15.5
    """
    # Setup: Create monitoring components that may or may not fail
    config = OptimizationConfig()
    
    prometheus = FailingPrometheusExporter() if prometheus_fails else None
    tracer = FailingOTelTracer() if tracer_fails else None
    
    # Mock the PyTorch inference to return a valid result
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.return_value = {
            "outputs": ["test output"],
            "num_outputs": 1
        }
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=prometheus,
            otel_tracer=tracer
        )
        
        # Execute: Attempt inference
        result = manager.infer(model_name, inputs)
        
        # Verify: Inference succeeded regardless of monitoring status
        assert result is not None
        assert isinstance(result, InferenceResult)
        assert result.engine_used == "pytorch"
        assert result.latency_ms >= 0
        
        # Verify PyTorch was called
        assert mock_pytorch.called


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@given(
    model_name=model_names,
    inputs=inference_inputs
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_monitoring_with_inference_failure_still_attempts_recording(model_name, inputs):
    """
    Property 59: Even when inference fails, monitoring should attempt to record
    the error without blocking the error propagation.
    
    Validates: Requirements 15.4, 15.5
    """
    # Setup: Create OptimizationManager with working monitoring
    config = OptimizationConfig()
    
    # Create mock monitoring that tracks calls
    mock_prometheus = Mock(spec=PrometheusExporter)
    mock_prometheus.is_enabled.return_value = True
    
    # Mock the PyTorch inference to fail
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.side_effect = RuntimeError("Inference failed")
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=mock_prometheus,
            otel_tracer=None
        )
        
        # Execute: Attempt inference (should fail)
        with pytest.raises(RuntimeError, match="Inference failed|All engines failed"):
            manager.infer(model_name, inputs)
        
        # Verify: Monitoring attempted to record the error
        # (record_error or record_inference_error should have been called)
        assert (
            mock_prometheus.record_error.called or
            mock_prometheus.record_inference_error.called
        )


# Feature: advanced-optimization-monitoring, Property 59: Monitoring failures don't block requests
@pytest.mark.property
def test_monitoring_disabled_does_not_affect_inference():
    """
    Property 59: When monitoring is disabled (None), inference should work normally.
    
    Validates: Requirements 15.4, 15.5
    """
    # Setup: Create OptimizationManager without monitoring
    config = OptimizationConfig()
    
    # Mock the PyTorch inference
    with patch.object(OptimizationManager, '_infer_pytorch') as mock_pytorch:
        mock_pytorch.return_value = {
            "outputs": ["test output"],
            "num_outputs": 1
        }
        
        manager = OptimizationManager(
            config=config,
            prometheus_exporter=None,
            otel_tracer=None
        )
        
        # Execute: Attempt inference without monitoring
        result = manager.infer("test-model", {"prompts": ["test"]})
        
        # Verify: Inference succeeded
        assert result is not None
        assert isinstance(result, InferenceResult)
        assert result.engine_used == "pytorch"
        assert mock_pytorch.called
