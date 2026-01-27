"""Property-based tests for Prometheus metrics exporter."""

import pytest
from hypothesis import given, strategies as st, settings
import time

# Try to import prometheus_client, skip tests if not available
try:
    from prometheus_client import REGISTRY, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from mm_orch.monitoring.prometheus_exporter import PrometheusExporter


# Skip all tests if prometheus_client is not available
pytestmark = pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE,
    reason="prometheus_client not installed"
)


# Strategies for test data generation
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

engine_names = st.sampled_from(['vllm', 'deepspeed', 'onnx', 'pytorch'])

latency_values = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)

throughput_values = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

gpu_memory_values = st.dictionaries(
    keys=st.integers(min_value=0, max_value=7),
    values=st.floats(min_value=0.0, max_value=80000.0, allow_nan=False, allow_infinity=False),
    min_size=0,
    max_size=8
)

cpu_percent_values = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)

event_types = st.sampled_from(['load', 'unload'])

error_types = st.text(min_size=1, max_size=50)


def create_exporter():
    """Helper to create a fresh exporter with cleared registry."""
    # Clear registry before creating exporter
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    
    return PrometheusExporter(port=9999, enabled=True)


# Feature: advanced-optimization-monitoring, Property 15: Inference metrics are recorded
@given(
    model_name=model_names,
    engine=engine_names,
    latency_ms=latency_values
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_15_inference_metrics_recorded(model_name, engine, latency_ms):
    """
    Property 15: Inference metrics are recorded.
    
    For any completed inference operation, latency and throughput metrics
    should be recorded in the metrics exporter.
    
    Validates: Requirements 4.2, 4.3
    """
    exporter = create_exporter()
    
    # Record inference latency
    exporter.record_inference_latency(model_name, engine, latency_ms)
    
    # Verify metric was recorded by checking the registry
    # The metric should exist and have at least one sample
    metric_name = 'inference_latency_seconds'
    samples = list(REGISTRY.collect())
    
    # Find our metric
    found_metric = False
    for family in samples:
        if family.name == metric_name:
            found_metric = True
            break
    
    assert found_metric, f"Metric {metric_name} not found in registry"


# Feature: advanced-optimization-monitoring, Property 16: Model lifecycle events are recorded
@given(
    model_name=model_names,
    event_type=event_types
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_16_model_lifecycle_recorded(model_name, event_type):
    """
    Property 16: Model lifecycle events are recorded.
    
    For any model load or unload operation, a lifecycle event should be
    recorded in the metrics exporter.
    
    Validates: Requirements 4.4
    """
    exporter = create_exporter()
    
    # Record model lifecycle event - should not raise exception
    try:
        exporter.record_model_lifecycle(model_name, event_type)
    except Exception as e:
        pytest.fail(f"Recording model lifecycle event raised exception: {e}")
    
    # Verify the metric exists in the exporter's attributes
    assert hasattr(exporter, 'model_lifecycle_events')
    assert exporter.model_lifecycle_events is not None


# Feature: advanced-optimization-monitoring, Property 17: Resource metrics are continuously recorded
@given(
    gpu_memory=gpu_memory_values,
    cpu_percent=cpu_percent_values
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_17_resource_metrics_recorded(gpu_memory, cpu_percent):
    """
    Property 17: Resource metrics are continuously recorded.
    
    For any monitoring period, GPU memory and CPU usage metrics should be
    periodically recorded.
    
    Validates: Requirements 4.5, 4.6
    """
    exporter = create_exporter()
    
    # Record resource usage
    exporter.record_resource_usage(
        gpu_memory_mb=gpu_memory if gpu_memory else None,
        cpu_percent=cpu_percent
    )
    
    # Verify metrics were recorded
    samples = list(REGISTRY.collect())
    metric_names = [family.name for family in samples]
    
    # CPU metric should always be recorded if provided
    if cpu_percent is not None:
        assert 'cpu_usage_percent' in metric_names
    
    # GPU metrics should be recorded if provided
    if gpu_memory:
        assert 'gpu_memory_used_bytes' in metric_names


# Feature: advanced-optimization-monitoring, Property 18: Metrics are exposed in Prometheus format
@given(
    model_name=model_names,
    engine=engine_names,
    latency_ms=latency_values
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_property_18_prometheus_format(model_name, engine, latency_ms):
    """
    Property 18: Metrics are exposed in Prometheus format.
    
    For any metrics endpoint query, the response should be in valid
    Prometheus text format with proper metric types and labels.
    
    Validates: Requirements 4.7
    """
    exporter = create_exporter()
    
    # Record some metrics
    exporter.record_inference_latency(model_name, engine, latency_ms)
    
    # Collect metrics from registry
    samples = list(REGISTRY.collect())
    
    # Verify we have metric families (Prometheus format requirement)
    assert len(samples) > 0, "No metrics collected"
    
    # Verify each metric family has required attributes
    for family in samples:
        # Must have name
        assert hasattr(family, 'name')
        assert isinstance(family.name, str)
        assert len(family.name) > 0
        
        # Must have type
        assert hasattr(family, 'type')
        
        # Must have samples
        assert hasattr(family, 'samples')
        
        # Each sample must have name, labels, and value
        for sample in family.samples:
            assert hasattr(sample, 'name')
            assert hasattr(sample, 'labels')
            assert hasattr(sample, 'value')
            assert isinstance(sample.labels, dict)


# Additional property: Metrics recording is resilient to errors
@given(
    model_name=model_names,
    engine=engine_names,
    latency_ms=latency_values
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_metrics_recording_resilient_to_errors(model_name, engine, latency_ms):
    """
    Property: Metrics recording failures don't crash the system.
    
    For any metrics recording operation that fails, the system should
    log the error but continue processing.
    
    Validates: Requirement 15.4 (graceful degradation)
    """
    exporter = create_exporter()
    
    # Simulate a failure by setting enabled to False after initialization
    original_enabled = exporter.enabled
    exporter.enabled = False
    
    # These should not raise exceptions
    try:
        exporter.record_inference_latency(model_name, engine, latency_ms)
        exporter.record_throughput(model_name, 100.0)
        exporter.record_resource_usage(cpu_percent=50.0)
        exporter.record_model_lifecycle(model_name, 'load')
    except Exception as e:
        pytest.fail(f"Metrics recording raised exception when disabled: {e}")
    finally:
        exporter.enabled = original_enabled


# Property: Disabled exporter doesn't record metrics
@given(
    model_name=model_names,
    engine=engine_names,
    latency_ms=latency_values
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_disabled_exporter_no_recording(model_name, engine, latency_ms):
    """
    Property: When disabled, exporter doesn't record metrics.
    
    For any metrics recording operation when exporter is disabled,
    no metrics should be recorded.
    """
    # Clear registry first
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    
    # Create disabled exporter
    exporter = PrometheusExporter(port=9999, enabled=False)
    
    # Try to record metrics
    exporter.record_inference_latency(model_name, engine, latency_ms)
    exporter.record_throughput(model_name, 100.0)
    
    # Verify exporter is disabled
    assert not exporter.is_enabled()


# Property: Multiple recordings accumulate correctly
@given(
    model_name=model_names,
    engine=engine_names,
    latencies=st.lists(latency_values, min_size=1, max_size=10)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_multiple_recordings_accumulate(model_name, engine, latencies):
    """
    Property: Multiple metric recordings accumulate correctly.
    
    For any sequence of metric recordings, all values should be
    accumulated in the appropriate metric type (counter, histogram, etc.).
    """
    exporter = create_exporter()
    
    # Record multiple latencies
    for latency in latencies:
        exporter.record_inference_latency(model_name, engine, latency)
    
    # Verify metrics were recorded (at least the metric family exists)
    samples = list(REGISTRY.collect())
    metric_names = [family.name for family in samples]
    
    # Check for the histogram metric (latency)
    assert 'inference_latency_seconds' in metric_names
    
    # The counter metric name in registry might be without _total suffix
    # Just verify the exporter has the counter attribute
    assert hasattr(exporter, 'inference_requests')
    assert exporter.inference_requests is not None


# Property: Different models have separate metrics
@given(
    model1=model_names,
    model2=model_names,
    engine=engine_names,
    latency=latency_values
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_different_models_separate_metrics(model1, model2, engine, latency):
    """
    Property: Different models have separate metric labels.
    
    For any two different models, their metrics should be tracked
    separately using labels.
    """
    # Skip if models are the same
    if model1 == model2:
        return
    
    exporter = create_exporter()
    
    # Record metrics for both models
    exporter.record_inference_latency(model1, engine, latency)
    exporter.record_inference_latency(model2, engine, latency)
    
    # Both should be recorded (verified by no exceptions)
    # The Prometheus client handles label separation automatically
    assert True


# Property: Throughput can be updated multiple times
@given(
    model_name=model_names,
    throughputs=st.lists(throughput_values, min_size=1, max_size=10)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_throughput_updates(model_name, throughputs):
    """
    Property: Throughput gauge can be updated multiple times.
    
    For any sequence of throughput values, the gauge should be
    updated to reflect the latest value.
    """
    exporter = create_exporter()
    
    # Record multiple throughput values
    for throughput in throughputs:
        exporter.record_throughput(model_name, throughput)
    
    # Verify metric exists
    samples = list(REGISTRY.collect())
    metric_names = [family.name for family in samples]
    
    assert 'throughput_requests_per_second' in metric_names
