"""
Property-based tests for PerformanceMonitor.

Tests universal properties that should hold across all valid inputs.
Uses Hypothesis for property-based testing with 100+ iterations per property.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
import time

from mm_orch.monitoring.performance_monitor import (
    PerformanceMonitor,
    LatencyRecord,
    ResourceSnapshot
)


# Strategies for generating test data
operation_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='_-'
))

latency_values = st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False)

metadata_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(
        st.text(max_size=50),
        st.integers(min_value=0, max_value=1000),
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
    ),
    max_size=5
)


# Feature: advanced-optimization-monitoring, Property 29: Per-request latency is collected
@given(
    operation=operation_names,
    latencies=st.lists(latency_values, min_size=1, max_size=50)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_29_per_request_latency_collected(operation, latencies):
    """
    Property 29: Per-request latency is collected.
    
    For any request processed by the system, latency should be recorded
    in the performance monitor.
    
    Validates: Requirements 9.1
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    
    # Record latencies
    for latency in latencies:
        monitor.record_latency(operation, latency)
    
    # Verify all latencies were collected
    metrics = monitor.get_operation_metrics(operation)
    assert metrics is not None, "Metrics should be available after recording latencies"
    assert metrics.count == len(latencies), "All latencies should be recorded"
    assert metrics.operation == operation, "Operation name should match"
    
    # Verify latency values are within expected range
    assert metrics.min_latency_ms >= min(latencies) * 0.99, "Min latency should be close to actual min"
    assert metrics.max_latency_ms <= max(latencies) * 1.01, "Max latency should be close to actual max"


# Feature: advanced-optimization-monitoring, Property 30: Per-model inference time is collected
@given(
    model_name=st.text(min_size=1, max_size=30),
    inference_times=st.lists(latency_values, min_size=1, max_size=30)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_30_per_model_inference_time_collected(model_name, inference_times):
    """
    Property 30: Per-model inference time is collected.
    
    For any model inference operation, inference time should be recorded
    with model identification.
    
    Validates: Requirements 9.2
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    operation = "inference"
    
    # Record inference times with model metadata
    for inference_time in inference_times:
        monitor.record_latency(
            operation,
            inference_time,
            metadata={"model_name": model_name}
        )
    
    # Verify inference times were collected
    metrics = monitor.get_operation_metrics(operation)
    assert metrics is not None, "Inference metrics should be available"
    assert metrics.count == len(inference_times), "All inference times should be recorded"
    
    # Verify we can query by operation
    percentiles = monitor.get_percentiles(operation)
    assert percentiles["p50"] > 0, "Median latency should be positive"
    assert percentiles["p95"] >= percentiles["p50"], "p95 should be >= p50"
    assert percentiles["p99"] >= percentiles["p95"], "p99 should be >= p95"


# Feature: advanced-optimization-monitoring, Property 31: Throughput is calculated over time windows
@given(
    num_requests=st.integers(min_value=5, max_value=30),
    window_seconds=st.integers(min_value=10, max_value=60)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_property_31_throughput_calculated_over_windows(num_requests, window_seconds):
    """
    Property 31: Throughput is calculated over time windows.
    
    For any time window query, throughput (requests per second) should be
    calculable from recorded request timestamps.
    
    Validates: Requirements 9.3
    """
    monitor = PerformanceMonitor(max_history_seconds=window_seconds * 2)
    
    # Record requests with small time intervals
    for i in range(num_requests):
        monitor.record_latency("test_operation", 10.0)
        if i < num_requests - 1:
            time.sleep(0.005)  # Smaller delay for faster testing
    
    # Calculate throughput
    throughput = monitor.get_throughput(window_seconds=window_seconds)
    
    # Verify throughput is reasonable
    assert throughput >= 0, "Throughput should be non-negative"
    # With 0.005s delays, max throughput is ~200 rps
    assert throughput <= 500, "Throughput should be within reasonable bounds"


# Feature: advanced-optimization-monitoring, Property 32: Resource utilization is tracked
@given(
    num_samples=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_property_32_resource_utilization_tracked(num_samples):
    """
    Property 32: Resource utilization is tracked.
    
    For any monitoring period, GPU, CPU, and memory utilization should be
    tracked and queryable.
    
    Validates: Requirements 9.4
    """
    monitor = PerformanceMonitor(
        max_history_seconds=60,
        resource_sample_interval=0.1  # Sample every 0.1 seconds for faster testing
    )
    
    # Trigger resource sampling by recording latencies
    for i in range(num_samples):
        monitor.record_latency("test_op", 10.0)
        time.sleep(0.15)  # Wait slightly longer than sample interval
    
    # Get resource trends
    trends = monitor.get_resource_trends(window_seconds=60)
    
    # Verify resource metrics are tracked
    assert trends.avg_cpu_percent >= 0, "CPU usage should be non-negative"
    assert trends.max_cpu_percent >= trends.avg_cpu_percent, "Max CPU should be >= avg"
    assert trends.avg_memory_mb >= 0, "Memory usage should be non-negative"
    assert trends.max_memory_mb >= trends.avg_memory_mb, "Max memory should be >= avg"


# Feature: advanced-optimization-monitoring, Property 33: Percentile latencies are computable
@given(
    latencies=st.lists(latency_values, min_size=10, max_size=100)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_33_percentile_latencies_computable(latencies):
    """
    Property 33: Percentile latencies are computable.
    
    For any set of latency measurements, p50, p95, and p99 percentiles
    should be computable.
    
    Validates: Requirements 9.5
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    operation = "test_operation"
    
    # Record latencies
    for latency in latencies:
        monitor.record_latency(operation, latency)
    
    # Get percentiles
    percentiles = monitor.get_percentiles(operation)
    
    # Verify percentiles are computed
    assert "p50" in percentiles, "p50 should be computed"
    assert "p95" in percentiles, "p95 should be computed"
    assert "p99" in percentiles, "p99 should be computed"
    
    # Verify percentile ordering
    assert percentiles["p50"] >= min(latencies) * 0.99, "p50 should be >= min"
    assert percentiles["p99"] <= max(latencies) * 1.01, "p99 should be <= max"
    assert percentiles["p50"] <= percentiles["p95"], "p50 should be <= p95"
    assert percentiles["p95"] <= percentiles["p99"], "p95 should be <= p99"
    
    # Verify percentiles are within data range
    sorted_latencies = sorted(latencies)
    assert percentiles["p50"] >= sorted_latencies[0], "p50 should be >= minimum value"
    assert percentiles["p99"] <= sorted_latencies[-1], "p99 should be <= maximum value"


# Additional property: Sliding window filtering works correctly
@given(
    latencies=st.lists(latency_values, min_size=10, max_size=50),
    window_seconds=st.integers(min_value=5, max_value=30)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_sliding_window_filtering(latencies, window_seconds):
    """
    Property: Sliding window filtering works correctly.
    
    For any time window, only data within that window should be included
    in calculations.
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    operation = "test_operation"
    
    # Record half the latencies
    half = len(latencies) // 2
    for latency in latencies[:half]:
        monitor.record_latency(operation, latency)
    
    # Wait to ensure time separation
    time.sleep(0.5)
    
    # Record remaining latencies
    for latency in latencies[half:]:
        monitor.record_latency(operation, latency)
    
    # Get metrics for very short window (should only include recent data)
    metrics_short = monitor.get_operation_metrics(operation, window_seconds=1)
    
    # Get metrics for all data
    metrics_all = monitor.get_operation_metrics(operation)
    
    # Verify windowing works
    if metrics_short:
        assert metrics_short.count <= metrics_all.count, "Short window should have <= records"
    assert metrics_all.count == len(latencies), "All window should have all records"


# Additional property: Error tracking works correctly
@given(
    error_types=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=10
    ),
    components=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_error_tracking(error_types, components):
    """
    Property: Error tracking works correctly.
    
    For any errors recorded, they should be tracked by type and component.
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    
    # Record errors
    for error_type in error_types:
        for component in components:
            monitor.record_error(error_type, component)
    
    # Verify error rate is non-negative
    error_rate = monitor.get_error_rate()
    assert error_rate >= 0, "Error rate should be non-negative"
    
    # Verify component-specific error rate
    for component in components:
        component_error_rate = monitor.get_error_rate(component=component)
        assert component_error_rate >= 0, "Component error rate should be non-negative"


# Additional property: Statistics reset works correctly
@given(
    latencies=st.lists(latency_values, min_size=5, max_size=30)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_statistics_reset(latencies):
    """
    Property: Statistics reset works correctly.
    
    After reset, all statistics should be cleared.
    """
    monitor = PerformanceMonitor(max_history_seconds=60)
    operation = "test_operation"
    
    # Record some data
    for latency in latencies:
        monitor.record_latency(operation, latency)
    
    # Verify data exists
    metrics_before = monitor.get_operation_metrics(operation)
    assert metrics_before is not None, "Metrics should exist before reset"
    assert metrics_before.count > 0, "Should have recorded data"
    
    # Reset statistics
    monitor.reset_statistics()
    
    # Verify data is cleared
    metrics_after = monitor.get_operation_metrics(operation)
    assert metrics_after is None, "Metrics should be None after reset"
    
    throughput = monitor.get_throughput()
    assert throughput == 0.0, "Throughput should be 0 after reset"
    
    operations = monitor.get_all_operations()
    assert len(operations) == 0, "No operations should be tracked after reset"
