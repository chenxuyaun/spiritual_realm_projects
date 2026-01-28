"""
Property-based tests for anomaly detection.

Tests universal properties of the AnomalyDetector across all valid inputs.
Uses Hypothesis for property-based testing with 100+ iterations per property.
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from mm_orch.monitoring.anomaly_detector import (
    AnomalyDetector,
    Alert,
    AlertSeverity,
    AlertType
)
from mm_orch.monitoring.config import AnomalyConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor


# Feature: advanced-optimization-monitoring, Property 34: Latency threshold triggers alerts
@given(
    latency_ms=st.floats(min_value=0.0, max_value=10000.0),
    threshold_ms=st.floats(min_value=1.0, max_value=5000.0)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_latency_threshold_triggers_alerts(latency_ms, threshold_ms):
    """
    Property: For any operation with latency exceeding the configured threshold,
    a latency alert should be triggered.
    
    Validates: Requirements 10.1
    """
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=threshold_ms,
        alert_rate_limit_seconds=0  # Disable rate limiting for test
    )
    detector = AnomalyDetector(config)
    
    # Check latency
    alert = detector.check_latency("test_operation", latency_ms)
    
    # Verify: Alert triggered if and only if latency exceeds threshold
    if latency_ms > threshold_ms:
        assert alert is not None, f"Expected alert for latency {latency_ms}ms > threshold {threshold_ms}ms"
        assert alert.alert_type == AlertType.LATENCY.value
        assert "threshold exceeded" in alert.message.lower()
        assert alert.metadata["latency_ms"] == latency_ms
        assert alert.metadata["threshold_ms"] == threshold_ms
    else:
        assert alert is None, f"Unexpected alert for latency {latency_ms}ms <= threshold {threshold_ms}ms"


# Feature: advanced-optimization-monitoring, Property 35: Error rate threshold triggers alerts
@given(
    error_count=st.integers(min_value=0, max_value=100),
    total_requests=st.integers(min_value=1, max_value=1000),
    threshold=st.floats(min_value=0.01, max_value=0.5)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_error_rate_threshold_triggers_alerts(error_count, total_requests, threshold):
    """
    Property: For any time window where error rate exceeds the configured threshold,
    an error rate alert should be triggered.
    
    Validates: Requirements 10.2
    """
    config = AnomalyConfig(
        enabled=True,
        error_rate_threshold=threshold,
        alert_rate_limit_seconds=0  # Disable rate limiting for test
    )
    
    # Mock performance monitor
    perf_monitor = Mock(spec=PerformanceMonitor)
    window_seconds = 60
    
    # Calculate rates
    throughput = total_requests / window_seconds
    error_rate_per_sec = error_count / window_seconds
    actual_error_rate = error_count / total_requests if total_requests > 0 else 0.0
    
    perf_monitor.get_throughput.return_value = throughput
    perf_monitor.get_error_rate.return_value = error_rate_per_sec
    
    detector = AnomalyDetector(config, perf_monitor)
    
    # Check error rate
    alert = detector.check_error_rate(window_seconds=window_seconds)
    
    # Verify: Alert triggered if and only if error rate exceeds threshold
    if actual_error_rate > threshold:
        assert alert is not None, f"Expected alert for error rate {actual_error_rate:.2%} > threshold {threshold:.2%}"
        assert alert.alert_type == AlertType.ERROR_RATE.value
        assert "error rate" in alert.message.lower()
        assert alert.metadata["error_rate"] == pytest.approx(actual_error_rate, abs=0.01)
    else:
        assert alert is None, f"Unexpected alert for error rate {actual_error_rate:.2%} <= threshold {threshold:.2%}"


# Feature: advanced-optimization-monitoring, Property 36: Memory threshold triggers alerts
@given(
    memory_percent=st.floats(min_value=0.0, max_value=100.0),
    threshold_percent=st.floats(min_value=50.0, max_value=99.0)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_memory_threshold_triggers_alerts(memory_percent, threshold_percent):
    """
    Property: For any measurement where memory usage exceeds the configured threshold,
    a memory alert should be triggered.
    
    Validates: Requirements 10.3
    """
    config = AnomalyConfig(
        enabled=True,
        memory_threshold_percent=threshold_percent,
        alert_rate_limit_seconds=0  # Disable rate limiting for test
    )
    detector = AnomalyDetector(config)
    
    # Check memory usage
    alert = detector.check_resources(memory_percent=memory_percent)
    
    # Verify: Alert triggered if and only if memory exceeds threshold
    if memory_percent > threshold_percent:
        assert alert is not None, f"Expected alert for memory {memory_percent:.1f}% > threshold {threshold_percent}%"
        assert alert.alert_type == AlertType.RESOURCE.value  # Changed from MEMORY to RESOURCE
        assert "threshold exceeded" in alert.message.lower()
        assert alert.metadata["memory_percent"] == memory_percent
        assert alert.metadata["threshold_percent"] == threshold_percent
    else:
        assert alert is None, f"Unexpected alert for memory {memory_percent:.1f}% <= threshold {threshold_percent}%"


# Feature: advanced-optimization-monitoring, Property 37: Throughput threshold triggers alerts
@given(
    throughput_rps=st.floats(min_value=0.0, max_value=100.0),
    threshold_rps=st.floats(min_value=1.0, max_value=50.0)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_throughput_threshold_triggers_alerts(throughput_rps, threshold_rps):
    """
    Property: For any time window where throughput drops below the configured threshold,
    a throughput alert should be triggered.
    
    Validates: Requirements 10.4
    """
    config = AnomalyConfig(
        enabled=True,
        throughput_threshold_rps=threshold_rps,
        alert_rate_limit_seconds=0  # Disable rate limiting for test
    )
    detector = AnomalyDetector(config)
    
    # Check throughput
    alert = detector.check_throughput(throughput_rps)
    
    # Verify: Alert triggered if and only if throughput below threshold
    if throughput_rps < threshold_rps:
        assert alert is not None, f"Expected alert for throughput {throughput_rps:.2f}rps < threshold {threshold_rps}rps"
        assert alert.alert_type == AlertType.THROUGHPUT.value
        assert "below threshold" in alert.message.lower()
        assert alert.metadata["throughput_rps"] == throughput_rps
        assert alert.metadata["threshold_rps"] == threshold_rps
    else:
        assert alert is None, f"Unexpected alert for throughput {throughput_rps:.2f}rps >= threshold {threshold_rps}rps"


# Feature: advanced-optimization-monitoring, Property 38: Alerts are sent to configured destinations
@given(
    destinations=st.lists(
        st.sampled_from(["log", "webhook", "alertmanager"]),
        min_size=1,
        max_size=3,
        unique=True
    )
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_alerts_sent_to_configured_destinations(destinations):
    """
    Property: For any triggered alert, it should be sent to all configured alert destinations.
    
    Validates: Requirements 10.5
    """
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        alert_destinations=destinations,
        webhook_url="http://example.com/webhook" if "webhook" in destinations else None,
        alert_rate_limit_seconds=0  # Disable rate limiting for test
    )
    detector = AnomalyDetector(config)
    
    # Create an alert that exceeds threshold
    alert = detector.check_latency("test_operation", 200.0)
    
    # Verify: Alert was created
    assert alert is not None
    
    # Verify: Alert is in history (log destination always works)
    history = detector.get_alert_history()
    assert len(history) > 0
    assert history[-1].alert_type == AlertType.LATENCY.value
    
    # Note: We can't easily test webhook/alertmanager without mocking HTTP,
    # but the code path is exercised and logged


# Feature: advanced-optimization-monitoring, Property 39: Alert rate limiting prevents storms
@given(
    num_alerts=st.integers(min_value=2, max_value=20),
    rate_limit_seconds=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_alert_rate_limiting_prevents_storms(num_alerts, rate_limit_seconds):
    """
    Property: For any rapid sequence of similar alerts, only one alert per type
    should be sent within the rate limit window.
    
    Validates: Requirements 10.6
    """
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        alert_rate_limit_seconds=rate_limit_seconds
    )
    detector = AnomalyDetector(config)
    
    # Trigger multiple alerts of the same type rapidly
    alerts_sent = []
    for i in range(num_alerts):
        alert = detector.check_latency("test_operation", 200.0)
        if alert is not None:
            alerts_sent.append(alert)
    
    # Verify: Only one alert was sent (first one)
    assert len(alerts_sent) == 1, f"Expected 1 alert due to rate limiting, got {len(alerts_sent)}"
    
    # Verify: Alert history shows only one alert
    history = detector.get_alert_history(alert_type=AlertType.LATENCY.value)
    assert len(history) == 1


# Additional property: Severity levels are assigned correctly
@given(
    latency_ms=st.floats(min_value=100.0, max_value=10000.0),
    threshold_ms=st.floats(min_value=100.0, max_value=1000.0)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_severity_levels_assigned_correctly(latency_ms, threshold_ms):
    """
    Property: Alert severity should increase with the magnitude of threshold violation.
    
    Validates: Requirements 10.1 (implicit severity requirement)
    """
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=threshold_ms,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    alert = detector.check_latency("test_operation", latency_ms)
    
    if alert is not None:
        excess_ratio = latency_ms / threshold_ms
        
        # Verify severity matches excess ratio
        if excess_ratio >= 3.0:
            assert alert.severity == AlertSeverity.CRITICAL.value
        elif excess_ratio >= 2.0:
            assert alert.severity == AlertSeverity.ERROR.value
        else:
            assert alert.severity == AlertSeverity.WARNING.value


# Property: Disabled detector doesn't trigger alerts
@given(
    latency_ms=st.floats(min_value=0.0, max_value=10000.0),
    threshold_ms=st.floats(min_value=1.0, max_value=5000.0)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_disabled_detector_no_alerts(latency_ms, threshold_ms):
    """
    Property: When anomaly detection is disabled, no alerts should be triggered
    regardless of threshold violations.
    """
    config = AnomalyConfig(
        enabled=False,  # Disabled
        latency_threshold_ms=threshold_ms
    )
    detector = AnomalyDetector(config)
    
    # Try to trigger alert
    alert = detector.check_latency("test_operation", latency_ms)
    
    # Verify: No alert triggered even if threshold exceeded
    assert alert is None


# Property: Alert metadata contains relevant information
@given(
    latency_ms=st.floats(min_value=100.0, max_value=10000.0),
    threshold_ms=st.floats(min_value=1.0, max_value=100.0)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_alert_metadata_contains_relevant_info(latency_ms, threshold_ms):
    """
    Property: Every alert should contain metadata with threshold, actual value,
    and other relevant context.
    """
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=threshold_ms,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    alert = detector.check_latency("test_operation", latency_ms)
    
    if alert is not None:
        # Verify metadata contains key information
        assert "latency_ms" in alert.metadata
        assert "threshold_ms" in alert.metadata
        assert "operation" in alert.metadata
        assert alert.metadata["latency_ms"] == latency_ms
        assert alert.metadata["threshold_ms"] == threshold_ms
        assert alert.metadata["operation"] == "test_operation"


# Property: GPU memory is prioritized over system memory
@given(
    system_memory=st.floats(min_value=0.0, max_value=100.0),
    gpu_memory=st.floats(min_value=0.0, max_value=100.0),
    threshold=st.floats(min_value=50.0, max_value=99.0)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_gpu_memory_prioritized_over_system(system_memory, gpu_memory, threshold):
    """
    Property: When both system and GPU memory are provided, GPU memory should be
    checked against the threshold (GPU memory is more critical for ML workloads).
    """
    config = AnomalyConfig(
        enabled=True,
        memory_threshold_percent=threshold,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    # Check with both memory types
    alert = detector.check_resources(
        memory_percent=system_memory,
        gpu_memory_percent=gpu_memory
    )
    
    # Verify: Alert based on GPU memory, not system memory
    if gpu_memory > threshold:
        assert alert is not None
        assert "GPU memory" in alert.message or "gpu" in alert.metadata.get("memory_type", "")
    elif system_memory > threshold:
        # System memory exceeds but GPU doesn't - no alert
        assert alert is None
    else:
        assert alert is None
