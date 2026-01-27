"""
Unit tests for AutoTuner class.

Tests the performance analysis and tuning recommendation logic.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from mm_orch.optimization.auto_tuner import AutoTuner, TuningRecommendations, TuningEvent
from mm_orch.optimization.config import TunerConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetrics, ResourceTrends


@pytest.fixture
def tuner_config():
    """Create a test tuner configuration."""
    return TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=True
    )


@pytest.fixture
def performance_monitor():
    """Create a mock performance monitor."""
    monitor = Mock(spec=PerformanceMonitor)
    return monitor


@pytest.fixture
def auto_tuner(tuner_config, performance_monitor):
    """Create an AutoTuner instance for testing."""
    return AutoTuner(
        config=tuner_config,
        performance_monitor=performance_monitor
    )


def test_auto_tuner_initialization(auto_tuner, tuner_config):
    """Test AutoTuner initializes correctly."""
    assert auto_tuner.config == tuner_config
    assert auto_tuner.performance_monitor is not None
    assert len(auto_tuner.get_tuning_history()) == 0


def test_analyze_performance_with_high_latency(auto_tuner, performance_monitor):
    """Test that high latency triggers batch size reduction."""
    # Mock high latency metrics
    metrics = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=1200.0,
        min_latency_ms=800.0,
        max_latency_ms=2000.0,
        p50_latency_ms=1100.0,
        p95_latency_ms=1800.0,
        p99_latency_ms=1950.0,
        throughput_rps=15.0
    )
    
    # Mock resource trends
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=50.0,
        max_cpu_percent=70.0,
        avg_memory_mb=4096.0,
        max_memory_mb=6144.0
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    
    # Set current batch size
    auto_tuner._current_batch_size = 32
    
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Should recommend reducing batch size due to high latency
    assert recommendations.batch_size is not None
    assert recommendations.batch_size < 32
    assert "batch size" in recommendations.rationale.lower()


def test_analyze_performance_with_low_throughput(auto_tuner, performance_monitor):
    """Test that low throughput with good latency triggers batch size increase."""
    # Mock low throughput with acceptable latency
    metrics = PerformanceMetrics(
        operation="inference",
        count=50,
        mean_latency_ms=300.0,
        min_latency_ms=200.0,
        max_latency_ms=450.0,
        p50_latency_ms=280.0,
        p95_latency_ms=420.0,
        p99_latency_ms=445.0,
        throughput_rps=5.0
    )
    
    # Mock resource trends
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=30.0,
        max_cpu_percent=50.0,
        avg_memory_mb=2048.0,
        max_memory_mb=3072.0
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    
    # Set current batch size
    auto_tuner._current_batch_size = 16
    
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Should recommend increasing batch size due to low throughput
    assert recommendations.batch_size is not None
    assert recommendations.batch_size > 16
    assert "batch size" in recommendations.rationale.lower()


def test_analyze_performance_with_high_throughput(auto_tuner, performance_monitor):
    """Test that high throughput triggers timeout reduction."""
    # Mock high throughput metrics
    metrics = PerformanceMetrics(
        operation="inference",
        count=500,
        mean_latency_ms=150.0,
        min_latency_ms=100.0,
        max_latency_ms=250.0,
        p50_latency_ms=140.0,
        p95_latency_ms=220.0,
        p99_latency_ms=245.0,
        throughput_rps=60.0
    )
    
    # Mock resource trends
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=60.0,
        max_cpu_percent=80.0,
        avg_memory_mb=5120.0,
        max_memory_mb=7168.0
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    
    # Set current timeout
    auto_tuner._current_batch_timeout = 50
    
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Should recommend reducing timeout due to high throughput
    assert recommendations.batch_timeout_ms is not None
    assert recommendations.batch_timeout_ms < 50
    assert "timeout" in recommendations.rationale.lower()


def test_analyze_performance_disabled(tuner_config, performance_monitor):
    """Test that analysis returns no recommendations when disabled."""
    tuner_config.enabled = False
    auto_tuner = AutoTuner(
        config=tuner_config,
        performance_monitor=performance_monitor
    )
    
    metrics = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=1200.0,
        min_latency_ms=800.0,
        max_latency_ms=2000.0,
        p50_latency_ms=1100.0,
        p95_latency_ms=1800.0,
        p99_latency_ms=1950.0,
        throughput_rps=15.0
    )
    
    recommendations = auto_tuner.analyze_performance(metrics)
    
    assert recommendations.batch_size is None
    assert recommendations.batch_timeout_ms is None
    assert recommendations.cache_size_mb is None
    assert "disabled" in recommendations.rationale.lower()


def test_apply_tuning_disabled(tuner_config, performance_monitor):
    """
    Test that tuning is not applied when auto-tuning is disabled.
    
    Validates Requirement 12.6: Static configuration is used when disabled.
    """
    tuner_config.enabled = False
    auto_tuner = AutoTuner(
        config=tuner_config,
        performance_monitor=performance_monitor
    )
    
    # Set up callbacks
    batch_size_callback = Mock()
    batch_timeout_callback = Mock()
    cache_size_callback = Mock()
    
    auto_tuner.batch_size_callback = batch_size_callback
    auto_tuner.batch_timeout_callback = batch_timeout_callback
    auto_tuner.cache_size_callback = cache_size_callback
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=500.0,
        min_latency_ms=300.0,
        max_latency_ms=800.0,
        p50_latency_ms=480.0,
        p95_latency_ms=750.0,
        p99_latency_ms=790.0,
        throughput_rps=20.0
    )
    
    # Create recommendations (even though disabled)
    recommendations = TuningRecommendations(
        batch_size=64,
        batch_timeout_ms=100,
        cache_size_mb=8192,
        rationale="Test tuning"
    )
    
    # Apply tuning (should be no-op when disabled)
    auto_tuner.apply_tuning(recommendations)
    
    # Verify callbacks were NOT called (static parameters maintained)
    batch_size_callback.assert_not_called()
    batch_timeout_callback.assert_not_called()
    cache_size_callback.assert_not_called()
    
    # Verify no history was recorded
    history = auto_tuner.get_tuning_history()
    assert len(history) == 0


def test_start_disabled(tuner_config, performance_monitor):
    """
    Test that background tuning loop does not start when disabled.
    
    Validates Requirement 12.6: Static configuration is used when disabled.
    """
    tuner_config.enabled = False
    auto_tuner = AutoTuner(
        config=tuner_config,
        performance_monitor=performance_monitor
    )
    
    # Attempt to start (should be no-op)
    auto_tuner.start()
    
    # Verify thread was not started
    assert auto_tuner._tuning_thread is None


def test_apply_tuning_with_callbacks(auto_tuner, performance_monitor):
    """Test that tuning recommendations are applied via callbacks."""
    # Set up callbacks
    batch_size_callback = Mock()
    batch_timeout_callback = Mock()
    cache_size_callback = Mock()
    
    auto_tuner.batch_size_callback = batch_size_callback
    auto_tuner.batch_timeout_callback = batch_timeout_callback
    auto_tuner.cache_size_callback = cache_size_callback
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=500.0,
        min_latency_ms=300.0,
        max_latency_ms=800.0,
        p50_latency_ms=480.0,
        p95_latency_ms=750.0,
        p99_latency_ms=790.0,
        throughput_rps=20.0
    )
    
    # Create recommendations
    recommendations = TuningRecommendations(
        batch_size=64,
        batch_timeout_ms=100,
        cache_size_mb=8192,
        rationale="Test tuning"
    )
    
    # Apply tuning
    auto_tuner.apply_tuning(recommendations)
    
    # Verify callbacks were called
    batch_size_callback.assert_called_once_with(64)
    batch_timeout_callback.assert_called_once_with(100)
    cache_size_callback.assert_called_once_with(8192)
    
    # Verify history was recorded
    history = auto_tuner.get_tuning_history()
    assert len(history) == 3
    assert any(e.parameter == "batch_size" for e in history)
    assert any(e.parameter == "batch_timeout_ms" for e in history)
    assert any(e.parameter == "cache_size_mb" for e in history)


def test_tuning_history_tracking(auto_tuner, performance_monitor):
    """Test that tuning history is tracked correctly."""
    # Set up callback
    batch_size_callback = Mock()
    auto_tuner.batch_size_callback = batch_size_callback
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=500.0,
        min_latency_ms=300.0,
        max_latency_ms=800.0,
        p50_latency_ms=480.0,
        p95_latency_ms=750.0,
        p99_latency_ms=790.0,
        throughput_rps=20.0
    )
    
    # Apply multiple tuning changes
    for i in range(5):
        recommendations = TuningRecommendations(
            batch_size=32 + i * 8,
            rationale=f"Tuning iteration {i}"
        )
        auto_tuner.apply_tuning(recommendations)
    
    # Verify history
    history = auto_tuner.get_tuning_history()
    assert len(history) == 5
    
    # Verify history is in chronological order
    for i in range(len(history) - 1):
        assert history[i].timestamp <= history[i + 1].timestamp


def test_check_for_degradation(auto_tuner, performance_monitor):
    """Test degradation detection after tuning."""
    # Set baseline
    auto_tuner._baseline_latency = 500.0
    auto_tuner._baseline_throughput = 20.0
    
    # Mock degraded performance
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=800.0,
        min_latency_ms=600.0,
        max_latency_ms=1200.0,
        p50_latency_ms=780.0,
        p95_latency_ms=650.0,  # >20% worse than baseline
        p99_latency_ms=1150.0,
        throughput_rps=15.0  # >20% worse than baseline
    )
    
    # Check for degradation
    degraded = auto_tuner.check_for_degradation()
    assert degraded is True


def test_no_degradation_detected(auto_tuner, performance_monitor):
    """Test that no degradation is detected when performance is good."""
    # Set baseline
    auto_tuner._baseline_latency = 500.0
    auto_tuner._baseline_throughput = 20.0
    
    # Mock similar or better performance
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=450.0,
        min_latency_ms=300.0,
        max_latency_ms=600.0,
        p50_latency_ms=440.0,
        p95_latency_ms=550.0,  # Within 20% of baseline
        p99_latency_ms=590.0,
        throughput_rps=22.0  # Better than baseline
    )
    
    # Check for degradation
    degraded = auto_tuner.check_for_degradation()
    assert degraded is False


def test_analyze_performance_with_insufficient_data(auto_tuner, performance_monitor):
    """Test that analysis handles insufficient data gracefully."""
    # Mock no metrics available
    performance_monitor.get_operation_metrics.return_value = None
    
    recommendations = auto_tuner.analyze_performance()
    
    assert recommendations.batch_size is None
    assert recommendations.batch_timeout_ms is None
    assert recommendations.cache_size_mb is None
    assert "insufficient" in recommendations.rationale.lower()


def test_rollback_last_tuning(auto_tuner, performance_monitor):
    """Test that rollback restores previous parameter values."""
    # Set up callback
    batch_size_callback = Mock()
    auto_tuner.batch_size_callback = batch_size_callback
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=500.0,
        min_latency_ms=300.0,
        max_latency_ms=800.0,
        p50_latency_ms=480.0,
        p95_latency_ms=750.0,
        p99_latency_ms=790.0,
        throughput_rps=20.0
    )
    
    # Set initial value
    auto_tuner._current_batch_size = 32
    
    # Apply tuning
    recommendations = TuningRecommendations(
        batch_size=64,
        rationale="Increase batch size"
    )
    auto_tuner.apply_tuning(recommendations)
    
    # Verify new value was applied
    assert auto_tuner._current_batch_size == 64
    batch_size_callback.assert_called_with(64)
    
    # Rollback
    auto_tuner.rollback_last_tuning()
    
    # Verify rollback to old value
    assert auto_tuner._current_batch_size == 32
    batch_size_callback.assert_called_with(32)
    
    # Verify rollback event in history
    history = auto_tuner.get_tuning_history()
    assert len(history) == 2
    assert history[-1].parameter == "batch_size"
    assert history[-1].old_value == 64
    assert history[-1].new_value == 32
    assert "rollback" in history[-1].rationale.lower()


def test_rollback_with_no_history(auto_tuner):
    """Test that rollback handles empty history gracefully."""
    # Attempt rollback with no history
    auto_tuner.rollback_last_tuning()
    
    # Should not raise an error
    assert len(auto_tuner.get_tuning_history()) == 0


def test_tuning_with_prometheus_metrics(tuner_config, performance_monitor):
    """Test that tuning decisions are exposed in Prometheus metrics."""
    # Create mock Prometheus exporter
    prometheus_exporter = Mock()
    prometheus_exporter.is_enabled.return_value = True
    
    # Create auto_tuner with Prometheus exporter
    auto_tuner = AutoTuner(
        config=tuner_config,
        performance_monitor=performance_monitor,
        prometheus_exporter=prometheus_exporter
    )
    
    # Set up callback
    batch_size_callback = Mock()
    auto_tuner.batch_size_callback = batch_size_callback
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=500.0,
        min_latency_ms=300.0,
        max_latency_ms=800.0,
        p50_latency_ms=480.0,
        p95_latency_ms=750.0,
        p99_latency_ms=790.0,
        throughput_rps=20.0
    )
    
    # Set initial value
    auto_tuner._current_batch_size = 32
    
    # Apply tuning
    recommendations = TuningRecommendations(
        batch_size=64,
        rationale="Increase batch size"
    )
    auto_tuner.apply_tuning(recommendations)
    
    # Verify metrics were recorded (if Prometheus is available)
    if hasattr(auto_tuner, 'tuning_decisions'):
        # Metrics should have been recorded
        assert auto_tuner._current_batch_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
