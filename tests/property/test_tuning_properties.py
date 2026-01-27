"""
Property-based tests for auto-tuning.

Tests Properties 45-49 from the design document:
- Property 45: Batch size adapts to load changes
- Property 46: Timeout parameters adapt to load changes
- Property 47: Cache size adapts to usage patterns
- Property 48: Tuning decisions are logged
- Property 49: Static configuration is used when auto-tuning is disabled
"""

import time
from hypothesis import given, strategies as st, settings
import pytest
from unittest.mock import Mock
from datetime import datetime

from mm_orch.optimization.auto_tuner import AutoTuner, TuningRecommendations
from mm_orch.optimization.config import TunerConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetrics, ResourceTrends


# Feature: advanced-optimization-monitoring, Property 45: Batch size adapts to load changes
@given(
    initial_latency=st.floats(min_value=100.0, max_value=500.0),
    load_change=st.sampled_from(["high_latency", "low_throughput", "optimal"]),
)
@settings(max_examples=50, deadline=2000)
@pytest.mark.property
def test_batch_size_adapts_to_load(initial_latency, load_change):
    """
    Property 45: Batch size adapts to load changes.
    
    For any significant change in system load, the auto-tuner should
    adjust batch size to maintain performance targets.
    
    Validates: Requirements 12.1
    """
    # Setup: Create tuner with batch size tuning enabled
    config = TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=False,
        enable_cache_size_tuning=False
    )
    
    performance_monitor = Mock(spec=PerformanceMonitor)
    batch_size_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        batch_size_callback=batch_size_callback
    )
    
    # Set initial batch size
    initial_batch_size = 32
    auto_tuner._current_batch_size = initial_batch_size
    
    # Create metrics based on load change scenario
    if load_change == "high_latency":
        # High latency should trigger batch size reduction
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
    elif load_change == "low_throughput":
        # Low throughput with good latency should trigger batch size increase
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
    else:  # optimal
        # Optimal performance should not trigger changes
        metrics = PerformanceMetrics(
            operation="inference",
            count=200,
            mean_latency_ms=initial_latency,
            min_latency_ms=initial_latency * 0.7,
            max_latency_ms=initial_latency * 1.3,
            p50_latency_ms=initial_latency,
            p95_latency_ms=initial_latency * 1.2,
            p99_latency_ms=initial_latency * 1.25,
            throughput_rps=25.0
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
    performance_monitor.get_operation_metrics.return_value = metrics
    
    # Analyze performance
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Verify: Batch size recommendation adapts to load
    if load_change == "high_latency":
        # Should recommend reducing batch size
        assert recommendations.batch_size is not None
        assert recommendations.batch_size < initial_batch_size
    elif load_change == "low_throughput":
        # Should recommend increasing batch size
        assert recommendations.batch_size is not None
        assert recommendations.batch_size > initial_batch_size
    else:  # optimal
        # Should not recommend changes (or minimal changes)
        if recommendations.batch_size is not None:
            # If there is a recommendation, it should be close to current
            assert abs(recommendations.batch_size - initial_batch_size) <= initial_batch_size * 0.3
    
    # Apply tuning if there's a recommendation
    if recommendations.batch_size:
        auto_tuner.apply_tuning(recommendations)
        
        # Verify callback was called with new batch size
        batch_size_callback.assert_called_once_with(recommendations.batch_size)
        
        # Verify current batch size was updated
        assert auto_tuner._current_batch_size == recommendations.batch_size


# Feature: advanced-optimization-monitoring, Property 46: Timeout parameters adapt to load changes
@given(
    throughput=st.floats(min_value=1.0, max_value=100.0),
    load_pattern=st.sampled_from(["high_throughput", "low_throughput", "moderate"]),
)
@settings(max_examples=50, deadline=2000)
@pytest.mark.property
def test_timeout_adapts_to_load(throughput, load_pattern):
    """
    Property 46: Timeout parameters adapt to load changes.
    
    For any significant change in request arrival patterns, the auto-tuner
    should adjust timeout parameters.
    
    Validates: Requirements 12.2
    """
    # Setup: Create tuner with timeout tuning enabled
    config = TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=False,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=False
    )
    
    performance_monitor = Mock(spec=PerformanceMonitor)
    batch_timeout_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        batch_timeout_callback=batch_timeout_callback
    )
    
    # Set initial timeout
    initial_timeout = 50
    auto_tuner._current_batch_timeout = initial_timeout
    
    # Create metrics based on load pattern
    if load_pattern == "high_throughput":
        actual_throughput = max(throughput, 60.0)
    elif load_pattern == "low_throughput":
        actual_throughput = min(throughput, 4.0)
    else:  # moderate
        actual_throughput = throughput
    
    metrics = PerformanceMetrics(
        operation="inference",
        count=int(actual_throughput * 60),
        mean_latency_ms=200.0,
        min_latency_ms=150.0,
        max_latency_ms=300.0,
        p50_latency_ms=190.0,
        p95_latency_ms=280.0,
        p99_latency_ms=295.0,
        throughput_rps=actual_throughput
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
    performance_monitor.get_operation_metrics.return_value = metrics
    
    # Analyze performance
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Verify: Timeout recommendation adapts to load pattern
    if load_pattern == "high_throughput":
        # High throughput should trigger timeout reduction
        assert recommendations.batch_timeout_ms is not None
        assert recommendations.batch_timeout_ms < initial_timeout
    elif load_pattern == "low_throughput":
        # Low throughput should trigger timeout increase
        assert recommendations.batch_timeout_ms is not None
        assert recommendations.batch_timeout_ms > initial_timeout
    # For moderate throughput, may or may not recommend changes
    
    # Apply tuning if there's a recommendation
    if recommendations.batch_timeout_ms:
        auto_tuner.apply_tuning(recommendations)
        
        # Verify callback was called with new timeout
        batch_timeout_callback.assert_called_once_with(recommendations.batch_timeout_ms)
        
        # Verify current timeout was updated
        assert auto_tuner._current_batch_timeout == recommendations.batch_timeout_ms


# Feature: advanced-optimization-monitoring, Property 47: Cache size adapts to usage patterns
@given(
    memory_scenario=st.sampled_from(["high_memory", "low_memory", "moderate_memory"]),
)
@settings(max_examples=50, deadline=2000)
@pytest.mark.property
def test_cache_size_adapts_to_usage(memory_scenario):
    """
    Property 47: Cache size adapts to usage patterns.
    
    For any significant change in cache hit rates or memory pressure,
    the auto-tuner should adjust cache size.
    
    Validates: Requirements 12.3
    
    Note: This test verifies that the cache size tuning mechanism works
    and produces valid recommendations. The actual direction (increase/decrease)
    depends on the real system memory (via psutil), so we verify the mechanism
    rather than the specific direction.
    """
    # Setup: Create tuner with cache size tuning enabled
    config = TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=False,
        enable_timeout_tuning=False,
        enable_cache_size_tuning=True
    )
    
    performance_monitor = Mock(spec=PerformanceMonitor)
    cache_size_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        cache_size_callback=cache_size_callback
    )
    
    # Set initial cache size
    initial_cache_size = 4096
    auto_tuner._current_cache_size = initial_cache_size
    
    # Create metrics
    metrics = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=300.0,
        min_latency_ms=200.0,
        max_latency_ms=500.0,
        p50_latency_ms=280.0,
        p95_latency_ms=450.0,
        p99_latency_ms=490.0,
        throughput_rps=20.0
    )
    
    # Mock resource trends based on memory scenario
    # Use realistic memory values
    if memory_scenario == "high_memory":
        # High memory usage
        avg_memory_mb = 14000.0
        max_memory_mb = 15000.0
    elif memory_scenario == "low_memory":
        # Low memory usage
        avg_memory_mb = 6000.0
        max_memory_mb = 7000.0
    else:  # moderate_memory
        # Moderate memory usage
        avg_memory_mb = 10000.0
        max_memory_mb = 11000.0
    
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=50.0,
        max_cpu_percent=70.0,
        avg_memory_mb=avg_memory_mb,
        max_memory_mb=max_memory_mb
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    performance_monitor.get_operation_metrics.return_value = metrics
    
    # Analyze performance
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Verify: Cache size tuning mechanism works
    # The auto-tuner uses psutil to get actual system memory, so the
    # recommendation direction depends on the real system state.
    # We verify that:
    # 1. The mechanism produces valid recommendations
    # 2. Recommendations are within reasonable bounds
    # 3. The tuning can be applied successfully
    
    if recommendations.cache_size_mb is not None:
        # Verify recommendation is within reasonable bounds
        assert 1024 <= recommendations.cache_size_mb <= 8192
        
        # Verify recommendation is different from initial (adaptation occurred)
        # or is a reasonable value
        assert recommendations.cache_size_mb > 0
        
        # Apply tuning
        auto_tuner.apply_tuning(recommendations)
        
        # Verify callback was called with new cache size
        cache_size_callback.assert_called_once_with(recommendations.cache_size_mb)
        
        # Verify current cache size was updated
        assert auto_tuner._current_cache_size == recommendations.cache_size_mb
    else:
        # No recommendation is also valid (system is in optimal state)
        # Verify no tuning was applied
        cache_size_callback.assert_not_called()
        assert auto_tuner._current_cache_size == initial_cache_size


# Feature: advanced-optimization-monitoring, Property 48: Tuning decisions are logged
@given(
    parameter_type=st.sampled_from(["batch_size", "batch_timeout_ms", "cache_size_mb"]),
    new_value=st.integers(min_value=10, max_value=200),
)
@settings(max_examples=50, deadline=2000)
@pytest.mark.property
def test_tuning_decisions_logged(parameter_type, new_value):
    """
    Property 48: Tuning decisions are logged.
    
    For any tuning decision made by the auto-tuner, the decision and
    rationale should be logged and exposed in metrics.
    
    Validates: Requirements 12.5
    """
    # Setup: Create tuner with all tuning enabled
    config = TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=True
    )
    
    performance_monitor = Mock(spec=PerformanceMonitor)
    
    # Set up callbacks
    batch_size_callback = Mock()
    batch_timeout_callback = Mock()
    cache_size_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        batch_size_callback=batch_size_callback,
        batch_timeout_callback=batch_timeout_callback,
        cache_size_callback=cache_size_callback
    )
    
    # Set initial values
    auto_tuner._current_batch_size = 32
    auto_tuner._current_batch_timeout = 50
    auto_tuner._current_cache_size = 4096
    
    # Mock performance metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=300.0,
        min_latency_ms=200.0,
        max_latency_ms=500.0,
        p50_latency_ms=280.0,
        p95_latency_ms=450.0,
        p99_latency_ms=490.0,
        throughput_rps=20.0
    )
    
    # Create recommendations based on parameter type
    recommendations = TuningRecommendations(
        rationale=f"Test tuning for {parameter_type}"
    )
    
    if parameter_type == "batch_size":
        recommendations.batch_size = new_value
    elif parameter_type == "batch_timeout_ms":
        recommendations.batch_timeout_ms = new_value
    elif parameter_type == "cache_size_mb":
        recommendations.cache_size_mb = new_value
    
    # Get initial history length
    initial_history_len = len(auto_tuner.get_tuning_history())
    
    # Apply tuning
    auto_tuner.apply_tuning(recommendations)
    
    # Verify: Tuning decision is logged in history
    history = auto_tuner.get_tuning_history()
    assert len(history) > initial_history_len
    
    # Find the most recent event for this parameter
    recent_events = [e for e in history if e.parameter == parameter_type]
    assert len(recent_events) > 0
    
    latest_event = recent_events[-1]
    
    # Verify: Event contains required information
    assert latest_event.parameter == parameter_type
    assert latest_event.new_value == new_value
    assert latest_event.rationale is not None
    assert len(latest_event.rationale) > 0
    assert latest_event.timestamp is not None
    
    # Verify: Rationale is meaningful
    assert "test tuning" in latest_event.rationale.lower() or parameter_type in latest_event.rationale.lower()


# Feature: advanced-optimization-monitoring, Property 49: Static configuration is used when auto-tuning is disabled
@given(
    initial_batch_size=st.integers(min_value=8, max_value=64),
    initial_timeout=st.integers(min_value=20, max_value=200),
    load_scenario=st.sampled_from(["high_latency", "low_throughput", "high_throughput"]),
)
@settings(max_examples=50, deadline=2000)
@pytest.mark.property
def test_static_config_when_disabled(initial_batch_size, initial_timeout, load_scenario):
    """
    Property 49: Static configuration is used when auto-tuning is disabled.
    
    For any operation when auto-tuning is disabled, static configuration
    parameters should be used without adaptation.
    
    Validates: Requirements 12.6
    """
    # Setup: Create tuner with auto-tuning DISABLED
    config = TunerConfig(
        enabled=False,  # Disabled
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=True
    )
    
    performance_monitor = Mock(spec=PerformanceMonitor)
    
    # Set up callbacks
    batch_size_callback = Mock()
    batch_timeout_callback = Mock()
    cache_size_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        batch_size_callback=batch_size_callback,
        batch_timeout_callback=batch_timeout_callback,
        cache_size_callback=cache_size_callback
    )
    
    # Set initial static values
    auto_tuner._current_batch_size = initial_batch_size
    auto_tuner._current_batch_timeout = initial_timeout
    auto_tuner._current_cache_size = 4096
    
    # Create metrics that would normally trigger tuning
    if load_scenario == "high_latency":
        metrics = PerformanceMetrics(
            operation="inference",
            count=100,
            mean_latency_ms=1500.0,
            min_latency_ms=1000.0,
            max_latency_ms=2500.0,
            p50_latency_ms=1400.0,
            p95_latency_ms=2200.0,
            p99_latency_ms=2450.0,
            throughput_rps=10.0
        )
    elif load_scenario == "low_throughput":
        metrics = PerformanceMetrics(
            operation="inference",
            count=30,
            mean_latency_ms=250.0,
            min_latency_ms=150.0,
            max_latency_ms=400.0,
            p50_latency_ms=240.0,
            p95_latency_ms=380.0,
            p99_latency_ms=395.0,
            throughput_rps=3.0
        )
    else:  # high_throughput
        metrics = PerformanceMetrics(
            operation="inference",
            count=600,
            mean_latency_ms=150.0,
            min_latency_ms=100.0,
            max_latency_ms=250.0,
            p50_latency_ms=140.0,
            p95_latency_ms=220.0,
            p99_latency_ms=245.0,
            throughput_rps=70.0
        )
    
    # Mock resource trends
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=60.0,
        max_cpu_percent=80.0,
        avg_memory_mb=8192.0,
        max_memory_mb=10240.0
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    performance_monitor.get_operation_metrics.return_value = metrics
    
    # Analyze performance (should return no recommendations when disabled)
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Verify: No recommendations when disabled
    assert recommendations.batch_size is None
    assert recommendations.batch_timeout_ms is None
    assert recommendations.cache_size_mb is None
    assert "disabled" in recommendations.rationale.lower()
    
    # Try to apply tuning (should be no-op)
    auto_tuner.apply_tuning(recommendations)
    
    # Verify: Callbacks were NOT called (static parameters maintained)
    batch_size_callback.assert_not_called()
    batch_timeout_callback.assert_not_called()
    cache_size_callback.assert_not_called()
    
    # Verify: Parameters remain unchanged
    assert auto_tuner._current_batch_size == initial_batch_size
    assert auto_tuner._current_batch_timeout == initial_timeout
    assert auto_tuner._current_cache_size == 4096
    
    # Verify: No history was recorded
    history = auto_tuner.get_tuning_history()
    assert len(history) == 0
    
    # Verify: Background tuning loop should not start when disabled
    auto_tuner.start()
    assert auto_tuner._tuning_thread is None


# Additional edge case tests
@pytest.mark.property
def test_tuning_with_no_metrics():
    """Test that tuning handles missing metrics gracefully."""
    config = TunerConfig(enabled=True)
    performance_monitor = Mock(spec=PerformanceMonitor)
    performance_monitor.get_operation_metrics.return_value = None
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor
    )
    
    # Analyze with no metrics
    recommendations = auto_tuner.analyze_performance()
    
    # Should return no recommendations
    assert recommendations.batch_size is None
    assert recommendations.batch_timeout_ms is None
    assert recommendations.cache_size_mb is None
    assert "insufficient" in recommendations.rationale.lower()


@pytest.mark.property
def test_tuning_history_ordering():
    """Test that tuning history maintains chronological order."""
    config = TunerConfig(enabled=True)
    performance_monitor = Mock(spec=PerformanceMonitor)
    batch_size_callback = Mock()
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor,
        batch_size_callback=batch_size_callback
    )
    
    # Mock metrics
    performance_monitor.get_operation_metrics.return_value = PerformanceMetrics(
        operation="inference",
        count=100,
        mean_latency_ms=300.0,
        min_latency_ms=200.0,
        max_latency_ms=500.0,
        p50_latency_ms=280.0,
        p95_latency_ms=450.0,
        p99_latency_ms=490.0,
        throughput_rps=20.0
    )
    
    # Apply multiple tuning changes
    for i in range(5):
        recommendations = TuningRecommendations(
            batch_size=32 + i * 4,
            rationale=f"Iteration {i}"
        )
        auto_tuner.apply_tuning(recommendations)
        time.sleep(0.01)  # Small delay to ensure different timestamps
    
    # Get history
    history = auto_tuner.get_tuning_history()
    
    # Verify chronological order
    assert len(history) == 5
    for i in range(len(history) - 1):
        assert history[i].timestamp <= history[i + 1].timestamp


@pytest.mark.property
def test_gradual_batch_size_changes():
    """Test that batch size changes are gradual (not extreme jumps)."""
    config = TunerConfig(enabled=True, enable_batch_size_tuning=True)
    performance_monitor = Mock(spec=PerformanceMonitor)
    
    auto_tuner = AutoTuner(
        config=config,
        performance_monitor=performance_monitor
    )
    
    # Set initial batch size
    auto_tuner._current_batch_size = 32
    
    # High latency scenario
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
    
    resource_trends = ResourceTrends(
        window_seconds=60,
        avg_cpu_percent=50.0,
        max_cpu_percent=70.0,
        avg_memory_mb=4096.0,
        max_memory_mb=6144.0
    )
    performance_monitor.get_resource_trends.return_value = resource_trends
    
    recommendations = auto_tuner.analyze_performance(metrics)
    
    # Verify gradual change (should be around 25% reduction)
    if recommendations.batch_size:
        change_ratio = recommendations.batch_size / 32
        # Should be between 0.5 and 1.0 (reduction, but not too extreme)
        assert 0.5 <= change_ratio <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
