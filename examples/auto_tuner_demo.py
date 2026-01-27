"""
Demo script for AutoTuner functionality.

This script demonstrates how the AutoTuner analyzes performance metrics
and generates tuning recommendations for batch size, timeout, and cache size.
"""

import time
from mm_orch.optimization.auto_tuner import AutoTuner
from mm_orch.optimization.config import TunerConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetrics


def batch_size_callback(new_size: int):
    """Callback for batch size changes."""
    print(f"  → Batch size changed to: {new_size}")


def batch_timeout_callback(new_timeout: int):
    """Callback for timeout changes."""
    print(f"  → Batch timeout changed to: {new_timeout}ms")


def cache_size_callback(new_size: int):
    """Callback for cache size changes."""
    print(f"  → Cache size changed to: {new_size}MB")


def main():
    """Run AutoTuner demo."""
    print("=" * 70)
    print("AutoTuner Demo - Performance Analysis and Tuning")
    print("=" * 70)
    print()
    
    # Create configuration
    print("1. Creating AutoTuner configuration...")
    config = TunerConfig(
        enabled=True,
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=True
    )
    print(f"   ✓ Observation window: {config.observation_window_seconds}s")
    print(f"   ✓ Tuning interval: {config.tuning_interval_seconds}s")
    print()
    
    # Create performance monitor
    print("2. Creating PerformanceMonitor...")
    monitor = PerformanceMonitor(
        max_history_seconds=3600,
        resource_sample_interval=10
    )
    print("   ✓ Performance monitor initialized")
    print()
    
    # Create AutoTuner
    print("3. Creating AutoTuner with callbacks...")
    tuner = AutoTuner(
        config=config,
        performance_monitor=monitor,
        batch_size_callback=batch_size_callback,
        batch_timeout_callback=batch_timeout_callback,
        cache_size_callback=cache_size_callback
    )
    print("   ✓ AutoTuner initialized")
    print()
    
    # Scenario 1: High latency
    print("=" * 70)
    print("Scenario 1: High Latency Detection")
    print("=" * 70)
    print("Simulating high latency scenario (p95 = 1800ms)...")
    
    high_latency_metrics = PerformanceMetrics(
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
    
    tuner._current_batch_size = 32
    
    # Mock resource trends to avoid blocking psutil calls
    from mm_orch.monitoring.performance_monitor import ResourceTrends
    monitor.get_resource_trends = lambda window_seconds: ResourceTrends(
        window_seconds=window_seconds,
        avg_cpu_percent=50.0,
        max_cpu_percent=70.0,
        avg_memory_mb=4096.0,
        max_memory_mb=6144.0
    )
    
    recommendations = tuner.analyze_performance(high_latency_metrics)
    
    print(f"\nAnalysis Results:")
    print(f"  • Batch size recommendation: {recommendations.batch_size}")
    print(f"  • Timeout recommendation: {recommendations.batch_timeout_ms}")
    print(f"  • Cache size recommendation: {recommendations.cache_size_mb}")
    print(f"  • Rationale: {recommendations.rationale}")
    
    if recommendations.batch_size:
        print(f"\nApplying tuning...")
        tuner.apply_tuning(recommendations)
    print()
    
    # Scenario 2: Low throughput with good latency
    print("=" * 70)
    print("Scenario 2: Low Throughput with Good Latency")
    print("=" * 70)
    print("Simulating low throughput scenario (5 rps, p95 = 420ms)...")
    
    low_throughput_metrics = PerformanceMetrics(
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
    
    tuner._current_batch_size = 16
    recommendations = tuner.analyze_performance(low_throughput_metrics)
    
    print(f"\nAnalysis Results:")
    print(f"  • Batch size recommendation: {recommendations.batch_size}")
    print(f"  • Timeout recommendation: {recommendations.batch_timeout_ms}")
    print(f"  • Cache size recommendation: {recommendations.cache_size_mb}")
    print(f"  • Rationale: {recommendations.rationale}")
    
    if recommendations.batch_size:
        print(f"\nApplying tuning...")
        tuner.apply_tuning(recommendations)
    print()
    
    # Scenario 3: High throughput
    print("=" * 70)
    print("Scenario 3: High Throughput")
    print("=" * 70)
    print("Simulating high throughput scenario (60 rps)...")
    
    high_throughput_metrics = PerformanceMetrics(
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
    
    tuner._current_batch_timeout = 50
    recommendations = tuner.analyze_performance(high_throughput_metrics)
    
    print(f"\nAnalysis Results:")
    print(f"  • Batch size recommendation: {recommendations.batch_size}")
    print(f"  • Timeout recommendation: {recommendations.batch_timeout_ms}")
    print(f"  • Cache size recommendation: {recommendations.cache_size_mb}")
    print(f"  • Rationale: {recommendations.rationale}")
    
    if recommendations.batch_timeout_ms:
        print(f"\nApplying tuning...")
        tuner.apply_tuning(recommendations)
    print()
    
    # Show tuning history
    print("=" * 70)
    print("Tuning History")
    print("=" * 70)
    history = tuner.get_tuning_history()
    
    if history:
        for i, event in enumerate(history, 1):
            print(f"\n{i}. {event.parameter}")
            print(f"   Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Change: {event.old_value} → {event.new_value}")
            print(f"   Rationale: {event.rationale}")
    else:
        print("No tuning events recorded yet.")
    print()
    
    # Scenario 4: Optimal performance
    print("=" * 70)
    print("Scenario 4: Optimal Performance")
    print("=" * 70)
    print("Simulating optimal performance scenario...")
    
    optimal_metrics = PerformanceMetrics(
        operation="inference",
        count=200,
        mean_latency_ms=400.0,
        min_latency_ms=300.0,
        max_latency_ms=600.0,
        p50_latency_ms=380.0,
        p95_latency_ms=550.0,
        p99_latency_ms=590.0,
        throughput_rps=25.0
    )
    
    recommendations = tuner.analyze_performance(optimal_metrics)
    
    print(f"\nAnalysis Results:")
    print(f"  • Batch size recommendation: {recommendations.batch_size}")
    print(f"  • Timeout recommendation: {recommendations.batch_timeout_ms}")
    print(f"  • Cache size recommendation: {recommendations.cache_size_mb}")
    print(f"  • Rationale: {recommendations.rationale}")
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • AutoTuner analyzes performance metrics to detect patterns")
    print("  • High latency triggers batch size reduction")
    print("  • Low throughput with good latency triggers batch size increase")
    print("  • High throughput triggers timeout reduction")
    print("  • All tuning decisions are logged with rationale")
    print("  • Tuning decisions are exposed in Prometheus metrics (if enabled)")
    print("  • Automatic rollback on performance degradation")
    print("  • Callbacks allow integration with actual system components")
    print()
    
    # Scenario 5: Degradation detection and rollback
    print("=" * 70)
    print("Scenario 5: Degradation Detection and Rollback")
    print("=" * 70)
    print("Simulating performance degradation after tuning...")
    
    # Set baseline
    tuner._baseline_latency = 500.0
    tuner._baseline_throughput = 20.0
    tuner._current_batch_size = 32
    
    # Apply a tuning change
    print("\nApplying tuning change (batch size 32 → 64)...")
    recommendations = PerformanceMetrics(
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
    
    from mm_orch.optimization.auto_tuner import TuningRecommendations
    tuning_rec = TuningRecommendations(
        batch_size=64,
        rationale="Test tuning for degradation demo"
    )
    tuner.apply_tuning(tuning_rec)
    
    # Simulate degraded performance
    print("\nChecking performance after tuning...")
    monitor.get_operation_metrics = lambda op, window_seconds: PerformanceMetrics(
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
    
    degraded = tuner.check_for_degradation()
    print(f"  • Degradation detected: {degraded}")
    
    if degraded:
        print("\nInitiating rollback...")
        tuner.rollback_last_tuning()
        print("  ✓ Rollback complete")
        
        # Show updated history
        print("\nUpdated tuning history:")
        history = tuner.get_tuning_history()
        for event in history[-2:]:
            print(f"  • {event.parameter}: {event.old_value} → {event.new_value}")
            print(f"    Rationale: {event.rationale}")
    print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • AutoTuner analyzes performance metrics to detect patterns")
    print("  • High latency triggers batch size reduction")
    print("  • Low throughput with good latency triggers batch size increase")
    print("  • High throughput triggers timeout reduction")
    print("  • All tuning decisions are logged with rationale")
    print("  • Tuning decisions are exposed in Prometheus metrics (if enabled)")
    print("  • Automatic rollback on performance degradation (>20% worse)")
    print("  • Callbacks allow integration with actual system components")
    print()
    
    # Scenario 6: Disabled auto-tuning
    print("=" * 70)
    print("Scenario 6: Auto-Tuning Disabled Mode")
    print("=" * 70)
    print("Demonstrating static configuration when auto-tuning is disabled...")
    
    # Create disabled configuration
    disabled_config = TunerConfig(
        enabled=False,  # Disable auto-tuning
        observation_window_seconds=60,
        tuning_interval_seconds=30,
        enable_batch_size_tuning=True,
        enable_timeout_tuning=True,
        enable_cache_size_tuning=True
    )
    
    # Create disabled tuner
    disabled_tuner = AutoTuner(
        config=disabled_config,
        performance_monitor=monitor,
        batch_size_callback=batch_size_callback,
        batch_timeout_callback=batch_timeout_callback,
        cache_size_callback=cache_size_callback
    )
    
    print(f"\n✓ Created AutoTuner with enabled={disabled_config.enabled}")
    
    # Try to start (should not start)
    print("\nAttempting to start background tuning loop...")
    disabled_tuner.start()
    print("  → Background loop not started (auto-tuning is disabled)")
    
    # Try to analyze performance
    print("\nAnalyzing performance with high latency...")
    disabled_tuner._current_batch_size = 32
    recommendations = disabled_tuner.analyze_performance(high_latency_metrics)
    
    print(f"\nAnalysis Results:")
    print(f"  • Batch size recommendation: {recommendations.batch_size}")
    print(f"  • Timeout recommendation: {recommendations.batch_timeout_ms}")
    print(f"  • Cache size recommendation: {recommendations.cache_size_mb}")
    print(f"  • Rationale: {recommendations.rationale}")
    
    # Try to apply tuning
    print("\nAttempting to apply tuning...")
    test_recommendations = TuningRecommendations(
        batch_size=64,
        batch_timeout_ms=100,
        cache_size_mb=8192,
        rationale="Test tuning (should not be applied)"
    )
    disabled_tuner.apply_tuning(test_recommendations)
    print("  → Tuning not applied (auto-tuning is disabled)")
    print("  → Static configuration parameters maintained")
    
    # Verify no history
    disabled_history = disabled_tuner.get_tuning_history()
    print(f"\nTuning history: {len(disabled_history)} events")
    print("  → No tuning events recorded (as expected)")
    
    print("\n" + "=" * 70)
    print("Key Takeaways - Disabled Mode:")
    print("=" * 70)
    print("  • When auto-tuning is disabled (enabled=False):")
    print("    - Background tuning loop does not start")
    print("    - analyze_performance() returns empty recommendations")
    print("    - apply_tuning() does not modify parameters")
    print("    - Static configuration parameters are maintained")
    print("    - No tuning history is recorded")
    print("  • This satisfies Requirement 12.6:")
    print("    'WHERE auto-tuning is disabled, THE System SHALL use")
    print("     static configuration parameters'")
    print()


if __name__ == "__main__":
    main()
