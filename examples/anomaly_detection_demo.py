"""
Demo of anomaly detection and alerting capabilities.

This example demonstrates:
1. Configuring anomaly detection thresholds
2. Checking various metrics against thresholds
3. Alert rate limiting
4. Alert history tracking
"""

import time
from mm_orch.monitoring.anomaly_detector import AnomalyDetector, AlertType
from mm_orch.monitoring.config import AnomalyConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor


def demo_latency_alerts():
    """Demonstrate latency threshold alerts."""
    print("\n=== Latency Alerts Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=500.0,
        alert_rate_limit_seconds=5
    )
    detector = AnomalyDetector(config)
    
    # Normal latency - no alert
    print("\n1. Normal latency (300ms):")
    alert = detector.check_latency("inference", 300.0)
    print(f"   Alert triggered: {alert is not None}")
    
    # High latency - warning alert
    print("\n2. High latency (800ms - 1.6x threshold):")
    alert = detector.check_latency("inference", 800.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")
    
    # Very high latency - error alert
    print("\n3. Very high latency (1200ms - 2.4x threshold):")
    alert = detector.check_latency("inference", 1200.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")
    
    # Extremely high latency - critical alert
    print("\n4. Extremely high latency (2000ms - 4x threshold):")
    alert = detector.check_latency("inference", 2000.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")


def demo_rate_limiting():
    """Demonstrate alert rate limiting."""
    print("\n=== Alert Rate Limiting Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        alert_rate_limit_seconds=3
    )
    detector = AnomalyDetector(config)
    
    print("\nTriggering multiple alerts rapidly:")
    for i in range(5):
        alert = detector.check_latency("test", 200.0)
        status = "SENT" if alert else "BLOCKED (rate limited)"
        print(f"   Alert {i+1}: {status}")
        time.sleep(0.5)
    
    print("\nWaiting for rate limit window to expire...")
    time.sleep(3)
    
    print("\nTriggering alert after rate limit window:")
    alert = detector.check_latency("test", 200.0)
    status = "SENT" if alert else "BLOCKED"
    print(f"   Alert: {status}")


def demo_memory_alerts():
    """Demonstrate memory threshold alerts."""
    print("\n=== Memory Alerts Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        memory_threshold_percent=85.0,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    # Normal memory usage
    print("\n1. Normal memory usage (70%):")
    alert = detector.check_resources(memory_percent=70.0)
    print(f"   Alert triggered: {alert is not None}")
    
    # High memory usage
    print("\n2. High memory usage (90%):")
    alert = detector.check_resources(memory_percent=90.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")
    
    # GPU memory prioritized over system memory
    print("\n3. System memory high (95%), GPU memory normal (80%):")
    alert = detector.check_resources(
        memory_percent=95.0,
        gpu_memory_percent=80.0
    )
    print(f"   Alert triggered: {alert is not None} (GPU memory is checked)")
    
    print("\n4. System memory normal (80%), GPU memory high (95%):")
    alert = detector.check_resources(
        memory_percent=80.0,
        gpu_memory_percent=95.0
    )
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")


def demo_throughput_alerts():
    """Demonstrate throughput threshold alerts."""
    print("\n=== Throughput Alerts Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        throughput_threshold_rps=10.0,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    # Normal throughput
    print("\n1. Normal throughput (15 rps):")
    alert = detector.check_throughput(15.0)
    print(f"   Alert triggered: {alert is not None}")
    
    # Low throughput
    print("\n2. Low throughput (5 rps):")
    alert = detector.check_throughput(5.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")
    
    # Very low throughput
    print("\n3. Very low throughput (2 rps):")
    alert = detector.check_throughput(2.0)
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")


def demo_error_rate_alerts():
    """Demonstrate error rate threshold alerts."""
    print("\n=== Error Rate Alerts Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        error_rate_threshold=0.05,  # 5%
        alert_rate_limit_seconds=0
    )
    
    # Create mock performance monitor
    from unittest.mock import Mock
    perf_monitor = Mock(spec=PerformanceMonitor)
    
    detector = AnomalyDetector(config, perf_monitor)
    
    # Normal error rate (2%)
    print("\n1. Normal error rate (2%):")
    perf_monitor.get_throughput.return_value = 10.0  # 10 rps
    perf_monitor.get_error_rate.return_value = 0.2  # 0.2 errors/sec = 2%
    alert = detector.check_error_rate()
    print(f"   Alert triggered: {alert is not None}")
    
    # High error rate (10%)
    print("\n2. High error rate (10%):")
    perf_monitor.get_throughput.return_value = 10.0
    perf_monitor.get_error_rate.return_value = 1.0  # 1 error/sec = 10%
    alert = detector.check_error_rate()
    if alert:
        print(f"   Alert: {alert.severity.upper()} - {alert.message}")


def demo_alert_history():
    """Demonstrate alert history tracking."""
    print("\n=== Alert History Demo ===")
    
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        throughput_threshold_rps=10.0,
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    
    # Trigger various alerts
    print("\nTriggering multiple alerts:")
    detector.check_latency("inference", 200.0)
    detector.check_latency("preprocessing", 150.0)
    detector.check_throughput(5.0)
    detector.check_latency("postprocessing", 300.0)
    
    # Get all alerts
    print("\nAll alerts:")
    all_alerts = detector.get_alert_history()
    for i, alert in enumerate(all_alerts, 1):
        print(f"   {i}. [{alert.alert_type}] {alert.message}")
    
    # Get filtered alerts
    print("\nLatency alerts only:")
    latency_alerts = detector.get_alert_history(alert_type=AlertType.LATENCY.value)
    for i, alert in enumerate(latency_alerts, 1):
        print(f"   {i}. {alert.message}")
    
    print("\nThroughput alerts only:")
    throughput_alerts = detector.get_alert_history(alert_type=AlertType.THROUGHPUT.value)
    for i, alert in enumerate(throughput_alerts, 1):
        print(f"   {i}. {alert.message}")


def demo_alert_destinations():
    """Demonstrate alert delivery to different destinations."""
    print("\n=== Alert Destinations Demo ===")
    
    # Log only (default)
    print("\n1. Log destination only:")
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        alert_destinations=["log"],
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    alert = detector.check_latency("test", 200.0)
    print(f"   Alert sent to: {config.alert_destinations}")
    
    # Multiple destinations
    print("\n2. Multiple destinations (log + webhook):")
    config = AnomalyConfig(
        enabled=True,
        latency_threshold_ms=100.0,
        alert_destinations=["log", "webhook"],
        webhook_url="http://example.com/webhook",
        alert_rate_limit_seconds=0
    )
    detector = AnomalyDetector(config)
    alert = detector.check_latency("test", 200.0)
    print(f"   Alert sent to: {config.alert_destinations}")
    print(f"   Webhook URL: {config.webhook_url}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Anomaly Detection and Alerting Demo")
    print("=" * 60)
    
    demo_latency_alerts()
    demo_rate_limiting()
    demo_memory_alerts()
    demo_throughput_alerts()
    demo_error_rate_alerts()
    demo_alert_history()
    demo_alert_destinations()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
