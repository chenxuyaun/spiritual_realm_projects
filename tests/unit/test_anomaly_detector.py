"""
Unit tests for AnomalyDetector.

Tests specific examples, edge cases, and integration scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from mm_orch.monitoring.anomaly_detector import (
    AnomalyDetector,
    Alert,
    AlertSeverity,
    AlertType
)
from mm_orch.monitoring.config import AnomalyConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor


class TestAnomalyDetectorBasics:
    """Test basic anomaly detector functionality."""
    
    def test_initialization(self):
        """Test detector initialization with config."""
        config = AnomalyConfig(
            enabled=True,
            latency_threshold_ms=500.0,
            error_rate_threshold=0.05
        )
        detector = AnomalyDetector(config)
        
        assert detector.config == config
        assert detector.performance_monitor is None
    
    def test_initialization_with_performance_monitor(self):
        """Test detector initialization with performance monitor."""
        config = AnomalyConfig()
        perf_monitor = Mock(spec=PerformanceMonitor)
        detector = AnomalyDetector(config, perf_monitor)
        
        assert detector.performance_monitor == perf_monitor
    
    def test_disabled_detector(self):
        """Test that disabled detector doesn't trigger alerts."""
        config = AnomalyConfig(enabled=False, latency_threshold_ms=100.0)
        detector = AnomalyDetector(config)
        
        # Try to trigger alerts
        assert detector.check_latency("test", 1000.0) is None
        assert detector.check_throughput(0.1) is None
        assert detector.check_resources(memory_percent=99.0) is None


class TestLatencyAlerts:
    """Test latency threshold checking."""
    
    def test_latency_below_threshold(self):
        """Test no alert when latency is below threshold."""
        config = AnomalyConfig(latency_threshold_ms=1000.0)
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 500.0)
        assert alert is None
    
    def test_latency_at_threshold(self):
        """Test no alert when latency equals threshold."""
        config = AnomalyConfig(latency_threshold_ms=1000.0)
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 1000.0)
        assert alert is None
    
    def test_latency_above_threshold_warning(self):
        """Test warning alert for moderate latency excess."""
        config = AnomalyConfig(
            latency_threshold_ms=1000.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 1500.0)
        assert alert is not None
        assert alert.alert_type == AlertType.LATENCY.value
        assert alert.severity == AlertSeverity.WARNING.value
        assert "1500.00ms" in alert.message
    
    def test_latency_above_threshold_error(self):
        """Test error alert for high latency excess (2x threshold)."""
        config = AnomalyConfig(
            latency_threshold_ms=1000.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 2500.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.ERROR.value
    
    def test_latency_above_threshold_critical(self):
        """Test critical alert for very high latency excess (3x threshold)."""
        config = AnomalyConfig(
            latency_threshold_ms=1000.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 3500.0)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL.value


class TestErrorRateAlerts:
    """Test error rate threshold checking."""
    
    def test_error_rate_requires_performance_monitor(self):
        """Test that error rate checking requires performance monitor."""
        config = AnomalyConfig()
        detector = AnomalyDetector(config)
        
        # Without performance monitor, should return None
        alert = detector.check_error_rate()
        assert alert is None
    
    def test_error_rate_below_threshold(self):
        """Test no alert when error rate is below threshold."""
        config = AnomalyConfig(error_rate_threshold=0.05)
        perf_monitor = Mock(spec=PerformanceMonitor)
        perf_monitor.get_throughput.return_value = 10.0  # 10 rps
        perf_monitor.get_error_rate.return_value = 0.2  # 0.2 errors/sec = 2% error rate
        
        detector = AnomalyDetector(config, perf_monitor)
        alert = detector.check_error_rate()
        
        assert alert is None
    
    def test_error_rate_above_threshold(self):
        """Test alert when error rate exceeds threshold."""
        config = AnomalyConfig(
            error_rate_threshold=0.05,
            alert_rate_limit_seconds=0
        )
        perf_monitor = Mock(spec=PerformanceMonitor)
        perf_monitor.get_throughput.return_value = 10.0  # 10 rps
        perf_monitor.get_error_rate.return_value = 1.0  # 1 error/sec = 10% error rate
        
        detector = AnomalyDetector(config, perf_monitor)
        alert = detector.check_error_rate()
        
        assert alert is not None
        assert alert.alert_type == AlertType.ERROR_RATE.value
        assert "10.00%" in alert.message or "0.10" in alert.message


class TestMemoryAlerts:
    """Test memory threshold checking."""
    
    def test_memory_below_threshold(self):
        """Test no alert when memory is below threshold."""
        config = AnomalyConfig(memory_threshold_percent=90.0)
        detector = AnomalyDetector(config)
        
        alert = detector.check_resources(memory_percent=80.0)
        assert alert is None
    
    def test_memory_above_threshold(self):
        """Test alert when memory exceeds threshold."""
        config = AnomalyConfig(
            memory_threshold_percent=90.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_resources(memory_percent=95.0)
        assert alert is not None
        assert alert.alert_type == AlertType.RESOURCE.value  # Changed from MEMORY to RESOURCE
        assert "95.0%" in alert.message
    
    def test_gpu_memory_prioritized(self):
        """Test that GPU memory is checked when both are provided."""
        config = AnomalyConfig(
            memory_threshold_percent=90.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        # System memory high, GPU memory low - no alert
        alert = detector.check_resources(
            memory_percent=95.0,
            gpu_memory_percent=80.0
        )
        assert alert is None
        
        # System memory low, GPU memory high - alert
        alert = detector.check_resources(
            memory_percent=80.0,
            gpu_memory_percent=95.0
        )
        assert alert is not None
        assert "GPU" in alert.message


class TestThroughputAlerts:
    """Test throughput threshold checking."""
    
    def test_throughput_above_threshold(self):
        """Test no alert when throughput is above threshold."""
        config = AnomalyConfig(throughput_threshold_rps=1.0)
        detector = AnomalyDetector(config)
        
        alert = detector.check_throughput(5.0)
        assert alert is None
    
    def test_throughput_below_threshold(self):
        """Test alert when throughput drops below threshold."""
        config = AnomalyConfig(
            throughput_threshold_rps=10.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_throughput(5.0)
        assert alert is not None
        assert alert.alert_type == AlertType.THROUGHPUT.value
        assert "5.00rps" in alert.message


class TestAlertRateLimiting:
    """Test alert rate limiting functionality."""
    
    def test_rate_limiting_prevents_duplicate_alerts(self):
        """Test that rate limiting prevents alert storms."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_rate_limit_seconds=60
        )
        detector = AnomalyDetector(config)
        
        # First alert should be sent
        alert1 = detector.check_latency("test", 200.0)
        assert alert1 is not None
        
        # Second alert within rate limit window should be blocked
        alert2 = detector.check_latency("test", 300.0)
        assert alert2 is None
        
        # Third alert also blocked
        alert3 = detector.check_latency("test", 400.0)
        assert alert3 is None
    
    def test_rate_limiting_per_alert_type(self):
        """Test that rate limiting is per alert type."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            throughput_threshold_rps=10.0,
            alert_rate_limit_seconds=60
        )
        detector = AnomalyDetector(config)
        
        # Latency alert
        alert1 = detector.check_latency("test", 200.0)
        assert alert1 is not None
        
        # Throughput alert (different type) should not be blocked
        alert2 = detector.check_throughput(5.0)
        assert alert2 is not None
        
        # Another latency alert should be blocked
        alert3 = detector.check_latency("test", 300.0)
        assert alert3 is None


class TestAlertDelivery:
    """Test alert delivery to different destinations."""
    
    def test_alert_logged(self):
        """Test that alerts are always logged."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_destinations=["log"],
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("test", 200.0)
        assert alert is not None
        
        # Check alert is in history
        history = detector.get_alert_history()
        assert len(history) == 1
        assert history[0].alert_type == AlertType.LATENCY.value
    
    @patch('mm_orch.monitoring.anomaly_detector.requests')
    def test_webhook_delivery(self, mock_requests):
        """Test alert delivery to webhook."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_destinations=["log", "webhook"],
            webhook_url="http://example.com/webhook",
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        # Mock successful webhook response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        alert = detector.check_latency("test", 200.0)
        assert alert is not None
        
        # Verify webhook was called
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "http://example.com/webhook"
        assert "json" in call_args[1]


class TestAlertHistory:
    """Test alert history tracking."""
    
    def test_get_all_alert_history(self):
        """Test retrieving all alert history."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        # Trigger multiple alerts
        detector.check_latency("test1", 200.0)
        detector.check_latency("test2", 300.0)
        detector.check_latency("test3", 400.0)
        
        history = detector.get_alert_history()
        assert len(history) == 3
    
    def test_get_filtered_alert_history(self):
        """Test retrieving filtered alert history by type."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            throughput_threshold_rps=10.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        # Trigger different types of alerts
        detector.check_latency("test", 200.0)
        detector.check_throughput(5.0)
        detector.check_latency("test", 300.0)
        
        # Get only latency alerts
        latency_history = detector.get_alert_history(alert_type=AlertType.LATENCY.value)
        assert len(latency_history) == 2
        assert all(a.alert_type == AlertType.LATENCY.value for a in latency_history)
        
        # Get only throughput alerts
        throughput_history = detector.get_alert_history(alert_type=AlertType.THROUGHPUT.value)
        assert len(throughput_history) == 1
        assert throughput_history[0].alert_type == AlertType.THROUGHPUT.value
    
    def test_alert_history_limit(self):
        """Test that alert history respects limit parameter."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        # Trigger many alerts
        for i in range(10):
            detector.check_latency(f"test{i}", 200.0)
        
        # Get limited history
        history = detector.get_alert_history(limit=5)
        assert len(history) == 5


class TestAlertMetadata:
    """Test alert metadata and serialization."""
    
    def test_alert_to_dict(self):
        """Test alert serialization to dictionary."""
        alert = Alert(
            alert_type=AlertType.LATENCY.value,
            severity=AlertSeverity.WARNING.value,
            message="Test alert",
            timestamp=datetime.now(),
            metadata={"key": "value"}
        )
        
        alert_dict = alert.to_dict()
        assert alert_dict["alert_type"] == AlertType.LATENCY.value
        assert alert_dict["severity"] == AlertSeverity.WARNING.value
        assert alert_dict["message"] == "Test alert"
        assert "timestamp" in alert_dict
        assert alert_dict["metadata"]["key"] == "value"
    
    def test_alert_metadata_contains_context(self):
        """Test that alerts contain relevant metadata."""
        config = AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_rate_limit_seconds=0
        )
        detector = AnomalyDetector(config)
        
        alert = detector.check_latency("inference", 250.0)
        assert alert is not None
        
        # Check metadata
        assert "operation" in alert.metadata
        assert "latency_ms" in alert.metadata
        assert "threshold_ms" in alert.metadata
        assert "excess_ratio" in alert.metadata
        assert alert.metadata["operation"] == "inference"
        assert alert.metadata["latency_ms"] == 250.0
        assert alert.metadata["threshold_ms"] == 100.0
