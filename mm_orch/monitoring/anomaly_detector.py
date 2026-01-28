"""
Anomaly detection and alerting for the MuAI Multi-Model Orchestration System.

This module provides anomaly detection based on configurable thresholds for latency,
error rates, resource usage, and throughput. Alerts are sent to configured destinations
with rate limiting to prevent alert storms.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from mm_orch.monitoring.config import AnomalyConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE = "resource"  # Changed from MEMORY to RESOURCE
    THROUGHPUT = "throughput"


@dataclass
class Alert:
    """
    Alert for anomaly detection.

    Attributes:
        alert_type: Type of alert (latency, error_rate, memory, throughput)
        severity: Alert severity (warning, error, critical)
        message: Human-readable alert message
        timestamp: When the alert was triggered
        metadata: Additional context (thresholds, actual values, etc.)
    """

    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AnomalyDetector:
    """
    Detects performance anomalies and triggers alerts.

    Monitors:
    - Latency exceeding thresholds
    - Error rates exceeding thresholds
    - Memory usage exceeding thresholds
    - Throughput dropping below thresholds

    Supports configurable alert destinations:
    - Structured logging (always enabled)
    - Webhook notifications
    - Prometheus Alertmanager

    Implements alert rate limiting to prevent alert storms.
    """

    def __init__(
        self, config: AnomalyConfig, performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize anomaly detector.

        Args:
            config: Anomaly detection configuration
            performance_monitor: Optional performance monitor for metrics queries
        """
        self.config = config
        self.performance_monitor = performance_monitor

        # Alert rate limiting
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_lock = Lock()

        # Alert history
        self._alert_history: List[Alert] = []
        self._history_lock = Lock()

        # Request tracking for error rate calculation
        self._request_history: List[Dict[str, Any]] = []
        self._request_lock = Lock()

        logger.info(
            f"AnomalyDetector initialized with thresholds: "
            f"latency={config.latency_threshold_ms}ms, "
            f"error_rate={config.error_rate_threshold}, "
            f"memory={config.memory_threshold_percent}%, "
            f"throughput={config.throughput_threshold_rps}rps"
        )

    def check_latency(self, operation: str, latency_ms: float) -> Optional[Alert]:
        """
        Check if latency exceeds threshold.

        Args:
            operation: Name of the operation
            latency_ms: Measured latency in milliseconds

        Returns:
            Alert if threshold exceeded and not rate-limited, None otherwise
        """
        if not self.config.enabled:
            return None

        if latency_ms <= self.config.latency_threshold_ms:
            return None

        # Determine severity based on how much threshold is exceeded
        excess_ratio = latency_ms / self.config.latency_threshold_ms
        if excess_ratio >= 3.0:
            severity = AlertSeverity.CRITICAL
        elif excess_ratio >= 2.0:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        alert = Alert(
            alert_type=AlertType.LATENCY.value,
            severity=severity.value,
            message=f"Latency threshold exceeded for {operation}: {latency_ms:.2f}ms (threshold: {self.config.latency_threshold_ms}ms)",
            timestamp=datetime.now(),
            metadata={
                "operation": operation,
                "latency_ms": latency_ms,
                "threshold_ms": self.config.latency_threshold_ms,
                "excess_ratio": excess_ratio,
            },
        )

        return self._maybe_send_alert(alert)

    def check_error_rate(
        self, component: Optional[str] = None, window_seconds: int = 60
    ) -> Optional[Alert]:
        """
        Check if error rate exceeds threshold.

        Uses performance_monitor if available, otherwise uses internal request history.

        Args:
            component: Component name (None = all components)
            window_seconds: Time window for rate calculation

        Returns:
            Alert if threshold exceeded and not rate-limited, None otherwise
        """
        if not self.config.enabled:
            return None

        # Try to get error rate from performance monitor first
        if self.performance_monitor:
            # Get total requests and errors from performance monitor
            throughput = self.performance_monitor.get_throughput(window_seconds)
            error_rate_per_sec = self.performance_monitor.get_error_rate(component, window_seconds)

            if throughput == 0:
                return None

            # Calculate error rate as fraction of requests
            total_requests = throughput * window_seconds
            total_errors = error_rate_per_sec * window_seconds
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0
        else:
            # Fall back to internal request history
            error_rate = self._calculate_error_rate_from_history(component, window_seconds)

            # Get request count from history
            with self._request_lock:
                cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
                recent_requests = [r for r in self._request_history if r["timestamp"] > cutoff_time]
                if component:
                    recent_requests = [
                        r for r in recent_requests if r.get("component") == component
                    ]
                total_requests = len(recent_requests)
                total_errors = sum(1 for r in recent_requests if not r["success"])

        if error_rate <= self.config.error_rate_threshold:
            return None

        # Determine severity
        excess_ratio = error_rate / self.config.error_rate_threshold
        if excess_ratio >= 3.0:
            severity = AlertSeverity.CRITICAL
        elif excess_ratio >= 2.0:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        component_str = f" in {component}" if component else ""
        alert = Alert(
            alert_type=AlertType.ERROR_RATE.value,
            severity=severity.value,
            message=f"Error rate threshold exceeded{component_str}: {error_rate:.2%} (threshold: {self.config.error_rate_threshold:.2%})",
            timestamp=datetime.now(),
            metadata={
                "component": component,
                "error_rate": error_rate,
                "threshold": self.config.error_rate_threshold,
                "window_seconds": window_seconds,
                "total_requests": total_requests if "total_requests" in locals() else 0,
                "total_errors": total_errors if "total_errors" in locals() else 0,
            },
        )

        return self._maybe_send_alert(alert)

    def check_resources(
        self,
        cpu_percent: Optional[float] = None,
        memory_percent: Optional[float] = None,
        gpu_memory_percent: Optional[float] = None,
    ) -> Optional[Alert]:
        """
        Check if resource usage exceeds thresholds.

        Args:
            cpu_percent: CPU usage percentage (0-100)
            memory_percent: Memory usage percentage (0-100)
            gpu_memory_percent: GPU memory usage percentage (0-100)

        Returns:
            Alert if threshold exceeded and not rate-limited, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check memory threshold (prioritize GPU memory if provided)
        memory_to_check = gpu_memory_percent if gpu_memory_percent is not None else memory_percent

        if memory_to_check is None or memory_to_check <= self.config.memory_threshold_percent:
            return None

        # Determine severity
        excess_ratio = memory_to_check / self.config.memory_threshold_percent
        if excess_ratio >= 1.1:  # 110% of threshold
            severity = AlertSeverity.CRITICAL
        elif excess_ratio >= 1.05:  # 105% of threshold
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        memory_type = "GPU memory" if gpu_memory_percent is not None else "Memory"
        alert = Alert(
            alert_type=AlertType.RESOURCE.value,  # Changed from MEMORY to RESOURCE
            severity=severity.value,
            message=f"{memory_type} threshold exceeded: {memory_to_check:.1f}% (threshold: {self.config.memory_threshold_percent}%)",
            timestamp=datetime.now(),
            metadata={
                "memory_percent": memory_to_check,
                "threshold_percent": self.config.memory_threshold_percent,
                "cpu_percent": cpu_percent,
                "memory_type": "gpu" if gpu_memory_percent is not None else "system",
            },
        )

        return self._maybe_send_alert(alert)

    def check_throughput(self, throughput_rps: float, window_seconds: int = 60) -> Optional[Alert]:
        """
        Check if throughput drops below threshold.

        Args:
            throughput_rps: Current throughput in requests per second
            window_seconds: Time window for throughput calculation

        Returns:
            Alert if threshold violated and not rate-limited, None otherwise
        """
        if not self.config.enabled:
            return None

        if throughput_rps >= self.config.throughput_threshold_rps:
            return None

        # Determine severity based on how far below threshold
        deficit_ratio = (
            self.config.throughput_threshold_rps / throughput_rps
            if throughput_rps > 0
            else float("inf")
        )
        if deficit_ratio >= 3.0:
            severity = AlertSeverity.CRITICAL
        elif deficit_ratio >= 2.0:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        alert = Alert(
            alert_type=AlertType.THROUGHPUT.value,
            severity=severity.value,
            message=f"Throughput below threshold: {throughput_rps:.2f}rps (threshold: {self.config.throughput_threshold_rps}rps)",
            timestamp=datetime.now(),
            metadata={
                "throughput_rps": throughput_rps,
                "threshold_rps": self.config.throughput_threshold_rps,
                "window_seconds": window_seconds,
                "deficit_ratio": deficit_ratio,
            },
        )

        return self._maybe_send_alert(alert)

    def send_alert(self, alert: Alert):
        """
        Send alert to configured destinations.

        Always logs the alert. Optionally sends to webhook or Alertmanager.

        Args:
            alert: Alert to send
        """
        # Always log
        self._log_alert(alert)

        # Store in history
        with self._history_lock:
            self._alert_history.append(alert)
            # Keep only last 1000 alerts
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-1000:]

        # Send to additional destinations
        for destination in self.config.alert_destinations:
            if destination == "log":
                # Already logged above
                continue
            elif destination == "webhook":
                self._send_webhook(alert)
            elif destination == "alertmanager":
                self._send_alertmanager(alert)

    def get_alert_history(self, alert_type: Optional[str] = None, limit: int = 100) -> List[Alert]:
        """
        Get recent alert history.

        Args:
            alert_type: Filter by alert type (None = all types)
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        with self._history_lock:
            if alert_type:
                filtered = [a for a in self._alert_history if a.alert_type == alert_type]
            else:
                filtered = self._alert_history

            return filtered[-limit:]

    # Private helper methods

    def _maybe_send_alert(self, alert: Alert) -> Optional[Alert]:
        """
        Send alert if not rate-limited.

        Args:
            alert: Alert to potentially send

        Returns:
            Alert if sent, None if rate-limited
        """
        with self._alert_lock:
            # Check rate limiting
            alert_key = alert.alert_type
            last_time = self._last_alert_time.get(alert_key)

            if last_time:
                time_since_last = (datetime.now() - last_time).total_seconds()
                if time_since_last < self.config.alert_rate_limit_seconds:
                    logger.debug(
                        f"Alert rate-limited: {alert.alert_type} "
                        f"(last alert {time_since_last:.1f}s ago)"
                    )
                    return None

            # Send alert
            self.send_alert(alert)
            self._last_alert_time[alert_key] = datetime.now()

            return alert

    def _record_request(self, success: bool = True, component: Optional[str] = None):
        """
        Record a request for error rate tracking.

        This is an internal method used to track request success/failure
        for error rate calculation when performance_monitor is not available.

        Args:
            success: Whether the request was successful
            component: Optional component name
        """
        with self._request_lock:
            self._request_history.append(
                {"timestamp": datetime.now(), "success": success, "component": component}
            )

            # Keep only last hour of history
            cutoff_time = datetime.now() - timedelta(hours=1)
            self._request_history = [
                r for r in self._request_history if r["timestamp"] > cutoff_time
            ]

    def _calculate_error_rate_from_history(
        self, component: Optional[str] = None, window_seconds: int = 60
    ) -> float:
        """
        Calculate error rate from internal request history.

        Args:
            component: Component name (None = all components)
            window_seconds: Time window for rate calculation

        Returns:
            Error rate as a fraction (0.0 to 1.0)
        """
        with self._request_lock:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)

            # Filter requests in time window
            recent_requests = [r for r in self._request_history if r["timestamp"] > cutoff_time]

            # Filter by component if specified
            if component:
                recent_requests = [r for r in recent_requests if r.get("component") == component]

            if not recent_requests:
                return 0.0

            # Calculate error rate
            total_requests = len(recent_requests)
            failed_requests = sum(1 for r in recent_requests if not r["success"])

            return failed_requests / total_requests if total_requests > 0 else 0.0

    def _log_alert(self, alert: Alert):
        """Log alert with structured data."""
        log_level = {
            AlertSeverity.WARNING.value: logging.WARNING,
            AlertSeverity.ERROR.value: logging.ERROR,
            AlertSeverity.CRITICAL.value: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            f"ALERT [{alert.severity.upper()}] {alert.message}",
            extra={
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
                "metadata": alert.metadata,
            },
        )

    def _send_webhook(self, alert: Alert):
        """Send alert to webhook."""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, cannot send webhook")
            return

        if not self.config.webhook_url:
            logger.warning("webhook destination enabled but webhook_url not configured")
            return

        try:
            payload = alert.to_dict()
            response = requests.post(self.config.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.debug(f"Alert sent to webhook: {alert.alert_type}")
        except Exception as e:
            logger.error(f"Failed to send alert to webhook: {e}")

    def _send_alertmanager(self, alert: Alert):
        """Send alert to Prometheus Alertmanager."""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, cannot send to Alertmanager")
            return

        # Alertmanager expects alerts in a specific format
        # This is a simplified implementation
        try:
            # Default Alertmanager endpoint
            alertmanager_url = "http://localhost:9093/api/v1/alerts"

            payload = [
                {
                    "labels": {
                        "alertname": alert.alert_type,
                        "severity": alert.severity,
                        "service": "muai-orchestration",
                    },
                    "annotations": {
                        "summary": alert.message,
                        "description": json.dumps(alert.metadata),
                    },
                    "startsAt": alert.timestamp.isoformat(),
                }
            ]

            response = requests.post(alertmanager_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.debug(f"Alert sent to Alertmanager: {alert.alert_type}")
        except Exception as e:
            logger.error(f"Failed to send alert to Alertmanager: {e}")
