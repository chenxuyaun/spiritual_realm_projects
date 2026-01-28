"""
GPU monitoring integration.

This module integrates GPU utilities with the monitoring system to expose
per-GPU metrics through Prometheus.
"""

import threading
import time
from typing import Optional

from mm_orch.logger import get_logger
from mm_orch.optimization.gpu_utils import get_gpu_manager

logger = get_logger(__name__)


class GPUMonitor:
    """
    Monitor GPU metrics and expose them through Prometheus.

    Continuously monitors GPU health, memory, utilization, and temperature,
    and records metrics for Prometheus scraping.

    Attributes:
        gpu_manager: GPU manager instance
        prometheus_exporter: Prometheus exporter instance (optional)
        monitoring_interval: Interval between monitoring updates (seconds)
        _monitoring_thread: Background monitoring thread
        _stop_monitoring: Flag to stop monitoring

    Example:
        >>> from mm_orch.monitoring import PrometheusExporter
        >>> exporter = PrometheusExporter(port=9090)
        >>> monitor = GPUMonitor(exporter, monitoring_interval=5.0)
        >>> monitor.start()
        >>> # ... monitoring runs in background ...
        >>> monitor.stop()
    """

    def __init__(
        self, prometheus_exporter: Optional[object] = None, monitoring_interval: float = 5.0
    ):
        """
        Initialize GPU monitor.

        Args:
            prometheus_exporter: PrometheusExporter instance (optional)
            monitoring_interval: Interval between monitoring updates in seconds
        """
        self.gpu_manager = get_gpu_manager()
        self.prometheus_exporter = prometheus_exporter
        self.monitoring_interval = monitoring_interval
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        logger.info(f"GPUMonitor initialized with interval={monitoring_interval}s")

    def start(self):
        """
        Start background GPU monitoring.

        Launches a background thread that continuously monitors GPU metrics
        and records them to Prometheus.

        Example:
            >>> monitor = GPUMonitor(exporter)
            >>> monitor.start()
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            logger.warning("GPU monitoring already running")
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="GPUMonitor"
        )
        self._monitoring_thread.start()
        logger.info("GPU monitoring started")

    def stop(self):
        """
        Stop background GPU monitoring.

        Signals the monitoring thread to stop and waits for it to finish.

        Example:
            >>> monitor = GPUMonitor(exporter)
            >>> monitor.start()
            >>> # ... later ...
            >>> monitor.stop()
        """
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            logger.warning("GPU monitoring not running")
            return

        logger.info("Stopping GPU monitoring...")
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=10.0)

        if self._monitoring_thread.is_alive():
            logger.warning("GPU monitoring thread did not stop gracefully")
        else:
            logger.info("GPU monitoring stopped")

    def _monitoring_loop(self):
        """
        Background monitoring loop.

        Continuously monitors GPU metrics and records them to Prometheus
        until stop is signaled.
        """
        logger.info("GPU monitoring loop started")

        while not self._stop_monitoring.is_set():
            try:
                # Collect GPU metrics
                self._collect_and_record_metrics()

                # Wait for next interval
                self._stop_monitoring.wait(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}", exc_info=True)
                # Continue monitoring despite errors
                time.sleep(self.monitoring_interval)

        logger.info("GPU monitoring loop stopped")

    def _collect_and_record_metrics(self):
        """
        Collect GPU metrics and record them to Prometheus.

        Collects metrics for all GPUs including:
        - Memory usage and availability
        - Utilization percentage
        - Temperature
        - Health status
        """
        try:
            # Get all GPU metrics
            all_metrics = self.gpu_manager.get_all_gpu_metrics()

            # Get GPU health status
            health_status = self.gpu_manager.monitor_gpu_health()

            # Record to Prometheus if exporter is available
            if self.prometheus_exporter and self.prometheus_exporter.is_enabled():
                for gpu_id, metrics in all_metrics.items():
                    # Add health status to metrics
                    is_healthy, _ = health_status.get(gpu_id, (True, None))
                    metrics["is_healthy"] = is_healthy

                    # Record per-GPU metrics
                    self.prometheus_exporter.record_per_gpu_metrics(
                        gpu_id=gpu_id,
                        memory_used_mb=metrics.get("memory_used_mb", 0),
                        memory_available_mb=metrics.get("memory_available_mb", 0),
                        utilization_percent=metrics.get("utilization_percent", 0),
                        temperature_celsius=metrics.get("temperature_celsius"),
                        is_healthy=is_healthy,
                    )

            # Log summary
            gpu_count = len(all_metrics)
            unhealthy_count = sum(1 for is_healthy, _ in health_status.values() if not is_healthy)

            if unhealthy_count > 0:
                logger.warning(f"GPU health check: {unhealthy_count}/{gpu_count} GPUs unhealthy")
            else:
                logger.debug(f"GPU health check: all {gpu_count} GPUs healthy")

        except Exception as e:
            logger.error(f"Failed to collect and record GPU metrics: {e}", exc_info=True)

    def get_current_metrics(self):
        """
        Get current GPU metrics snapshot.

        Returns:
            Dictionary mapping GPU ID to metrics dictionary

        Example:
            >>> monitor = GPUMonitor()
            >>> metrics = monitor.get_current_metrics()
            >>> print(metrics[0]['memory_used_mb'])
        """
        return self.gpu_manager.get_all_gpu_metrics()

    def is_monitoring(self) -> bool:
        """
        Check if monitoring is currently running.

        Returns:
            True if monitoring thread is active, False otherwise

        Example:
            >>> monitor = GPUMonitor()
            >>> monitor.start()
            >>> print(monitor.is_monitoring())
            True
        """
        return (
            self._monitoring_thread is not None
            and self._monitoring_thread.is_alive()
            and not self._stop_monitoring.is_set()
        )


def create_gpu_monitor(
    prometheus_exporter: Optional[object] = None,
    monitoring_interval: float = 5.0,
    auto_start: bool = True,
) -> GPUMonitor:
    """
    Create and optionally start a GPU monitor.

    Args:
        prometheus_exporter: PrometheusExporter instance (optional)
        monitoring_interval: Interval between monitoring updates in seconds
        auto_start: Whether to automatically start monitoring

    Returns:
        GPUMonitor instance

    Example:
        >>> from mm_orch.monitoring import PrometheusExporter
        >>> exporter = PrometheusExporter(port=9090)
        >>> monitor = create_gpu_monitor(exporter, auto_start=True)
    """
    monitor = GPUMonitor(prometheus_exporter, monitoring_interval)

    if auto_start:
        monitor.start()

    return monitor
