"""
Auto-tuning for performance optimization.

This module implements automatic performance tuning that adjusts system parameters
based on workload patterns and performance metrics. Supports tuning of batch sizes,
timeout parameters, and cache sizes with gradual adaptation and rollback on degradation.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread, Lock, Event
from typing import Dict, List, Optional, Any, Callable
import logging

from mm_orch.logger import get_logger
from mm_orch.optimization.config import TunerConfig
from mm_orch.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetrics
from mm_orch.monitoring.prometheus_exporter import PrometheusExporter

logger = get_logger(__name__)


@dataclass
class TuningRecommendations:
    """
    Recommendations from auto-tuner.

    Attributes:
        batch_size: Recommended batch size (None = no change)
        batch_timeout_ms: Recommended batch timeout in milliseconds (None = no change)
        cache_size_mb: Recommended cache size in megabytes (None = no change)
        rationale: Explanation for the recommendations
        timestamp: When recommendations were generated
    """

    batch_size: Optional[int] = None
    batch_timeout_ms: Optional[int] = None
    cache_size_mb: Optional[int] = None
    rationale: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TuningEvent:
    """
    Record of a tuning decision.

    Attributes:
        timestamp: When tuning was applied
        parameter: Parameter that was tuned
        old_value: Previous value
        new_value: New value
        rationale: Reason for the change
        performance_before: Performance metrics before tuning
        performance_after: Performance metrics after tuning (populated later)
    """

    timestamp: datetime
    parameter: str
    old_value: Any
    new_value: Any
    rationale: str
    performance_before: Optional[PerformanceMetrics] = None
    performance_after: Optional[PerformanceMetrics] = None


class AutoTuner:
    """
    Automatically adjusts performance parameters based on workload patterns.

    Monitors system performance and adapts:
    - Batch sizes based on latency and throughput targets
    - Timeout parameters based on request arrival patterns
    - Cache sizes based on hit rates and memory pressure

    Implements gradual adaptation with rollback on performance degradation.
    """

    def __init__(
        self,
        config: TunerConfig,
        performance_monitor: PerformanceMonitor,
        batch_size_callback: Optional[Callable[[int], None]] = None,
        batch_timeout_callback: Optional[Callable[[int], None]] = None,
        cache_size_callback: Optional[Callable[[int], None]] = None,
        prometheus_exporter: Optional[PrometheusExporter] = None,
    ):
        """
        Initialize auto-tuner.

        Args:
            config: Tuner configuration
            performance_monitor: Performance monitor for metrics
            batch_size_callback: Callback to apply batch size changes
            batch_timeout_callback: Callback to apply timeout changes
            cache_size_callback: Callback to apply cache size changes
            prometheus_exporter: Optional Prometheus exporter for metrics
        """
        self.config = config
        self.performance_monitor = performance_monitor
        self.prometheus_exporter = prometheus_exporter

        # Callbacks for applying tuning
        self.batch_size_callback = batch_size_callback
        self.batch_timeout_callback = batch_timeout_callback
        self.cache_size_callback = cache_size_callback

        # Tuning history
        self._tuning_history: deque[TuningEvent] = deque(maxlen=100)
        self._history_lock = Lock()

        # Background tuning thread
        self._tuning_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Current parameter values (tracked for rollback)
        self._current_batch_size: Optional[int] = None
        self._current_batch_timeout: Optional[int] = None
        self._current_cache_size: Optional[int] = None

        # Performance baselines for rollback detection
        self._baseline_latency: Optional[float] = None
        self._baseline_throughput: Optional[float] = None

        # Rollback state
        self._last_tuning_time: Optional[datetime] = None
        self._rollback_pending: bool = False

        # Initialize Prometheus metrics if exporter provided
        if self.prometheus_exporter and self.prometheus_exporter.is_enabled():
            self._init_metrics()

        logger.info(
            f"AutoTuner initialized with {config.observation_window_seconds}s observation window, "
            f"{config.tuning_interval_seconds}s tuning interval"
        )

    def _init_metrics(self):
        """Initialize Prometheus metrics for tuning decisions."""
        try:
            from prometheus_client import Counter, Gauge

            # Counter for tuning decisions
            self.tuning_decisions = Counter(
                "auto_tuning_decisions_total",
                "Total number of auto-tuning decisions",
                ["parameter", "direction"],  # direction: increase, decrease, no_change
            )

            # Gauge for current parameter values
            self.current_batch_size_metric = Gauge(
                "auto_tuning_batch_size", "Current batch size set by auto-tuner"
            )

            self.current_batch_timeout_metric = Gauge(
                "auto_tuning_batch_timeout_ms",
                "Current batch timeout in milliseconds set by auto-tuner",
            )

            self.current_cache_size_metric = Gauge(
                "auto_tuning_cache_size_mb", "Current cache size in MB set by auto-tuner"
            )

            # Counter for rollbacks
            self.rollback_counter = Counter(
                "auto_tuning_rollbacks_total",
                "Total number of auto-tuning rollbacks due to degradation",
                ["parameter"],
            )

            logger.info("Auto-tuning Prometheus metrics initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize auto-tuning metrics: {e}")

    def start(self):
        """Start background tuning loop."""
        if not self.config.enabled:
            logger.info("Auto-tuning is disabled, not starting tuning loop")
            return

        if self._tuning_thread and self._tuning_thread.is_alive():
            logger.warning("Tuning thread already running")
            return

        self._stop_event.clear()
        self._tuning_thread = Thread(target=self._tuning_loop, daemon=True)
        self._tuning_thread.start()
        logger.info("Auto-tuning loop started")

    def stop(self):
        """Stop background tuning loop."""
        if not self._tuning_thread:
            return

        self._stop_event.set()
        self._tuning_thread.join(timeout=5.0)
        logger.info("Auto-tuning loop stopped")

    def analyze_performance(
        self, metrics: Optional[PerformanceMetrics] = None
    ) -> TuningRecommendations:
        """
        Analyze performance metrics and generate tuning recommendations.

        Args:
            metrics: Performance metrics to analyze (None = query from monitor)

        Returns:
            TuningRecommendations with suggested parameter changes
        """
        if not self.config.enabled:
            return TuningRecommendations(rationale="Auto-tuning is disabled")

        # Get metrics from monitor if not provided
        if metrics is None:
            metrics = self.performance_monitor.get_operation_metrics(
                "inference", window_seconds=self.config.observation_window_seconds
            )

        if metrics is None:
            return TuningRecommendations(rationale="Insufficient performance data for analysis")

        recommendations = TuningRecommendations()
        rationale_parts = []

        # Analyze batch size
        if self.config.enable_batch_size_tuning:
            batch_rec = self._analyze_batch_size(metrics)
            if batch_rec:
                recommendations.batch_size = batch_rec
                rationale_parts.append(f"Batch size: {batch_rec}")

        # Analyze timeout
        if self.config.enable_timeout_tuning:
            timeout_rec = self._analyze_timeout(metrics)
            if timeout_rec:
                recommendations.batch_timeout_ms = timeout_rec
                rationale_parts.append(f"Timeout: {timeout_rec}ms")

        # Analyze cache size
        if self.config.enable_cache_size_tuning:
            cache_rec = self._analyze_cache_size(metrics)
            if cache_rec:
                recommendations.cache_size_mb = cache_rec
                rationale_parts.append(f"Cache: {cache_rec}MB")

        if rationale_parts:
            recommendations.rationale = "Recommendations: " + ", ".join(rationale_parts)
        else:
            recommendations.rationale = "No tuning needed, performance is optimal"

        logger.info(f"Performance analysis: {recommendations.rationale}")
        return recommendations

    def apply_tuning(self, recommendations: TuningRecommendations):
        """
        Apply tuning recommendations to the system.

        Args:
            recommendations: Tuning recommendations to apply

        Requirement 12.5: Log all tuning decisions and expose in metrics
        """
        if not self.config.enabled:
            logger.info("Auto-tuning is disabled, not applying recommendations")
            return

        # Get current performance baseline
        current_metrics = self.performance_monitor.get_operation_metrics(
            "inference", window_seconds=self.config.observation_window_seconds
        )

        # Apply batch size tuning
        if recommendations.batch_size and self.batch_size_callback:
            old_value = self._current_batch_size
            self.batch_size_callback(recommendations.batch_size)
            self._current_batch_size = recommendations.batch_size

            event = TuningEvent(
                timestamp=datetime.now(),
                parameter="batch_size",
                old_value=old_value,
                new_value=recommendations.batch_size,
                rationale=recommendations.rationale,
                performance_before=current_metrics,
            )

            with self._history_lock:
                self._tuning_history.append(event)

            # Log tuning decision with rationale
            logger.info(
                f"Applied batch size tuning: {old_value} -> {recommendations.batch_size}. "
                f"Rationale: {recommendations.rationale}"
            )

            # Expose in metrics
            self._record_tuning_metric("batch_size", old_value, recommendations.batch_size)

        # Apply timeout tuning
        if recommendations.batch_timeout_ms and self.batch_timeout_callback:
            old_value = self._current_batch_timeout
            self.batch_timeout_callback(recommendations.batch_timeout_ms)
            self._current_batch_timeout = recommendations.batch_timeout_ms

            event = TuningEvent(
                timestamp=datetime.now(),
                parameter="batch_timeout_ms",
                old_value=old_value,
                new_value=recommendations.batch_timeout_ms,
                rationale=recommendations.rationale,
                performance_before=current_metrics,
            )

            with self._history_lock:
                self._tuning_history.append(event)

            # Log tuning decision with rationale
            logger.info(
                f"Applied timeout tuning: {old_value} -> {recommendations.batch_timeout_ms}ms. "
                f"Rationale: {recommendations.rationale}"
            )

            # Expose in metrics
            self._record_tuning_metric(
                "batch_timeout_ms", old_value, recommendations.batch_timeout_ms
            )

        # Apply cache size tuning
        if recommendations.cache_size_mb and self.cache_size_callback:
            old_value = self._current_cache_size
            self.cache_size_callback(recommendations.cache_size_mb)
            self._current_cache_size = recommendations.cache_size_mb

            event = TuningEvent(
                timestamp=datetime.now(),
                parameter="cache_size_mb",
                old_value=old_value,
                new_value=recommendations.cache_size_mb,
                rationale=recommendations.rationale,
                performance_before=current_metrics,
            )

            with self._history_lock:
                self._tuning_history.append(event)

            # Log tuning decision with rationale
            logger.info(
                f"Applied cache size tuning: {old_value} -> {recommendations.cache_size_mb}MB. "
                f"Rationale: {recommendations.rationale}"
            )

            # Expose in metrics
            self._record_tuning_metric("cache_size_mb", old_value, recommendations.cache_size_mb)

        # Update baseline for rollback detection
        if current_metrics:
            self._baseline_latency = current_metrics.p95_latency_ms
            self._baseline_throughput = current_metrics.throughput_rps

        # Track when tuning was applied
        self._last_tuning_time = datetime.now()

    def get_tuning_history(self) -> List[TuningEvent]:
        """
        Get history of tuning decisions.

        Returns:
            List of tuning events in chronological order
        """
        with self._history_lock:
            return list(self._tuning_history)

    def check_for_degradation(self) -> bool:
        """
        Check if recent tuning caused performance degradation.

        Returns:
            True if degradation detected, False otherwise
        """
        if not self._baseline_latency or not self._baseline_throughput:
            return False

        current_metrics = self.performance_monitor.get_operation_metrics(
            "inference", window_seconds=60  # Check recent performance
        )

        if not current_metrics:
            return False

        # Check for significant degradation (>20% worse)
        latency_degradation = current_metrics.p95_latency_ms > self._baseline_latency * 1.2
        throughput_degradation = current_metrics.throughput_rps < self._baseline_throughput * 0.8

        if latency_degradation or throughput_degradation:
            logger.warning(
                f"Performance degradation detected: "
                f"latency {self._baseline_latency:.1f} -> {current_metrics.p95_latency_ms:.1f}ms, "
                f"throughput {self._baseline_throughput:.1f} -> {current_metrics.throughput_rps:.1f} rps"
            )
            return True

        return False

    def rollback_last_tuning(self):
        """
        Rollback the last tuning decision due to performance degradation.

        Requirement 12.5: Add rollback on performance degradation
        """
        if not self._tuning_history:
            logger.warning("No tuning history to rollback")
            return

        with self._history_lock:
            # Get the most recent tuning event
            last_event = self._tuning_history[-1]

        # Check if we already rolled back this event
        if self._rollback_pending:
            logger.warning("Rollback already in progress")
            return

        self._rollback_pending = True

        try:
            # Rollback based on parameter type
            if last_event.parameter == "batch_size" and self.batch_size_callback:
                if last_event.old_value is not None:
                    self.batch_size_callback(last_event.old_value)
                    self._current_batch_size = last_event.old_value
                    logger.warning(
                        f"Rolled back batch size: {last_event.new_value} -> {last_event.old_value}"
                    )
                    self._record_rollback_metric("batch_size")

            elif last_event.parameter == "batch_timeout_ms" and self.batch_timeout_callback:
                if last_event.old_value is not None:
                    self.batch_timeout_callback(last_event.old_value)
                    self._current_batch_timeout = last_event.old_value
                    logger.warning(
                        f"Rolled back batch timeout: {last_event.new_value} -> {last_event.old_value}ms"
                    )
                    self._record_rollback_metric("batch_timeout_ms")

            elif last_event.parameter == "cache_size_mb" and self.cache_size_callback:
                if last_event.old_value is not None:
                    self.cache_size_callback(last_event.old_value)
                    self._current_cache_size = last_event.old_value
                    logger.warning(
                        f"Rolled back cache size: {last_event.new_value} -> {last_event.old_value}MB"
                    )
                    self._record_rollback_metric("cache_size_mb")

            # Create rollback event in history
            rollback_event = TuningEvent(
                timestamp=datetime.now(),
                parameter=last_event.parameter,
                old_value=last_event.new_value,
                new_value=last_event.old_value,
                rationale=f"Rollback due to performance degradation after tuning at {last_event.timestamp}",
                performance_before=None,
            )

            with self._history_lock:
                self._tuning_history.append(rollback_event)

        finally:
            self._rollback_pending = False

    # Private helper methods

    def _record_tuning_metric(self, parameter: str, old_value: Any, new_value: Any):
        """
        Record tuning decision in Prometheus metrics.

        Args:
            parameter: Parameter that was tuned
            old_value: Previous value
            new_value: New value
        """
        if not self.prometheus_exporter or not self.prometheus_exporter.is_enabled():
            return

        try:
            # Determine direction of change
            if old_value is None:
                direction = "initial"
            elif new_value > old_value:
                direction = "increase"
            elif new_value < old_value:
                direction = "decrease"
            else:
                direction = "no_change"

            # Increment decision counter
            self.tuning_decisions.labels(parameter=parameter, direction=direction).inc()

            # Update current value gauge
            if parameter == "batch_size":
                self.current_batch_size_metric.set(new_value)
            elif parameter == "batch_timeout_ms":
                self.current_batch_timeout_metric.set(new_value)
            elif parameter == "cache_size_mb":
                self.current_cache_size_metric.set(new_value)

        except Exception as e:
            logger.warning(f"Failed to record tuning metric: {e}")

    def _record_rollback_metric(self, parameter: str):
        """
        Record rollback in Prometheus metrics.

        Args:
            parameter: Parameter that was rolled back
        """
        if not self.prometheus_exporter or not self.prometheus_exporter.is_enabled():
            return

        try:
            self.rollback_counter.labels(parameter=parameter).inc()
        except Exception as e:
            logger.warning(f"Failed to record rollback metric: {e}")

    def _tuning_loop(self):
        """Background loop for periodic tuning."""
        logger.info("Tuning loop started")

        while not self._stop_event.is_set():
            try:
                # Wait for tuning interval
                if self._stop_event.wait(timeout=self.config.tuning_interval_seconds):
                    break

                # Analyze and apply tuning
                recommendations = self.analyze_performance()

                if (
                    recommendations.batch_size
                    or recommendations.batch_timeout_ms
                    or recommendations.cache_size_mb
                ):
                    self.apply_tuning(recommendations)

                    # Wait a bit and check for degradation
                    time.sleep(30)
                    if self.check_for_degradation():
                        logger.warning("Degradation detected after tuning, initiating rollback")
                        self.rollback_last_tuning()

            except Exception as e:
                logger.error(f"Error in tuning loop: {e}", exc_info=True)
                time.sleep(10)  # Back off on error

        logger.info("Tuning loop stopped")

    def _analyze_batch_size(self, metrics: PerformanceMetrics) -> Optional[int]:
        """
        Analyze batch size and recommend adjustments.

        Strategy:
        - High latency (>1s p95) -> reduce batch size
        - Low throughput (<10 rps) and low latency -> increase batch size
        - Gradual changes (±25% per adjustment)

        Args:
            metrics: Performance metrics

        Returns:
            Recommended batch size or None
        """
        current_batch = self._current_batch_size or 32  # Default assumption

        # High latency - reduce batch size
        if metrics.p95_latency_ms > 1000:
            new_batch = max(1, int(current_batch * 0.75))
            if new_batch != current_batch:
                logger.info(
                    f"High latency detected ({metrics.p95_latency_ms:.1f}ms), "
                    f"reducing batch size to {new_batch}"
                )
                return new_batch

        # Low throughput and acceptable latency - increase batch size
        if metrics.throughput_rps < 10 and metrics.p95_latency_ms < 500:
            new_batch = min(128, int(current_batch * 1.25))
            if new_batch != current_batch:
                logger.info(
                    f"Low throughput ({metrics.throughput_rps:.1f} rps) with good latency, "
                    f"increasing batch size to {new_batch}"
                )
                return new_batch

        return None

    def _analyze_timeout(self, metrics: PerformanceMetrics) -> Optional[int]:
        """
        Analyze batch timeout and recommend adjustments.

        Strategy:
        - High throughput (>50 rps) -> reduce timeout for faster batching
        - Low throughput (<5 rps) -> increase timeout to accumulate requests

        Args:
            metrics: Performance metrics

        Returns:
            Recommended timeout in milliseconds or None
        """
        current_timeout = self._current_batch_timeout or 50  # Default assumption

        # High throughput - reduce timeout
        if metrics.throughput_rps > 50:
            new_timeout = max(10, int(current_timeout * 0.75))
            if new_timeout != current_timeout:
                logger.info(
                    f"High throughput ({metrics.throughput_rps:.1f} rps), "
                    f"reducing timeout to {new_timeout}ms"
                )
                return new_timeout

        # Low throughput - increase timeout
        if metrics.throughput_rps < 5:
            new_timeout = min(200, int(current_timeout * 1.5))
            if new_timeout != current_timeout:
                logger.info(
                    f"Low throughput ({metrics.throughput_rps:.1f} rps), "
                    f"increasing timeout to {new_timeout}ms"
                )
                return new_timeout

        return None

    def _analyze_cache_size(self, metrics: PerformanceMetrics) -> Optional[int]:
        """
        Analyze cache size and recommend adjustments.

        Strategy:
        - Get resource trends to check memory pressure
        - If memory usage is high (>80%), reduce cache
        - If memory usage is low (<50%) and cache could help, increase cache

        Args:
            metrics: Performance metrics

        Returns:
            Recommended cache size in MB or None
        """
        current_cache = self._current_cache_size or 4096  # Default assumption

        # Get resource trends
        resource_trends = self.performance_monitor.get_resource_trends(
            window_seconds=self.config.observation_window_seconds
        )

        # High memory usage - reduce cache
        if resource_trends.max_memory_mb > 0:
            # Estimate total memory (rough approximation)
            import psutil

            total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
            memory_percent = (resource_trends.avg_memory_mb / total_memory_mb) * 100

            if memory_percent > 80:
                new_cache = max(1024, int(current_cache * 0.75))
                if new_cache != current_cache:
                    logger.info(
                        f"High memory usage ({memory_percent:.1f}%), "
                        f"reducing cache to {new_cache}MB"
                    )
                    return new_cache

            # Low memory usage - increase cache
            if memory_percent < 50:
                new_cache = min(8192, int(current_cache * 1.25))
                if new_cache != current_cache:
                    logger.info(
                        f"Low memory usage ({memory_percent:.1f}%), "
                        f"increasing cache to {new_cache}MB"
                    )
                    return new_cache

        return None
