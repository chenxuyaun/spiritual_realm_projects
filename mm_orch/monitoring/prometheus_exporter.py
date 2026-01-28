"""Prometheus metrics exporter for system monitoring."""

import threading
from typing import Dict, Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from mm_orch.logger import get_logger

logger = get_logger(__name__)


class PrometheusExporter:
    """
    Exports performance metrics in Prometheus format.

    Provides HTTP endpoint for Prometheus scraping and methods to record
    various system metrics including inference latency, throughput, and
    resource usage.

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
    """

    def __init__(self, config_or_port=None, enabled: bool = True):
        """
        Initialize Prometheus exporter.

        Args:
            config_or_port: Either a PrometheusConfig object or an integer port number
            enabled: Whether metrics export is enabled (default: True, ignored if config provided)
        """
        # Handle both PrometheusConfig and legacy port parameter
        if config_or_port is None:
            self.port = 9090
            self.enabled = enabled
            self.start_server = True
        elif isinstance(config_or_port, int):
            # Legacy: port number
            self.port = config_or_port
            self.enabled = enabled
            self.start_server = True
        else:
            # New: PrometheusConfig object
            config = config_or_port
            self.port = config.port
            self.enabled = config.enabled
            self.start_server = config.start_server

        self._server_started = False
        self._lock = threading.Lock()
        self._degraded = False
        self._degradation_reason = None

        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not available, metrics export disabled. "
                "Install with: pip install prometheus-client"
            )
            self.enabled = False
            return

        if not self.enabled:
            logger.info("Prometheus metrics export disabled by configuration")
            return

        # Initialize metrics (with duplicate handling)
        self._init_metrics()
        logger.info(f"PrometheusExporter initialized on port {config_or_port}")

    def _get_or_create_metric(self, metric_class, name, description, labelnames=None, **kwargs):
        """
        Get existing metric or create new one.

        This prevents duplicate metric registration errors in tests.
        """
        try:
            # Try to get existing metric from registry
            for collector in list(REGISTRY._collector_to_names.keys()):
                if hasattr(collector, "_name") and collector._name == name:
                    return collector

            # Create new metric if not found
            if labelnames:
                return metric_class(name, description, labelnames, **kwargs)
            else:
                return metric_class(name, description, **kwargs)
        except Exception as e:
            # If metric already exists, try to retrieve it
            logger.debug(f"Metric {name} may already exist: {e}")

            # Return a dummy metric that does nothing
            class DummyMetric:
                def labels(self, **kwargs):
                    return self

                def observe(self, value):
                    pass

                def inc(self, amount=1):
                    pass

                def set(self, value):
                    pass

            return DummyMetric()

    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Inference latency histogram (in seconds)
        # Requirement 4.2: Record inference latency metrics
        self.inference_latency = self._get_or_create_metric(
            Histogram,
            "inference_latency_seconds",
            "Inference latency in seconds",
            ["model", "engine"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

        # Inference request counter
        # Requirement 4.2: Record inference requests
        self.inference_requests = self._get_or_create_metric(
            Counter,
            "inference_requests_total",
            "Total number of inference requests",
            ["model", "engine", "status"],
        )

        # Throughput gauge (requests per second)
        # Requirement 4.3: Record throughput metrics
        self.throughput = self._get_or_create_metric(
            Gauge,
            "throughput_requests_per_second",
            "Current throughput in requests per second",
            ["model"],
        )

        # GPU memory usage gauge (in bytes)
        # Requirement 4.5: Record GPU memory usage
        self.gpu_memory_used = self._get_or_create_metric(
            Gauge, "gpu_memory_used_bytes", "GPU memory used in bytes", ["gpu_id"]
        )

        # GPU memory available gauge (in bytes)
        # Requirement 11.5: Per-GPU metrics
        self.gpu_memory_available = self._get_or_create_metric(
            Gauge, "gpu_memory_available_bytes", "GPU memory available in bytes", ["gpu_id"]
        )

        # GPU utilization gauge (percentage)
        # Requirement 11.5: Per-GPU metrics
        self.gpu_utilization = self._get_or_create_metric(
            Gauge, "gpu_utilization_percent", "GPU utilization percentage", ["gpu_id"]
        )

        # GPU temperature gauge (Celsius)
        # Requirement 11.5: Per-GPU metrics
        self.gpu_temperature = self._get_or_create_metric(
            Gauge, "gpu_temperature_celsius", "GPU temperature in Celsius", ["gpu_id"]
        )

        # GPU health status gauge (1=healthy, 0=unhealthy)
        # Requirement 11.4: GPU failure detection
        self.gpu_health_status = self._get_or_create_metric(
            Gauge, "gpu_health_status", "GPU health status (1=healthy, 0=unhealthy)", ["gpu_id"]
        )

        # CPU usage gauge (percentage)
        # Requirement 4.6: Record CPU usage
        self.cpu_usage = self._get_or_create_metric(
            Gauge, "cpu_usage_percent", "CPU usage percentage"
        )

        # Model lifecycle events counter
        # Requirement 4.4: Record model lifecycle events
        self.model_lifecycle_events = self._get_or_create_metric(
            Counter,
            "model_lifecycle_events_total",
            "Model lifecycle events (load/unload)",
            ["model", "event_type"],
        )

        # KV cache hit rate gauge
        self.kv_cache_hit_rate = self._get_or_create_metric(
            Gauge, "kv_cache_hit_rate", "KV cache hit rate", ["model"]
        )

        # Batch size histogram
        self.batch_size = self._get_or_create_metric(
            Histogram,
            "batch_size",
            "Batch size distribution",
            ["model"],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
        )

        # System uptime gauge
        self.system_uptime = self._get_or_create_metric(
            Gauge, "system_uptime_seconds", "System uptime in seconds"
        )

        # Error counter
        self.errors = self._get_or_create_metric(
            Counter, "errors_total", "Total number of errors", ["component", "error_type"]
        )

        logger.info("Prometheus metrics initialized")

    def start_server(self):
        """
        Start HTTP server for metrics endpoint.

        Requirement 4.1, 4.7: Expose metrics on HTTP endpoint
        """
        if not self.enabled:
            logger.debug("Metrics export disabled, not starting server")
            return

        with self._lock:
            if self._server_started:
                logger.warning(f"Metrics server already started on port {self.port}")
                return

            try:
                start_http_server(self.port)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
            except OSError as e:
                logger.error(f"Failed to start metrics server on port {self.port}: {e}")
                self.enabled = False
            except Exception as e:
                logger.error(f"Unexpected error starting metrics server: {e}")
                self.enabled = False

    def record_inference_latency(self, model_name: str, engine: str, latency_ms: float):
        """
        Record inference latency metric.

        Args:
            model_name: Name of the model
            engine: Inference engine used (vllm, deepspeed, onnx, pytorch)
            latency_ms: Latency in milliseconds

        Requirement 4.2: Record inference latency metrics
        """
        if not self.enabled:
            return

        try:
            # Convert milliseconds to seconds for Prometheus
            latency_seconds = latency_ms / 1000.0
            self.inference_latency.labels(model=model_name, engine=engine).observe(latency_seconds)

            # Also increment request counter with success status
            self.inference_requests.labels(model=model_name, engine=engine, status="success").inc()

        except Exception as e:
            logger.error(f"Failed to record inference latency: {e}")

    def record_inference_error(self, model_name: str, engine: str, error_type: str):
        """
        Record inference error.

        Args:
            model_name: Name of the model
            engine: Inference engine used
            error_type: Type of error encountered
        """
        if not self.enabled:
            return

        try:
            self.inference_requests.labels(model=model_name, engine=engine, status="error").inc()

            self.errors.labels(component="inference", error_type=error_type).inc()

        except Exception as e:
            logger.error(f"Failed to record inference error: {e}")

    def record_throughput(self, model_name: str, requests_per_second: float):
        """
        Record throughput metric.

        Args:
            model_name: Name of the model
            requests_per_second: Current throughput

        Requirement 4.3: Record throughput metrics
        """
        if not self.enabled:
            return

        try:
            self.throughput.labels(model=model_name).set(requests_per_second)
        except Exception as e:
            logger.error(f"Failed to record throughput: {e}")

    def record_resource_usage(
        self, gpu_memory_mb: Optional[Dict[int, float]] = None, cpu_percent: Optional[float] = None
    ):
        """
        Record resource utilization metrics.

        Args:
            gpu_memory_mb: Dictionary mapping GPU ID to memory usage in MB
            cpu_percent: CPU usage percentage (0-100)

        Requirements 4.5, 4.6: Record GPU and CPU usage
        """
        if not self.enabled:
            return

        try:
            if gpu_memory_mb is not None:
                for gpu_id, memory_mb in gpu_memory_mb.items():
                    # Convert MB to bytes
                    memory_bytes = memory_mb * 1024 * 1024
                    self.gpu_memory_used.labels(gpu_id=str(gpu_id)).set(memory_bytes)

            if cpu_percent is not None:
                self.cpu_usage.set(cpu_percent)

        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}")

    def record_per_gpu_metrics(
        self,
        gpu_id: int,
        memory_used_mb: float,
        memory_available_mb: float,
        utilization_percent: float,
        temperature_celsius: Optional[float] = None,
        is_healthy: bool = True,
    ):
        """
        Record comprehensive per-GPU metrics.

        Args:
            gpu_id: GPU device ID
            memory_used_mb: GPU memory used in MB
            memory_available_mb: GPU memory available in MB
            utilization_percent: GPU utilization percentage (0-100)
            temperature_celsius: GPU temperature in Celsius (optional)
            is_healthy: Whether GPU is healthy

        Requirement 11.5: Per-GPU metrics are exposed
        """
        if not self.enabled:
            return

        try:
            gpu_label = str(gpu_id)

            # Memory metrics
            self.gpu_memory_used.labels(gpu_id=gpu_label).set(
                memory_used_mb * 1024 * 1024  # Convert to bytes
            )
            self.gpu_memory_available.labels(gpu_id=gpu_label).set(
                memory_available_mb * 1024 * 1024  # Convert to bytes
            )

            # Utilization metric
            self.gpu_utilization.labels(gpu_id=gpu_label).set(utilization_percent)

            # Temperature metric (if available)
            if temperature_celsius is not None:
                self.gpu_temperature.labels(gpu_id=gpu_label).set(temperature_celsius)

            # Health status metric
            self.gpu_health_status.labels(gpu_id=gpu_label).set(1 if is_healthy else 0)

        except Exception as e:
            logger.error(f"Failed to record per-GPU metrics for GPU {gpu_id}: {e}")

    def record_all_gpu_metrics(self, gpu_metrics: Dict[int, Dict[str, float]]):
        """
        Record metrics for all GPUs at once.

        Args:
            gpu_metrics: Dictionary mapping GPU ID to metrics dictionary
                        Each metrics dict should contain:
                        - memory_used_mb
                        - memory_available_mb
                        - utilization_percent
                        - temperature_celsius (optional)
                        - is_healthy (optional, defaults to True)

        Requirement 11.5: Per-GPU metrics are exposed

        Example:
            >>> exporter = PrometheusExporter()
            >>> metrics = {
            ...     0: {
            ...         'memory_used_mb': 8192,
            ...         'memory_available_mb': 7808,
            ...         'utilization_percent': 75.5,
            ...         'temperature_celsius': 65.0,
            ...         'is_healthy': True
            ...     },
            ...     1: {
            ...         'memory_used_mb': 4096,
            ...         'memory_available_mb': 11904,
            ...         'utilization_percent': 45.2,
            ...     }
            ... }
            >>> exporter.record_all_gpu_metrics(metrics)
        """
        if not self.enabled:
            return

        for gpu_id, metrics in gpu_metrics.items():
            self.record_per_gpu_metrics(
                gpu_id=gpu_id,
                memory_used_mb=metrics.get("memory_used_mb", 0),
                memory_available_mb=metrics.get("memory_available_mb", 0),
                utilization_percent=metrics.get("utilization_percent", 0),
                temperature_celsius=metrics.get("temperature_celsius"),
                is_healthy=metrics.get("is_healthy", True),
            )

    def record_model_lifecycle(
        self, model_name: str, event: str, duration_ms: Optional[float] = None
    ):
        """
        Record model lifecycle event.

        Args:
            model_name: Name of the model
            event: Type of event ('load' or 'unload')
            duration_ms: Optional duration of the operation in milliseconds

        Requirement 4.4: Record model lifecycle events
        """
        if not self.enabled:
            return

        try:
            self.model_lifecycle_events.labels(model=model_name, event_type=event).inc()
        except Exception as e:
            logger.error(f"Failed to record model lifecycle event: {e}")

    def record_cache_hit_rate(self, model_name: str, hit_rate: float):
        """
        Record KV cache hit rate.

        Args:
            model_name: Name of the model
            hit_rate: Cache hit rate (0.0 to 1.0)
        """
        if not self.enabled:
            return

        try:
            self.kv_cache_hit_rate.labels(model=model_name).set(hit_rate)
        except Exception as e:
            logger.error(f"Failed to record cache hit rate: {e}")

    def record_batch_size(self, model_name: str, batch_size: int):
        """
        Record batch size.

        Args:
            model_name: Name of the model
            batch_size: Size of the batch processed
        """
        if not self.enabled:
            return

        try:
            self.batch_size.labels(model=model_name).observe(batch_size)
        except Exception as e:
            logger.error(f"Failed to record batch size: {e}")

    def record_uptime(self, uptime_seconds: float):
        """
        Record system uptime.

        Args:
            uptime_seconds: System uptime in seconds
        """
        if not self.enabled:
            return

        try:
            self.system_uptime.set(uptime_seconds)
        except Exception as e:
            logger.error(f"Failed to record uptime: {e}")

    def record_error(self, component: str, error_type: str):
        """
        Record generic error.

        Args:
            component: Component where error occurred
            error_type: Type of error
        """
        if not self.enabled:
            return

        try:
            self.errors.labels(component=component, error_type=error_type).inc()
        except Exception as e:
            logger.error(f"Failed to record error: {e}")

    def is_enabled(self) -> bool:
        """Check if metrics export is enabled."""
        return self.enabled

    def is_server_started(self) -> bool:
        """Check if HTTP server is started."""
        return self._server_started

    def get_metrics(self) -> Dict[str, any]:
        """
        Get current metrics as a dictionary.

        This is primarily for testing and debugging purposes.

        Returns:
            Dictionary containing current metric values
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return {}

        try:
            from prometheus_client import generate_latest

            # Get metrics in Prometheus format
            metrics_bytes = generate_latest(REGISTRY)
            metrics_text = metrics_bytes.decode("utf-8")

            return {
                "enabled": self.enabled,
                "server_started": self._server_started,
                "port": self.port,
                "metrics_text": metrics_text,
                "metrics_count": len(
                    [line for line in metrics_text.split("\n") if line and not line.startswith("#")]
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}

    def format_metrics(self) -> str:
        """
        Format metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus text format

        Requirement 4.7: Expose metrics in Prometheus format
        """
        if not self.enabled or not PROMETHEUS_AVAILABLE:
            return ""

        try:
            from prometheus_client import generate_latest

            metrics_bytes = generate_latest(REGISTRY)
            return metrics_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to format metrics: {e}")
            return f"# Error formatting metrics: {e}\n"

    def get_status(self) -> Dict[str, any]:
        """
        Get the current status of the Prometheus exporter.

        Returns:
            Dictionary containing status information including:
            - enabled: Whether metrics export is enabled
            - server_started: Whether HTTP server is running
            - port: HTTP port for metrics endpoint
            - degraded: Whether the exporter is in degraded state
            - degradation_reason: Reason for degradation (if degraded)
        """
        status = {
            "enabled": self.enabled,
            "server_started": self._server_started,
            "port": self.port,
            "degraded": self._degraded,
        }

        if self._degraded and self._degradation_reason:
            status["reason"] = self._degradation_reason
            status["degradation_reason"] = self._degradation_reason

        return status
