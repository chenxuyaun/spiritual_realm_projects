"""Prometheus metrics exporter for system monitoring."""

import logging
import threading
from typing import Dict, Optional
from datetime import datetime

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
    
    def __init__(self, port: int = 9090, enabled: bool = True):
        """
        Initialize Prometheus exporter.
        
        Args:
            port: HTTP port for metrics endpoint (default: 9090)
            enabled: Whether metrics export is enabled (default: True)
        """
        self.port = port
        self.enabled = enabled
        self._server_started = False
        self._lock = threading.Lock()
        
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
        
        # Initialize metrics
        self._init_metrics()
        logger.info(f"PrometheusExporter initialized on port {port}")
    
    def _init_metrics(self):
        """Initialize Prometheus metrics."""
        # Inference latency histogram (in seconds)
        # Requirement 4.2: Record inference latency metrics
        self.inference_latency = Histogram(
            'inference_latency_seconds',
            'Inference latency in seconds',
            ['model', 'engine'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Inference request counter
        # Requirement 4.2: Record inference requests
        self.inference_requests = Counter(
            'inference_requests_total',
            'Total number of inference requests',
            ['model', 'engine', 'status']
        )
        
        # Throughput gauge (requests per second)
        # Requirement 4.3: Record throughput metrics
        self.throughput = Gauge(
            'throughput_requests_per_second',
            'Current throughput in requests per second',
            ['model']
        )
        
        # GPU memory usage gauge (in bytes)
        # Requirement 4.5: Record GPU memory usage
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id']
        )
        
        # CPU usage gauge (percentage)
        # Requirement 4.6: Record CPU usage
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Model lifecycle events counter
        # Requirement 4.4: Record model lifecycle events
        self.model_lifecycle_events = Counter(
            'model_lifecycle_events_total',
            'Model lifecycle events (load/unload)',
            ['model', 'event_type']
        )
        
        # KV cache hit rate gauge
        self.kv_cache_hit_rate = Gauge(
            'kv_cache_hit_rate',
            'KV cache hit rate',
            ['model']
        )
        
        # Batch size histogram
        self.batch_size = Histogram(
            'batch_size',
            'Batch size distribution',
            ['model'],
            buckets=(1, 2, 4, 8, 16, 32, 64, 128)
        )
        
        # System uptime gauge
        self.system_uptime = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds'
        )
        
        # Error counter
        self.errors = Counter(
            'errors_total',
            'Total number of errors',
            ['component', 'error_type']
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
    
    def record_inference_latency(
        self,
        model_name: str,
        engine: str,
        latency_ms: float
    ):
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
            self.inference_latency.labels(
                model=model_name,
                engine=engine
            ).observe(latency_seconds)
            
            # Also increment request counter with success status
            self.inference_requests.labels(
                model=model_name,
                engine=engine,
                status='success'
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record inference latency: {e}")
    
    def record_inference_error(
        self,
        model_name: str,
        engine: str,
        error_type: str
    ):
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
            self.inference_requests.labels(
                model=model_name,
                engine=engine,
                status='error'
            ).inc()
            
            self.errors.labels(
                component='inference',
                error_type=error_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record inference error: {e}")
    
    def record_throughput(
        self,
        model_name: str,
        requests_per_second: float
    ):
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
        self,
        gpu_memory_mb: Optional[Dict[int, float]] = None,
        cpu_percent: Optional[float] = None
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
    
    def record_model_lifecycle(
        self,
        model_name: str,
        event_type: str
    ):
        """
        Record model lifecycle event.
        
        Args:
            model_name: Name of the model
            event_type: Type of event ('load' or 'unload')
            
        Requirement 4.4: Record model lifecycle events
        """
        if not self.enabled:
            return
        
        try:
            self.model_lifecycle_events.labels(
                model=model_name,
                event_type=event_type
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record model lifecycle event: {e}")
    
    def record_cache_hit_rate(
        self,
        model_name: str,
        hit_rate: float
    ):
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
    
    def record_batch_size(
        self,
        model_name: str,
        batch_size: int
    ):
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
    
    def record_error(
        self,
        component: str,
        error_type: str
    ):
        """
        Record generic error.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
        """
        if not self.enabled:
            return
        
        try:
            self.errors.labels(
                component=component,
                error_type=error_type
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def is_enabled(self) -> bool:
        """Check if metrics export is enabled."""
        return self.enabled
    
    def is_server_started(self) -> bool:
        """Check if HTTP server is started."""
        return self._server_started
