"""
Performance monitoring for the MuAI Multi-Model Orchestration System.

This module provides detailed performance statistics collection and analysis,
including latency tracking, throughput calculation, and resource utilization monitoring.
"""

import time
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Any, Deque
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class LatencyRecord:
    """Record of a single latency measurement."""
    operation: str
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource utilization at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[Dict[int, float]] = None
    gpu_utilization: Optional[Dict[int, float]] = None


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for an operation."""
    operation: str
    count: int
    mean_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float


@dataclass
class ResourceTrends:
    """Resource utilization trends over time."""
    window_seconds: int
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_memory_mb: float
    max_memory_mb: float
    avg_gpu_memory_mb: Optional[Dict[int, float]] = None
    max_gpu_memory_mb: Optional[Dict[int, float]] = None
    avg_gpu_utilization: Optional[Dict[int, float]] = None
    max_gpu_utilization: Optional[Dict[int, float]] = None


class PerformanceMonitor:
    """
    Collects and analyzes detailed performance statistics.
    
    Tracks:
    - Per-operation latency distributions
    - Throughput over sliding time windows
    - Resource utilization trends (GPU, CPU, memory)
    - Error rates by type and component
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        max_history_seconds: int = 3600,
        resource_sample_interval: int = 10
    ):
        """
        Initialize performance monitor.
        
        Args:
            max_history_seconds: Maximum time to keep historical data (default: 1 hour)
            resource_sample_interval: Seconds between resource samples (default: 10s)
        """
        self.max_history_seconds = max_history_seconds
        self.resource_sample_interval = resource_sample_interval
        
        # Latency tracking
        self._latency_records: Dict[str, Deque[LatencyRecord]] = defaultdict(deque)
        self._latency_lock = Lock()
        
        # Request tracking for throughput
        self._request_timestamps: Deque[datetime] = deque()
        self._request_lock = Lock()
        
        # Resource tracking
        self._resource_snapshots: Deque[ResourceSnapshot] = deque()
        self._resource_lock = Lock()
        self._last_resource_sample = datetime.now()
        
        # Error tracking
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._error_lock = Lock()
        
        logger.info(
            f"PerformanceMonitor initialized with {max_history_seconds}s history, "
            f"{resource_sample_interval}s resource sampling"
        )
    
    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record latency for an operation.
        
        Args:
            operation: Name of the operation (e.g., "inference", "preprocessing")
            latency_ms: Latency in milliseconds
            metadata: Optional metadata (model_name, batch_size, etc.)
        """
        record = LatencyRecord(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._latency_lock:
            self._latency_records[operation].append(record)
            self._cleanup_old_latency_records(operation)
        
        # Also record as a request for throughput calculation
        with self._request_lock:
            self._request_timestamps.append(datetime.now())
            self._cleanup_old_requests()
        
        # Sample resources periodically
        self._maybe_sample_resources()
    
    def record_error(self, error_type: str, component: str):
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error (e.g., "OOMError", "TimeoutError")
            component: Component where error occurred (e.g., "vllm_engine", "batcher")
        """
        key = f"{component}:{error_type}"
        with self._error_lock:
            self._error_counts[key] += 1
    
    def get_percentiles(
        self,
        operation: str,
        window_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get percentile latencies for an operation.
        
        Args:
            operation: Name of the operation
            window_seconds: Time window to consider (None = all history)
            
        Returns:
            Dictionary with p50, p95, p99 latencies in milliseconds
        """
        with self._latency_lock:
            records = self._latency_records.get(operation, deque())
            
            if not records:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
            # Filter by time window if specified
            if window_seconds:
                cutoff = datetime.now() - timedelta(seconds=window_seconds)
                latencies = [
                    r.latency_ms for r in records
                    if r.timestamp >= cutoff
                ]
            else:
                latencies = [r.latency_ms for r in records]
            
            if not latencies:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
            # Sort for percentile calculation
            sorted_latencies = sorted(latencies)
            
            return {
                "p50": self._percentile(sorted_latencies, 50),
                "p95": self._percentile(sorted_latencies, 95),
                "p99": self._percentile(sorted_latencies, 99)
            }
    
    def get_throughput(self, window_seconds: int = 60) -> float:
        """
        Calculate requests per second over a time window.
        
        Args:
            window_seconds: Time window in seconds (default: 60)
            
        Returns:
            Requests per second
        """
        with self._request_lock:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            recent_requests = [
                ts for ts in self._request_timestamps
                if ts >= cutoff
            ]
            
            if not recent_requests:
                return 0.0
            
            # Calculate actual time span
            time_span = (recent_requests[-1] - recent_requests[0]).total_seconds()
            if time_span == 0:
                return 0.0
            
            return len(recent_requests) / time_span
    
    def get_resource_trends(
        self,
        window_seconds: int = 300
    ) -> ResourceTrends:
        """
        Get resource usage trends over a time window.
        
        Args:
            window_seconds: Time window in seconds (default: 300 = 5 minutes)
            
        Returns:
            ResourceTrends with aggregated statistics
        """
        with self._resource_lock:
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            recent_snapshots = [
                s for s in self._resource_snapshots
                if s.timestamp >= cutoff
            ]
            
            if not recent_snapshots:
                # Return current snapshot if no history
                self._sample_resources()
                recent_snapshots = list(self._resource_snapshots)
            
            if not recent_snapshots:
                # Still no data, return zeros
                return ResourceTrends(
                    window_seconds=window_seconds,
                    avg_cpu_percent=0.0,
                    max_cpu_percent=0.0,
                    avg_memory_mb=0.0,
                    max_memory_mb=0.0
                )
            
            # Calculate CPU and memory trends
            cpu_values = [s.cpu_percent for s in recent_snapshots]
            memory_values = [s.memory_mb for s in recent_snapshots]
            
            trends = ResourceTrends(
                window_seconds=window_seconds,
                avg_cpu_percent=sum(cpu_values) / len(cpu_values),
                max_cpu_percent=max(cpu_values),
                avg_memory_mb=sum(memory_values) / len(memory_values),
                max_memory_mb=max(memory_values)
            )
            
            # Calculate GPU trends if available
            if recent_snapshots[0].gpu_memory_mb:
                gpu_ids = recent_snapshots[0].gpu_memory_mb.keys()
                
                trends.avg_gpu_memory_mb = {}
                trends.max_gpu_memory_mb = {}
                trends.avg_gpu_utilization = {}
                trends.max_gpu_utilization = {}
                
                for gpu_id in gpu_ids:
                    gpu_mem_values = [
                        s.gpu_memory_mb.get(gpu_id, 0.0)
                        for s in recent_snapshots
                        if s.gpu_memory_mb
                    ]
                    gpu_util_values = [
                        s.gpu_utilization.get(gpu_id, 0.0)
                        for s in recent_snapshots
                        if s.gpu_utilization
                    ]
                    
                    if gpu_mem_values:
                        trends.avg_gpu_memory_mb[gpu_id] = sum(gpu_mem_values) / len(gpu_mem_values)
                        trends.max_gpu_memory_mb[gpu_id] = max(gpu_mem_values)
                    
                    if gpu_util_values:
                        trends.avg_gpu_utilization[gpu_id] = sum(gpu_util_values) / len(gpu_util_values)
                        trends.max_gpu_utilization[gpu_id] = max(gpu_util_values)
            
            return trends
    
    def get_operation_metrics(
        self,
        operation: str,
        window_seconds: Optional[int] = None
    ) -> Optional[PerformanceMetrics]:
        """
        Get aggregated metrics for an operation.
        
        Args:
            operation: Name of the operation
            window_seconds: Time window to consider (None = all history)
            
        Returns:
            PerformanceMetrics or None if no data
        """
        with self._latency_lock:
            records = self._latency_records.get(operation, deque())
            
            if not records:
                return None
            
            # Filter by time window if specified
            if window_seconds:
                cutoff = datetime.now() - timedelta(seconds=window_seconds)
                filtered_records = [r for r in records if r.timestamp >= cutoff]
            else:
                filtered_records = list(records)
            
            if not filtered_records:
                return None
            
            latencies = [r.latency_ms for r in filtered_records]
            sorted_latencies = sorted(latencies)
            
            # Calculate throughput
            time_span = (filtered_records[-1].timestamp - filtered_records[0].timestamp).total_seconds()
            throughput = len(filtered_records) / time_span if time_span > 0 else 0.0
            
            return PerformanceMetrics(
                operation=operation,
                count=len(filtered_records),
                mean_latency_ms=sum(latencies) / len(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p50_latency_ms=self._percentile(sorted_latencies, 50),
                p95_latency_ms=self._percentile(sorted_latencies, 95),
                p99_latency_ms=self._percentile(sorted_latencies, 99),
                throughput_rps=throughput
            )
    
    def get_error_rate(
        self,
        component: Optional[str] = None,
        window_seconds: int = 60
    ) -> float:
        """
        Get error rate for a component or overall.
        
        Args:
            component: Component name (None = all components)
            window_seconds: Time window for rate calculation
            
        Returns:
            Error rate (errors per second)
        """
        with self._error_lock:
            if component:
                # Filter errors for specific component
                total_errors = sum(
                    count for key, count in self._error_counts.items()
                    if key.startswith(f"{component}:")
                )
            else:
                # All errors
                total_errors = sum(self._error_counts.values())
            
            return total_errors / window_seconds
    
    def get_all_operations(self) -> List[str]:
        """Get list of all tracked operations."""
        with self._latency_lock:
            return list(self._latency_records.keys())
    
    def reset_statistics(self):
        """Reset all collected statistics."""
        with self._latency_lock:
            self._latency_records.clear()
        
        with self._request_lock:
            self._request_timestamps.clear()
        
        with self._resource_lock:
            self._resource_snapshots.clear()
        
        with self._error_lock:
            self._error_counts.clear()
        
        logger.info("Performance statistics reset")
    
    # Private helper methods
    
    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        n = len(sorted_values)
        index = (percentile / 100.0) * (n - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            # Linear interpolation
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
    
    def _cleanup_old_latency_records(self, operation: str):
        """Remove latency records older than max_history_seconds."""
        cutoff = datetime.now() - timedelta(seconds=self.max_history_seconds)
        records = self._latency_records[operation]
        
        while records and records[0].timestamp < cutoff:
            records.popleft()
    
    def _cleanup_old_requests(self):
        """Remove request timestamps older than max_history_seconds."""
        cutoff = datetime.now() - timedelta(seconds=self.max_history_seconds)
        
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
    
    def _cleanup_old_resource_snapshots(self):
        """Remove resource snapshots older than max_history_seconds."""
        cutoff = datetime.now() - timedelta(seconds=self.max_history_seconds)
        
        while self._resource_snapshots and self._resource_snapshots[0].timestamp < cutoff:
            self._resource_snapshots.popleft()
    
    def _maybe_sample_resources(self):
        """Sample resources if enough time has passed since last sample."""
        now = datetime.now()
        if (now - self._last_resource_sample).total_seconds() >= self.resource_sample_interval:
            self._sample_resources()
            self._last_resource_sample = now
    
    def _sample_resources(self):
        """Sample current resource utilization."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            # GPU metrics if available
            gpu_memory_mb = None
            gpu_utilization = None
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_memory_mb = {}
                gpu_utilization = {}
                
                for i in range(torch.cuda.device_count()):
                    # Memory
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    gpu_memory_mb[i] = mem_allocated
                    
                    # Utilization (approximation based on memory usage)
                    mem_total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                    gpu_utilization[i] = (mem_allocated / mem_total) * 100 if mem_total > 0 else 0.0
            
            snapshot = ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization
            )
            
            with self._resource_lock:
                self._resource_snapshots.append(snapshot)
                self._cleanup_old_resource_snapshots()
        
        except Exception as e:
            logger.warning(f"Failed to sample resources: {e}")
