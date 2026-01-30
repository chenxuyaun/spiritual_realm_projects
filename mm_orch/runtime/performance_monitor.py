"""
Performance monitoring for backend inference operations.

This module provides performance tracking and comparison capabilities
for different inference backends (PyTorch, OpenVINO).
"""

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np

from mm_orch.logger import get_logger


@dataclass
class InferenceMetrics:
    """Metrics for a single inference operation."""
    
    backend: str
    model_name: str
    latency_ms: float
    tokens_generated: int
    throughput_tokens_per_sec: float
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Monitor and compare backend performance.
    
    Tracks inference latency, throughput, and other metrics for
    different backends to enable performance comparison and optimization.
    
    Requirements:
    - 7.1: Track inference latency for each backend
    - 7.2: Track throughput (tokens/second) for each backend
    - 7.3: Expose performance metrics through monitoring system
    - 7.4: Provide comparative metrics when both backends are used
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self._logger = get_logger()
        
        # Storage for raw metrics: {backend:model_name:metric_type -> [values]}
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Storage for detailed inference records
        self._inference_records: List[InferenceMetrics] = []
        
        # Backend statistics cache
        self._backend_stats: Dict[str, Dict[str, Any]] = {}
        
        self._logger.debug("PerformanceMonitor initialized")
    
    def record_inference(
        self,
        backend: str,
        model_name: str,
        latency: float,
        tokens: int
    ) -> None:
        """
        Record inference metrics.
        
        Args:
            backend: Backend name ('pytorch' or 'openvino')
            model_name: Model identifier
            latency: Inference latency in seconds
            tokens: Number of tokens generated
        
        Requirements:
        - 7.1: Track inference latency for each backend
        - 7.2: Track throughput (tokens/second) for each backend
        """
        # Calculate throughput
        throughput = tokens / latency if latency > 0 else 0.0
        
        # Store metrics in defaultdict with unambiguous keys
        key_prefix = f"{backend}::{model_name}"  # Use :: as separator to avoid ambiguity
        self._metrics[f"{key_prefix}::latency"].append(latency)
        self._metrics[f"{key_prefix}::throughput"].append(throughput)
        self._metrics[f"{key_prefix}::tokens"].append(float(tokens))
        
        # Store detailed record
        record = InferenceMetrics(
            backend=backend,
            model_name=model_name,
            latency_ms=latency * 1000,  # Convert to milliseconds
            tokens_generated=tokens,
            throughput_tokens_per_sec=throughput
        )
        self._inference_records.append(record)
        
        # Invalidate cached stats for this backend
        if backend in self._backend_stats:
            del self._backend_stats[backend]
        
        self._logger.debug(
            f"Recorded inference metrics",
            context={
                "backend": backend,
                "model": model_name,
                "latency_ms": latency * 1000,
                "throughput": throughput
            }
        )
    
    def get_backend_stats(self, backend: str) -> Dict[str, Any]:
        """
        Get aggregated statistics for a backend.
        
        Args:
            backend: Backend name ('pytorch' or 'openvino')
        
        Returns:
            Dictionary containing aggregated statistics:
            - backend: Backend name
            - avg_latency: Average latency in seconds
            - p50_latency: Median latency in seconds
            - p95_latency: 95th percentile latency in seconds
            - avg_throughput: Average throughput in tokens/second
            - sample_count: Number of inference samples
        
        Requirements:
        - 7.3: Expose performance metrics through monitoring system
        - 7.4: Provide comparative metrics
        """
        # Check cache
        if backend in self._backend_stats:
            return self._backend_stats[backend]
        
        # Collect all latencies and throughputs for this backend
        latencies = []
        throughputs = []
        
        for key, values in self._metrics.items():
            if key.startswith(f"{backend}::"):
                # Check if key ends with the metric type
                if key.endswith("::latency"):
                    latencies.extend(values)
                elif key.endswith("::throughput"):
                    throughputs.extend(values)
        
        # Return empty dict if no data
        if not latencies:
            return {}
        
        # Calculate statistics
        stats = {
            "backend": backend,
            "avg_latency": float(np.mean(latencies)),
            "p50_latency": float(np.percentile(latencies, 50)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "avg_throughput": float(np.mean(throughputs)) if throughputs else 0.0,
            "sample_count": len(latencies)
        }
        
        # Cache the stats
        self._backend_stats[backend] = stats
        
        return stats
    
    def compare_backends(
        self,
        backend1: str,
        backend2: str
    ) -> Dict[str, Any]:
        """
        Compare performance between two backends.
        
        Args:
            backend1: First backend name
            backend2: Second backend name
        
        Returns:
            Dictionary containing comparison metrics:
            - latency_improvement: Ratio of backend1 latency to backend2 latency
            - throughput_improvement: Ratio of backend2 throughput to backend1 throughput
            - {backend1}_stats: Statistics for backend1
            - {backend2}_stats: Statistics for backend2
        
        Requirements:
        - 7.4: Provide comparative metrics when both backends are used
        """
        stats1 = self.get_backend_stats(backend1)
        stats2 = self.get_backend_stats(backend2)
        
        # Return empty dict if either backend has no data
        if not stats1 or not stats2:
            self._logger.warning(
                f"Cannot compare backends: insufficient data",
                context={
                    "backend1": backend1,
                    "backend2": backend2,
                    "backend1_samples": stats1.get("sample_count", 0),
                    "backend2_samples": stats2.get("sample_count", 0)
                }
            )
            return {}
        
        # Calculate improvement ratios
        latency_improvement = (
            stats1["avg_latency"] / stats2["avg_latency"]
            if stats2["avg_latency"] > 0 else 0.0
        )
        
        throughput_improvement = (
            stats2["avg_throughput"] / stats1["avg_throughput"]
            if stats1["avg_throughput"] > 0 else 0.0
        )
        
        comparison = {
            "latency_improvement": latency_improvement,
            "throughput_improvement": throughput_improvement,
            f"{backend1}_stats": stats1,
            f"{backend2}_stats": stats2
        }
        
        self._logger.info(
            f"Backend comparison: {backend1} vs {backend2}",
            context={
                "latency_improvement": f"{latency_improvement:.2f}x",
                "throughput_improvement": f"{throughput_improvement:.2f}x"
            }
        )
        
        return comparison
    
    def get_model_stats(self, model_name: str, backend: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific model.
        
        Args:
            model_name: Model identifier
            backend: Optional backend filter
        
        Returns:
            Dictionary containing model-specific statistics
        """
        latencies = []
        throughputs = []
        
        for key, values in self._metrics.items():
            # Filter by model name and optionally by backend
            if f"::{model_name}::" in key:
                if backend is None or key.startswith(f"{backend}::"):
                    if key.endswith("::latency"):
                        latencies.extend(values)
                    elif key.endswith("::throughput"):
                        throughputs.extend(values)
        
        if not latencies:
            return {}
        
        return {
            "model_name": model_name,
            "backend": backend,
            "avg_latency": float(np.mean(latencies)),
            "p50_latency": float(np.percentile(latencies, 50)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "avg_throughput": float(np.mean(throughputs)) if throughputs else 0.0,
            "sample_count": len(latencies)
        }
    
    def get_all_records(self) -> List[InferenceMetrics]:
        """
        Get all inference records.
        
        Returns:
            List of InferenceMetrics objects (deep copies)
        """
        import copy
        return copy.deepcopy(self._inference_records)
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._metrics.clear()
        self._inference_records.clear()
        self._backend_stats.clear()
        self._logger.debug("Performance metrics cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all performance data.
        
        Returns:
            Dictionary containing summary statistics for all backends
        """
        # Get unique backends
        backends = set()
        for key in self._metrics.keys():
            backend = key.split("::")[0]
            backends.add(backend)
        
        summary = {
            "total_inferences": len(self._inference_records),
            "backends": {}
        }
        
        for backend in backends:
            stats = self.get_backend_stats(backend)
            if stats:
                summary["backends"][backend] = stats
        
        return summary
