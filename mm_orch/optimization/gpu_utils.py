"""
GPU utilities for multi-GPU support.

This module provides utilities for GPU detection, allocation, health monitoring,
and failure detection for multi-GPU inference with tensor and pipeline parallelism.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from mm_orch.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """
    Information about a single GPU.

    Attributes:
        device_id: GPU device ID
        name: GPU name/model
        total_memory_mb: Total GPU memory in MB
        available_memory_mb: Available GPU memory in MB
        utilization_percent: GPU utilization percentage (0-100)
        temperature_celsius: GPU temperature in Celsius (None if unavailable)
        is_healthy: Whether GPU is healthy and operational
        error_message: Error message if GPU is unhealthy
    """

    device_id: int
    name: str
    total_memory_mb: float
    available_memory_mb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    is_healthy: bool = True
    error_message: Optional[str] = None


class GPUManager:
    """
    Manager for GPU detection, allocation, and health monitoring.

    Provides utilities for:
    - Detecting available GPUs
    - Allocating GPUs for tensor/pipeline parallelism
    - Monitoring GPU health and detecting failures
    - Collecting per-GPU metrics
    - Recovering from GPU failures

    Example:
        >>> manager = GPUManager()
        >>> gpus = manager.detect_gpus()
        >>> print(f"Found {len(gpus)} GPUs")
        >>> allocation = manager.allocate_gpus(tensor_parallel=2)
        >>> print(f"Allocated GPUs: {allocation}")
    """

    def __init__(self):
        """Initialize GPU manager."""
        self._gpu_cache: Dict[int, GPUInfo] = {}
        self._last_health_check: float = 0
        self._health_check_interval: float = 5.0  # seconds
        self._failed_gpus: set = set()  # Track failed GPUs

        logger.info("GPUManager initialized")

    def detect_gpus(self, force_refresh: bool = False) -> List[GPUInfo]:
        """
        Detect all available GPUs in the system.

        Args:
            force_refresh: Force refresh of GPU information

        Returns:
            List of GPUInfo objects for all detected GPUs

        Example:
            >>> manager = GPUManager()
            >>> gpus = manager.detect_gpus()
            >>> for gpu in gpus:
            ...     print(f"GPU {gpu.device_id}: {gpu.name}")
        """
        # Check if we can use cached data
        current_time = time.time()
        if (
            not force_refresh
            and self._gpu_cache
            and current_time - self._last_health_check < self._health_check_interval
        ):
            return list(self._gpu_cache.values())

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, no GPUs detected")
            return []

        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} CUDA GPUs")

        gpus = []
        for device_id in range(gpu_count):
            try:
                gpu_info = self._get_gpu_info(device_id)
                gpus.append(gpu_info)
                self._gpu_cache[device_id] = gpu_info
            except Exception as e:
                logger.error(f"Failed to get info for GPU {device_id}: {e}", exc_info=True)
                # Create unhealthy GPU entry
                gpu_info = GPUInfo(
                    device_id=device_id,
                    name="Unknown",
                    total_memory_mb=0,
                    available_memory_mb=0,
                    utilization_percent=0,
                    is_healthy=False,
                    error_message=str(e),
                )
                gpus.append(gpu_info)
                self._gpu_cache[device_id] = gpu_info

        self._last_health_check = current_time
        return gpus

    def _get_gpu_info(self, device_id: int) -> GPUInfo:
        """
        Get detailed information about a specific GPU.

        Args:
            device_id: GPU device ID

        Returns:
            GPUInfo object with GPU details
        """
        # Get GPU properties
        props = torch.cuda.get_device_properties(device_id)
        name = props.name
        total_memory_mb = props.total_memory / (1024**2)

        # Get memory info
        torch.cuda.set_device(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**2)
        available_memory_mb = total_memory_mb - memory_reserved

        # Calculate utilization (approximation based on memory usage)
        utilization_percent = (memory_reserved / total_memory_mb) * 100

        # Try to get temperature (requires nvidia-ml-py3)
        temperature = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except (ImportError, Exception):
            # Temperature monitoring not available
            pass

        return GPUInfo(
            device_id=device_id,
            name=name,
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            utilization_percent=utilization_percent,
            temperature_celsius=temperature,
            is_healthy=True,
            error_message=None,
        )

    def allocate_gpus(
        self,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        min_memory_mb: Optional[float] = None,
    ) -> Tuple[List[int], str]:
        """
        Allocate GPUs for tensor and/or pipeline parallelism.

        Selects GPUs with sufficient memory and health status for the
        requested parallelism configuration.

        Args:
            tensor_parallel: Number of GPUs for tensor parallelism
            pipeline_parallel: Number of GPUs for pipeline parallelism
            min_memory_mb: Minimum required memory per GPU (optional)

        Returns:
            Tuple of (list of allocated GPU IDs, allocation strategy description)

        Raises:
            RuntimeError: If insufficient GPUs are available

        Example:
            >>> manager = GPUManager()
            >>> gpu_ids, strategy = manager.allocate_gpus(tensor_parallel=2)
            >>> print(f"Allocated GPUs {gpu_ids} using {strategy}")
        """
        gpus = self.detect_gpus(force_refresh=True)

        # Filter healthy GPUs
        healthy_gpus = [gpu for gpu in gpus if gpu.is_healthy]

        if not healthy_gpus:
            raise RuntimeError("No healthy GPUs available")

        # Filter by memory if specified
        if min_memory_mb:
            healthy_gpus = [gpu for gpu in healthy_gpus if gpu.available_memory_mb >= min_memory_mb]

            if not healthy_gpus:
                raise RuntimeError(f"No GPUs with at least {min_memory_mb}MB available memory")

        # Calculate total GPUs needed
        total_gpus_needed = tensor_parallel * pipeline_parallel

        if len(healthy_gpus) < total_gpus_needed:
            raise RuntimeError(
                f"Insufficient GPUs: need {total_gpus_needed}, "
                f"have {len(healthy_gpus)} healthy GPUs"
            )

        # Sort GPUs by available memory (descending)
        healthy_gpus.sort(key=lambda g: g.available_memory_mb, reverse=True)

        # Allocate top GPUs
        allocated_gpu_ids = [gpu.device_id for gpu in healthy_gpus[:total_gpus_needed]]

        # Determine allocation strategy
        if tensor_parallel > 1 and pipeline_parallel > 1:
            strategy = f"hybrid (TP={tensor_parallel}, PP={pipeline_parallel})"
        elif tensor_parallel > 1:
            strategy = f"tensor_parallel (TP={tensor_parallel})"
        elif pipeline_parallel > 1:
            strategy = f"pipeline_parallel (PP={pipeline_parallel})"
        else:
            strategy = "single_gpu"

        logger.info(
            f"Allocated {len(allocated_gpu_ids)} GPUs {allocated_gpu_ids} " f"for {strategy}"
        )

        return allocated_gpu_ids, strategy

    def check_gpu_health(self, device_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check health status of a specific GPU.

        Performs basic health checks including:
        - CUDA availability
        - Memory allocation test
        - Basic computation test

        Args:
            device_id: GPU device ID to check

        Returns:
            Tuple of (is_healthy, error_message)

        Example:
            >>> manager = GPUManager()
            >>> is_healthy, error = manager.check_gpu_health(0)
            >>> if not is_healthy:
            ...     print(f"GPU 0 unhealthy: {error}")
        """
        try:
            # Check if device exists
            if device_id >= torch.cuda.device_count():
                return False, f"GPU {device_id} does not exist"

            # Try to allocate memory
            torch.cuda.set_device(device_id)
            test_tensor = torch.zeros(1024, 1024, device=f"cuda:{device_id}")

            # Try basic computation
            result = test_tensor + 1
            result = result.cpu()

            # Clean up
            del test_tensor
            del result
            torch.cuda.empty_cache()

            return True, None

        except torch.cuda.OutOfMemoryError:
            return False, "Out of memory"
        except RuntimeError as e:
            return False, f"CUDA runtime error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def monitor_gpu_health(self) -> Dict[int, Tuple[bool, Optional[str]]]:
        """
        Monitor health of all GPUs.

        Returns:
            Dictionary mapping GPU ID to (is_healthy, error_message)

        Example:
            >>> manager = GPUManager()
            >>> health_status = manager.monitor_gpu_health()
            >>> for gpu_id, (healthy, error) in health_status.items():
            ...     if not healthy:
            ...         print(f"GPU {gpu_id} failed: {error}")
        """
        gpus = self.detect_gpus(force_refresh=True)
        health_status = {}

        for gpu in gpus:
            if not gpu.is_healthy:
                health_status[gpu.device_id] = (False, gpu.error_message)
            else:
                is_healthy, error = self.check_gpu_health(gpu.device_id)
                health_status[gpu.device_id] = (is_healthy, error)

                # Update cache
                if gpu.device_id in self._gpu_cache:
                    self._gpu_cache[gpu.device_id].is_healthy = is_healthy
                    self._gpu_cache[gpu.device_id].error_message = error

        return health_status

    def get_gpu_metrics(self, device_id: int) -> Dict[str, float]:
        """
        Get current metrics for a specific GPU.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with GPU metrics (memory_used_mb, memory_available_mb,
            utilization_percent, temperature_celsius)

        Example:
            >>> manager = GPUManager()
            >>> metrics = manager.get_gpu_metrics(0)
            >>> print(f"GPU 0 memory: {metrics['memory_used_mb']}MB")
        """
        gpu_info = self._get_gpu_info(device_id)

        metrics = {
            "memory_used_mb": gpu_info.total_memory_mb - gpu_info.available_memory_mb,
            "memory_available_mb": gpu_info.available_memory_mb,
            "memory_total_mb": gpu_info.total_memory_mb,
            "utilization_percent": gpu_info.utilization_percent,
        }

        if gpu_info.temperature_celsius is not None:
            metrics["temperature_celsius"] = gpu_info.temperature_celsius

        return metrics

    def get_all_gpu_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Get current metrics for all GPUs.

        Returns:
            Dictionary mapping GPU ID to metrics dictionary

        Example:
            >>> manager = GPUManager()
            >>> all_metrics = manager.get_all_gpu_metrics()
            >>> for gpu_id, metrics in all_metrics.items():
            ...     print(f"GPU {gpu_id}: {metrics['memory_used_mb']}MB used")
        """
        gpus = self.detect_gpus(force_refresh=True)
        all_metrics = {}

        for gpu in gpus:
            try:
                metrics = self.get_gpu_metrics(gpu.device_id)
                all_metrics[gpu.device_id] = metrics
            except Exception as e:
                logger.error(f"Failed to get metrics for GPU {gpu.device_id}: {e}", exc_info=True)

        return all_metrics

    def balance_load(self, tensor_parallel: int, pipeline_parallel: int = 1) -> List[List[int]]:
        """
        Balance computational load across GPUs for parallelism.

        Creates a balanced allocation plan for tensor and pipeline parallelism
        that distributes work evenly across available GPUs.

        Args:
            tensor_parallel: Number of GPUs for tensor parallelism
            pipeline_parallel: Number of stages for pipeline parallelism

        Returns:
            List of GPU ID lists, one per pipeline stage

        Example:
            >>> manager = GPUManager()
            >>> # 4 GPUs with TP=2, PP=2
            >>> allocation = manager.balance_load(tensor_parallel=2, pipeline_parallel=2)
            >>> print(allocation)  # [[0, 1], [2, 3]]
        """
        gpu_ids, _ = self.allocate_gpus(tensor_parallel, pipeline_parallel)

        # Group GPUs by pipeline stage
        allocation = []
        for stage in range(pipeline_parallel):
            start_idx = stage * tensor_parallel
            end_idx = start_idx + tensor_parallel
            stage_gpus = gpu_ids[start_idx:end_idx]
            allocation.append(stage_gpus)

        logger.info(
            f"Load balanced across {pipeline_parallel} pipeline stages "
            f"with {tensor_parallel} GPUs per stage: {allocation}"
        )

        return allocation

    def detect_gpu_failure(self, device_id: int) -> Tuple[bool, Optional[str]]:
        """
        Detect if a GPU has failed during inference.

        Performs comprehensive health checks to detect various failure modes:
        - CUDA errors
        - Out of memory errors
        - Computation errors
        - Communication errors (for multi-GPU)

        Args:
            device_id: GPU device ID to check

        Returns:
            Tuple of (has_failed, failure_reason)

        Example:
            >>> manager = GPUManager()
            >>> has_failed, reason = manager.detect_gpu_failure(0)
            >>> if has_failed:
            ...     print(f"GPU 0 failed: {reason}")
        """
        is_healthy, error = self.check_gpu_health(device_id)

        if not is_healthy:
            # Mark GPU as failed
            self._failed_gpus.add(device_id)
            logger.error(f"GPU {device_id} failure detected: {error}")
            return True, error

        # GPU is healthy, remove from failed set if it was there
        if device_id in self._failed_gpus:
            logger.info(f"GPU {device_id} recovered from failure")
            self._failed_gpus.discard(device_id)

        return False, None

    def get_failed_gpus(self) -> List[int]:
        """
        Get list of GPUs that have failed.

        Returns:
            List of failed GPU IDs

        Example:
            >>> manager = GPUManager()
            >>> failed = manager.get_failed_gpus()
            >>> print(f"Failed GPUs: {failed}")
        """
        return list(self._failed_gpus)

    def recover_from_failure(
        self, failed_gpu_id: int, tensor_parallel: int, pipeline_parallel: int = 1
    ) -> Tuple[Optional[List[int]], str]:
        """
        Attempt to recover from GPU failure by reallocating to healthy GPUs.

        Recovery strategies:
        1. Reallocate to different healthy GPUs
        2. Reduce parallelism if insufficient healthy GPUs
        3. Fall back to CPU if no GPUs available

        Args:
            failed_gpu_id: ID of the failed GPU
            tensor_parallel: Current tensor parallelism level
            pipeline_parallel: Current pipeline parallelism level

        Returns:
            Tuple of (new_gpu_allocation, recovery_strategy)
            Returns (None, strategy) if recovery is not possible

        Example:
            >>> manager = GPUManager()
            >>> # GPU 1 failed during inference
            >>> new_gpus, strategy = manager.recover_from_failure(1, tensor_parallel=2)
            >>> if new_gpus:
            ...     print(f"Recovered using {strategy}: {new_gpus}")
        """
        logger.warning(
            f"Attempting recovery from GPU {failed_gpu_id} failure "
            f"(TP={tensor_parallel}, PP={pipeline_parallel})"
        )

        # Mark GPU as failed
        self._failed_gpus.add(failed_gpu_id)

        # Get all healthy GPUs
        gpus = self.detect_gpus(force_refresh=True)
        healthy_gpus = [
            gpu for gpu in gpus if gpu.is_healthy and gpu.device_id not in self._failed_gpus
        ]

        total_gpus_needed = tensor_parallel * pipeline_parallel

        # Strategy 1: Reallocate to different healthy GPUs
        if len(healthy_gpus) >= total_gpus_needed:
            try:
                new_allocation, strategy = self.allocate_gpus(
                    tensor_parallel=tensor_parallel, pipeline_parallel=pipeline_parallel
                )
                logger.info(f"Recovery successful: reallocated to GPUs {new_allocation}")
                return new_allocation, f"reallocate_{strategy}"
            except RuntimeError as e:
                logger.warning(f"Reallocation failed: {e}")

        # Strategy 2: Reduce parallelism
        if len(healthy_gpus) > 0:
            # Try reducing pipeline parallelism first
            if pipeline_parallel > 1:
                new_pp = min(pipeline_parallel - 1, len(healthy_gpus) // tensor_parallel)
                if new_pp >= 1:
                    try:
                        new_allocation, strategy = self.allocate_gpus(
                            tensor_parallel=tensor_parallel, pipeline_parallel=new_pp
                        )
                        logger.info(
                            f"Recovery successful: reduced pipeline parallelism "
                            f"from {pipeline_parallel} to {new_pp}"
                        )
                        return new_allocation, f"reduce_pp_{strategy}"
                    except RuntimeError:
                        pass

            # Try reducing tensor parallelism
            if tensor_parallel > 1:
                new_tp = min(tensor_parallel - 1, len(healthy_gpus))
                if new_tp >= 1:
                    try:
                        new_allocation, strategy = self.allocate_gpus(
                            tensor_parallel=new_tp, pipeline_parallel=1
                        )
                        logger.info(
                            f"Recovery successful: reduced tensor parallelism "
                            f"from {tensor_parallel} to {new_tp}"
                        )
                        return new_allocation, f"reduce_tp_{strategy}"
                    except RuntimeError:
                        pass

            # Try single GPU
            try:
                new_allocation, strategy = self.allocate_gpus(
                    tensor_parallel=1, pipeline_parallel=1
                )
                logger.info("Recovery successful: fell back to single GPU")
                return new_allocation, "single_gpu"
            except RuntimeError:
                pass

        # Strategy 3: Fall back to CPU
        logger.warning("No healthy GPUs available, falling back to CPU")
        return None, "cpu_fallback"

    def reset_failed_gpus(self):
        """
        Reset the failed GPU tracking.

        Useful for testing or after manual GPU recovery.

        Example:
            >>> manager = GPUManager()
            >>> manager.reset_failed_gpus()
        """
        logger.info(f"Resetting failed GPU tracking: {self._failed_gpus}")
        self._failed_gpus.clear()


def get_gpu_manager() -> GPUManager:
    """
    Get singleton GPU manager instance.

    Returns:
        GPUManager singleton instance

    Example:
        >>> manager = get_gpu_manager()
        >>> gpus = manager.detect_gpus()
    """
    if not hasattr(get_gpu_manager, "_instance"):
        get_gpu_manager._instance = GPUManager()
    return get_gpu_manager._instance
