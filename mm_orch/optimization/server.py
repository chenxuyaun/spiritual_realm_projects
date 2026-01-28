"""
Inference Server for long-running deployments.

This module provides the InferenceServer class that manages server lifecycle,
request queuing, model pre-loading, and graceful shutdown for production
deployments.
"""

import asyncio
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue, Full, Empty
from typing import Any, Dict, List, Optional

from mm_orch.logger import get_logger
from mm_orch.optimization.config import OptimizationConfig, ServerConfig
from mm_orch.optimization.manager import OptimizationManager, InferenceResult

logger = get_logger(__name__)


class ServerStatus(Enum):
    """Enumeration of server status states."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class HealthStatus:
    """
    Health status information for the server.
    
    Attributes:
        status: Current server status (healthy, degraded, unhealthy)
        uptime_seconds: Server uptime in seconds
        models_loaded: List of model names currently loaded
        engines_available: List of available inference engines
        queue_size: Current number of requests in queue
        queue_capacity: Maximum queue capacity
        degradation_reasons: List of reasons for degraded status
    """
    status: str
    uptime_seconds: float
    models_loaded: List[str]
    engines_available: List[str]
    queue_size: int
    queue_capacity: int
    degradation_reasons: List[str] = None
    
    def __post_init__(self):
        """Initialize degradation_reasons if not provided."""
        if self.degradation_reasons is None:
            self.degradation_reasons = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize HealthStatus to dictionary.
        
        Returns:
            Dictionary representation of health status
        """
        return {
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
            "models_loaded": self.models_loaded,
            "engines_available": self.engines_available,
            "queue_size": self.queue_size,
            "queue_capacity": self.queue_capacity,
            "degradation_reasons": self.degradation_reasons
        }


@dataclass
class ReadinessStatus:
    """
    Readiness status information for the server.
    
    Attributes:
        ready: Whether the server is ready to accept requests
        models_loaded: Number of models loaded
        queue_available: Whether the queue has capacity
        engines_available: Number of available engines
    """
    ready: bool
    models_loaded: int
    queue_available: bool
    engines_available: int
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize ReadinessStatus to dictionary.
        
        Returns:
            Dictionary representation of readiness status
        """
        return {
            "ready": self.ready,
            "models_loaded": self.models_loaded,
            "queue_available": self.queue_available,
            "engines_available": self.engines_available
        }


@dataclass
class InferenceRequest:
    """
    Request for inference execution.
    
    Attributes:
        request_id: Unique identifier for the request
        model_name: Name of the model to use
        inputs: Input data for inference
        engine_preference: Optional engine preference
        timestamp: Request creation timestamp
    """
    request_id: str
    model_name: str
    inputs: Dict[str, Any]
    engine_preference: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp to current time if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


class InferenceServer:
    """
    Long-running inference server with request queuing and lifecycle management.
    
    The InferenceServer provides a production-ready deployment mode with:
    - Model pre-loading at startup
    - Request queue management with capacity limits
    - Graceful shutdown with pending request completion
    - Health and readiness endpoints
    - Degradation status tracking
    
    Attributes:
        config: Server configuration
        opt_config: Optimization configuration
        _status: Current server status
        _start_time: Server start timestamp
        _request_queue: Queue for incoming inference requests
        _result_futures: Dictionary mapping request IDs to result futures
        _worker_thread: Background thread for processing requests
        _shutdown_event: Event to signal shutdown
        _opt_manager: OptimizationManager for inference execution
    
    Example:
        >>> config = OptimizationConfig()
        >>> server = InferenceServer(config)
        >>> server.start()
        >>> # Server is now running and accepting requests
        >>> health = server.health_check()
        >>> print(health.status)
        'healthy'
        >>> server.stop()
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize InferenceServer with configuration.
        
        Args:
            config: Optimization configuration including server settings
        """
        self.config = config
        self.server_config = config.server
        self._status = ServerStatus.STOPPED
        self._start_time: Optional[datetime] = None
        
        # Request queue and results
        self._request_queue: Queue[InferenceRequest] = Queue(
            maxsize=self.server_config.queue_capacity
        )
        self._result_futures: Dict[str, Any] = {}
        self._result_lock = threading.Lock()
        
        # Worker thread for processing requests
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Optimization manager (initialized in start())
        self._opt_manager: Optional[OptimizationManager] = None
        
        # Track loaded models
        self._loaded_models: List[str] = []
        
        # Track degradation reasons
        self._degradation_reasons: List[str] = []
        
        logger.info("InferenceServer initialized")
    
    def start(self):
        """
        Start the inference server.
        
        Performs the following:
        1. Initialize OptimizationManager
        2. Pre-load configured models
        3. Start background worker thread
        4. Set up signal handlers for graceful shutdown
        
        Raises:
            RuntimeError: If server is already running
        
        Example:
            >>> server = InferenceServer(config)
            >>> server.start()
            >>> # Server is now running
        """
        if self._status != ServerStatus.STOPPED:
            raise RuntimeError(f"Server already running (status: {self._status.value})")
        
        logger.info("Starting InferenceServer...")
        self._status = ServerStatus.STARTING
        self._start_time = datetime.now()
        self._shutdown_event.clear()
        
        try:
            # Initialize OptimizationManager
            logger.info("Initializing OptimizationManager...")
            self._opt_manager = OptimizationManager(self.config)
            
            # Pre-load configured models
            if self.server_config.preload_models:
                logger.info(
                    f"Pre-loading {len(self.server_config.preload_models)} models..."
                )
                self._preload_models()
            
            # Start worker thread
            logger.info("Starting worker thread...")
            self._worker_thread = threading.Thread(
                target=self._process_requests,
                name="InferenceServerWorker",
                daemon=False
            )
            self._worker_thread.start()
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Update status based on engine availability
            self._update_status()
            
            logger.info(f"InferenceServer started (status: {self._status.value})")
            
        except Exception as e:
            logger.error(f"Failed to start InferenceServer: {e}", exc_info=True)
            self._status = ServerStatus.UNHEALTHY
            self._degradation_reasons.append(f"Startup failed: {e}")
            raise
    
    def stop(self, timeout: Optional[int] = None):
        """
        Stop the inference server gracefully.
        
        Completes pending requests before shutting down. If timeout is reached,
        forces shutdown.
        
        Args:
            timeout: Maximum time to wait for pending requests (seconds).
                    Uses graceful_shutdown_timeout from config if not provided.
        
        Example:
            >>> server.stop(timeout=30)
            >>> # Server stopped after completing pending requests
        """
        if self._status == ServerStatus.STOPPED:
            logger.warning("Server already stopped")
            return
        
        if timeout is None:
            timeout = self.server_config.graceful_shutdown_timeout
        
        logger.info(f"Stopping InferenceServer (timeout: {timeout}s)...")
        self._status = ServerStatus.SHUTTING_DOWN
        
        # Signal shutdown to worker thread
        self._shutdown_event.set()
        
        # Wait for worker thread to complete pending requests
        if self._worker_thread and self._worker_thread.is_alive():
            logger.info("Waiting for worker thread to complete pending requests...")
            self._worker_thread.join(timeout=timeout)
            
            if self._worker_thread.is_alive():
                logger.warning(
                    f"Worker thread did not complete within {timeout}s, "
                    "forcing shutdown"
                )
        
        # Clear any remaining requests
        remaining = self._request_queue.qsize()
        if remaining > 0:
            logger.warning(f"Discarding {remaining} pending requests")
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                except Empty:
                    break
        
        self._status = ServerStatus.STOPPED
        logger.info("InferenceServer stopped")
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals (SIGINT, SIGTERM).
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def _preload_models(self):
        """
        Pre-load configured models at startup.
        
        Attempts to load each model in the preload list. Failures are logged
        but don't prevent server startup (degraded mode).
        """
        for model_name in self.server_config.preload_models:
            try:
                logger.info(f"Pre-loading model: {model_name}")
                
                # Perform a dummy inference to trigger model loading
                # This ensures the model is loaded and cached
                dummy_inputs = {"prompts": ["test"]}
                result = self._opt_manager.infer(model_name, dummy_inputs)
                
                self._loaded_models.append(model_name)
                logger.info(
                    f"Model {model_name} pre-loaded successfully "
                    f"(engine: {result.engine_used})"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to pre-load model {model_name}: {e}",
                    exc_info=True
                )
                self._degradation_reasons.append(
                    f"Failed to pre-load model {model_name}: {e}"
                )
    
    def _update_status(self):
        """
        Update server status based on current state.
        
        Determines if server is healthy, degraded, or unhealthy based on:
        - Engine availability
        - Model loading status
        - Queue capacity
        """
        if self._status == ServerStatus.SHUTTING_DOWN or self._status == ServerStatus.STOPPED:
            return
        
        # Check engine availability
        available_engines = self._opt_manager.get_available_engines()
        
        # Determine status
        if not available_engines:
            self._status = ServerStatus.UNHEALTHY
            if "No engines available" not in self._degradation_reasons:
                self._degradation_reasons.append("No engines available")
        elif len(self._degradation_reasons) > 0:
            self._status = ServerStatus.DEGRADED
        else:
            self._status = ServerStatus.HEALTHY
            self._degradation_reasons.clear()
    
    def _process_requests(self):
        """
        Background worker thread that processes requests from the queue.
        
        Continuously polls the request queue and executes inference requests
        until shutdown is signaled. Completes pending requests during graceful
        shutdown.
        """
        logger.info("Worker thread started")
        
        while not self._shutdown_event.is_set() or not self._request_queue.empty():
            try:
                # Get request from queue with timeout
                try:
                    request = self._request_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                logger.debug(f"Processing request {request.request_id}")
                
                # Execute inference
                try:
                    result = self._opt_manager.infer(
                        model_name=request.model_name,
                        inputs=request.inputs,
                        engine_preference=request.engine_preference
                    )
                    
                    # Store result
                    with self._result_lock:
                        self._result_futures[request.request_id] = {
                            "status": "success",
                            "result": result,
                            "error": None
                        }
                    
                    logger.debug(
                        f"Request {request.request_id} completed successfully "
                        f"(engine: {result.engine_used}, latency: {result.latency_ms:.2f}ms)"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Request {request.request_id} failed: {e}",
                        exc_info=True
                    )
                    
                    # Store error
                    with self._result_lock:
                        self._result_futures[request.request_id] = {
                            "status": "error",
                            "result": None,
                            "error": str(e)
                        }
                
                finally:
                    self._request_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker thread error: {e}", exc_info=True)
        
        logger.info("Worker thread stopped")
    
    def submit_request(
        self,
        request_id: str,
        model_name: str,
        inputs: Dict[str, Any],
        engine_preference: Optional[str] = None
    ) -> bool:
        """
        Submit an inference request to the queue.
        
        Args:
            request_id: Unique identifier for the request
            model_name: Name of the model to use
            inputs: Input data for inference
            engine_preference: Optional engine preference
        
        Returns:
            True if request was queued, False if queue is full
        
        Raises:
            RuntimeError: If server is not running
        
        Example:
            >>> success = server.submit_request(
            ...     request_id="req-123",
            ...     model_name="qwen-chat",
            ...     inputs={"prompts": ["Hello"]}
            ... )
            >>> if success:
            ...     result = server.get_result("req-123", timeout=10.0)
        """
        if self._status not in (ServerStatus.HEALTHY, ServerStatus.DEGRADED):
            raise RuntimeError(
                f"Server not ready to accept requests (status: {self._status.value})"
            )
        
        request = InferenceRequest(
            request_id=request_id,
            model_name=model_name,
            inputs=inputs,
            engine_preference=engine_preference
        )
        
        try:
            self._request_queue.put_nowait(request)
            logger.debug(f"Request {request_id} queued")
            return True
        except Full:
            logger.warning(f"Request {request_id} rejected: queue full")
            return False
    
    def get_result(
        self,
        request_id: str,
        timeout: Optional[float] = None
    ) -> Optional[InferenceResult]:
        """
        Get the result for a submitted request.
        
        Blocks until the result is available or timeout is reached.
        
        Args:
            request_id: Unique identifier for the request
            timeout: Maximum time to wait for result (seconds)
        
        Returns:
            InferenceResult if successful, None if timeout or error
        
        Raises:
            RuntimeError: If request failed with error
        
        Example:
            >>> result = server.get_result("req-123", timeout=10.0)
            >>> if result:
            ...     print(result.outputs)
        """
        start_time = time.time()
        
        while True:
            # Check if result is available
            with self._result_lock:
                if request_id in self._result_futures:
                    future = self._result_futures.pop(request_id)
                    
                    if future["status"] == "success":
                        return future["result"]
                    else:
                        raise RuntimeError(
                            f"Request {request_id} failed: {future['error']}"
                        )
            
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Request {request_id} timed out after {timeout}s")
                    return None
            
            # Wait before checking again
            time.sleep(0.1)
    
    def health_check(self) -> HealthStatus:
        """
        Get current health status of the server.
        
        Returns:
            HealthStatus with server health information
        
        Example:
            >>> health = server.health_check()
            >>> print(f"Status: {health.status}")
            >>> print(f"Uptime: {health.uptime_seconds}s")
            >>> print(f"Queue: {health.queue_size}/{health.queue_capacity}")
        """
        # Calculate uptime
        uptime = 0.0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()
        
        # Get available engines
        engines_available = []
        if self._opt_manager:
            engines_available = self._opt_manager.get_available_engines()
        
        # Get queue size
        queue_size = self._request_queue.qsize()
        
        return HealthStatus(
            status=self._status.value,
            uptime_seconds=uptime,
            models_loaded=self._loaded_models.copy(),
            engines_available=engines_available,
            queue_size=queue_size,
            queue_capacity=self.server_config.queue_capacity,
            degradation_reasons=self._degradation_reasons.copy()
        )
    
    def readiness_check(self) -> ReadinessStatus:
        """
        Get current readiness status of the server.
        
        Returns:
            ReadinessStatus indicating if server is ready for requests
        
        Example:
            >>> readiness = server.readiness_check()
            >>> if readiness.ready:
            ...     print("Server is ready to accept requests")
        """
        # Server is ready if status is healthy or degraded
        ready = self._status in (ServerStatus.HEALTHY, ServerStatus.DEGRADED)
        
        # Get available engines count
        engines_count = 0
        if self._opt_manager:
            engines_count = len(self._opt_manager.get_available_engines())
        
        # Check if queue has capacity
        queue_available = self._request_queue.qsize() < self.server_config.queue_capacity
        
        return ReadinessStatus(
            ready=ready,
            models_loaded=len(self._loaded_models),
            queue_available=queue_available,
            engines_available=engines_count
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current server status as a dictionary.
        
        This is a convenience method that returns health information
        in dictionary format for easier testing and debugging.
        
        Returns:
            Dictionary with server status information
        """
        health = self.health_check()
        return {
            "status": health.status,
            "uptime_seconds": health.uptime_seconds,
            "models_loaded": health.models_loaded,
            "engines_available": health.engines_available,
            "queue_size": health.queue_size,
            "queue_capacity": health.queue_capacity,
            "degradation_reasons": health.degradation_reasons
        }
