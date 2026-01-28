"""
Dynamic batching for inference requests.

This module implements dynamic batching to group compatible inference requests
for batch processing, maximizing GPU utilization and throughput. Supports
adaptive batch sizing, timeout-based triggers, and graceful degradation.
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from threading import Thread, Lock, Event
import queue

from mm_orch.logger import get_logger
from mm_orch.optimization.config import BatcherConfig

logger = get_logger(__name__)


@dataclass
class InferenceRequest:
    """
    Represents a single inference request.

    Attributes:
        request_id: Unique identifier for the request
        model_name: Name of the model to use
        inputs: Input data for inference
        parameters: Additional inference parameters
        timestamp: Request creation timestamp
        sequence_length: Length of input sequence (for padding)
    """

    request_id: str
    model_name: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    sequence_length: Optional[int] = None

    def __post_init__(self):
        """Calculate sequence length if not provided."""
        if self.sequence_length is None and "input_ids" in self.inputs:
            # Assume input_ids is a list or tensor-like
            input_ids = self.inputs["input_ids"]
            if hasattr(input_ids, "__len__"):
                self.sequence_length = len(input_ids)
            else:
                self.sequence_length = 0


@dataclass
class BatchedRequest:
    """
    Represents a batch of compatible requests.

    Attributes:
        batch_id: Unique identifier for the batch
        model_name: Model name (all requests in batch use same model)
        requests: List of individual requests in the batch
        max_sequence_length: Maximum sequence length in batch
        created_at: Batch creation timestamp
    """

    batch_id: str
    model_name: str
    requests: List[InferenceRequest]
    max_sequence_length: int
    created_at: float = field(default_factory=time.time)

    @property
    def batch_size(self) -> int:
        """Return number of requests in batch."""
        return len(self.requests)

    def get_padded_inputs(self) -> Dict[str, Any]:
        """
        Get batch inputs with padding applied.

        Pads variable-length sequences to the maximum length in the batch.
        For input_ids, applies padding token (0 by default).
        For attention_mask, sets padded positions to 0.

        Returns:
            Dictionary with batched and padded inputs
        """
        if not self.requests:
            return {}

        # Collect all inputs
        all_input_ids = []
        all_attention_masks = []
        original_lengths = []

        for req in self.requests:
            inputs = req.inputs

            # Extract input_ids
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                if isinstance(input_ids, list):
                    all_input_ids.append(input_ids)
                    original_lengths.append(len(input_ids))
                else:
                    # Handle tensor-like objects
                    all_input_ids.append(input_ids)
                    original_lengths.append(len(input_ids) if hasattr(input_ids, "__len__") else 0)

            # Extract attention_mask if present
            if "attention_mask" in inputs:
                attention_mask = inputs["attention_mask"]
                if isinstance(attention_mask, list):
                    all_attention_masks.append(attention_mask)
                else:
                    all_attention_masks.append(attention_mask)

        # Apply padding to max_sequence_length
        padded_input_ids = []
        padded_attention_masks = []

        for i, input_ids in enumerate(all_input_ids):
            current_length = original_lengths[i]
            padding_length = self.max_sequence_length - current_length

            if isinstance(input_ids, list):
                # Pad with 0 (standard padding token)
                padded_ids = input_ids + [0] * padding_length
                padded_input_ids.append(padded_ids)

                # Create or pad attention mask
                if i < len(all_attention_masks):
                    attention_mask = all_attention_masks[i]
                    if isinstance(attention_mask, list):
                        padded_mask = attention_mask + [0] * padding_length
                    else:
                        # Assume it's already correct length
                        padded_mask = attention_mask
                else:
                    # Create attention mask: 1 for real tokens, 0 for padding
                    padded_mask = [1] * current_length + [0] * padding_length

                padded_attention_masks.append(padded_mask)
            else:
                # For non-list inputs, pass through as-is
                padded_input_ids.append(input_ids)
                if i < len(all_attention_masks):
                    padded_attention_masks.append(all_attention_masks[i])

        # Build batched inputs
        batched_inputs = {
            "batch_size": self.batch_size,
            "max_length": self.max_sequence_length,
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "original_lengths": original_lengths,
            "request_ids": [req.request_id for req in self.requests],
        }

        return batched_inputs


@dataclass
class InferenceResult:
    """
    Result from inference execution.

    Attributes:
        request_id: ID of the original request
        outputs: Inference outputs
        latency_ms: Inference latency in milliseconds
        metadata: Additional metadata
    """

    request_id: str
    outputs: Dict[str, Any]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicBatcher:
    """
    Dynamic batcher for grouping compatible inference requests.

    Features:
    - Async request queue with background batching loop
    - Request ID generation and tracking
    - Batch formation based on size and timeout triggers
    - Compatible request grouping by model
    - Adaptive batch sizing based on system load
    - Variable-length sequence padding
    - Disable mode for individual request processing

    The batcher runs a background thread that continuously monitors the request
    queue and forms batches when either the size threshold is reached or the
    timeout expires.
    """

    def __init__(self, config: BatcherConfig):
        """
        Initialize dynamic batcher.

        Args:
            config: Batcher configuration
        """
        self.config = config
        self._request_queues: Dict[str, deque] = defaultdict(deque)
        self._pending_results: Dict[str, queue.Queue] = {}
        self._lock = Lock()
        self._stop_event = Event()
        self._batch_thread: Optional[Thread] = None
        self._running = False

        # Performance tracking for adaptive batching
        self._latency_history: deque = deque(maxlen=100)
        self._throughput_history: deque = deque(maxlen=100)
        self._current_batch_size = config.max_batch_size
        self._last_adaptation_time = time.time()
        self._adaptation_interval = 10.0  # Adapt every 10 seconds

        # Adaptive batching targets
        self._target_latency_ms = 100.0  # Target 100ms latency
        self._latency_tolerance = 0.2  # 20% tolerance

        logger.info(
            f"DynamicBatcher initialized: "
            f"enabled={config.enabled}, "
            f"max_batch_size={config.max_batch_size}, "
            f"timeout={config.batch_timeout_ms}ms, "
            f"adaptive={config.adaptive_batching}"
        )

    def start(self):
        """Start background batching loop."""
        if self._running:
            logger.warning("DynamicBatcher already running")
            return

        if not self.config.enabled:
            logger.info("DynamicBatcher disabled, not starting background thread")
            return

        self._running = True
        self._stop_event.clear()
        self._batch_thread = Thread(target=self._batching_loop, daemon=True)
        self._batch_thread.start()
        logger.info("DynamicBatcher background thread started")

    def stop(self, timeout: float = 5.0):
        """
        Stop batching and process remaining requests.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self._running:
            return

        logger.info("Stopping DynamicBatcher...")
        self._stop_event.set()

        if self._batch_thread:
            self._batch_thread.join(timeout=timeout)
            if self._batch_thread.is_alive():
                logger.warning("DynamicBatcher thread did not stop gracefully")

        # Process any remaining requests
        with self._lock:
            for model_name, req_queue in self._request_queues.items():
                if req_queue:
                    logger.info(f"Processing {len(req_queue)} remaining requests for {model_name}")
                    # In production, these would be processed or cancelled

        self._running = False
        logger.info("DynamicBatcher stopped")

    def add_request(
        self, model_name: str, inputs: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add inference request to batch queue.

        If batching is disabled, processes request immediately.

        Args:
            model_name: Name of the model to use
            inputs: Input data for inference
            parameters: Additional inference parameters

        Returns:
            Unique request ID for tracking
        """
        request_id = str(uuid.uuid4())

        request = InferenceRequest(
            request_id=request_id, model_name=model_name, inputs=inputs, parameters=parameters or {}
        )

        # Create result queue for this request
        result_queue = queue.Queue(maxsize=1)

        with self._lock:
            self._pending_results[request_id] = result_queue

        # If batching is disabled, process immediately
        if not self.config.enabled:
            logger.debug(f"Batching disabled, processing request {request_id} individually")
            self._process_individual_request(request)
        else:
            # Add to queue for batching
            with self._lock:
                self._request_queues[model_name].append(request)

            logger.debug(
                f"Request {request_id} added to queue for model {model_name}, "
                f"queue_size={len(self._request_queues[model_name])}"
            )

        return request_id

    def _process_individual_request(self, request: InferenceRequest):
        """
        Process a single request without batching.

        Args:
            request: Request to process
        """
        start_time = time.time()

        # Simulate processing (in production, this would call actual inference)
        processing_time = 0.01  # 10ms
        time.sleep(processing_time)

        latency_ms = (time.time() - start_time) * 1000

        # Create result
        result = InferenceResult(
            request_id=request.request_id,
            outputs={"simulated": True, "individual": True},
            latency_ms=latency_ms,
            metadata={"batch_size": 1, "model_name": request.model_name, "batching_disabled": True},
        )

        # Put result in queue
        with self._lock:
            if request.request_id in self._pending_results:
                self._pending_results[request.request_id].put(result)

        logger.debug(f"Processed individual request {request.request_id} " f"in {latency_ms:.2f}ms")

    def get_result(self, request_id: str, timeout: Optional[float] = None) -> InferenceResult:
        """
        Wait for and retrieve result for a request.

        Args:
            request_id: ID of the request
            timeout: Maximum time to wait (None for infinite)

        Returns:
            InferenceResult for the request

        Raises:
            KeyError: If request_id is not found
            queue.Empty: If timeout expires before result is ready
        """
        with self._lock:
            if request_id not in self._pending_results:
                raise KeyError(f"Request {request_id} not found")
            result_queue = self._pending_results[request_id]

        try:
            result = result_queue.get(timeout=timeout)

            # Clean up
            with self._lock:
                del self._pending_results[request_id]

            return result
        except queue.Empty:
            logger.warning(f"Timeout waiting for result of request {request_id}")
            raise

    async def get_result_async(
        self, request_id: str, timeout: Optional[float] = None
    ) -> InferenceResult:
        """
        Async version of get_result.

        Args:
            request_id: ID of the request
            timeout: Maximum time to wait (None for infinite)

        Returns:
            InferenceResult for the request
        """
        # Run blocking get_result in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_result, request_id, timeout)

    def _batching_loop(self):
        """
        Background loop for batch formation and processing.

        Continuously monitors request queues and forms batches when:
        1. Batch size threshold is reached
        2. Timeout expires with pending requests
        """
        logger.info("Batching loop started")

        while not self._stop_event.is_set():
            try:
                # Get list of models WITHOUT holding lock for entire loop
                with self._lock:
                    models_to_process = list(self._request_queues.keys())

                # Process each model's queue
                for model_name in models_to_process:
                    if self._stop_event.is_set():
                        break
                    self._process_model_queue(model_name)

                # Wait with shorter timeout to be more responsive
                # Use wait() instead of sleep() so we can be interrupted
                self._stop_event.wait(timeout=0.005)  # 5ms

            except Exception as e:
                logger.error(f"Error in batching loop: {e}", exc_info=True)
                if self._stop_event.is_set():
                    break

        # Process any remaining requests before exit
        self._process_remaining_requests()

        logger.info("Batching loop stopped")

    def _process_remaining_requests(self):
        """Process any remaining requests in queues before shutdown."""
        logger.info("Processing remaining requests before shutdown")

        with self._lock:
            models_with_requests = [
                (model_name, list(req_queue))
                for model_name, req_queue in self._request_queues.items()
                if req_queue
            ]

        for model_name, requests in models_with_requests:
            logger.info(f"Processing {len(requests)} remaining requests for {model_name}")

            with self._lock:
                req_queue = self._request_queues[model_name]

                while req_queue:
                    # Force process remaining requests
                    batch = self._form_batch(model_name, req_queue)

                    if not batch:
                        break

            # Process batch outside lock
            if batch:
                logger.debug(
                    f"Processing remaining batch {batch.batch_id}: {batch.batch_size} requests"
                )
                self._simulate_batch_processing(batch)

    def _process_model_queue(self, model_name: str):
        """
        Process request queue for a specific model.

        Args:
            model_name: Name of the model
        """
        # Check and form batch while holding lock
        with self._lock:
            if model_name not in self._request_queues:
                return

            req_queue = self._request_queues[model_name]

            if not req_queue:
                return

            # Check if we should form a batch
            should_batch = False
            reason = ""

            # Trigger 1: Batch size threshold reached
            if len(req_queue) >= self._current_batch_size:
                should_batch = True
                reason = "size_threshold"

            # Trigger 2: Timeout expired for oldest request
            elif req_queue:
                oldest_request = req_queue[0]
                age_ms = (time.time() - oldest_request.timestamp) * 1000
                if age_ms >= self.config.batch_timeout_ms:
                    should_batch = True
                    reason = "timeout"

            if not should_batch:
                return

            # Form batch (removes requests from queue)
            batch = self._form_batch(model_name, req_queue)

        # Process batch OUTSIDE the lock to avoid blocking other operations
        if batch:
            logger.debug(
                f"Formed batch {batch.batch_id} for {model_name}: "
                f"size={batch.batch_size}, reason={reason}"
            )

            # In production, this would trigger actual inference
            # For now, we'll simulate processing
            self._simulate_batch_processing(batch)

    def _form_batch(self, model_name: str, req_queue: deque) -> Optional[BatchedRequest]:
        """
        Form a batch from the request queue.

        Groups compatible requests by model. All requests in a batch must:
        - Use the same model
        - Have compatible parameters (same generation settings)

        Args:
            model_name: Name of the model
            req_queue: Queue of requests for this model

        Returns:
            BatchedRequest or None if queue is empty
        """
        if not req_queue:
            return None

        # Determine batch size
        batch_size = min(len(req_queue), self._current_batch_size)

        # Extract requests
        requests = []
        max_seq_len = 0

        for _ in range(batch_size):
            if not req_queue:
                break
            req = req_queue.popleft()

            # Check compatibility with existing requests in batch
            if requests and not self._are_requests_compatible(requests[0], req):
                # Put incompatible request back
                req_queue.appendleft(req)
                break

            requests.append(req)
            if req.sequence_length:
                max_seq_len = max(max_seq_len, req.sequence_length)

        if not requests:
            return None

        batch_id = str(uuid.uuid4())
        batch = BatchedRequest(
            batch_id=batch_id,
            model_name=model_name,
            requests=requests,
            max_sequence_length=max_seq_len,
        )

        return batch

    def _are_requests_compatible(self, req1: InferenceRequest, req2: InferenceRequest) -> bool:
        """
        Check if two requests are compatible for batching.

        Requests are compatible if they have the same:
        - Model name (already guaranteed by queue structure)
        - Generation parameters (temperature, top_p, max_length, etc.)

        Args:
            req1: First request
            req2: Second request

        Returns:
            True if requests can be batched together
        """
        # Model name already matches (same queue)

        # Check if parameters match
        # For simplicity, we check a subset of common parameters
        param_keys = ["temperature", "top_p", "top_k", "max_length", "do_sample"]

        for key in param_keys:
            val1 = req1.parameters.get(key)
            val2 = req2.parameters.get(key)

            # If both have the parameter, they must match
            if val1 is not None and val2 is not None and val1 != val2:
                return False

        return True

    def _simulate_batch_processing(self, batch: BatchedRequest):
        """
        Simulate batch processing (placeholder for actual inference).

        Args:
            batch: Batch to process
        """
        # Simulate processing time
        processing_time = 0.01 * batch.batch_size  # 10ms per request
        time.sleep(processing_time)

        # Create results for each request
        for request in batch.requests:
            result = InferenceResult(
                request_id=request.request_id,
                outputs={"simulated": True, "batch_id": batch.batch_id},
                latency_ms=processing_time * 1000,
                metadata={"batch_size": batch.batch_size, "model_name": batch.model_name},
            )

            # Put result in queue
            with self._lock:
                if request.request_id in self._pending_results:
                    self._pending_results[request.request_id].put(result)

        # Track performance for adaptive batching
        if self.config.adaptive_batching:
            self._latency_history.append(processing_time * 1000)
            self._throughput_history.append(batch.batch_size / processing_time)

            # Periodically adapt batch size
            self._adapt_batch_size()

    def _adapt_batch_size(self):
        """
        Adapt batch size based on system load and performance metrics.

        Strategy:
        - If latency is too high, reduce batch size
        - If latency is low and throughput can improve, increase batch size
        - Adjust gradually to avoid oscillation
        """
        current_time = time.time()

        # Only adapt at intervals
        if current_time - self._last_adaptation_time < self._adaptation_interval:
            return

        self._last_adaptation_time = current_time

        if not self._latency_history or not self.config.adaptive_batching:
            return

        # Calculate average latency
        avg_latency = sum(self._latency_history) / len(self._latency_history)

        # Calculate target range
        target_min = self._target_latency_ms * (1 - self._latency_tolerance)
        target_max = self._target_latency_ms * (1 + self._latency_tolerance)

        old_batch_size = self._current_batch_size

        # Adjust batch size based on latency
        if avg_latency > target_max:
            # Latency too high, reduce batch size
            reduction = max(1, int(self._current_batch_size * 0.1))  # Reduce by 10%
            self._current_batch_size = max(
                self.config.min_batch_size, self._current_batch_size - reduction
            )
            reason = "high_latency"
        elif avg_latency < target_min:
            # Latency low, can increase batch size
            increase = max(1, int(self._current_batch_size * 0.1))  # Increase by 10%
            self._current_batch_size = min(
                self.config.max_batch_size, self._current_batch_size + increase
            )
            reason = "low_latency"
        else:
            # Latency in target range, no change
            return

        if self._current_batch_size != old_batch_size:
            logger.info(
                f"Adapted batch size: {old_batch_size} -> {self._current_batch_size} "
                f"(reason={reason}, avg_latency={avg_latency:.2f}ms, "
                f"target={self._target_latency_ms:.2f}ms)"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get batcher statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_queued = sum(len(q) for q in self._request_queues.values())
            pending_results = len(self._pending_results)

        avg_latency = (
            sum(self._latency_history) / len(self._latency_history)
            if self._latency_history
            else 0.0
        )

        avg_throughput = (
            sum(self._throughput_history) / len(self._throughput_history)
            if self._throughput_history
            else 0.0
        )

        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "total_queued_requests": total_queued,
            "pending_results": pending_results,
            "current_batch_size": self._current_batch_size,
            "avg_latency_ms": avg_latency,
            "avg_throughput_rps": avg_throughput,
            "queues_by_model": {model: len(queue) for model, queue in self._request_queues.items()},
        }
