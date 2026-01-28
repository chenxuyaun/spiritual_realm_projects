"""
Optimization Manager for inference engine orchestration.

This module provides the OptimizationManager class that coordinates inference
engine selection and execution with fallback strategies. It manages a registry
of available engines (vLLM, DeepSpeed, ONNX, PyTorch) and implements the
fallback chain logic for graceful degradation.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from mm_orch.logger import get_logger
from mm_orch.optimization.config import OptimizationConfig
from mm_orch.monitoring.prometheus_exporter import PrometheusExporter
from mm_orch.monitoring.otel_tracer import OTelTracer

logger = get_logger(__name__)

# Lazy import to avoid circular dependencies
_vllm_engine = None


class EngineType(Enum):
    """Enumeration of supported inference engines."""

    VLLM = "vllm"
    DEEPSPEED = "deepspeed"
    ONNX = "onnx"
    PYTORCH = "pytorch"


@dataclass
class EngineStatus:
    """
    Status information for an inference engine.

    Attributes:
        name: Engine name (vllm, deepspeed, onnx, pytorch)
        available: Whether the engine is available and functional
        models_loaded: List of model names currently loaded in this engine
        error_message: Error message if engine is unavailable
        last_check: Timestamp of last availability check
    """

    name: str
    available: bool
    models_loaded: List[str]
    error_message: Optional[str] = None
    last_check: Optional[datetime] = None

    def __post_init__(self):
        """Set last_check to current time if not provided."""
        if self.last_check is None:
            self.last_check = datetime.now()


@dataclass
class InferenceResult:
    """
    Result from inference execution.

    Attributes:
        outputs: Dictionary of output tensors/data
        engine_used: Name of the engine that performed inference
        latency_ms: Inference latency in milliseconds
        batch_size: Number of requests in the batch
        cache_hit: Whether KV cache was used (if applicable)
        metadata: Additional metadata about the inference
    """

    outputs: Dict[str, Any]
    engine_used: str
    latency_ms: float
    batch_size: int = 1
    cache_hit: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize InferenceResult to dictionary.

        Returns:
            Dictionary representation of the inference result

        Example:
            >>> result = InferenceResult(
            ...     outputs={"text": "Hello"},
            ...     engine_used="pytorch",
            ...     latency_ms=10.5
            ... )
            >>> result_dict = result.to_dict()
            >>> result_dict["engine_used"]
            'pytorch'
        """
        return {
            "outputs": self.outputs,
            "engine_used": self.engine_used,
            "latency_ms": self.latency_ms,
            "batch_size": self.batch_size,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """
        Deserialize InferenceResult from dictionary.

        Args:
            data: Dictionary containing inference result data

        Returns:
            InferenceResult instance

        Raises:
            KeyError: If required fields are missing
            TypeError: If field types are invalid

        Example:
            >>> data = {
            ...     "outputs": {"text": "Hello"},
            ...     "engine_used": "pytorch",
            ...     "latency_ms": 10.5,
            ...     "batch_size": 1,
            ...     "cache_hit": False,
            ...     "metadata": {}
            ... }
            >>> result = InferenceResult.from_dict(data)
            >>> result.engine_used
            'pytorch'
        """
        # Validate required fields
        required_fields = ["outputs", "engine_used", "latency_ms"]
        for field in required_fields:
            if field not in data:
                raise KeyError(f"Missing required field: {field}")

        # Create instance with all fields
        return cls(
            outputs=data["outputs"],
            engine_used=data["engine_used"],
            latency_ms=data["latency_ms"],
            batch_size=data.get("batch_size", 1),
            cache_hit=data.get("cache_hit", False),
            metadata=data.get("metadata", {}),
        )


class OptimizationManager:
    """
    Manages inference engine selection and execution with fallback strategies.

    The OptimizationManager coordinates multiple inference engines (vLLM,
    DeepSpeed, ONNX Runtime, PyTorch) and implements a fallback chain for
    graceful degradation when engines fail or are unavailable.

    Fallback chain: vLLM → DeepSpeed → ONNX → PyTorch

    Attributes:
        config: Optimization configuration
        _engine_registry: Registry of available engines with their status
        _engine_preference: Ordered list of preferred engines

    Example:
        >>> config = OptimizationConfig()
        >>> manager = OptimizationManager(config)
        >>> result = manager.infer("qwen-chat", {"input": "Hello"})
        >>> print(f"Used engine: {result.engine_used}")
    """

    def __init__(
        self,
        config: OptimizationConfig,
        prometheus_exporter: Optional[PrometheusExporter] = None,
        otel_tracer: Optional[OTelTracer] = None,
    ):
        """
        Initialize OptimizationManager with configuration.

        Args:
            config: Optimization configuration specifying engine settings
            prometheus_exporter: Optional Prometheus metrics exporter
            otel_tracer: Optional OpenTelemetry tracer
        """
        self.config = config
        self._engine_registry: Dict[str, EngineStatus] = {}
        self._engine_preference = config.engine_preference.copy()

        # Monitoring components (optional)
        self._prometheus = prometheus_exporter
        self._tracer = otel_tracer

        # Engine instances (lazy initialization)
        self._vllm_engine = None
        self._deepspeed_engine = None
        self._onnx_engine = None

        # Initialize engine registry
        self._initialize_engine_registry()

        logger.info(f"OptimizationManager initialized with engines: {self._engine_preference}")

    def _initialize_engine_registry(self):
        """
        Initialize the engine registry by detecting available engines.

        Checks each engine's availability and populates the registry with
        status information. Engines that fail initialization are marked as
        unavailable but remain in the registry for potential recovery.
        """
        # Check vLLM availability
        vllm_status = self._check_vllm_availability()
        self._engine_registry[EngineType.VLLM.value] = vllm_status

        # Check DeepSpeed availability
        deepspeed_status = self._check_deepspeed_availability()
        self._engine_registry[EngineType.DEEPSPEED.value] = deepspeed_status

        # Check ONNX Runtime availability
        onnx_status = self._check_onnx_availability()
        self._engine_registry[EngineType.ONNX.value] = onnx_status

        # PyTorch is always available (fallback)
        pytorch_status = EngineStatus(
            name=EngineType.PYTORCH.value,
            available=True,
            models_loaded=[],
            error_message=None,
            last_check=datetime.now(),
        )
        self._engine_registry[EngineType.PYTORCH.value] = pytorch_status

        # Log availability summary
        available_engines = [
            name for name, status in self._engine_registry.items() if status.available
        ]
        logger.info(f"Available engines: {available_engines}")

        unavailable_engines = [
            (name, status.error_message)
            for name, status in self._engine_registry.items()
            if not status.available
        ]
        if unavailable_engines:
            for name, error in unavailable_engines:
                logger.warning(f"Engine {name} unavailable: {error}")

    def _check_vllm_availability(self) -> EngineStatus:
        """
        Check if vLLM is available and functional.

        Returns:
            EngineStatus with availability information
        """
        if not self.config.vllm.enabled:
            return EngineStatus(
                name=EngineType.VLLM.value,
                available=False,
                models_loaded=[],
                error_message="vLLM disabled in configuration",
                last_check=datetime.now(),
            )

        try:
            # Try to import vLLM
            import vllm  # noqa: F401

            return EngineStatus(
                name=EngineType.VLLM.value,
                available=True,
                models_loaded=[],
                error_message=None,
                last_check=datetime.now(),
            )
        except ImportError as e:
            return EngineStatus(
                name=EngineType.VLLM.value,
                available=False,
                models_loaded=[],
                error_message=f"vLLM not installed: {e}",
                last_check=datetime.now(),
            )
        except Exception as e:
            return EngineStatus(
                name=EngineType.VLLM.value,
                available=False,
                models_loaded=[],
                error_message=f"vLLM initialization error: {e}",
                last_check=datetime.now(),
            )

    def _check_deepspeed_availability(self) -> EngineStatus:
        """
        Check if DeepSpeed is available and functional.

        Returns:
            EngineStatus with availability information
        """
        if not self.config.deepspeed.enabled:
            return EngineStatus(
                name=EngineType.DEEPSPEED.value,
                available=False,
                models_loaded=[],
                error_message="DeepSpeed disabled in configuration",
                last_check=datetime.now(),
            )

        try:
            # Try to import DeepSpeed
            import deepspeed  # noqa: F401

            return EngineStatus(
                name=EngineType.DEEPSPEED.value,
                available=True,
                models_loaded=[],
                error_message=None,
                last_check=datetime.now(),
            )
        except ImportError as e:
            return EngineStatus(
                name=EngineType.DEEPSPEED.value,
                available=False,
                models_loaded=[],
                error_message=f"DeepSpeed not installed: {e}",
                last_check=datetime.now(),
            )
        except Exception as e:
            return EngineStatus(
                name=EngineType.DEEPSPEED.value,
                available=False,
                models_loaded=[],
                error_message=f"DeepSpeed initialization error: {e}",
                last_check=datetime.now(),
            )

    def _check_onnx_availability(self) -> EngineStatus:
        """
        Check if ONNX Runtime is available and functional.

        Returns:
            EngineStatus with availability information
        """
        if not self.config.onnx.enabled:
            return EngineStatus(
                name=EngineType.ONNX.value,
                available=False,
                models_loaded=[],
                error_message="ONNX Runtime disabled in configuration",
                last_check=datetime.now(),
            )

        try:
            # Try to import ONNX Runtime
            import onnxruntime  # noqa: F401

            return EngineStatus(
                name=EngineType.ONNX.value,
                available=True,
                models_loaded=[],
                error_message=None,
                last_check=datetime.now(),
            )
        except ImportError as e:
            return EngineStatus(
                name=EngineType.ONNX.value,
                available=False,
                models_loaded=[],
                error_message=f"ONNX Runtime not installed: {e}",
                last_check=datetime.now(),
            )
        except Exception as e:
            return EngineStatus(
                name=EngineType.ONNX.value,
                available=False,
                models_loaded=[],
                error_message=f"ONNX Runtime initialization error: {e}",
                last_check=datetime.now(),
            )

    def get_available_engines(self, model_name: Optional[str] = None) -> List[str]:
        """
        Get list of available engines for a model.

        Args:
            model_name: Optional model name to check compatibility (not used yet)

        Returns:
            List of available engine names in preference order

        Example:
            >>> manager.get_available_engines()
            ['vllm', 'pytorch']
        """
        available = []
        for engine_name in self._engine_preference:
            status = self._engine_registry.get(engine_name)
            if status and status.available:
                available.append(engine_name)

        return available

    def get_engine_status(self, engine_name: Optional[str] = None) -> Dict[str, EngineStatus]:
        """
        Get status information for engines.

        Args:
            engine_name: Optional specific engine name, or None for all engines

        Returns:
            Dictionary mapping engine names to their status

        Example:
            >>> status = manager.get_engine_status("vllm")
            >>> print(status["vllm"].available)
            True
        """
        if engine_name:
            status = self._engine_registry.get(engine_name)
            if status:
                return {engine_name: status}
            else:
                return {}

        # Return all engine statuses
        return self._engine_registry.copy()

    def infer(
        self, model_name: str, inputs: Dict[str, Any], engine_preference: Optional[str] = None
    ) -> InferenceResult:
        """
        Execute inference with optimal engine and fallback support.

        Attempts inference with the preferred engine, falling back to the next
        available engine in the fallback chain if the preferred engine fails.

        Fallback chain: vLLM → DeepSpeed → ONNX → PyTorch

        Args:
            model_name: Name of the model to use for inference
            inputs: Input data for inference (format depends on engine)
            engine_preference: Optional engine hint to try first

        Returns:
            InferenceResult with outputs and metadata

        Raises:
            RuntimeError: If all engines fail (including PyTorch fallback)

        Example:
            >>> result = manager.infer(
            ...     "qwen-chat",
            ...     {"input": "Hello, world!"},
            ...     engine_preference="vllm"
            ... )
            >>> print(result.engine_used)
            'vllm'
        """
        # Create tracing span for inference operation
        # Requirement 15.4: Handle monitoring failures gracefully
        span = None
        try:
            if self._tracer:
                span = self._tracer.trace_inference(
                    model_name=model_name, engine=engine_preference or "auto"
                ).__enter__()
        except Exception as e:
            # Monitoring failure should not block request
            logger.warning(f"Failed to create tracing span: {e}")
            span = None

        try:
            # Determine engine order to try
            if engine_preference and engine_preference in self._engine_preference:
                # Try preferred engine first, then fallback chain
                engines_to_try = [engine_preference] + [
                    e for e in self._engine_preference if e != engine_preference
                ]
            else:
                # Use default preference order
                engines_to_try = self._engine_preference.copy()

            # Filter to only available engines
            available_engines = [
                engine
                for engine in engines_to_try
                if self._engine_registry.get(engine, EngineStatus(engine, False, [])).available
            ]

            if not available_engines:
                error_msg = (
                    f"No available engines for model {model_name}. "
                    f"All engines unavailable or disabled."
                )
                # Record error in monitoring
                self._record_monitoring_error(model_name, "no_engines_available")
                raise RuntimeError(error_msg)

            # Try each engine in order
            last_error = None
            for engine_name in available_engines:
                try:
                    logger.debug(f"Attempting inference with engine: {engine_name}")
                    result = self._infer_with_engine(engine_name, model_name, inputs)

                    logger.info(
                        f"Inference successful with {engine_name} "
                        f"(latency: {result.latency_ms:.2f}ms)"
                    )

                    # Record successful inference metrics
                    # Requirement 15.4: Handle monitoring failures gracefully
                    self._record_inference_metrics(result)

                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(f"Inference failed with {engine_name}: {e}", exc_info=True)

                    # Record error in monitoring
                    self._record_monitoring_error(model_name, engine_name, str(e))

                    # Update engine status if it's a critical failure
                    if self._is_critical_error(e):
                        self._mark_engine_unavailable(engine_name, str(e))

                    # Continue to next engine if fallback is enabled
                    if not self.config.fallback_on_error:
                        raise

            # All engines failed
            error_msg = f"All engines failed for model {model_name}. " f"Last error: {last_error}"
            raise RuntimeError(error_msg)

        except Exception as e:
            # Record error in span if available
            if span and self._tracer:
                try:
                    self._tracer._record_error_in_span(span, e)
                except Exception as trace_error:
                    logger.warning(f"Failed to record error in span: {trace_error}")
            raise
        finally:
            # Close span if it was created
            if span:
                try:
                    span.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Failed to close tracing span: {e}")

    def _infer_with_engine(
        self, engine_name: str, model_name: str, inputs: Dict[str, Any]
    ) -> InferenceResult:
        """
        Execute inference with a specific engine.

        Args:
            engine_name: Name of the engine to use
            model_name: Name of the model
            inputs: Input data for inference

        Returns:
            InferenceResult with outputs and metadata

        Raises:
            NotImplementedError: Engine-specific inference not yet implemented
            RuntimeError: Engine execution failed
        """
        start_time = time.time()

        # Route to appropriate engine implementation
        if engine_name == EngineType.VLLM.value:
            outputs = self._infer_vllm(model_name, inputs)
        elif engine_name == EngineType.DEEPSPEED.value:
            outputs = self._infer_deepspeed(model_name, inputs)
        elif engine_name == EngineType.ONNX.value:
            outputs = self._infer_onnx(model_name, inputs)
        elif engine_name == EngineType.PYTORCH.value:
            outputs = self._infer_pytorch(model_name, inputs)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResult(
            outputs=outputs,
            engine_used=engine_name,
            latency_ms=latency_ms,
            batch_size=1,  # Will be updated when batching is implemented
            cache_hit=False,  # Will be updated when caching is implemented
            metadata={"model_name": model_name, "timestamp": datetime.now().isoformat()},
        )

    def _infer_vllm(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference with vLLM engine.

        Args:
            model_name: Name of the model
            inputs: Input data for inference (must contain 'prompts' key)

        Returns:
            Dictionary of output data

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If vLLM inference fails
        """
        # Lazy initialization of vLLM engine
        if self._vllm_engine is None:
            from mm_orch.optimization.vllm_engine import VLLMEngine

            self._vllm_engine = VLLMEngine(self.config.vllm)

        # Validate inputs
        if "prompts" not in inputs:
            raise ValueError("vLLM inference requires 'prompts' key in inputs")

        prompts = inputs["prompts"]
        if not isinstance(prompts, list):
            prompts = [prompts]

        # Load model if not already loaded
        if self._vllm_engine.get_loaded_model() != model_name:
            success = self._vllm_engine.load_model(model_name)
            if not success:
                raise RuntimeError(f"Failed to load model {model_name} with vLLM")

            # Update registry
            if model_name not in self._engine_registry[EngineType.VLLM.value].models_loaded:
                self._engine_registry[EngineType.VLLM.value].models_loaded.append(model_name)

        # Get sampling parameters if provided
        sampling_params = inputs.get("sampling_params", None)

        # Generate with vLLM
        outputs = self._vllm_engine.generate(prompts, sampling_params)

        # Return in standard format
        return {"outputs": outputs, "num_outputs": len(outputs)}

    def _infer_deepspeed(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference with DeepSpeed engine.

        Args:
            model_name: Name of the model
            inputs: Input data for inference

        Returns:
            Dictionary of output data

        Raises:
            NotImplementedError: DeepSpeed inference not yet implemented
        """
        # TODO: Implement DeepSpeed inference in task 4.2
        raise NotImplementedError("DeepSpeed inference not yet implemented")

    def _infer_onnx(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference with ONNX Runtime engine.

        Args:
            model_name: Name of the model
            inputs: Input data for inference

        Returns:
            Dictionary of output data

        Raises:
            NotImplementedError: ONNX inference not yet implemented
        """
        # TODO: Implement ONNX inference in task 5.2
        raise NotImplementedError("ONNX inference not yet implemented")

    def _infer_pytorch(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference with PyTorch (fallback engine).

        Args:
            model_name: Name of the model
            inputs: Input data for inference

        Returns:
            Dictionary of output data

        Raises:
            NotImplementedError: PyTorch inference integration not yet implemented
        """
        # TODO: Integrate with existing ModelManager in task 19.1
        raise NotImplementedError("PyTorch inference integration not yet implemented")

    def _is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error is critical enough to mark engine as unavailable.

        Args:
            error: Exception that occurred during inference

        Returns:
            True if error is critical, False otherwise
        """
        # Critical errors that indicate engine is broken
        critical_error_types = (
            ImportError,
            RuntimeError,
            MemoryError,
        )

        return isinstance(error, critical_error_types)

    def _mark_engine_unavailable(self, engine_name: str, error_message: str):
        """
        Mark an engine as unavailable due to critical failure.

        Args:
            engine_name: Name of the engine to mark unavailable
            error_message: Error message describing the failure
        """
        if engine_name in self._engine_registry:
            status = self._engine_registry[engine_name]
            status.available = False
            status.error_message = error_message
            status.last_check = datetime.now()

            logger.error(f"Engine {engine_name} marked as unavailable: {error_message}")

    def _record_inference_metrics(self, result: InferenceResult):
        """
        Record inference metrics to Prometheus.

        Handles monitoring failures gracefully per Requirement 15.4.

        Args:
            result: InferenceResult containing metrics to record
        """
        if not self._prometheus:
            return

        try:
            # Record inference latency
            self._prometheus.record_inference_latency(
                model_name=result.metadata.get("model_name", "unknown"),
                engine=result.engine_used,
                latency_ms=result.latency_ms,
            )

            # Record batch size if available
            if result.batch_size > 0:
                self._prometheus.record_batch_size(
                    model_name=result.metadata.get("model_name", "unknown"),
                    batch_size=result.batch_size,
                )

            # Record cache hit rate if available
            if result.cache_hit:
                # Cache hit rate tracking would be done by KV cache manager
                pass

        except Exception as e:
            # Monitoring failure should not block request
            # Requirement 15.4: Handle monitoring failures gracefully
            logger.warning(f"Failed to record inference metrics: {e}")

    def _record_monitoring_error(
        self, model_name: str, engine_or_error: str, error_message: Optional[str] = None
    ):
        """
        Record monitoring error to Prometheus.

        Handles monitoring failures gracefully per Requirement 15.4.

        Args:
            model_name: Name of the model
            engine_or_error: Engine name or error type
            error_message: Optional error message
        """
        if not self._prometheus:
            return

        try:
            if error_message:
                # This is an inference error
                self._prometheus.record_inference_error(
                    model_name=model_name,
                    engine=engine_or_error,
                    error_type=(
                        type(error_message).__name__
                        if hasattr(error_message, "__class__")
                        else "unknown"
                    ),
                )
            else:
                # This is a general error
                self._prometheus.record_error(
                    component="optimization_manager", error_type=engine_or_error
                )
        except Exception as e:
            # Monitoring failure should not block request
            # Requirement 15.5: Tracing failures don't block requests
            logger.warning(f"Failed to record monitoring error: {e}")
