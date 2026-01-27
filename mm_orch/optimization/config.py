"""
Configuration data models for optimization features.

This module defines configuration dataclasses for all optimization engines
(vLLM, DeepSpeed, ONNX Runtime), dynamic batching, KV caching, and auto-tuning.
Supports loading from YAML files with environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from mm_orch.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VLLMConfig:
    """
    Configuration for vLLM inference engine.
    
    Attributes:
        enabled: Whether vLLM is enabled
        tensor_parallel_size: Number of GPUs for tensor parallelism
        dtype: Data type for model weights ('auto', 'fp16', 'fp32', 'bf16')
        max_model_len: Maximum sequence length (None for model default)
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        swap_space: CPU swap space in GB for offloading
    """
    enabled: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        
        valid_dtypes = {"auto", "fp16", "fp32", "bf16"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")
        
        if not 0.0 <= self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")
        
        if self.swap_space < 0:
            raise ValueError("swap_space must be >= 0")
        
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1 or None")


@dataclass
class DeepSpeedConfig:
    """
    Configuration for DeepSpeed inference engine.
    
    Attributes:
        enabled: Whether DeepSpeed is enabled
        tensor_parallel: Number of GPUs for tensor parallelism
        pipeline_parallel: Number of GPUs for pipeline parallelism
        dtype: Data type for model weights ('fp16', 'fp32', 'bf16')
        replace_with_kernel_inject: Use DeepSpeed kernel injection
    """
    enabled: bool = True
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    dtype: str = "fp16"
    replace_with_kernel_inject: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tensor_parallel < 1:
            raise ValueError("tensor_parallel must be >= 1")
        
        if self.pipeline_parallel < 1:
            raise ValueError("pipeline_parallel must be >= 1")
        
        valid_dtypes = {"fp16", "fp32", "bf16"}
        if self.dtype not in valid_dtypes:
            raise ValueError(f"dtype must be one of {valid_dtypes}")


@dataclass
class ONNXConfig:
    """
    Configuration for ONNX Runtime inference engine.
    
    Attributes:
        enabled: Whether ONNX Runtime is enabled
        execution_providers: List of execution providers in priority order
        optimization_level: ONNX optimization level ('none', 'basic', 'extended', 'all')
        enable_quantization: Enable dynamic quantization during conversion
    """
    enabled: bool = True
    execution_providers: List[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    optimization_level: str = "all"
    enable_quantization: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_opt_levels = {"none", "basic", "extended", "all"}
        if self.optimization_level not in valid_opt_levels:
            raise ValueError(f"optimization_level must be one of {valid_opt_levels}")
        
        valid_providers = {
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
            "OpenVINOExecutionProvider",
        }
        for provider in self.execution_providers:
            if provider not in valid_providers:
                logger.warning(f"Unknown execution provider: {provider}")


@dataclass
class BatcherConfig:
    """
    Configuration for dynamic batching.
    
    Attributes:
        enabled: Whether dynamic batching is enabled
        max_batch_size: Maximum number of requests per batch
        batch_timeout_ms: Maximum wait time for batch formation (milliseconds)
        adaptive_batching: Enable dynamic batch size adjustment
        min_batch_size: Minimum batch size for processing
    """
    enabled: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    adaptive_batching: bool = True
    min_batch_size: int = 1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        
        if self.batch_timeout_ms < 0:
            raise ValueError("batch_timeout_ms must be >= 0")
        
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        
        if self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size must be <= max_batch_size")


@dataclass
class CacheConfig:
    """
    Configuration for KV cache management.
    
    Attributes:
        enabled: Whether KV caching is enabled
        max_memory_mb: Maximum cache memory in megabytes
        eviction_policy: Cache eviction policy ('lru', 'fifo')
    """
    enabled: bool = True
    max_memory_mb: int = 4096
    eviction_policy: str = "lru"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_memory_mb < 0:
            raise ValueError("max_memory_mb must be >= 0")
        
        valid_policies = {"lru", "fifo"}
        if self.eviction_policy not in valid_policies:
            raise ValueError(f"eviction_policy must be one of {valid_policies}")


@dataclass
class TunerConfig:
    """
    Configuration for auto-tuning.
    
    Attributes:
        enabled: Whether auto-tuning is enabled
        observation_window_seconds: Time window for performance observation
        tuning_interval_seconds: Interval between tuning adjustments
        enable_batch_size_tuning: Enable batch size auto-tuning
        enable_timeout_tuning: Enable timeout auto-tuning
        enable_cache_size_tuning: Enable cache size auto-tuning
    """
    enabled: bool = False
    observation_window_seconds: int = 300
    tuning_interval_seconds: int = 60
    enable_batch_size_tuning: bool = True
    enable_timeout_tuning: bool = True
    enable_cache_size_tuning: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.observation_window_seconds < 1:
            raise ValueError("observation_window_seconds must be >= 1")
        
        if self.tuning_interval_seconds < 1:
            raise ValueError("tuning_interval_seconds must be >= 1")


@dataclass
class OptimizationConfig:
    """
    Top-level configuration for optimization features.
    
    Attributes:
        enabled: Whether optimization features are enabled globally
        vllm: vLLM engine configuration
        deepspeed: DeepSpeed engine configuration
        onnx: ONNX Runtime engine configuration
        batcher: Dynamic batching configuration
        cache: KV cache configuration
        tuner: Auto-tuning configuration
        engine_preference: Ordered list of preferred engines
        fallback_on_error: Enable fallback to next engine on error
    """
    enabled: bool = True
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    onnx: ONNXConfig = field(default_factory=ONNXConfig)
    batcher: BatcherConfig = field(default_factory=BatcherConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)
    engine_preference: List[str] = field(
        default_factory=lambda: ["vllm", "deepspeed", "onnx", "pytorch"]
    )
    fallback_on_error: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        valid_engines = {"vllm", "deepspeed", "onnx", "pytorch"}
        for engine in self.engine_preference:
            if engine not in valid_engines:
                raise ValueError(f"Invalid engine in preference list: {engine}")
        
        if "pytorch" not in self.engine_preference:
            logger.warning("pytorch not in engine_preference, adding as fallback")
            self.engine_preference.append("pytorch")


def load_optimization_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> OptimizationConfig:
    """
    Load optimization configuration from YAML file or dictionary.
    
    Supports environment variable overrides with MUAI_OPT_ prefix.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary (overrides file)
    
    Returns:
        OptimizationConfig instance
    
    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If configuration is invalid
    
    Environment Variables:
        MUAI_OPT_ENABLED: Override optimization.enabled
        MUAI_OPT_VLLM_ENABLED: Override optimization.vllm.enabled
        MUAI_OPT_VLLM_TENSOR_PARALLEL: Override optimization.vllm.tensor_parallel_size
        MUAI_OPT_DEEPSPEED_ENABLED: Override optimization.deepspeed.enabled
        MUAI_OPT_ONNX_ENABLED: Override optimization.onnx.enabled
        MUAI_OPT_BATCHER_ENABLED: Override optimization.batcher.enabled
        MUAI_OPT_BATCHER_MAX_SIZE: Override optimization.batcher.max_batch_size
        MUAI_OPT_CACHE_ENABLED: Override optimization.cache.enabled
        MUAI_OPT_TUNER_ENABLED: Override optimization.tuner.enabled
    
    Example:
        >>> config = load_optimization_config("config/optimization.yaml")
        >>> config = load_optimization_config(config_dict={"enabled": True})
    """
    # Load from file if provided
    if config_path:
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path_obj, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        
        # Extract optimization section
        opt_config = file_config.get("optimization", {})
    elif config_dict:
        opt_config = config_dict
    else:
        # Use defaults
        opt_config = {}
    
    # Apply environment variable overrides
    opt_config = _apply_env_overrides(opt_config)
    
    # Build configuration objects
    try:
        vllm_config = VLLMConfig(**opt_config.get("vllm", {}))
        deepspeed_config = DeepSpeedConfig(**opt_config.get("deepspeed", {}))
        onnx_config = ONNXConfig(**opt_config.get("onnx", {}))
        batcher_config = BatcherConfig(**opt_config.get("batcher", {}))
        cache_config = CacheConfig(**opt_config.get("cache", {}))
        tuner_config = TunerConfig(**opt_config.get("tuner", {}))
        
        # Build top-level config
        config = OptimizationConfig(
            enabled=opt_config.get("enabled", True),
            vllm=vllm_config,
            deepspeed=deepspeed_config,
            onnx=onnx_config,
            batcher=batcher_config,
            cache=cache_config,
            tuner=tuner_config,
            engine_preference=opt_config.get(
                "engine_preference", ["vllm", "deepspeed", "onnx", "pytorch"]
            ),
            fallback_on_error=opt_config.get("fallback_on_error", True),
        )
        
        logger.info("Optimization configuration loaded successfully")
        return config
        
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid optimization configuration: {e}")
        raise ValueError(f"Configuration validation failed: {e}") from e


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Updated configuration dictionary
    """
    # Top-level overrides
    if "MUAI_OPT_ENABLED" in os.environ:
        config["enabled"] = os.environ["MUAI_OPT_ENABLED"].lower() in ("true", "1", "yes")
    
    # vLLM overrides
    vllm_config = config.setdefault("vllm", {})
    if "MUAI_OPT_VLLM_ENABLED" in os.environ:
        vllm_config["enabled"] = os.environ["MUAI_OPT_VLLM_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_OPT_VLLM_TENSOR_PARALLEL" in os.environ:
        vllm_config["tensor_parallel_size"] = int(os.environ["MUAI_OPT_VLLM_TENSOR_PARALLEL"])
    if "MUAI_OPT_VLLM_DTYPE" in os.environ:
        vllm_config["dtype"] = os.environ["MUAI_OPT_VLLM_DTYPE"]
    if "MUAI_OPT_VLLM_GPU_MEMORY" in os.environ:
        vllm_config["gpu_memory_utilization"] = float(os.environ["MUAI_OPT_VLLM_GPU_MEMORY"])
    
    # DeepSpeed overrides
    deepspeed_config = config.setdefault("deepspeed", {})
    if "MUAI_OPT_DEEPSPEED_ENABLED" in os.environ:
        deepspeed_config["enabled"] = os.environ["MUAI_OPT_DEEPSPEED_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_OPT_DEEPSPEED_TENSOR_PARALLEL" in os.environ:
        deepspeed_config["tensor_parallel"] = int(os.environ["MUAI_OPT_DEEPSPEED_TENSOR_PARALLEL"])
    if "MUAI_OPT_DEEPSPEED_PIPELINE_PARALLEL" in os.environ:
        deepspeed_config["pipeline_parallel"] = int(os.environ["MUAI_OPT_DEEPSPEED_PIPELINE_PARALLEL"])
    
    # ONNX overrides
    onnx_config = config.setdefault("onnx", {})
    if "MUAI_OPT_ONNX_ENABLED" in os.environ:
        onnx_config["enabled"] = os.environ["MUAI_OPT_ONNX_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_OPT_ONNX_OPTIMIZATION_LEVEL" in os.environ:
        onnx_config["optimization_level"] = os.environ["MUAI_OPT_ONNX_OPTIMIZATION_LEVEL"]
    
    # Batcher overrides
    batcher_config = config.setdefault("batcher", {})
    if "MUAI_OPT_BATCHER_ENABLED" in os.environ:
        batcher_config["enabled"] = os.environ["MUAI_OPT_BATCHER_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_OPT_BATCHER_MAX_SIZE" in os.environ:
        batcher_config["max_batch_size"] = int(os.environ["MUAI_OPT_BATCHER_MAX_SIZE"])
    if "MUAI_OPT_BATCHER_TIMEOUT" in os.environ:
        batcher_config["batch_timeout_ms"] = int(os.environ["MUAI_OPT_BATCHER_TIMEOUT"])
    
    # Cache overrides
    cache_config = config.setdefault("cache", {})
    if "MUAI_OPT_CACHE_ENABLED" in os.environ:
        cache_config["enabled"] = os.environ["MUAI_OPT_CACHE_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_OPT_CACHE_MAX_MEMORY" in os.environ:
        cache_config["max_memory_mb"] = int(os.environ["MUAI_OPT_CACHE_MAX_MEMORY"])
    
    # Tuner overrides
    tuner_config = config.setdefault("tuner", {})
    if "MUAI_OPT_TUNER_ENABLED" in os.environ:
        tuner_config["enabled"] = os.environ["MUAI_OPT_TUNER_ENABLED"].lower() in ("true", "1", "yes")
    
    return config
