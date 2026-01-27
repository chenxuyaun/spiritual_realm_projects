"""
Optimization module for advanced inference engines.

This module provides integration with high-performance inference engines
including vLLM, DeepSpeed, and ONNX Runtime, with graceful fallback to
standard PyTorch inference.
"""

from mm_orch.optimization.config import (
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig,
    BatcherConfig,
    CacheConfig,
    TunerConfig,
    ServerConfig,
    load_optimization_config,
)
from mm_orch.optimization.manager import (
    OptimizationManager,
    EngineStatus,
    EngineType,
    InferenceResult,
)
from mm_orch.optimization.vllm_engine import VLLMEngine
from mm_orch.optimization.deepspeed_engine import DeepSpeedEngine
from mm_orch.optimization.onnx_engine import ONNXEngine
from mm_orch.optimization.batcher import (
    DynamicBatcher,
    InferenceRequest,
    BatchedRequest,
)
from mm_orch.optimization.kv_cache_manager import (
    KVCacheManager,
    KVCache,
    CacheStats,
)
from mm_orch.optimization.server import (
    InferenceServer,
    ServerStatus,
    HealthStatus,
    ReadinessStatus,
)

__all__ = [
    # Configuration
    "OptimizationConfig",
    "VLLMConfig",
    "DeepSpeedConfig",
    "ONNXConfig",
    "BatcherConfig",
    "CacheConfig",
    "TunerConfig",
    "ServerConfig",
    "load_optimization_config",
    # Manager
    "OptimizationManager",
    "EngineStatus",
    "EngineType",
    "InferenceResult",
    # Engines
    "VLLMEngine",
    "DeepSpeedEngine",
    "ONNXEngine",
    # Batcher
    "DynamicBatcher",
    "InferenceRequest",
    "BatchedRequest",
    # KV Cache
    "KVCacheManager",
    "KVCache",
    "CacheStats",
    # Server
    "InferenceServer",
    "ServerStatus",
    "HealthStatus",
    "ReadinessStatus",
]
