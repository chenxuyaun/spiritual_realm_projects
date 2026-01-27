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

__all__ = [
    # Configuration
    "OptimizationConfig",
    "VLLMConfig",
    "DeepSpeedConfig",
    "ONNXConfig",
    "BatcherConfig",
    "CacheConfig",
    "TunerConfig",
    "load_optimization_config",
    # Manager
    "OptimizationManager",
    "EngineStatus",
    "EngineType",
    "InferenceResult",
    # Engines
    "VLLMEngine",
    "DeepSpeedEngine",
]
