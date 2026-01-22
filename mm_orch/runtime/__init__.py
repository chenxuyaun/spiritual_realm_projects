"""Runtime management for model loading, caching, and vector database."""

from mm_orch.runtime.model_manager import (
    ModelManager,
    CachedModel,
    get_model_manager,
    configure_model_manager,
)

from mm_orch.runtime.vector_db import (
    VectorDBManager,
    IndexMetadata,
    get_vector_db,
    configure_vector_db,
)

# Real Model Integration (Phase 1)
from mm_orch.runtime.quantization import (
    QuantizationManager,
    QuantizationConfig,
)

from mm_orch.runtime.model_loader import (
    ModelLoader,
    ModelConfig,
    LoadedModel,
)

from mm_orch.runtime.memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    MemoryAlert,
)

from mm_orch.runtime.real_model_manager import (
    RealModelManager,
    CachedModel as RealCachedModel,
)

from mm_orch.runtime.flash_attention import (
    FlashAttentionManager,
    FlashAttentionInfo,
    is_flash_attention_available,
    get_flash_attention_info,
    get_best_attention_implementation,
)

from mm_orch.runtime.inference_engine import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)

from mm_orch.runtime.conversation import (
    ConversationManager,
    Conversation,
    Message,
)

__all__ = [
    # Model Manager (Mock)
    "ModelManager",
    "CachedModel",
    "get_model_manager",
    "configure_model_manager",
    # Vector DB
    "VectorDBManager",
    "IndexMetadata",
    "get_vector_db",
    "configure_vector_db",
    # Quantization
    "QuantizationManager",
    "QuantizationConfig",
    # Model Loader
    "ModelLoader",
    "ModelConfig",
    "LoadedModel",
    # Memory Monitor
    "MemoryMonitor",
    "MemorySnapshot",
    "MemoryAlert",
    # Real Model Manager
    "RealModelManager",
    "RealCachedModel",
    # Flash Attention
    "FlashAttentionManager",
    "FlashAttentionInfo",
    "is_flash_attention_available",
    "get_flash_attention_info",
    "get_best_attention_implementation",
    # Inference Engine
    "InferenceEngine",
    "GenerationConfig",
    "GenerationResult",
    # Conversation
    "ConversationManager",
    "Conversation",
    "Message",
]
