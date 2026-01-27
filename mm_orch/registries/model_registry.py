"""
Model Registry for managing ML models.

This module provides a registry for discovering and managing ML models
with metadata about their capabilities, resource requirements, and
device policies. Models are registered with detailed metadata to enable
intelligent loading and scheduling decisions.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.

    Attributes:
        name: Unique identifier for the model
        capabilities: List of capability tags (e.g., ["summarize", "generate", "qa", "embed"])
        expected_vram_mb: Expected VRAM usage in megabytes
        supports_quant: Whether the model supports quantization (8-bit/4-bit)
        preferred_device_policy: Device loading policy ("gpu_on_demand", "cpu_only", "gpu_resident")
        model_path: Path to model files or HuggingFace model identifier
        quantization_config: Optional configuration for quantization
    """

    name: str
    capabilities: List[str]
    expected_vram_mb: int
    supports_quant: bool
    preferred_device_policy: str
    model_path: str
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        # Validate required fields
        required_fields = {
            "name": self.name,
            "capabilities": self.capabilities,
            "expected_vram_mb": self.expected_vram_mb,
            "supports_quant": self.supports_quant,
            "preferred_device_policy": self.preferred_device_policy,
            "model_path": self.model_path,
        }

        for field_name, field_value in required_fields.items():
            if field_value is None or (isinstance(field_value, str) and not field_value):
                raise ValueError(f"Missing required field: {field_name}")

        # Validate capabilities is not empty
        if not self.capabilities:
            raise ValueError(f"Model '{self.name}' must have at least one capability")

        # Validate expected_vram_mb is positive
        if self.expected_vram_mb < 0:
            raise ValueError(
                f"Model '{self.name}' expected_vram_mb must be non-negative, got {self.expected_vram_mb}"
            )

        # Validate preferred_device_policy
        valid_policies = ["gpu_on_demand", "cpu_only", "gpu_resident"]
        if self.preferred_device_policy not in valid_policies:
            raise ValueError(
                f"Model '{self.name}' preferred_device_policy must be one of {valid_policies}, "
                f"got '{self.preferred_device_policy}'"
            )


class ModelRegistry:
    """
    Registry for ML models.

    Provides centralized registration and discovery of ML models
    with metadata about their capabilities, resource requirements,
    and device policies.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register(
        ...     ModelMetadata(
        ...         name="t5-small",
        ...         capabilities=["summarize"],
        ...         expected_vram_mb=1200,
        ...         supports_quant=True,
        ...         preferred_device_policy="gpu_on_demand",
        ...         model_path="t5-small"
        ...     )
        ... )
        >>> metadata = registry.get("t5-small")
        >>> summarizers = registry.find_by_capability("summarize")
    """

    def __init__(self):
        """Initialize an empty model registry."""
        self._models: Dict[str, ModelMetadata] = {}
        logger.debug("Model registry initialized")

    def register(self, metadata: ModelMetadata) -> None:
        """
        Register a model with metadata.

        Args:
            metadata: ModelMetadata describing the model

        Raises:
            ValueError: If metadata validation fails or model is already registered
        """
        # Metadata validation happens in __post_init__
        name = metadata.name

        if name in self._models:
            logger.warning(f"Model '{name}' is already registered, overwriting")

        self._models[name] = metadata

        logger.info(
            f"Registered model '{name}'",
            capabilities=metadata.capabilities,
            expected_vram_mb=metadata.expected_vram_mb,
            device_policy=metadata.preferred_device_policy,
        )

    def get(self, name: str) -> ModelMetadata:
        """
        Retrieve model metadata by name.

        Args:
            name: Model identifier

        Returns:
            ModelMetadata for the model

        Raises:
            KeyError: If model is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")

        return self._models[name]

    def find_by_capability(self, capability: str) -> List[ModelMetadata]:
        """
        Find models with a specific capability.

        Args:
            capability: Capability tag to search for

        Returns:
            List of ModelMetadata for models that have the specified capability
        """
        matching_models = [
            meta for meta in self._models.values() if capability in meta.capabilities
        ]

        logger.debug(f"Found {len(matching_models)} models with capability '{capability}'")

        return matching_models

    def has(self, name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            name: Model identifier

        Returns:
            True if model is registered, False otherwise
        """
        return name in self._models

    def list_all(self) -> List[str]:
        """
        List all registered model names.

        Returns:
            List of all registered model names
        """
        return list(self._models.keys())

    def unregister(self, name: str) -> None:
        """
        Unregister a model.

        Args:
            name: Model identifier

        Raises:
            KeyError: If model is not registered
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")

        del self._models[name]

        logger.info(f"Unregistered model '{name}'")


# Global registry instance
_global_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        Global ModelRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
        _register_default_models(_global_registry)
    return _global_registry


def _register_default_models(registry: ModelRegistry) -> None:
    """
    Register default models in the registry.

    Args:
        registry: ModelRegistry to register models in
    """
    # Register T5-small for summarization
    registry.register(
        ModelMetadata(
            name="t5-small",
            capabilities=["summarize", "text2text"],
            expected_vram_mb=1200,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="t5-small",
        )
    )

    # Register BART for summarization
    registry.register(
        ModelMetadata(
            name="facebook/bart-large-cnn",
            capabilities=["summarize", "text2text"],
            expected_vram_mb=1600,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="facebook/bart-large-cnn",
        )
    )

    # Register GPT-2 for generation
    registry.register(
        ModelMetadata(
            name="gpt2",
            capabilities=["generate", "text_generation"],
            expected_vram_mb=500,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="gpt2",
        )
    )

    # Register DistilGPT-2 for generation (smaller)
    registry.register(
        ModelMetadata(
            name="distilgpt2",
            capabilities=["generate", "text_generation"],
            expected_vram_mb=350,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="distilgpt2",
        )
    )

    # Register Qwen Chat for generation
    registry.register(
        ModelMetadata(
            name="Qwen/Qwen-1_8B-Chat",
            capabilities=["generate", "chat", "text_generation"],
            expected_vram_mb=3600,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="Qwen/Qwen-1_8B-Chat",
            quantization_config={
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            },
        )
    )

    # Register MiniLM for embeddings
    registry.register(
        ModelMetadata(
            name="sentence-transformers/all-MiniLM-L6-v2",
            capabilities=["embed", "sentence_embedding"],
            expected_vram_mb=400,
            supports_quant=False,
            preferred_device_policy="gpu_resident",
            model_path="sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    logger.info(
        "Registered default models: t5-small, bart-large-cnn, gpt2, distilgpt2, Qwen-1_8B-Chat, MiniLM"
    )
