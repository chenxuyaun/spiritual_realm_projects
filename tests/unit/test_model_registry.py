"""
Unit tests for Model Registry.

Tests the registration, retrieval, and querying of ML models
with metadata about capabilities and resource requirements.
"""

import pytest
from mm_orch.registries.model_registry import (
    ModelRegistry,
    ModelMetadata,
    get_model_registry,
)


class TestModelMetadata:
    """Test ModelMetadata dataclass validation."""

    def test_valid_metadata(self):
        """Test creating valid model metadata."""
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["summarize"],
            expected_vram_mb=1000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model",
        )

        assert metadata.name == "test-model"
        assert metadata.capabilities == ["summarize"]
        assert metadata.expected_vram_mb == 1000
        assert metadata.supports_quant is True
        assert metadata.preferred_device_policy == "gpu_on_demand"
        assert metadata.model_path == "test/model"
        assert metadata.quantization_config is None

    def test_metadata_with_quantization_config(self):
        """Test metadata with quantization configuration."""
        quant_config = {"load_in_8bit": True, "llm_int8_threshold": 6.0}
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["generate"],
            expected_vram_mb=2000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model",
            quantization_config=quant_config,
        )

        assert metadata.quantization_config == quant_config

    def test_missing_name(self):
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError, match="Missing required field: name"):
            ModelMetadata(
                name="",
                capabilities=["summarize"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="test/model",
            )

    def test_missing_capabilities(self):
        """Test that empty capabilities raises ValueError."""
        with pytest.raises(ValueError, match="must have at least one capability"):
            ModelMetadata(
                name="test-model",
                capabilities=[],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="test/model",
            )

    def test_missing_model_path(self):
        """Test that missing model_path raises ValueError."""
        with pytest.raises(ValueError, match="Missing required field: model_path"):
            ModelMetadata(
                name="test-model",
                capabilities=["summarize"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="",
            )

    def test_negative_vram(self):
        """Test that negative VRAM raises ValueError."""
        with pytest.raises(ValueError, match="expected_vram_mb must be non-negative"):
            ModelMetadata(
                name="test-model",
                capabilities=["summarize"],
                expected_vram_mb=-100,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="test/model",
            )

    def test_invalid_device_policy(self):
        """Test that invalid device policy raises ValueError."""
        with pytest.raises(ValueError, match="preferred_device_policy must be one of"):
            ModelMetadata(
                name="test-model",
                capabilities=["summarize"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="invalid_policy",
                model_path="test/model",
            )

    def test_valid_device_policies(self):
        """Test all valid device policies."""
        valid_policies = ["gpu_on_demand", "cpu_only", "gpu_resident"]
        
        for policy in valid_policies:
            metadata = ModelMetadata(
                name=f"test-model-{policy}",
                capabilities=["summarize"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy=policy,
                model_path="test/model",
            )
            assert metadata.preferred_device_policy == policy



class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_register_and_get(self):
        """Test registering and retrieving a model."""
        registry = ModelRegistry()
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["summarize"],
            expected_vram_mb=1000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model",
        )

        registry.register(metadata)
        retrieved = registry.get("test-model")

        assert retrieved.name == "test-model"
        assert retrieved.capabilities == ["summarize"]
        assert retrieved.expected_vram_mb == 1000

    def test_get_nonexistent_model(self):
        """Test that getting nonexistent model raises KeyError."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="is not registered"):
            registry.get("nonexistent-model")

    def test_register_duplicate_model(self):
        """Test that registering duplicate model overwrites with warning."""
        registry = ModelRegistry()
        metadata1 = ModelMetadata(
            name="test-model",
            capabilities=["summarize"],
            expected_vram_mb=1000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model/v1",
        )
        metadata2 = ModelMetadata(
            name="test-model",
            capabilities=["generate"],
            expected_vram_mb=2000,
            supports_quant=False,
            preferred_device_policy="cpu_only",
            model_path="test/model/v2",
        )

        registry.register(metadata1)
        registry.register(metadata2)

        # Should have the second registration
        retrieved = registry.get("test-model")
        assert retrieved.capabilities == ["generate"]
        assert retrieved.expected_vram_mb == 2000
        assert retrieved.model_path == "test/model/v2"

    def test_find_by_capability(self):
        """Test finding models by capability."""
        registry = ModelRegistry()
        
        # Register models with different capabilities
        registry.register(
            ModelMetadata(
                name="summarizer-1",
                capabilities=["summarize", "text2text"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="summarizer-1",
            )
        )
        registry.register(
            ModelMetadata(
                name="summarizer-2",
                capabilities=["summarize"],
                expected_vram_mb=1200,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="summarizer-2",
            )
        )
        registry.register(
            ModelMetadata(
                name="generator-1",
                capabilities=["generate", "text_generation"],
                expected_vram_mb=500,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="generator-1",
            )
        )

        # Find summarizers
        summarizers = registry.find_by_capability("summarize")
        assert len(summarizers) == 2
        assert all(m.name.startswith("summarizer") for m in summarizers)

        # Find generators
        generators = registry.find_by_capability("generate")
        assert len(generators) == 1
        assert generators[0].name == "generator-1"

        # Find text2text
        text2text = registry.find_by_capability("text2text")
        assert len(text2text) == 1
        assert text2text[0].name == "summarizer-1"

    def test_find_by_capability_no_matches(self):
        """Test finding models with no matches."""
        registry = ModelRegistry()
        registry.register(
            ModelMetadata(
                name="test-model",
                capabilities=["summarize"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="test/model",
            )
        )

        results = registry.find_by_capability("nonexistent")
        assert results == []

    def test_has_model(self):
        """Test checking if model is registered."""
        registry = ModelRegistry()
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["summarize"],
            expected_vram_mb=1000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model",
        )

        assert not registry.has("test-model")
        registry.register(metadata)
        assert registry.has("test-model")

    def test_list_all(self):
        """Test listing all registered models."""
        registry = ModelRegistry()
        
        # Empty registry
        assert registry.list_all() == []

        # Register models
        for i in range(3):
            registry.register(
                ModelMetadata(
                    name=f"model-{i}",
                    capabilities=["test"],
                    expected_vram_mb=1000,
                    supports_quant=True,
                    preferred_device_policy="gpu_on_demand",
                    model_path=f"model-{i}",
                )
            )

        all_models = registry.list_all()
        assert len(all_models) == 3
        assert set(all_models) == {"model-0", "model-1", "model-2"}

    def test_unregister(self):
        """Test unregistering a model."""
        registry = ModelRegistry()
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["summarize"],
            expected_vram_mb=1000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/model",
        )

        registry.register(metadata)
        assert registry.has("test-model")

        registry.unregister("test-model")
        assert not registry.has("test-model")

    def test_unregister_nonexistent(self):
        """Test that unregistering nonexistent model raises KeyError."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="is not registered"):
            registry.unregister("nonexistent-model")



class TestGlobalRegistry:
    """Test global registry singleton."""

    def test_get_global_registry(self):
        """Test getting global registry instance."""
        registry1 = get_model_registry()
        registry2 = get_model_registry()

        # Should return same instance
        assert registry1 is registry2

    def test_default_models_registered(self):
        """Test that default models are registered."""
        registry = get_model_registry()

        # Check that default models exist
        default_models = [
            "t5-small",
            "facebook/bart-large-cnn",
            "gpt2",
            "distilgpt2",
            "Qwen/Qwen-1_8B-Chat",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        for model_name in default_models:
            assert registry.has(model_name), f"Default model '{model_name}' not registered"

    def test_default_model_capabilities(self):
        """Test that default models have correct capabilities."""
        registry = get_model_registry()

        # Check T5 has summarize capability
        t5 = registry.get("t5-small")
        assert "summarize" in t5.capabilities

        # Check GPT-2 has generate capability
        gpt2 = registry.get("gpt2")
        assert "generate" in gpt2.capabilities

        # Check MiniLM has embed capability
        minilm = registry.get("sentence-transformers/all-MiniLM-L6-v2")
        assert "embed" in minilm.capabilities

    def test_find_summarizers(self):
        """Test finding summarization models."""
        registry = get_model_registry()

        summarizers = registry.find_by_capability("summarize")
        assert len(summarizers) >= 2  # At least T5 and BART

        summarizer_names = [m.name for m in summarizers]
        assert "t5-small" in summarizer_names
        assert "facebook/bart-large-cnn" in summarizer_names

    def test_find_generators(self):
        """Test finding generation models."""
        registry = get_model_registry()

        generators = registry.find_by_capability("generate")
        assert len(generators) >= 3  # At least GPT-2, DistilGPT-2, Qwen

        generator_names = [m.name for m in generators]
        assert "gpt2" in generator_names
        assert "distilgpt2" in generator_names
        assert "Qwen/Qwen-1_8B-Chat" in generator_names

    def test_find_embedders(self):
        """Test finding embedding models."""
        registry = get_model_registry()

        embedders = registry.find_by_capability("embed")
        assert len(embedders) >= 1  # At least MiniLM

        embedder_names = [m.name for m in embedders]
        assert "sentence-transformers/all-MiniLM-L6-v2" in embedder_names



class TestModelMetadataValidation:
    """Test comprehensive metadata validation scenarios."""

    def test_zero_vram_allowed(self):
        """Test that zero VRAM is allowed (for CPU-only models)."""
        metadata = ModelMetadata(
            name="cpu-model",
            capabilities=["test"],
            expected_vram_mb=0,
            supports_quant=False,
            preferred_device_policy="cpu_only",
            model_path="cpu-model",
        )
        assert metadata.expected_vram_mb == 0

    def test_multiple_capabilities(self):
        """Test model with multiple capabilities."""
        metadata = ModelMetadata(
            name="multi-model",
            capabilities=["summarize", "generate", "qa", "embed"],
            expected_vram_mb=2000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="multi-model",
        )
        assert len(metadata.capabilities) == 4
        assert "summarize" in metadata.capabilities
        assert "generate" in metadata.capabilities
        assert "qa" in metadata.capabilities
        assert "embed" in metadata.capabilities

    def test_quantization_config_structure(self):
        """Test various quantization config structures."""
        configs = [
            {"load_in_8bit": True},
            {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"},
            {"load_in_8bit": True, "llm_int8_threshold": 6.0, "llm_int8_has_fp16_weight": False},
        ]

        for config in configs:
            metadata = ModelMetadata(
                name="quant-model",
                capabilities=["test"],
                expected_vram_mb=1000,
                supports_quant=True,
                preferred_device_policy="gpu_on_demand",
                model_path="quant-model",
                quantization_config=config,
            )
            assert metadata.quantization_config == config

    def test_none_quantization_config(self):
        """Test that None quantization_config is allowed."""
        metadata = ModelMetadata(
            name="no-quant-model",
            capabilities=["test"],
            expected_vram_mb=1000,
            supports_quant=False,
            preferred_device_policy="cpu_only",
            model_path="no-quant-model",
            quantization_config=None,
        )
        assert metadata.quantization_config is None
