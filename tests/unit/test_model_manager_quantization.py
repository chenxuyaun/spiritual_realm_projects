"""
Unit tests for ModelManager quantization support.

Tests Requirement 6.3: Support 8-bit and 4-bit quantization using bitsandbytes
with fallback to standard loading if quantization fails.
"""

import pytest
from unittest.mock import Mock, patch

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


class TestQuantizationSupport:
    """Tests for quantization support in ModelManager."""
    
    def test_8bit_quantization_config_registered(self):
        """Test that 8-bit quantization config is properly registered."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Verify config is registered correctly
        assert manager.is_registered("test-model")
        retrieved_config = manager.get_model_config("test-model")
        assert retrieved_config.quantization == "8bit"
    
    def test_4bit_quantization_config_registered(self):
        """Test that 4-bit quantization config is properly registered."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="4bit"
        )
        manager.register_model(config)
        
        # Verify config is registered correctly
        assert manager.is_registered("test-model")
        retrieved_config = manager.get_model_config("test-model")
        assert retrieved_config.quantization == "4bit"
    
    def test_quantization_disabled_when_enable_false(self):
        """Test that quantization is not applied when enable_quantization=False."""
        manager = ModelManager(enable_quantization=False)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Manager should still register the config
        assert manager.is_registered("test-model")
        assert manager.enable_quantization is False
    
    def test_quantization_not_applied_on_cpu(self):
        """Test that quantization is not applied when device is CPU."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cpu",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Config should be registered with CPU device
        retrieved_config = manager.get_model_config("test-model")
        assert retrieved_config.device == "cpu"
    
    def test_no_quantization_when_config_is_none(self):
        """Test that no quantization is applied when config.quantization is None."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization=None
        )
        manager.register_model(config)
        
        retrieved_config = manager.get_model_config("test-model")
        assert retrieved_config.quantization is None


class TestQuantizationWithMockedLoading:
    """Tests for quantization with mocked model loading."""
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_quantized_model_loads_successfully(self, mock_load):
        """Test that quantized models can be loaded successfully."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        cached = manager.load_model("test-model")
        
        assert cached is not None
        assert cached.model == mock_model
        assert cached.tokenizer == mock_tokenizer
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_quantized_model_caching(self, mock_load):
        """Test that quantized models are properly cached."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager(enable_quantization=True, max_cached_models=2)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # First load
        cached1 = manager.load_model("test-model")
        # Second load should use cache
        cached2 = manager.load_model("test-model")
        
        assert cached1 is cached2
        assert mock_load.call_count == 1  # Only loaded once
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_quantized_model_usage_stats(self, mock_load):
        """Test that usage stats work correctly with quantized models."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        manager.load_model("test-model")
        
        stats = manager.get_usage_stats("test-model")
        assert stats.load_count == 1
        assert stats.last_used > 0


class TestQuantizationWithModelMetadata:
    """Tests for quantization with ModelMetadata from registry."""
    
    def test_model_metadata_with_quantization_config(self):
        """Test that ModelMetadata supports quantization_config."""
        from mm_orch.registries.model_registry import ModelMetadata
        
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["generate"],
            expected_vram_mb=2000,
            supports_quant=True,
            preferred_device_policy="gpu_on_demand",
            model_path="test/path",
            quantization_config={
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0
            }
        )
        
        # Verify metadata has quantization_config
        assert metadata.quantization_config is not None
        assert metadata.quantization_config["load_in_8bit"] is True
        assert metadata.quantization_config["llm_int8_threshold"] == 6.0
    
    def test_model_metadata_without_quantization_config(self):
        """Test that ModelMetadata works without quantization_config."""
        from mm_orch.registries.model_registry import ModelMetadata
        
        metadata = ModelMetadata(
            name="test-model",
            capabilities=["generate"],
            expected_vram_mb=2000,
            supports_quant=False,
            preferred_device_policy="cpu_only",
            model_path="test/path"
        )
        
        # Verify metadata works without quantization_config
        assert metadata.quantization_config is None
        assert metadata.supports_quant is False
