"""
Unit tests for ModelManager.

Tests the core functionality of model loading, caching, and management.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from mm_orch.runtime.model_manager import (
    ModelManager,
    CachedModel,
    get_model_manager,
    configure_model_manager,
)
from mm_orch.schemas import ModelConfig
from mm_orch.exceptions import ModelError


class TestCachedModel:
    """Tests for CachedModel dataclass."""
    
    def test_cached_model_creation(self):
        """Test creating a CachedModel instance."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = ModelConfig(
            name="test-model",
            model_path="test/path"
        )
        
        cached = CachedModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            device="cpu"
        )
        
        assert cached.model == mock_model
        assert cached.tokenizer == mock_tokenizer
        assert cached.config == config
        assert cached.device == "cpu"
        assert cached.use_count == 0
    
    def test_update_usage(self):
        """Test that update_usage increments count and updates timestamp."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = ModelConfig(name="test", model_path="path")
        
        cached = CachedModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            device="cpu"
        )
        
        initial_time = cached.last_used
        initial_count = cached.use_count
        
        time.sleep(0.01)  # Small delay to ensure time difference
        cached.update_usage()
        
        assert cached.use_count == initial_count + 1
        assert cached.last_used >= initial_time


class TestModelManager:
    """Tests for ModelManager class."""
    
    def test_initialization(self):
        """Test ModelManager initialization with default values."""
        manager = ModelManager()
        
        assert manager.max_cached_models == 3
        assert manager.default_device == "auto"
        assert manager.enable_quantization is True
        assert len(manager._cache) == 0
        assert len(manager._model_configs) == 0
    
    def test_initialization_with_custom_values(self):
        """Test ModelManager initialization with custom values."""
        manager = ModelManager(
            max_cached_models=5,
            default_device="cpu",
            enable_quantization=False
        )
        
        assert manager.max_cached_models == 5
        assert manager.default_device == "cpu"
        assert manager.enable_quantization is False
    
    def test_register_model(self):
        """Test registering a model configuration."""
        manager = ModelManager()
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cpu"
        )
        
        manager.register_model(config)
        
        assert "test-model" in manager._model_configs
        assert manager._model_configs["test-model"] == config
    
    def test_is_registered(self):
        """Test checking if a model is registered."""
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="path")
        
        assert manager.is_registered("test-model") is False
        
        manager.register_model(config)
        
        assert manager.is_registered("test-model") is True
    
    def test_get_registered_models(self):
        """Test getting list of registered models."""
        manager = ModelManager()
        
        assert manager.get_registered_models() == []
        
        manager.register_model(ModelConfig(name="model1", model_path="path1"))
        manager.register_model(ModelConfig(name="model2", model_path="path2"))
        
        registered = manager.get_registered_models()
        assert "model1" in registered
        assert "model2" in registered
        assert len(registered) == 2
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        manager = ModelManager()
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            temperature=0.5
        )
        
        manager.register_model(config)
        
        retrieved = manager.get_model_config("test-model")
        assert retrieved == config
        assert retrieved.temperature == 0.5
        
        # Non-existent model
        assert manager.get_model_config("non-existent") is None
    
    def test_load_unregistered_model_raises_error(self):
        """Test that loading an unregistered model raises ModelError."""
        manager = ModelManager()
        
        with pytest.raises(ModelError) as exc_info:
            manager.load_model("non-existent")
        
        assert "not registered" in str(exc_info.value)
    
    def test_is_loaded(self):
        """Test checking if a model is loaded."""
        manager = ModelManager()
        
        assert manager.is_loaded("test-model") is False
    
    def test_get_cached_models_empty(self):
        """Test getting cached models when cache is empty."""
        manager = ModelManager()
        
        assert manager.get_cached_models() == []
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        manager = ModelManager(max_cached_models=5)
        
        info = manager.get_cache_info()
        
        assert info["max_cached_models"] == 5
        assert info["current_cached"] == 0
        assert info["registered_models"] == 0
        assert "cuda_available" in info
        assert info["models"] == {}
    
    def test_repr(self):
        """Test string representation."""
        manager = ModelManager(max_cached_models=5)
        manager.register_model(ModelConfig(name="test", model_path="path"))
        
        repr_str = repr(manager)
        
        assert "ModelManager" in repr_str
        assert "max_cached=5" in repr_str
        assert "registered=1" in repr_str


class TestModelManagerDeviceSelection:
    """Tests for device selection logic."""
    
    def test_select_device_cpu_requested(self):
        """Test that CPU is returned when explicitly requested."""
        manager = ModelManager()
        
        device = manager._select_device("cpu")
        
        assert device == "cpu"
    
    def test_select_device_cuda_fallback_to_cpu(self):
        """Test fallback to CPU when CUDA is not available."""
        manager = ModelManager()
        manager._cuda_available = False
        
        device = manager._select_device("cuda")
        
        assert device == "cpu"
    
    def test_select_device_auto_without_cuda(self):
        """Test auto device selection without CUDA."""
        manager = ModelManager()
        manager._cuda_available = False
        
        device = manager._select_device("auto")
        
        assert device == "cpu"


class TestModelManagerLRUCache:
    """Tests for LRU cache behavior."""
    
    def test_evict_lru_model_empty_cache(self):
        """Test eviction with empty cache returns None."""
        manager = ModelManager()
        
        result = manager._evict_lru_model()
        
        assert result is None
    
    def test_cache_order_maintained(self):
        """Test that cache maintains insertion order for LRU."""
        manager = ModelManager(max_cached_models=3)
        
        # Manually add items to cache to test ordering
        for i in range(3):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            cached = CachedModel(
                model=Mock(),
                tokenizer=Mock(),
                config=config,
                device="cpu"
            )
            manager._cache[f"model{i}"] = cached
        
        # First item should be the oldest
        oldest = next(iter(manager._cache))
        assert oldest == "model0"
    
    def test_unload_model_not_in_cache(self):
        """Test unloading a model that's not in cache."""
        manager = ModelManager()
        
        result = manager.unload_model("non-existent")
        
        assert result is False
    
    def test_unload_all_empty_cache(self):
        """Test unloading all models from empty cache."""
        manager = ModelManager()
        
        count = manager.unload_all()
        
        assert count == 0
    
    def test_clear_cache(self):
        """Test clearing the cache."""
        manager = ModelManager()
        
        # Add some items manually
        config = ModelConfig(name="test", model_path="path")
        cached = CachedModel(
            model=Mock(),
            tokenizer=Mock(),
            config=config,
            device="cpu"
        )
        manager._cache["test"] = cached
        
        manager.clear_cache()
        
        assert len(manager._cache) == 0


class TestModelManagerGlobalInstance:
    """Tests for global model manager instance."""
    
    def test_get_model_manager_creates_instance(self):
        """Test that get_model_manager creates a new instance."""
        # Reset global instance
        import mm_orch.runtime.model_manager as mm
        mm._global_model_manager = None
        
        manager = get_model_manager()
        
        assert manager is not None
        assert isinstance(manager, ModelManager)
    
    def test_get_model_manager_returns_same_instance(self):
        """Test that get_model_manager returns the same instance."""
        import mm_orch.runtime.model_manager as mm
        mm._global_model_manager = None
        
        manager1 = get_model_manager()
        manager2 = get_model_manager()
        
        assert manager1 is manager2
    
    def test_configure_model_manager(self):
        """Test configuring the global model manager."""
        import mm_orch.runtime.model_manager as mm
        mm._global_model_manager = None
        
        manager = configure_model_manager(
            max_cached_models=10,
            default_device="cpu",
            enable_quantization=False
        )
        
        assert manager.max_cached_models == 10
        assert manager.default_device == "cpu"
        assert manager.enable_quantization is False


class TestModelManagerWithMockedLoading:
    """Tests for model loading with mocked transformers."""
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_load_model_success(self, mock_load):
        """Test successful model loading."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path")
        manager.register_model(config)
        
        cached = manager.load_model("test-model")
        
        assert cached.model == mock_model
        assert cached.tokenizer == mock_tokenizer
        assert manager.is_loaded("test-model")
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_load_model_uses_cache(self, mock_load):
        """Test that loading a cached model doesn't reload it."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path")
        manager.register_model(config)
        
        # First load
        cached1 = manager.load_model("test-model")
        # Second load should use cache
        cached2 = manager.load_model("test-model")
        
        assert cached1 is cached2
        assert mock_load.call_count == 1  # Only called once
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_get_model_loads_if_not_cached(self, mock_load):
        """Test that get_model loads the model if not cached."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path")
        manager.register_model(config)
        
        cached = manager.get_model("test-model")
        
        assert cached.model == mock_model
        mock_load.assert_called_once()
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_lru_eviction_when_cache_full(self, mock_load):
        """Test LRU eviction when cache is full."""
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=2)
        
        # Register 3 models
        for i in range(3):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Load first two models
        manager.load_model("model0")
        manager.load_model("model1")
        
        assert len(manager._cache) == 2
        assert manager.is_loaded("model0")
        assert manager.is_loaded("model1")
        
        # Load third model - should evict model0 (LRU)
        manager.load_model("model2")
        
        assert len(manager._cache) == 2
        assert not manager.is_loaded("model0")  # Evicted
        assert manager.is_loaded("model1")
        assert manager.is_loaded("model2")
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_cache_updates_on_access(self, mock_load):
        """Test that accessing a cached model updates its position."""
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=2)
        
        # Register 3 models
        for i in range(3):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Load first two models
        manager.load_model("model0")
        manager.load_model("model1")
        
        # Access model0 to make it recently used
        manager.load_model("model0")
        
        # Load third model - should evict model1 (now LRU)
        manager.load_model("model2")
        
        assert manager.is_loaded("model0")  # Still loaded (recently used)
        assert not manager.is_loaded("model1")  # Evicted
        assert manager.is_loaded("model2")



class TestModelManagerEnhancedFeatures:
    """Tests for enhanced model manager features (Phase B)."""
    
    def test_usage_stats_initialization(self):
        """Test that ModelUsageStats is properly initialized."""
        from mm_orch.runtime.model_manager import ModelUsageStats
        
        stats = ModelUsageStats()
        assert stats.load_count == 0
        assert stats.last_used == 0.0
        assert stats.total_inference_time == 0.0
        assert stats.peak_vram_mb == 0
    
    def test_residency_seconds_parameter(self):
        """Test that residency_seconds parameter is properly set."""
        manager = ModelManager(residency_seconds=60)
        assert manager.residency_seconds == 60
    
    def test_usage_counter_increment_on_load(self):
        """Test Requirement 6.1: Usage counter increments on model load."""
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        # Mock the loading process
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # First load
            manager.load_model("test-model")
            stats = manager.get_usage_stats("test-model")
            assert stats.load_count == 1
            
            # Second load (from cache)
            manager.load_model("test-model")
            stats = manager.get_usage_stats("test-model")
            assert stats.load_count == 2
    
    def test_cleanup_stale_models(self):
        """Test Requirement 6.2: Cleanup models not used for residency_seconds."""
        manager = ModelManager(residency_seconds=1)  # 1 second for testing
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        # Mock the loading process
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load model
            manager.load_model("test-model")
            assert manager.is_loaded("test-model")
            
            # Wait for residency timeout
            time.sleep(1.1)
            
            # Cleanup stale models
            unloaded = manager.cleanup_stale()
            assert "test-model" in unloaded
            assert not manager.is_loaded("test-model")
    
    def test_cleanup_stale_does_not_remove_recent_models(self):
        """Test that cleanup_stale does not remove recently used models."""
        manager = ModelManager(residency_seconds=2)
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load model
            manager.load_model("test-model")
            assert manager.is_loaded("test-model")
            
            # Wait less than residency timeout
            time.sleep(0.5)
            
            # Cleanup should not remove the model
            unloaded = manager.cleanup_stale()
            assert "test-model" not in unloaded
            assert manager.is_loaded("test-model")
    
    def test_calculate_model_priority(self):
        """Test Requirement 6.4: Priority calculation based on usage patterns."""
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load model
            manager.load_model("test-model")
            
            # Calculate priority
            priority = manager._calculate_model_priority("test-model")
            assert priority > 0.0
            
            # Load again to increase usage count
            manager.load_model("test-model")
            new_priority = manager._calculate_model_priority("test-model")
            assert new_priority > priority  # Higher usage should increase priority
    
    def test_priority_with_device_policy(self):
        """Test that device policy affects priority calculation."""
        manager = ModelManager()
        
        # Create config with device_policy attribute
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        config.device_policy = "gpu_resident"
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            manager.load_model("test-model")
            priority = manager._calculate_model_priority("test-model")
            
            # GPU resident models should have higher priority
            assert priority > 0.0
    
    def test_evict_by_priority(self):
        """Test Requirement 6.4: Eviction based on priority."""
        manager = ModelManager(max_cached_models=2)
        
        config1 = ModelConfig(name="model1", model_path="path1", device="cpu")
        config2 = ModelConfig(name="model2", model_path="path2", device="cpu")
        manager.register_model(config1)
        manager.register_model(config2)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load both models
            manager.load_model("model1")
            manager.load_model("model2")
            
            # Access model2 multiple times to increase its priority
            for _ in range(5):
                manager.load_model("model2")
            
            # Evict by priority
            evicted = manager._evict_by_priority()
            
            # model1 should be evicted as it has lower priority
            assert evicted == "model1"
            assert not manager.is_loaded("model1")
            assert manager.is_loaded("model2")
    
    def test_get_usage_stats(self):
        """Test retrieving usage statistics for a model."""
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load model
            manager.load_model("test-model")
            
            # Get stats
            stats = manager.get_usage_stats("test-model")
            assert stats.load_count == 1
            assert stats.last_used > 0
    
    def test_get_usage_stats_for_unloaded_model(self):
        """Test getting stats for a model that was never loaded."""
        manager = ModelManager()
        stats = manager.get_usage_stats("nonexistent-model")
        
        # Should return default stats
        assert stats.load_count == 0
        assert stats.last_used == 0.0
        assert stats.total_inference_time == 0.0
        assert stats.peak_vram_mb == 0
    
    def test_cache_info_includes_usage_stats(self):
        """Test that get_cache_info includes usage statistics."""
        manager = ModelManager()
        config = ModelConfig(name="test-model", model_path="test/path", device="cpu")
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            manager.load_model("test-model")
            
            info = manager.get_cache_info()
            assert "residency_seconds" in info
            assert "test-model" in info["models"]
            
            model_info = info["models"]["test-model"]
            assert "load_count" in model_info
            assert "total_inference_time" in model_info
            assert "peak_vram_mb" in model_info
            assert "priority" in model_info
    
    def test_inference_time_tracking(self):
        """Test that inference time is tracked during model inference."""
        manager = ModelManager()
        config = ModelConfig(
            name="test-model",
            model_path="test/path",
            device="cpu",
            max_length=512,
            temperature=0.7
        )
        manager.register_model(config)
        
        with patch.object(manager, '_load_model_from_path') as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            
            # Mock tokenizer behavior
            mock_tokenizer.return_value = {
                "input_ids": Mock(),
                "attention_mask": Mock()
            }
            mock_tokenizer.batch_decode.return_value = ["test output"]
            mock_tokenizer.pad_token_id = 0
            
            # Mock model generate with a small delay to simulate inference
            def mock_generate(*args, **kwargs):
                time.sleep(0.01)  # Small delay to simulate inference
                return Mock()
            
            mock_model.generate = mock_generate
            
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Perform inference
            manager.infer("test-model", "test input")
            
            # Check that inference time was tracked
            stats = manager.get_usage_stats("test-model")
            assert stats.total_inference_time > 0
    
    def test_global_manager_with_residency_seconds(self):
        """Test that global manager functions accept residency_seconds."""
        from mm_orch.runtime.model_manager import _global_model_manager
        
        # Reset global manager
        import mm_orch.runtime.model_manager as mm
        mm._global_model_manager = None
        
        manager = get_model_manager(residency_seconds=45)
        assert manager.residency_seconds == 45
        
        # Reset for other tests
        mm._global_model_manager = None
    
    def test_configure_model_manager_with_residency_seconds(self):
        """Test configuring model manager with residency_seconds."""
        manager = configure_model_manager(
            max_cached_models=5,
            residency_seconds=120
        )
        
        assert manager.max_cached_models == 5
        assert manager.residency_seconds == 120



class TestQuantizationSupport:
    """Tests for quantization support in ModelManager (Requirement 6.3)."""
    
    def test_8bit_quantization_config_registered(self):
        """Test that 8-bit quantization config is properly registered."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model-8bit",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Verify config is registered correctly
        assert manager.is_registered("test-model-8bit")
        retrieved_config = manager.get_model_config("test-model-8bit")
        assert retrieved_config.quantization == "8bit"
    
    def test_4bit_quantization_config_registered(self):
        """Test that 4-bit quantization config is properly registered."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model-4bit",
            model_path="test/path",
            device="cuda",
            quantization="4bit"
        )
        manager.register_model(config)
        
        # Verify config is registered correctly
        assert manager.is_registered("test-model-4bit")
        retrieved_config = manager.get_model_config("test-model-4bit")
        assert retrieved_config.quantization == "4bit"
    
    def test_quantization_disabled_when_enable_false(self):
        """Test that quantization is not applied when enable_quantization=False."""
        manager = ModelManager(enable_quantization=False)
        config = ModelConfig(
            name="test-model-no-quant",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Manager should still register the config
        assert manager.is_registered("test-model-no-quant")
        assert manager.enable_quantization is False
    
    def test_quantization_not_applied_on_cpu(self):
        """Test that quantization is not applied when device is CPU."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model-cpu",
            model_path="test/path",
            device="cpu",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # Config should be registered with CPU device
        retrieved_config = manager.get_model_config("test-model-cpu")
        assert retrieved_config.device == "cpu"
    
    def test_no_quantization_when_config_is_none(self):
        """Test that no quantization is applied when config.quantization is None."""
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model-none",
            model_path="test/path",
            device="cuda",
            quantization=None
        )
        manager.register_model(config)
        
        retrieved_config = manager.get_model_config("test-model-none")
        assert retrieved_config.quantization is None
    
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_quantized_model_loads_successfully(self, mock_load):
        """Test that quantized models can be loaded successfully."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager(enable_quantization=True)
        config = ModelConfig(
            name="test-model-load",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        cached = manager.load_model("test-model-load")
        
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
            name="test-model-cache",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        # First load
        cached1 = manager.load_model("test-model-cache")
        # Second load should use cache
        cached2 = manager.load_model("test-model-cache")
        
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
            name="test-model-stats",
            model_path="test/path",
            device="cuda",
            quantization="8bit"
        )
        manager.register_model(config)
        
        manager.load_model("test-model-stats")
        
        stats = manager.get_usage_stats("test-model-stats")
        assert stats.load_count == 1
        assert stats.last_used > 0
    
    def test_model_metadata_with_quantization_config(self):
        """Test that ModelMetadata supports quantization_config."""
        from mm_orch.registries.model_registry import ModelMetadata
        
        metadata = ModelMetadata(
            name="test-model-metadata",
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
            name="test-model-no-config",
            capabilities=["generate"],
            expected_vram_mb=2000,
            supports_quant=False,
            preferred_device_policy="cpu_only",
            model_path="test/path"
        )
        
        # Verify metadata works without quantization_config
        assert metadata.quantization_config is None
        assert metadata.supports_quant is False
