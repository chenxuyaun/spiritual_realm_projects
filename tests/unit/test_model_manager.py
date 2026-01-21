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
