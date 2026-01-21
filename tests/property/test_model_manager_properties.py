"""
Property-based tests for ModelManager.

Tests the correctness properties defined in the design document:
- Property 24: 模型延迟加载
- Property 25: 模型设备选择
- Property 26: 模型缓存复用
- Property 27: 模型LRU淘汰

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
import time

from mm_orch.runtime.model_manager import ModelManager, CachedModel
from mm_orch.schemas import ModelConfig
from mm_orch.exceptions import ModelError


# Strategies for generating test data
model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() == x and len(x) > 0)

model_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/-_.'),
    min_size=1,
    max_size=100
).filter(lambda x: len(x.strip()) > 0)

device_strategy = st.sampled_from(["auto", "cuda", "cpu"])

max_cached_models_strategy = st.integers(min_value=1, max_value=10)


class TestProperty24LazyLoading:
    """
    Property 24: 模型延迟加载
    
    对于任何模型，在系统启动时该模型不应该被加载到内存，
    只有在首次调用get_model或infer方法时才应该触发加载。
    
    **Validates: Requirements 10.1**
    """
    
    @given(
        model_name=model_name_strategy,
        model_path=model_path_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_model_not_loaded_on_registration(self, model_name, model_path):
        """
        Feature: muai-orchestration-system, Property 24: 模型延迟加载
        
        对于任何注册的模型，注册后不应该自动加载到缓存中。
        """
        manager = ModelManager()
        config = ModelConfig(name=model_name, model_path=model_path)
        
        # Register the model
        manager.register_model(config)
        
        # Model should be registered but NOT loaded
        assert manager.is_registered(model_name)
        assert not manager.is_loaded(model_name)
        assert len(manager._cache) == 0
    
    @given(
        model_names=st.lists(model_name_strategy, min_size=1, max_size=5, unique=True)
    )
    @settings(max_examples=50)
    def test_multiple_models_not_loaded_on_registration(self, model_names):
        """
        Feature: muai-orchestration-system, Property 24: 模型延迟加载
        
        对于任何数量的注册模型，注册后都不应该自动加载。
        """
        manager = ModelManager()
        
        for i, name in enumerate(model_names):
            config = ModelConfig(name=name, model_path=f"path/{i}")
            manager.register_model(config)
        
        # All models should be registered but none loaded
        assert len(manager.get_registered_models()) == len(model_names)
        assert len(manager.get_cached_models()) == 0
        
        for name in model_names:
            assert manager.is_registered(name)
            assert not manager.is_loaded(name)
    
    @given(model_name=model_name_strategy)
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_model_loaded_on_first_get(self, mock_load, model_name):
        """
        Feature: muai-orchestration-system, Property 24: 模型延迟加载
        
        模型只有在首次调用get_model时才会被加载。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager()
        config = ModelConfig(name=model_name, model_path="test/path")
        manager.register_model(config)
        
        # Before get_model, not loaded
        assert not manager.is_loaded(model_name)
        
        # After get_model, should be loaded
        manager.get_model(model_name)
        
        assert manager.is_loaded(model_name)
        mock_load.assert_called_once()


class TestProperty25DeviceSelection:
    """
    Property 25: 模型设备选择
    
    对于任何模型加载请求，当device参数为'auto'时，如果CUDA可用且GPU内存充足，
    模型应该被加载到GPU；否则应该加载到CPU。
    
    **Validates: Requirements 10.2, 10.3**
    """
    
    @given(device=device_strategy)
    @settings(max_examples=100)
    def test_cpu_always_returns_cpu(self, device):
        """
        Feature: muai-orchestration-system, Property 25: 模型设备选择
        
        当请求CPU设备时，总是返回CPU。
        """
        manager = ModelManager()
        
        if device == "cpu":
            result = manager._select_device(device)
            assert result == "cpu"
    
    @given(device=st.sampled_from(["auto", "cuda"]))
    @settings(max_examples=50)
    def test_fallback_to_cpu_without_cuda(self, device):
        """
        Feature: muai-orchestration-system, Property 25: 模型设备选择
        
        当CUDA不可用时，auto和cuda请求都应该回退到CPU。
        """
        manager = ModelManager()
        manager._cuda_available = False
        
        result = manager._select_device(device)
        
        assert result == "cpu"
    
    @given(
        model_name=model_name_strategy,
        requested_device=device_strategy
    )
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_device_selection_consistency(self, mock_load, model_name, requested_device):
        """
        Feature: muai-orchestration-system, Property 25: 模型设备选择
        
        设备选择结果应该是有效的设备类型。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager()
        manager._cuda_available = False  # Force CPU for predictable testing
        
        config = ModelConfig(name=model_name, model_path="path", device=requested_device)
        manager.register_model(config)
        
        cached = manager.load_model(model_name)
        
        # Device should be a valid option
        assert cached.device in ["cuda", "cpu"]
        
        # Without CUDA, should always be CPU
        assert cached.device == "cpu"


class TestProperty26CacheReuse:
    """
    Property 26: 模型缓存复用
    
    对于任何模型，当该模型已经在缓存中时，再次请求该模型应该直接返回缓存的实例，
    而不应该重新加载。
    
    **Validates: Requirements 10.4**
    """
    
    @given(model_name=model_name_strategy)
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_cached_model_reused(self, mock_load, model_name):
        """
        Feature: muai-orchestration-system, Property 26: 模型缓存复用
        
        对于任何已缓存的模型，再次请求应该返回相同的实例。
        """
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        
        manager = ModelManager()
        config = ModelConfig(name=model_name, model_path="path")
        manager.register_model(config)
        
        # First load
        cached1 = manager.get_model(model_name)
        
        # Second load should return same instance
        cached2 = manager.get_model(model_name)
        
        assert cached1 is cached2
        assert mock_load.call_count == 1  # Only loaded once
    
    @given(
        model_name=model_name_strategy,
        access_count=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_multiple_accesses_use_cache(self, mock_load, model_name, access_count):
        """
        Feature: muai-orchestration-system, Property 26: 模型缓存复用
        
        对于任何数量的访问，都应该只加载一次模型。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager()
        config = ModelConfig(name=model_name, model_path="path")
        manager.register_model(config)
        
        # Access multiple times
        cached_instances = [manager.get_model(model_name) for _ in range(access_count)]
        
        # All should be the same instance
        assert all(c is cached_instances[0] for c in cached_instances)
        
        # Only one load call
        assert mock_load.call_count == 1
    
    @given(model_name=model_name_strategy)
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_cache_updates_usage_stats(self, mock_load, model_name):
        """
        Feature: muai-orchestration-system, Property 26: 模型缓存复用
        
        每次访问缓存的模型时，使用统计应该更新。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager()
        config = ModelConfig(name=model_name, model_path="path")
        manager.register_model(config)
        
        # First access
        cached = manager.get_model(model_name)
        initial_count = cached.use_count
        
        # Second access
        manager.get_model(model_name)
        
        # Use count should increase
        assert cached.use_count > initial_count


class TestProperty27LRUEviction:
    """
    Property 27: 模型LRU淘汰
    
    对于任何模型管理器，当缓存的模型数量超过max_cached_models限制时，
    最久未使用的模型应该被卸载以释放内存。
    
    **Validates: Requirements 10.5**
    """
    
    @given(max_cached=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_cache_respects_max_limit(self, mock_load, max_cached):
        """
        Feature: muai-orchestration-system, Property 27: 模型LRU淘汰
        
        缓存的模型数量不应该超过max_cached_models限制。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=max_cached)
        
        # Register more models than cache limit
        num_models = max_cached + 3
        for i in range(num_models):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Load all models
        for i in range(num_models):
            manager.load_model(f"model{i}")
        
        # Cache should not exceed limit
        assert len(manager._cache) <= max_cached
    
    @given(max_cached=st.integers(min_value=2, max_value=5))
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_lru_model_evicted_first(self, mock_load, max_cached):
        """
        Feature: muai-orchestration-system, Property 27: 模型LRU淘汰
        
        当需要淘汰时，最久未使用的模型应该被首先淘汰。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=max_cached)
        
        # Register models
        for i in range(max_cached + 1):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Load first max_cached models
        for i in range(max_cached):
            manager.load_model(f"model{i}")
        
        # model0 is the oldest (LRU)
        assert manager.is_loaded("model0")
        
        # Load one more model - should evict model0
        manager.load_model(f"model{max_cached}")
        
        # model0 should be evicted
        assert not manager.is_loaded("model0")
        # New model should be loaded
        assert manager.is_loaded(f"model{max_cached}")
    
    @given(max_cached=st.integers(min_value=2, max_value=5))
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_recently_used_model_not_evicted(self, mock_load, max_cached):
        """
        Feature: muai-orchestration-system, Property 27: 模型LRU淘汰
        
        最近使用的模型不应该被淘汰。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=max_cached)
        
        # Register models
        for i in range(max_cached + 1):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Load first max_cached models
        for i in range(max_cached):
            manager.load_model(f"model{i}")
        
        # Access model0 to make it recently used
        manager.get_model("model0")
        
        # Load one more model - should evict model1 (now LRU), not model0
        manager.load_model(f"model{max_cached}")
        
        # model0 should still be loaded (recently used)
        assert manager.is_loaded("model0")
        # model1 should be evicted (LRU)
        assert not manager.is_loaded("model1")
    
    @given(
        max_cached=st.integers(min_value=2, max_value=4),
        access_pattern=st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=5,
            max_size=20
        )
    )
    @settings(max_examples=50)
    @patch('mm_orch.runtime.model_manager.ModelManager._load_model_from_path')
    def test_cache_invariant_under_access_pattern(self, mock_load, max_cached, access_pattern):
        """
        Feature: muai-orchestration-system, Property 27: 模型LRU淘汰
        
        对于任何访问模式，缓存大小不应该超过限制。
        """
        mock_load.return_value = (Mock(), Mock())
        
        manager = ModelManager(max_cached_models=max_cached)
        
        # Register models
        num_models = max(access_pattern) + 1 if access_pattern else 1
        for i in range(num_models):
            config = ModelConfig(name=f"model{i}", model_path=f"path{i}")
            manager.register_model(config)
        
        # Apply access pattern
        for model_idx in access_pattern:
            if model_idx < num_models:
                manager.load_model(f"model{model_idx}")
                
                # Invariant: cache size never exceeds limit
                assert len(manager._cache) <= max_cached


class TestModelManagerInvariants:
    """
    Additional invariant tests for ModelManager.
    """
    
    @given(
        model_names=st.lists(model_name_strategy, min_size=1, max_size=10, unique=True)
    )
    @settings(max_examples=50)
    def test_registered_models_count_invariant(self, model_names):
        """
        注册的模型数量应该等于注册操作的次数。
        """
        manager = ModelManager()
        
        for i, name in enumerate(model_names):
            config = ModelConfig(name=name, model_path=f"path{i}")
            manager.register_model(config)
            
            # Invariant: registered count equals number of registrations
            assert len(manager.get_registered_models()) == i + 1
    
    @given(model_name=model_name_strategy)
    @settings(max_examples=50)
    def test_unregistered_model_raises_error(self, model_name):
        """
        尝试加载未注册的模型应该抛出ModelError。
        """
        manager = ModelManager()
        
        with pytest.raises(ModelError):
            manager.load_model(model_name)
    
    @given(max_cached=st.integers(min_value=1, max_value=10))
    @settings(max_examples=50)
    def test_cache_info_consistency(self, max_cached):
        """
        缓存信息应该与实际状态一致。
        """
        manager = ModelManager(max_cached_models=max_cached)
        
        info = manager.get_cache_info()
        
        assert info["max_cached_models"] == max_cached
        assert info["current_cached"] == len(manager._cache)
        assert info["registered_models"] == len(manager._model_configs)
