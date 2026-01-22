"""
真实模型管理器单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import torch

from mm_orch.runtime.real_model_manager import (
    RealModelManager,
    CachedModel,
)
from mm_orch.runtime.model_loader import ModelConfig, LoadedModel
from mm_orch.exceptions import ModelLoadError, OutOfMemoryError


class TestCachedModel:
    """CachedModel测试"""
    
    def test_create_cached_model(self):
        """测试创建缓存模型"""
        mock_loaded = Mock(spec=LoadedModel)
        cached = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now()
        )
        
        assert cached.loaded_model == mock_loaded
        assert cached.access_count == 0
    
    def test_touch(self):
        """测试更新访问时间"""
        mock_loaded = Mock(spec=LoadedModel)
        cached = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now()
        )
        
        old_time = cached.last_accessed
        cached.touch()
        
        assert cached.access_count == 1
        assert cached.last_accessed >= old_time


class TestRealModelManager:
    """RealModelManager测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        manager = RealModelManager()
        
        assert manager.max_cached_models == 3
        assert manager.memory_threshold == 0.8
        assert manager.fallback_to_cpu is True
        assert len(manager._model_cache) == 0
    
    def test_init_custom(self):
        """测试自定义初始化"""
        manager = RealModelManager(
            max_cached_models=5,
            memory_threshold=0.9,
            fallback_to_cpu=False
        )
        
        assert manager.max_cached_models == 5
        assert manager.memory_threshold == 0.9
        assert manager.fallback_to_cpu is False
    
    def test_is_loaded_false(self):
        """测试模型未加载"""
        manager = RealModelManager()
        assert manager.is_loaded("gpt2") is False
    
    def test_list_loaded_models_empty(self):
        """测试列出已加载模型（空）"""
        manager = RealModelManager()
        assert manager.list_loaded_models() == []
    
    def test_get_model_not_loaded(self):
        """测试获取未加载的模型"""
        manager = RealModelManager()
        assert manager.get_model("gpt2") is None
    
    def test_get_model_info_not_loaded(self):
        """测试获取未加载模型的信息"""
        manager = RealModelManager()
        assert manager.get_model_info("gpt2") is None
    
    def test_unload_model_not_loaded(self):
        """测试卸载未加载的模型"""
        manager = RealModelManager()
        assert manager.unload_model("gpt2") is False
    
    def test_get_stats_initial(self):
        """测试获取初始统计信息"""
        manager = RealModelManager()
        stats = manager.get_stats()
        
        assert stats["loads"] == 0
        assert stats["unloads"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["cached_models"] == 0
    
    def test_clear_cache_empty(self):
        """测试清空空缓存"""
        manager = RealModelManager()
        count = manager.clear_cache()
        assert count == 0
    
    def test_load_model_cache_hit(self):
        """测试缓存命中"""
        manager = RealModelManager()
        
        # 预先添加模型到缓存
        mock_loaded = Mock(spec=LoadedModel)
        mock_loaded.device = "cuda"
        mock_loaded.dtype = torch.float16
        mock_loaded.memory_footprint = 1024**3
        mock_loaded.config = ModelConfig(model_name="gpt2")
        
        manager._model_cache["gpt2"] = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now()
        )
        
        # 加载模型（应该命中缓存）
        config = ModelConfig(model_name="gpt2")
        result = manager.load_model(config)
        
        assert result == mock_loaded
        assert manager._stats["cache_hits"] == 1
    
    @patch.object(RealModelManager, "_check_memory_before_load")
    def test_load_model_cache_miss(self, mock_check_mem):
        """测试缓存未命中"""
        manager = RealModelManager()
        
        # Mock模型加载器
        mock_loaded = Mock(spec=LoadedModel)
        mock_loaded.device = "cpu"
        mock_loaded.dtype = torch.float32
        mock_loaded.memory_footprint = 500 * 1024**2
        mock_loaded.config = ModelConfig(model_name="gpt2")
        
        manager._model_loader = Mock()
        manager._model_loader.load_model.return_value = mock_loaded
        
        # 加载模型
        config = ModelConfig(model_name="gpt2", device="cpu")
        result = manager.load_model(config)
        
        assert result == mock_loaded
        assert manager._stats["cache_misses"] == 1
        assert manager._stats["loads"] == 1
        assert manager.is_loaded("gpt2") is True
    
    def test_unload_model_success(self):
        """测试成功卸载模型"""
        manager = RealModelManager()
        
        # 添加模型到缓存
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_loaded = Mock(spec=LoadedModel)
        mock_loaded.model = mock_model
        mock_loaded.tokenizer = mock_tokenizer
        
        manager._model_cache["gpt2"] = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now()
        )
        
        # 卸载模型
        with patch("torch.cuda.is_available", return_value=False):
            result = manager.unload_model("gpt2")
        
        assert result is True
        assert manager._stats["unloads"] == 1
        assert manager.is_loaded("gpt2") is False
    
    def test_lru_eviction(self):
        """测试LRU驱逐策略"""
        manager = RealModelManager(max_cached_models=2)
        
        # 添加两个模型
        for name in ["model1", "model2"]:
            mock_loaded = Mock(spec=LoadedModel)
            mock_loaded.model = Mock()
            mock_loaded.tokenizer = Mock()
            manager._model_cache[name] = CachedModel(
                loaded_model=mock_loaded,
                last_accessed=datetime.now()
            )
        
        # 确保缓存空间（应该驱逐model1）
        with patch("torch.cuda.is_available", return_value=False):
            manager._ensure_cache_space()
        
        # model1应该被驱逐
        assert len(manager._model_cache) == 1
    
    def test_get_model_info_loaded(self):
        """测试获取已加载模型的信息"""
        manager = RealModelManager()
        
        # 添加模型到缓存
        mock_loaded = Mock(spec=LoadedModel)
        mock_loaded.device = "cuda"
        mock_loaded.dtype = torch.float16
        mock_loaded.memory_footprint = 1024**3
        mock_loaded.config = Mock()
        mock_loaded.config.model_type = "gpt2"
        mock_loaded.config.quantization = None
        mock_loaded.config.flash_attention = False
        
        manager._model_cache["gpt2"] = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now(),
            access_count=5
        )
        
        info = manager.get_model_info("gpt2")
        
        assert info is not None
        assert info["model_name"] == "gpt2"
        assert info["device"] == "cuda"
        assert info["access_count"] == 5
    
    def test_switch_device_not_loaded(self):
        """测试切换未加载模型的设备"""
        manager = RealModelManager()
        result = manager.switch_device("gpt2", "cpu")
        assert result is False
    
    def test_switch_device_same_device(self):
        """测试切换到相同设备"""
        manager = RealModelManager()
        
        mock_loaded = Mock(spec=LoadedModel)
        mock_loaded.device = "cpu"
        
        manager._model_cache["gpt2"] = CachedModel(
            loaded_model=mock_loaded,
            last_accessed=datetime.now()
        )
        
        result = manager.switch_device("gpt2", "cpu")
        assert result is True
    
    def test_get_memory_summary(self):
        """测试获取内存摘要"""
        manager = RealModelManager()
        
        with patch.object(manager._memory_monitor, "get_summary", return_value={
            "gpu": {"available": True, "allocated_gb": 4.0},
            "cpu": {"used_gb": 8.0}
        }):
            summary = manager.get_memory_summary()
            
            assert "gpu" in summary
            assert "cpu" in summary
