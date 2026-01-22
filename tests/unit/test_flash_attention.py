"""
FlashAttention单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import torch

from mm_orch.runtime.flash_attention import (
    FlashAttentionManager,
    FlashAttentionInfo,
    is_flash_attention_available,
    get_flash_attention_info,
    get_best_attention_implementation,
)


class TestFlashAttentionInfo:
    """FlashAttentionInfo测试"""
    
    def test_create_available(self):
        """测试创建可用信息"""
        info = FlashAttentionInfo(
            available=True,
            version="2.3.0",
            cuda_version="11.8"
        )
        
        assert info.available is True
        assert info.version == "2.3.0"
        assert info.cuda_version == "11.8"
        assert info.reason_unavailable is None
    
    def test_create_unavailable(self):
        """测试创建不可用信息"""
        info = FlashAttentionInfo(
            available=False,
            reason_unavailable="CUDA not available"
        )
        
        assert info.available is False
        assert info.reason_unavailable == "CUDA not available"


class TestFlashAttentionManager:
    """FlashAttentionManager测试"""
    
    def setup_method(self):
        """每个测试前重置状态"""
        FlashAttentionManager.reset()
    
    @patch("torch.cuda.is_available", return_value=False)
    def test_is_available_no_cuda(self, mock_available):
        """测试无CUDA时不可用"""
        result = FlashAttentionManager.is_available()
        
        assert result is False
        info = FlashAttentionManager.get_info()
        assert "CUDA not available" in info.reason_unavailable
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_is_available_low_compute_capability(self, mock_available):
        """测试计算能力不足时不可用"""
        mock_props = Mock()
        mock_props.major = 7
        mock_props.minor = 5
        
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            result = FlashAttentionManager.is_available()
        
        assert result is False
        info = FlashAttentionManager.get_info()
        assert "compute capability" in info.reason_unavailable.lower()
    
    @patch("torch.cuda.is_available", return_value=True)
    def test_is_available_no_flash_attn_package(self, mock_available):
        """测试flash-attn包未安装时不可用"""
        mock_props = Mock()
        mock_props.major = 8
        mock_props.minor = 0
        
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            with patch.dict("sys.modules", {"flash_attn": None}):
                # 模拟ImportError
                import builtins
                original_import = builtins.__import__
                
                def mock_import(name, *args, **kwargs):
                    if name == "flash_attn":
                        raise ImportError("No module named 'flash_attn'")
                    return original_import(name, *args, **kwargs)
                
                with patch.object(builtins, "__import__", mock_import):
                    FlashAttentionManager.reset()
                    result = FlashAttentionManager.is_available()
        
        assert result is False
    
    def test_get_info_cached(self):
        """测试信息缓存"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        FlashAttentionManager._info = FlashAttentionInfo(
            available=True,
            version="2.3.0"
        )
        
        info = FlashAttentionManager.get_info()
        
        assert info.available is True
        assert info.version == "2.3.0"
    
    def test_get_attention_implementation_flash_available(self):
        """测试FlashAttention可用时的实现选择"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        impl = FlashAttentionManager.get_attention_implementation(prefer_flash=True)
        
        assert impl == "flash_attention_2"
    
    def test_get_attention_implementation_flash_unavailable(self):
        """测试FlashAttention不可用时的实现选择"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = False
        
        # 检查是否有SDPA
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            impl = FlashAttentionManager.get_attention_implementation(prefer_flash=True)
            assert impl == "sdpa"
        else:
            impl = FlashAttentionManager.get_attention_implementation(prefer_flash=True)
            assert impl == "eager"
    
    def test_get_attention_implementation_prefer_flash_false(self):
        """测试不优先使用FlashAttention"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        impl = FlashAttentionManager.get_attention_implementation(prefer_flash=False)
        
        # 应该返回SDPA或eager，而不是flash_attention_2
        assert impl in ["sdpa", "eager"]
    
    def test_configure_model_for_flash_attention_enabled(self):
        """测试配置模型启用FlashAttention"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        model_kwargs = {}
        result = FlashAttentionManager.configure_model_for_flash_attention(
            model_kwargs,
            enable_flash=True
        )
        
        assert result["attn_implementation"] == "flash_attention_2"
    
    def test_configure_model_for_flash_attention_disabled(self):
        """测试配置模型禁用FlashAttention"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        model_kwargs = {}
        result = FlashAttentionManager.configure_model_for_flash_attention(
            model_kwargs,
            enable_flash=False
        )
        
        assert "attn_implementation" not in result or result.get("attn_implementation") != "flash_attention_2"
    
    def test_reset(self):
        """测试重置状态"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        FlashAttentionManager._info = FlashAttentionInfo(available=True)
        
        FlashAttentionManager.reset()
        
        assert FlashAttentionManager._checked is False
        assert FlashAttentionManager._available is False
        assert FlashAttentionManager._info is None


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def setup_method(self):
        """每个测试前重置状态"""
        FlashAttentionManager.reset()
    
    def test_is_flash_attention_available(self):
        """测试is_flash_attention_available函数"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        result = is_flash_attention_available()
        assert result is True
    
    def test_get_flash_attention_info(self):
        """测试get_flash_attention_info函数"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._info = FlashAttentionInfo(
            available=True,
            version="2.3.0"
        )
        
        info = get_flash_attention_info()
        assert info.version == "2.3.0"
    
    def test_get_best_attention_implementation(self):
        """测试get_best_attention_implementation函数"""
        FlashAttentionManager._checked = True
        FlashAttentionManager._available = True
        
        impl = get_best_attention_implementation()
        assert impl == "flash_attention_2"
