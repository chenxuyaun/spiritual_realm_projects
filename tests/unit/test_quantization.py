"""
量化管理器单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import torch

from mm_orch.runtime.quantization import (
    QuantizationManager,
    QuantizationConfig,
)
from mm_orch.exceptions import QuantizationError


class TestQuantizationConfig:
    """QuantizationConfig测试"""
    
    def test_create_8bit_config(self):
        """测试创建8-bit配置"""
        config = QuantizationConfig(
            quant_type="8bit",
            llm_int8_threshold=6.0
        )
        assert config.quant_type == "8bit"
        assert config.llm_int8_threshold == 6.0
    
    def test_create_4bit_config(self):
        """测试创建4-bit配置"""
        config = QuantizationConfig(
            quant_type="4bit",
            compute_dtype=torch.float16,
            use_double_quant=True,
            quant_method="nf4"
        )
        assert config.quant_type == "4bit"
        assert config.compute_dtype == torch.float16
        assert config.use_double_quant is True
        assert config.quant_method == "nf4"


class TestQuantizationManager:
    """QuantizationManager测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        manager = QuantizationManager()
        assert manager.config == {}
    
    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {"default_quant_type": "8bit"}
        manager = QuantizationManager(config)
        assert manager.config == config
    
    def test_supported_quant_types(self):
        """测试支持的量化类型"""
        assert "8bit" in QuantizationManager.SUPPORTED_QUANT_TYPES
        assert "4bit" in QuantizationManager.SUPPORTED_QUANT_TYPES
        assert "gptq" in QuantizationManager.SUPPORTED_QUANT_TYPES
    
    @patch("mm_orch.runtime.quantization.HAS_TRANSFORMERS", True)
    def test_get_quantization_config_8bit(self):
        """测试获取8-bit量化配置"""
        with patch("mm_orch.runtime.quantization.BitsAndBytesConfig") as mock_config:
            mock_config.return_value = Mock()
            
            config = QuantizationManager.get_quantization_config("8bit")
            
            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["load_in_8bit"] is True
    
    @patch("mm_orch.runtime.quantization.HAS_TRANSFORMERS", True)
    def test_get_quantization_config_4bit(self):
        """测试获取4-bit量化配置"""
        with patch("mm_orch.runtime.quantization.BitsAndBytesConfig") as mock_config:
            mock_config.return_value = Mock()
            
            config = QuantizationManager.get_quantization_config("4bit")
            
            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["load_in_4bit"] is True
    
    @patch("mm_orch.runtime.quantization.HAS_TRANSFORMERS", False)
    def test_get_quantization_config_no_transformers(self):
        """测试transformers不可用时的错误"""
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationManager.get_quantization_config("8bit")
        
        assert "transformers" in str(exc_info.value).lower()
    
    def test_get_quantization_config_unsupported(self):
        """测试不支持的量化类型"""
        with patch("mm_orch.runtime.quantization.HAS_TRANSFORMERS", True):
            config = QuantizationManager.get_quantization_config("unsupported")
            assert config is None
    
    @patch("mm_orch.runtime.quantization.HAS_AUTO_GPTQ", False)
    def test_load_gptq_model_no_library(self):
        """测试auto-gptq不可用时的错误"""
        with pytest.raises(QuantizationError) as exc_info:
            QuantizationManager.load_gptq_model("test-model")
        
        assert "auto-gptq" in str(exc_info.value).lower()
    
    def test_is_quantization_available_8bit(self):
        """测试8-bit量化可用性检查"""
        # 这个测试依赖于bitsandbytes是否安装
        result = QuantizationManager.is_quantization_available("8bit")
        assert isinstance(result, bool)
    
    def test_is_quantization_available_gptq(self):
        """测试GPTQ量化可用性检查"""
        result = QuantizationManager.is_quantization_available("gptq")
        assert isinstance(result, bool)
    
    def test_get_recommended_quant_type_no_quant_needed(self):
        """测试不需要量化的情况"""
        manager = QuantizationManager()
        result = manager.get_recommended_quant_type(
            model_size_gb=4.0,
            available_memory_gb=16.0
        )
        assert result is None
    
    def test_get_recommended_quant_type_8bit(self):
        """测试推荐8-bit量化"""
        manager = QuantizationManager()
        result = manager.get_recommended_quant_type(
            model_size_gb=14.0,
            available_memory_gb=16.0
        )
        assert result == "8bit"
    
    def test_get_recommended_quant_type_4bit(self):
        """测试推荐4-bit量化"""
        manager = QuantizationManager()
        result = manager.get_recommended_quant_type(
            model_size_gb=14.0,
            available_memory_gb=8.0
        )
        assert result == "4bit"
    
    def test_estimate_memory_usage_no_quant(self):
        """测试无量化内存估算"""
        manager = QuantizationManager()
        result = manager.estimate_memory_usage(14.0, None)
        assert result == 14.0
    
    def test_estimate_memory_usage_8bit(self):
        """测试8-bit量化内存估算"""
        manager = QuantizationManager()
        result = manager.estimate_memory_usage(14.0, "8bit")
        assert result == 7.0  # 14 * 0.5
    
    def test_estimate_memory_usage_4bit(self):
        """测试4-bit量化内存估算"""
        manager = QuantizationManager()
        result = manager.estimate_memory_usage(14.0, "4bit")
        assert result == 3.5  # 14 * 0.25
