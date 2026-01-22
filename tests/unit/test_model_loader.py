"""
模型加载器单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import torch

from mm_orch.runtime.model_loader import (
    ModelLoader,
    ModelConfig,
    LoadedModel,
)
from mm_orch.exceptions import ModelLoadError, OutOfMemoryError


class TestModelConfig:
    """ModelConfig测试"""
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        config = ModelConfig(model_name="gpt2")
        assert config.model_name == "gpt2"
        assert config.model_type == "auto"
        assert config.device == "auto"
        assert config.dtype == "auto"
        assert config.quantization is None
        assert config.trust_remote_code is False
    
    def test_create_qwen_config(self):
        """测试创建Qwen配置"""
        config = ModelConfig(
            model_name="Qwen/Qwen-7B-Chat",
            model_type="qwen-chat",
            device="cuda",
            dtype="bf16",
            trust_remote_code=True,
            flash_attention=True
        )
        assert config.model_name == "Qwen/Qwen-7B-Chat"
        assert config.model_type == "qwen-chat"
        assert config.trust_remote_code is True
        assert config.flash_attention is True
    
    def test_create_quantized_config(self):
        """测试创建量化配置"""
        config = ModelConfig(
            model_name="gpt2-medium",
            quantization="8bit"
        )
        assert config.quantization == "8bit"


class TestLoadedModel:
    """LoadedModel测试"""
    
    def test_create_loaded_model(self):
        """测试创建LoadedModel"""
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = ModelConfig(model_name="gpt2")
        
        loaded = LoadedModel(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
            device="cuda",
            dtype=torch.float16,
            memory_footprint=1024 * 1024 * 1024  # 1GB
        )
        
        assert loaded.model == mock_model
        assert loaded.tokenizer == mock_tokenizer
        assert loaded.device == "cuda"
        assert loaded.dtype == torch.float16
        assert loaded.memory_footprint == 1024 * 1024 * 1024


class TestModelLoader:
    """ModelLoader测试"""
    
    @patch("mm_orch.runtime.model_loader.HAS_TRANSFORMERS", True)
    def test_init_default(self):
        """测试默认初始化"""
        loader = ModelLoader()
        assert loader.config == {}
        assert loader.quantization_manager is not None
    
    @patch("mm_orch.runtime.model_loader.HAS_TRANSFORMERS", False)
    def test_init_no_transformers(self):
        """测试transformers不可用时的错误"""
        with pytest.raises(ModelLoadError) as exc_info:
            ModelLoader()
        assert "transformers" in str(exc_info.value).lower()
    
    def test_detect_device_auto_with_cuda(self):
        """测试自动设备检测（有CUDA）"""
        with patch("torch.cuda.is_available", return_value=True):
            device = ModelLoader.detect_device("auto")
            assert device == "cuda"
    
    def test_detect_device_auto_without_cuda(self):
        """测试自动设备检测（无CUDA）"""
        with patch("torch.cuda.is_available", return_value=False):
            device = ModelLoader.detect_device("auto")
            assert device == "cpu"
    
    def test_detect_device_cuda_available(self):
        """测试指定CUDA设备（可用）"""
        with patch("torch.cuda.is_available", return_value=True):
            device = ModelLoader.detect_device("cuda")
            assert device == "cuda"
    
    def test_detect_device_cuda_unavailable(self):
        """测试指定CUDA设备（不可用）"""
        with patch("torch.cuda.is_available", return_value=False):
            device = ModelLoader.detect_device("cuda")
            assert device == "cpu"
    
    def test_detect_device_specific_gpu(self):
        """测试指定特定GPU"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                device = ModelLoader.detect_device("cuda:1")
                assert device == "cuda:1"
    
    def test_detect_device_cpu(self):
        """测试指定CPU"""
        device = ModelLoader.detect_device("cpu")
        assert device == "cpu"
    
    def test_detect_dtype_auto_cpu(self):
        """测试自动数据类型（CPU）"""
        dtype = ModelLoader.detect_dtype("auto", "cpu")
        assert dtype == torch.float32
    
    def test_detect_dtype_auto_cuda_bf16(self):
        """测试自动数据类型（CUDA支持bf16）"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                dtype = ModelLoader.detect_dtype("auto", "cuda")
                assert dtype == torch.bfloat16
    
    def test_detect_dtype_auto_cuda_fp16(self):
        """测试自动数据类型（CUDA不支持bf16）"""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=False):
                dtype = ModelLoader.detect_dtype("auto", "cuda")
                assert dtype == torch.float16
    
    def test_detect_dtype_explicit(self):
        """测试显式指定数据类型"""
        assert ModelLoader.detect_dtype("fp32", "cuda") == torch.float32
        assert ModelLoader.detect_dtype("fp16", "cuda") == torch.float16
        assert ModelLoader.detect_dtype("bf16", "cuda") == torch.bfloat16
    
    def test_get_gpu_memory_info_no_cuda(self):
        """测试获取GPU内存信息（无CUDA）"""
        with patch("torch.cuda.is_available", return_value=False):
            info = ModelLoader.get_gpu_memory_info()
            assert info["total"] == 0
            assert info["allocated"] == 0
            assert info["free"] == 0
    
    @patch("mm_orch.runtime.model_loader.HAS_TRANSFORMERS", True)
    def test_load_tokenizer(self):
        """测试加载tokenizer"""
        with patch("mm_orch.runtime.model_loader.AutoTokenizer") as mock_tokenizer:
            mock_tok = Mock()
            mock_tok.pad_token = None
            mock_tok.eos_token = "<eos>"
            mock_tokenizer.from_pretrained.return_value = mock_tok
            
            loader = ModelLoader()
            tokenizer = loader.load_tokenizer("gpt2")
            
            mock_tokenizer.from_pretrained.assert_called_once()
            assert mock_tok.pad_token == "<eos>"


class TestModelLoaderIntegration:
    """ModelLoader集成测试（需要mock）"""
    
    @patch("mm_orch.runtime.model_loader.HAS_TRANSFORMERS", True)
    @patch("mm_orch.runtime.model_loader.AutoModelForCausalLM")
    @patch("mm_orch.runtime.model_loader.AutoTokenizer")
    def test_load_model_success(self, mock_tokenizer_cls, mock_model_cls):
        """测试成功加载模型"""
        # 设置mock
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.parameters.return_value = []
        mock_model.buffers.return_value = []
        mock_model_cls.from_pretrained.return_value = mock_model
        
        # 加载模型
        loader = ModelLoader()
        config = ModelConfig(model_name="gpt2", device="cpu")
        
        with patch.object(loader, "detect_device", return_value="cpu"):
            with patch.object(loader, "detect_dtype", return_value=torch.float32):
                loaded = loader.load_model(config)
        
        assert loaded.model == mock_model
        assert loaded.tokenizer == mock_tokenizer
        assert loaded.device == "cpu"
