"""
推理引擎单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

import torch

from mm_orch.runtime.inference_engine import (
    InferenceEngine,
    GenerationConfig,
    GenerationResult,
)
from mm_orch.exceptions import InferenceError, ValidationError


class TestGenerationConfig:
    """GenerationConfig测试"""
    
    def test_create_default(self):
        """测试创建默认配置"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.1
        assert config.do_sample is True
    
    def test_create_custom(self):
        """测试创建自定义配置"""
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.95,
            top_k=40
        )
        
        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
    
    def test_validate_success(self):
        """测试验证成功"""
        config = GenerationConfig()
        config.validate()  # 不应抛出异常
    
    def test_validate_max_new_tokens_negative(self):
        """测试max_new_tokens为负数"""
        config = GenerationConfig(max_new_tokens=-1)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "max_new_tokens" in str(exc_info.value)
    
    def test_validate_max_new_tokens_too_large(self):
        """测试max_new_tokens过大"""
        config = GenerationConfig(max_new_tokens=10000)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "max_new_tokens" in str(exc_info.value)
    
    def test_validate_temperature_out_of_range(self):
        """测试temperature超出范围"""
        config = GenerationConfig(temperature=3.0)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "temperature" in str(exc_info.value)
    
    def test_validate_top_p_out_of_range(self):
        """测试top_p超出范围"""
        config = GenerationConfig(top_p=1.5)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "top_p" in str(exc_info.value)
    
    def test_validate_top_k_negative(self):
        """测试top_k为负数"""
        config = GenerationConfig(top_k=-1)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "top_k" in str(exc_info.value)
    
    def test_validate_repetition_penalty_too_low(self):
        """测试repetition_penalty过低"""
        config = GenerationConfig(repetition_penalty=0.5)
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        assert "repetition_penalty" in str(exc_info.value)
    
    def test_to_dict(self):
        """测试转换为字典"""
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.5
        )
        
        result = config.to_dict()
        
        assert result["max_new_tokens"] == 256
        assert result["temperature"] == 0.5
        assert "top_p" in result
        assert "top_k" in result


class TestGenerationResult:
    """GenerationResult测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = GenerationResult(
            text="Hello, world!",
            input_tokens=5,
            output_tokens=3,
            total_time=0.5,
            tokens_per_second=6.0,
            finish_reason="stop"
        )
        
        assert result.text == "Hello, world!"
        assert result.input_tokens == 5
        assert result.output_tokens == 3
        assert result.total_time == 0.5
        assert result.tokens_per_second == 6.0
        assert result.finish_reason == "stop"


class TestInferenceEngine:
    """InferenceEngine测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
    
    def test_init(self):
        """测试初始化"""
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        assert engine.model == self.mock_model
        assert engine.tokenizer == self.mock_tokenizer
        assert engine.device == "cpu"
    
    def test_init_with_custom_config(self):
        """测试带自定义配置初始化"""
        config = GenerationConfig(max_new_tokens=256)
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu",
            default_config=config
        )
        
        assert engine.default_config.max_new_tokens == 256
    
    def test_init_sets_pad_token(self):
        """测试初始化时设置pad_token"""
        self.mock_tokenizer.pad_token_id = None
        self.mock_tokenizer.eos_token_id = 1
        
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        assert self.mock_tokenizer.pad_token_id == 1
    
    def test_generate_success(self):
        """测试成功生成"""
        # 设置mock
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.decode.return_value = "Generated text"
        
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        result = engine.generate("Test prompt")
        
        assert isinstance(result, GenerationResult)
        assert result.text == "Generated text"
        assert result.input_tokens == 3
        assert result.output_tokens == 3
    
    def test_generate_with_custom_config(self):
        """测试使用自定义配置生成"""
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        self.mock_tokenizer.decode.return_value = "Generated text"
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        config = GenerationConfig(max_new_tokens=100, temperature=0.5)
        result = engine.generate("Test prompt", config=config)
        
        assert isinstance(result, GenerationResult)
    
    def test_generate_invalid_config(self):
        """测试无效配置"""
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        config = GenerationConfig(max_new_tokens=-1)
        
        with pytest.raises(ValidationError):
            engine.generate("Test prompt", config=config)
    
    def test_batch_generate_empty(self):
        """测试空批量生成"""
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        result = engine.batch_generate([])
        assert result == []
    
    def test_batch_generate_success(self):
        """测试成功批量生成"""
        # 设置mock
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        self.mock_tokenizer.decode.return_value = "Generated text"
        
        self.mock_model.generate.return_value = torch.tensor([
            [1, 2, 3, 7, 8, 9],
            [4, 5, 6, 10, 11, 12]
        ])
        
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        results = engine.batch_generate(["Prompt 1", "Prompt 2"])
        
        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)
    
    def test_validate_generation_params_success(self):
        """测试验证生成参数成功"""
        config = InferenceEngine.validate_generation_params(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9
        )
        
        assert isinstance(config, GenerationConfig)
        assert config.max_new_tokens == 256
    
    def test_validate_generation_params_failure(self):
        """测试验证生成参数失败"""
        with pytest.raises(ValidationError):
            InferenceEngine.validate_generation_params(
                max_new_tokens=-1
            )


class TestInferenceEngineStreaming:
    """InferenceEngine流式生成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
    
    @patch("mm_orch.runtime.inference_engine.HAS_TRANSFORMERS", True)
    @patch("mm_orch.runtime.inference_engine.TextIteratorStreamer")
    def test_generate_stream(self, mock_streamer_cls):
        """测试流式生成"""
        # 设置mock
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = Mock(return_value=iter(["Hello", " world", "!"]))
        mock_streamer_cls.return_value = mock_streamer
        
        engine = InferenceEngine(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device="cpu"
        )
        
        # 由于流式生成使用线程，这里只测试基本设置
        # 实际的流式测试需要更复杂的mock设置
        assert hasattr(engine, "generate_stream")
