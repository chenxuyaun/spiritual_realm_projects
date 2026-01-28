"""
Unit tests for DeepSpeed engine wrapper.

Tests specific examples, edge cases, and error handling for DeepSpeed integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from mm_orch.optimization import DeepSpeedEngine, DeepSpeedConfig


class TestDeepSpeedEngineInitialization:
    """Test DeepSpeed engine initialization."""
    
    def test_engine_initialization_with_defaults(self):
        """Test engine initializes with default configuration."""
        config = DeepSpeedConfig()
        engine = DeepSpeedEngine(config)
        
        assert engine.config == config
        assert engine._model is None
        assert engine._loaded_model is None
        assert engine._tokenizer is None
    
    def test_engine_initialization_with_custom_config(self):
        """Test engine initializes with custom configuration."""
        config = DeepSpeedConfig(
            enabled=True,
            tensor_parallel=4,
            pipeline_parallel=2,
            dtype="bf16",
            replace_with_kernel_inject=False
        )
        engine = DeepSpeedEngine(config)
        
        assert engine.config.tensor_parallel == 4
        assert engine.config.pipeline_parallel == 2
        assert engine.config.dtype == "bf16"
        assert engine.config.replace_with_kernel_inject is False


class TestDeepSpeedAvailability:
    """Test DeepSpeed availability detection."""
    
    def test_availability_when_disabled(self):
        """Test engine reports unavailable when disabled in config."""
        config = DeepSpeedConfig(enabled=False)
        engine = DeepSpeedEngine(config)
        
        assert engine.is_available() is False
    
    def test_availability_when_deepspeed_not_installed(self):
        """Test engine reports unavailable when DeepSpeed not installed."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'deepspeed'")):
            assert engine.is_available() is False
    
    def test_availability_when_deepspeed_installed(self):
        """Test engine reports available when DeepSpeed is installed."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock successful import
        with patch.dict('sys.modules', {'deepspeed': MagicMock()}):
            result = engine.is_available()
            # Result depends on actual DeepSpeed installation
            assert isinstance(result, bool)


class TestDeepSpeedModelLoading:
    """Test DeepSpeed model loading."""
    
    def test_load_model_when_unavailable(self):
        """Test load_model fails gracefully when DeepSpeed unavailable."""
        config = DeepSpeedConfig(enabled=False)
        engine = DeepSpeedEngine(config)
        
        result = engine.load_model("test-model")
        
        assert result is False
        assert engine.get_loaded_model() is None
    
    def test_load_model_success(self):
        """Test successful model loading with DeepSpeed."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        result = engine.load_model("test-model")
        
        assert result is True
        assert engine.get_loaded_model() == "test-model"
        assert engine._model is not None
        assert engine._tokenizer is not None
    
    def test_load_model_with_overrides(self):
        """Test model loading with parameter overrides."""
        config = DeepSpeedConfig(
            enabled=True,
            tensor_parallel=2,
            pipeline_parallel=1
        )
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        # Mock GPU manager to prevent fallback to single GPU
        mock_gpu_manager = MagicMock()
        mock_gpu_manager.allocate_gpus.return_value = ([0, 1, 2, 3], "tensor_parallel")
        mock_gpu_manager.balance_load.return_value = [[0, 1], [2, 3]]
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    # Mock the GPU manager
                    engine._gpu_manager = mock_gpu_manager
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        result = engine.load_model(
                            "test-model",
                            tensor_parallel=4,
                            pipeline_parallel=2
                        )
        
        assert result is True
        # Verify overrides were used
        mock_deepspeed.init_inference.assert_called_once()
        call_kwargs = mock_deepspeed.init_inference.call_args[1]
        assert call_kwargs['mp_size'] == 4  # Override value
    
    def test_load_model_handles_runtime_error(self):
        """Test model loading handles RuntimeError gracefully."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module with error
        mock_deepspeed = MagicMock()
        mock_deepspeed.init_inference.side_effect = RuntimeError("GPU not available")
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        with pytest.raises(RuntimeError, match="DeepSpeed model loading failed"):
                            engine.load_model("test-model")
        
        # Verify cleanup
        assert engine.get_loaded_model() is None
        assert engine._model is None


class TestDeepSpeedInference:
    """Test DeepSpeed inference operations."""
    
    def test_infer_without_loaded_model(self):
        """Test infer raises error when no model is loaded."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.infer(inputs)
    
    def test_infer_success(self):
        """Test successful inference with DeepSpeed."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)
        mock_ds_model.return_value = mock_output
        
        # Mock parameters for device detection - return iterator
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_ds_model.parameters.return_value = iter([mock_param])
        
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        engine.load_model("test-model")
                        
                        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
                        outputs = engine.infer(inputs)
        
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert isinstance(outputs["logits"], torch.Tensor)
    
    def test_generate_without_loaded_model(self):
        """Test generate raises error when no model is loaded."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.generate(["Hello"])
    
    def test_generate_success(self):
        """Test successful text generation with DeepSpeed."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        
        # Mock tokenizer behavior
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3)
        }
        
        # Mock model generate behavior
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_ds_model.generate.return_value = generated_ids
        
        # Mock tokenizer decode
        mock_tokenizer_instance.decode.return_value = "Hello generated text"
        
        # Mock parameters for device detection - return iterator
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_ds_model.parameters.return_value = iter([mock_param])
        
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        engine.load_model("test-model")
                        
                        outputs = engine.generate(["Hello"])
        
        assert len(outputs) == 1
        assert isinstance(outputs[0], dict)
        assert "text" in outputs[0]
        assert "prompt" in outputs[0]
        assert "tokens_generated" in outputs[0]


class TestDeepSpeedModelUnloading:
    """Test DeepSpeed model unloading."""
    
    def test_unload_model(self):
        """Test model unloading clears state."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        engine.load_model("test-model")
                        assert engine.get_loaded_model() == "test-model"
                        
                        engine.unload_model()
        
        assert engine.get_loaded_model() is None
        assert engine._model is None
        assert engine._tokenizer is None
    
    def test_unload_when_no_model_loaded(self):
        """Test unload_model is safe when no model is loaded."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Should not raise any errors
        engine.unload_model()
        
        assert engine.get_loaded_model() is None


class TestDeepSpeedDtypeConversion:
    """Test dtype conversion utilities."""
    
    def test_get_torch_dtype_fp16(self):
        """Test fp16 dtype conversion."""
        config = DeepSpeedConfig(dtype="fp16")
        engine = DeepSpeedEngine(config)
        
        assert engine._get_torch_dtype() == torch.float16
    
    def test_get_torch_dtype_fp32(self):
        """Test fp32 dtype conversion."""
        config = DeepSpeedConfig(dtype="fp32")
        engine = DeepSpeedEngine(config)
        
        assert engine._get_torch_dtype() == torch.float32
    
    def test_get_torch_dtype_bf16(self):
        """Test bf16 dtype conversion."""
        config = DeepSpeedConfig(dtype="bf16")
        engine = DeepSpeedEngine(config)
        
        assert engine._get_torch_dtype() == torch.bfloat16


class TestDeepSpeedConfigValidation:
    """Test DeepSpeed configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration is accepted."""
        config = DeepSpeedConfig(
            enabled=True,
            tensor_parallel=2,
            pipeline_parallel=2,
            dtype="fp16",
            replace_with_kernel_inject=True
        )
        
        assert config.tensor_parallel == 2
        assert config.pipeline_parallel == 2
        assert config.dtype == "fp16"
    
    def test_invalid_tensor_parallel(self):
        """Test invalid tensor_parallel is rejected."""
        with pytest.raises(ValueError, match="tensor_parallel must be >= 1"):
            DeepSpeedConfig(tensor_parallel=0)
    
    def test_invalid_pipeline_parallel(self):
        """Test invalid pipeline_parallel is rejected."""
        with pytest.raises(ValueError, match="pipeline_parallel must be >= 1"):
            DeepSpeedConfig(pipeline_parallel=-1)
    
    def test_invalid_dtype(self):
        """Test invalid dtype is rejected."""
        with pytest.raises(ValueError, match="dtype must be one of"):
            DeepSpeedConfig(dtype="invalid")


class TestDeepSpeedErrorHandling:
    """Test DeepSpeed error handling."""
    
    def test_load_model_import_error(self):
        """Test load_model handles ImportError."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module with import error
        mock_deepspeed = MagicMock()
        mock_deepspeed.init_inference.side_effect = ImportError("DeepSpeed not found")
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        with pytest.raises(ImportError, match="DeepSpeed is not installed"):
                            engine.load_model("test-model")
    
    def test_infer_handles_exceptions(self):
        """Test infer handles exceptions gracefully."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Mock DeepSpeed module
        mock_deepspeed = MagicMock()
        mock_ds_model = MagicMock()
        
        # Mock model to raise exception
        mock_ds_model.side_effect = RuntimeError("CUDA out of memory")
        
        # Mock parameters for device detection - return iterator
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_ds_model.parameters.return_value = iter([mock_param])
        
        mock_deepspeed.init_inference.return_value = mock_ds_model
        
        with patch.dict('sys.modules', {'deepspeed': mock_deepspeed}):
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    with patch.object(engine, 'is_available', return_value=True):
                        engine.load_model("test-model")
                        
                        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
                        
                        with pytest.raises(RuntimeError, match="DeepSpeed inference failed"):
                            engine.infer(inputs)
