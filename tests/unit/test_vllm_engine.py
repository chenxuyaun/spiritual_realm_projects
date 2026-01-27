"""
Unit tests for vLLM engine error handling.

Feature: advanced-optimization-monitoring
Tests error handling, initialization failures, and fallback behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization import VLLMEngine, VLLMConfig


class TestVLLMEngineInitialization:
    """Test vLLM engine initialization and configuration validation."""
    
    def test_initialization_with_valid_config(self):
        """Test engine initializes correctly with valid configuration."""
        config = VLLMConfig(
            enabled=True,
            tensor_parallel_size=2,
            dtype="fp16"
        )
        engine = VLLMEngine(config)
        
        assert engine.config == config
        assert engine._llm is None
        assert engine._loaded_model is None
    
    def test_initialization_with_disabled_config(self):
        """Test engine initializes but reports unavailable when disabled."""
        config = VLLMConfig(enabled=False)
        engine = VLLMEngine(config)
        
        assert engine.is_available() is False
    
    def test_invalid_tensor_parallel_size_rejected(self):
        """Test that invalid tensor_parallel_size raises ValueError."""
        with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
            VLLMConfig(tensor_parallel_size=0)
        
        with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
            VLLMConfig(tensor_parallel_size=-1)
    
    def test_invalid_gpu_memory_utilization_rejected(self):
        """Test that invalid gpu_memory_utilization raises ValueError."""
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0.0 and 1.0"):
            VLLMConfig(gpu_memory_utilization=-0.1)
        
        with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0.0 and 1.0"):
            VLLMConfig(gpu_memory_utilization=1.5)
    
    def test_invalid_dtype_rejected(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be one of"):
            VLLMConfig(dtype="invalid_dtype")


class TestVLLMEngineAvailability:
    """Test vLLM availability detection and error handling."""
    
    def test_availability_when_vllm_not_installed(self):
        """Test availability check returns False when vLLM is not installed."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock ImportError when trying to import vLLM
        with patch('builtins.__import__', side_effect=ImportError("No module named 'vllm'")):
            assert engine.is_available() is False
    
    def test_availability_when_vllm_import_fails(self):
        """Test availability check handles vLLM import errors gracefully."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock exception during vLLM import
        with patch('builtins.__import__', side_effect=RuntimeError("CUDA error")):
            assert engine.is_available() is False
    
    def test_availability_when_disabled_in_config(self):
        """Test availability returns False when disabled in configuration."""
        config = VLLMConfig(enabled=False)
        engine = VLLMEngine(config)
        
        # Should return False without attempting import
        assert engine.is_available() is False


class TestVLLMEngineModelLoading:
    """Test model loading error handling and fallback."""
    
    def test_load_model_when_vllm_unavailable(self):
        """Test load_model returns False when vLLM is unavailable."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock vLLM as unavailable
        with patch.object(engine, 'is_available', return_value=False):
            result = engine.load_model("test-model")
            
            assert result is False
            assert engine._llm is None
            assert engine._loaded_model is None
    
    def test_load_model_import_error_raises(self):
        """Test load_model raises ImportError when vLLM import fails."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock vLLM as available but import fails during load_model
        with patch.object(engine, 'is_available', return_value=True):
            # Patch the import inside load_model
            with patch('builtins.__import__', side_effect=ImportError("vLLM not found")):
                with pytest.raises(ImportError, match="vLLM is not installed"):
                    engine.load_model("test-model")
    
    def test_load_model_runtime_error_raises(self):
        """Test load_model raises RuntimeError on model loading failure."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock vLLM as available but model loading fails
        with patch.object(engine, 'is_available', return_value=True):
            with patch('vllm.LLM', side_effect=RuntimeError("CUDA out of memory")):
                with pytest.raises(RuntimeError, match="vLLM model loading failed"):
                    engine.load_model("test-model")
                
                # Verify cleanup occurred
                assert engine._llm is None
                assert engine._loaded_model is None
    
    def test_load_model_unexpected_error_returns_false(self):
        """Test load_model returns False on unexpected errors."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock vLLM as available but unexpected error occurs
        with patch.object(engine, 'is_available', return_value=True):
            with patch('vllm.LLM', side_effect=ValueError("Invalid config")):
                result = engine.load_model("test-model")
                
                assert result is False
                assert engine._llm is None
                assert engine._loaded_model is None
    
    def test_load_model_with_overrides(self):
        """Test load_model applies parameter overrides correctly."""
        config = VLLMConfig(
            enabled=True,
            tensor_parallel_size=1,
            dtype="fp32"
        )
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock vLLM as available
        with patch.object(engine, 'is_available', return_value=True):
            with patch('mm_orch.optimization.vllm_engine.LLM') as mock_llm:
                mock_llm.return_value = MagicMock()
                
                # Load with overrides
                engine.load_model(
                    "test-model",
                    tensor_parallel_size=4,
                    dtype="fp16"
                )
                
                # Verify overrides were used
                call_kwargs = mock_llm.call_args[1]
                assert call_kwargs['tensor_parallel_size'] == 4
                assert call_kwargs['dtype'] == "fp16"


class TestVLLMEngineGeneration:
    """Test generation error handling and fallback."""
    
    def test_generate_without_loaded_model_raises(self):
        """Test generate raises RuntimeError when no model is loaded."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.generate(["test prompt"])
    
    def test_generate_with_vllm_error_raises(self):
        """Test generate raises RuntimeError on vLLM generation failure."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock loaded model
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        
        # Mock generation failure
        engine._llm.generate.side_effect = RuntimeError("CUDA error during generation")
        
        with pytest.raises(RuntimeError, match="vLLM generation failed"):
            engine.generate(["test prompt"])
    
    def test_generate_with_empty_prompts(self):
        """Test generate handles empty prompt list."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock loaded model
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        engine._llm.generate.return_value = []
        
        outputs = engine.generate([])
        assert outputs == []
    
    def test_generate_with_default_sampling_params(self):
        """Test generate uses default sampling params when none provided."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Skip if vLLM not installed
        try:
            import vllm  # noqa: F401
        except ImportError:
            pytest.skip("vLLM not installed")
        
        # Mock loaded model and vLLM output
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        
        mock_output = MagicMock()
        mock_output.prompt = "test prompt"
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].text = "generated text"
        mock_output.outputs[0].finish_reason = "stop"
        mock_output.outputs[0].token_ids = [1, 2, 3]
        
        engine._llm.generate.return_value = [mock_output]
        
        with patch('mm_orch.optimization.vllm_engine.SamplingParams') as mock_params:
            mock_params.return_value = MagicMock()
            
            outputs = engine.generate(["test prompt"])
            
            # Verify default params were created
            mock_params.assert_called_once()
            assert len(outputs) == 1
            assert outputs[0]["text"] == "generated text"


class TestVLLMEngineUnload:
    """Test model unloading and resource cleanup."""
    
    def test_unload_model_clears_state(self):
        """Test unload_model clears model state correctly."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock loaded model
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        
        # Unload
        engine.unload_model()
        
        assert engine._llm is None
        assert engine._loaded_model is None
    
    def test_unload_model_when_no_model_loaded(self):
        """Test unload_model handles case when no model is loaded."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Should not raise error
        engine.unload_model()
        
        assert engine._llm is None
        assert engine._loaded_model is None
    
    def test_get_loaded_model_returns_none_after_unload(self):
        """Test get_loaded_model returns None after unload."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock loaded model
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        
        assert engine.get_loaded_model() == "test-model"
        
        engine.unload_model()
        
        assert engine.get_loaded_model() is None


class TestVLLMEngineDestructor:
    """Test engine cleanup on destruction."""
    
    def test_destructor_unloads_model(self):
        """Test __del__ unloads model when engine is destroyed."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Mock loaded model
        engine._llm = MagicMock()
        engine._loaded_model = "test-model"
        
        # Mock unload_model to verify it's called
        with patch.object(engine, 'unload_model') as mock_unload:
            engine.__del__()
            mock_unload.assert_called_once()
    
    def test_destructor_handles_no_model(self):
        """Test __del__ handles case when no model is loaded."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Should not raise error
        engine.__del__()
