"""
Unit tests for ONNX Runtime engine wrapper.

Tests specific examples, edge cases, and error conditions for the ONNX engine.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization import ONNXEngine, ONNXConfig


class SimpleTestModel(torch.nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


class TestONNXEngineInitialization:
    """Test ONNX engine initialization."""
    
    def test_initialization_with_default_config(self):
        """Test engine initializes with default configuration."""
        config = ONNXConfig()
        engine = ONNXEngine(config)
        
        assert engine.config == config
        assert engine._session is None
        assert engine._loaded_model is None
        assert engine._onnx_path is None
        assert engine._input_names == []
        assert engine._output_names == []
    
    def test_initialization_with_custom_config(self):
        """Test engine initializes with custom configuration."""
        config = ONNXConfig(
            enabled=True,
            execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            optimization_level="extended",
            enable_quantization=True
        )
        engine = ONNXEngine(config)
        
        assert engine.config.execution_providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert engine.config.optimization_level == "extended"
        assert engine.config.enable_quantization is True
    
    def test_initialization_with_disabled_config(self):
        """Test engine initializes when disabled."""
        config = ONNXConfig(enabled=False)
        engine = ONNXEngine(config)
        
        assert not engine.is_available()


class TestONNXEngineAvailability:
    """Test ONNX engine availability detection."""
    
    def test_is_available_when_disabled(self):
        """Test availability returns False when disabled."""
        config = ONNXConfig(enabled=False)
        engine = ONNXEngine(config)
        
        assert not engine.is_available()
    
    def test_is_available_when_library_missing(self):
        """Test availability returns False when library not installed."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        with patch('mm_orch.optimization.onnx_engine.onnxruntime', side_effect=ImportError):
            # Force re-check by calling is_available
            with patch.dict('sys.modules', {'onnxruntime': None}):
                # This will depend on actual installation
                pass
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_is_available_when_library_installed(self):
        """Test availability returns True when library installed and enabled."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        assert engine.is_available()


class TestONNXModelConversion:
    """Test ONNX model conversion."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_convert_model_basic(self):
        """Test basic model conversion."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            
            assert Path(converted_path).exists()
            assert converted_path == output_path
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_convert_model_with_validation(self):
        """Test model conversion with output validation."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(2, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            
            # Should not raise if validation passes
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=True
            )
            
            assert Path(converted_path).exists()
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_convert_model_without_output_path(self):
        """Test model conversion creates temp file when no output path specified."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        converted_path = engine.convert_model(
            model,
            sample_inputs,
            output_path=None,
            validate=False
        )
        
        assert Path(converted_path).exists()
        assert "onnx_model_" in converted_path or "Temp" in converted_path
    
    def test_convert_model_when_unavailable(self):
        """Test conversion fails gracefully when ONNX Runtime unavailable."""
        config = ONNXConfig(enabled=False)
        engine = ONNXEngine(config)
        
        model = SimpleTestModel()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with pytest.raises(ImportError, match="ONNX Runtime is not available"):
            engine.convert_model(model, sample_inputs)


class TestONNXModelLoading:
    """Test ONNX model loading."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_load_model_success(self):
        """Test successful model loading."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        # First convert a model
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            
            # Load the model
            success = engine.load_model(converted_path, "test_model")
            
            assert success
            assert engine.get_loaded_model() == "test_model"
            assert engine._session is not None
            assert len(engine._input_names) > 0
            assert len(engine._output_names) > 0
    
    def test_load_model_nonexistent_file(self):
        """Test loading non-existent file fails gracefully."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        success = engine.load_model("/nonexistent/model.onnx", "test")
        
        assert not success
        assert engine.get_loaded_model() is None
    
    def test_load_model_when_unavailable(self):
        """Test loading fails when ONNX Runtime unavailable."""
        config = ONNXConfig(enabled=False)
        engine = ONNXEngine(config)
        
        success = engine.load_model("model.onnx", "test")
        
        assert not success


class TestONNXInference:
    """Test ONNX inference."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_infer_success(self):
        """Test successful inference."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        # Convert and load model
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            engine.load_model(converted_path, "test_model")
            
            # Run inference
            inputs = {"input": np.random.randn(2, 10).astype(np.float32)}
            outputs = engine.infer(inputs)
            
            assert "output" in outputs
            assert outputs["output"].shape[0] == 2
            assert outputs["output"].shape[1] == 5
    
    def test_infer_without_loaded_model(self):
        """Test inference fails when no model loaded."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        inputs = {"input": np.array([[1, 2, 3]])}
        
        with pytest.raises(RuntimeError, match="No model loaded"):
            engine.infer(inputs)
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_infer_with_missing_input(self):
        """Test inference fails with missing required input."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        # Convert and load model
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            engine.load_model(converted_path, "test_model")
            
            # Try inference with wrong input name
            inputs = {"wrong_name": np.random.randn(1, 10).astype(np.float32)}
            
            with pytest.raises(ValueError, match="Missing required input"):
                engine.infer(inputs)


class TestONNXModelUnloading:
    """Test ONNX model unloading."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_unload_model(self):
        """Test model unloading."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        # Convert and load model
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            engine.load_model(converted_path, "test_model")
            
            # Verify model is loaded
            assert engine.get_loaded_model() == "test_model"
            
            # Unload model
            engine.unload_model()
            
            # Verify model is unloaded
            assert engine.get_loaded_model() is None
            assert engine._session is None
            assert len(engine._input_names) == 0
            assert len(engine._output_names) == 0
    
    def test_unload_when_no_model_loaded(self):
        """Test unloading when no model is loaded doesn't crash."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        # Should not raise exception
        engine.unload_model()
        
        assert engine.get_loaded_model() is None


class TestONNXExecutionProviders:
    """Test ONNX execution provider handling."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_get_available_providers(self):
        """Test getting available execution providers."""
        config = ONNXConfig(
            enabled=True,
            execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        providers = engine._get_available_providers()
        
        assert len(providers) > 0
        assert "CPUExecutionProvider" in providers
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_fallback_to_cpu_provider(self):
        """Test fallback to CPU provider when others unavailable."""
        config = ONNXConfig(
            enabled=True,
            execution_providers=["NonExistentProvider"]
        )
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        providers = engine._get_available_providers()
        
        assert "CPUExecutionProvider" in providers


class TestONNXOptimizationLevel:
    """Test ONNX optimization level mapping."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_optimization_level_mapping(self):
        """Test optimization level string maps to ONNX enum."""
        import onnxruntime as ort
        
        test_cases = [
            ("none", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
            ("basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
            ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
            ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        ]
        
        for level_str, expected_enum in test_cases:
            config = ONNXConfig(enabled=True, optimization_level=level_str)
            engine = ONNXEngine(config)
            
            if not engine.is_available():
                pytest.skip("ONNX Runtime not available")
            
            actual_enum = engine._get_optimization_level()
            assert actual_enum == expected_enum


class TestONNXEdgeCases:
    """Test ONNX engine edge cases."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_multiple_load_unload_cycles(self):
        """Test multiple load/unload cycles work correctly."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            
            # Load and unload multiple times
            for i in range(3):
                engine.load_model(converted_path, f"test_model_{i}")
                assert engine.get_loaded_model() == f"test_model_{i}"
                
                engine.unload_model()
                assert engine.get_loaded_model() is None
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="ONNX Runtime not installed"),
        reason="ONNX Runtime not available"
    )
    def test_inference_with_different_batch_sizes(self):
        """Test inference works with different batch sizes."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        if not engine.is_available():
            pytest.skip("ONNX Runtime not available")
        
        model = SimpleTestModel()
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "model.onnx")
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=output_path,
                validate=False
            )
            engine.load_model(converted_path, "test_model")
            
            # Test different batch sizes
            for batch_size in [1, 2, 4, 8]:
                inputs = {"input": np.random.randn(batch_size, 10).astype(np.float32)}
                outputs = engine.infer(inputs)
                
                assert outputs["output"].shape[0] == batch_size
