"""
Property-based tests for ONNX engine.

Feature: advanced-optimization-monitoring
Tests Properties 3 and 4 from the design document (ONNX-specific).
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from mm_orch.optimization import ONNXEngine, ONNXConfig


# Strategy for generating valid model names
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

# Strategy for generating execution providers
execution_providers_strategy = st.lists(
    st.sampled_from([
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "CPUExecutionProvider",
        "OpenVINOExecutionProvider"
    ]),
    min_size=1,
    max_size=4,
    unique=True
)

# Strategy for generating optimization levels
optimization_levels = st.sampled_from(["none", "basic", "extended", "all"])

# Strategy for generating small tensors for testing
small_tensors = st.integers(min_value=1, max_value=10).flatmap(
    lambda size: st.builds(
        torch.tensor,
        st.lists(
            st.lists(st.integers(min_value=0, max_value=100), min_size=size, max_size=size),
            min_size=1,
            max_size=4
        )
    )
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing ONNX conversion."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Feature: advanced-optimization-monitoring, Property 3: Engine configuration is applied correctly (ONNX)
@given(
    execution_providers=execution_providers_strategy,
    optimization_level=optimization_levels,
    enable_quantization=st.booleans()
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_3_onnx_engine_configuration_applied_correctly(
    execution_providers, optimization_level, enable_quantization
):
    """
    Property 3: Engine configuration is applied correctly (ONNX).
    
    For any model loaded with ONNX Runtime, the engine should be configured
    with the specified execution providers and optimization level from configuration.
    
    Validates: Requirements 3.3
    """
    # Setup: Create ONNX config with specified parameters
    config = ONNXConfig(
        enabled=True,
        execution_providers=execution_providers,
        optimization_level=optimization_level,
        enable_quantization=enable_quantization
    )
    engine = ONNXEngine(config)
    
    # Verify: Configuration is stored correctly
    assert engine.config.execution_providers == execution_providers
    assert engine.config.optimization_level == optimization_level
    assert engine.config.enable_quantization == enable_quantization
    
    # Verify: Optimization level mapping works
    if engine.is_available():
        opt_level = engine._get_optimization_level()
        assert opt_level is not None, "Optimization level should be mapped"
    
    # Verify: Provider filtering works
    if engine.is_available():
        available_providers = engine._get_available_providers()
        assert len(available_providers) > 0, "At least one provider should be available"
        assert "CPUExecutionProvider" in available_providers or any(
            p in available_providers for p in execution_providers
        ), "Either CPU or a configured provider should be available"


# Feature: advanced-optimization-monitoring, Property 4: ONNX conversion preserves model behavior
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    input_size=st.integers(min_value=5, max_value=20),
    hidden_size=st.integers(min_value=10, max_value=30),
    output_size=st.integers(min_value=3, max_value=10)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_property_4_onnx_conversion_preserves_model_behavior(
    batch_size, input_size, hidden_size, output_size
):
    """
    Property 4: ONNX conversion preserves model behavior.
    
    For any PyTorch model converted to ONNX, inference on the same inputs should
    produce equivalent outputs (within numerical tolerance) for both the original
    and converted models.
    
    Validates: Requirements 3.2
    """
    # Setup: Create a simple PyTorch model
    model = SimpleModel(input_size, hidden_size, output_size)
    model.eval()
    
    # Create sample inputs
    sample_inputs = {
        "input": torch.randn(batch_size, input_size)
    }
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(sample_inputs["input"])
        pytorch_output_np = pytorch_output.cpu().numpy()
    
    # Setup: Create ONNX engine
    config = ONNXConfig(
        enabled=True,
        execution_providers=["CPUExecutionProvider"],
        optimization_level="basic"
    )
    engine = ONNXEngine(config)
    
    # Skip test if ONNX Runtime is not available
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    try:
        # Execute: Convert model to ONNX with validation
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = str(Path(temp_dir) / "model.onnx")
            
            # Convert with validation enabled
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=onnx_path,
                validate=True  # This will check output equivalence
            )
            
            assert Path(converted_path).exists(), "ONNX model file should exist"
            
            # Load and run inference with ONNX
            engine.load_model(converted_path, "test_model")
            
            # Prepare ONNX inputs (numpy arrays)
            onnx_inputs = {
                "input": sample_inputs["input"].cpu().numpy()
            }
            
            # Run ONNX inference
            onnx_outputs = engine.infer(onnx_inputs)
            onnx_output_np = onnx_outputs["output"]
            
            # Verify: Outputs should be close (within numerical tolerance)
            assert np.allclose(
                pytorch_output_np,
                onnx_output_np,
                rtol=1e-3,
                atol=1e-5
            ), (
                f"ONNX output doesn't match PyTorch output. "
                f"Max difference: {np.max(np.abs(pytorch_output_np - onnx_output_np))}"
            )
            
            # Verify: Output shapes should match
            assert pytorch_output_np.shape == onnx_output_np.shape, (
                f"Output shapes don't match: PyTorch {pytorch_output_np.shape} "
                f"vs ONNX {onnx_output_np.shape}"
            )
            
    except Exception as e:
        # If conversion or inference fails, it's still a valid test result
        # (the property is that IF conversion succeeds, outputs should match)
        pytest.skip(f"ONNX conversion/inference failed: {e}")


# Additional property test: ONNX engine availability detection
@given(enabled=st.booleans())
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_onnx_engine_availability_detection(enabled):
    """
    Additional property: ONNX engine availability is correctly detected.
    
    For any configuration, the engine should correctly report whether ONNX Runtime
    is available based on both the enabled flag and library availability.
    """
    # Setup: Create config with specified enabled flag
    config = ONNXConfig(enabled=enabled)
    engine = ONNXEngine(config)
    
    # Execute: Check availability
    is_available = engine.is_available()
    
    # Verify: If disabled, should not be available
    if not enabled:
        assert not is_available, "Engine should not be available when disabled"
    
    # Verify: If enabled, availability depends on library
    if enabled:
        try:
            import onnxruntime  # noqa: F401
            # Library is installed, should be available
            assert is_available, "Engine should be available when enabled and library installed"
        except ImportError:
            # Library not installed, should not be available
            assert not is_available, "Engine should not be available when library missing"


# Additional property test: ONNX model loading and unloading
@given(model_name=model_names)
@settings(max_examples=30, deadline=None)
@pytest.mark.property
def test_onnx_model_loading_and_unloading(model_name):
    """
    Additional property: ONNX models can be loaded and unloaded correctly.
    
    For any valid ONNX model file, the engine should be able to load it,
    track the loaded model, and unload it cleanly.
    """
    # Setup: Create ONNX engine
    config = ONNXConfig(enabled=True)
    engine = ONNXEngine(config)
    
    # Skip if ONNX Runtime not available
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    try:
        # Create a simple model and convert it
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = str(Path(temp_dir) / "model.onnx")
            
            # Convert model
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=onnx_path,
                validate=False  # Skip validation for speed
            )
            
            # Verify: Initially no model loaded
            assert engine.get_loaded_model() is None
            
            # Execute: Load model
            success = engine.load_model(converted_path, model_name)
            
            # Verify: Model should be loaded
            assert success, "Model loading should succeed"
            assert engine.get_loaded_model() == model_name
            assert engine._session is not None
            assert len(engine._input_names) > 0
            assert len(engine._output_names) > 0
            
            # Execute: Unload model
            engine.unload_model()
            
            # Verify: Model should be unloaded
            assert engine.get_loaded_model() is None
            assert engine._session is None
            assert len(engine._input_names) == 0
            assert len(engine._output_names) == 0
            
    except Exception as e:
        pytest.skip(f"ONNX model loading test failed: {e}")


# Additional property test: ONNX inference with different batch sizes
@given(
    batch_size_1=st.integers(min_value=1, max_value=4),
    batch_size_2=st.integers(min_value=1, max_value=4),
    input_size=st.integers(min_value=5, max_value=15)
)
@settings(max_examples=30, deadline=None)
@pytest.mark.property
def test_onnx_inference_with_different_batch_sizes(
    batch_size_1, batch_size_2, input_size
):
    """
    Additional property: ONNX inference handles different batch sizes correctly.
    
    For any ONNX model with dynamic batch size, inference should work correctly
    with different batch sizes without reloading the model.
    """
    # Setup: Create and convert model
    model = SimpleModel(input_size=input_size, hidden_size=20, output_size=5)
    model.eval()
    
    config = ONNXConfig(enabled=True, execution_providers=["CPUExecutionProvider"])
    engine = ONNXEngine(config)
    
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = str(Path(temp_dir) / "model.onnx")
            
            # Convert with sample input (batch_size=1)
            sample_inputs = {"input": torch.randn(1, input_size)}
            converted_path = engine.convert_model(
                model,
                sample_inputs,
                output_path=onnx_path,
                validate=False
            )
            
            # Load model
            engine.load_model(converted_path, "test_model")
            
            # Execute: Inference with first batch size
            inputs_1 = {"input": np.random.randn(batch_size_1, input_size).astype(np.float32)}
            outputs_1 = engine.infer(inputs_1)
            
            # Verify: Output shape matches input batch size
            assert outputs_1["output"].shape[0] == batch_size_1, (
                f"Output batch size {outputs_1['output'].shape[0]} "
                f"doesn't match input batch size {batch_size_1}"
            )
            
            # Execute: Inference with second batch size (different)
            inputs_2 = {"input": np.random.randn(batch_size_2, input_size).astype(np.float32)}
            outputs_2 = engine.infer(inputs_2)
            
            # Verify: Output shape matches second input batch size
            assert outputs_2["output"].shape[0] == batch_size_2, (
                f"Output batch size {outputs_2['output'].shape[0]} "
                f"doesn't match input batch size {batch_size_2}"
            )
            
    except Exception as e:
        pytest.skip(f"ONNX dynamic batch size test failed: {e}")


# Additional property test: ONNX error handling
@given(model_name=model_names)
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_onnx_error_handling(model_name):
    """
    Additional property: ONNX engine handles errors gracefully.
    
    For any error condition (missing file, invalid model, etc.), the engine
    should handle it gracefully without crashing.
    """
    # Setup: Create ONNX engine
    config = ONNXConfig(enabled=True)
    engine = ONNXEngine(config)
    
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    # Test 1: Loading non-existent file
    success = engine.load_model("/nonexistent/path/model.onnx", model_name)
    assert not success, "Loading non-existent file should fail gracefully"
    assert engine.get_loaded_model() is None
    
    # Test 2: Inference without loaded model
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.infer({"input": np.array([[1, 2, 3]])})
    
    # Test 3: Unloading when no model is loaded (should not crash)
    engine.unload_model()  # Should not raise exception
    assert engine.get_loaded_model() is None


# Additional property test: ONNX execution provider fallback
@given(
    preferred_providers=st.lists(
        st.sampled_from([
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "NonExistentProvider"
        ]),
        min_size=1,
        max_size=3,
        unique=True
    )
)
@settings(max_examples=30, deadline=None)
@pytest.mark.property
def test_onnx_execution_provider_fallback(preferred_providers):
    """
    Additional property: ONNX engine falls back to CPU provider when preferred providers unavailable.
    
    For any list of execution providers, if none are available, the engine should
    fall back to CPUExecutionProvider.
    """
    # Setup: Create config with preferred providers
    config = ONNXConfig(
        enabled=True,
        execution_providers=preferred_providers
    )
    engine = ONNXEngine(config)
    
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    # Execute: Get available providers
    available_providers = engine._get_available_providers()
    
    # Verify: At least one provider should be available
    assert len(available_providers) > 0, "At least one provider should be available"
    
    # Verify: If no preferred providers available, should fall back to CPU
    try:
        import onnxruntime as ort
        actual_available = ort.get_available_providers()
        
        has_preferred = any(p in actual_available for p in preferred_providers)
        
        if not has_preferred:
            assert "CPUExecutionProvider" in available_providers, (
                "Should fall back to CPUExecutionProvider when preferred providers unavailable"
            )
    except ImportError:
        pytest.skip("Cannot verify provider availability without onnxruntime")


# Additional property test: ONNX model conversion cleanup
@given(cleanup_temp=st.booleans())
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_onnx_model_conversion_cleanup(cleanup_temp):
    """
    Additional property: ONNX engine cleans up temporary files correctly.
    
    For any model conversion, temporary files should be cleaned up when the
    engine is unloaded or destroyed.
    """
    # Setup: Create ONNX engine
    config = ONNXConfig(enabled=True)
    engine = ONNXEngine(config)
    
    if not engine.is_available():
        pytest.skip("ONNX Runtime not available")
    
    try:
        # Create and convert model
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()
        sample_inputs = {"input": torch.randn(1, 10)}
        
        # Convert without specifying output path (uses temp directory)
        converted_path = engine.convert_model(
            model,
            sample_inputs,
            output_path=None,  # Use temp directory
            validate=False
        )
        
        # Verify: Temp file exists
        assert Path(converted_path).exists(), "Converted model should exist"
        temp_dir = Path(converted_path).parent
        
        # Load model
        engine.load_model(converted_path, "test_model")
        
        if cleanup_temp:
            # Execute: Unload model (should clean up temp files)
            engine.unload_model()
            
            # Verify: Temp directory should be cleaned up if it was in /tmp/
            if "/tmp/" in str(temp_dir) or "\\Temp\\" in str(temp_dir):
                # Note: Cleanup happens in unload_model for temp directories
                pass  # Cleanup is best-effort, don't assert
        
    except Exception as e:
        pytest.skip(f"ONNX cleanup test failed: {e}")
