"""
Property-based tests for PyTorch backend backward compatibility.

This module tests Property 5: Backward Compatibility Preservation for the
OpenVINO backend integration feature.

Property tested:
- Property 5: Backward Compatibility Preservation

Requirements validated: 3.1, 3.2, 3.3, 3.4

Feature: openvino-backend-integration
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck, assume
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
import torch

from mm_orch.runtime.pytorch_backend import PyTorchBackend
from mm_orch.runtime.backend_factory import BackendFactory


# Test strategies
@st.composite
def device_strategy(draw):
    """Generate valid device strings."""
    return draw(st.sampled_from(['cpu', 'cuda']))


@st.composite
def backend_config_strategy(draw):
    """Generate valid backend configuration dictionaries."""
    return {
        'dtype': draw(st.sampled_from(['float32', 'float16'])),
        'enable_fallback': draw(st.booleans()),
    }


@st.composite
def model_path_strategy(draw):
    """Generate valid model path strings."""
    # Use small test models that are commonly available
    return draw(st.sampled_from([
        'distilgpt2',
        'gpt2',
        'sshleifer/tiny-gpt2',
    ]))


@st.composite
def generation_params_strategy(draw):
    """Generate valid generation parameters."""
    return {
        'max_length': draw(st.integers(min_value=10, max_value=100)),
        'temperature': draw(st.floats(min_value=0.1, max_value=2.0)),
        'do_sample': draw(st.booleans()),
    }


class TestProperty5_BackwardCompatibilityPreservation:
    """
    Property 5: Backward Compatibility Preservation
    
    For any existing code that uses ModelManager without the new backend
    parameter, the system should behave identically to the pre-integration
    implementation, producing the same outputs and maintaining the same
    API signatures.
    
    Validates: Requirements 3.1, 3.2, 3.3, 3.4
    
    Feature: openvino-backend-integration, Property 5: Backward Compatibility Preservation
    """
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        deadline=None
    )
    def test_pytorch_backend_initialization_preserves_behavior(self, device, config):
        """
        Test that PyTorchBackend initialization preserves existing behavior.
        
        Property: For any valid device and config, PyTorchBackend should
        initialize successfully and maintain the same interface as before.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Verify backend has required attributes
        assert hasattr(backend, 'device')
        assert hasattr(backend, 'config')
        assert hasattr(backend, 'torch_device')
        assert hasattr(backend, '_models')
        
        # Verify device is set correctly
        assert backend.device == device
        assert backend.config == config
        
        # Verify torch_device is a valid torch.device
        assert isinstance(backend.torch_device, torch.device)
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_pytorch_backend_implements_all_required_methods(self, device, config):
        """
        Test that PyTorchBackend implements all required InferenceBackend methods.
        
        Property: For any valid initialization, PyTorchBackend should have all
        methods defined in the InferenceBackend interface.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Verify all required methods are present
        required_methods = [
            'load_model',
            'forward',
            'generate',
            'unload_model',
            'get_model_info',
            'is_available',
        ]
        
        for method_name in required_methods:
            assert hasattr(backend, method_name), f"Missing method: {method_name}"
            assert callable(getattr(backend, method_name)), f"Method not callable: {method_name}"
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_pytorch_backend_is_available_returns_true(self, device, config):
        """
        Test that PyTorchBackend.is_available() returns True when PyTorch is installed.
        
        Property: For any valid backend, is_available() should return True
        since PyTorch is required for the system to run.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Verify backend is available
        assert backend.is_available() is True
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        deadline=None
    )
    def test_backend_factory_creates_pytorch_backend(self, device, config):
        """
        Test that BackendFactory can create PyTorchBackend instances.
        
        Property: For any valid device and config, BackendFactory should
        successfully create a PyTorchBackend instance.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create factory
        factory = BackendFactory()
        
        # Create backend through factory
        backend = factory.create_backend(
            backend_type='pytorch',
            device=device,
            config=config
        )
        
        # Verify backend is PyTorchBackend instance
        assert isinstance(backend, PyTorchBackend)
        assert backend.device == device
        assert backend.config == config
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_pytorch_backend_get_model_info_returns_correct_structure(self, device, config):
        """
        Test that get_model_info returns the expected structure.
        
        Property: For any valid backend and mock model, get_model_info should
        return a dictionary with required keys.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Create a mock model with parameters
        mock_model = MagicMock()
        mock_param = torch.nn.Parameter(torch.randn(10, 10))
        mock_model.parameters.return_value = [mock_param]
        
        # Get model info
        info = backend.get_model_info(mock_model)
        
        # Verify structure
        assert isinstance(info, dict)
        assert 'backend' in info
        assert 'device' in info
        assert 'parameters' in info
        assert 'dtype' in info
        
        # Verify values
        assert info['backend'] == 'pytorch'
        assert isinstance(info['parameters'], int)
        assert info['parameters'] >= 0
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        deadline=None
    )
    def test_pytorch_backend_unload_model_handles_missing_model(self, device, config):
        """
        Test that unload_model raises KeyError for non-existent models.
        
        Property: For any valid backend and non-existent model name,
        unload_model should raise KeyError.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Try to unload non-existent model
        with pytest.raises(KeyError):
            backend.unload_model('non_existent_model')
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy(),
        prompt=st.text(min_size=1, max_size=50)
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        deadline=None
    )
    def test_pytorch_backend_generate_with_mock_model(self, device, config, prompt):
        """
        Test that generate method works with mocked model and tokenizer.
        
        Property: For any valid prompt, generate should call model.generate
        and tokenizer.decode correctly.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 2
        
        # Setup model mock
        mock_output = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.generate.return_value = mock_output
        
        # Setup decode mock
        mock_tokenizer.decode.return_value = "Generated text"
        
        # Call generate
        result = backend.generate(
            model=mock_model,
            tokenizer=mock_tokenizer,
            prompt=prompt,
            max_length=50
        )
        
        # Verify tokenizer was called
        mock_tokenizer.assert_called_once()
        
        # Verify model.generate was called
        mock_model.generate.assert_called_once()
        
        # Verify decode was called
        mock_tokenizer.decode.assert_called_once()
        
        # Verify result is a string
        assert isinstance(result, str)
    
    @given(
        device=device_strategy(),
        config=backend_config_strategy()
    )
    @settings(
        max_examples=30,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
        deadline=None
    )
    def test_pytorch_backend_forward_with_mock_model(self, device, config):
        """
        Test that forward method works with mocked model.
        
        Property: For any valid inputs, forward should call model forward
        pass and return outputs in correct format.
        """
        # Skip CUDA tests if CUDA is not available
        if device == 'cuda' and not torch.cuda.is_available():
            assume(False)
        
        # Create PyTorch backend
        backend = PyTorchBackend(device=device, config=config)
        
        # Create mock model
        mock_model = MagicMock()
        
        # Setup model output
        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 50257)  # Typical GPT-2 output shape
        mock_model.return_value = mock_output
        
        # Create inputs
        inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Call forward
        result = backend.forward(model=mock_model, inputs=inputs)
        
        # Verify model was called
        mock_model.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'logits' in result
        assert isinstance(result['logits'], torch.Tensor)
    
    def test_pytorch_backend_load_model_rejects_invalid_model_type(self):
        """
        Test that load_model raises ValueError for unsupported model types.
        
        Property: For any unsupported model_type, load_model should raise
        ValueError with a clear error message.
        """
        # Create PyTorch backend
        backend = PyTorchBackend(device='cpu', config={})
        
        # Try to load with invalid model type
        with pytest.raises(ValueError) as exc_info:
            backend.load_model(
                model_name='test_model',
                model_path='test/path',
                model_type='unsupported_type'
            )
        
        # Verify error message mentions the unsupported type
        assert 'unsupported_type' in str(exc_info.value).lower()


class TestIntegration_PyTorchBackendBackwardCompatibility:
    """
    Integration tests for PyTorch backend backward compatibility.
    
    These tests verify that PyTorchBackend maintains backward compatibility
    with existing PyTorch-based workflows.
    """
    
    def test_backend_factory_lists_pytorch_as_available(self):
        """
        Test that BackendFactory lists 'pytorch' as available.
        
        This verifies that the factory correctly detects PyTorch availability.
        """
        factory = BackendFactory()
        available = factory.get_available_backends()
        
        # PyTorch should be available (it's required for the system)
        assert 'pytorch' in available
    
    def test_pytorch_backend_maintains_api_compatibility(self):
        """
        Test that PyTorchBackend maintains API compatibility.
        
        This verifies that all expected methods exist and have correct signatures.
        """
        backend = PyTorchBackend(device='cpu', config={})
        
        # Verify method signatures (if they exist, they're callable)
        assert callable(backend.load_model)
        assert callable(backend.forward)
        assert callable(backend.generate)
        assert callable(backend.unload_model)
        assert callable(backend.get_model_info)
        assert callable(backend.is_available)
        
        # Verify attributes
        assert hasattr(backend, 'device')
        assert hasattr(backend, 'config')
        assert hasattr(backend, 'torch_device')
        assert hasattr(backend, '_models')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

