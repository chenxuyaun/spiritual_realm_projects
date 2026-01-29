"""
Property-based tests for OpenVINO backend fallback behavior.

These tests verify that the OpenVINO backend correctly falls back to PyTorch
when operations fail, and that fallback can be disabled when needed.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, patch, MagicMock
import logging


# Feature: openvino-backend-integration, Property 6: Automatic Fallback on Failure
# Validates: Requirements 4.1, 4.2, 4.3, 4.4


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    model_path=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P'))),
    enable_fallback=st.just(True),  # Fallback enabled for this property
)
def test_property_automatic_fallback_on_model_load_failure(device, model_name, model_path, enable_fallback):
    """
    Property 6: Automatic Fallback on Failure
    
    For any operation (model loading or inference) that fails in the OpenVINO backend
    when fallback is enabled, the system should automatically retry the operation using
    the PyTorch backend and log the fallback event.
    
    This test verifies model loading fallback behavior.
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4
    """
    # Skip invalid inputs
    assume(len(model_name.strip()) > 0)
    assume(len(model_path.strip()) > 0)
    assume('/' not in model_name and '\\' not in model_name)
    
    # Create backend with fallback enabled
    config = {'enable_fallback': enable_fallback}
    
    # Patch the imports inside _initialize_openvino
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not available")):
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            # Mock PyTorch backend to simulate successful fallback
            mock_pytorch = Mock()
            mock_pytorch_model = Mock()
            mock_pytorch.load_model.return_value = mock_pytorch_model
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend (should initialize fallback)
            backend = OpenVINOBackend(device=device, config=config)
            
            # Verify fallback backend was initialized
            assert backend._fallback_backend is not None, \
                "Fallback backend should be initialized when OpenVINO fails"
            
            # Attempt to load model (should use fallback)
            result = backend.load_model(model_name, model_path, 'transformers')
            
            # Verify fallback was used
            # Property: Fallback backend should be called when OpenVINO fails
            assert mock_pytorch.load_model.called, \
                "Fallback backend should be called when OpenVINO fails"
            
            # Property: Result should be from fallback backend
            assert result == mock_pytorch_model, \
                "Result should be from fallback backend"


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    prompt=st.text(min_size=1, max_size=100),
    max_length=st.integers(min_value=10, max_value=200),
    enable_fallback=st.just(True),
)
def test_property_automatic_fallback_on_inference_failure(device, prompt, max_length, enable_fallback):
    """
    Property 6: Automatic Fallback on Failure (Inference)
    
    For any inference operation that fails in the OpenVINO backend when fallback
    is enabled, the system should automatically retry using PyTorch backend.
    
    This test verifies inference fallback behavior.
    
    Validates: Requirements 4.1, 4.2, 4.3, 4.4
    """
    # Skip invalid inputs
    assume(len(prompt.strip()) > 0)
    
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not available")):
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            # Mock PyTorch backend
            mock_pytorch = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
            mock_tokenizer.decode.return_value = "Generated text"
            
            mock_pytorch.generate.return_value = "Generated text"
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend
            backend = OpenVINOBackend(device=device, config=config)
            
            # Create a mock model
            mock_model = Mock()
            
            # Generate should use fallback
            result = backend.generate(mock_model, mock_tokenizer, prompt, max_length)
            
            # Property: Fallback should be called
            assert mock_pytorch.generate.called, \
                "Fallback backend should be called when OpenVINO is not available"
            
            # Property: Result should be from fallback
            assert result == "Generated text", \
                "Should return result from fallback backend"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    enable_fallback=st.just(True),
)
def test_property_fallback_logging(device, enable_fallback, caplog):
    """
    Property 6: Automatic Fallback on Failure (Logging)
    
    When fallback occurs, the system should log a warning with the failure reason.
    
    This test verifies that fallback events are properly logged.
    
    Validates: Requirement 4.3
    """
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not installed")):
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend
            with caplog.at_level(logging.WARNING):
                backend = OpenVINOBackend(device=device, config=config)
                
                # Property: Warning should be logged when OpenVINO initialization fails
                warning_logs = [record for record in caplog.records if record.levelname == 'WARNING']
                
                # At least one warning should mention the failure
                has_failure_warning = any(
                    'failed' in record.message.lower() or 'openvino' in record.message.lower()
                    for record in warning_logs
                )
                
                assert has_failure_warning, \
                    "Should log warning when OpenVINO initialization fails"


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    enable_fallback=st.just(True),
)
def test_property_fallback_preserves_functionality(device, enable_fallback):
    """
    Property 6: Automatic Fallback on Failure (Functionality Preservation)
    
    When fallback occurs, the system should still provide the same functionality
    as if the operation had succeeded with OpenVINO.
    
    This test verifies that fallback preserves the expected API behavior.
    
    Validates: Requirement 4.4
    """
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO unavailable")):
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_model = Mock()
            mock_pytorch.load_model.return_value = mock_model
            mock_pytorch.get_model_info.return_value = {
                'backend': 'pytorch',
                'device': 'cpu',
                'parameters': 1000,
            }
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend
            backend = OpenVINOBackend(device=device, config=config)
            
            # Load model (will use fallback)
            model = backend.load_model('test_model', 'test/path', 'transformers')
            
            # Property: Model should be loaded successfully via fallback
            assert model is not None, "Model should be loaded via fallback"
            assert model == mock_model, "Should return the fallback model"
            
            # Property: get_model_info should work on fallback model
            with patch.object(backend, '_is_fallback_model', return_value=True):
                info = backend.get_model_info(model)
                assert 'backend' in info, "Model info should contain backend field"
                assert info['is_fallback'] == True, "Should indicate this is a fallback model"

