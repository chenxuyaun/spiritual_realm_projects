"""
Property-based tests for OpenVINO backend fallback disable behavior.

These tests verify that when fallback is disabled, the OpenVINO backend
raises the original error instead of falling back to PyTorch.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
import logging


# Feature: openvino-backend-integration, Property 7: Fallback Disable Behavior
# Validates: Requirement 4.5


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    enable_fallback=st.just(False),  # Fallback disabled for this property
)
def test_property_fallback_disabled_raises_original_error(device, enable_fallback):
    """
    Property 7: Fallback Disable Behavior
    
    For any operation that fails in the OpenVINO backend when fallback is disabled
    in configuration, the system should raise the original error without attempting
    PyTorch fallback.
    
    This test verifies that disabling fallback prevents automatic fallback to PyTorch.
    
    Validates: Requirement 4.5
    """
    config = {'enable_fallback': enable_fallback}
    
    # Patch to simulate OpenVINO not available
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not installed")):
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Property: Creating backend with fallback disabled should raise error
            # when OpenVINO is not available
            with pytest.raises(RuntimeError) as exc_info:
                backend = OpenVINOBackend(device=device, config=config)
            
            # Property: Error message should mention OpenVINO failure
            assert "openvino" in str(exc_info.value).lower(), \
                "Error should mention OpenVINO failure"
            
            # Property: PyTorch fallback should NOT be initialized
            # (we can't check this directly since the exception is raised,
            # but we verify the exception is raised which means fallback didn't happen)


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    model_path=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P'))),
    enable_fallback=st.just(False),
)
def test_property_fallback_disabled_on_model_load_failure(device, model_name, model_path, enable_fallback):
    """
    Property 7: Fallback Disable Behavior (Model Loading)
    
    When fallback is disabled and model loading fails, the system should raise
    the original error without attempting PyTorch fallback.
    
    Validates: Requirement 4.5
    """
    # Skip invalid inputs
    assume(len(model_name.strip()) > 0)
    assume(len(model_path.strip()) > 0)
    assume('/' not in model_name and '\\' not in model_name)
    
    config = {'enable_fallback': enable_fallback}
    
    # Create a mock OpenVINO manager that will succeed initialization
    # but fail on model loading
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
        mock_manager = Mock()
        mock_manager.load_model.side_effect = RuntimeError("Model load failed")
        mock_manager_class.return_value = mock_manager
        
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend (should succeed since OpenVINO manager is available)
            backend = OpenVINOBackend(device=device, config=config)
            
            # Verify fallback backend was NOT initialized
            assert backend._fallback_backend is None, \
                "Fallback backend should not be initialized when fallback is disabled"
            
            # Property: Model loading should raise error without fallback
            with pytest.raises((RuntimeError, FileNotFoundError)) as exc_info:
                with patch('os.path.exists', return_value=False):
                    backend.load_model(model_name, model_path, 'transformers')
            
            # Property: PyTorch fallback should NOT be called
            assert not mock_pytorch.load_model.called, \
                "PyTorch fallback should not be called when fallback is disabled"


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    enable_fallback=st.just(False),
)
def test_property_fallback_disabled_no_fallback_backend_created(device, enable_fallback):
    """
    Property 7: Fallback Disable Behavior (No Fallback Backend)
    
    When fallback is disabled, the system should not create a fallback backend
    instance, even if OpenVINO initialization succeeds.
    
    Validates: Requirement 4.5
    """
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend
            backend = OpenVINOBackend(device=device, config=config)
            
            # Property: Fallback backend should be None when disabled
            assert backend._fallback_backend is None, \
                "Fallback backend should be None when fallback is disabled"
            
            # Property: Fallback enabled flag should be False
            assert backend._fallback_enabled == False, \
                "Fallback enabled flag should be False"
            
            # Property: PyTorchBackend should not be instantiated
            assert not mock_pytorch_class.called, \
                "PyTorchBackend should not be instantiated when fallback is disabled"


@settings(max_examples=100, deadline=None)
@given(
    device=st.sampled_from(['CPU', 'GPU', 'AUTO']),
    enable_fallback=st.sampled_from([True, False]),
)
def test_property_fallback_config_respected(device, enable_fallback):
    """
    Property 7: Fallback Disable Behavior (Config Respected)
    
    The system should respect the enable_fallback configuration parameter,
    enabling or disabling fallback behavior accordingly.
    
    Validates: Requirement 4.5
    """
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
            mock_pytorch = Mock()
            mock_pytorch_class.return_value = mock_pytorch
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend
            backend = OpenVINOBackend(device=device, config=config)
            
            # Property: Fallback enabled flag should match configuration
            assert backend._fallback_enabled == enable_fallback, \
                f"Fallback enabled flag should be {enable_fallback}"
            
            # Property: Fallback backend existence should match configuration
            if enable_fallback:
                # When enabled, fallback backend may or may not be created
                # depending on OpenVINO availability
                pass  # No assertion needed
            else:
                # When disabled, fallback backend should never be created
                assert backend._fallback_backend is None, \
                    "Fallback backend should be None when fallback is disabled"
