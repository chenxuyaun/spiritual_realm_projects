"""
Property-based tests for OpenVINO backend device fallback behavior.

These tests verify that the OpenVINO backend correctly falls back to CPU
when the requested device is unavailable.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from unittest.mock import Mock, patch, MagicMock
import logging


# Feature: openvino-backend-integration, Property 9: Device Fallback on Unavailability
# Validates: Requirements 6.3


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    requested_device=st.sampled_from(['GPU', 'NPU']),  # Devices that might not be available
    enable_fallback=st.booleans(),
)
def test_property_device_fallback_to_cpu_on_unavailability(requested_device, enable_fallback):
    """
    Property 9: Device Fallback on Unavailability
    
    For any requested OpenVINO device that is unavailable on the system,
    the backend should fall back to CPU device with a warning logged.
    
    This test verifies that when a device (GPU, NPU) is not available,
    the backend automatically falls back to CPU and logs an appropriate warning.
    
    Validates: Requirements 6.3
    """
    config = {'enable_fallback': enable_fallback}
    
    # Mock OpenVINO to simulate device unavailability
    with patch('mm_orch.runtime.openvino_backend.logger') as mock_logger:
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            # Mock the OpenVINO Core to report no GPU/NPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                # Only CPU is available
                mock_core.available_devices = ['CPU']
                mock_core_class.return_value = mock_core
                
                # Mock the manager initialization
                mock_manager = Mock()
                mock_manager_class.return_value = mock_manager
                
                # Import after patching
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                # Create backend with unavailable device
                backend = OpenVINOBackend(device=requested_device, config=config)
                
                # Property 1: Device should be set to CPU after fallback
                assert backend.device == 'CPU', \
                    f"Device should fall back to CPU when {requested_device} is unavailable, got {backend.device}"
                
                # Property 2: Warning should be logged about device fallback
                warning_logged = any(
                    call[0][0].startswith(f"Device '{requested_device}' is not available")
                    for call in mock_logger.warning.call_args_list
                )
                assert warning_logged, \
                    f"Warning should be logged when device {requested_device} is unavailable"
                
                # Property 3: OpenVINO manager should be initialized with CPU device
                if mock_manager_class.called:
                    call_kwargs = mock_manager_class.call_args[1]
                    assert call_kwargs['default_device'] == 'CPU', \
                        "OpenVINO manager should be initialized with CPU device after fallback"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    device=st.sampled_from(['CPU', 'AUTO']),  # Devices that are always available
    enable_fallback=st.booleans(),
)
def test_property_no_device_fallback_for_available_devices(device, enable_fallback):
    """
    Property 9: Device Fallback on Unavailability (No Fallback Case)
    
    For devices that are always available (CPU, AUTO), no device fallback
    should occur, and the requested device should be used.
    
    This test verifies that CPU and AUTO devices don't trigger fallback.
    
    Validates: Requirements 6.3
    """
    config = {'enable_fallback': enable_fallback}
    
    with patch('mm_orch.runtime.openvino_backend.logger') as mock_logger:
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            # Mock the manager initialization
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend with always-available device
            backend = OpenVINOBackend(device=device, config=config)
            
            # Property 1: Device should remain as requested (no fallback)
            assert backend.device == device, \
                f"Device should remain {device} for always-available devices, got {backend.device}"
            
            # Property 2: No device fallback warning should be logged
            device_fallback_warnings = [
                call for call in mock_logger.warning.call_args_list
                if call[0][0].startswith(f"Device '{device}' is not available")
            ]
            assert len(device_fallback_warnings) == 0, \
                f"No device fallback warning should be logged for {device}"
            
            # Property 3: OpenVINO manager should be initialized with requested device
            if mock_manager_class.called:
                call_kwargs = mock_manager_class.call_args[1]
                assert call_kwargs['default_device'] == device, \
                    f"OpenVINO manager should be initialized with requested device {device}"


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    requested_device=st.sampled_from(['GPU', 'NPU']),
    model_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    model_path=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'P'))),
)
def test_property_device_fallback_preserves_functionality(requested_device, model_name, model_path):
    """
    Property 9: Device Fallback on Unavailability (Functionality Preservation)
    
    When device fallback occurs, the backend should still function correctly
    with the fallback device (CPU), and model operations should succeed.
    
    This test verifies that device fallback doesn't break functionality.
    
    Validates: Requirements 6.3
    """
    # Skip invalid inputs
    assume(len(model_name.strip()) > 0)
    assume(len(model_path.strip()) > 0)
    assume('/' not in model_name and '\\' not in model_name)
    
    config = {'enable_fallback': True}
    
    with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
        # Mock OpenVINO Core to simulate device unavailability
        with patch('openvino.Core') as mock_core_class:
            mock_core = Mock()
            mock_core.available_devices = ['CPU']  # Only CPU available
            mock_core_class.return_value = mock_core
            
            # Mock the manager
            mock_manager = Mock()
            mock_model = Mock()
            mock_cached = Mock()
            mock_cached.model = mock_model
            mock_manager.load_model.return_value = mock_cached
            mock_manager_class.return_value = mock_manager
            
            # Import after patching
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Create backend with unavailable device (should fallback to CPU)
            backend = OpenVINOBackend(device=requested_device, config=config)
            
            # Property 1: Backend should be functional after device fallback
            assert backend.is_available(), \
                "Backend should be available after device fallback to CPU"
            
            # Property 2: Device should be CPU after fallback
            assert backend.device == 'CPU', \
                f"Device should be CPU after fallback from {requested_device}"
            
            # Property 3: Model loading should work with fallback device
            # Mock the necessary components for model loading
            with patch('os.path.exists', return_value=True):
                with patch('mm_orch.schemas.ModelConfig'):
                    try:
                        # This should not raise an error
                        result = backend.load_model(model_name, model_path, 'transformers')
                        # If we get here, functionality is preserved
                        functionality_preserved = True
                        # Verify we got a model back
                        assert result is not None, "Model loading should return a model object"
                    except Exception as e:
                        # Device fallback should not break basic functionality
                        functionality_preserved = False
                        # Log the error for debugging
                        print(f"Model loading failed after device fallback: {e}")
                    
                    assert functionality_preserved, \
                        f"Model loading should work after device fallback to CPU"


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    invalid_device=st.text(min_size=1, max_size=20).filter(
        lambda x: x.upper() not in ['CPU', 'GPU', 'AUTO', 'NPU']
    ),
)
def test_property_invalid_device_raises_error(invalid_device):
    """
    Property 9: Device Fallback on Unavailability (Invalid Device Validation)
    
    For any device parameter that is not a valid OpenVINO device,
    the backend should raise a ValueError during initialization.
    
    This test verifies that invalid device names are rejected early.
    
    Validates: Requirements 6.1
    """
    # Skip empty or whitespace-only strings
    assume(len(invalid_device.strip()) > 0)
    
    config = {'enable_fallback': True}
    
    # Import the backend class
    from mm_orch.runtime.openvino_backend import OpenVINOBackend
    
    # Property: Invalid device should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        OpenVINOBackend(device=invalid_device, config=config)
    
    # Property: Error message should mention the invalid device
    error_message = str(exc_info.value)
    assert invalid_device in error_message or invalid_device.upper() in error_message, \
        f"Error message should mention the invalid device '{invalid_device}'"
    
    # Property: Error message should list supported devices
    assert 'CPU' in error_message and 'GPU' in error_message, \
        "Error message should list supported devices"
