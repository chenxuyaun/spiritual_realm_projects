"""
Unit tests for BackendFactory.

Tests the factory's ability to create backend instances, validate parameters,
and detect available backends on the system.

Validates Requirements: 1.1, 1.3

Note: These tests focus on the BackendFactory's validation and detection logic.
Tests that require actual backend implementations will be added when those
backends are implemented in later tasks.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys

from mm_orch.runtime.backend_factory import BackendFactory
from mm_orch.runtime.inference_backend import InferenceBackend


class TestBackendFactoryValidation:
    """Test backend parameter validation."""
    
    def test_invalid_backend_name_raises_value_error(self):
        """Test that invalid backend names raise ValueError."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('invalid_backend', 'cpu', {})
        
        error_message = str(exc_info.value)
        assert 'Invalid backend type' in error_message
        assert 'invalid_backend' in error_message
    
    def test_empty_backend_name_raises_value_error(self):
        """Test that empty backend name raises ValueError."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError):
            factory.create_backend('', 'cpu', {})
    
    def test_none_backend_name_raises_error(self):
        """Test that None backend name raises appropriate error."""
        factory = BackendFactory()
        
        with pytest.raises((ValueError, TypeError, AttributeError)):
            factory.create_backend(None, 'cpu', {})

    def test_uppercase_backend_name_raises_value_error(self):
        """Test that uppercase backend names are rejected (case-sensitive)."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError):
            factory.create_backend('PYTORCH', 'cpu', {})
        
        with pytest.raises(ValueError):
            factory.create_backend('PyTorch', 'cpu', {})
    
    def test_error_message_includes_valid_backends(self):
        """Test that error messages mention valid backend options."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('tensorflow', 'cpu', {})
        
        error_message = str(exc_info.value)
        # Should mention valid options
        assert 'pytorch' in error_message.lower() or 'openvino' in error_message.lower()
    
    def test_error_message_includes_available_backends(self):
        """Test that error messages include available backends on system."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('invalid', 'cpu', {})
        
        error_message = str(exc_info.value)
        # Should mention available backends
        assert 'available' in error_message.lower() or 'supported' in error_message.lower()


class TestBackendAvailabilityDetection:
    """Test backend availability detection."""
    
    def test_get_available_backends_returns_list(self):
        """Test that get_available_backends returns a list."""
        factory = BackendFactory()
        available = factory.get_available_backends()
        
        assert isinstance(available, list)
    
    def test_get_available_backends_returns_valid_names(self):
        """Test that get_available_backends only returns valid backend names."""
        factory = BackendFactory()
        available = factory.get_available_backends()
        
        valid_backends = {'pytorch', 'openvino'}
        for backend in available:
            assert backend in valid_backends

    def test_pytorch_in_available_when_torch_installed(self):
        """Test that 'pytorch' is in available backends when torch is installed."""
        factory = BackendFactory()
        
        # Check if torch is actually installed
        try:
            import torch
            torch_installed = True
        except ImportError:
            torch_installed = False
        
        available = factory.get_available_backends()
        
        if torch_installed:
            assert 'pytorch' in available
        else:
            assert 'pytorch' not in available
    
    def test_openvino_in_available_when_openvino_installed(self):
        """Test that 'openvino' is in available backends when openvino is installed."""
        factory = BackendFactory()
        
        # Check if openvino is actually installed
        try:
            import openvino
            openvino_installed = True
        except ImportError:
            openvino_installed = False
        
        available = factory.get_available_backends()
        
        if openvino_installed:
            assert 'openvino' in available
        else:
            assert 'openvino' not in available
    
    def test_get_available_backends_is_consistent(self):
        """Test that get_available_backends returns consistent results."""
        factory = BackendFactory()
        
        # Call multiple times
        available1 = factory.get_available_backends()
        available2 = factory.get_available_backends()
        available3 = factory.get_available_backends()
        
        # Should return the same results
        assert available1 == available2 == available3
    
    def test_get_available_backends_no_duplicates(self):
        """Test that get_available_backends doesn't return duplicates."""
        factory = BackendFactory()
        available = factory.get_available_backends()
        
        # Should have no duplicates
        assert len(available) == len(set(available))



class TestBackendFactoryInstantiation:
    """Test BackendFactory instantiation."""
    
    def test_factory_can_be_instantiated(self):
        """Test that BackendFactory can be instantiated."""
        factory = BackendFactory()
        assert factory is not None
        assert isinstance(factory, BackendFactory)
    
    def test_factory_has_create_backend_method(self):
        """Test that factory has create_backend method."""
        factory = BackendFactory()
        assert hasattr(factory, 'create_backend')
        assert callable(factory.create_backend)
    
    def test_factory_has_get_available_backends_method(self):
        """Test that factory has get_available_backends method."""
        factory = BackendFactory()
        assert hasattr(factory, 'get_available_backends')
        assert callable(factory.get_available_backends)


class TestBackendFactoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_create_backend_with_special_characters_in_name(self):
        """Test that backend names with special characters are rejected."""
        factory = BackendFactory()
        
        invalid_names = [
            'pytorch!',
            'open@vino',
            'backend#1',
            'my-backend',
            'backend_name'
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                factory.create_backend(name, 'cpu', {})
    
    def test_create_backend_with_numeric_name(self):
        """Test that numeric backend names are rejected."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError):
            factory.create_backend('123', 'cpu', {})
    
    def test_create_backend_with_whitespace_name(self):
        """Test that backend names with whitespace are rejected."""
        factory = BackendFactory()
        
        invalid_names = [
            ' pytorch',
            'pytorch ',
            'py torch',
            '\tpytorch',
            'pytorch\n'
        ]
        
        for name in invalid_names:
            with pytest.raises(ValueError):
                factory.create_backend(name, 'cpu', {})

    def test_create_backend_accepts_various_device_values(self):
        """Test that various device values are accepted (validation is backend-specific)."""
        factory = BackendFactory()
        
        # These should not raise ValueError for invalid backend type
        # (they may raise other errors when trying to import backends)
        devices = ['cpu', 'cuda', 'GPU', 'CPU', 'AUTO']
        
        for device in devices:
            with pytest.raises((ValueError, RuntimeError, ImportError)) as exc_info:
                factory.create_backend('invalid_backend', device, {})
            
            # Should be ValueError for invalid backend, not device
            assert isinstance(exc_info.value, ValueError)
            assert 'Invalid backend type' in str(exc_info.value)
    
    def test_create_backend_accepts_various_config_values(self):
        """Test that various config values are accepted."""
        factory = BackendFactory()
        
        configs = [
            {},
            {'key': 'value'},
            {'enable_fallback': True},
            {'device': 'CPU', 'num_streams': 1}
        ]
        
        for config in configs:
            with pytest.raises((ValueError, RuntimeError, ImportError)) as exc_info:
                factory.create_backend('invalid_backend', 'cpu', config)
            
            # Should be ValueError for invalid backend, not config
            assert isinstance(exc_info.value, ValueError)
            assert 'Invalid backend type' in str(exc_info.value)


class TestBackendFactoryErrorMessages:
    """Test error message quality and content."""
    
    def test_error_message_is_descriptive(self):
        """Test that error messages are descriptive and helpful."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('tensorflow', 'cpu', {})
        
        error_message = str(exc_info.value)
        
        # Should include the invalid backend name
        assert 'tensorflow' in error_message
        
        # Should mention it's invalid
        assert 'invalid' in error_message.lower() or 'unknown' in error_message.lower()
        
        # Should mention valid options
        assert 'pytorch' in error_message.lower() or 'openvino' in error_message.lower()
    
    def test_error_message_suggests_alternatives(self):
        """Test that error messages suggest valid alternatives."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('torch', 'cpu', {})
        
        error_message = str(exc_info.value)
        
        # Should suggest the correct backend name
        assert 'pytorch' in error_message.lower()
    
    def test_error_message_includes_system_info(self):
        """Test that error messages include system-specific information."""
        factory = BackendFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('invalid', 'cpu', {})
        
        error_message = str(exc_info.value)
        
        # Should include available backends on this system
        available = factory.get_available_backends()
        # Error message should reference availability
        assert 'available' in error_message.lower() or 'supported' in error_message.lower()



class TestBackendFactoryIntegration:
    """Integration tests for BackendFactory."""
    
    def test_factory_workflow_with_invalid_backend(self):
        """Test complete workflow with invalid backend."""
        factory = BackendFactory()
        
        # Check available backends
        available = factory.get_available_backends()
        assert isinstance(available, list)
        
        # Try to create invalid backend
        with pytest.raises(ValueError) as exc_info:
            factory.create_backend('invalid', 'cpu', {})
        
        # Error should be descriptive
        assert 'Invalid backend type' in str(exc_info.value)
    
    def test_multiple_factory_instances_are_independent(self):
        """Test that multiple factory instances work independently."""
        factory1 = BackendFactory()
        factory2 = BackendFactory()
        
        # Both should work independently
        available1 = factory1.get_available_backends()
        available2 = factory2.get_available_backends()
        
        # Should return same results
        assert available1 == available2
        
        # Both should reject invalid backends
        with pytest.raises(ValueError):
            factory1.create_backend('invalid', 'cpu', {})
        
        with pytest.raises(ValueError):
            factory2.create_backend('invalid', 'cpu', {})
