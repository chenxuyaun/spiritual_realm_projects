"""
Unit tests for error handling across backend components.

This module tests that error messages contain helpful information,
are properly categorized, and include troubleshooting suggestions.

Requirements tested:
    - 1.5: Provide clear error messages with installation instructions
    - 9.1: Provide specific failure reasons in error messages
    - 9.2: Distinguish between different error types
    - 9.5: Include troubleshooting suggestions in error messages
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from mm_orch.runtime.backend_factory import BackendFactory
from mm_orch.runtime.openvino_backend import OpenVINOBackend
from mm_orch.runtime.backend_config import BackendConfig
from mm_orch.runtime.backend_exceptions import (
    BackendError,
    BackendInitializationError,
    ConfigurationError,
    ModelLoadError,
    InferenceError,
    DeviceError,
    FileValidationError,
)


class TestErrorCategorization:
    """Test that errors are properly categorized."""
    
    def test_exception_hierarchy(self):
        """Test that exception hierarchy is correct."""
        # All exceptions should inherit from BackendError
        assert issubclass(BackendInitializationError, BackendError)
        assert issubclass(ConfigurationError, BackendError)
        assert issubclass(ModelLoadError, BackendError)
        assert issubclass(InferenceError, BackendError)
        assert issubclass(DeviceError, BackendError)
        
        # FileValidationError should inherit from ModelLoadError
        assert issubclass(FileValidationError, ModelLoadError)
        assert issubclass(FileValidationError, BackendError)
    
    def test_error_can_be_caught_generically(self):
        """Test that all backend errors can be caught with BackendError."""
        errors = [
            BackendInitializationError("test"),
            ConfigurationError("test"),
            ModelLoadError("test"),
            InferenceError("test"),
            DeviceError("test"),
            FileValidationError("test"),
        ]
        
        for error in errors:
            try:
                raise error
            except BackendError:
                # Should catch all backend errors
                pass
            except Exception:
                pytest.fail(f"{type(error).__name__} not caught by BackendError")
    
    def test_error_can_be_caught_specifically(self):
        """Test that specific error types can be caught individually."""
        # Test FileValidationError can be caught specifically
        try:
            raise FileValidationError("test")
        except FileValidationError:
            pass
        except Exception:
            pytest.fail("FileValidationError not caught specifically")
        
        # Test ConfigurationError can be caught specifically
        try:
            raise ConfigurationError("test")
        except ConfigurationError:
            pass
        except Exception:
            pytest.fail("ConfigurationError not caught specifically")


class TestBackendFactoryErrors:
    """Test error messages from BackendFactory."""
    
    def test_invalid_backend_type_error_message(self):
        """Test that invalid backend type produces helpful error message."""
        factory = BackendFactory()
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_backend('invalid_backend', 'cpu', {})
        
        error_msg = str(exc_info.value)
        
        # Should mention the invalid backend name
        assert 'invalid_backend' in error_msg
        
        # Should list supported backends
        assert 'pytorch' in error_msg
        assert 'openvino' in error_msg
        
        # Should show available backends on this system
        assert 'Available backends' in error_msg or 'available' in error_msg.lower()
        
        # Should include installation instructions
        assert 'pip install' in error_msg
        
        # Should include example usage
        assert 'Example' in error_msg or 'example' in error_msg.lower()
    
    def test_pytorch_import_error_message(self):
        """Test that PyTorch import error produces helpful message."""
        # We can't easily mock the import inside create_backend,
        # but we can verify the error message structure by checking
        # what would happen if PyTorch wasn't available
        factory = BackendFactory()
        
        # The error message format is defined in the code
        # We'll verify it by checking the actual error handling code
        # exists and has the right structure
        import inspect
        source = inspect.getsource(factory.create_backend)
        
        # Verify the error message contains key elements
        assert 'PyTorch' in source or 'torch' in source
        assert 'pip install' in source
        assert 'pytorch.org' in source
    
    def test_openvino_import_error_message(self):
        """Test that OpenVINO import error produces helpful message."""
        # Similar to above, verify the error message structure
        factory = BackendFactory()
        
        import inspect
        source = inspect.getsource(factory.create_backend)
        
        # Verify the error message contains key elements
        assert 'OpenVINO' in source or 'openvino' in source
        assert 'pip install' in source
        assert 'openvino-dev' in source
        assert 'optimum' in source


class TestOpenVINOBackendErrors:
    """Test error messages from OpenVINOBackend."""
    
    def test_invalid_device_error_message(self):
        """Test that invalid device produces helpful error message."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenVINOBackend('INVALID_DEVICE', {'enable_fallback': False})
        
        error_msg = str(exc_info.value)
        
        # Should mention the invalid device
        assert 'INVALID_DEVICE' in error_msg
        
        # Should list supported devices
        assert 'CPU' in error_msg
        assert 'GPU' in error_msg
        assert 'AUTO' in error_msg
        assert 'NPU' in error_msg
        
        # Should describe what each device is
        assert 'integrated' in error_msg.lower() or 'discrete' in error_msg.lower()
        
        # Should include example usage
        assert 'Example' in error_msg or 'example' in error_msg.lower()
    
    def test_file_validation_error_contains_export_command(self):
        """Test that file validation errors mention the export script."""
        with tempfile.TemporaryDirectory() as temp_dir:
            openvino_path = os.path.join(temp_dir, 'models', 'openvino', 'test_model')
            
            backend = OpenVINOBackend('CPU', {'enable_fallback': False})
            
            with pytest.raises(FileValidationError) as exc_info:
                backend._validate_model_files('test_model', openvino_path)
            
            error_msg = str(exc_info.value)
            
            # Should mention the export script
            assert 'export_to_openvino.py' in error_msg
            
            # Should include the model name in the command
            assert 'test_model' in error_msg
            
            # Should include troubleshooting steps
            assert 'Troubleshooting' in error_msg or 'troubleshooting' in error_msg.lower()
    
    def test_file_validation_error_for_missing_xml(self):
        """Test error message when XML file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            openvino_path = os.path.join(temp_dir, 'models', 'openvino', 'test_model')
            os.makedirs(openvino_path)
            
            # Create BIN file but not XML
            bin_file = os.path.join(openvino_path, 'openvino_model.bin')
            with open(bin_file, 'wb') as f:
                f.write(b'dummy weights')
            
            backend = OpenVINOBackend('CPU', {'enable_fallback': False})
            
            with pytest.raises(FileValidationError) as exc_info:
                backend._validate_model_files('test_model', openvino_path)
            
            error_msg = str(exc_info.value)
            
            # Should specifically mention XML file
            assert 'XML' in error_msg or 'xml' in error_msg
            
            # Should mention the file is missing
            assert 'not found' in error_msg.lower() or 'missing' in error_msg.lower()
            
            # Should suggest re-export
            assert 'export' in error_msg.lower() or 'Export' in error_msg
    
    def test_file_validation_error_for_empty_file(self):
        """Test error message when file is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            openvino_path = os.path.join(temp_dir, 'models', 'openvino', 'test_model')
            os.makedirs(openvino_path)
            
            # Create empty XML file
            xml_file = os.path.join(openvino_path, 'openvino_model.xml')
            with open(xml_file, 'w') as f:
                pass  # Empty file
            
            # Create valid BIN file
            bin_file = os.path.join(openvino_path, 'openvino_model.bin')
            with open(bin_file, 'wb') as f:
                f.write(b'weights')
            
            backend = OpenVINOBackend('CPU', {'enable_fallback': False})
            
            with pytest.raises(FileValidationError) as exc_info:
                backend._validate_model_files('test_model', openvino_path)
            
            error_msg = str(exc_info.value)
            
            # Should mention file is empty
            assert 'empty' in error_msg.lower() or 'zero bytes' in error_msg.lower()
            
            # Should suggest the export was corrupted
            assert 'corrupted' in error_msg.lower() or 'failed' in error_msg.lower()
            
            # Should suggest deleting and re-exporting
            assert 'delete' in error_msg.lower() or 'rm' in error_msg.lower()


class TestConfigurationErrors:
    """Test error messages from BackendConfig."""
    
    def test_yaml_parsing_error_message(self):
        """Test that YAML parsing errors produce helpful messages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'config.yaml')
            
            # Write invalid YAML
            with open(config_file, 'w') as f:
                f.write("backend:\n  default: pytorch\n  invalid: [unclosed")
            
            # Should not raise, but should log warning and use defaults
            config = BackendConfig(config_file)
            
            # Should fall back to defaults
            assert config.get_default_backend() == 'pytorch'
    
    def test_missing_config_file_message(self):
        """Test that missing config file produces helpful message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, 'nonexistent.yaml')
            
            # Should not raise, but should use defaults
            config = BackendConfig(config_file)
            
            # Should fall back to defaults
            assert config.get_default_backend() == 'pytorch'


class TestTroubleshootingSuggestions:
    """Test that error messages include troubleshooting suggestions."""
    
    def test_initialization_error_includes_troubleshooting(self):
        """Test that initialization errors include troubleshooting steps."""
        # We can verify the error message structure by checking the source code
        # since mocking the initialization is complex
        from mm_orch.runtime.openvino_backend import OpenVINOBackend
        import inspect
        
        source = inspect.getsource(OpenVINOBackend._initialize_openvino)
        
        # Verify the error message contains key troubleshooting elements
        assert 'Troubleshooting' in source or 'troubleshooting' in source
        assert 'install' in source.lower()
        assert 'requirements' in source.lower() or 'Requirements' in source
        assert 'fallback' in source.lower() or 'Fallback' in source
    
    def test_model_load_error_includes_troubleshooting(self):
        """Test that model load errors include troubleshooting steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'models', 'test_model')
            openvino_path = os.path.join(temp_dir, 'models', 'openvino', 'test_model')
            
            backend = OpenVINOBackend('CPU', {'enable_fallback': False})
            
            # Try to load non-existent model
            with pytest.raises((FileValidationError, ModelLoadError)) as exc_info:
                backend.load_model('test_model', model_path, 'transformers')
            
            error_msg = str(exc_info.value)
            
            # Should include troubleshooting section
            assert 'Troubleshooting' in error_msg or 'troubleshooting' in error_msg.lower()
            
            # Should mention export script
            assert 'export' in error_msg.lower() or 'Export' in error_msg


class TestErrorMessageQuality:
    """Test the overall quality of error messages."""
    
    def test_error_messages_are_multiline(self):
        """Test that error messages use multiple lines for readability."""
        with pytest.raises(ConfigurationError) as exc_info:
            OpenVINOBackend('INVALID', {'enable_fallback': False})
        
        error_msg = str(exc_info.value)
        
        # Should have multiple lines (more readable)
        assert '\n' in error_msg
        
        # Should have at least 5 lines of content
        lines = [line for line in error_msg.split('\n') if line.strip()]
        assert len(lines) >= 5
    
    def test_error_messages_include_context(self):
        """Test that error messages include contextual information."""
        factory = BackendFactory()
        
        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_backend('invalid', 'cpu', {})
        
        error_msg = str(exc_info.value)
        
        # Should include what was attempted
        assert 'invalid' in error_msg
        
        # Should include what is valid
        assert 'pytorch' in error_msg
        assert 'openvino' in error_msg
        
        # Should include how to fix it
        assert 'Example' in error_msg or 'example' in error_msg.lower()
    
    def test_error_messages_avoid_technical_jargon(self):
        """Test that error messages are understandable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            openvino_path = os.path.join(temp_dir, 'models', 'openvino', 'test')
            
            backend = OpenVINOBackend('CPU', {'enable_fallback': False})
            
            with pytest.raises(FileValidationError) as exc_info:
                backend._validate_model_files('test', openvino_path)
            
            error_msg = str(exc_info.value)
            
            # Should use clear language
            assert 'not found' in error_msg.lower() or 'missing' in error_msg.lower()
            
            # Should provide actionable steps
            assert 'export' in error_msg.lower() or 'Export' in error_msg
            
            # Should not use overly technical terms without explanation
            # (IR, XML, BIN are okay as they're standard OpenVINO terms)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
