"""
Unit tests for OpenVINO backend implementation.

These tests verify specific behaviors and edge cases of the OpenVINO backend,
including initialization, model loading, inference, and fallback scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestOpenVINOBackendInitialization:
    """Test OpenVINO backend initialization scenarios."""
    
    def test_initialization_with_openvino_available(self):
        """Test successful initialization when OpenVINO is available."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            assert backend._openvino_manager is not None
            assert backend.device == 'CPU'
            assert backend._fallback_enabled == True
    
    def test_initialization_with_openvino_unavailable_fallback_enabled(self):
        """Test initialization when OpenVINO is unavailable but fallback is enabled."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                assert backend._openvino_manager is None
                assert backend._fallback_backend is not None
                assert backend._fallback_enabled == True
    
    def test_initialization_with_openvino_unavailable_fallback_disabled(self):
        """Test initialization fails when OpenVINO is unavailable and fallback is disabled."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            from mm_orch.runtime.backend_exceptions import BackendInitializationError
            
            with pytest.raises(BackendInitializationError) as exc_info:
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': False})
            
            assert "openvino" in str(exc_info.value).lower()
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration parameters."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            config = {
                'enable_fallback': False,
                'cache_dir': 'custom/cache',
                'num_streams': 2
            }
            
            backend = OpenVINOBackend(device='GPU', config=config)
            
            assert backend.device == 'GPU'
            assert backend._fallback_enabled == False
            assert backend.config['cache_dir'] == 'custom/cache'


class TestOpenVINOBackendModelLoading:
    """Test model loading scenarios."""
    
    def test_load_model_with_valid_openvino_files(self):
        """Test loading model when OpenVINO files exist."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_model = Mock()
            mock_cached = Mock()
            mock_cached.model = mock_model
            mock_manager.load_model.return_value = mock_cached
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            with patch('os.path.exists', return_value=True):
                with patch('mm_orch.schemas.ModelConfig'):
                    model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
            
            assert model == mock_model
            assert 'gpt2' in backend._models
    
    def test_load_model_with_missing_openvino_files_fallback_enabled(self):
        """Test loading model when OpenVINO files are missing but fallback is enabled."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_model = Mock()
                mock_pytorch.load_model.return_value = mock_pytorch_model
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                # Initialize fallback backend manually since OpenVINO is available
                backend._fallback_backend = mock_pytorch
                
                # Simulate missing OpenVINO files
                with patch('os.path.exists', return_value=False):
                    model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
                
                assert model == mock_pytorch_model
                assert mock_pytorch.load_model.called
    
    def test_load_model_with_missing_openvino_files_fallback_disabled(self):
        """Test loading model fails when OpenVINO files are missing and fallback is disabled."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            from mm_orch.runtime.backend_exceptions import ModelLoadError
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': False})
            
            with patch('os.path.exists', return_value=False):
                with pytest.raises(ModelLoadError) as exc_info:
                    backend.load_model('gpt2', 'models/gpt2', 'transformers')
                
                assert "openvino" in str(exc_info.value).lower()
    
    def test_load_model_openvino_manager_failure_triggers_fallback(self):
        """Test that OpenVINO manager failures trigger fallback when enabled."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.load_model.side_effect = RuntimeError("OpenVINO load failed")
            mock_manager_class.return_value = mock_manager
            
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_model = Mock()
                mock_pytorch.load_model.return_value = mock_pytorch_model
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                # Initialize fallback backend manually
                backend._fallback_backend = mock_pytorch
                
                with patch('os.path.exists', return_value=True):
                    with patch('mm_orch.schemas.ModelConfig'):
                        model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
                
                assert model == mock_pytorch_model
                assert mock_pytorch.load_model.called


class TestOpenVINOBackendInference:
    """Test inference methods."""
    
    def test_forward_with_openvino_model(self):
        """Test forward inference with OpenVINO model."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # Create mock model with forward method
            mock_model = Mock()
            mock_output = Mock()
            mock_output.logits = Mock()
            mock_model.forward.return_value = mock_output
            
            backend._models['test_model'] = mock_model
            
            inputs = {'input_ids': Mock(), 'attention_mask': Mock()}
            result = backend.forward(mock_model, inputs)
            
            assert 'logits' in result
            assert mock_model.forward.called
    
    def test_generate_with_openvino_model(self):
        """Test text generation with OpenVINO model."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # Create mock model and tokenizer
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer.decode.return_value = "Generated text"
            
            # Mock generate to return a list-like object
            mock_output = [Mock()]  # Make it subscriptable
            mock_model.generate.return_value = mock_output
            
            result = backend.generate(mock_model, mock_tokenizer, "Hello", max_length=50)
            
            assert result == "Generated text"
            assert mock_model.generate.called
            assert mock_tokenizer.decode.called
    
    def test_generate_with_fallback_model(self):
        """Test text generation with fallback PyTorch model."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch.generate.return_value = "Fallback generated text"
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                mock_model = Mock()
                mock_tokenizer = Mock()
                
                result = backend.generate(mock_model, mock_tokenizer, "Hello", max_length=50)
                
                assert result == "Fallback generated text"
                assert mock_pytorch.generate.called


class TestOpenVINOBackendModelManagement:
    """Test model management operations."""
    
    def test_unload_model_success(self):
        """Test successful model unloading."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.unload_model.return_value = True
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # Add a model to the registry
            backend._models['test_model'] = Mock()
            
            # Unload the model
            backend.unload_model('test_model')
            
            assert 'test_model' not in backend._models
    
    def test_unload_model_not_found(self):
        """Test unloading a model that doesn't exist."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            with pytest.raises(KeyError) as exc_info:
                backend.unload_model('nonexistent_model')
            
            assert 'nonexistent_model' in str(exc_info.value)
    
    def test_get_model_info_openvino_model(self):
        """Test getting model info for OpenVINO model."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            mock_model = Mock()
            mock_model.compiled_model = Mock()  # OpenVINO-specific attribute
            
            info = backend.get_model_info(mock_model)
            
            assert info['backend'] == 'openvino'
            assert info['device'] == 'CPU'
            assert info['is_fallback'] == False
    
    def test_get_model_info_fallback_model(self):
        """Test getting model info for fallback PyTorch model."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch.get_model_info.return_value = {
                    'backend': 'pytorch',
                    'device': 'cpu',
                    'parameters': 1000,
                }
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                mock_model = Mock()
                
                # Patch _is_fallback_model to return True
                with patch.object(backend, '_is_fallback_model', return_value=True):
                    info = backend.get_model_info(mock_model)
                
                assert info['is_fallback'] == True
                assert 'backend' in info


class TestOpenVINOBackendHelperMethods:
    """Test helper methods."""
    
    def test_get_openvino_path(self):
        """Test OpenVINO path conversion."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # Test path conversion
            pytorch_path = 'models/gpt2'
            openvino_path = backend._get_openvino_path(pytorch_path)
            
            assert 'openvino' in openvino_path
            assert 'gpt2' in openvino_path
            # Use os.path.join for platform-independent path comparison
            expected_path = os.path.join('models', 'openvino', 'gpt2')
            assert openvino_path == expected_path
    
    def test_is_fallback_model_detection(self):
        """Test fallback model detection."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                # PyTorch model (no OpenVINO attributes)
                pytorch_model = Mock(spec=['forward', 'generate'])
                assert backend._is_fallback_model(pytorch_model) == True
                
                # OpenVINO model (has compiled_model attribute)
                openvino_model = Mock()
                openvino_model.compiled_model = Mock()
                assert backend._is_fallback_model(openvino_model) == False
    
    def test_is_available(self):
        """Test backend availability check."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            assert backend.is_available() == True
        
        # Test when OpenVINO is not available
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_class.return_value = mock_pytorch
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                assert backend.is_available() == False


class TestOpenVINOBackendDeviceSelection:
    """Test device selection and validation scenarios."""
    
    def test_valid_device_cpu(self):
        """Test initialization with valid CPU device."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            assert backend.device == 'CPU'
            # Verify OpenVINO manager was initialized with CPU device
            call_kwargs = mock_manager_class.call_args[1]
            assert call_kwargs['default_device'] == 'CPU'
    
    def test_valid_device_gpu(self):
        """Test initialization with valid GPU device."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report GPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU', 'GPU.0']
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='GPU', config={'enable_fallback': True})
                
                assert backend.device == 'GPU'
                # Verify OpenVINO manager was initialized with GPU device
                call_kwargs = mock_manager_class.call_args[1]
                assert call_kwargs['default_device'] == 'GPU'
    
    def test_valid_device_auto(self):
        """Test initialization with AUTO device selection."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='AUTO', config={'enable_fallback': True})
            
            assert backend.device == 'AUTO'
            # Verify OpenVINO manager was initialized with AUTO device
            call_kwargs = mock_manager_class.call_args[1]
            assert call_kwargs['default_device'] == 'AUTO'
    
    def test_valid_device_npu(self):
        """Test initialization with NPU device."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report NPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU', 'NPU.0']
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='NPU', config={'enable_fallback': True})
                
                assert backend.device == 'NPU'
                # Verify OpenVINO manager was initialized with NPU device
                call_kwargs = mock_manager_class.call_args[1]
                assert call_kwargs['default_device'] == 'NPU'
    
    def test_invalid_device_raises_error(self):
        """Test that invalid device names raise ConfigurationError."""
        from mm_orch.runtime.openvino_backend import OpenVINOBackend
        from mm_orch.runtime.backend_exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError) as exc_info:
            backend = OpenVINOBackend(device='INVALID', config={'enable_fallback': True})
        
        error_message = str(exc_info.value)
        assert 'INVALID' in error_message
        assert 'CPU' in error_message
        assert 'GPU' in error_message
    
    def test_device_case_insensitive(self):
        """Test that device names are case-insensitive."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            # Test lowercase
            backend1 = OpenVINOBackend(device='cpu', config={'enable_fallback': True})
            assert backend1.device == 'CPU'
            
            # Test mixed case
            backend2 = OpenVINOBackend(device='Gpu', config={'enable_fallback': True})
            assert backend2.device == 'GPU'


class TestOpenVINOBackendDeviceFallback:
    """Test device fallback scenarios."""
    
    def test_gpu_fallback_to_cpu_when_unavailable(self):
        """Test that GPU falls back to CPU when GPU is not available."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report only CPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU']  # No GPU
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='GPU', config={'enable_fallback': True})
                
                # Device should have fallen back to CPU
                assert backend.device == 'CPU'
                # Verify OpenVINO manager was initialized with CPU device
                call_kwargs = mock_manager_class.call_args[1]
                assert call_kwargs['default_device'] == 'CPU'
    
    def test_npu_fallback_to_cpu_when_unavailable(self):
        """Test that NPU falls back to CPU when NPU is not available."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report only CPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU']  # No NPU
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='NPU', config={'enable_fallback': True})
                
                # Device should have fallen back to CPU
                assert backend.device == 'CPU'
                # Verify OpenVINO manager was initialized with CPU device
                call_kwargs = mock_manager_class.call_args[1]
                assert call_kwargs['default_device'] == 'CPU'
    
    def test_cpu_no_fallback_needed(self):
        """Test that CPU device doesn't trigger fallback (always available)."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # Device should remain CPU (no fallback)
            assert backend.device == 'CPU'
            # Verify OpenVINO manager was initialized with CPU device
            call_kwargs = mock_manager_class.call_args[1]
            assert call_kwargs['default_device'] == 'CPU'
    
    def test_auto_no_fallback_needed(self):
        """Test that AUTO device doesn't trigger fallback."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='AUTO', config={'enable_fallback': True})
            
            # Device should remain AUTO (no fallback)
            assert backend.device == 'AUTO'
            # Verify OpenVINO manager was initialized with AUTO device
            call_kwargs = mock_manager_class.call_args[1]
            assert call_kwargs['default_device'] == 'AUTO'
    
    def test_device_fallback_logs_warning(self):
        """Test that device fallback logs a warning message."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report only CPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU']
                mock_core_class.return_value = mock_core
                
                with patch('mm_orch.runtime.openvino_backend.logger') as mock_logger:
                    from mm_orch.runtime.openvino_backend import OpenVINOBackend
                    
                    backend = OpenVINOBackend(device='GPU', config={'enable_fallback': True})
                    
                    # Verify warning was logged
                    assert mock_logger.warning.called
                    warning_message = mock_logger.warning.call_args[0][0]
                    assert 'GPU' in warning_message
                    assert 'not available' in warning_message
                    assert 'CPU' in warning_message
    
    def test_device_availability_check_with_openvino_not_installed(self):
        """Test device availability check when OpenVINO is not installed."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager', side_effect=ImportError("OpenVINO not found")):
            with patch('mm_orch.runtime.pytorch_backend.PyTorchBackend') as mock_pytorch_class:
                mock_pytorch = Mock()
                mock_pytorch_class.return_value = mock_pytorch
                
                # Mock openvino import to fail
                with patch('openvino.Core', side_effect=ImportError("OpenVINO not installed")):
                    from mm_orch.runtime.openvino_backend import OpenVINOBackend
                    
                    # Should not raise error, should fall back to PyTorch
                    backend = OpenVINOBackend(device='GPU', config={'enable_fallback': True})
                    
                    # Should have fallen back to PyTorch backend
                    assert backend._fallback_backend is not None
                    assert backend._openvino_manager is None


class TestOpenVINOBackendDeviceValidation:
    """Test device validation methods."""
    
    def test_validate_device_normalizes_to_uppercase(self):
        """Test that _validate_device normalizes device names to uppercase."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='cpu', config={'enable_fallback': True})
            
            # Device should be normalized to uppercase
            assert backend.device == 'CPU'
    
    def test_check_device_availability_cpu_always_available(self):
        """Test that CPU is always reported as available."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # CPU should always be available
            assert backend._check_device_availability('CPU') == True
    
    def test_check_device_availability_auto_always_available(self):
        """Test that AUTO is always reported as available."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            from mm_orch.runtime.openvino_backend import OpenVINOBackend
            
            backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
            
            # AUTO should always be available
            assert backend._check_device_availability('AUTO') == True
    
    def test_check_device_availability_gpu_when_present(self):
        """Test GPU availability check when GPU is present."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report GPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU', 'GPU.0']
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                # GPU should be reported as available
                assert backend._check_device_availability('GPU') == True
    
    def test_check_device_availability_gpu_when_absent(self):
        """Test GPU availability check when GPU is not present."""
        with patch('mm_orch.runtime.openvino_manager.OpenVINOModelManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            # Mock OpenVINO Core to report only CPU available
            with patch('openvino.Core') as mock_core_class:
                mock_core = Mock()
                mock_core.available_devices = ['CPU']  # No GPU
                mock_core_class.return_value = mock_core
                
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                
                backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
                
                # GPU should be reported as not available
                assert backend._check_device_availability('GPU') == False
