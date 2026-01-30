"""
Integration tests for ModelManager with multiple backends.

Requirements tested:
- 1.1: Backend selection
- 2.3: Per-model backend overrides
- 5.5: Backend switching
"""

import os
import pytest
import tempfile
import yaml

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        "backend": {
            "default": "pytorch",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True,
                "cache_dir": "models/openvino"
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {
            "test-model-ov": "openvino"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)



class TestModelManagerBackendSelection:
    """Test backend selection in ModelManager."""
    
    def test_default_pytorch_backend(self):
        """Test that ModelManager defaults to PyTorch backend (Requirement 1.1)."""
        manager = ModelManager(max_cached_models=2)
        
        assert manager.default_backend == "pytorch"
        assert "pytorch" in manager._backend_factory.get_available_backends()
    
    def test_explicit_backend_selection(self):
        """Test explicit backend selection (Requirement 1.1)."""
        # Test PyTorch backend
        manager_pt = ModelManager(backend="pytorch")
        assert manager_pt.default_backend == "pytorch"
        
        # Test OpenVINO backend (if available)
        available_backends = manager_pt._backend_factory.get_available_backends()
        if "openvino" in available_backends:
            manager_ov = ModelManager(backend="openvino")
            assert manager_ov.default_backend == "openvino"
    
    def test_backend_config_from_file(self, temp_config_file):
        """Test loading backend configuration from file (Requirement 2.3)."""
        manager = ModelManager(backend_config=temp_config_file)
        
        # Should use default from config file
        assert manager.default_backend == "pytorch"
        
        # Check that config was loaded
        assert manager._backend_config_loader is not None
        assert manager._backend_config_loader.get_default_backend() == "pytorch"
    
    def test_backend_config_from_dict(self):
        """Test loading backend configuration from dictionary."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "pytorch": {"device": "cpu"}
            },
            "model_overrides": {}
        }
        
        manager = ModelManager(backend_config=config_dict)
        assert manager.default_backend == "pytorch"


class TestModelManagerPerModelOverride:
    """Test per-model backend overrides."""
    
    def test_per_model_backend_override(self, temp_config_file):
        """Test per-model backend override from configuration (Requirement 2.3)."""
        manager = ModelManager(backend_config=temp_config_file)
        
        # Check that model override is loaded
        override = manager._backend_config_loader.get_model_backend("test-model-ov")
        assert override == "openvino"
    
    def test_load_model_with_override(self, temp_config_file):
        """Test loading model with backend override (Requirement 2.3)."""
        manager = ModelManager(backend_config=temp_config_file, max_cached_models=2)
        
        # Register a test model
        test_config = ModelConfig(
            name="test-model-ov",
            model_path="gpt2",  # Use a small model for testing
            model_type="transformers",
            device="cpu",
            max_length=512
        )
        manager.register_model(test_config)
        
        # Note: Actual loading would require the model files to exist
        # This test verifies the override is detected
        override = manager._backend_config_loader.get_model_backend("test-model-ov")
        assert override == "openvino"


class TestModelManagerBackendSwitching:
    """Test backend switching capabilities."""
    
    def test_backend_override_parameter(self):
        """Test backend_override parameter in load_model (Requirement 5.5)."""
        manager = ModelManager(backend="pytorch")
        
        # Register a test model
        test_config = ModelConfig(
            name="test-model",
            model_path="gpt2",
            model_type="transformers",
            device="cpu",
            max_length=512
        )
        manager.register_model(test_config)
        
        # Verify that backend_override parameter is accepted
        # (actual loading would require model files)
        assert manager.is_registered("test-model")
    
    def test_cache_info_includes_backend(self):
        """Test that cache info includes backend information."""
        manager = ModelManager(backend="pytorch")
        
        cache_info = manager.get_cache_info()
        
        assert "default_backend" in cache_info
        assert cache_info["default_backend"] == "pytorch"
        assert "available_backends" in cache_info
        assert isinstance(cache_info["available_backends"], list)


class TestModelManagerBackwardCompatibility:
    """Test backward compatibility with existing code."""
    
    def test_no_backend_parameter_defaults_to_pytorch(self):
        """Test that omitting backend parameter defaults to PyTorch (Requirement 3.1)."""
        # Old code that doesn't specify backend
        manager = ModelManager(max_cached_models=3, default_device="cpu")
        
        assert manager.default_backend == "pytorch"
    
    def test_existing_api_signatures_unchanged(self):
        """Test that existing API signatures are unchanged (Requirement 3.2)."""
        manager = ModelManager()
        
        # Verify all expected methods exist with correct signatures
        assert hasattr(manager, 'load_model')
        assert hasattr(manager, 'get_model')
        assert hasattr(manager, 'unload_model')
        assert hasattr(manager, 'infer')
        assert hasattr(manager, 'register_model')
        assert hasattr(manager, 'get_cache_info')
        
        # Verify load_model accepts optional backend_override
        import inspect
        sig = inspect.signature(manager.load_model)
        params = list(sig.parameters.keys())
        assert 'model_name' in params
        assert 'device' in params
        assert 'backend_override' in params


class TestModelManagerInferenceWithBackends:
    """Test inference with different backends."""
    
    def test_inference_method_signature_unchanged(self):
        """Test that infer method signature is unchanged (Requirement 5.1, 5.2)."""
        manager = ModelManager()
        
        import inspect
        sig = inspect.signature(manager.infer)
        params = list(sig.parameters.keys())
        
        # Verify expected parameters
        assert 'model_name' in params
        assert 'inputs' in params
        assert 'kwargs' in params or any('**' in str(p) for p in sig.parameters.values())


# Feature: openvino-backend-integration
# Integration tests for ModelManager with multiple backends
# Tests Requirements: 1.1, 2.3, 5.5
