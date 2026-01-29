"""
Unit tests for BackendConfig.

Tests specific examples and edge cases for configuration loading and validation.
"""

import os
import tempfile
import yaml
import pytest
from mm_orch.runtime.backend_config import BackendConfig


def create_temp_config(config_dict):
    """Helper to create temporary config file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    yaml.dump(config_dict, temp_file)
    temp_file.close()
    return temp_file.name


class TestBackendConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_valid_configuration_file(self):
        """Test loading a valid configuration file."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "GPU",
                    "enable_fallback": False,
                    "cache_dir": "custom/path",
                    "num_streams": 4
                },
                "pytorch": {
                    "device": "cuda",
                    "dtype": "float16"
                }
            },
            "model_overrides": {
                "gpt2": "pytorch",
                "t5-small": "openvino"
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            
            # Verify all settings are loaded correctly
            assert backend_config.get_default_backend() == "openvino"
            
            ov_config = backend_config.get_backend_config("openvino")
            assert ov_config["device"] == "GPU"
            assert ov_config["enable_fallback"] is False
            assert ov_config["cache_dir"] == "custom/path"
            assert ov_config["num_streams"] == 4
            
            pt_config = backend_config.get_backend_config("pytorch")
            assert pt_config["device"] == "cuda"
            assert pt_config["dtype"] == "float16"
            
            assert backend_config.get_model_backend("gpt2") == "pytorch"
            assert backend_config.get_model_backend("t5-small") == "openvino"
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_handle_missing_configuration_file(self):
        """Test handling of missing configuration file."""
        backend_config = BackendConfig(config_path="non_existent_file.yaml")
        
        # Should use default values
        assert backend_config.get_default_backend() == "pytorch"
        
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == "CPU"
        assert ov_config["enable_fallback"] is True
        
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["device"] == "cpu"
        assert pt_config["dtype"] == "float32"
    
    def test_handle_empty_configuration_file(self):
        """Test handling of empty configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        temp_file.write("")
        temp_file.close()
        
        try:
            backend_config = BackendConfig(config_path=temp_file.name)
            
            # Should use default values
            assert backend_config.get_default_backend() == "pytorch"
            
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
    
    def test_handle_malformed_yaml(self):
        """Test handling of malformed YAML file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
        temp_file.write("backend:\n  default: pytorch\n  invalid yaml: [unclosed")
        temp_file.close()
        
        try:
            backend_config = BackendConfig(config_path=temp_file.name)
            
            # Should fall back to defaults
            assert backend_config.get_default_backend() == "pytorch"
            
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)


class TestBackendConfigValidation:
    """Test configuration validation functionality."""
    
    def test_validate_invalid_default_backend(self):
        """Test validation of invalid default backend."""
        config_dict = {
            "backend": {
                "default": "tensorflow",  # Invalid
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            
            # Should fall back to 'pytorch'
            assert backend_config.get_default_backend() == "pytorch"
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_openvino_device(self):
        """Test validation of invalid OpenVINO device."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "TPU",  # Invalid
                    "enable_fallback": True
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            # Should fall back to 'CPU'
            assert ov_config["device"] == "CPU"
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_pytorch_device(self):
        """Test validation of invalid PyTorch device."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {
                    "device": "tpu"  # Invalid
                }
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            pt_config = backend_config.get_backend_config("pytorch")
            
            # Should fall back to 'cpu'
            assert pt_config["device"] == "cpu"
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_dtype(self):
        """Test validation of invalid PyTorch dtype."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {
                    "device": "cpu",
                    "dtype": "int8"  # Invalid
                }
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            pt_config = backend_config.get_backend_config("pytorch")
            
            # Should fall back to 'float32'
            assert pt_config["dtype"] == "float32"
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_enable_fallback(self):
        """Test validation of invalid enable_fallback value."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "CPU",
                    "enable_fallback": "yes"  # Should be boolean
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            # Should fall back to True
            assert ov_config["enable_fallback"] is True
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_num_streams_negative(self):
        """Test validation of negative num_streams."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "CPU",
                    "num_streams": -5  # Invalid
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            # Should fall back to 1
            assert ov_config["num_streams"] == 1
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_num_streams_zero(self):
        """Test validation of zero num_streams."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "CPU",
                    "num_streams": 0  # Invalid
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            # Should fall back to 1
            assert ov_config["num_streams"] == 1
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_num_streams_string(self):
        """Test validation of string num_streams."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {
                    "device": "CPU",
                    "num_streams": "four"  # Invalid
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            # Should fall back to 1
            assert ov_config["num_streams"] == 1
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_model_override(self):
        """Test validation of invalid model override."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            },
            "model_overrides": {
                "gpt2": "tensorflow"  # Invalid
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            
            # Invalid override should be removed
            assert backend_config.get_model_backend("gpt2") is None
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_validate_invalid_model_overrides_type(self):
        """Test validation of invalid model_overrides type."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            },
            "model_overrides": ["gpt2", "t5-small"]  # Should be dict
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            
            # Should fall back to empty dict
            assert backend_config.get_model_backend("gpt2") is None
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestBackendConfigDefaults:
    """Test default configuration values."""
    
    def test_default_backend_is_pytorch(self):
        """Test that default backend is 'pytorch'."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        assert backend_config.get_default_backend() == "pytorch"
    
    def test_default_openvino_device_is_cpu(self):
        """Test that default OpenVINO device is 'CPU'."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == "CPU"
    
    def test_default_openvino_fallback_is_enabled(self):
        """Test that default OpenVINO fallback is enabled."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["enable_fallback"] is True
    
    def test_default_openvino_cache_dir(self):
        """Test that default OpenVINO cache_dir is set."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["cache_dir"] == "models/openvino"
    
    def test_default_openvino_num_streams(self):
        """Test that default OpenVINO num_streams is 1."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["num_streams"] == 1
    
    def test_default_pytorch_device_is_cpu(self):
        """Test that default PyTorch device is 'cpu'."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["device"] == "cpu"
    
    def test_default_pytorch_dtype_is_float32(self):
        """Test that default PyTorch dtype is 'float32'."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["dtype"] == "float32"
    
    def test_default_model_overrides_is_empty(self):
        """Test that default model_overrides is empty."""
        backend_config = BackendConfig(config_path="non_existent.yaml")
        assert backend_config.get_model_backend("any_model") is None


class TestBackendConfigAPI:
    """Test BackendConfig API methods."""
    
    def test_get_default_backend(self):
        """Test get_default_backend method."""
        config_dict = {
            "backend": {
                "default": "openvino",
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            assert backend_config.get_default_backend() == "openvino"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_get_backend_config_openvino(self):
        """Test get_backend_config for OpenVINO."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {
                    "device": "GPU",
                    "enable_fallback": False
                },
                "pytorch": {"device": "cpu"}
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            ov_config = backend_config.get_backend_config("openvino")
            
            assert ov_config["device"] == "GPU"
            assert ov_config["enable_fallback"] is False
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_get_backend_config_pytorch(self):
        """Test get_backend_config for PyTorch."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {
                    "device": "cuda",
                    "dtype": "float16"
                }
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            pt_config = backend_config.get_backend_config("pytorch")
            
            assert pt_config["device"] == "cuda"
            assert pt_config["dtype"] == "float16"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_get_model_backend_with_override(self):
        """Test get_model_backend with override."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            },
            "model_overrides": {
                "gpt2": "openvino"
            }
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            assert backend_config.get_model_backend("gpt2") == "openvino"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_get_model_backend_without_override(self):
        """Test get_model_backend without override."""
        config_dict = {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU"},
                "pytorch": {"device": "cpu"}
            },
            "model_overrides": {}
        }
        
        config_path = create_temp_config(config_dict)
        
        try:
            backend_config = BackendConfig(config_path=config_path)
            assert backend_config.get_model_backend("gpt2") is None
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
