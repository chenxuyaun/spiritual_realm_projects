"""Unit tests for configuration management system."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from mm_orch.config import ConfigError, ConfigLoader, ConfigValidationError


class TestConfigLoader:
    """Test suite for ConfigLoader class."""
    
    def test_default_config_loaded(self):
        """Test that default configuration is loaded when no file provided."""
        loader = ConfigLoader()
        
        assert loader.get("system.log_level") == "INFO"
        assert loader.get("system.max_cached_models") == 3
        assert loader.get("system.development_stage") == "adult"
    
    def test_get_with_dot_notation(self):
        """Test getting nested configuration values with dot notation."""
        loader = ConfigLoader()
        
        # Test nested access
        assert loader.get("system.log_level") == "INFO"
        assert loader.get("storage.vector_db_path") == "data/vector_db"
        
        # Test non-existent key returns default
        assert loader.get("nonexistent.key", "default") == "default"
    
    def test_get_without_default(self):
        """Test getting non-existent key without default returns None."""
        loader = ConfigLoader()
        
        assert loader.get("nonexistent.key") is None
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        # Create temporary YAML config
        config_data = {
            "system": {
                "log_level": "DEBUG",
                "max_cached_models": 5,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # Check overridden values
            assert loader.get("system.log_level") == "DEBUG"
            assert loader.get("system.max_cached_models") == 5
            
            # Check default values still present
            assert loader.get("system.development_stage") == "adult"
        finally:
            Path(temp_path).unlink()
    
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        # Create temporary JSON config
        config_data = {
            "system": {
                "log_level": "WARNING",
                "device": "cpu",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # Check overridden values
            assert loader.get("system.log_level") == "WARNING"
            assert loader.get("system.device") == "cpu"
            
            # Check default values still present
            assert loader.get("system.max_cached_models") == 3
        finally:
            Path(temp_path).unlink()
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported file format raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some text")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigError, match="Unsupported configuration format"):
                ConfigLoader(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_yaml_raises_error(self):
        """Test that invalid YAML raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigError, match="Failed to parse configuration file"):
                ConfigLoader(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ConfigError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigError, match="Failed to parse configuration file"):
                ConfigLoader(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_nonexistent_file_uses_defaults(self):
        """Test that nonexistent file path uses default configuration."""
        loader = ConfigLoader("nonexistent_config.yaml")
        
        # Should use defaults
        assert loader.get("system.log_level") == "INFO"
    
    def test_deep_merge_config(self):
        """Test that configuration merging works correctly."""
        config_data = {
            "system": {
                "log_level": "ERROR",
            },
            "models": {
                "custom_model": {
                    "model_path": "custom/path",
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            # Check merged values
            assert loader.get("system.log_level") == "ERROR"
            assert loader.get("system.max_cached_models") == 3  # Default preserved
            assert loader.get("models.custom_model.model_path") == "custom/path"
        finally:
            Path(temp_path).unlink()
    
    def test_get_all_returns_copy(self):
        """Test that get_all returns a copy of configuration."""
        loader = ConfigLoader()
        
        config1 = loader.get_all()
        config2 = loader.get_all()
        
        # Modify one copy
        config1["system"]["log_level"] = "MODIFIED"
        
        # Other copy should be unchanged
        assert config2["system"]["log_level"] == "INFO"
        
        # Original should be unchanged
        assert loader.get("system.log_level") == "INFO"
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        loader = ConfigLoader()
        
        assert loader.validate() is True
    
    def test_validate_invalid_log_level(self):
        """Test validation fails for invalid log level."""
        config_data = {
            "system": {
                "log_level": "INVALID_LEVEL",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            with pytest.raises(ConfigValidationError, match="Invalid log_level"):
                loader.validate()
        finally:
            Path(temp_path).unlink()
    
    def test_validate_invalid_max_cached_models(self):
        """Test validation fails for invalid max_cached_models."""
        config_data = {
            "system": {
                "max_cached_models": -1,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            with pytest.raises(ConfigValidationError, match="must be a positive integer"):
                loader.validate()
        finally:
            Path(temp_path).unlink()
    
    def test_validate_invalid_device(self):
        """Test validation fails for invalid device."""
        config_data = {
            "system": {
                "device": "invalid_device",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            
            with pytest.raises(ConfigValidationError, match="Invalid device"):
                loader.validate()
        finally:
            Path(temp_path).unlink()
    
    def test_reload_config(self):
        """Test reloading configuration from file."""
        config_data = {
            "system": {
                "log_level": "DEBUG",
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader(temp_path)
            assert loader.get("system.log_level") == "DEBUG"
            
            # Modify file
            with open(temp_path, 'w') as f:
                yaml.dump({"system": {"log_level": "ERROR"}}, f)
            
            # Reload
            loader.reload()
            assert loader.get("system.log_level") == "ERROR"
        finally:
            Path(temp_path).unlink()
    
    def test_repr(self):
        """Test string representation of ConfigLoader."""
        loader = ConfigLoader("test_config.yaml")
        
        repr_str = repr(loader)
        assert "ConfigLoader" in repr_str
        assert "test_config.yaml" in repr_str
