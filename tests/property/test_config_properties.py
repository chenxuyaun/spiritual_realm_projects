"""Property-based tests for configuration management system.

These tests verify universal properties that should hold for all valid inputs.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st

from mm_orch.config import ConfigError, ConfigLoader, ConfigValidationError


# Strategies for generating test data
valid_log_levels = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
valid_devices = st.sampled_from(["auto", "cuda", "cpu"])
positive_integers = st.integers(min_value=1, max_value=100)
config_keys = st.text(min_size=1, max_size=20, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll'), min_codepoint=97, max_codepoint=122
))


@given(log_level=valid_log_levels)
def test_property_28_yaml_format_support(log_level):
    """
    Feature: muai-orchestration-system, Property 28: 配置格式支持
    
    **Validates: Requirements 11.1, 11.2**
    
    对于任何有效的YAML格式配置文件，系统应该能够成功解析并加载配置，
    且加载后的配置对象应该包含所有必需的字段。
    """
    # Create valid YAML config
    config_data = {
        "system": {
            "log_level": log_level,
            "max_cached_models": 3,
        },
        "storage": {
            "vector_db_path": "data/vector_db",
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        # Should successfully load
        loader = ConfigLoader(temp_path)
        
        # Should contain all required fields
        assert loader.get("system") is not None
        assert loader.get("storage") is not None
        
        # Should contain loaded values
        assert loader.get("system.log_level") == log_level
        
        # Should validate successfully
        assert loader.validate() is True
    finally:
        Path(temp_path).unlink()


@given(device=valid_devices, max_models=positive_integers)
def test_property_28_json_format_support(device, max_models):
    """
    Feature: muai-orchestration-system, Property 28: 配置格式支持
    
    **Validates: Requirements 11.1, 11.2**
    
    对于任何有效的JSON格式配置文件，系统应该能够成功解析并加载配置，
    且加载后的配置对象应该包含所有必需的字段。
    """
    # Create valid JSON config
    config_data = {
        "system": {
            "device": device,
            "max_cached_models": max_models,
        },
        "storage": {
            "vector_db_path": "data/vector_db",
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    try:
        # Should successfully load
        loader = ConfigLoader(temp_path)
        
        # Should contain all required fields
        assert loader.get("system") is not None
        assert loader.get("storage") is not None
        
        # Should contain loaded values
        assert loader.get("system.device") == device
        assert loader.get("system.max_cached_models") == max_models
        
        # Should validate successfully
        assert loader.validate() is True
    finally:
        Path(temp_path).unlink()


@given(
    missing_key=st.sampled_from([
        "nonexistent.key",
        "system.nonexistent",
        "missing.nested.key",
        "a.b.c.d.e"
    ])
)
def test_property_29_default_value_fallback(missing_key):
    """
    Feature: muai-orchestration-system, Property 29: 配置默认值回退
    
    **Validates: Requirements 11.3**
    
    对于任何配置项，当该项在配置文件中缺失时，系统应该使用预定义的默认值，
    且不应该抛出异常。
    """
    # Create minimal config (missing many keys)
    config_data = {
        "system": {},
        "storage": {}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Should not raise exception when accessing missing key
        result = loader.get(missing_key)
        
        # Should return None (default) for missing keys
        assert result is None
        
        # Should return custom default if provided
        custom_default = "custom_value"
        result_with_default = loader.get(missing_key, custom_default)
        assert result_with_default == custom_default
    finally:
        Path(temp_path).unlink()


@given(log_level=valid_log_levels)
def test_property_29_system_defaults_present(log_level):
    """
    Feature: muai-orchestration-system, Property 29: 配置默认值回退
    
    **Validates: Requirements 11.3**
    
    当配置文件只包含部分配置项时，系统应该为缺失的配置项使用默认值。
    """
    # Create config with only log_level
    config_data = {
        "system": {
            "log_level": log_level,
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Loaded value should be present
        assert loader.get("system.log_level") == log_level
        
        # Default values should be present for missing keys
        assert loader.get("system.max_cached_models") == 3
        assert loader.get("system.development_stage") == "adult"
        assert loader.get("system.device") == "auto"
        
        # Storage defaults should be present
        assert loader.get("storage.vector_db_path") is not None
        assert loader.get("storage.chat_history_path") is not None
    finally:
        Path(temp_path).unlink()


@given(
    key1=config_keys,
    key2=config_keys,
    value=st.text(min_size=0, max_size=50)
)
def test_property_29_no_exception_on_missing_keys(key1, key2, value):
    """
    Feature: muai-orchestration-system, Property 29: 配置默认值回退
    
    **Validates: Requirements 11.3**
    
    访问任何不存在的配置键都不应该抛出异常。
    """
    loader = ConfigLoader()
    
    # Construct arbitrary nested key
    nested_key = f"{key1}.{key2}"
    
    # Should not raise exception
    result = loader.get(nested_key)
    
    # Should return None or provided default
    assert result is None or result == loader.get(nested_key, value)


@given(
    log_level=valid_log_levels,
    device=valid_devices,
    max_models=positive_integers
)
def test_property_config_merge_preserves_defaults(log_level, device, max_models):
    """
    Property: Configuration merging preserves default values
    
    When loading a partial configuration, default values should be preserved
    for keys not specified in the loaded config.
    """
    # Create partial config
    config_data = {
        "system": {
            "log_level": log_level,
            "device": device,
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Loaded values should override defaults
        assert loader.get("system.log_level") == log_level
        assert loader.get("system.device") == device
        
        # Unspecified values should use defaults
        assert loader.get("system.max_cached_models") == 3
        assert loader.get("system.development_stage") == "adult"
        
        # Entire sections not in config should use defaults
        assert loader.get("router.confidence_threshold") == 0.6
        assert loader.get("api.port") == 8000
    finally:
        Path(temp_path).unlink()


@given(
    path1=st.text(min_size=1, max_size=20),
    path2=st.text(min_size=1, max_size=20)
)
def test_property_get_all_returns_independent_copy(path1, path2):
    """
    Property: get_all() returns independent copies
    
    Multiple calls to get_all() should return independent copies that don't
    affect each other or the original configuration.
    """
    loader = ConfigLoader()
    
    # Get two copies
    config1 = loader.get_all()
    config2 = loader.get_all()
    
    # Modify first copy
    if "custom" not in config1:
        config1["custom"] = {}
    config1["custom"]["path1"] = path1
    
    # Second copy should be unaffected
    assert config2.get("custom", {}).get("path1") != path1
    
    # Original should be unaffected
    assert loader.get("custom.path1") != path1


@given(
    invalid_level=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
)
def test_property_validation_rejects_invalid_log_level(invalid_level):
    """
    Property: Validation rejects invalid log levels
    
    For any invalid log level, validation should fail with ConfigValidationError.
    """
    config_data = {
        "system": {
            "log_level": invalid_level,
        },
        "storage": {
            "vector_db_path": "data/vector_db",
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Validation should fail
        with pytest.raises(ConfigValidationError, match="Invalid log_level"):
            loader.validate()
    finally:
        Path(temp_path).unlink()


@given(
    invalid_max_models=st.integers(max_value=0)
)
def test_property_validation_rejects_invalid_max_models(invalid_max_models):
    """
    Property: Validation rejects invalid max_cached_models
    
    For any non-positive integer, validation should fail.
    """
    config_data = {
        "system": {
            "max_cached_models": invalid_max_models,
        },
        "storage": {
            "vector_db_path": "data/vector_db",
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Validation should fail
        with pytest.raises(ConfigValidationError, match="must be a positive integer"):
            loader.validate()
    finally:
        Path(temp_path).unlink()


@given(
    invalid_device=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in ["auto", "cuda", "cpu"]
    )
)
def test_property_validation_rejects_invalid_device(invalid_device):
    """
    Property: Validation rejects invalid device values
    
    For any device value not in the valid set, validation should fail.
    """
    config_data = {
        "system": {
            "device": invalid_device,
        },
        "storage": {
            "vector_db_path": "data/vector_db",
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loader = ConfigLoader(temp_path)
        
        # Validation should fail
        with pytest.raises(ConfigValidationError, match="Invalid device"):
            loader.validate()
    finally:
        Path(temp_path).unlink()
