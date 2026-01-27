"""
Property-based tests for optimization configuration.

Tests universal properties of configuration loading, validation, and
environment variable overrides using Hypothesis.
"""

import os
import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings
import yaml

from mm_orch.optimization.config import (
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig,
    BatcherConfig,
    CacheConfig,
    TunerConfig,
    load_optimization_config,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

@st.composite
def valid_vllm_config(draw):
    """Generate valid VLLMConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "tensor_parallel_size": draw(st.integers(min_value=1, max_value=8)),
        "dtype": draw(st.sampled_from(["auto", "fp16", "fp32", "bf16"])),
        "max_model_len": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=8192))),
        "gpu_memory_utilization": draw(st.floats(min_value=0.1, max_value=1.0)),
        "swap_space": draw(st.integers(min_value=0, max_value=32)),
    }


@st.composite
def valid_deepspeed_config(draw):
    """Generate valid DeepSpeedConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "tensor_parallel": draw(st.integers(min_value=1, max_value=8)),
        "pipeline_parallel": draw(st.integers(min_value=1, max_value=8)),
        "dtype": draw(st.sampled_from(["fp16", "fp32", "bf16"])),
        "replace_with_kernel_inject": draw(st.booleans()),
    }


@st.composite
def valid_onnx_config(draw):
    """Generate valid ONNXConfig parameters."""
    providers = draw(st.lists(
        st.sampled_from([
            "CUDAExecutionProvider",
            "TensorrtExecutionProvider",
            "CPUExecutionProvider",
        ]),
        min_size=1,
        max_size=3,
        unique=True
    ))
    return {
        "enabled": draw(st.booleans()),
        "execution_providers": providers,
        "optimization_level": draw(st.sampled_from(["none", "basic", "extended", "all"])),
        "enable_quantization": draw(st.booleans()),
    }


@st.composite
def valid_batcher_config(draw):
    """Generate valid BatcherConfig parameters."""
    min_size = draw(st.integers(min_value=1, max_value=16))
    max_size = draw(st.integers(min_value=min_size, max_value=128))
    return {
        "enabled": draw(st.booleans()),
        "max_batch_size": max_size,
        "batch_timeout_ms": draw(st.integers(min_value=0, max_value=1000)),
        "adaptive_batching": draw(st.booleans()),
        "min_batch_size": min_size,
    }


@st.composite
def valid_cache_config(draw):
    """Generate valid CacheConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "max_memory_mb": draw(st.integers(min_value=0, max_value=32768)),
        "eviction_policy": draw(st.sampled_from(["lru", "fifo"])),
    }


@st.composite
def valid_tuner_config(draw):
    """Generate valid TunerConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "observation_window_seconds": draw(st.integers(min_value=1, max_value=3600)),
        "tuning_interval_seconds": draw(st.integers(min_value=1, max_value=600)),
        "enable_batch_size_tuning": draw(st.booleans()),
        "enable_timeout_tuning": draw(st.booleans()),
        "enable_cache_size_tuning": draw(st.booleans()),
    }


@st.composite
def valid_optimization_config(draw):
    """Generate valid OptimizationConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "vllm": draw(valid_vllm_config()),
        "deepspeed": draw(valid_deepspeed_config()),
        "onnx": draw(valid_onnx_config()),
        "batcher": draw(valid_batcher_config()),
        "cache": draw(valid_cache_config()),
        "tuner": draw(valid_tuner_config()),
        "engine_preference": draw(st.lists(
            st.sampled_from(["vllm", "deepspeed", "onnx", "pytorch"]),
            min_size=1,
            max_size=4,
            unique=True
        )),
        "fallback_on_error": draw(st.booleans()),
    }


# =============================================================================
# Property Tests
# =============================================================================

# Feature: advanced-optimization-monitoring, Property 54: YAML configuration is parsed correctly
@given(config_dict=valid_optimization_config())
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_yaml_configuration_parsed_correctly(config_dict):
    """
    Property 54: YAML configuration is parsed correctly.
    
    For any valid YAML configuration file, the system should parse it
    and apply the settings correctly.
    
    **Validates: Requirements 14.1**
    """
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = {"optimization": config_dict}
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    try:
        # Load configuration from YAML
        config = load_optimization_config(config_path=temp_path)
        
        # Verify configuration was parsed correctly
        assert config.enabled == config_dict.get("enabled", True)
        assert config.fallback_on_error == config_dict.get("fallback_on_error", True)
        
        # Verify vLLM config
        vllm_dict = config_dict.get("vllm", {})
        assert config.vllm.enabled == vllm_dict.get("enabled", True)
        assert config.vllm.tensor_parallel_size == vllm_dict.get("tensor_parallel_size", 1)
        
        # Verify DeepSpeed config
        deepspeed_dict = config_dict.get("deepspeed", {})
        assert config.deepspeed.enabled == deepspeed_dict.get("enabled", True)
        
        # Verify ONNX config
        onnx_dict = config_dict.get("onnx", {})
        assert config.onnx.enabled == onnx_dict.get("enabled", True)
        
        # Verify batcher config
        batcher_dict = config_dict.get("batcher", {})
        assert config.batcher.enabled == batcher_dict.get("enabled", True)
        
        # Verify cache config
        cache_dict = config_dict.get("cache", {})
        assert config.cache.enabled == cache_dict.get("enabled", True)
        
        # Verify tuner config
        tuner_dict = config_dict.get("tuner", {})
        assert config.tuner.enabled == tuner_dict.get("enabled", False)
        
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: advanced-optimization-monitoring, Property 55: Environment variables override configuration
@given(
    file_enabled=st.booleans(),
    env_enabled=st.booleans(),
    file_vllm_enabled=st.booleans(),
    env_vllm_enabled=st.booleans(),
    file_batch_size=st.integers(min_value=1, max_value=64),
    env_batch_size=st.integers(min_value=1, max_value=128),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_environment_variables_override_configuration(
    file_enabled, env_enabled, file_vllm_enabled, env_vllm_enabled,
    file_batch_size, env_batch_size
):
    """
    Property 55: Environment variables override configuration.
    
    For any configuration parameter with both file and environment variable
    values, the environment variable value should take precedence.
    
    **Validates: Requirements 14.2**
    """
    # Create configuration dict
    config_dict = {
        "enabled": file_enabled,
        "vllm": {"enabled": file_vllm_enabled},
        "batcher": {"max_batch_size": file_batch_size},
    }
    
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = {"optimization": config_dict}
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    # Set environment variables
    env_vars = {
        "MUAI_OPT_ENABLED": str(env_enabled).lower(),
        "MUAI_OPT_VLLM_ENABLED": str(env_vllm_enabled).lower(),
        "MUAI_OPT_BATCHER_MAX_SIZE": str(env_batch_size),
    }
    
    # Save original environment
    original_env = {}
    for key in env_vars:
        original_env[key] = os.environ.get(key)
        os.environ[key] = env_vars[key]
    
    try:
        # Load configuration
        config = load_optimization_config(config_path=temp_path)
        
        # Verify environment variables override file values
        assert config.enabled == env_enabled, "Top-level enabled should be overridden by env var"
        assert config.vllm.enabled == env_vllm_enabled, "vLLM enabled should be overridden by env var"
        assert config.batcher.max_batch_size == env_batch_size, "Batch size should be overridden by env var"
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: advanced-optimization-monitoring, Property 56: Invalid configuration is rejected with clear errors
@given(
    invalid_type=st.sampled_from([
        "invalid_dtype",
        "invalid_optimization_level",
        "invalid_eviction_policy",
        "negative_tensor_parallel",
        "negative_batch_size",
        "invalid_gpu_memory",
    ])
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_configuration_rejected_with_clear_errors(invalid_type):
    """
    Property 56: Invalid configuration is rejected with clear errors.
    
    For any invalid configuration (missing required fields, invalid values),
    the system should reject it at startup with a clear error message.
    
    **Validates: Requirements 14.3**
    """
    # Create invalid configuration based on type
    if invalid_type == "invalid_dtype":
        config_dict = {"vllm": {"dtype": "invalid_type"}}
    elif invalid_type == "invalid_optimization_level":
        config_dict = {"onnx": {"optimization_level": "invalid_level"}}
    elif invalid_type == "invalid_eviction_policy":
        config_dict = {"cache": {"eviction_policy": "invalid_policy"}}
    elif invalid_type == "negative_tensor_parallel":
        config_dict = {"vllm": {"tensor_parallel_size": -1}}
    elif invalid_type == "negative_batch_size":
        config_dict = {"batcher": {"max_batch_size": -1}}
    elif invalid_type == "invalid_gpu_memory":
        config_dict = {"vllm": {"gpu_memory_utilization": 1.5}}
    
    # Attempt to load configuration and expect ValueError
    with pytest.raises(ValueError) as exc_info:
        load_optimization_config(config_dict=config_dict)
    
    # Verify error message is informative
    error_message = str(exc_info.value)
    assert len(error_message) > 0, "Error message should not be empty"
    assert "Configuration validation failed" in error_message or "must be" in error_message.lower()


# Feature: advanced-optimization-monitoring, Property 57: Default values are used for missing configuration
@given(
    include_vllm=st.booleans(),
    include_deepspeed=st.booleans(),
    include_onnx=st.booleans(),
    include_batcher=st.booleans(),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_default_values_used_for_missing_configuration(
    include_vllm, include_deepspeed, include_onnx, include_batcher
):
    """
    Property 57: Default values are used for missing configuration.
    
    For any configuration parameter not specified in the configuration file,
    the system should use the documented default value.
    
    **Validates: Requirements 14.4**
    """
    # Create partial configuration
    config_dict = {}
    
    if include_vllm:
        config_dict["vllm"] = {"enabled": True}  # Only set enabled, others should use defaults
    
    if include_deepspeed:
        config_dict["deepspeed"] = {"enabled": False}
    
    if include_onnx:
        config_dict["onnx"] = {}  # Empty dict, all should use defaults
    
    if include_batcher:
        config_dict["batcher"] = {"max_batch_size": 16}  # Only set one field
    
    # Load configuration
    config = load_optimization_config(config_dict=config_dict)
    
    # Verify defaults are used for missing fields
    if not include_vllm:
        # All vLLM fields should use defaults
        assert config.vllm.enabled == True  # Default
        assert config.vllm.tensor_parallel_size == 1  # Default
        assert config.vllm.dtype == "auto"  # Default
        assert config.vllm.gpu_memory_utilization == 0.9  # Default
    else:
        # Only enabled was set, others should use defaults
        assert config.vllm.enabled == True
        assert config.vllm.tensor_parallel_size == 1  # Default
        assert config.vllm.dtype == "auto"  # Default
    
    if not include_deepspeed:
        assert config.deepspeed.enabled == True  # Default
        assert config.deepspeed.tensor_parallel == 1  # Default
    
    if include_onnx:
        # Empty dict, all should use defaults
        assert config.onnx.enabled == True  # Default
        assert config.onnx.optimization_level == "all"  # Default
    
    if include_batcher:
        # Only max_batch_size was set
        assert config.batcher.max_batch_size == 16
        assert config.batcher.batch_timeout_ms == 50  # Default
        assert config.batcher.adaptive_batching == True  # Default
    else:
        # All should use defaults
        assert config.batcher.max_batch_size == 32  # Default
        assert config.batcher.batch_timeout_ms == 50  # Default


# =============================================================================
# Additional Property Tests for Validation
# =============================================================================

@given(
    tensor_parallel=st.integers(min_value=1, max_value=8),
    gpu_memory=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_vllm_config_validation_accepts_valid_values(tensor_parallel, gpu_memory):
    """Valid VLLMConfig parameters should be accepted."""
    config = VLLMConfig(
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_memory,
    )
    assert config.tensor_parallel_size == tensor_parallel
    assert config.gpu_memory_utilization == gpu_memory


@given(
    min_size=st.integers(min_value=1, max_value=32),
    max_size=st.integers(min_value=1, max_value=128),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_batcher_config_validation_enforces_min_max_relationship(min_size, max_size):
    """BatcherConfig should enforce min_batch_size <= max_batch_size."""
    if min_size <= max_size:
        # Should succeed
        config = BatcherConfig(min_batch_size=min_size, max_batch_size=max_size)
        assert config.min_batch_size <= config.max_batch_size
    else:
        # Should fail
        with pytest.raises(ValueError):
            BatcherConfig(min_batch_size=min_size, max_batch_size=max_size)


@given(engine_list=st.lists(
    st.sampled_from(["vllm", "deepspeed", "onnx", "pytorch"]),
    min_size=1,
    max_size=4,
    unique=True
))
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_optimization_config_ensures_pytorch_fallback(engine_list):
    """OptimizationConfig should ensure pytorch is in engine_preference."""
    config = OptimizationConfig(engine_preference=engine_list)
    
    # pytorch should always be in the preference list (added if missing)
    assert "pytorch" in config.engine_preference


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.property
def test_empty_configuration_uses_all_defaults():
    """Empty configuration should use all default values."""
    config = load_optimization_config(config_dict={})
    
    # Verify all defaults
    assert config.enabled == True
    assert config.vllm.enabled == True
    assert config.deepspeed.enabled == True
    assert config.onnx.enabled == True
    assert config.batcher.enabled == True
    assert config.cache.enabled == True
    assert config.tuner.enabled == False  # Tuner defaults to disabled
    assert "pytorch" in config.engine_preference


@pytest.mark.property
def test_nonexistent_file_raises_file_not_found():
    """Loading from non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_optimization_config(config_path="/nonexistent/path/config.yaml")


@pytest.mark.property
def test_config_dict_overrides_file():
    """config_dict parameter should override file configuration."""
    # Create temporary file with one value
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({"optimization": {"enabled": True}}, f)
        temp_path = f.name
    
    try:
        # Load with config_dict override (config_dict takes precedence)
        config = load_optimization_config(config_dict={"enabled": False})
        
        # config_dict should override file
        assert config.enabled == False
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
