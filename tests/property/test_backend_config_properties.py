"""
Property-based tests for BackendConfig.

Feature: openvino-backend-integration
"""

import os
import tempfile
import yaml
import pytest
from hypothesis import given, strategies as st, settings
from mm_orch.runtime.backend_config import BackendConfig


# Strategy for valid backend names
valid_backends = st.sampled_from(["pytorch", "openvino"])

# Strategy for valid OpenVINO devices
valid_ov_devices = st.sampled_from(["CPU", "GPU", "AUTO"])

# Strategy for valid PyTorch devices
valid_pt_devices = st.sampled_from(["cpu", "cuda", "auto"])

# Strategy for valid dtypes
valid_dtypes = st.sampled_from(["float32", "float16", "bfloat16"])

# Strategy for boolean values
booleans = st.booleans()

# Strategy for positive integers
positive_ints = st.integers(min_value=1, max_value=10)

# Strategy for model names
model_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=ord('a')),
    min_size=1,
    max_size=20
)


def create_temp_config(config_dict):
    """Helper to create temporary config file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    yaml.dump(config_dict, temp_file)
    temp_file.close()
    return temp_file.name


@settings(max_examples=100, deadline=None)
@given(
    default_backend=valid_backends,
    ov_device=valid_ov_devices,
    pt_device=valid_pt_devices,
    enable_fallback=booleans,
    num_streams=positive_ints,
    dtype=valid_dtypes
)
def test_property_2_configuration_loading_and_validation_valid_config(
    default_backend,
    ov_device,
    pt_device,
    enable_fallback,
    num_streams,
    dtype
):
    """
    Property 2: Configuration Loading and Validation
    
    For any valid configuration file content, the system should correctly
    parse and apply backend settings.
    
    Validates: Requirements 2.1, 2.2, 2.6
    """
    # Create valid configuration
    config_dict = {
        "backend": {
            "default": default_backend,
            "openvino": {
                "device": ov_device,
                "enable_fallback": enable_fallback,
                "cache_dir": "models/openvino",
                "num_streams": num_streams
            },
            "pytorch": {
                "device": pt_device,
                "dtype": dtype
            }
        },
        "model_overrides": {}
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        # Load configuration
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify default backend is correctly loaded
        assert backend_config.get_default_backend() == default_backend
        
        # Verify OpenVINO config is correctly loaded
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == ov_device
        assert ov_config["enable_fallback"] == enable_fallback
        assert ov_config["num_streams"] == num_streams
        
        # Verify PyTorch config is correctly loaded
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["device"] == pt_device
        assert pt_config["dtype"] == dtype
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


@settings(max_examples=100, deadline=None)
@given(
    invalid_backend=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["pytorch", "openvino"]),
    invalid_ov_device=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["CPU", "GPU", "AUTO"]),
    invalid_pt_device=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["cpu", "cuda", "auto"])
)
def test_property_2_configuration_validation_invalid_values(
    invalid_backend,
    invalid_ov_device,
    invalid_pt_device
):
    """
    Property 2: Configuration Loading and Validation
    
    For any invalid configuration, the system should either reject it with
    clear errors or fall back to safe defaults.
    
    Validates: Requirements 2.1, 2.2, 2.6
    """
    # Test invalid default backend - should fall back to 'pytorch'
    config_dict = {
        "backend": {
            "default": invalid_backend,
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
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
    
    # Test invalid OpenVINO device - should fall back to 'CPU'
    config_dict = {
        "backend": {
            "default": "openvino",
            "openvino": {
                "device": invalid_ov_device,
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
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
    
    # Test invalid PyTorch device - should fall back to 'cpu'
    config_dict = {
        "backend": {
            "default": "pytorch",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": invalid_pt_device
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


@settings(max_examples=100, deadline=None)
@given(
    random_suffix=st.integers(min_value=0, max_value=1000000)
)
def test_property_2_missing_config_file_uses_defaults(random_suffix):
    """
    Property 2: Configuration Loading and Validation
    
    When configuration file is missing, the system should use safe defaults.
    
    Validates: Requirements 2.1, 2.2, 2.6
    """
    # Use a non-existent path
    non_existent_path = f"non_existent_config_{random_suffix}.yaml"
    
    backend_config = BackendConfig(config_path=non_existent_path)
    
    # Should use default values
    assert backend_config.get_default_backend() == "pytorch"
    
    ov_config = backend_config.get_backend_config("openvino")
    assert ov_config["device"] == "CPU"
    assert ov_config["enable_fallback"] is True
    assert ov_config["cache_dir"] == "models/openvino"
    assert ov_config["num_streams"] == 1
    
    pt_config = backend_config.get_backend_config("pytorch")
    assert pt_config["device"] == "cpu"
    assert pt_config["dtype"] == "float32"


@settings(max_examples=100, deadline=None)
@given(st.random_module())
def test_property_2_empty_config_file_uses_defaults(random_module):
    """
    Property 2: Configuration Loading and Validation
    
    When configuration file is empty, the system should use safe defaults.
    
    Validates: Requirements 2.1, 2.2, 2.6
    """
    # Create empty config file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    temp_file.write("")
    temp_file.close()
    
    try:
        backend_config = BackendConfig(config_path=temp_file.name)
        
        # Should use default values
        assert backend_config.get_default_backend() == "pytorch"
        
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == "CPU"
        assert ov_config["enable_fallback"] is True
        
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["device"] == "cpu"
        assert pt_config["dtype"] == "float32"
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@settings(max_examples=100, deadline=None)
@given(
    invalid_num_streams=st.one_of(
        st.integers(max_value=0),
        st.text(min_size=1, max_size=10),
        st.floats(allow_nan=False, allow_infinity=False)
    )
)
def test_property_2_invalid_num_streams_falls_back_to_default(invalid_num_streams):
    """
    Property 2: Configuration Loading and Validation
    
    When num_streams is invalid, the system should fall back to default value of 1.
    
    Validates: Requirements 2.1, 2.2, 2.6
    """
    config_dict = {
        "backend": {
            "default": "openvino",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True,
                "num_streams": invalid_num_streams
            },
            "pytorch": {
                "device": "cpu"
            }
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



@settings(max_examples=100, deadline=None)
@given(
    default_backend=valid_backends,
    model_name=model_names,
    override_backend=valid_backends
)
def test_property_3_per_model_backend_override(
    default_backend,
    model_name,
    override_backend
):
    """
    Property 3: Per-Model Backend Override
    
    For any model with a backend override in configuration, the system should
    use the override backend instead of the default backend for that specific model.
    
    Validates: Requirements 2.3
    """
    # Create configuration with model override
    config_dict = {
        "backend": {
            "default": default_backend,
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {
            model_name: override_backend
        }
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify default backend is correctly loaded
        assert backend_config.get_default_backend() == default_backend
        
        # Verify model override is correctly loaded
        model_backend = backend_config.get_model_backend(model_name)
        assert model_backend == override_backend
        
        # Verify that a different model name returns None (no override)
        other_model = f"other_{model_name}"
        assert backend_config.get_model_backend(other_model) is None
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


@settings(max_examples=100, deadline=None)
@given(
    default_backend=valid_backends,
    model_names_list=st.lists(model_names, min_size=1, max_size=5, unique=True),
    override_backends=st.lists(valid_backends, min_size=1, max_size=5)
)
def test_property_3_multiple_model_overrides(
    default_backend,
    model_names_list,
    override_backends
):
    """
    Property 3: Per-Model Backend Override
    
    For multiple models with backend overrides, each model should use its
    specific override backend.
    
    Validates: Requirements 2.3
    """
    # Create overrides for multiple models
    model_overrides = {}
    for i, model_name in enumerate(model_names_list):
        backend_idx = i % len(override_backends)
        model_overrides[model_name] = override_backends[backend_idx]
    
    config_dict = {
        "backend": {
            "default": default_backend,
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": model_overrides
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify each model override is correctly loaded
        for model_name, expected_backend in model_overrides.items():
            actual_backend = backend_config.get_model_backend(model_name)
            assert actual_backend == expected_backend
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


@settings(max_examples=100, deadline=None)
@given(
    default_backend=valid_backends,
    model_name=model_names,
    invalid_backend=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["pytorch", "openvino"])
)
def test_property_3_invalid_model_override_is_removed(
    default_backend,
    model_name,
    invalid_backend
):
    """
    Property 3: Per-Model Backend Override
    
    For any model with an invalid backend override, the system should remove
    the invalid override and fall back to the default backend.
    
    Validates: Requirements 2.3
    """
    # Create configuration with invalid model override
    config_dict = {
        "backend": {
            "default": default_backend,
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {
            model_name: invalid_backend
        }
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify invalid override is removed (returns None)
        model_backend = backend_config.get_model_backend(model_name)
        assert model_backend is None
        
        # Default backend should still be correct
        assert backend_config.get_default_backend() == default_backend
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)



@settings(max_examples=100, deadline=None)
@given(
    ov_device=valid_ov_devices,
    pt_device=valid_pt_devices
)
def test_property_4_device_configuration_consistency(
    ov_device,
    pt_device
):
    """
    Property 4: Device Configuration Consistency
    
    For any valid device configuration (CPU, GPU, AUTO), the OpenVINO backend
    should respect the device setting and use it for model loading and inference.
    
    Validates: Requirements 2.4, 6.1
    """
    # Create configuration with specific devices
    config_dict = {
        "backend": {
            "default": "openvino",
            "openvino": {
                "device": ov_device,
                "enable_fallback": True
            },
            "pytorch": {
                "device": pt_device
            }
        },
        "model_overrides": {}
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify OpenVINO device is correctly loaded
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == ov_device
        
        # Verify PyTorch device is correctly loaded
        pt_config = backend_config.get_backend_config("pytorch")
        assert pt_config["device"] == pt_device
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


@settings(max_examples=100, deadline=None)
@given(
    default_backend=valid_backends,
    ov_device=valid_ov_devices,
    enable_fallback=booleans,
    num_streams=positive_ints
)
def test_property_4_openvino_device_config_preserved(
    default_backend,
    ov_device,
    enable_fallback,
    num_streams
):
    """
    Property 4: Device Configuration Consistency
    
    For any OpenVINO configuration, all device-related settings should be
    preserved and accessible through the configuration API.
    
    Validates: Requirements 2.4, 6.1
    """
    config_dict = {
        "backend": {
            "default": default_backend,
            "openvino": {
                "device": ov_device,
                "enable_fallback": enable_fallback,
                "cache_dir": "models/openvino",
                "num_streams": num_streams
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {}
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        
        # Verify all OpenVINO settings are preserved
        ov_config = backend_config.get_backend_config("openvino")
        assert ov_config["device"] == ov_device
        assert ov_config["enable_fallback"] == enable_fallback
        assert ov_config["cache_dir"] == "models/openvino"
        assert ov_config["num_streams"] == num_streams
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)


@settings(max_examples=100, deadline=None)
@given(
    invalid_ov_device=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["CPU", "GPU", "AUTO"]),
    invalid_pt_device=st.text(
        alphabet=st.characters(whitelist_categories=("Ll",)),
        min_size=1,
        max_size=20
    ).filter(lambda x: x not in ["cpu", "cuda", "auto"])
)
def test_property_4_invalid_device_falls_back_to_default(
    invalid_ov_device,
    invalid_pt_device
):
    """
    Property 4: Device Configuration Consistency
    
    For any invalid device configuration, the system should fall back to
    safe default devices (CPU for OpenVINO, cpu for PyTorch).
    
    Validates: Requirements 2.4, 6.1
    """
    # Test invalid OpenVINO device
    config_dict = {
        "backend": {
            "default": "openvino",
            "openvino": {
                "device": invalid_ov_device,
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
        }
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        ov_config = backend_config.get_backend_config("openvino")
        # Should fall back to CPU
        assert ov_config["device"] == "CPU"
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    # Test invalid PyTorch device
    config_dict = {
        "backend": {
            "default": "pytorch",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": invalid_pt_device
            }
        }
    }
    
    config_path = create_temp_config(config_dict)
    
    try:
        backend_config = BackendConfig(config_path=config_path)
        pt_config = backend_config.get_backend_config("pytorch")
        # Should fall back to cpu
        assert pt_config["device"] == "cpu"
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)
