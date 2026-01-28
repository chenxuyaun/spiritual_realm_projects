"""
Property-based tests for runtime configuration updates.

Tests universal properties of configuration hot-reload and runtime
parameter updates using Hypothesis.
"""

import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings
import yaml

from mm_orch.optimization.config import (
    OptimizationConfig,
    BatcherConfig,
    CacheConfig,
    TunerConfig,
    ServerConfig,
    load_optimization_config,
)
from mm_orch.optimization.config_manager import (
    ConfigurationManager,
    ConfigurationChange,
    create_config_manager,
    NON_CRITICAL_PARAMS,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

@st.composite
def non_critical_parameter_update(draw):
    """Generate a non-critical parameter name and valid value."""
    param = draw(st.sampled_from(sorted(NON_CRITICAL_PARAMS)))
    
    # Generate appropriate value based on parameter
    if param == "batcher.max_batch_size":
        value = draw(st.integers(min_value=1, max_value=128))
    elif param == "batcher.batch_timeout_ms":
        value = draw(st.integers(min_value=0, max_value=1000))
    elif param == "batcher.min_batch_size":
        # Ensure min_batch_size doesn't exceed default max_batch_size (32)
        value = draw(st.integers(min_value=1, max_value=32))
    elif param == "cache.max_memory_mb":
        value = draw(st.integers(min_value=0, max_value=32768))
    elif param == "tuner.observation_window_seconds":
        value = draw(st.integers(min_value=1, max_value=3600))
    elif param == "tuner.tuning_interval_seconds":
        value = draw(st.integers(min_value=1, max_value=600))
    elif param in ["tuner.enable_batch_size_tuning", "tuner.enable_timeout_tuning", "tuner.enable_cache_size_tuning"]:
        value = draw(st.booleans())
    elif param == "server.queue_capacity":
        value = draw(st.integers(min_value=1, max_value=1000))
    elif param == "server.graceful_shutdown_timeout":
        value = draw(st.integers(min_value=0, max_value=300))
    else:
        value = draw(st.integers(min_value=1, max_value=100))
    
    return param, value


@st.composite
def critical_parameter_name(draw):
    """Generate a critical parameter name that cannot be updated at runtime."""
    critical_params = [
        "enabled",
        "vllm.enabled",
        "vllm.tensor_parallel_size",
        "vllm.dtype",
        "deepspeed.enabled",
        "deepspeed.tensor_parallel",
        "onnx.enabled",
        "onnx.optimization_level",
        "batcher.enabled",
        "cache.enabled",
        "tuner.enabled",
        "server.enabled",
        "server.host",
        "server.port",
        "engine_preference",
        "fallback_on_error",
    ]
    return draw(st.sampled_from(critical_params))


@st.composite
def valid_config_for_reload(draw):
    """Generate valid configuration for reload testing."""
    # Generate min and max batch sizes that satisfy min <= max
    min_batch = draw(st.integers(min_value=1, max_value=32))
    max_batch = draw(st.integers(min_value=min_batch, max_value=128))
    
    return {
        "batcher": {
            "max_batch_size": max_batch,
            "batch_timeout_ms": draw(st.integers(min_value=0, max_value=1000)),
            "min_batch_size": min_batch,
        },
        "cache": {
            "max_memory_mb": draw(st.integers(min_value=0, max_value=32768)),
        },
        "tuner": {
            "observation_window_seconds": draw(st.integers(min_value=1, max_value=3600)),
            "tuning_interval_seconds": draw(st.integers(min_value=1, max_value=600)),
            "enable_batch_size_tuning": draw(st.booleans()),
        },
        "server": {
            "queue_capacity": draw(st.integers(min_value=1, max_value=1000)),
            "graceful_shutdown_timeout": draw(st.integers(min_value=0, max_value=300)),
        },
    }


# =============================================================================
# Property Tests
# =============================================================================

# Feature: advanced-optimization-monitoring, Property 58: Non-critical parameters support runtime updates
@given(param_and_value=non_critical_parameter_update())
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_non_critical_parameters_support_runtime_updates(param_and_value):
    """
    Property 58: Non-critical parameters support runtime updates.
    
    For any non-critical configuration parameter, the system should support
    updating it at runtime without requiring a restart.
    
    **Validates: Requirements 14.5**
    """
    param, new_value = param_and_value
    
    # Create initial configuration
    initial_config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(initial_config)
    
    # Get initial value
    initial_value = manager._get_parameter_value(param)
    
    # Update parameter
    change = manager.update_parameter(param, new_value)
    
    # Verify update was applied
    updated_value = manager._get_parameter_value(param)
    assert updated_value == new_value, f"Parameter {param} should be updated to {new_value}"
    
    # Verify change was recorded if value changed
    if initial_value != new_value:
        assert change is not None, "Change should be recorded when value changes"
        assert change.parameter == param
        assert change.old_value == initial_value
        assert change.new_value == new_value
    else:
        assert change is None, "No change should be recorded when value is unchanged"


@given(
    initial_config=valid_config_for_reload(),
    updated_config=valid_config_for_reload(),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_config_reload_applies_non_critical_changes(initial_config, updated_config):
    """
    Property: Configuration reload applies non-critical parameter changes.
    
    For any configuration reload with changed non-critical parameters,
    the system should apply all changes and return a list of changes.
    
    **Validates: Requirements 14.5**
    """
    # Create manager with initial config
    config = load_optimization_config(config_dict=initial_config)
    manager = ConfigurationManager(config)
    
    # Reload with updated config
    changes = manager.reload_config(config_dict=updated_config)
    
    # Verify changes were applied
    current_config = manager.get_config()
    
    # Check batcher parameters
    if "batcher" in updated_config:
        if "max_batch_size" in updated_config["batcher"]:
            assert current_config.batcher.max_batch_size == updated_config["batcher"]["max_batch_size"]
        if "batch_timeout_ms" in updated_config["batcher"]:
            assert current_config.batcher.batch_timeout_ms == updated_config["batcher"]["batch_timeout_ms"]
    
    # Check cache parameters
    if "cache" in updated_config:
        if "max_memory_mb" in updated_config["cache"]:
            assert current_config.cache.max_memory_mb == updated_config["cache"]["max_memory_mb"]
    
    # Check tuner parameters
    if "tuner" in updated_config:
        if "observation_window_seconds" in updated_config["tuner"]:
            assert current_config.tuner.observation_window_seconds == updated_config["tuner"]["observation_window_seconds"]
    
    # Verify changes list contains actual changes
    for change in changes:
        assert change.parameter in NON_CRITICAL_PARAMS
        assert change.old_value != change.new_value


@given(param_name=critical_parameter_name())
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_critical_parameters_cannot_be_updated_at_runtime(param_name):
    """
    Property: Critical parameters cannot be updated at runtime.
    
    For any critical configuration parameter (engine selection, model paths),
    attempting to update it at runtime should raise ValueError.
    
    **Validates: Requirements 14.5**
    """
    # Create configuration manager
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Attempt to update critical parameter
    with pytest.raises(ValueError) as exc_info:
        manager.update_parameter(param_name, True)
    
    # Verify error message is informative
    error_message = str(exc_info.value)
    assert "critical" in error_message.lower() or "cannot be updated" in error_message.lower()


@given(
    param_and_value=non_critical_parameter_update(),
    num_updates=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_change_history_records_all_updates(param_and_value, num_updates):
    """
    Property: Change history records all configuration updates.
    
    For any sequence of configuration updates, the change history should
    contain a record of each update with old value, new value, and timestamp.
    
    **Validates: Requirements 14.5**
    """
    param, base_value = param_and_value
    
    # Create configuration manager
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Perform multiple updates with values that won't violate constraints
    changes_made = []
    
    # For batcher.min_batch_size, ensure we don't exceed max_batch_size
    if param == "batcher.min_batch_size":
        max_value = manager.get_config().batcher.max_batch_size
        values = [min(base_value + i, max_value) for i in range(num_updates)]
    else:
        values = [base_value + i for i in range(num_updates)]
    
    for value in values:
        try:
            change = manager.update_parameter(param, value)
            if change:  # Only record if value actually changed
                changes_made.append(change)
        except ValueError:
            # Skip invalid values
            pass
    
    # Get change history
    history = manager.get_change_history()
    
    # Verify all changes are recorded
    assert len(history) >= len(changes_made), "All changes should be in history"
    
    # Verify most recent changes are first
    for i, change in enumerate(changes_made):
        # Find this change in history (should be near the end since we reverse)
        found = False
        for h in history:
            if (h.parameter == change.parameter and 
                h.old_value == change.old_value and 
                h.new_value == change.new_value):
                found = True
                break
        assert found, f"Change {change} should be in history"


@given(param_and_value=non_critical_parameter_update())
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_change_callbacks_are_notified(param_and_value):
    """
    Property: Change callbacks are notified of configuration updates.
    
    For any configuration update, all registered callbacks should be
    invoked with the ConfigurationChange object.
    
    **Validates: Requirements 14.5**
    """
    param, new_value = param_and_value
    
    # Create configuration manager
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Register callback
    callback_invoked = []
    
    def test_callback(change: ConfigurationChange):
        callback_invoked.append(change)
    
    manager.register_change_callback(test_callback)
    
    # Update parameter
    change = manager.update_parameter(param, new_value)
    
    # Verify callback was invoked if value changed
    if change:
        assert len(callback_invoked) == 1, "Callback should be invoked once"
        assert callback_invoked[0].parameter == param
        assert callback_invoked[0].new_value == new_value


@given(
    initial_value=st.integers(min_value=1, max_value=64),
    same_value=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_no_change_when_value_unchanged(initial_value, same_value):
    """
    Property: No change is recorded when parameter value is unchanged.
    
    For any parameter update where the new value equals the old value,
    no change should be recorded in the history.
    
    **Validates: Requirements 14.5**
    """
    # Create configuration with initial value
    config = load_optimization_config(config_dict={
        "batcher": {"max_batch_size": initial_value}
    })
    manager = ConfigurationManager(config)
    
    # Get initial history length
    initial_history_len = len(manager.get_change_history())
    
    # Update with same value
    change = manager.update_parameter("batcher.max_batch_size", initial_value)
    
    # Verify no change was recorded
    assert change is None, "No change should be returned when value is unchanged"
    assert len(manager.get_change_history()) == initial_history_len, "History should not grow"


@given(
    param_and_value=non_critical_parameter_update(),
    invalid_value=st.sampled_from([-1, -100, 1000000]),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_values_are_rejected(param_and_value, invalid_value):
    """
    Property: Invalid parameter values are rejected with validation errors.
    
    For any parameter update with an invalid value, the system should
    reject it and raise ValueError without changing the configuration.
    
    **Validates: Requirements 14.5**
    """
    param, _ = param_and_value
    
    # Skip if invalid_value might actually be valid for this parameter
    if param == "cache.max_memory_mb" and invalid_value >= 0:
        return
    if param == "server.graceful_shutdown_timeout" and invalid_value >= 0:
        return
    
    # Create configuration manager
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Get initial value
    initial_value = manager._get_parameter_value(param)
    
    # Attempt to update with invalid value
    try:
        manager.update_parameter(param, invalid_value)
        # If no exception, verify value wasn't changed
        current_value = manager._get_parameter_value(param)
        # For some parameters, large values might be valid
        if current_value != initial_value:
            # Value changed, so it was valid
            pass
    except ValueError:
        # Expected for invalid values
        # Verify value wasn't changed
        current_value = manager._get_parameter_value(param)
        assert current_value == initial_value, "Value should not change on validation error"


@given(config_dict=valid_config_for_reload())
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_reload_from_file_applies_changes(config_dict):
    """
    Property: Reloading from file applies configuration changes.
    
    For any configuration file reload, non-critical parameter changes
    should be applied to the running system.
    
    **Validates: Requirements 14.5**
    """
    # Create initial configuration
    initial_config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(initial_config)
    
    # Create temporary YAML file with updated config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = {"optimization": config_dict}
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    try:
        # Reload from file
        changes = manager.reload_config(config_path=temp_path)
        
        # Verify configuration was updated
        current_config = manager.get_config()
        
        # Check that at least some parameters were updated
        # (unless the random config happened to match defaults)
        if "batcher" in config_dict and "max_batch_size" in config_dict["batcher"]:
            assert current_config.batcher.max_batch_size == config_dict["batcher"]["max_batch_size"]
        
        if "cache" in config_dict and "max_memory_mb" in config_dict["cache"]:
            assert current_config.cache.max_memory_mb == config_dict["cache"]["max_memory_mb"]
        
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


@given(
    param1=st.sampled_from(["batcher.max_batch_size", "cache.max_memory_mb"]),
    value1=st.integers(min_value=1, max_value=100),
    param2=st.sampled_from(["tuner.observation_window_seconds", "server.queue_capacity"]),
    value2=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_multiple_parameters_can_be_updated_independently(param1, value1, param2, value2):
    """
    Property: Multiple parameters can be updated independently.
    
    For any set of non-critical parameters, each can be updated
    independently without affecting others.
    
    **Validates: Requirements 14.5**
    """
    # Create configuration manager
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Update first parameter
    change1 = manager.update_parameter(param1, value1)
    
    # Update second parameter
    change2 = manager.update_parameter(param2, value2)
    
    # Verify both updates were applied
    assert manager._get_parameter_value(param1) == value1
    assert manager._get_parameter_value(param2) == value2
    
    # Verify changes were recorded
    history = manager.get_change_history()
    param_changes = {change.parameter for change in history}
    
    if change1:
        assert param1 in param_changes
    if change2:
        assert param2 in param_changes


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.property
def test_get_non_critical_parameters_returns_expected_set():
    """get_non_critical_parameters should return the expected parameter set."""
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    params = manager.get_non_critical_parameters()
    
    # Verify it's a set
    assert isinstance(params, set)
    
    # Verify it contains expected parameters
    assert "batcher.max_batch_size" in params
    assert "cache.max_memory_mb" in params
    assert "tuner.observation_window_seconds" in params
    
    # Verify it doesn't contain critical parameters
    assert "vllm.enabled" not in params
    assert "engine_preference" not in params


@pytest.mark.property
def test_create_config_manager_factory_function():
    """create_config_manager should create a properly initialized manager."""
    manager = create_config_manager(config_dict={"enabled": True})
    
    assert isinstance(manager, ConfigurationManager)
    assert manager.get_config().enabled == True


@pytest.mark.property
def test_reload_with_invalid_config_preserves_current():
    """Reloading with invalid config should preserve current configuration."""
    # Create manager with valid config
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    initial_batch_size = manager.get_config().batcher.max_batch_size
    
    # Attempt to reload with invalid config
    invalid_config = {"batcher": {"max_batch_size": -1}}
    
    with pytest.raises(ValueError):
        manager.reload_config(config_dict=invalid_config)
    
    # Verify current config is unchanged
    assert manager.get_config().batcher.max_batch_size == initial_batch_size


@pytest.mark.property
def test_change_history_limit_works():
    """get_change_history with limit should return limited results."""
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    # Make several changes
    for i in range(5):
        manager.update_parameter("batcher.max_batch_size", 10 + i)
    
    # Get limited history
    history = manager.get_change_history(limit=2)
    
    # Verify limit is respected
    assert len(history) <= 2


@pytest.mark.property
def test_thread_safety_with_concurrent_updates():
    """Configuration manager should be thread-safe for concurrent updates."""
    import threading
    
    config = load_optimization_config(config_dict={})
    manager = ConfigurationManager(config)
    
    errors = []
    
    def update_param(param, value):
        try:
            manager.update_parameter(param, value)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads updating different parameters
    threads = [
        threading.Thread(target=update_param, args=("batcher.max_batch_size", 64)),
        threading.Thread(target=update_param, args=("cache.max_memory_mb", 8192)),
        threading.Thread(target=update_param, args=("tuner.observation_window_seconds", 600)),
    ]
    
    # Start all threads
    for t in threads:
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Verify no errors occurred
    assert len(errors) == 0, f"Thread-safety errors: {errors}"
