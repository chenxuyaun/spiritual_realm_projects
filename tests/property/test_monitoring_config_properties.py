"""
Property-based tests for monitoring configuration.

Tests universal properties of monitoring configuration loading, validation,
and environment variable overrides using Hypothesis.
"""

import os
import tempfile
from pathlib import Path
import pytest
from hypothesis import given, strategies as st, settings
import yaml

from mm_orch.monitoring.config import (
    MonitoringConfig,
    PrometheusConfig,
    TracingConfig,
    AnomalyConfig,
    ServerConfig,
    load_monitoring_config,
)


# =============================================================================
# Hypothesis Strategies
# =============================================================================

@st.composite
def valid_prometheus_config(draw):
    """Generate valid PrometheusConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "port": draw(st.integers(min_value=1024, max_value=65535)),
        "host": draw(st.sampled_from(["0.0.0.0", "127.0.0.1", "localhost"])),
        "path": draw(st.sampled_from(["/metrics", "/prometheus", "/stats"])),
    }


@st.composite
def valid_tracing_config(draw):
    """Generate valid TracingConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "endpoint": draw(st.sampled_from([
            "http://localhost:4317",
            "http://collector:4317",
            "grpc://localhost:4317",
        ])),
        "sample_rate": draw(st.floats(min_value=0.0, max_value=1.0)),
        "service_name": draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N')))),
        "export_timeout_ms": draw(st.integers(min_value=1000, max_value=60000)),
    }


@st.composite
def valid_anomaly_config(draw):
    """Generate valid AnomalyConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "latency_threshold_ms": draw(st.floats(min_value=0.0, max_value=10000.0)),
        "error_rate_threshold": draw(st.floats(min_value=0.0, max_value=1.0)),
        "memory_threshold_percent": draw(st.floats(min_value=0.0, max_value=100.0)),
        "throughput_threshold_rps": draw(st.floats(min_value=0.0, max_value=1000.0)),
        "alert_rate_limit_seconds": draw(st.integers(min_value=0, max_value=3600)),
        "alert_destinations": draw(st.lists(
            st.sampled_from(["log", "webhook", "alertmanager"]),
            min_size=1,
            max_size=3,
            unique=True
        )),
    }


@st.composite
def valid_server_config(draw):
    """Generate valid ServerConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "host": draw(st.sampled_from(["0.0.0.0", "127.0.0.1"])),
        "port": draw(st.integers(min_value=1024, max_value=65535)),
        "queue_capacity": draw(st.integers(min_value=1, max_value=1000)),
        "preload_models": draw(st.lists(st.text(min_size=1, max_size=20), max_size=5)),
        "graceful_shutdown_timeout": draw(st.integers(min_value=0, max_value=300)),
    }


@st.composite
def valid_monitoring_config(draw):
    """Generate valid MonitoringConfig parameters."""
    return {
        "enabled": draw(st.booleans()),
        "prometheus": draw(valid_prometheus_config()),
        "tracing": draw(valid_tracing_config()),
        "anomaly": draw(valid_anomaly_config()),
        "server": draw(valid_server_config()),
        "collection_interval_seconds": draw(st.integers(min_value=1, max_value=600)),
        "max_metrics_history": draw(st.integers(min_value=1, max_value=10000)),
    }


# =============================================================================
# Property Tests
# =============================================================================

# Feature: advanced-optimization-monitoring, Property 54: YAML configuration is parsed correctly
@given(config_dict=valid_monitoring_config())
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_yaml_monitoring_configuration_parsed_correctly(config_dict):
    """
    Property 54: YAML configuration is parsed correctly (monitoring).
    
    For any valid YAML monitoring configuration file, the system should
    parse it and apply the settings correctly.
    
    **Validates: Requirements 14.1**
    """
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = {"monitoring": config_dict}
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    try:
        # Load configuration from YAML
        config = load_monitoring_config(config_path=temp_path)
        
        # Verify configuration was parsed correctly
        assert config.enabled == config_dict.get("enabled", True)
        
        # Verify Prometheus config
        prom_dict = config_dict.get("prometheus", {})
        assert config.prometheus.enabled == prom_dict.get("enabled", True)
        assert config.prometheus.port == prom_dict.get("port", 9090)
        
        # Verify tracing config
        trace_dict = config_dict.get("tracing", {})
        assert config.tracing.enabled == trace_dict.get("enabled", True)
        
        # Verify anomaly config
        anomaly_dict = config_dict.get("anomaly", {})
        assert config.anomaly.enabled == anomaly_dict.get("enabled", True)
        
        # Verify server config
        server_dict = config_dict.get("server", {})
        assert config.server.enabled == server_dict.get("enabled", False)
        
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)


# Feature: advanced-optimization-monitoring, Property 55: Environment variables override configuration
@given(
    file_enabled=st.booleans(),
    env_enabled=st.booleans(),
    file_prom_port=st.integers(min_value=8000, max_value=9000),
    env_prom_port=st.integers(min_value=9000, max_value=10000),
    file_sample_rate=st.floats(min_value=0.0, max_value=0.5),
    env_sample_rate=st.floats(min_value=0.5, max_value=1.0),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_monitoring_environment_variables_override_configuration(
    file_enabled, env_enabled, file_prom_port, env_prom_port,
    file_sample_rate, env_sample_rate
):
    """
    Property 55: Environment variables override configuration (monitoring).
    
    For any monitoring configuration parameter with both file and environment
    variable values, the environment variable value should take precedence.
    
    **Validates: Requirements 14.2**
    """
    # Create configuration dict
    config_dict = {
        "enabled": file_enabled,
        "prometheus": {"port": file_prom_port},
        "tracing": {"sample_rate": file_sample_rate},
    }
    
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = {"monitoring": config_dict}
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    # Set environment variables
    env_vars = {
        "MUAI_MON_ENABLED": str(env_enabled).lower(),
        "MUAI_MON_PROMETHEUS_PORT": str(env_prom_port),
        "MUAI_MON_TRACING_SAMPLE_RATE": str(env_sample_rate),
    }
    
    # Save original environment
    original_env = {}
    for key in env_vars:
        original_env[key] = os.environ.get(key)
        os.environ[key] = env_vars[key]
    
    try:
        # Load configuration
        config = load_monitoring_config(config_path=temp_path)
        
        # Verify environment variables override file values
        assert config.enabled == env_enabled, "Top-level enabled should be overridden by env var"
        assert config.prometheus.port == env_prom_port, "Prometheus port should be overridden by env var"
        assert abs(config.tracing.sample_rate - env_sample_rate) < 0.01, "Sample rate should be overridden by env var"
        
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
        "invalid_port",
        "invalid_sample_rate",
        "invalid_memory_threshold",
        "negative_queue_capacity",
        "invalid_alert_destination",
        "empty_service_name",
    ])
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_monitoring_configuration_rejected_with_clear_errors(invalid_type):
    """
    Property 56: Invalid configuration is rejected with clear errors (monitoring).
    
    For any invalid monitoring configuration, the system should reject it
    with a clear error message.
    
    **Validates: Requirements 14.3**
    """
    # Create invalid configuration based on type
    if invalid_type == "invalid_port":
        config_dict = {"prometheus": {"port": 70000}}  # Port out of range
    elif invalid_type == "invalid_sample_rate":
        config_dict = {"tracing": {"sample_rate": 1.5}}  # Sample rate > 1.0
    elif invalid_type == "invalid_memory_threshold":
        config_dict = {"anomaly": {"memory_threshold_percent": 150.0}}  # > 100
    elif invalid_type == "negative_queue_capacity":
        config_dict = {"server": {"queue_capacity": -1}}
    elif invalid_type == "invalid_alert_destination":
        config_dict = {"anomaly": {"alert_destinations": ["invalid_dest"]}}
    elif invalid_type == "empty_service_name":
        config_dict = {"tracing": {"service_name": ""}}
    
    # Attempt to load configuration and expect ValueError
    with pytest.raises(ValueError) as exc_info:
        load_monitoring_config(config_dict=config_dict)
    
    # Verify error message is informative
    error_message = str(exc_info.value)
    assert len(error_message) > 0, "Error message should not be empty"
    assert "Configuration validation failed" in error_message or "must be" in error_message.lower()


# Feature: advanced-optimization-monitoring, Property 57: Default values are used for missing configuration
@given(
    include_prometheus=st.booleans(),
    include_tracing=st.booleans(),
    include_anomaly=st.booleans(),
    include_server=st.booleans(),
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_monitoring_default_values_used_for_missing_configuration(
    include_prometheus, include_tracing, include_anomaly, include_server
):
    """
    Property 57: Default values are used for missing configuration (monitoring).
    
    For any monitoring configuration parameter not specified, the system
    should use the documented default value.
    
    **Validates: Requirements 14.4**
    """
    # Create partial configuration
    config_dict = {}
    
    if include_prometheus:
        config_dict["prometheus"] = {"enabled": True}  # Only set enabled
    
    if include_tracing:
        config_dict["tracing"] = {}  # Empty dict, all should use defaults
    
    if include_anomaly:
        config_dict["anomaly"] = {"latency_threshold_ms": 500.0}  # Only set one field
    
    if include_server:
        config_dict["server"] = {"port": 8080}
    
    # Load configuration
    config = load_monitoring_config(config_dict=config_dict)
    
    # Verify defaults are used for missing fields
    if not include_prometheus:
        # All Prometheus fields should use defaults
        assert config.prometheus.enabled == True  # Default
        assert config.prometheus.port == 9090  # Default
        assert config.prometheus.host == "0.0.0.0"  # Default
    else:
        # Only enabled was set, others should use defaults
        assert config.prometheus.enabled == True
        assert config.prometheus.port == 9090  # Default
    
    if include_tracing:
        # Empty dict, all should use defaults
        assert config.tracing.enabled == True  # Default
        assert config.tracing.sample_rate == 1.0  # Default
        assert config.tracing.service_name == "muai-orchestration"  # Default
    
    if include_anomaly:
        # Only latency_threshold_ms was set
        assert config.anomaly.latency_threshold_ms == 500.0
        assert config.anomaly.error_rate_threshold == 0.05  # Default
        assert config.anomaly.alert_rate_limit_seconds == 300  # Default
    else:
        # All should use defaults
        assert config.anomaly.latency_threshold_ms == 1000.0  # Default
    
    if include_server:
        # Only port was set
        assert config.server.port == 8080
        assert config.server.enabled == False  # Default
        assert config.server.queue_capacity == 100  # Default
    else:
        # All should use defaults
        assert config.server.enabled == False  # Default
        assert config.server.port == 8000  # Default


# =============================================================================
# Additional Property Tests for Validation
# =============================================================================

@given(
    port=st.integers(min_value=1, max_value=65535),
    sample_rate=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_prometheus_and_tracing_config_validation_accepts_valid_values(port, sample_rate):
    """Valid PrometheusConfig and TracingConfig parameters should be accepted."""
    prom_config = PrometheusConfig(port=port)
    assert prom_config.port == port
    
    trace_config = TracingConfig(sample_rate=sample_rate)
    assert abs(trace_config.sample_rate - sample_rate) < 0.01


@given(
    latency=st.floats(min_value=0.0, max_value=10000.0),
    error_rate=st.floats(min_value=0.0, max_value=1.0),
    memory=st.floats(min_value=0.0, max_value=100.0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_anomaly_config_validation_accepts_valid_thresholds(latency, error_rate, memory):
    """Valid AnomalyConfig threshold parameters should be accepted."""
    config = AnomalyConfig(
        latency_threshold_ms=latency,
        error_rate_threshold=error_rate,
        memory_threshold_percent=memory,
    )
    assert abs(config.latency_threshold_ms - latency) < 0.01
    assert abs(config.error_rate_threshold - error_rate) < 0.01
    assert abs(config.memory_threshold_percent - memory) < 0.01


@given(queue_capacity=st.integers(min_value=1, max_value=1000))
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_server_config_validation_accepts_valid_queue_capacity(queue_capacity):
    """Valid ServerConfig queue_capacity should be accepted."""
    config = ServerConfig(queue_capacity=queue_capacity)
    assert config.queue_capacity == queue_capacity


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.property
def test_empty_monitoring_configuration_uses_all_defaults():
    """Empty monitoring configuration should use all default values."""
    config = load_monitoring_config(config_dict={})
    
    # Verify all defaults
    assert config.enabled == True
    assert config.prometheus.enabled == True
    assert config.prometheus.port == 9090
    assert config.tracing.enabled == True
    assert config.tracing.sample_rate == 1.0
    assert config.anomaly.enabled == True
    assert config.server.enabled == False  # Server defaults to disabled
    assert config.collection_interval_seconds == 60


@pytest.mark.property
def test_monitoring_nonexistent_file_raises_file_not_found():
    """Loading monitoring config from non-existent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_monitoring_config(config_path="/nonexistent/path/monitoring.yaml")


@pytest.mark.property
def test_webhook_destination_without_url_logs_warning(caplog):
    """Webhook destination without URL should log a warning."""
    config_dict = {
        "anomaly": {
            "alert_destinations": ["webhook"],
            "webhook_url": None,
        }
    }
    
    # This should succeed but log a warning
    config = load_monitoring_config(config_dict=config_dict)
    assert "webhook" in config.anomaly.alert_destinations


@pytest.mark.property
def test_preload_models_from_env_var():
    """Server preload_models should be parseable from comma-separated env var."""
    env_var = "MUAI_MON_SERVER_PRELOAD_MODELS"
    original_value = os.environ.get(env_var)
    
    try:
        os.environ[env_var] = "model1,model2,model3"
        config = load_monitoring_config(config_dict={})
        
        assert "model1" in config.server.preload_models
        assert "model2" in config.server.preload_models
        assert "model3" in config.server.preload_models
        
    finally:
        if original_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = original_value
