"""
Configuration data models for monitoring features.

This module defines configuration dataclasses for Prometheus metrics export,
OpenTelemetry distributed tracing, anomaly detection, and inference server mode.
Supports loading from YAML files with environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from mm_orch.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PrometheusConfig:
    """
    Configuration for Prometheus metrics export.
    
    Attributes:
        enabled: Whether Prometheus metrics export is enabled
        port: HTTP port for metrics endpoint
        host: Host address to bind metrics server
        path: URL path for metrics endpoint
    """
    enabled: bool = True
    port: int = 9090
    host: str = "0.0.0.0"
    path: str = "/metrics"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if not self.path.startswith("/"):
            raise ValueError("path must start with /")


@dataclass
class TracingConfig:
    """
    Configuration for OpenTelemetry distributed tracing.
    
    Attributes:
        enabled: Whether distributed tracing is enabled
        endpoint: OpenTelemetry collector endpoint (OTLP)
        sample_rate: Trace sampling rate (0.0-1.0)
        service_name: Service name for traces
        export_timeout_ms: Timeout for trace export in milliseconds
    """
    enabled: bool = True
    endpoint: str = "http://localhost:4317"
    sample_rate: float = 1.0
    service_name: str = "muai-orchestration"
    export_timeout_ms: int = 30000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        
        if self.export_timeout_ms < 0:
            raise ValueError("export_timeout_ms must be >= 0")
        
        if not self.service_name:
            raise ValueError("service_name cannot be empty")


@dataclass
class AnomalyConfig:
    """
    Configuration for anomaly detection and alerting.
    
    Attributes:
        enabled: Whether anomaly detection is enabled
        latency_threshold_ms: Latency threshold for alerts (milliseconds)
        error_rate_threshold: Error rate threshold (0.0-1.0)
        memory_threshold_percent: Memory usage threshold (0-100)
        throughput_threshold_rps: Minimum throughput threshold (requests/second)
        alert_rate_limit_seconds: Minimum time between alerts of same type
        alert_destinations: List of alert destinations ('log', 'webhook', 'alertmanager')
        webhook_url: Webhook URL for alerts (if webhook destination enabled)
    """
    enabled: bool = True
    latency_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05
    memory_threshold_percent: float = 90.0
    throughput_threshold_rps: float = 1.0
    alert_rate_limit_seconds: int = 300
    alert_destinations: List[str] = field(default_factory=lambda: ["log"])
    webhook_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.latency_threshold_ms < 0:
            raise ValueError("latency_threshold_ms must be >= 0")
        
        if not 0.0 <= self.error_rate_threshold <= 1.0:
            raise ValueError("error_rate_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.memory_threshold_percent <= 100.0:
            raise ValueError("memory_threshold_percent must be between 0 and 100")
        
        if self.throughput_threshold_rps < 0:
            raise ValueError("throughput_threshold_rps must be >= 0")
        
        if self.alert_rate_limit_seconds < 0:
            raise ValueError("alert_rate_limit_seconds must be >= 0")
        
        valid_destinations = {"log", "webhook", "alertmanager"}
        for dest in self.alert_destinations:
            if dest not in valid_destinations:
                raise ValueError(f"Invalid alert destination: {dest}")
        
        if "webhook" in self.alert_destinations and not self.webhook_url:
            logger.warning("webhook destination enabled but webhook_url not set")


@dataclass
class ServerConfig:
    """
    Configuration for inference server mode.
    
    Attributes:
        enabled: Whether server mode is enabled
        host: Host address to bind server
        port: HTTP port for server
        queue_capacity: Maximum request queue size
        preload_models: List of model names to pre-load at startup
        graceful_shutdown_timeout: Timeout for graceful shutdown (seconds)
        health_check_path: URL path for health check endpoint
        readiness_check_path: URL path for readiness check endpoint
    """
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    queue_capacity: int = 100
    preload_models: List[str] = field(default_factory=list)
    graceful_shutdown_timeout: int = 30
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        
        if self.graceful_shutdown_timeout < 0:
            raise ValueError("graceful_shutdown_timeout must be >= 0")
        
        if not self.health_check_path.startswith("/"):
            raise ValueError("health_check_path must start with /")
        
        if not self.readiness_check_path.startswith("/"):
            raise ValueError("readiness_check_path must start with /")


@dataclass
class MonitoringConfig:
    """
    Top-level configuration for monitoring features.
    
    Attributes:
        enabled: Whether monitoring features are enabled globally
        prometheus: Prometheus metrics configuration
        tracing: OpenTelemetry tracing configuration
        anomaly: Anomaly detection configuration
        server: Inference server configuration
        collection_interval_seconds: Interval for metrics collection
        max_metrics_history: Maximum number of historical metrics to retain
    """
    enabled: bool = True
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    collection_interval_seconds: int = 60
    max_metrics_history: int = 1000
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.collection_interval_seconds < 1:
            raise ValueError("collection_interval_seconds must be >= 1")
        
        if self.max_metrics_history < 1:
            raise ValueError("max_metrics_history must be >= 1")


def load_monitoring_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> MonitoringConfig:
    """
    Load monitoring configuration from YAML file or dictionary.
    
    Supports environment variable overrides with MUAI_MON_ prefix.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary (overrides file)
    
    Returns:
        MonitoringConfig instance
    
    Raises:
        FileNotFoundError: If config_path doesn't exist
        ValueError: If configuration is invalid
    
    Environment Variables:
        MUAI_MON_ENABLED: Override monitoring.enabled
        MUAI_MON_PROMETHEUS_ENABLED: Override monitoring.prometheus.enabled
        MUAI_MON_PROMETHEUS_PORT: Override monitoring.prometheus.port
        MUAI_MON_TRACING_ENABLED: Override monitoring.tracing.enabled
        MUAI_MON_TRACING_ENDPOINT: Override monitoring.tracing.endpoint
        MUAI_MON_TRACING_SAMPLE_RATE: Override monitoring.tracing.sample_rate
        MUAI_MON_ANOMALY_ENABLED: Override monitoring.anomaly.enabled
        MUAI_MON_ANOMALY_LATENCY_THRESHOLD: Override monitoring.anomaly.latency_threshold_ms
        MUAI_MON_SERVER_ENABLED: Override monitoring.server.enabled
        MUAI_MON_SERVER_PORT: Override monitoring.server.port
        MUAI_MON_SERVER_QUEUE_CAPACITY: Override monitoring.server.queue_capacity
    
    Example:
        >>> config = load_monitoring_config("config/monitoring.yaml")
        >>> config = load_monitoring_config(config_dict={"enabled": True})
    """
    # Load from file if provided
    if config_path:
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path_obj, "r", encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
        
        # Extract monitoring section
        mon_config = file_config.get("monitoring", {})
    elif config_dict:
        mon_config = config_dict
    else:
        # Use defaults
        mon_config = {}
    
    # Apply environment variable overrides
    mon_config = _apply_env_overrides(mon_config)
    
    # Build configuration objects
    try:
        prometheus_config = PrometheusConfig(**mon_config.get("prometheus", {}))
        tracing_config = TracingConfig(**mon_config.get("tracing", {}))
        anomaly_config = AnomalyConfig(**mon_config.get("anomaly", {}))
        server_config = ServerConfig(**mon_config.get("server", {}))
        
        # Build top-level config
        config = MonitoringConfig(
            enabled=mon_config.get("enabled", True),
            prometheus=prometheus_config,
            tracing=tracing_config,
            anomaly=anomaly_config,
            server=server_config,
            collection_interval_seconds=mon_config.get("collection_interval_seconds", 60),
            max_metrics_history=mon_config.get("max_metrics_history", 1000),
        )
        
        logger.info("Monitoring configuration loaded successfully")
        return config
        
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid monitoring configuration: {e}")
        raise ValueError(f"Configuration validation failed: {e}") from e


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Updated configuration dictionary
    """
    # Top-level overrides
    if "MUAI_MON_ENABLED" in os.environ:
        config["enabled"] = os.environ["MUAI_MON_ENABLED"].lower() in ("true", "1", "yes")
    
    if "MUAI_MON_COLLECTION_INTERVAL" in os.environ:
        config["collection_interval_seconds"] = int(os.environ["MUAI_MON_COLLECTION_INTERVAL"])
    
    # Prometheus overrides
    prometheus_config = config.setdefault("prometheus", {})
    if "MUAI_MON_PROMETHEUS_ENABLED" in os.environ:
        prometheus_config["enabled"] = os.environ["MUAI_MON_PROMETHEUS_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_MON_PROMETHEUS_PORT" in os.environ:
        prometheus_config["port"] = int(os.environ["MUAI_MON_PROMETHEUS_PORT"])
    if "MUAI_MON_PROMETHEUS_HOST" in os.environ:
        prometheus_config["host"] = os.environ["MUAI_MON_PROMETHEUS_HOST"]
    
    # Tracing overrides
    tracing_config = config.setdefault("tracing", {})
    if "MUAI_MON_TRACING_ENABLED" in os.environ:
        tracing_config["enabled"] = os.environ["MUAI_MON_TRACING_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_MON_TRACING_ENDPOINT" in os.environ:
        tracing_config["endpoint"] = os.environ["MUAI_MON_TRACING_ENDPOINT"]
    if "MUAI_MON_TRACING_SAMPLE_RATE" in os.environ:
        tracing_config["sample_rate"] = float(os.environ["MUAI_MON_TRACING_SAMPLE_RATE"])
    if "MUAI_MON_TRACING_SERVICE_NAME" in os.environ:
        tracing_config["service_name"] = os.environ["MUAI_MON_TRACING_SERVICE_NAME"]
    
    # Anomaly detection overrides
    anomaly_config = config.setdefault("anomaly", {})
    if "MUAI_MON_ANOMALY_ENABLED" in os.environ:
        anomaly_config["enabled"] = os.environ["MUAI_MON_ANOMALY_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_MON_ANOMALY_LATENCY_THRESHOLD" in os.environ:
        anomaly_config["latency_threshold_ms"] = float(os.environ["MUAI_MON_ANOMALY_LATENCY_THRESHOLD"])
    if "MUAI_MON_ANOMALY_ERROR_RATE" in os.environ:
        anomaly_config["error_rate_threshold"] = float(os.environ["MUAI_MON_ANOMALY_ERROR_RATE"])
    if "MUAI_MON_ANOMALY_MEMORY_THRESHOLD" in os.environ:
        anomaly_config["memory_threshold_percent"] = float(os.environ["MUAI_MON_ANOMALY_MEMORY_THRESHOLD"])
    if "MUAI_MON_ANOMALY_WEBHOOK_URL" in os.environ:
        anomaly_config["webhook_url"] = os.environ["MUAI_MON_ANOMALY_WEBHOOK_URL"]
    
    # Server mode overrides
    server_config = config.setdefault("server", {})
    if "MUAI_MON_SERVER_ENABLED" in os.environ:
        server_config["enabled"] = os.environ["MUAI_MON_SERVER_ENABLED"].lower() in ("true", "1", "yes")
    if "MUAI_MON_SERVER_PORT" in os.environ:
        server_config["port"] = int(os.environ["MUAI_MON_SERVER_PORT"])
    if "MUAI_MON_SERVER_HOST" in os.environ:
        server_config["host"] = os.environ["MUAI_MON_SERVER_HOST"]
    if "MUAI_MON_SERVER_QUEUE_CAPACITY" in os.environ:
        server_config["queue_capacity"] = int(os.environ["MUAI_MON_SERVER_QUEUE_CAPACITY"])
    if "MUAI_MON_SERVER_PRELOAD_MODELS" in os.environ:
        # Comma-separated list of model names
        models = os.environ["MUAI_MON_SERVER_PRELOAD_MODELS"].split(",")
        server_config["preload_models"] = [m.strip() for m in models if m.strip()]
    
    return config
