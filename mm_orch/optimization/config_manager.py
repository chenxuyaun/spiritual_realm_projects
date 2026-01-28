"""
Configuration manager with runtime reload support.

This module provides a ConfigurationManager that supports hot-reloading
of non-critical configuration parameters at runtime without requiring
system restart.
"""

import threading
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from mm_orch.logger import get_logger
from mm_orch.optimization.config import (
    BatcherConfig,
    CacheConfig,
    OptimizationConfig,
    TunerConfig,
    load_optimization_config,
)

logger = get_logger(__name__)


# Non-critical parameters that can be updated at runtime
# Critical parameters (like engine selection, model loading) require restart
NON_CRITICAL_PARAMS = {
    # Batcher parameters
    "batcher.max_batch_size",
    "batcher.batch_timeout_ms",
    "batcher.min_batch_size",
    # Cache parameters
    "cache.max_memory_mb",
    # Tuner parameters
    "tuner.observation_window_seconds",
    "tuner.tuning_interval_seconds",
    "tuner.enable_batch_size_tuning",
    "tuner.enable_timeout_tuning",
    "tuner.enable_cache_size_tuning",
    # Server parameters
    "server.queue_capacity",
    "server.graceful_shutdown_timeout",
}


class ConfigurationChange:
    """Represents a configuration change event."""
    
    def __init__(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        timestamp: Optional[datetime] = None,
    ):
        """
        Initialize configuration change.
        
        Args:
            parameter: Dot-notation parameter path (e.g., "batcher.max_batch_size")
            old_value: Previous value
            new_value: New value
            timestamp: When the change occurred (defaults to now)
        """
        self.parameter = parameter
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self) -> str:
        return (
            f"ConfigurationChange(parameter={self.parameter}, "
            f"old_value={self.old_value}, new_value={self.new_value}, "
            f"timestamp={self.timestamp})"
        )


class ConfigurationManager:
    """
    Manages optimization configuration with runtime reload support.
    
    Supports hot-reloading of non-critical parameters without system restart.
    Critical parameters (engine selection, model paths) require restart.
    
    Thread-safe for concurrent access.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        config_path: Optional[str] = None,
    ):
        """
        Initialize configuration manager.
        
        Args:
            config: Initial optimization configuration
            config_path: Optional path to configuration file for reloading
        """
        self._config = config
        self._config_path = config_path
        self._lock = threading.RLock()
        self._change_history: List[ConfigurationChange] = []
        self._change_callbacks: List[Callable[[ConfigurationChange], None]] = []
        
        logger.info("Configuration manager initialized")
    
    def get_config(self) -> OptimizationConfig:
        """
        Get current configuration (thread-safe).
        
        Returns:
            Current OptimizationConfig instance
        """
        with self._lock:
            return self._config
    
    def reload_config(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> List[ConfigurationChange]:
        """
        Reload configuration from file or dictionary.
        
        Only non-critical parameters are updated. Critical parameters
        (engine selection, model paths) are ignored and logged as warnings.
        
        Args:
            config_path: Path to YAML configuration file (overrides stored path)
            config_dict: Configuration dictionary (overrides file)
        
        Returns:
            List of ConfigurationChange objects for applied changes
        
        Raises:
            ValueError: If configuration validation fails
            FileNotFoundError: If config_path doesn't exist
        
        Example:
            >>> manager = ConfigurationManager(config)
            >>> changes = manager.reload_config("config/optimization.yaml")
            >>> for change in changes:
            ...     print(f"Updated {change.parameter}: {change.old_value} -> {change.new_value}")
        """
        with self._lock:
            # Load new configuration
            path = config_path or self._config_path
            try:
                new_config = load_optimization_config(
                    config_path=path,
                    config_dict=config_dict,
                )
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise
            
            # Validate new configuration
            try:
                self._validate_config(new_config)
            except ValueError as e:
                logger.error(f"Configuration validation failed: {e}")
                raise
            
            # Apply non-critical parameter updates
            changes = self._apply_non_critical_updates(new_config)
            
            # Log changes
            if changes:
                logger.info(f"Applied {len(changes)} configuration changes")
                for change in changes:
                    logger.info(
                        f"  {change.parameter}: {change.old_value} -> {change.new_value}"
                    )
                    self._change_history.append(change)
                    
                    # Notify callbacks
                    for callback in self._change_callbacks:
                        try:
                            callback(change)
                        except Exception as e:
                            logger.error(f"Configuration change callback failed: {e}")
            else:
                logger.info("No configuration changes detected")
            
            return changes
    
    def update_parameter(
        self,
        parameter: str,
        value: Any,
    ) -> Optional[ConfigurationChange]:
        """
        Update a single configuration parameter.
        
        Args:
            parameter: Dot-notation parameter path (e.g., "batcher.max_batch_size")
            value: New value for the parameter
        
        Returns:
            ConfigurationChange if parameter was updated, None if unchanged
        
        Raises:
            ValueError: If parameter is critical or invalid
        
        Example:
            >>> manager.update_parameter("batcher.max_batch_size", 64)
            >>> manager.update_parameter("cache.max_memory_mb", 8192)
        """
        with self._lock:
            # Check if parameter is non-critical
            if parameter not in NON_CRITICAL_PARAMS:
                raise ValueError(
                    f"Parameter '{parameter}' is critical and cannot be updated at runtime. "
                    f"Non-critical parameters: {sorted(NON_CRITICAL_PARAMS)}"
                )
            
            # Get current value
            old_value = self._get_parameter_value(parameter)
            
            # Check if value changed
            if old_value == value:
                logger.debug(f"Parameter {parameter} unchanged: {value}")
                return None
            
            # Validate new value
            self._validate_parameter_value(parameter, value)
            
            # Update parameter
            self._set_parameter_value(parameter, value)
            
            # Create change record
            change = ConfigurationChange(parameter, old_value, value)
            self._change_history.append(change)
            
            logger.info(f"Updated {parameter}: {old_value} -> {value}")
            
            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Configuration change callback failed: {e}")
            
            return change
    
    def register_change_callback(
        self,
        callback: Callable[[ConfigurationChange], None],
    ):
        """
        Register a callback to be notified of configuration changes.
        
        Args:
            callback: Function that accepts ConfigurationChange
        
        Example:
            >>> def on_config_change(change):
            ...     print(f"Config changed: {change.parameter}")
            >>> manager.register_change_callback(on_config_change)
        """
        with self._lock:
            self._change_callbacks.append(callback)
            logger.debug(f"Registered configuration change callback: {callback.__name__}")
    
    def get_change_history(
        self,
        limit: Optional[int] = None,
    ) -> List[ConfigurationChange]:
        """
        Get history of configuration changes.
        
        Args:
            limit: Maximum number of recent changes to return (None for all)
        
        Returns:
            List of ConfigurationChange objects (most recent first)
        """
        with self._lock:
            history = list(reversed(self._change_history))
            if limit:
                history = history[:limit]
            return history
    
    def get_non_critical_parameters(self) -> Set[str]:
        """
        Get set of non-critical parameters that can be updated at runtime.
        
        Returns:
            Set of parameter names in dot notation
        """
        return NON_CRITICAL_PARAMS.copy()
    
    def _validate_config(self, config: OptimizationConfig):
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validation is performed by dataclass __post_init__ methods
        # This method can be extended for additional validation
        pass
    
    def _apply_non_critical_updates(
        self,
        new_config: OptimizationConfig,
    ) -> List[ConfigurationChange]:
        """
        Apply non-critical parameter updates from new configuration.
        
        Args:
            new_config: New configuration to apply
        
        Returns:
            List of ConfigurationChange objects
        """
        changes = []
        
        # Special handling for batcher min/max relationship
        # If both are changing, update them together to avoid validation errors
        batcher_min_param = "batcher.min_batch_size"
        batcher_max_param = "batcher.max_batch_size"
        
        old_min = self._get_parameter_value(batcher_min_param)
        new_min = self._get_parameter_value(batcher_min_param, new_config)
        old_max = self._get_parameter_value(batcher_max_param)
        new_max = self._get_parameter_value(batcher_max_param, new_config)
        
        # If both min and max are changing, update them atomically
        if old_min != new_min or old_max != new_max:
            try:
                # Validate the new min/max relationship
                if new_min > new_max:
                    raise ValueError(f"min_batch_size ({new_min}) must be <= max_batch_size ({new_max})")
                
                # Update both values directly (skip individual validation)
                if old_max != new_max:
                    self._set_parameter_value(batcher_max_param, new_max)
                    changes.append(ConfigurationChange(batcher_max_param, old_max, new_max))
                if old_min != new_min:
                    self._set_parameter_value(batcher_min_param, new_min)
                    changes.append(ConfigurationChange(batcher_min_param, old_min, new_min))
            except ValueError as e:
                logger.warning(f"Skipping batcher min/max update: {e}")
        
        # Check other non-critical parameters
        for param in NON_CRITICAL_PARAMS:
            # Skip batcher min/max as we already handled them
            if param in [batcher_min_param, batcher_max_param]:
                continue
                
            old_value = self._get_parameter_value(param)
            new_value = self._get_parameter_value(param, new_config)
            
            if old_value != new_value:
                try:
                    self._validate_parameter_value(param, new_value)
                    self._set_parameter_value(param, new_value)
                    change = ConfigurationChange(param, old_value, new_value)
                    changes.append(change)
                except ValueError as e:
                    logger.warning(f"Skipping invalid parameter {param}: {e}")
        
        return changes
    
    def _get_parameter_value(
        self,
        parameter: str,
        config: Optional[OptimizationConfig] = None,
    ) -> Any:
        """
        Get parameter value from configuration.
        
        Args:
            parameter: Dot-notation parameter path
            config: Configuration to read from (defaults to current)
        
        Returns:
            Parameter value
        
        Raises:
            ValueError: If parameter path is invalid
        """
        if config is None:
            config = self._config
        
        parts = parameter.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid parameter path: {parameter}")
        
        section, param = parts
        
        # Get section config
        if section == "batcher":
            section_config = config.batcher
        elif section == "cache":
            section_config = config.cache
        elif section == "tuner":
            section_config = config.tuner
        elif section == "server":
            section_config = config.server
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        # Get parameter value
        if not hasattr(section_config, param):
            raise ValueError(f"Unknown parameter: {parameter}")
        
        return getattr(section_config, param)
    
    def _set_parameter_value(self, parameter: str, value: Any):
        """
        Set parameter value in configuration.
        
        Args:
            parameter: Dot-notation parameter path
            value: New value
        
        Raises:
            ValueError: If parameter path is invalid
        """
        parts = parameter.split(".")
        if len(parts) != 2:
            raise ValueError(f"Invalid parameter path: {parameter}")
        
        section, param = parts
        
        # Get section config
        if section == "batcher":
            section_config = self._config.batcher
        elif section == "cache":
            section_config = self._config.cache
        elif section == "tuner":
            section_config = self._config.tuner
        elif section == "server":
            section_config = self._config.server
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        # Set parameter value
        if not hasattr(section_config, param):
            raise ValueError(f"Unknown parameter: {parameter}")
        
        setattr(section_config, param, value)
    
    def _validate_parameter_value(self, parameter: str, value: Any):
        """
        Validate parameter value.
        
        Args:
            parameter: Dot-notation parameter path
            value: Value to validate
        
        Raises:
            ValueError: If value is invalid
        """
        parts = parameter.split(".")
        section, param = parts
        
        # Special validation for batcher min/max relationship
        if section == "batcher":
            current_config = self._config.batcher
            if param == "min_batch_size":
                # Validate against current max_batch_size
                if value > current_config.max_batch_size:
                    raise ValueError(
                        f"min_batch_size ({value}) must be <= max_batch_size ({current_config.max_batch_size})"
                    )
            elif param == "max_batch_size":
                # Validate against current min_batch_size
                if value < current_config.min_batch_size:
                    raise ValueError(
                        f"max_batch_size ({value}) must be >= min_batch_size ({current_config.min_batch_size})"
                    )
        
        # Create temporary config object to validate
        if section == "batcher":
            temp_config = replace(self._config.batcher, **{param: value})
        elif section == "cache":
            temp_config = replace(self._config.cache, **{param: value})
        elif section == "tuner":
            temp_config = replace(self._config.tuner, **{param: value})
        elif section == "server":
            temp_config = replace(self._config.server, **{param: value})
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        # Validation happens in __post_init__


def create_config_manager(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> ConfigurationManager:
    """
    Create a configuration manager with initial configuration.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary (overrides file)
    
    Returns:
        ConfigurationManager instance
    
    Example:
        >>> manager = create_config_manager("config/optimization.yaml")
        >>> config = manager.get_config()
    """
    config = load_optimization_config(
        config_path=config_path,
        config_dict=config_dict,
    )
    
    return ConfigurationManager(config, config_path=config_path)
