"""Configuration management system for MuAI Orchestration System.

This module provides configuration loading, validation, and default value handling
for both YAML and JSON configuration files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass


class ConfigValidationError(ConfigError):
    """Exception raised when configuration validation fails."""
    pass


class ConfigLoader:
    """Configuration loader with support for YAML and JSON formats.
    
    Supports:
    - Loading from YAML and JSON files
    - Configuration validation
    - Default value fallback
    - Nested configuration access
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        "system": {
            "log_level": "INFO",
            "max_cached_models": 3,
            "development_stage": "adult",
            "device": "auto",
        },
        "models": {},
        "storage": {
            "vector_db_path": "data/vector_db",
            "chat_history_path": "data/chat_history",
            "consciousness_state_path": ".consciousness/state.json",
        },
        "router": {
            "confidence_threshold": 0.6,
            "enable_logging": True,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "enable_auth": False,
        },
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the configuration loader.
        
        Args:
            config_path: Path to configuration file (YAML or JSON).
                        If None, uses default configuration only.
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or use defaults."""
        # Start with default configuration
        self._config = self._deep_copy_dict(self.DEFAULT_CONFIG)
        
        # If config file provided, load and merge
        if self.config_path and self.config_path.exists():
            try:
                loaded_config = self._load_from_file(self.config_path)
                self._merge_config(loaded_config)
            except Exception as e:
                raise ConfigError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _load_from_file(self, path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            ConfigError: If file format is unsupported or parsing fails
        """
        suffix = path.suffix.lower()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif suffix == '.json':
                    return json.load(f)
                else:
                    raise ConfigError(f"Unsupported configuration format: {suffix}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to parse configuration file: {e}")
    
    def _merge_config(self, loaded_config: Dict[str, Any]) -> None:
        """Merge loaded configuration with defaults.
        
        Args:
            loaded_config: Configuration loaded from file
        """
        self._deep_merge(self._config, loaded_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge override dictionary into base dictionary.
        
        Args:
            base: Base dictionary to merge into
            override: Dictionary with values to override
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary.
        
        Args:
            d: Dictionary to copy
            
        Returns:
            Deep copy of the dictionary
        """
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Supports nested keys using dot notation (e.g., "system.log_level").
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self._deep_copy_dict(self._config)
    
    def validate(self) -> bool:
        """Validate the configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate required top-level keys
        required_keys = ["system", "storage"]
        for key in required_keys:
            if key not in self._config:
                raise ConfigValidationError(f"Missing required configuration key: {key}")
        
        # Validate system configuration
        system_config = self._config.get("system", {})
        if "log_level" in system_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if system_config["log_level"] not in valid_levels:
                raise ConfigValidationError(
                    f"Invalid log_level: {system_config['log_level']}. "
                    f"Must be one of {valid_levels}"
                )
        
        if "max_cached_models" in system_config:
            if not isinstance(system_config["max_cached_models"], int) or \
               system_config["max_cached_models"] < 1:
                raise ConfigValidationError(
                    "max_cached_models must be a positive integer"
                )
        
        if "device" in system_config:
            valid_devices = ["auto", "cuda", "cpu"]
            if system_config["device"] not in valid_devices:
                raise ConfigValidationError(
                    f"Invalid device: {system_config['device']}. "
                    f"Must be one of {valid_devices}"
                )
        
        return True
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
    
    def __repr__(self) -> str:
        """String representation of ConfigLoader."""
        return f"ConfigLoader(config_path={self.config_path})"
