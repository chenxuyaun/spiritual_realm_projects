"""
Backend configuration management for inference backends.

This module provides configuration loading, validation, and access for
PyTorch and OpenVINO backends.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

from mm_orch.runtime.backend_exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class BackendConfig:
    """Backend configuration container."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize backend configuration.
        
        Args:
            config_path: Path to configuration file. Defaults to config/system.yaml
        """
        self.config_path = config_path or "config/system.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If configuration file is severely malformed
        """
        if not os.path.exists(self.config_path):
            logger.info(
                f"Configuration file not found: {self.config_path}\n"
                f"Using default configuration. To customize, create:\n"
                f"  {self.config_path}\n\n"
                f"Example configuration:\n"
                f"  backend:\n"
                f"    default: pytorch\n"
                f"    openvino:\n"
                f"      device: CPU\n"
                f"      enable_fallback: true"
            )
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                logger.warning(
                    f"Empty configuration file: {self.config_path}\n"
                    f"Using default configuration."
                )
                return self._get_default_config()
            
            return self._validate_config(config)
            
        except yaml.YAMLError as e:
            error_msg = (
                f"YAML parsing error in configuration file: {self.config_path}\n\n"
                f"Error details: {str(e)}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Check YAML syntax (indentation, colons, quotes)\n"
                f"2. Validate YAML online: https://www.yamllint.com/\n"
                f"3. Compare with example configuration:\n"
                f"   backend:\n"
                f"     default: pytorch\n"
                f"     openvino:\n"
                f"       device: CPU\n"
                f"       enable_fallback: true\n\n"
                f"4. Backup and recreate the file if needed"
            )
            logger.warning(error_msg)
            logger.info("Using default configuration due to YAML error")
            return self._get_default_config()
            
        except Exception as e:
            error_type = type(e).__name__
            logger.warning(
                f"Configuration loading failed: {error_type}: {str(e)}\n"
                f"Using default configuration."
            )
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary with safe defaults
        """
        return {
            "backend": {
                "default": "pytorch",
                "openvino": {
                    "device": "CPU",
                    "enable_fallback": True,
                    "cache_dir": "models/openvino",
                    "num_streams": 1
                },
                "pytorch": {
                    "device": "cpu",
                    "dtype": "float32"
                }
            },
            "model_overrides": {}
        }
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration values.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration dictionary
        """
        # Ensure backend section exists
        if "backend" not in config:
            logger.warning("Missing 'backend' section in config, using defaults")
            config["backend"] = self._get_default_config()["backend"]
        
        backend_config = config["backend"]
        
        # Validate default backend
        default_backend = backend_config.get("default", "pytorch")
        if default_backend not in ["pytorch", "openvino"]:
            logger.warning(
                f"Invalid default backend: {default_backend}, "
                f"must be 'pytorch' or 'openvino'. Using 'pytorch'."
            )
            backend_config["default"] = "pytorch"
        
        # Validate OpenVINO configuration
        if "openvino" in backend_config:
            ov_config = backend_config["openvino"]
            
            # Validate device
            ov_device = ov_config.get("device", "CPU")
            valid_devices = ["CPU", "GPU", "AUTO"]
            if ov_device not in valid_devices:
                logger.warning(
                    f"Invalid OpenVINO device: {ov_device}, "
                    f"must be one of {valid_devices}. Using 'CPU'."
                )
                ov_config["device"] = "CPU"
            
            # Validate enable_fallback
            if "enable_fallback" in ov_config:
                if not isinstance(ov_config["enable_fallback"], bool):
                    logger.warning(
                        f"Invalid enable_fallback value: {ov_config['enable_fallback']}, "
                        f"must be boolean. Using True."
                    )
                    ov_config["enable_fallback"] = True
            else:
                ov_config["enable_fallback"] = True
            
            # Validate cache_dir
            if "cache_dir" not in ov_config:
                ov_config["cache_dir"] = "models/openvino"
            
            # Validate num_streams
            if "num_streams" in ov_config:
                try:
                    # Convert to int, but reject floats that aren't whole numbers
                    num_streams_value = ov_config["num_streams"]
                    if isinstance(num_streams_value, float):
                        # Reject floats - they should be integers
                        logger.warning(
                            f"Invalid num_streams value: {num_streams_value}, "
                            f"must be integer. Using 1."
                        )
                        ov_config["num_streams"] = 1
                    else:
                        num_streams = int(num_streams_value)
                        if num_streams < 1:
                            logger.warning(
                                f"Invalid num_streams: {num_streams}, must be >= 1. Using 1."
                            )
                            ov_config["num_streams"] = 1
                        else:
                            ov_config["num_streams"] = num_streams
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid num_streams value: {ov_config['num_streams']}, "
                        f"must be integer. Using 1."
                    )
                    ov_config["num_streams"] = 1
            else:
                ov_config["num_streams"] = 1
        else:
            # Add default OpenVINO config if missing
            backend_config["openvino"] = self._get_default_config()["backend"]["openvino"]
        
        # Validate PyTorch configuration
        if "pytorch" in backend_config:
            pt_config = backend_config["pytorch"]
            
            # Validate device
            pt_device = pt_config.get("device", "cpu")
            valid_pt_devices = ["cpu", "cuda", "auto"]
            if pt_device not in valid_pt_devices:
                logger.warning(
                    f"Invalid PyTorch device: {pt_device}, "
                    f"must be one of {valid_pt_devices}. Using 'cpu'."
                )
                pt_config["device"] = "cpu"
            
            # Validate dtype
            if "dtype" in pt_config:
                valid_dtypes = ["float32", "float16", "bfloat16"]
                if pt_config["dtype"] not in valid_dtypes:
                    logger.warning(
                        f"Invalid PyTorch dtype: {pt_config['dtype']}, "
                        f"must be one of {valid_dtypes}. Using 'float32'."
                    )
                    pt_config["dtype"] = "float32"
            else:
                pt_config["dtype"] = "float32"
        else:
            # Add default PyTorch config if missing
            backend_config["pytorch"] = self._get_default_config()["backend"]["pytorch"]
        
        # Validate model_overrides
        if "model_overrides" not in config:
            config["model_overrides"] = {}
        
        if not isinstance(config["model_overrides"], dict):
            logger.warning(
                f"Invalid model_overrides type: {type(config['model_overrides'])}, "
                f"must be dict. Using empty dict."
            )
            config["model_overrides"] = {}
        
        # Validate each model override
        for model_name, backend in list(config["model_overrides"].items()):
            if backend not in ["pytorch", "openvino"]:
                logger.warning(
                    f"Invalid backend override for model '{model_name}': {backend}, "
                    f"must be 'pytorch' or 'openvino'. Removing override."
                )
                del config["model_overrides"][model_name]
        
        return config
    
    def get_default_backend(self) -> str:
        """
        Get default backend name.
        
        Returns:
            Default backend name ('pytorch' or 'openvino')
        """
        return self._config["backend"]["default"]
    
    def get_backend_config(self, backend: str) -> Dict[str, Any]:
        """
        Get configuration for specific backend.
        
        Args:
            backend: Backend name ('pytorch' or 'openvino')
            
        Returns:
            Backend-specific configuration dictionary
        """
        return self._config["backend"].get(backend, {})
    
    def get_model_backend(self, model_name: str) -> Optional[str]:
        """
        Get backend override for specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Backend name if override exists, None otherwise
        """
        return self._config.get("model_overrides", {}).get(model_name)
