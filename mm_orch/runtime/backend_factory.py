"""
Backend factory for creating inference backend instances.

This module provides a factory class for creating and managing inference backends
(PyTorch, OpenVINO, etc.) based on configuration and availability.
"""

import logging
from typing import Dict, Any, List

from mm_orch.runtime.inference_backend import InferenceBackend

logger = logging.getLogger(__name__)


class BackendFactory:
    """
    Factory for creating inference backend instances.
    
    This factory handles the creation of different backend types (PyTorch, OpenVINO)
    and provides methods to detect which backends are available on the current system.
    
    Example:
        >>> factory = BackendFactory()
        >>> available = factory.get_available_backends()
        >>> print(available)  # ['pytorch', 'openvino']
        >>> backend = factory.create_backend('pytorch', 'cpu', {})
    """
    
    def create_backend(
        self,
        backend_type: str,
        device: str,
        config: Dict[str, Any]
    ) -> InferenceBackend:
        """
        Create an inference backend instance.
        
        Args:
            backend_type: Type of backend to create ('pytorch' or 'openvino').
            device: Target device for inference (e.g., 'cpu', 'cuda', 'GPU', 'AUTO').
            config: Backend-specific configuration dictionary.
            
        Returns:
            An instance of InferenceBackend (PyTorchBackend or OpenVINOBackend).
            
        Raises:
            ValueError: If backend_type is not 'pytorch' or 'openvino'.
            RuntimeError: If backend initialization fails.
            
        Example:
            >>> factory = BackendFactory()
            >>> backend = factory.create_backend('pytorch', 'cpu', {})
            >>> # or
            >>> backend = factory.create_backend('openvino', 'CPU', {'enable_fallback': True})
        """
        # Validate backend_type parameter
        if backend_type not in ['pytorch', 'openvino']:
            raise ValueError(
                f"Invalid backend type: '{backend_type}'. "
                f"Supported backends: 'pytorch', 'openvino'. "
                f"Available backends on this system: {self.get_available_backends()}"
            )
        
        # Create PyTorch backend
        if backend_type == 'pytorch':
            try:
                from mm_orch.runtime.pytorch_backend import PyTorchBackend
                logger.info(f"Creating PyTorch backend with device: {device}")
                return PyTorchBackend(device, config)
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to create PyTorch backend: {e}. "
                    f"Please ensure PyTorch is installed: pip install torch"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize PyTorch backend: {e}"
                ) from e
        
        # Create OpenVINO backend
        elif backend_type == 'openvino':
            try:
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                logger.info(f"Creating OpenVINO backend with device: {device}")
                return OpenVINOBackend(device, config)
            except ImportError as e:
                raise RuntimeError(
                    f"Failed to create OpenVINO backend: {e}. "
                    f"OpenVINO is not installed. Install with: "
                    f"pip install openvino openvino-dev"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize OpenVINO backend: {e}"
                ) from e
    
    def get_available_backends(self) -> List[str]:
        """
        Detect which inference backends are available on this system.
        
        This method checks for the availability of PyTorch and OpenVINO by attempting
        to import their respective modules. It returns a list of backend names that
        can be used with create_backend().
        
        Returns:
            List of available backend names (e.g., ['pytorch', 'openvino']).
            
        Example:
            >>> factory = BackendFactory()
            >>> available = factory.get_available_backends()
            >>> if 'openvino' in available:
            ...     backend = factory.create_backend('openvino', 'CPU', {})
            ... else:
            ...     backend = factory.create_backend('pytorch', 'cpu', {})
        """
        available = []
        
        # Check PyTorch availability
        try:
            import torch
            available.append('pytorch')
            logger.debug("PyTorch backend is available")
        except ImportError:
            logger.debug("PyTorch backend is not available")
        
        # Check OpenVINO availability
        try:
            import openvino
            available.append('openvino')
            logger.debug("OpenVINO backend is available")
        except ImportError:
            logger.debug("OpenVINO backend is not available")
        
        if not available:
            logger.warning(
                "No inference backends are available. "
                "Please install PyTorch (pip install torch) or "
                "OpenVINO (pip install openvino openvino-dev)"
            )
        
        return available
