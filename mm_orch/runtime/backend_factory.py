"""
Backend factory for creating inference backend instances.

This module provides a factory class for creating and managing inference backends
(PyTorch, OpenVINO, etc.) based on configuration and availability.
"""

import logging
from typing import Dict, Any, List

from mm_orch.runtime.inference_backend import InferenceBackend
from mm_orch.runtime.backend_exceptions import (
    BackendInitializationError,
    ConfigurationError,
)

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
            ConfigurationError: If backend_type is not 'pytorch' or 'openvino'.
            BackendInitializationError: If backend initialization fails.
            
        Example:
            >>> factory = BackendFactory()
            >>> backend = factory.create_backend('pytorch', 'cpu', {})
            >>> # or
            >>> backend = factory.create_backend('openvino', 'CPU', {'enable_fallback': True})
        """
        # Validate backend_type parameter
        if backend_type not in ['pytorch', 'openvino']:
            available = self.get_available_backends()
            raise ConfigurationError(
                f"Invalid backend type: '{backend_type}'.\n\n"
                f"Supported backend types: 'pytorch', 'openvino'\n"
                f"Available backends on this system: {', '.join(available) if available else '(none)'}\n\n"
                f"Backend descriptions:\n"
                f"  - pytorch: Use PyTorch for inference (CPU or CUDA)\n"
                f"  - openvino: Use OpenVINO for accelerated CPU/GPU inference\n\n"
                f"Installation instructions:\n"
                f"  PyTorch:  pip install torch\n"
                f"  OpenVINO: pip install openvino openvino-dev\n\n"
                f"Example usage:\n"
                f"  factory = BackendFactory()\n"
                f"  backend = factory.create_backend('pytorch', 'cpu', {{}})"
            )
        
        # Create PyTorch backend
        if backend_type == 'pytorch':
            try:
                from mm_orch.runtime.pytorch_backend import PyTorchBackend
                logger.info(f"Creating PyTorch backend with device: {device}")
                return PyTorchBackend(device, config)
            except ImportError as e:
                raise BackendInitializationError(
                    f"Failed to create PyTorch backend: PyTorch is not installed.\n\n"
                    f"Installation instructions:\n"
                    f"1. Install PyTorch (CPU version):\n"
                    f"   pip install torch torchvision torchaudio\n\n"
                    f"2. Install PyTorch (GPU version with CUDA):\n"
                    f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n\n"
                    f"3. Verify installation:\n"
                    f"   python -c \"import torch; print(torch.__version__)\"\n\n"
                    f"For more information, visit:\n"
                    f"https://pytorch.org/get-started/locally/"
                ) from e
            except Exception as e:
                error_type = type(e).__name__
                raise BackendInitializationError(
                    f"Failed to initialize PyTorch backend.\n\n"
                    f"Error type: {error_type}\n"
                    f"Error details: {str(e)}\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Verify PyTorch is properly installed:\n"
                    f"   pip install --upgrade torch\n\n"
                    f"2. Check device parameter is valid:\n"
                    f"   - For CPU: device='cpu'\n"
                    f"   - For GPU: device='cuda' (requires CUDA installation)\n\n"
                    f"3. Check for conflicting packages:\n"
                    f"   pip list | grep torch"
                ) from e
        
        # Create OpenVINO backend
        elif backend_type == 'openvino':
            try:
                from mm_orch.runtime.openvino_backend import OpenVINOBackend
                logger.info(f"Creating OpenVINO backend with device: {device}")
                return OpenVINOBackend(device, config)
            except ImportError as e:
                raise BackendInitializationError(
                    f"Failed to create OpenVINO backend: OpenVINO is not installed.\n\n"
                    f"Installation instructions:\n"
                    f"1. Install OpenVINO runtime:\n"
                    f"   pip install openvino\n\n"
                    f"2. Install OpenVINO development tools (for model export):\n"
                    f"   pip install openvino-dev\n\n"
                    f"3. Install Optimum Intel (for transformers integration):\n"
                    f"   pip install optimum[openvino]\n\n"
                    f"4. Verify installation:\n"
                    f"   python -c \"import openvino; print(openvino.__version__)\"\n\n"
                    f"Alternative: Use PyTorch backend instead:\n"
                    f"  factory.create_backend('pytorch', 'cpu', {{}})\n\n"
                    f"For more information, visit:\n"
                    f"https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
                ) from e
            except Exception as e:
                error_type = type(e).__name__
                raise BackendInitializationError(
                    f"Failed to initialize OpenVINO backend.\n\n"
                    f"Error type: {error_type}\n"
                    f"Error details: {str(e)}\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Verify OpenVINO is properly installed:\n"
                    f"   pip install --upgrade openvino openvino-dev\n\n"
                    f"2. Check device parameter is valid for OpenVINO:\n"
                    f"   - CPU: Always available\n"
                    f"   - GPU: Requires Intel GPU drivers\n"
                    f"   - AUTO: Let OpenVINO choose automatically\n"
                    f"   - NPU: Intel Core Ultra processors only\n\n"
                    f"3. Enable fallback to PyTorch:\n"
                    f"   config = {{'enable_fallback': True}}\n"
                    f"   backend = factory.create_backend('openvino', 'CPU', config)\n\n"
                    f"4. Use PyTorch backend as alternative:\n"
                    f"   backend = factory.create_backend('pytorch', 'cpu', {{}})"
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
