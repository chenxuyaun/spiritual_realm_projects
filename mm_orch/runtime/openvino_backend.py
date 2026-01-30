"""
OpenVINO inference backend implementation.

This module provides an OpenVINO-based implementation of the InferenceBackend interface,
with automatic fallback to PyTorch when OpenVINO operations fail.
"""

import logging
import os
from typing import Any, Dict, Optional, List

from mm_orch.runtime.inference_backend import InferenceBackend
from mm_orch.runtime.backend_exceptions import (
    BackendInitializationError,
    ConfigurationError,
    DeviceError,
    FileValidationError,
    InferenceError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


class OpenVINOBackend(InferenceBackend):
    """
    OpenVINO inference backend with automatic PyTorch fallback.
    
    This backend uses OpenVINO for accelerated inference on CPU/GPU/NPU devices,
    with automatic fallback to PyTorch when OpenVINO operations fail. This ensures
    robustness while providing 2-3x performance improvements when OpenVINO is available.
    
    Requirements:
        - 1.4: Verify OpenVINO availability before initialization
        - 4.1: Automatic fallback to PyTorch on OpenVINO failures
        - 4.3: Log warnings on fallback with failure reason
        - 4.5: Support disabling fallback via configuration
    
    Example:
        >>> # With fallback enabled (default)
        >>> backend = OpenVINOBackend(device='CPU', config={'enable_fallback': True})
        >>> model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
        >>> 
        >>> # With fallback disabled
        >>> backend = OpenVINOBackend(device='CPU', config={'enable_fallback': False})
    """
    
    def __init__(self, device: str, config: Dict[str, Any]):
        """
        Initialize OpenVINO backend with optional PyTorch fallback.
        
        Args:
            device: Target device for OpenVINO ('CPU', 'GPU', 'AUTO', 'NPU').
            config: Backend-specific configuration:
                - enable_fallback (bool): Enable automatic PyTorch fallback (default: True)
                - cache_dir (str): Directory for OpenVINO model cache
                - num_streams (int): Number of parallel inference streams
        
        Raises:
            RuntimeError: If OpenVINO initialization fails and fallback is disabled.
            ValueError: If device parameter is invalid.
        """
        # Validate and normalize device parameter (Requirement 6.1)
        validated_device = self._validate_device(device)
        
        super().__init__(validated_device, config)
        
        self._openvino_manager = None
        self._fallback_backend = None
        self._fallback_enabled = config.get('enable_fallback', True)
        self._original_device = device  # Store original device request
        
        # Initialize OpenVINO with fallback handling
        self._initialize_openvino()
        
        logger.info(
            f"Initialized OpenVINO backend with device: {self.device}, "
            f"fallback_enabled: {self._fallback_enabled}"
        )
    
    def _validate_device(self, device: str) -> str:
        """
        Validate and normalize device parameter.
        
        Validates that the device parameter is one of the supported OpenVINO devices.
        Normalizes the device string to uppercase for consistency.
        
        Requirements:
            - 6.1: Validate device parameter in __init__
            - 9.2: Add available devices to device selection errors
        
        Args:
            device: Device string to validate.
            
        Returns:
            Normalized device string (uppercase).
            
        Raises:
            ConfigurationError: If device is not a supported OpenVINO device.
            
        Example:
            >>> backend._validate_device('cpu')
            'CPU'
            >>> backend._validate_device('GPU')
            'GPU'
            >>> backend._validate_device('invalid')
            ConfigurationError: Invalid device 'invalid'. Supported devices: CPU, GPU, AUTO, NPU
        """
        # Normalize to uppercase
        normalized_device = device.upper()
        
        # List of supported OpenVINO devices
        supported_devices = ['CPU', 'GPU', 'AUTO', 'NPU']
        
        if normalized_device not in supported_devices:
            # Try to get available devices on this system
            try:
                import openvino as ov
                core = ov.Core()
                available_devices = core.available_devices
                available_info = f"\n\nDevices available on this system:\n  " + "\n  ".join(available_devices) if available_devices else "\n\n(No devices detected)"
            except Exception:
                available_info = "\n\n(Could not detect available devices - OpenVINO may not be installed)"
            
            raise ConfigurationError(
                f"Invalid device '{device}' for OpenVINO backend.\n\n"
                f"Supported device names: {', '.join(supported_devices)}{available_info}\n\n"
                f"Device descriptions:\n"
                f"  - CPU: Use CPU for inference (always available)\n"
                f"  - GPU: Use integrated or discrete GPU (requires GPU drivers)\n"
                f"  - AUTO: Let OpenVINO automatically select the best device\n"
                f"  - NPU: Use Neural Processing Unit (Intel Core Ultra only)\n\n"
                f"Example usage:\n"
                f"  backend = OpenVINOBackend('CPU', config)\n"
                f"  backend = OpenVINOBackend('AUTO', config)"
            )
        
        return normalized_device
    
    def _check_device_availability(self, device: str) -> bool:
        """
        Check if a specific device is available on the system.
        
        Queries OpenVINO to determine if the requested device is available.
        This helps provide early feedback about device availability.
        
        Requirements:
            - 6.1: Implement device availability checking
        
        Args:
            device: Device to check ('CPU', 'GPU', 'AUTO', 'NPU').
            
        Returns:
            True if device is available, False otherwise.
            
        Example:
            >>> backend._check_device_availability('CPU')
            True
            >>> backend._check_device_availability('GPU')
            False  # If no GPU present
        """
        # CPU is always available
        if device == 'CPU':
            return True
        
        # AUTO lets OpenVINO choose, so it's always "available"
        if device == 'AUTO':
            return True
        
        # For GPU and NPU, we need to check with OpenVINO
        try:
            import openvino as ov
            core = ov.Core()
            available_devices = core.available_devices
            
            # Check if the requested device is in the list
            # OpenVINO returns devices like 'GPU.0', 'NPU.0', etc.
            for available_device in available_devices:
                if available_device.startswith(device):
                    return True
            
            return False
            
        except ImportError:
            # If OpenVINO is not installed, we can't check
            logger.warning("Cannot check device availability: OpenVINO not installed")
            return False
        except Exception as e:
            # If there's any error checking, assume device is not available
            logger.warning(f"Error checking device availability: {e}")
            return False
    
    def _fallback_device_to_cpu(self, requested_device: str) -> str:
        """
        Fallback to CPU device if requested device is unavailable.
        
        Checks if the requested device is available. If not, falls back to CPU
        and logs a warning.
        
        Requirements:
            - 6.3: Add device fallback to CPU on unavailability
            - 6.3: Log warnings on device fallback
        
        Args:
            requested_device: The device that was requested.
            
        Returns:
            'CPU' if fallback occurred, otherwise the requested device.
            
        Example:
            >>> backend._fallback_device_to_cpu('GPU')
            'CPU'  # If GPU not available, with warning logged
            >>> backend._fallback_device_to_cpu('CPU')
            'CPU'  # No fallback needed
        """
        # Check if device is available
        if not self._check_device_availability(requested_device):
            logger.warning(
                f"Device '{requested_device}' is not available on this system. "
                f"Falling back to CPU device. "
                f"Available devices can be checked with: "
                f"python -c \"import openvino as ov; print(ov.Core().available_devices)\""
            )
            return 'CPU'
        
        return requested_device
    
    def _initialize_openvino(self) -> None:
        """
        Initialize OpenVINO manager with automatic fallback to PyTorch.
        
        Attempts to initialize the OpenVINO_Manager. If initialization fails
        and fallback is enabled, creates a PyTorchBackend as fallback. If fallback
        is disabled, raises the original error.
        
        Also checks device availability and falls back to CPU if the requested
        device is not available.
        
        Requirements:
            - 6.3: Device fallback to CPU on unavailability
            - 1.5: Provide clear error messages with installation instructions
        
        Raises:
            BackendInitializationError: If OpenVINO initialization fails and fallback is disabled.
        """
        try:
            # Attempt to import and initialize OpenVINO manager
            from mm_orch.runtime.openvino_manager import OpenVINOModelManager
            
            # Check device availability and fallback to CPU if needed (Requirement 6.3)
            actual_device = self._fallback_device_to_cpu(self.device)
            
            # Update device if fallback occurred
            if actual_device != self.device:
                self.device = actual_device
            
            self._openvino_manager = OpenVINOModelManager(
                default_device=self.device,
                enable_openvino=True,
                fallback_to_pytorch=False,  # We handle fallback at backend level
                openvino_cache_dir=self.config.get('cache_dir', 'models/openvino')
            )
            
            logger.info("OpenVINO manager initialized successfully")
            
        except ImportError as e:
            error_msg = (
                f"OpenVINO initialization failed: OpenVINO library not found.\n\n"
                f"OpenVINO is not installed on this system.\n\n"
                f"Installation instructions:\n"
                f"1. Install OpenVINO runtime:\n"
                f"   pip install openvino\n\n"
                f"2. Install OpenVINO development tools (for model export):\n"
                f"   pip install openvino-dev\n\n"
                f"3. Install Optimum Intel (for transformers integration):\n"
                f"   pip install optimum[openvino]\n\n"
                f"4. Verify installation:\n"
                f"   python -c \"import openvino; print(openvino.__version__)\"\n\n"
                f"For more information, visit:\n"
                f"https://docs.openvino.ai/latest/openvino_docs_install_guides_overview.html"
            )
            logger.warning(error_msg)
            
            if self._fallback_enabled:
                logger.info("Fallback to PyTorch is enabled, initializing PyTorch backend")
                self._initialize_fallback_backend()
            else:
                raise BackendInitializationError(error_msg) from e
                
        except Exception as e:
            error_type = type(e).__name__
            error_msg = (
                f"OpenVINO initialization failed: {error_type}: {str(e)}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Verify OpenVINO is properly installed:\n"
                f"   pip install --upgrade openvino openvino-dev\n\n"
                f"2. Check for conflicting package versions:\n"
                f"   pip list | grep openvino\n\n"
                f"3. Try reinstalling OpenVINO:\n"
                f"   pip uninstall openvino openvino-dev\n"
                f"   pip install openvino openvino-dev\n\n"
                f"4. Check system requirements:\n"
                f"   - Python 3.8 or later\n"
                f"   - 64-bit operating system\n"
                f"   - Sufficient RAM (4GB minimum)\n\n"
                f"5. Enable fallback to PyTorch in configuration:\n"
                f"   backend:\n"
                f"     openvino:\n"
                f"       enable_fallback: true"
            )
            logger.warning(error_msg)
            
            if self._fallback_enabled:
                logger.info("Fallback to PyTorch is enabled, initializing PyTorch backend")
                self._initialize_fallback_backend()
            else:
                raise BackendInitializationError(error_msg) from e
    
    def _initialize_fallback_backend(self) -> None:
        """
        Initialize PyTorch fallback backend.
        
        Creates a PyTorchBackend instance to use when OpenVINO operations fail.
        The fallback backend uses CPU device for maximum compatibility.
        """
        try:
            from mm_orch.runtime.pytorch_backend import PyTorchBackend
            
            # Use CPU for fallback to ensure compatibility
            fallback_device = 'cpu' if self.device.upper() != 'CUDA' else 'cuda'
            
            self._fallback_backend = PyTorchBackend(
                device=fallback_device,
                config=self.config
            )
            
            logger.info(
                f"Initialized PyTorch fallback backend with device: {fallback_device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback backend: {e}")
            raise RuntimeError(
                f"Both OpenVINO and PyTorch fallback initialization failed: {e}"
            ) from e
    
    def _get_openvino_path(self, pytorch_path: str) -> str:
        """
        Convert PyTorch model path to OpenVINO IR model path.
        
        Follows the convention: models/{model_name} -> models/openvino/{model_name}
        
        Args:
            pytorch_path: Path to PyTorch model directory.
            
        Returns:
            Path to OpenVINO IR model directory.
            
        Example:
            >>> backend._get_openvino_path('models/gpt2')
            'models/openvino/gpt2'
            >>> backend._get_openvino_path('/path/to/models/t5-small')
            '/path/to/models/openvino/t5-small'
        """
        # Split path into directory and model name
        base_dir = os.path.dirname(pytorch_path)
        model_name = os.path.basename(pytorch_path)
        
        # Construct OpenVINO path
        openvino_path = os.path.join(base_dir, 'openvino', model_name)
        
        return openvino_path
    
    def _validate_model_files(self, model_name: str, openvino_path: str) -> None:
        """
        Validate that all required OpenVINO model files exist.
        
        Checks for the presence of required OpenVINO IR files (.xml and .bin)
        and provides detailed error messages with troubleshooting steps if files
        are missing.
        
        Requirements:
            - 9.3: Check model XML and BIN files exist before loading
            - 9.3: Provide clear error messages for missing files
            - 9.3: Include troubleshooting suggestions in errors
        
        Args:
            model_name: Name of the model being validated.
            openvino_path: Path to the OpenVINO model directory.
            
        Raises:
            FileValidationError: If required model files are missing or invalid.
            
        Example:
            >>> backend._validate_model_files('gpt2', 'models/openvino/gpt2')
            # Raises FileValidationError if files are missing
        """
        # Define required files
        xml_file = os.path.join(openvino_path, 'openvino_model.xml')
        bin_file = os.path.join(openvino_path, 'openvino_model.bin')
        
        # Check if directory exists
        if not os.path.exists(openvino_path):
            raise FileValidationError(
                f"OpenVINO model directory not found: {openvino_path}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Export the model to OpenVINO format:\n"
                f"   python scripts/export_to_openvino.py {model_name}\n\n"
                f"2. Verify the model exists in PyTorch format first\n\n"
                f"3. Check that the export script completed successfully\n\n"
                f"4. Ensure you have write permissions to the models directory"
            )
        
        # Check if XML file exists
        if not os.path.exists(xml_file):
            # List what files are actually present
            try:
                present_files = os.listdir(openvino_path)
                files_info = f"\nFiles present in {openvino_path}:\n  " + "\n  ".join(present_files) if present_files else "\n(directory is empty)"
            except Exception:
                files_info = ""
            
            raise FileValidationError(
                f"OpenVINO model XML file not found: {xml_file}\n\n"
                f"The model directory exists but the XML file is missing.{files_info}\n\n"
                f"Troubleshooting steps:\n"
                f"1. Re-export the model (export may have been incomplete):\n"
                f"   python scripts/export_to_openvino.py {model_name}\n\n"
                f"2. Check for export errors in the logs\n\n"
                f"3. Verify you have the correct model name\n\n"
                f"4. Ensure the export script supports this model type"
            )
        
        # Check if BIN file exists
        if not os.path.exists(bin_file):
            raise FileValidationError(
                f"OpenVINO model weights file not found: {bin_file}\n\n"
                f"The XML file exists but the weights file is missing. "
                f"This indicates an incomplete or corrupted export.\n\n"
                f"Troubleshooting steps:\n"
                f"1. Delete the incomplete export:\n"
                f"   rm -rf {openvino_path}\n\n"
                f"2. Re-export the model:\n"
                f"   python scripts/export_to_openvino.py {model_name}\n\n"
                f"3. Check available disk space (large models need several GB)\n\n"
                f"4. Verify the export completed without errors"
            )
        
        # Validate file sizes (basic sanity check)
        xml_size = os.path.getsize(xml_file)
        bin_size = os.path.getsize(bin_file)
        
        if xml_size == 0:
            raise FileValidationError(
                f"OpenVINO model XML file is empty: {xml_file}\n\n"
                f"The file exists but has zero bytes, indicating a failed export.\n\n"
                f"Troubleshooting steps:\n"
                f"1. Delete the corrupted export:\n"
                f"   rm -rf {openvino_path}\n\n"
                f"2. Re-export the model:\n"
                f"   python scripts/export_to_openvino.py {model_name}\n\n"
                f"3. Check for disk space or permission issues"
            )
        
        if bin_size == 0:
            raise FileValidationError(
                f"OpenVINO model weights file is empty: {bin_file}\n\n"
                f"The file exists but has zero bytes, indicating a failed export.\n\n"
                f"Troubleshooting steps:\n"
                f"1. Delete the corrupted export:\n"
                f"   rm -rf {openvino_path}\n\n"
                f"2. Re-export the model:\n"
                f"   python scripts/export_to_openvino.py {model_name}\n\n"
                f"3. Check for disk space issues (model weights can be several GB)"
            )
        
        # Log successful validation
        logger.debug(
            f"Model file validation passed for {model_name}: "
            f"XML={xml_size} bytes, BIN={bin_size} bytes"
        )
    
    def _is_fallback_model(self, model: Any) -> bool:
        """
        Check if a model is from the fallback PyTorch backend.
        
        Determines whether a model object was loaded by the fallback backend
        by checking for OpenVINO-specific attributes.
        
        Args:
            model: Model object to check.
            
        Returns:
            True if model is from fallback backend, False if from OpenVINO.
            
        Example:
            >>> pytorch_model = backend._fallback_backend.load_model(...)
            >>> backend._is_fallback_model(pytorch_model)
            True
            >>> openvino_model = backend._openvino_manager.load_model(...)
            >>> backend._is_fallback_model(openvino_model)
            False
        """
        if self._fallback_backend is None:
            return False
        
        # OpenVINO models have specific attributes that PyTorch models don't
        # Check for OpenVINO-specific attributes
        has_openvino_attrs = (
            hasattr(model, 'compiled_model') or
            hasattr(model, 'request') or
            hasattr(model, '_openvino_config')
        )
        
        return not has_openvino_attrs
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str
    ) -> Any:
        """
        Load a model using OpenVINO with automatic fallback to PyTorch.
        
        Attempts to load the model using OpenVINO. If OpenVINO is not available
        or loading fails, automatically falls back to PyTorch (if enabled).
        
        Requirements:
            - 4.1: Automatic fallback to PyTorch on OpenVINO failure
            - 4.3: Log warnings on fallback with failure reason
            - 9.3: Validate OpenVINO model files exist before loading
        
        Args:
            model_name: Unique identifier for the model.
            model_path: Path to the model files (PyTorch format).
            model_type: Type of model (e.g., 'transformers').
            
        Returns:
            Loaded model object (OpenVINO or PyTorch).
            
        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails and fallback is disabled.
            
        Example:
            >>> backend = OpenVINOBackend('CPU', {'enable_fallback': True})
            >>> model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
            >>> # If OpenVINO fails, automatically uses PyTorch
        """
        # If OpenVINO manager is not available, use fallback immediately
        if self._openvino_manager is None:
            if self._fallback_backend:
                logger.info(
                    f"OpenVINO not available, using PyTorch fallback for {model_name}"
                )
                return self._fallback_backend.load_model(
                    model_name, model_path, model_type
                )
            raise BackendInitializationError(
                f"OpenVINO backend not initialized and fallback is disabled.\n\n"
                f"The OpenVINO backend failed to initialize. This usually means:\n"
                f"1. OpenVINO is not installed\n"
                f"2. OpenVINO installation is corrupted\n"
                f"3. System requirements are not met\n\n"
                f"Solutions:\n"
                f"1. Install OpenVINO:\n"
                f"   pip install openvino openvino-dev\n\n"
                f"2. Enable fallback to PyTorch:\n"
                f"   backend:\n"
                f"     openvino:\n"
                f"       enable_fallback: true\n\n"
                f"3. Use PyTorch backend directly:\n"
                f"   backend = 'pytorch'"
            )
        
        # Try loading with OpenVINO
        try:
            # Convert PyTorch path to OpenVINO path
            openvino_path = self._get_openvino_path(model_path)
            
            # Validate OpenVINO model files exist (Requirement 9.3)
            # This provides detailed error messages with troubleshooting steps
            self._validate_model_files(model_name, openvino_path)
            
            logger.info(
                f"Loading model with OpenVINO: {model_name} from {openvino_path}"
            )
            
            # Register model config with OpenVINO manager
            from mm_orch.schemas import ModelConfig
            config = ModelConfig(
                name=model_name,
                model_path=model_path,
                model_type=model_type
            )
            self._openvino_manager.register_model(config)
            
            # Load model using OpenVINO manager
            cached_model = self._openvino_manager.load_model(
                model_name=model_name,
                device=self.device,
                force_backend='openvino'
            )
            
            # Store in internal registry
            self._models[model_name] = cached_model.model
            
            logger.info(
                f"Successfully loaded model {model_name} with OpenVINO on device {self.device}"
            )
            
            return cached_model.model
            
        except FileValidationError as e:
            # File validation errors have detailed troubleshooting info
            logger.warning(
                f"OpenVINO model file validation failed for {model_name}: {e}"
            )
            
            if self._fallback_backend:
                logger.info(
                    f"Falling back to PyTorch for {model_name} "
                    f"(reason: OpenVINO model files not found or invalid)"
                )
                return self._fallback_backend.load_model(
                    model_name, model_path, model_type
                )
            else:
                # Re-raise with additional context
                raise ModelLoadError(
                    f"Failed to load OpenVINO model '{model_name}' and fallback is disabled.\n\n"
                    f"Original error:\n{str(e)}"
                ) from e
                
        except FileNotFoundError as e:
            # Legacy file not found errors (shouldn't happen with new validation)
            logger.warning(
                f"OpenVINO model files not found for {model_name}: {e}"
            )
            
            if self._fallback_backend:
                logger.info(
                    f"Falling back to PyTorch for {model_name} "
                    f"(reason: OpenVINO model files not found)"
                )
                return self._fallback_backend.load_model(
                    model_name, model_path, model_type
                )
            else:
                raise ModelLoadError(
                    f"OpenVINO model files not found for '{model_name}' and fallback is disabled.\n\n"
                    f"Export the model first:\n"
                    f"  python scripts/export_to_openvino.py {model_name}"
                ) from e
                
        except Exception as e:
            # Other errors should also trigger fallback if enabled
            error_type = type(e).__name__
            logger.warning(
                f"OpenVINO model loading failed for {model_name}: {error_type}: {e}"
            )
            
            if self._fallback_backend:
                logger.info(
                    f"Falling back to PyTorch for {model_name} "
                    f"(reason: {error_type}: {str(e)})"
                )
                return self._fallback_backend.load_model(
                    model_name, model_path, model_type
                )
            else:
                raise ModelLoadError(
                    f"OpenVINO model loading failed for '{model_name}' and fallback is disabled.\n\n"
                    f"Error type: {error_type}\n"
                    f"Error details: {str(e)}\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Check that OpenVINO is properly installed:\n"
                    f"   pip install openvino openvino-dev\n\n"
                    f"2. Verify the model was exported correctly:\n"
                    f"   python scripts/export_to_openvino.py {model_name}\n\n"
                    f"3. Enable fallback to PyTorch in configuration:\n"
                    f"   backend:\n"
                    f"     openvino:\n"
                    f"       enable_fallback: true"
                ) from e
    
    def forward(
        self,
        model: Any,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run forward inference with automatic fallback.
        
        Delegates to OpenVINO manager or fallback backend based on model type.
        
        Requirements:
            - 4.2: Retry with PyTorch backend on OpenVINO inference failure
            - 5.2: Return results in same format regardless of backend
        
        Args:
            model: The loaded model object.
            inputs: Dictionary of input tensors/arrays.
            
        Returns:
            Dictionary containing model outputs (e.g., {'logits': tensor}).
            
        Raises:
            RuntimeError: If inference fails.
            
        Example:
            >>> inputs = {'input_ids': tensor, 'attention_mask': tensor}
            >>> outputs = backend.forward(model, inputs)
            >>> logits = outputs['logits']
        """
        # Check if this is a fallback model
        if self._is_fallback_model(model):
            if self._fallback_backend is None:
                raise RuntimeError("Model is from fallback backend but fallback is not initialized")
            return self._fallback_backend.forward(model, inputs)
        
        # Try OpenVINO inference
        try:
            # OpenVINO manager's infer method expects model name, not model object
            # We need to find the model name from our registry
            model_name = None
            for name, cached_model in self._models.items():
                if cached_model is model:
                    model_name = name
                    break
            
            if model_name is None:
                raise RuntimeError("Model not found in registry")
            
            # Use OpenVINO manager for inference
            # Note: OpenVINO manager's infer method handles tokenization internally
            # For forward pass, we need to work with the model directly
            logger.debug(f"Running OpenVINO forward inference for {model_name}")
            
            # For now, delegate to the model's forward method
            # OpenVINO models from optimum.intel have a forward method
            if hasattr(model, 'forward'):
                outputs = model.forward(**inputs)
                return {'logits': outputs.logits if hasattr(outputs, 'logits') else outputs}
            else:
                raise RuntimeError("Model does not support forward method")
                
        except Exception as e:
            error_type = type(e).__name__
            logger.warning(f"OpenVINO forward inference failed: {error_type}: {e}")
            
            if self._fallback_backend:
                logger.info("Retrying forward inference with PyTorch fallback")
                return self._fallback_backend.forward(model, inputs)
            else:
                raise InferenceError(
                    f"OpenVINO forward inference failed and fallback is disabled.\n\n"
                    f"Error type: {error_type}\n"
                    f"Error details: {str(e)}\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Verify input format matches model expectations\n"
                    f"2. Check that the model was loaded successfully\n"
                    f"3. Enable fallback to PyTorch for robustness:\n"
                    f"   backend:\n"
                    f"     openvino:\n"
                    f"       enable_fallback: true\n\n"
                    f"4. Try reloading the model:\n"
                    f"   backend.unload_model(model_name)\n"
                    f"   backend.load_model(model_name, ...)"
                ) from e
    
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Generate text with automatic fallback.
        
        Delegates to OpenVINO manager or fallback backend based on model type.
        
        Requirements:
            - 4.2: Retry with PyTorch backend on OpenVINO generation failure
            - 5.2: Return results in same format regardless of backend
        
        Args:
            model: The loaded model object.
            tokenizer: Tokenizer for encoding/decoding.
            prompt: Input text prompt.
            max_length: Maximum length of generated text.
            **kwargs: Additional generation parameters (temperature, top_p, etc.).
            
        Returns:
            Generated text string.
            
        Raises:
            RuntimeError: If generation fails.
            
        Example:
            >>> text = backend.generate(model, tokenizer, "Hello", max_length=50)
            >>> print(text)
            'Hello, how are you today?'
        """
        # Check if this is a fallback model
        if self._is_fallback_model(model):
            if self._fallback_backend is None:
                raise RuntimeError("Model is from fallback backend but fallback is not initialized")
            return self._fallback_backend.generate(model, tokenizer, prompt, max_length, **kwargs)
        
        # Try OpenVINO generation
        try:
            logger.debug(f"Running OpenVINO text generation")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt')
            
            # Prepare generation kwargs
            gen_kwargs = {
                'max_length': max_length,
                'temperature': kwargs.get('temperature', 0.7),
                'do_sample': kwargs.get('do_sample', True),
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            
            # Add optional parameters
            if 'top_p' in kwargs:
                gen_kwargs['top_p'] = kwargs['top_p']
            if 'top_k' in kwargs:
                gen_kwargs['top_k'] = kwargs['top_k']
            if 'num_return_sequences' in kwargs:
                gen_kwargs['num_return_sequences'] = kwargs['num_return_sequences']
            
            # Generate using OpenVINO model
            outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            error_type = type(e).__name__
            logger.warning(f"OpenVINO text generation failed: {error_type}: {e}")
            
            if self._fallback_backend:
                logger.info("Retrying text generation with PyTorch fallback")
                return self._fallback_backend.generate(model, tokenizer, prompt, max_length, **kwargs)
            else:
                raise InferenceError(
                    f"OpenVINO text generation failed and fallback is disabled.\n\n"
                    f"Error type: {error_type}\n"
                    f"Error details: {str(e)}\n\n"
                    f"Troubleshooting steps:\n"
                    f"1. Verify the prompt is valid and not empty\n"
                    f"2. Check generation parameters (max_length, temperature, etc.)\n"
                    f"3. Ensure the tokenizer matches the model\n"
                    f"4. Enable fallback to PyTorch for robustness:\n"
                    f"   backend:\n"
                    f"     openvino:\n"
                    f"       enable_fallback: true\n\n"
                    f"5. Try with simpler generation parameters:\n"
                    f"   backend.generate(model, tokenizer, prompt, max_length=50)"
                ) from e
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Removes the model from both the internal registry and the OpenVINO manager cache.
        
        Args:
            model_name: Unique identifier of the model to unload.
            
        Raises:
            KeyError: If model_name is not found.
            
        Example:
            >>> backend.unload_model('gpt2')
        """
        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' not found in loaded models. "
                f"Available models: {list(self._models.keys())}"
            )
        
        # Remove from internal registry
        del self._models[model_name]
        
        # Unload from OpenVINO manager if available
        if self._openvino_manager is not None:
            try:
                self._openvino_manager.unload_model(model_name)
            except Exception as e:
                logger.warning(f"Failed to unload model from OpenVINO manager: {e}")
        
        logger.info(f"Unloaded model: {model_name}")
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get metadata about a loaded model.
        
        Returns backend type, device, and other metadata. Automatically detects
        whether the model is from OpenVINO or fallback backend.
        
        Requirements:
            - 5.2: Provide consistent metadata format across backends
        
        Args:
            model: The loaded model object.
            
        Returns:
            Dictionary containing:
                - backend (str): 'openvino' or 'pytorch' (if fallback)
                - device (str): Device where model is loaded
                - model_type (str): Type of model
                - is_fallback (bool): Whether this is a fallback model
                
        Example:
            >>> info = backend.get_model_info(model)
            >>> print(info['backend'])  # 'openvino' or 'pytorch'
            >>> print(info['is_fallback'])  # True or False
        """
        # Check if this is a fallback model
        is_fallback = self._is_fallback_model(model)
        
        if is_fallback and self._fallback_backend:
            # Get info from fallback backend
            info = self._fallback_backend.get_model_info(model)
            info['is_fallback'] = True
            return info
        
        # Return OpenVINO model info
        return {
            'backend': 'openvino',
            'device': self.device,
            'model_type': 'IR',
            'is_fallback': False,
        }
    
    def is_available(self) -> bool:
        """
        Check if OpenVINO backend is available.
        
        Returns:
            True if OpenVINO manager is initialized, False otherwise.
        """
        return self._openvino_manager is not None
