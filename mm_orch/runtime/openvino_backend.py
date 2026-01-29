"""
OpenVINO inference backend implementation.

This module provides an OpenVINO-based implementation of the InferenceBackend interface,
with automatic fallback to PyTorch when OpenVINO operations fail.
"""

import logging
import os
from typing import Any, Dict, Optional

from mm_orch.runtime.inference_backend import InferenceBackend

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
        """
        super().__init__(device, config)
        
        self._openvino_manager = None
        self._fallback_backend = None
        self._fallback_enabled = config.get('enable_fallback', True)
        
        # Initialize OpenVINO with fallback handling
        self._initialize_openvino()
        
        logger.info(
            f"Initialized OpenVINO backend with device: {device}, "
            f"fallback_enabled: {self._fallback_enabled}"
        )
    
    def _initialize_openvino(self) -> None:
        """
        Initialize OpenVINO manager with automatic fallback to PyTorch.
        
        Attempts to initialize the OpenVINO_Manager. If initialization fails
        and fallback is enabled, creates a PyTorchBackend as fallback. If fallback
        is disabled, raises the original error.
        
        Raises:
            RuntimeError: If OpenVINO initialization fails and fallback is disabled.
        """
        try:
            # Attempt to import and initialize OpenVINO manager
            from mm_orch.runtime.openvino_manager import OpenVINOModelManager
            
            self._openvino_manager = OpenVINOModelManager(
                default_device=self.device,
                enable_openvino=True,
                fallback_to_pytorch=False,  # We handle fallback at backend level
                openvino_cache_dir=self.config.get('cache_dir', 'models/openvino')
            )
            
            logger.info("OpenVINO manager initialized successfully")
            
        except ImportError as e:
            error_msg = (
                f"OpenVINO initialization failed: {e}. "
                f"OpenVINO is not installed. Install with: "
                f"pip install openvino openvino-dev optimum[openvino]"
            )
            logger.warning(error_msg)
            
            if self._fallback_enabled:
                self._initialize_fallback_backend()
            else:
                raise RuntimeError(error_msg) from e
                
        except Exception as e:
            error_msg = f"OpenVINO initialization failed: {e}"
            logger.warning(error_msg)
            
            if self._fallback_enabled:
                self._initialize_fallback_backend()
            else:
                raise RuntimeError(error_msg) from e
    
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
            raise RuntimeError(
                f"OpenVINO backend not initialized and fallback is disabled"
            )
        
        # Try loading with OpenVINO
        try:
            # Convert PyTorch path to OpenVINO path
            openvino_path = self._get_openvino_path(model_path)
            
            # Validate OpenVINO model files exist (Requirement 9.3)
            openvino_xml = os.path.join(openvino_path, 'openvino_model.xml')
            openvino_bin = os.path.join(openvino_path, 'openvino_model.bin')
            
            if not os.path.exists(openvino_xml):
                raise FileNotFoundError(
                    f"OpenVINO model not found: {openvino_xml}. "
                    f"Please export the model first using: "
                    f"python scripts/export_to_openvino.py {model_name}"
                )
            
            if not os.path.exists(openvino_bin):
                raise FileNotFoundError(
                    f"OpenVINO model weights not found: {openvino_bin}. "
                    f"Model export may be incomplete. "
                    f"Re-export using: python scripts/export_to_openvino.py {model_name}"
                )
            
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
            
        except FileNotFoundError as e:
            # File not found errors should trigger fallback
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
                raise
                
        except Exception as e:
            # Other errors should also trigger fallback if enabled
            logger.warning(
                f"OpenVINO model loading failed for {model_name}: {e}"
            )
            
            if self._fallback_backend:
                logger.info(
                    f"Falling back to PyTorch for {model_name} "
                    f"(reason: {type(e).__name__}: {str(e)})"
                )
                return self._fallback_backend.load_model(
                    model_name, model_path, model_type
                )
            else:
                raise RuntimeError(
                    f"OpenVINO model loading failed and fallback is disabled: {e}"
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
            logger.warning(f"OpenVINO forward inference failed: {e}")
            
            if self._fallback_backend:
                logger.info("Retrying forward inference with PyTorch fallback")
                return self._fallback_backend.forward(model, inputs)
            else:
                raise RuntimeError(f"OpenVINO inference failed and fallback is disabled: {e}") from e
    
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
            logger.warning(f"OpenVINO text generation failed: {e}")
            
            if self._fallback_backend:
                logger.info("Retrying text generation with PyTorch fallback")
                return self._fallback_backend.generate(model, tokenizer, prompt, max_length, **kwargs)
            else:
                raise RuntimeError(f"OpenVINO generation failed and fallback is disabled: {e}") from e
    
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
