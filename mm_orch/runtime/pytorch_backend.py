"""
PyTorch inference backend implementation.

This module provides a PyTorch-based implementation of the InferenceBackend interface,
wrapping existing PyTorch functionality into the unified backend API.
"""

import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoModel

from mm_orch.runtime.inference_backend import InferenceBackend

logger = logging.getLogger(__name__)


class PyTorchBackend(InferenceBackend):
    """
    PyTorch inference backend.
    
    This backend wraps PyTorch models (primarily HuggingFace Transformers) into
    the unified InferenceBackend interface, providing backward compatibility with
    existing PyTorch-based workflows.
    
    Requirements:
        - 3.1: Maintain backward compatibility with existing PyTorch implementation
        - 3.2: Preserve existing ModelManager API methods and signatures
        - 3.3: Produce identical outputs to current implementation
    
    Example:
        >>> backend = PyTorchBackend(device='cpu', config={})
        >>> model = backend.load_model('gpt2', 'models/gpt2', 'transformers')
        >>> output = backend.generate(model, tokenizer, 'Hello', max_length=50)
    """
    
    def __init__(self, device: str, config: Dict[str, Any]):
        """
        Initialize PyTorch backend.
        
        Args:
            device: Target device ('cpu', 'cuda', or 'cuda:N' for specific GPU).
            config: Backend-specific configuration (e.g., dtype, quantization settings).
        """
        super().__init__(device, config)
        
        # Convert device string to torch.device
        self.torch_device = torch.device(device if device in ['cpu', 'cuda'] else 'cpu')
        
        logger.info(f"Initialized PyTorch backend with device: {self.torch_device}")
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str
    ) -> Any:
        """
        Load a PyTorch model from the specified path.
        
        Supports HuggingFace Transformers models with automatic fallback from
        AutoModelForCausalLM to AutoModel if needed.
        
        Args:
            model_name: Unique identifier for the model.
            model_path: Path to the model files (HuggingFace model name or local path).
            model_type: Type of model (currently supports 'transformers').
            
        Returns:
            Loaded PyTorch model in eval mode.
            
        Raises:
            ValueError: If model_type is not supported.
            RuntimeError: If model loading fails.
        """
        if model_type != 'transformers':
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"PyTorchBackend currently supports 'transformers' only."
            )
        
        try:
            logger.info(f"Loading PyTorch model: {model_name} from {model_path}")
            
            # Prepare loading kwargs
            load_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float32,  # Default to FP32 for compatibility
            }
            
            # Try loading as CausalLM first (most common for generation)
            model = None
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                logger.debug(f"Loaded {model_name} as AutoModelForCausalLM")
            except Exception as e:
                logger.debug(f"Failed to load as CausalLM: {e}, trying AutoModel")
                # Fallback to generic AutoModel
                model = AutoModel.from_pretrained(model_path, **load_kwargs)
                logger.debug(f"Loaded {model_name} as AutoModel")
            
            # Move model to target device
            model.to(self.torch_device)
            
            # Set to evaluation mode
            model.eval()
            
            # Store in internal registry
            self._models[model_name] = model
            
            logger.info(
                f"Successfully loaded model {model_name} on device {self.torch_device}"
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(
                f"Failed to load PyTorch model {model_name} from {model_path}: {e}"
            ) from e
    
    def forward(
        self,
        model: Any,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run forward inference on the PyTorch model.
        
        Executes model forward pass within torch.no_grad() context for efficiency.
        
        Args:
            model: The loaded PyTorch model.
            inputs: Dictionary of input tensors (e.g., {'input_ids': tensor, 'attention_mask': tensor}).
            
        Returns:
            Dictionary containing model outputs (e.g., {'logits': tensor}).
            
        Raises:
            RuntimeError: If inference fails.
        """
        try:
            # Move inputs to correct device
            device_inputs = {
                k: v.to(self.torch_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            
            # Run inference without gradient computation
            with torch.no_grad():
                outputs = model(**device_inputs)
            
            # Extract logits from model output
            result = {'logits': outputs.logits if hasattr(outputs, 'logits') else outputs}
            
            return result
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError(f"PyTorch forward inference failed: {e}") from e
    
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Generate text using the PyTorch model.
        
        Wraps model.generate() with proper tokenization and decoding.
        
        Args:
            model: The loaded PyTorch model.
            tokenizer: HuggingFace tokenizer for encoding/decoding.
            prompt: Input text prompt.
            max_length: Maximum length of generated text (total, including prompt).
            **kwargs: Additional generation parameters:
                - temperature (float): Sampling temperature (default: 0.7)
                - top_p (float): Nucleus sampling parameter
                - top_k (int): Top-k sampling parameter
                - do_sample (bool): Whether to use sampling (default: True)
                - num_return_sequences (int): Number of sequences to generate
                
        Returns:
            Generated text string (with prompt removed if applicable).
            
        Raises:
            RuntimeError: If generation fails.
        """
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors='pt')
            
            # Move inputs to correct device
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}
            
            # Prepare generation kwargs
            gen_kwargs = {
                'max_length': max_length,
                'temperature': kwargs.get('temperature', 0.7),
                'do_sample': kwargs.get('do_sample', True),
                'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            
            # Add optional parameters if provided
            if 'top_p' in kwargs:
                gen_kwargs['top_p'] = kwargs['top_p']
            if 'top_k' in kwargs:
                gen_kwargs['top_k'] = kwargs['top_k']
            if 'num_return_sequences' in kwargs:
                gen_kwargs['num_return_sequences'] = kwargs['num_return_sequences']
            
            # Generate text without gradient computation
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"PyTorch text generation failed: {e}") from e
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a PyTorch model from memory.
        
        Removes the model from the internal registry and clears CUDA cache if applicable.
        
        Args:
            model_name: Unique identifier of the model to unload.
            
        Raises:
            KeyError: If model_name is not found in loaded models.
        """
        if model_name not in self._models:
            raise KeyError(
                f"Model '{model_name}' not found in loaded models. "
                f"Available models: {list(self._models.keys())}"
            )
        
        # Remove model from registry
        del self._models[model_name]
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available() and self.torch_device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache after unloading {model_name}")
        
        logger.info(f"Unloaded model: {model_name}")
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get metadata and information about a loaded PyTorch model.
        
        Args:
            model: The loaded PyTorch model.
            
        Returns:
            Dictionary containing:
                - backend (str): 'pytorch'
                - device (str): Device where model is loaded
                - parameters (int): Total number of model parameters
                - dtype (str): Model data type
        """
        try:
            # Count total parameters
            params_list = list(model.parameters())
            total_params = sum(p.numel() for p in params_list)
            
            # Get model dtype (from first parameter)
            dtype = str(params_list[0].dtype) if len(params_list) > 0 else 'unknown'
            
            return {
                'backend': 'pytorch',
                'device': str(self.torch_device),
                'parameters': total_params,
                'dtype': dtype,
            }
        except Exception as e:
            logger.warning(f"Failed to get complete model info: {e}")
            return {
                'backend': 'pytorch',
                'device': str(self.torch_device),
                'parameters': 0,
                'dtype': 'unknown',
            }
    
    def is_available(self) -> bool:
        """
        Check if PyTorch backend is available on the current system.
        
        Returns:
            True if PyTorch is installed and functional, False otherwise.
        """
        try:
            import torch
            # Basic availability check - can we create a tensor?
            _ = torch.tensor([1.0])
            return True
        except Exception:
            return False
