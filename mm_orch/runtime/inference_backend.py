"""
Abstract base class for inference backends.

This module defines the interface that all inference backends (PyTorch, OpenVINO, etc.)
must implement to provide a unified API for model loading, inference, and management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    This class defines the interface that all inference backends must implement
    to ensure a consistent API regardless of the underlying inference engine.
    
    Attributes:
        device (str): The target device for inference (e.g., 'cpu', 'cuda', 'GPU').
        config (Dict[str, Any]): Backend-specific configuration parameters.
    """
    
    def __init__(self, device: str, config: Dict[str, Any]):
        """
        Initialize the inference backend.
        
        Args:
            device: Target device for inference (e.g., 'cpu', 'cuda', 'GPU', 'AUTO').
            config: Backend-specific configuration dictionary.
        """
        self.device = device
        self.config = config
        self._models: Dict[str, Any] = {}
    
    @abstractmethod
    def load_model(
        self,
        model_name: str,
        model_path: str,
        model_type: str
    ) -> Any:
        """
        Load a model from the specified path.
        
        Args:
            model_name: Unique identifier for the model.
            model_path: Path to the model files.
            model_type: Type of model (e.g., 'transformers', 'custom').
            
        Returns:
            Loaded model object.
            
        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        model: Any,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run forward inference on the model.
        
        Args:
            model: The loaded model object.
            inputs: Dictionary of input tensors/arrays.
            
        Returns:
            Dictionary containing model outputs (e.g., {'logits': tensor}).
            
        Raises:
            RuntimeError: If inference fails.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int,
        **kwargs
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            model: The loaded model object.
            tokenizer: Tokenizer for encoding/decoding text.
            prompt: Input text prompt.
            max_length: Maximum length of generated text.
            **kwargs: Additional generation parameters (temperature, top_p, etc.).
            
        Returns:
            Generated text string.
            
        Raises:
            RuntimeError: If generation fails.
        """
        pass
    
    @abstractmethod
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Unique identifier of the model to unload.
            
        Raises:
            KeyError: If model_name is not found in loaded models.
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get metadata and information about a loaded model.
        
        Args:
            model: The loaded model object.
            
        Returns:
            Dictionary containing model metadata (backend, device, parameters, etc.).
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if this backend is available on the current system.
        
        This method can be overridden by subclasses to perform backend-specific
        availability checks (e.g., checking if required libraries are installed).
        
        Returns:
            True if the backend is available, False otherwise.
        """
        return True
