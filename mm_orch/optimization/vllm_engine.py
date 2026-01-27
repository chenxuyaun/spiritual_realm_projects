"""
vLLM inference engine wrapper.

This module provides integration with vLLM for high-throughput LLM inference
with continuous batching and PagedAttention. Includes graceful error handling
and fallback strategies.
"""

import time
from typing import Any, Dict, List, Optional

from mm_orch.logger import get_logger
from mm_orch.optimization.config import VLLMConfig
from mm_orch.optimization.gpu_utils import get_gpu_manager

logger = get_logger(__name__)


class VLLMEngine:
    """
    Wrapper for vLLM inference engine.
    
    Provides high-throughput LLM inference with continuous batching and
    PagedAttention for memory efficiency. Supports tensor parallelism for
    multi-GPU inference.
    
    Attributes:
        config: vLLM configuration
        _llm: vLLM LLM instance (None if not initialized)
        _loaded_model: Name of currently loaded model (None if no model loaded)
    
    Example:
        >>> config = VLLMConfig(tensor_parallel_size=2)
        >>> engine = VLLMEngine(config)
        >>> if engine.is_available():
        ...     engine.load_model("qwen-chat")
        ...     outputs = engine.generate(["Hello"], sampling_params)
    """
    
    def __init__(self, config: VLLMConfig):
        """
        Initialize vLLM engine with configuration.
        
        Args:
            config: vLLM configuration specifying parallelism and memory settings
        """
        self.config = config
        self._llm = None
        self._loaded_model: Optional[str] = None
        self._gpu_manager = get_gpu_manager()
        self._allocated_gpus: Optional[List[int]] = None
        
        logger.info(
            f"VLLMEngine initialized with config: "
            f"tensor_parallel={config.tensor_parallel_size}, "
            f"dtype={config.dtype}, "
            f"gpu_memory={config.gpu_memory_utilization}"
        )
    
    def is_available(self) -> bool:
        """
        Check if vLLM is available and functional.
        
        Attempts to import vLLM and verify basic functionality.
        
        Returns:
            True if vLLM is available, False otherwise
        
        Example:
            >>> engine = VLLMEngine(VLLMConfig())
            >>> if engine.is_available():
            ...     print("vLLM is ready")
        """
        if not self.config.enabled:
            logger.debug("vLLM disabled in configuration")
            return False
        
        try:
            # Try to import vLLM
            import vllm  # noqa: F401
            from vllm import LLM  # noqa: F401
            
            logger.debug("vLLM is available")
            return True
            
        except ImportError as e:
            logger.warning(f"vLLM not available: {e}")
            return False
        except Exception as e:
            logger.error(f"vLLM availability check failed: {e}", exc_info=True)
            return False
    
    def load_model(
        self,
        model_name: str,
        tensor_parallel_size: Optional[int] = None,
        dtype: Optional[str] = None
    ) -> bool:
        """
        Load model into vLLM engine with tensor parallelism configuration.
        
        Automatically detects and allocates GPUs for tensor parallelism.
        
        Args:
            model_name: HuggingFace model name or path to load
            tensor_parallel_size: Override config tensor parallelism (optional)
            dtype: Override config dtype (optional)
        
        Returns:
            True if model loaded successfully, False otherwise
        
        Raises:
            ImportError: If vLLM is not installed
            RuntimeError: If model loading fails critically
        
        Example:
            >>> engine = VLLMEngine(VLLMConfig())
            >>> success = engine.load_model("qwen-chat", tensor_parallel_size=2)
            >>> if success:
            ...     print("Model loaded successfully")
        """
        if not self.is_available():
            logger.error("Cannot load model: vLLM is not available")
            return False
        
        try:
            from vllm import LLM
            
            # Use provided parameters or fall back to config
            tp_size = tensor_parallel_size or self.config.tensor_parallel_size
            model_dtype = dtype or self.config.dtype
            
            # Allocate GPUs if tensor parallelism is requested
            if tp_size > 1:
                try:
                    gpu_ids, strategy = self._gpu_manager.allocate_gpus(
                        tensor_parallel=tp_size
                    )
                    self._allocated_gpus = gpu_ids
                    logger.info(
                        f"Allocated {len(gpu_ids)} GPUs for tensor parallelism: {gpu_ids}"
                    )
                except RuntimeError as e:
                    logger.error(f"GPU allocation failed: {e}")
                    # Fall back to single GPU if allocation fails
                    logger.warning("Falling back to single GPU mode")
                    tp_size = 1
                    self._allocated_gpus = None
            
            logger.info(
                f"Loading model {model_name} with vLLM "
                f"(tensor_parallel={tp_size}, dtype={model_dtype})"
            )
            
            start_time = time.time()
            
            # Initialize vLLM with configuration
            self._llm = LLM(
                model=model_name,
                tensor_parallel_size=tp_size,
                dtype=model_dtype,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                swap_space=self.config.swap_space,
                trust_remote_code=True,  # Required for some models like Qwen
            )
            
            self._loaded_model = model_name
            
            load_time = time.time() - start_time
            logger.info(
                f"Model {model_name} loaded successfully with vLLM "
                f"in {load_time:.2f}s"
            )
            
            return True
            
        except ImportError as e:
            logger.error(f"vLLM import failed: {e}")
            raise ImportError(f"vLLM is not installed: {e}") from e
            
        except RuntimeError as e:
            logger.error(
                f"Failed to load model {model_name} with vLLM: {e}",
                exc_info=True
            )
            # Clean up partial initialization
            self._llm = None
            self._loaded_model = None
            self._allocated_gpus = None
            raise RuntimeError(f"vLLM model loading failed: {e}") from e
            
        except Exception as e:
            logger.error(
                f"Unexpected error loading model {model_name}: {e}",
                exc_info=True
            )
            # Clean up partial initialization
            self._llm = None
            self._loaded_model = None
            self._allocated_gpus = None
            return False
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text using vLLM with continuous batching.
        
        Args:
            prompts: List of input prompts to generate from
            sampling_params: vLLM SamplingParams object (optional)
        
        Returns:
            List of generation outputs, one per prompt
        
        Raises:
            RuntimeError: If no model is loaded or generation fails
        
        Example:
            >>> from vllm import SamplingParams
            >>> engine = VLLMEngine(VLLMConfig())
            >>> engine.load_model("qwen-chat")
            >>> params = SamplingParams(temperature=0.7, max_tokens=100)
            >>> outputs = engine.generate(["Hello"], params)
            >>> print(outputs[0]["text"])
        """
        if self._llm is None or self._loaded_model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() before generate()."
            )
        
        try:
            from vllm import SamplingParams
            
            # Use default sampling params if not provided
            if sampling_params is None:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=512,
                )
            
            logger.debug(
                f"Generating with vLLM for {len(prompts)} prompts "
                f"(model: {self._loaded_model})"
            )
            
            start_time = time.time()
            
            # Generate with vLLM (uses continuous batching internally)
            vllm_outputs = self._llm.generate(prompts, sampling_params)
            
            generation_time = time.time() - start_time
            
            # Convert vLLM outputs to standard format
            outputs = []
            for vllm_output in vllm_outputs:
                # Extract generated text from first completion
                generated_text = vllm_output.outputs[0].text if vllm_output.outputs else ""
                
                output = {
                    "text": generated_text,
                    "prompt": vllm_output.prompt,
                    "finish_reason": vllm_output.outputs[0].finish_reason if vllm_output.outputs else None,
                    "tokens_generated": len(vllm_output.outputs[0].token_ids) if vllm_output.outputs else 0,
                }
                outputs.append(output)
            
            logger.info(
                f"Generated {len(outputs)} outputs with vLLM "
                f"in {generation_time:.2f}s "
                f"({len(outputs)/generation_time:.2f} outputs/s)"
            )
            
            return outputs
            
        except Exception as e:
            logger.error(
                f"vLLM generation failed: {e}",
                exc_info=True
            )
            raise RuntimeError(f"vLLM generation failed: {e}") from e
    
    def unload_model(self):
        """
        Unload the currently loaded model and free resources.
        
        Example:
            >>> engine = VLLMEngine(VLLMConfig())
            >>> engine.load_model("qwen-chat")
            >>> # ... use model ...
            >>> engine.unload_model()
        """
        if self._llm is not None:
            logger.info(f"Unloading model {self._loaded_model} from vLLM")
            
            # vLLM doesn't have explicit unload, but we can clear references
            # and rely on Python garbage collection
            self._llm = None
            self._loaded_model = None
            self._allocated_gpus = None
            
            # Force garbage collection to free GPU memory
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("vLLM model unloaded and GPU memory cleared")
    
    def get_loaded_model(self) -> Optional[str]:
        """
        Get the name of the currently loaded model.
        
        Returns:
            Model name if a model is loaded, None otherwise
        
        Example:
            >>> engine = VLLMEngine(VLLMConfig())
            >>> engine.load_model("qwen-chat")
            >>> print(engine.get_loaded_model())
            'qwen-chat'
        """
        return self._loaded_model
    
    def get_allocated_gpus(self) -> Optional[List[int]]:
        """
        Get the list of GPUs allocated for tensor parallelism.
        
        Returns:
            List of GPU IDs if allocated, None otherwise
        
        Example:
            >>> engine = VLLMEngine(VLLMConfig(tensor_parallel_size=2))
            >>> engine.load_model("qwen-chat")
            >>> print(engine.get_allocated_gpus())
            [0, 1]
        """
        return self._allocated_gpus
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self._llm is not None:
            logger.debug("VLLMEngine destructor: unloading model")
            self.unload_model()
