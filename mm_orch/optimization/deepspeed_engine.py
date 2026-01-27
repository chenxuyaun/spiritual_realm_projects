"""
DeepSpeed inference engine wrapper.

This module provides integration with DeepSpeed for large model inference
with tensor and pipeline parallelism. Includes graceful error handling
and fallback strategies.
"""

import time
from typing import Any, Dict, List, Optional

import torch

from mm_orch.logger import get_logger
from mm_orch.optimization.config import DeepSpeedConfig

logger = get_logger(__name__)


class DeepSpeedEngine:
    """
    Wrapper for DeepSpeed inference engine.
    
    Provides large model inference with tensor parallelism and pipeline
    parallelism for multi-GPU distribution. Supports model sharding and
    DeepSpeed kernel optimizations.
    
    Attributes:
        config: DeepSpeed configuration
        _model: DeepSpeed-wrapped model (None if not initialized)
        _loaded_model: Name of currently loaded model (None if no model loaded)
        _tokenizer: Model tokenizer (None if not loaded)
    
    Example:
        >>> config = DeepSpeedConfig(tensor_parallel=2)
        >>> engine = DeepSpeedEngine(config)
        >>> if engine.is_available():
        ...     engine.load_model("qwen-chat")
        ...     outputs = engine.infer({"input_ids": tensor})
    """
    
    def __init__(self, config: DeepSpeedConfig):
        """
        Initialize DeepSpeed engine with configuration.
        
        Args:
            config: DeepSpeed configuration specifying parallelism and dtype
        """
        self.config = config
        self._model = None
        self._loaded_model: Optional[str] = None
        self._tokenizer = None
        
        logger.info(
            f"DeepSpeedEngine initialized with config: "
            f"tensor_parallel={config.tensor_parallel}, "
            f"pipeline_parallel={config.pipeline_parallel}, "
            f"dtype={config.dtype}, "
            f"kernel_inject={config.replace_with_kernel_inject}"
        )
    
    def is_available(self) -> bool:
        """
        Check if DeepSpeed is available and functional.
        
        Attempts to import DeepSpeed and verify basic functionality.
        
        Returns:
            True if DeepSpeed is available, False otherwise
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> if engine.is_available():
            ...     print("DeepSpeed is ready")
        """
        if not self.config.enabled:
            logger.debug("DeepSpeed disabled in configuration")
            return False
        
        try:
            # Try to import DeepSpeed
            import deepspeed  # noqa: F401
            
            logger.debug("DeepSpeed is available")
            return True
            
        except ImportError as e:
            logger.warning(f"DeepSpeed not available: {e}")
            return False
        except Exception as e:
            logger.error(f"DeepSpeed availability check failed: {e}", exc_info=True)
            return False
    
    def load_model(
        self,
        model_name: str,
        tensor_parallel: Optional[int] = None,
        pipeline_parallel: Optional[int] = None
    ) -> bool:
        """
        Load model with DeepSpeed optimizations and parallelism.
        
        Args:
            model_name: HuggingFace model name or path to load
            tensor_parallel: Override config tensor parallelism (optional)
            pipeline_parallel: Override config pipeline parallelism (optional)
        
        Returns:
            True if model loaded successfully, False otherwise
        
        Raises:
            ImportError: If DeepSpeed is not installed
            RuntimeError: If model loading fails critically
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> success = engine.load_model("qwen-chat", tensor_parallel=2)
            >>> if success:
            ...     print("Model loaded successfully")
        """
        if not self.is_available():
            logger.error("Cannot load model: DeepSpeed is not available")
            return False
        
        try:
            import deepspeed
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Use provided parameters or fall back to config
            tp_size = tensor_parallel or self.config.tensor_parallel
            pp_size = pipeline_parallel or self.config.pipeline_parallel
            
            logger.info(
                f"Loading model {model_name} with DeepSpeed "
                f"(tensor_parallel={tp_size}, pipeline_parallel={pp_size}, "
                f"dtype={self.config.dtype})"
            )
            
            start_time = time.time()
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self._get_torch_dtype(),
            )
            
            # Configure DeepSpeed inference
            ds_config = {
                "tensor_parallel": {
                    "tp_size": tp_size
                },
                "dtype": self.config.dtype,
                "replace_with_kernel_inject": self.config.replace_with_kernel_inject,
            }
            
            # Initialize DeepSpeed inference engine
            self._model = deepspeed.init_inference(
                model,
                mp_size=tp_size,
                dtype=self._get_torch_dtype(),
                replace_with_kernel_inject=self.config.replace_with_kernel_inject,
            )
            
            self._loaded_model = model_name
            
            load_time = time.time() - start_time
            logger.info(
                f"Model {model_name} loaded successfully with DeepSpeed "
                f"in {load_time:.2f}s"
            )
            
            return True
            
        except ImportError as e:
            logger.error(f"DeepSpeed import failed: {e}")
            raise ImportError(f"DeepSpeed is not installed: {e}") from e
            
        except RuntimeError as e:
            logger.error(
                f"Failed to load model {model_name} with DeepSpeed: {e}",
                exc_info=True
            )
            # Clean up partial initialization
            self._model = None
            self._loaded_model = None
            self._tokenizer = None
            raise RuntimeError(f"DeepSpeed model loading failed: {e}") from e
            
        except Exception as e:
            logger.error(
                f"Unexpected error loading model {model_name}: {e}",
                exc_info=True
            )
            # Clean up partial initialization
            self._model = None
            self._loaded_model = None
            self._tokenizer = None
            return False
    
    def infer(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute inference with DeepSpeed.
        
        Handles tensor parallelism and model sharding automatically.
        Converts inputs/outputs between formats as needed.
        
        Args:
            inputs: Dictionary with input tensors (e.g., input_ids, attention_mask)
        
        Returns:
            Dictionary with output tensors (e.g., logits, hidden_states)
        
        Raises:
            RuntimeError: If no model is loaded or inference fails
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> engine.load_model("qwen-chat")
            >>> inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
            >>> outputs = engine.infer(inputs)
            >>> print(outputs["logits"].shape)
        """
        if self._model is None or self._loaded_model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() before infer()."
            )
        
        try:
            logger.debug(
                f"Running inference with DeepSpeed "
                f"(model: {self._loaded_model}, "
                f"batch_size: {inputs.get('input_ids', torch.tensor([])).shape[0] if 'input_ids' in inputs else 'unknown'})"
            )
            
            start_time = time.time()
            
            # Move inputs to appropriate device
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Run inference with DeepSpeed
            with torch.no_grad():
                outputs = self._model(**inputs)
            
            inference_time = time.time() - start_time
            
            # Convert outputs to dictionary format
            if hasattr(outputs, 'logits'):
                output_dict = {
                    "logits": outputs.logits,
                }
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    output_dict["hidden_states"] = outputs.hidden_states
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    output_dict["attentions"] = outputs.attentions
            else:
                # Fallback: treat outputs as tensor
                output_dict = {"output": outputs}
            
            logger.info(
                f"DeepSpeed inference completed in {inference_time:.2f}s"
            )
            
            return output_dict
            
        except Exception as e:
            logger.error(
                f"DeepSpeed inference failed: {e}",
                exc_info=True
            )
            raise RuntimeError(f"DeepSpeed inference failed: {e}") from e
    
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate text using DeepSpeed inference.
        
        Args:
            prompts: List of input prompts to generate from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
        
        Returns:
            List of generation outputs, one per prompt
        
        Raises:
            RuntimeError: If no model is loaded or generation fails
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> engine.load_model("qwen-chat")
            >>> outputs = engine.generate(["Hello"], max_new_tokens=100)
            >>> print(outputs[0]["text"])
        """
        if self._model is None or self._loaded_model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() before generate()."
            )
        
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        try:
            logger.debug(
                f"Generating with DeepSpeed for {len(prompts)} prompts "
                f"(model: {self._loaded_model})"
            )
            
            start_time = time.time()
            
            # Tokenize inputs
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with DeepSpeed model
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    **kwargs
                )
            
            generation_time = time.time() - start_time
            
            # Decode outputs
            outputs = []
            for i, (prompt, gen_ids) in enumerate(zip(prompts, generated_ids)):
                # Decode generated text
                generated_text = self._tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True
                )
                
                # Remove prompt from generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                output = {
                    "text": generated_text,
                    "prompt": prompt,
                    "tokens_generated": len(gen_ids) - len(inputs["input_ids"][i]),
                }
                outputs.append(output)
            
            logger.info(
                f"Generated {len(outputs)} outputs with DeepSpeed "
                f"in {generation_time:.2f}s"
                + (f" ({len(outputs)/generation_time:.2f} outputs/s)" if generation_time > 0 else "")
            )
            
            return outputs
            
        except Exception as e:
            logger.error(
                f"DeepSpeed generation failed: {e}",
                exc_info=True
            )
            raise RuntimeError(f"DeepSpeed generation failed: {e}") from e
    
    def unload_model(self):
        """
        Unload the currently loaded model and free resources.
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> engine.load_model("qwen-chat")
            >>> # ... use model ...
            >>> engine.unload_model()
        """
        if self._model is not None:
            logger.info(f"Unloading model {self._loaded_model} from DeepSpeed")
            
            # Clear references
            self._model = None
            self._loaded_model = None
            self._tokenizer = None
            
            # Force garbage collection to free GPU memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("DeepSpeed model unloaded and GPU memory cleared")
    
    def get_loaded_model(self) -> Optional[str]:
        """
        Get the name of the currently loaded model.
        
        Returns:
            Model name if a model is loaded, None otherwise
        
        Example:
            >>> engine = DeepSpeedEngine(DeepSpeedConfig())
            >>> engine.load_model("qwen-chat")
            >>> print(engine.get_loaded_model())
            'qwen-chat'
        """
        return self._loaded_model
    
    def _get_torch_dtype(self) -> torch.dtype:
        """
        Convert config dtype string to torch dtype.
        
        Returns:
            torch.dtype corresponding to config dtype
        """
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float16)
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self._model is not None:
            logger.debug("DeepSpeedEngine destructor: unloading model")
            self.unload_model()
