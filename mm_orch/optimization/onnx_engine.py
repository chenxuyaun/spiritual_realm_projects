"""
ONNX Runtime inference engine wrapper.

This module provides integration with ONNX Runtime for cross-platform
accelerated inference. Includes PyTorch to ONNX model conversion with
validation, and supports multiple execution providers (CUDA, TensorRT, CPU).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil

import torch
import numpy as np

from mm_orch.logger import get_logger
from mm_orch.optimization.config import ONNXConfig

logger = get_logger(__name__)


class ONNXEngine:
    """
    Wrapper for ONNX Runtime inference engine.

    Provides cross-platform accelerated inference with automatic PyTorch to
    ONNX model conversion. Supports multiple execution providers (CUDA,
    TensorRT, CPU) and graph optimizations.

    Attributes:
        config: ONNX Runtime configuration
        _session: ONNX Runtime InferenceSession (None if not initialized)
        _loaded_model: Name of currently loaded model (None if no model loaded)
        _onnx_path: Path to ONNX model file (None if not converted)
        _input_names: List of model input names
        _output_names: List of model output names

    Example:
        >>> config = ONNXConfig(execution_providers=["CUDAExecutionProvider"])
        >>> engine = ONNXEngine(config)
        >>> if engine.is_available():
        ...     onnx_path = engine.convert_model(model, sample_inputs)
        ...     engine.load_model(onnx_path)
        ...     outputs = engine.infer({"input_ids": np.array([[1, 2, 3]])})
    """

    def __init__(self, config: ONNXConfig):
        """
        Initialize ONNX Runtime engine with configuration.

        Args:
            config: ONNX Runtime configuration specifying execution providers
        """
        self.config = config
        self._session = None
        self._loaded_model: Optional[str] = None
        self._onnx_path: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

        logger.info(
            f"ONNXEngine initialized with config: "
            f"providers={config.execution_providers}, "
            f"optimization={config.optimization_level}, "
            f"quantization={config.enable_quantization}"
        )

    def is_available(self) -> bool:
        """
        Check if ONNX Runtime is available and functional.

        Attempts to import ONNX Runtime and verify basic functionality.

        Returns:
            True if ONNX Runtime is available, False otherwise

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> if engine.is_available():
            ...     print("ONNX Runtime is ready")
        """
        if not self.config.enabled:
            logger.debug("ONNX Runtime disabled in configuration")
            return False

        try:
            # Try to import ONNX Runtime
            import onnxruntime  # noqa: F401

            logger.debug("ONNX Runtime is available")
            return True

        except ImportError as e:
            logger.warning(f"ONNX Runtime not available: {e}")
            return False
        except Exception as e:
            logger.error(f"ONNX Runtime availability check failed: {e}", exc_info=True)
            return False

    def convert_model(
        self,
        model: torch.nn.Module,
        sample_inputs: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        validate: bool = True,
    ) -> str:
        """
        Convert PyTorch model to ONNX format with validation.

        Args:
            model: PyTorch model to convert
            sample_inputs: Sample inputs for tracing (dict of tensors)
            output_path: Path to save ONNX model (None for temp file)
            validate: Whether to validate conversion by comparing outputs

        Returns:
            Path to converted ONNX model file

        Raises:
            ImportError: If ONNX or ONNX Runtime is not installed
            RuntimeError: If conversion or validation fails

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> model = MyModel()
            >>> sample_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
            >>> onnx_path = engine.convert_model(model, sample_inputs)
            >>> print(f"Model converted to {onnx_path}")
        """
        if not self.is_available():
            raise ImportError("ONNX Runtime is not available")

        try:
            import onnx
            import torch.onnx

            logger.info(f"Converting PyTorch model to ONNX format")

            # Create output path if not provided
            if output_path is None:
                temp_dir = tempfile.mkdtemp(prefix="onnx_model_")
                output_path = str(Path(temp_dir) / "model.onnx")

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Set model to eval mode
            model.eval()

            # Prepare input names and dynamic axes
            input_names = list(sample_inputs.keys())
            output_names = ["output"]

            # Create dynamic axes for variable batch size and sequence length
            dynamic_axes = {}
            for name in input_names:
                dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
            for name in output_names:
                dynamic_axes[name] = {0: "batch_size"}

            # Convert tuple of tensors for torch.onnx.export
            sample_input_tuple = tuple(sample_inputs.values())

            start_time = time.time()

            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input_tuple,
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=14,  # Use opset 14 for better compatibility
                    do_constant_folding=True,
                    export_params=True,
                )

            conversion_time = time.time() - start_time

            # Check ONNX model validity
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            logger.info(
                f"Model converted to ONNX successfully in {conversion_time:.2f}s "
                f"(saved to {output_path})"
            )

            # Validate conversion if requested
            if validate:
                self._validate_conversion(model, sample_inputs, output_path)

            return output_path

        except ImportError as e:
            logger.error(f"ONNX import failed: {e}")
            raise ImportError(f"ONNX or ONNX Runtime is not installed: {e}") from e

        except RuntimeError as e:
            logger.error(f"Model conversion failed: {e}", exc_info=True)
            raise RuntimeError(f"ONNX conversion failed: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error during conversion: {e}", exc_info=True)
            raise RuntimeError(f"ONNX conversion failed: {e}") from e

    def _validate_conversion(
        self,
        pytorch_model: torch.nn.Module,
        sample_inputs: Dict[str, torch.Tensor],
        onnx_path: str,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        """
        Validate ONNX conversion by comparing outputs with PyTorch model.

        Args:
            pytorch_model: Original PyTorch model
            sample_inputs: Sample inputs for comparison
            onnx_path: Path to ONNX model
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Raises:
            RuntimeError: If outputs don't match within tolerance
        """
        try:
            import onnxruntime as ort

            logger.info("Validating ONNX conversion by comparing outputs")

            # Get PyTorch output
            pytorch_model.eval()
            with torch.no_grad():
                sample_input_tuple = tuple(sample_inputs.values())
                pytorch_output = pytorch_model(*sample_input_tuple)

                # Handle different output types
                if isinstance(pytorch_output, torch.Tensor):
                    pytorch_output_np = pytorch_output.cpu().numpy()
                elif hasattr(pytorch_output, "logits"):
                    pytorch_output_np = pytorch_output.logits.cpu().numpy()
                else:
                    # Try to get first element if it's a tuple/list
                    pytorch_output_np = pytorch_output[0].cpu().numpy()

            # Get ONNX output
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = self._get_optimization_level()

            session = ort.InferenceSession(
                onnx_path, sess_options=sess_options, providers=self._get_available_providers()
            )

            # Prepare ONNX inputs
            onnx_inputs = {name: tensor.cpu().numpy() for name, tensor in sample_inputs.items()}

            onnx_outputs = session.run(None, onnx_inputs)
            onnx_output_np = onnx_outputs[0]

            # Compare outputs
            if not np.allclose(pytorch_output_np, onnx_output_np, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(pytorch_output_np - onnx_output_np))
                logger.error(
                    f"ONNX conversion validation failed: "
                    f"max difference = {max_diff:.6f} "
                    f"(rtol={rtol}, atol={atol})"
                )
                raise RuntimeError(
                    f"ONNX model outputs don't match PyTorch outputs " f"(max_diff={max_diff:.6f})"
                )

            logger.info("ONNX conversion validated successfully")

        except Exception as e:
            logger.error(f"Conversion validation failed: {e}", exc_info=True)
            raise RuntimeError(f"ONNX validation failed: {e}") from e

    def load_model(self, onnx_path: str, model_name: Optional[str] = None) -> bool:
        """
        Load ONNX model into runtime with configured execution providers.

        Args:
            onnx_path: Path to ONNX model file
            model_name: Optional name for the model (for logging)

        Returns:
            True if model loaded successfully, False otherwise

        Raises:
            ImportError: If ONNX Runtime is not installed
            RuntimeError: If model loading fails critically

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> success = engine.load_model("model.onnx", "qwen-chat")
            >>> if success:
            ...     print("Model loaded successfully")
        """
        if not self.is_available():
            logger.error("Cannot load model: ONNX Runtime is not available")
            return False

        if not Path(onnx_path).exists():
            logger.error(f"ONNX model file not found: {onnx_path}")
            return False

        try:
            import onnxruntime as ort

            model_name = model_name or Path(onnx_path).stem

            logger.info(
                f"Loading ONNX model {model_name} from {onnx_path} "
                f"with providers: {self.config.execution_providers}"
            )

            start_time = time.time()

            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = self._get_optimization_level()

            # Enable profiling for debugging (optional)
            # sess_options.enable_profiling = True

            # Create inference session with configured providers
            available_providers = self._get_available_providers()

            self._session = ort.InferenceSession(
                onnx_path, sess_options=sess_options, providers=available_providers
            )

            self._loaded_model = model_name
            self._onnx_path = onnx_path

            # Store input/output names for inference
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

            load_time = time.time() - start_time

            # Log which provider is actually being used
            actual_providers = self._session.get_providers()

            logger.info(
                f"Model {model_name} loaded successfully with ONNX Runtime "
                f"in {load_time:.2f}s "
                f"(using providers: {actual_providers})"
            )

            return True

        except ImportError as e:
            logger.error(f"ONNX Runtime import failed: {e}")
            raise ImportError(f"ONNX Runtime is not installed: {e}") from e

        except RuntimeError as e:
            logger.error(f"Failed to load ONNX model {model_name}: {e}", exc_info=True)
            # Clean up partial initialization
            self._session = None
            self._loaded_model = None
            self._onnx_path = None
            raise RuntimeError(f"ONNX model loading failed: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error loading ONNX model {model_name}: {e}", exc_info=True)
            # Clean up partial initialization
            self._session = None
            self._loaded_model = None
            self._onnx_path = None
            return False

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute inference with ONNX Runtime.

        Handles input/output tensor conversions and execution provider selection.

        Args:
            inputs: Dictionary with input arrays (numpy arrays)

        Returns:
            Dictionary with output arrays (numpy arrays)

        Raises:
            RuntimeError: If no model is loaded or inference fails

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> engine.load_model("model.onnx")
            >>> inputs = {"input_ids": np.array([[1, 2, 3]])}
            >>> outputs = engine.infer(inputs)
            >>> print(outputs["output"].shape)
        """
        if self._session is None or self._loaded_model is None:
            raise RuntimeError("No model loaded. Call load_model() before infer().")

        try:
            logger.debug(
                f"Running inference with ONNX Runtime "
                f"(model: {self._loaded_model}, "
                f"batch_size: {list(inputs.values())[0].shape[0] if inputs else 'unknown'})"
            )

            start_time = time.time()

            # Validate inputs
            for name in self._input_names:
                if name not in inputs:
                    raise ValueError(f"Missing required input: {name}")

            # Run inference
            onnx_outputs = self._session.run(self._output_names, inputs)

            inference_time = time.time() - start_time

            # Convert outputs to dictionary
            output_dict = {name: output for name, output in zip(self._output_names, onnx_outputs)}

            logger.info(f"ONNX Runtime inference completed in {inference_time:.2f}s")

            return output_dict

        except Exception as e:
            logger.error(f"ONNX Runtime inference failed: {e}", exc_info=True)
            raise RuntimeError(f"ONNX Runtime inference failed: {e}") from e

    def unload_model(self):
        """
        Unload the currently loaded model and free resources.

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> engine.load_model("model.onnx")
            >>> # ... use model ...
            >>> engine.unload_model()
        """
        if self._session is not None:
            logger.info(f"Unloading ONNX model {self._loaded_model}")

            # Clear references
            self._session = None
            self._loaded_model = None
            self._input_names = []
            self._output_names = []

            # Clean up temporary ONNX file if it was in a temp directory
            if self._onnx_path and "/tmp/" in self._onnx_path:
                try:
                    temp_dir = Path(self._onnx_path).parent
                    if temp_dir.exists() and temp_dir.name.startswith("onnx_model_"):
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temporary ONNX directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

            self._onnx_path = None

            logger.info("ONNX model unloaded")

    def get_loaded_model(self) -> Optional[str]:
        """
        Get the name of the currently loaded model.

        Returns:
            Model name if a model is loaded, None otherwise

        Example:
            >>> engine = ONNXEngine(ONNXConfig())
            >>> engine.load_model("model.onnx", "qwen-chat")
            >>> print(engine.get_loaded_model())
            'qwen-chat'
        """
        return self._loaded_model

    def _get_optimization_level(self):
        """
        Convert config optimization level to ONNX Runtime GraphOptimizationLevel.

        Returns:
            onnxruntime.GraphOptimizationLevel enum value
        """
        try:
            import onnxruntime as ort

            level_map = {
                "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }

            return level_map.get(
                self.config.optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

        except ImportError:
            logger.warning("ONNX Runtime not available for optimization level")
            return None

    def _get_available_providers(self) -> List[str]:
        """
        Get list of available execution providers from config.

        Filters config providers to only include those actually available
        in the current ONNX Runtime installation.

        Returns:
            List of available execution provider names
        """
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()

            # Filter config providers to only include available ones
            filtered_providers = [
                provider for provider in self.config.execution_providers if provider in available
            ]

            if not filtered_providers:
                logger.warning(
                    f"None of the configured providers {self.config.execution_providers} "
                    f"are available. Available providers: {available}. "
                    f"Falling back to CPUExecutionProvider."
                )
                filtered_providers = ["CPUExecutionProvider"]

            if filtered_providers != self.config.execution_providers:
                logger.info(
                    f"Using providers {filtered_providers} "
                    f"(requested: {self.config.execution_providers})"
                )

            return filtered_providers

        except ImportError:
            logger.warning("ONNX Runtime not available for provider detection")
            return ["CPUExecutionProvider"]

    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self._session is not None:
            logger.debug("ONNXEngine destructor: unloading model")
            self.unload_model()
