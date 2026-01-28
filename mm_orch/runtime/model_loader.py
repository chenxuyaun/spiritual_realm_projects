"""
模型加载器模块

提供HuggingFace模型的加载功能，支持设备自动检测、数据类型选择和量化加载。
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from mm_orch.exceptions import ModelLoadError, OutOfMemoryError
from mm_orch.runtime.quantization import QuantizationManager

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置数据类"""

    model_name: str
    model_type: str = "auto"  # "qwen-chat", "gpt2", "auto"
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0"
    dtype: str = "auto"  # "auto", "fp32", "fp16", "bf16"
    quantization: Optional[str] = None  # None, "8bit", "4bit", "gptq"
    trust_remote_code: bool = False
    flash_attention: bool = True
    max_memory: Optional[Dict[str, str]] = None
    low_cpu_mem_usage: bool = True
    torch_dtype: Optional[torch.dtype] = field(default=None, repr=False)


@dataclass
class LoadedModel:
    """已加载模型的容器"""

    model: Any
    tokenizer: Any
    config: ModelConfig
    device: str
    dtype: torch.dtype
    memory_footprint: int = 0  # bytes


class ModelLoader:
    """
    模型加载器

    负责加载HuggingFace模型，支持：
    - 设备自动检测和分配
    - 数据类型自动选择
    - 量化加载（8-bit, 4-bit, GPTQ）
    - 错误处理和回退策略
    """

    DTYPE_MAP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模型加载器

        Args:
            config: 全局配置字典
        """
        self.config = config or {}
        self.quantization_manager = QuantizationManager()
        self._validate_environment()

    def _validate_environment(self) -> None:
        """验证运行环境"""
        if not HAS_TRANSFORMERS:
            raise ModelLoadError(
                "transformers library not installed. " "Install with: pip install transformers"
            )

    @staticmethod
    def detect_device(preferred: str = "auto") -> str:
        """
        检测并返回最佳可用设备

        Args:
            preferred: 首选设备 ("auto", "cuda", "cpu", "cuda:N")

        Returns:
            设备字符串
        """
        if preferred == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.info("CUDA not available, falling back to CPU")
                return "cpu"
        elif preferred.startswith("cuda"):
            if torch.cuda.is_available():
                # 验证指定的GPU是否存在
                if ":" in preferred:
                    device_id = int(preferred.split(":")[1])
                    if device_id < torch.cuda.device_count():
                        return preferred
                    else:
                        logger.warning(f"GPU {device_id} not available, using cuda:0")
                        return "cuda:0"
                return preferred
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return preferred

    @staticmethod
    def detect_dtype(preferred: str = "auto", device: str = "cuda") -> torch.dtype:
        """
        检测并返回最佳数据类型

        Args:
            preferred: 首选数据类型 ("auto", "fp32", "fp16", "bf16")
            device: 目标设备

        Returns:
            torch.dtype
        """
        if preferred != "auto":
            return ModelLoader.DTYPE_MAP.get(preferred, torch.float32)

        # 自动选择最佳数据类型
        if device == "cpu":
            return torch.float32

        if torch.cuda.is_available():
            # 检查是否支持bf16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16

        return torch.float32

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, int]:
        """
        获取GPU内存信息

        Returns:
            包含total, allocated, free内存的字典（字节）
        """
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "free": 0}

        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total - allocated

        return {"total": total, "allocated": allocated, "free": free}

    def load_model(
        self,
        config: ModelConfig,
        fallback_to_cpu: bool = True,
        fallback_to_quantization: bool = True,
    ) -> LoadedModel:
        """
        加载模型

        Args:
            config: 模型配置
            fallback_to_cpu: GPU加载失败时是否回退到CPU
            fallback_to_quantization: 内存不足时是否尝试量化

        Returns:
            LoadedModel: 已加载的模型容器

        Raises:
            ModelLoadError: 模型加载失败
            OutOfMemoryError: 内存不足
        """
        # 检测设备和数据类型
        device = self.detect_device(config.device)
        dtype = self.detect_dtype(config.dtype, device)

        logger.info(f"Loading model {config.model_name} on {device} with dtype {dtype}")

        try:
            # 尝试加载模型
            return self._load_model_internal(config, device, dtype)

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e).lower()

            # 处理OOM错误
            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.warning(f"Memory error loading model: {e}")

                # 尝试量化回退
                if fallback_to_quantization and config.quantization is None:
                    logger.info("Attempting to load with 8-bit quantization")
                    config.quantization = "8bit"
                    try:
                        return self._load_model_internal(config, device, dtype)
                    except Exception:
                        logger.info("8-bit failed, trying 4-bit quantization")
                        config.quantization = "4bit"
                        try:
                            return self._load_model_internal(config, device, dtype)
                        except Exception:
                            pass

                # 尝试CPU回退
                if fallback_to_cpu and device != "cpu":
                    logger.info("Falling back to CPU")
                    config.quantization = None
                    return self._load_model_internal(config, "cpu", torch.float32)

                raise OutOfMemoryError(
                    f"Failed to load model {config.model_name}: {e}", device=device
                )

            raise ModelLoadError(
                f"Failed to load model {config.model_name}: {e}", model_name=config.model_name
            )

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model {config.model_name}: {e}", model_name=config.model_name
            )

    def _load_model_internal(
        self, config: ModelConfig, device: str, dtype: torch.dtype
    ) -> LoadedModel:
        """
        内部模型加载实现

        Args:
            config: 模型配置
            device: 目标设备
            dtype: 数据类型

        Returns:
            LoadedModel
        """
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=config.trust_remote_code, padding_side="left"
        )

        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 准备模型加载参数
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
        }

        # 处理量化
        if config.quantization == "gptq":
            model = self.quantization_manager.load_gptq_model(
                config.model_name, device=device, trust_remote_code=config.trust_remote_code
            )
        else:
            # 设置量化配置
            if config.quantization in ["8bit", "4bit"]:
                quant_config = self.quantization_manager.get_quantization_config(
                    config.quantization
                )
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"
            else:
                # 非量化加载
                model_kwargs["torch_dtype"] = dtype
                if device != "cpu":
                    model_kwargs["device_map"] = "auto"

            # 处理FlashAttention
            if config.flash_attention and device != "cpu":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        # 如果没有使用device_map，手动移动到设备
        if "device_map" not in model_kwargs and device != "cpu":
            model = model.to(device)

        # 设置为评估模式
        model.eval()

        # 计算内存占用
        memory_footprint = self._estimate_memory_footprint(model)

        logger.info(
            f"Model loaded successfully. Memory footprint: " f"{memory_footprint / 1024**3:.2f} GB"
        )

        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dtype=dtype,
            memory_footprint=memory_footprint,
        )

    def _estimate_memory_footprint(self, model: Any) -> int:
        """
        估算模型内存占用

        Args:
            model: PyTorch模型

        Returns:
            内存占用（字节）
        """
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return param_size + buffer_size
        except Exception:
            return 0

    def load_tokenizer(self, model_name: str, trust_remote_code: bool = False) -> Any:
        """
        单独加载tokenizer

        Args:
            model_name: 模型名称
            trust_remote_code: 是否信任远程代码

        Returns:
            tokenizer对象
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer
