"""
量化管理器模块

提供模型量化配置和加载功能，支持8-bit和4-bit量化。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

try:
    from transformers import BitsAndBytesConfig

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    BitsAndBytesConfig = None

try:
    from auto_gptq import AutoGPTQForCausalLM

    HAS_AUTO_GPTQ = True
except ImportError:
    HAS_AUTO_GPTQ = False
    AutoGPTQForCausalLM = None

from mm_orch.exceptions import QuantizationError

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """量化配置数据类"""

    quant_type: str  # "8bit", "4bit", "gptq"
    compute_dtype: torch.dtype = torch.float16
    quant_storage_dtype: Optional[torch.dtype] = None
    use_double_quant: bool = True
    quant_method: str = "nf4"  # "nf4" or "fp4" for 4-bit
    llm_int8_threshold: float = 6.0  # for 8-bit


class QuantizationManager:
    """
    量化管理器

    负责创建量化配置和加载量化模型。
    支持bitsandbytes的8-bit/4-bit量化和GPTQ量化。
    """

    SUPPORTED_QUANT_TYPES = ["8bit", "4bit", "gptq"]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化量化管理器

        Args:
            config: 量化配置字典，可选
        """
        self.config = config or {}
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """验证必要的依赖是否已安装"""
        if not HAS_TRANSFORMERS:
            logger.warning("transformers not installed, quantization features limited")

    @staticmethod
    def is_quantization_available(quant_type: str) -> bool:
        """
        检查指定的量化类型是否可用

        Args:
            quant_type: 量化类型 ("8bit", "4bit", "gptq")

        Returns:
            bool: 是否可用
        """
        if quant_type in ["8bit", "4bit"]:
            try:
                pass

                return True
            except ImportError:
                return False
        elif quant_type == "gptq":
            return HAS_AUTO_GPTQ
        return False

    @staticmethod
    def get_quantization_config(
        quant_type: str,
        compute_dtype: Optional[torch.dtype] = None,
        use_double_quant: bool = True,
        llm_int8_threshold: float = 6.0,
    ) -> Optional["BitsAndBytesConfig"]:
        """
        获取BitsAndBytes量化配置

        Args:
            quant_type: 量化类型 ("8bit" 或 "4bit")
            compute_dtype: 计算数据类型，默认fp16
            use_double_quant: 是否使用双重量化（仅4-bit）
            llm_int8_threshold: INT8阈值（仅8-bit）

        Returns:
            BitsAndBytesConfig: 量化配置对象，如果不支持则返回None

        Raises:
            QuantizationError: 量化配置创建失败
        """
        if not HAS_TRANSFORMERS:
            raise QuantizationError(
                "transformers library not installed", quantization_type=quant_type
            )

        if compute_dtype is None:
            compute_dtype = torch.float16

        try:
            if quant_type == "8bit":
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=llm_int8_threshold,
                    llm_int8_has_fp16_weight=True,
                )
            elif quant_type == "4bit":
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=use_double_quant,
                )
            else:
                logger.warning(f"Unsupported quant_type for BitsAndBytes: {quant_type}")
                return None
        except Exception as e:
            raise QuantizationError(
                f"Failed to create quantization config: {e}", quantization_type=quant_type
            )

    @staticmethod
    def load_gptq_model(
        model_name: str,
        device: str = "cuda",
        trust_remote_code: bool = False,
        use_safetensors: bool = True,
        **kwargs,
    ) -> Any:
        """
        加载GPTQ量化模型

        Args:
            model_name: HuggingFace模型名称或本地路径
            device: 目标设备 ("cuda" 或 "cpu")
            trust_remote_code: 是否信任远程代码
            use_safetensors: 是否使用safetensors格式
            **kwargs: 其他AutoGPTQ参数

        Returns:
            加载的GPTQ模型

        Raises:
            QuantizationError: GPTQ模型加载失败
        """
        if not HAS_AUTO_GPTQ:
            raise QuantizationError(
                "auto-gptq library not installed. Install with: pip install auto-gptq",
                quantization_type="gptq",
                model_name=model_name,
            )

        try:
            logger.info(f"Loading GPTQ model: {model_name}")

            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device=device,
                trust_remote_code=trust_remote_code,
                use_safetensors=use_safetensors,
                **kwargs,
            )

            logger.info(f"Successfully loaded GPTQ model: {model_name}")
            return model

        except Exception as e:
            raise QuantizationError(
                f"Failed to load GPTQ model: {e}", quantization_type="gptq", model_name=model_name
            )

    def get_recommended_quant_type(
        self, model_size_gb: float, available_memory_gb: float
    ) -> Optional[str]:
        """
        根据模型大小和可用内存推荐量化类型

        Args:
            model_size_gb: 模型大小（GB）
            available_memory_gb: 可用内存（GB）

        Returns:
            推荐的量化类型，如果不需要量化则返回None
        """
        # 估算不同量化级别的内存需求
        # 全精度: model_size_gb
        # 8-bit: model_size_gb * 0.5
        # 4-bit: model_size_gb * 0.25

        if model_size_gb <= available_memory_gb * 0.8:
            return None  # 不需要量化
        elif model_size_gb * 0.5 <= available_memory_gb * 0.8:
            return "8bit"
        elif model_size_gb * 0.25 <= available_memory_gb * 0.8:
            return "4bit"
        else:
            logger.warning(
                f"Model ({model_size_gb}GB) may not fit in memory ({available_memory_gb}GB) "
                "even with 4-bit quantization"
            )
            return "4bit"

    def estimate_memory_usage(
        self, model_size_gb: float, quant_type: Optional[str] = None
    ) -> float:
        """
        估算量化后的内存使用量

        Args:
            model_size_gb: 原始模型大小（GB）
            quant_type: 量化类型

        Returns:
            估算的内存使用量（GB）
        """
        if quant_type is None:
            return model_size_gb
        elif quant_type == "8bit":
            return model_size_gb * 0.5
        elif quant_type in ["4bit", "gptq"]:
            return model_size_gb * 0.25
        else:
            return model_size_gb
