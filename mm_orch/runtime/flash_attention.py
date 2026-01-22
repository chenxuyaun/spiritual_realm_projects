"""
FlashAttention集成模块

提供FlashAttention可用性检测和自动启用/回退功能。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class FlashAttentionInfo:
    """FlashAttention信息"""
    available: bool
    version: Optional[str] = None
    cuda_version: Optional[str] = None
    reason_unavailable: Optional[str] = None


class FlashAttentionManager:
    """
    FlashAttention管理器
    
    负责检测FlashAttention可用性并提供自动启用/回退逻辑。
    """
    
    _instance = None
    _checked = False
    _available = False
    _info: Optional[FlashAttentionInfo] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def is_available(cls) -> bool:
        """
        检查FlashAttention是否可用
        
        Returns:
            bool: 是否可用
        """
        if not cls._checked:
            cls._check_availability()
        return cls._available
    
    @classmethod
    def get_info(cls) -> FlashAttentionInfo:
        """
        获取FlashAttention详细信息
        
        Returns:
            FlashAttentionInfo: 详细信息
        """
        if not cls._checked:
            cls._check_availability()
        return cls._info or FlashAttentionInfo(available=False)
    
    @classmethod
    def _check_availability(cls) -> None:
        """检查FlashAttention可用性"""
        cls._checked = True
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            cls._info = FlashAttentionInfo(
                available=False,
                reason_unavailable="CUDA not available"
            )
            cls._available = False
            logger.info("FlashAttention unavailable: CUDA not available")
            return
        
        # 检查GPU计算能力
        try:
            device_props = torch.cuda.get_device_properties(0)
            compute_capability = (
                device_props.major,
                device_props.minor
            )
            
            # FlashAttention 2 需要 SM 80+ (Ampere及以上)
            if compute_capability < (8, 0):
                cls._info = FlashAttentionInfo(
                    available=False,
                    cuda_version=torch.version.cuda,
                    reason_unavailable=(
                        f"GPU compute capability {compute_capability} < (8, 0). "
                        "FlashAttention 2 requires Ampere (SM 80) or newer GPU."
                    )
                )
                cls._available = False
                logger.info(
                    f"FlashAttention unavailable: GPU compute capability "
                    f"{compute_capability} < (8, 0)"
                )
                return
        except Exception as e:
            cls._info = FlashAttentionInfo(
                available=False,
                reason_unavailable=f"Failed to check GPU properties: {e}"
            )
            cls._available = False
            return
        
        # 尝试导入flash_attn
        try:
            import flash_attn
            version = getattr(flash_attn, "__version__", "unknown")
            
            cls._info = FlashAttentionInfo(
                available=True,
                version=version,
                cuda_version=torch.version.cuda
            )
            cls._available = True
            logger.info(f"FlashAttention {version} available")
            
        except ImportError:
            cls._info = FlashAttentionInfo(
                available=False,
                cuda_version=torch.version.cuda,
                reason_unavailable=(
                    "flash-attn package not installed. "
                    "Install with: pip install flash-attn"
                )
            )
            cls._available = False
            logger.info("FlashAttention unavailable: package not installed")
        
        except Exception as e:
            cls._info = FlashAttentionInfo(
                available=False,
                cuda_version=torch.version.cuda,
                reason_unavailable=f"Failed to import flash_attn: {e}"
            )
            cls._available = False
            logger.warning(f"FlashAttention check failed: {e}")
    
    @classmethod
    def get_attention_implementation(
        cls,
        prefer_flash: bool = True
    ) -> str:
        """
        获取推荐的注意力实现
        
        Args:
            prefer_flash: 是否优先使用FlashAttention
            
        Returns:
            str: 注意力实现名称 ("flash_attention_2", "sdpa", "eager")
        """
        if prefer_flash and cls.is_available():
            return "flash_attention_2"
        
        # 检查是否支持SDPA (Scaled Dot Product Attention)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"
        
        return "eager"
    
    @classmethod
    def configure_model_for_flash_attention(
        cls,
        model_kwargs: Dict[str, Any],
        enable_flash: bool = True
    ) -> Dict[str, Any]:
        """
        配置模型参数以启用FlashAttention
        
        Args:
            model_kwargs: 模型加载参数
            enable_flash: 是否启用FlashAttention
            
        Returns:
            更新后的模型参数
        """
        if enable_flash and cls.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Configured model for FlashAttention 2")
        elif enable_flash:
            # FlashAttention不可用，尝试SDPA
            impl = cls.get_attention_implementation(prefer_flash=False)
            if impl != "eager":
                model_kwargs["attn_implementation"] = impl
                logger.info(f"FlashAttention unavailable, using {impl}")
        
        return model_kwargs
    
    @classmethod
    def reset(cls) -> None:
        """重置检查状态（用于测试）"""
        cls._checked = False
        cls._available = False
        cls._info = None


def is_flash_attention_available() -> bool:
    """
    便捷函数：检查FlashAttention是否可用
    
    Returns:
        bool: 是否可用
    """
    return FlashAttentionManager.is_available()


def get_flash_attention_info() -> FlashAttentionInfo:
    """
    便捷函数：获取FlashAttention信息
    
    Returns:
        FlashAttentionInfo: 详细信息
    """
    return FlashAttentionManager.get_info()


def get_best_attention_implementation() -> str:
    """
    便捷函数：获取最佳注意力实现
    
    Returns:
        str: 注意力实现名称
    """
    return FlashAttentionManager.get_attention_implementation()
