"""
模型管理器

负责模型的加载、缓存、卸载和推理管理。
支持延迟加载、自动设备选择、LRU缓存和模型量化。
"""

import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

from mm_orch.exceptions import ModelError, ResourceError
from mm_orch.logger import get_logger
from mm_orch.schemas import ModelConfig


@dataclass
class CachedModel:
    """缓存的模型信息"""

    model: Any
    tokenizer: Any
    config: ModelConfig
    device: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    def update_usage(self) -> None:
        """更新使用时间和计数"""
        self.last_used = time.time()
        self.use_count += 1


class ModelManager:
    """
    模型管理器

    功能:
    - 延迟加载：仅在首次使用时加载模型
    - 自动设备选择：优先GPU，内存不足时降级CPU
    - LRU缓存：最多缓存max_cached_models个模型
    - 模型量化：支持8bit/4bit量化以节省内存

    属性:
    - 属性24: 模型延迟加载
    - 属性25: 模型设备选择
    - 属性26: 模型缓存复用
    - 属性27: 模型LRU淘汰
    """

    def __init__(
        self,
        max_cached_models: int = 3,
        default_device: str = "auto",
        enable_quantization: bool = True,
    ):
        """
        初始化模型管理器

        Args:
            max_cached_models: 最大缓存模型数量
            default_device: 默认设备 ('auto', 'cuda', 'cpu')
            enable_quantization: 是否启用量化
        """
        self.max_cached_models = max_cached_models
        self.default_device = default_device
        self.enable_quantization = enable_quantization

        # 使用OrderedDict实现LRU缓存
        self._cache: OrderedDict[str, CachedModel] = OrderedDict()

        # 模型配置注册表
        self._model_configs: Dict[str, ModelConfig] = {}

        # 日志记录器
        self._logger = get_logger()

        # 检测可用设备
        self._cuda_available = self._check_cuda_available()

    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_memory_info(self) -> Tuple[float, float]:
        """
        获取GPU内存信息

        Returns:
            (已用内存GB, 总内存GB)
        """
        if not self._cuda_available:
            return (0.0, 0.0)

        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory
                allocated = torch.cuda.memory_allocated(device)
                return (allocated / 1e9, total / 1e9)
        except Exception:
            pass
        return (0.0, 0.0)

    def _has_sufficient_gpu_memory(self, required_gb: float = 2.0) -> bool:
        """
        检查是否有足够的GPU内存

        Args:
            required_gb: 所需内存（GB）

        Returns:
            是否有足够内存
        """
        if not self._cuda_available:
            return False

        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory
                allocated = torch.cuda.memory_allocated(device)
                free = (total - allocated) / 1e9
                return free >= required_gb
        except Exception:
            pass
        return False

    def _select_device(self, requested_device: str) -> str:
        """
        选择最佳设备

        Args:
            requested_device: 请求的设备 ('auto', 'cuda', 'cpu')

        Returns:
            实际使用的设备
        """
        if requested_device == "cpu":
            return "cpu"

        if requested_device == "cuda":
            if self._cuda_available and self._has_sufficient_gpu_memory():
                return "cuda"
            else:
                self._logger.warning(
                    "CUDA requested but not available or insufficient memory, falling back to CPU",
                    context={"requested": requested_device},
                )
                return "cpu"

        # auto模式：优先GPU
        if requested_device == "auto":
            if self._cuda_available and self._has_sufficient_gpu_memory():
                return "cuda"
            return "cpu"

        return "cpu"

    def register_model(self, config: ModelConfig) -> None:
        """
        注册模型配置

        Args:
            config: 模型配置
        """
        self._model_configs[config.name] = config
        self._logger.debug(
            f"Registered model configuration: {config.name}",
            context={"model_path": config.model_path, "device": config.device},
        )

    def _evict_lru_model(self) -> Optional[str]:
        """
        淘汰最久未使用的模型

        Returns:
            被淘汰的模型名称，如果没有则返回None
        """
        if not self._cache:
            return None

        # OrderedDict的第一个元素是最久未使用的
        oldest_name = next(iter(self._cache))
        self.unload_model(oldest_name)
        return oldest_name

    def _load_model_from_path(self, config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """
        从路径加载模型和分词器

        Args:
            config: 模型配置
            device: 目标设备

        Returns:
            (model, tokenizer) 元组

        Raises:
            ModelError: 加载失败时抛出
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

            model_path = config.model_path

            self._logger.info(
                f"Loading model: {config.name}", context={"path": model_path, "device": device}
            )

            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # 准备加载参数
            load_kwargs = {
                "trust_remote_code": True,
            }

            # 设置设备映射
            if device == "cuda":
                load_kwargs["device_map"] = "auto"

                # 应用量化
                if self.enable_quantization and config.quantization:
                    if config.quantization == "8bit":
                        load_kwargs["load_in_8bit"] = True
                    elif config.quantization == "4bit":
                        load_kwargs["load_in_4bit"] = True
            else:
                load_kwargs["device_map"] = "cpu"

            # 尝试加载为因果语言模型
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            except Exception:
                # 回退到通用模型加载
                model = AutoModel.from_pretrained(model_path, **load_kwargs)

            self._logger.info(
                f"Model loaded successfully: {config.name}", context={"device": device}
            )

            return model, tokenizer

        except ImportError as e:
            raise ModelError(
                f"Required library not installed: {e}", context={"model_name": config.name}
            )
        except Exception as e:
            raise ModelError(
                f"Failed to load model {config.name}: {e}",
                context={"model_path": config.model_path, "device": device},
            )

    def load_model(self, model_name: str, device: Optional[str] = None) -> CachedModel:
        """
        加载模型到内存

        Args:
            model_name: 模型标识符
            device: 目标设备 ('cuda', 'cpu', 'auto')，None使用默认值

        Returns:
            CachedModel对象

        Raises:
            ModelError: 模型未注册或加载失败
        """
        # 检查是否已缓存
        if model_name in self._cache:
            cached = self._cache[model_name]
            cached.update_usage()
            # 移动到OrderedDict末尾（最近使用）
            self._cache.move_to_end(model_name)
            self._logger.debug(
                f"Model retrieved from cache: {model_name}", context={"use_count": cached.use_count}
            )
            return cached

        # 检查模型是否已注册
        if model_name not in self._model_configs:
            raise ModelError(
                f"Model not registered: {model_name}",
                context={"available_models": list(self._model_configs.keys())},
            )

        config = self._model_configs[model_name]

        # 选择设备
        target_device = self._select_device(device or config.device)

        # 检查缓存容量，必要时淘汰
        while len(self._cache) >= self.max_cached_models:
            evicted = self._evict_lru_model()
            if evicted:
                self._logger.info(
                    f"Evicted LRU model: {evicted}", context={"cache_size": len(self._cache)}
                )

        # 加载模型
        model, tokenizer = self._load_model_from_path(config, target_device)

        # 创建缓存条目
        cached_model = CachedModel(
            model=model, tokenizer=tokenizer, config=config, device=target_device
        )

        # 添加到缓存
        self._cache[model_name] = cached_model

        self._logger.info(
            f"Model cached: {model_name}",
            context={"device": target_device, "cache_size": len(self._cache)},
        )

        return cached_model

    def get_model(self, model_name: str) -> CachedModel:
        """
        获取模型（从缓存或加载）

        Args:
            model_name: 模型标识符

        Returns:
            CachedModel对象
        """
        return self.load_model(model_name)

    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型释放内存

        Args:
            model_name: 模型标识符

        Returns:
            是否成功卸载
        """
        if model_name not in self._cache:
            return False

        cached = self._cache.pop(model_name)

        # 清理模型资源
        try:
            del cached.model
            del cached.tokenizer

            # 强制垃圾回收
            gc.collect()

            # 清理CUDA缓存
            if self._cuda_available:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            self._logger.info(
                f"Model unloaded: {model_name}", context={"cache_size": len(self._cache)}
            )
            return True

        except Exception as e:
            self._logger.error(f"Error unloading model: {model_name}", context={"error": str(e)})
            return False

    def unload_all(self) -> int:
        """
        卸载所有缓存的模型

        Returns:
            卸载的模型数量
        """
        model_names = list(self._cache.keys())
        count = 0
        for name in model_names:
            if self.unload_model(name):
                count += 1
        return count

    def infer(self, model_name: str, inputs: Union[str, List[str]], **kwargs) -> Any:
        """
        执行模型推理

        Args:
            model_name: 模型标识符
            inputs: 输入文本或文本列表
            **kwargs: 额外的生成参数

        Returns:
            模型输出

        Raises:
            ModelError: 推理失败时抛出
        """
        cached = self.get_model(model_name)

        try:
            model = cached.model
            tokenizer = cached.tokenizer
            config = cached.config

            # 准备输入
            if isinstance(inputs, str):
                inputs = [inputs]

            # 分词
            encoded = tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )

            # 移动到正确的设备
            if cached.device == "cuda":
                try:
                    import torch

                    encoded = {k: v.to("cuda") for k, v in encoded.items()}
                except Exception:
                    pass

            # 设置生成参数
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 256),
                "temperature": kwargs.get("temperature", config.temperature),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }

            # 生成
            with self._inference_context():
                outputs = model.generate(**encoded, **gen_kwargs)

            # 解码输出
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 更新使用统计
            cached.update_usage()

            return decoded[0] if len(decoded) == 1 else decoded

        except Exception as e:
            raise ModelError(
                f"Inference failed for model {model_name}: {e}",
                context={"inputs_count": len(inputs) if isinstance(inputs, list) else 1},
            )

    def _inference_context(self):
        """获取推理上下文管理器"""
        try:
            import torch

            return torch.no_grad()
        except ImportError:
            # 如果torch不可用，返回一个空的上下文管理器
            from contextlib import nullcontext

            return nullcontext()

    def is_loaded(self, model_name: str) -> bool:
        """
        检查模型是否已加载

        Args:
            model_name: 模型标识符

        Returns:
            是否已加载
        """
        return model_name in self._cache

    def is_registered(self, model_name: str) -> bool:
        """
        检查模型是否已注册

        Args:
            model_name: 模型标识符

        Returns:
            是否已注册
        """
        return model_name in self._model_configs

    def get_cached_models(self) -> List[str]:
        """
        获取当前缓存的模型列表

        Returns:
            模型名称列表
        """
        return list(self._cache.keys())

    def get_registered_models(self) -> List[str]:
        """
        获取已注册的模型列表

        Returns:
            模型名称列表
        """
        return list(self._model_configs.keys())

    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息

        Returns:
            缓存统计信息
        """
        info = {
            "max_cached_models": self.max_cached_models,
            "current_cached": len(self._cache),
            "registered_models": len(self._model_configs),
            "cuda_available": self._cuda_available,
            "models": {},
        }

        for name, cached in self._cache.items():
            info["models"][name] = {
                "device": cached.device,
                "loaded_at": cached.loaded_at,
                "last_used": cached.last_used,
                "use_count": cached.use_count,
            }

        return info

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        获取模型配置

        Args:
            model_name: 模型标识符

        Returns:
            模型配置，如果未注册则返回None
        """
        return self._model_configs.get(model_name)

    def clear_cache(self) -> None:
        """清空模型缓存"""
        self.unload_all()
        self._cache.clear()

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ModelManager(max_cached={self.max_cached_models}, "
            f"cached={len(self._cache)}, "
            f"registered={len(self._model_configs)})"
        )


# 全局模型管理器实例
_global_model_manager: Optional[ModelManager] = None


def get_model_manager(max_cached_models: int = 3, default_device: str = "auto") -> ModelManager:
    """
    获取全局模型管理器实例

    Args:
        max_cached_models: 最大缓存模型数量
        default_device: 默认设备

    Returns:
        ModelManager实例
    """
    global _global_model_manager

    if _global_model_manager is None:
        _global_model_manager = ModelManager(
            max_cached_models=max_cached_models, default_device=default_device
        )

    return _global_model_manager


def configure_model_manager(
    max_cached_models: int = 3, default_device: str = "auto", enable_quantization: bool = True
) -> ModelManager:
    """
    配置全局模型管理器

    Args:
        max_cached_models: 最大缓存模型数量
        default_device: 默认设备
        enable_quantization: 是否启用量化

    Returns:
        配置后的ModelManager实例
    """
    global _global_model_manager
    _global_model_manager = ModelManager(
        max_cached_models=max_cached_models,
        default_device=default_device,
        enable_quantization=enable_quantization,
    )
    return _global_model_manager
