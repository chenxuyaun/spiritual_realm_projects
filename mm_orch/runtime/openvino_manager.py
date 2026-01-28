"""
OpenVINO模型管理器

提供OpenVINO加速的模型加载和推理功能。
与现有ModelManager兼容，可无缝切换。
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from mm_orch.exceptions import ModelError
from mm_orch.logger import get_logger
from mm_orch.schemas import ModelConfig

# OpenVINO imports
try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    OVModelForCausalLM = None


@dataclass
class OpenVINOCachedModel:
    """OpenVINO缓存的模型信息"""
    model: Any
    tokenizer: Any
    config: ModelConfig
    device: str
    backend: str  # 'openvino' or 'pytorch'
    loaded_at: float
    last_used: float
    use_count: int = 0
    
    def update_usage(self) -> None:
        """更新使用时间和计数"""
        self.last_used = time.time()
        self.use_count += 1


class OpenVINOModelManager:
    """
    OpenVINO模型管理器
    
    提供与ModelManager兼容的接口，但使用OpenVINO加速。
    
    特性:
    - 自动模型导出到OpenVINO格式
    - CPU/iGPU/NPU设备支持
    - 2-3x性能提升
    - 向后兼容PyTorch
    """
    
    def __init__(
        self,
        openvino_cache_dir: str = "models/openvino",
        default_device: str = "CPU",
        enable_openvino: bool = True,
        fallback_to_pytorch: bool = True,
        max_cached_models: int = 3,
    ):
        """
        初始化OpenVINO模型管理器
        
        Args:
            openvino_cache_dir: OpenVINO模型缓存目录
            default_device: 默认设备 ('CPU', 'GPU', 'NPU')
            enable_openvino: 是否启用OpenVINO
            fallback_to_pytorch: OpenVINO失败时是否回退到PyTorch
            max_cached_models: 最大缓存模型数量
        """
        self.openvino_cache_dir = Path(openvino_cache_dir)
        self.default_device = default_device
        self.enable_openvino = enable_openvino and OPENVINO_AVAILABLE
        self.fallback_to_pytorch = fallback_to_pytorch
        self.max_cached_models = max_cached_models
        
        # 缓存
        self._cache: Dict[str, OpenVINOCachedModel] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        
        # 日志
        self._logger = get_logger()
        
        # 检查OpenVINO可用性
        if not OPENVINO_AVAILABLE:
            self._logger.warning(
                "OpenVINO not available. Install with: pip install optimum[openvino]"
            )
            if not fallback_to_pytorch:
                raise RuntimeError("OpenVINO not available and fallback disabled")
        
        # 创建缓存目录
        self.openvino_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger.info(
            f"OpenVINO Manager initialized",
            context={
                "openvino_available": OPENVINO_AVAILABLE,
                "default_device": default_device,
                "cache_dir": str(self.openvino_cache_dir)
            }
        )
    
    def _get_openvino_model_path(self, model_name: str) -> Path:
        """获取OpenVINO模型路径"""
        return self.openvino_cache_dir / model_name
    
    def _is_openvino_model_exported(self, model_name: str) -> bool:
        """检查模型是否已导出为OpenVINO格式"""
        model_path = self._get_openvino_model_path(model_name)
        return (model_path / "openvino_model.xml").exists()
    
    def _export_to_openvino(
        self,
        model_path: str,
        model_name: str
    ) -> Path:
        """
        导出模型到OpenVINO格式
        
        Args:
            model_path: 原始模型路径
            model_name: 模型名称
            
        Returns:
            导出的模型路径
        """
        if not OPENVINO_AVAILABLE:
            raise ModelError("OpenVINO not available for model export")
        
        output_path = self._get_openvino_model_path(model_name)
        
        # 如果已导出，直接返回
        if self._is_openvino_model_exported(model_name):
            self._logger.info(f"Model already exported: {model_name}")
            return output_path
        
        self._logger.info(
            f"Exporting model to OpenVINO: {model_name}",
            context={"source": model_path, "target": str(output_path)}
        )
        
        try:
            # 导出模型
            start_time = time.time()
            
            model = OVModelForCausalLM.from_pretrained(
                model_path,
                export=True,
                compile=False  # 不立即编译
            )
            
            # 保存导出的模型
            output_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_path)
            
            export_time = time.time() - start_time
            
            self._logger.info(
                f"Model exported successfully: {model_name}",
                context={"export_time": f"{export_time:.2f}s", "path": str(output_path)}
            )
            
            return output_path
            
        except Exception as e:
            self._logger.error(
                f"Failed to export model: {model_name}",
                context={"error": str(e)}
            )
            raise ModelError(f"Model export failed: {e}")
    
    def _load_openvino_model(
        self,
        model_name: str,
        model_path: Path,
        device: str
    ) -> Tuple[Any, Any]:
        """
        加载OpenVINO模型
        
        Args:
            model_name: 模型名称
            model_path: OpenVINO模型路径
            device: 目标设备
            
        Returns:
            (model, tokenizer) 元组
        """
        if not OPENVINO_AVAILABLE:
            raise ModelError("OpenVINO not available")
        
        self._logger.info(
            f"Loading OpenVINO model: {model_name}",
            context={"device": device, "path": str(model_path)}
        )
        
        try:
            start_time = time.time()
            
            # 加载模型
            model = OVModelForCausalLM.from_pretrained(
                model_path,
                device=device,
                compile=True
            )
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            load_time = time.time() - start_time
            
            self._logger.info(
                f"OpenVINO model loaded: {model_name}",
                context={"device": device, "load_time": f"{load_time:.2f}s"}
            )
            
            return model, tokenizer
            
        except Exception as e:
            self._logger.error(
                f"Failed to load OpenVINO model: {model_name}",
                context={"error": str(e), "device": device}
            )
            raise ModelError(f"OpenVINO model load failed: {e}")
    
    def _load_pytorch_model(
        self,
        model_path: str,
        device: str
    ) -> Tuple[Any, Any]:
        """
        加载PyTorch模型（回退方案）
        
        Args:
            model_path: 模型路径
            device: 目标设备
            
        Returns:
            (model, tokenizer) 元组
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._logger.info(
                f"Loading PyTorch model (fallback)",
                context={"path": model_path, "device": device}
            )
            
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
            
        except Exception as e:
            raise ModelError(f"PyTorch model load failed: {e}")
    
    def register_model(self, config: ModelConfig) -> None:
        """
        注册模型配置
        
        Args:
            config: 模型配置
        """
        self._model_configs[config.name] = config
        self._logger.debug(
            f"Registered model: {config.name}",
            context={"model_path": config.model_path}
        )
    
    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        force_backend: Optional[str] = None
    ) -> OpenVINOCachedModel:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            device: 目标设备
            force_backend: 强制使用的后端 ('openvino' or 'pytorch')
            
        Returns:
            OpenVINOCachedModel对象
        """
        # 检查缓存
        if model_name in self._cache:
            cached = self._cache[model_name]
            cached.update_usage()
            self._logger.debug(
                f"Model retrieved from cache: {model_name}",
                context={"backend": cached.backend, "use_count": cached.use_count}
            )
            return cached
        
        # 检查是否已注册
        if model_name not in self._model_configs:
            raise ModelError(
                f"Model not registered: {model_name}",
                context={"available": list(self._model_configs.keys())}
            )
        
        config = self._model_configs[model_name]
        target_device = device or self.default_device
        
        # 确定使用的后端
        use_openvino = (
            (force_backend == 'openvino' or (force_backend is None and self.enable_openvino))
            and OPENVINO_AVAILABLE
        )
        
        model = None
        tokenizer = None
        backend = 'pytorch'
        
        # 尝试OpenVINO
        if use_openvino:
            try:
                # 导出模型（如果需要）
                openvino_path = self._export_to_openvino(
                    config.model_path,
                    model_name
                )
                
                # 加载OpenVINO模型
                model, tokenizer = self._load_openvino_model(
                    model_name,
                    openvino_path,
                    target_device
                )
                backend = 'openvino'
                
            except Exception as e:
                self._logger.warning(
                    f"OpenVINO loading failed: {model_name}",
                    context={"error": str(e)}
                )
                
                if not self.fallback_to_pytorch:
                    raise
        
        # 回退到PyTorch
        if model is None and self.fallback_to_pytorch:
            self._logger.info(f"Falling back to PyTorch: {model_name}")
            model, tokenizer = self._load_pytorch_model(
                config.model_path,
                'cpu'  # PyTorch fallback uses CPU
            )
            backend = 'pytorch'
        
        if model is None:
            raise ModelError(f"Failed to load model: {model_name}")
        
        # 创建缓存条目
        cached_model = OpenVINOCachedModel(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=target_device,
            backend=backend,
            loaded_at=time.time(),
            last_used=time.time(),
            use_count=1
        )
        
        # 检查缓存容量
        while len(self._cache) >= self.max_cached_models:
            self._evict_lru_model()
        
        # 添加到缓存
        self._cache[model_name] = cached_model
        
        self._logger.info(
            f"Model cached: {model_name}",
            context={
                "backend": backend,
                "device": target_device,
                "cache_size": len(self._cache)
            }
        )
        
        return cached_model
    
    def _evict_lru_model(self) -> Optional[str]:
        """淘汰最久未使用的模型"""
        if not self._cache:
            return None
        
        # 找到最久未使用的模型
        lru_model = min(
            self._cache.items(),
            key=lambda x: x[1].last_used
        )
        
        model_name = lru_model[0]
        self.unload_model(model_name)
        return model_name
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功卸载
        """
        if model_name not in self._cache:
            return False
        
        cached = self._cache.pop(model_name)
        
        try:
            del cached.model
            del cached.tokenizer
            gc.collect()
            
            self._logger.info(
                f"Model unloaded: {model_name}",
                context={"cache_size": len(self._cache)}
            )
            return True
            
        except Exception as e:
            self._logger.error(
                f"Error unloading model: {model_name}",
                context={"error": str(e)}
            )
            return False
    
    def infer(
        self,
        model_name: str,
        inputs: Union[str, List[str]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        执行推理
        
        Args:
            model_name: 模型名称
            inputs: 输入文本或文本列表
            **kwargs: 生成参数
            
        Returns:
            生成的文本
        """
        cached = self.load_model(model_name)
        
        try:
            model = cached.model
            tokenizer = cached.tokenizer
            
            # 准备输入
            if isinstance(inputs, str):
                inputs_list = [inputs]
                single_input = True
            else:
                inputs_list = inputs
                single_input = False
            
            # 分词
            encoded = tokenizer(
                inputs_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=kwargs.get("max_length", 512)
            )
            
            # 生成参数
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 50),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            
            # 生成
            import torch
            with torch.no_grad():
                outputs = model.generate(**encoded, **gen_kwargs)
            
            # 解码
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 更新使用统计
            cached.update_usage()
            
            # 返回结果
            return decoded[0] if single_input else decoded
            
        except Exception as e:
            raise ModelError(
                f"Inference failed: {model_name}",
                context={"error": str(e)}
            )
    
    def get_model(self, model_name: str) -> OpenVINOCachedModel:
        """获取模型（从缓存或加载）"""
        return self.load_model(model_name)
    
    def is_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        return model_name in self._cache
    
    def is_registered(self, model_name: str) -> bool:
        """检查模型是否已注册"""
        return model_name in self._model_configs
    
    def get_cached_models(self) -> List[str]:
        """获取缓存的模型列表"""
        return list(self._cache.keys())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            "max_cached_models": self.max_cached_models,
            "current_cached": len(self._cache),
            "openvino_enabled": self.enable_openvino,
            "openvino_available": OPENVINO_AVAILABLE,
            "default_device": self.default_device,
            "models": {
                name: {
                    "backend": cached.backend,
                    "device": cached.device,
                    "loaded_at": cached.loaded_at,
                    "last_used": cached.last_used,
                    "use_count": cached.use_count,
                }
                for name, cached in self._cache.items()
            }
        }
    
    def unload_all(self) -> int:
        """卸载所有模型"""
        model_names = list(self._cache.keys())
        count = 0
        for name in model_names:
            if self.unload_model(name):
                count += 1
        return count


# 全局实例
_global_openvino_manager: Optional[OpenVINOModelManager] = None


def get_openvino_manager(
    openvino_cache_dir: str = "models/openvino",
    default_device: str = "CPU",
    enable_openvino: bool = True,
) -> OpenVINOModelManager:
    """
    获取全局OpenVINO管理器实例
    
    Args:
        openvino_cache_dir: OpenVINO缓存目录
        default_device: 默认设备
        enable_openvino: 是否启用OpenVINO
        
    Returns:
        OpenVINOModelManager实例
    """
    global _global_openvino_manager
    
    if _global_openvino_manager is None:
        _global_openvino_manager = OpenVINOModelManager(
            openvino_cache_dir=openvino_cache_dir,
            default_device=default_device,
            enable_openvino=enable_openvino,
        )
    
    return _global_openvino_manager
