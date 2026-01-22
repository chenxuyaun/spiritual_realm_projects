"""
真实模型管理器模块

提供真实LLM模型的加载、缓存和管理功能。
"""

import gc
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from mm_orch.exceptions import (
    ModelLoadError,
    OutOfMemoryError,
    InferenceError,
)
from mm_orch.runtime.model_loader import ModelConfig, ModelLoader, LoadedModel
from mm_orch.runtime.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class CachedModel:
    """缓存的模型信息"""
    loaded_model: LoadedModel
    last_accessed: datetime
    access_count: int = 0
    
    def touch(self) -> None:
        """更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class RealModelManager:
    """
    真实模型管理器
    
    负责管理真实LLM模型的生命周期：
    - 模型加载和卸载
    - LRU缓存策略
    - GPU/CPU自动切换
    - 内存监控和管理
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_cached_models: int = 3,
        memory_threshold: float = 0.8,
        fallback_to_cpu: bool = True
    ):
        """
        初始化模型管理器
        
        Args:
            config: 配置字典
            max_cached_models: 最大缓存模型数
            memory_threshold: 内存使用阈值（触发卸载）
            fallback_to_cpu: GPU不可用时是否回退到CPU
        """
        self.config = config or {}
        self.max_cached_models = max_cached_models
        self.memory_threshold = memory_threshold
        self.fallback_to_cpu = fallback_to_cpu
        
        # 模型缓存（LRU）
        self._model_cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._lock = threading.RLock()
        
        # 组件
        self._model_loader = ModelLoader(config)
        self._memory_monitor = MemoryMonitor(
            gpu_warning_threshold=memory_threshold,
            gpu_critical_threshold=0.95
        )
        
        # 注册内存告警回调
        self._memory_monitor.register_alert_callback(self._on_memory_alert)
        
        # 统计信息
        self._stats = {
            "loads": 0,
            "unloads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallbacks": 0,
        }
    
    def _on_memory_alert(self, alert) -> None:
        """内存告警回调"""
        if alert.alert_type in ["gpu_critical", "cpu_critical"]:
            logger.warning(f"Critical memory alert: {alert.message}")
            # 尝试卸载最少使用的模型
            self._evict_lru_model()
    
    def load_model(
        self,
        model_config: ModelConfig,
        force_reload: bool = False
    ) -> LoadedModel:
        """
        加载模型
        
        Args:
            model_config: 模型配置
            force_reload: 是否强制重新加载
            
        Returns:
            LoadedModel: 已加载的模型
            
        Raises:
            ModelLoadError: 模型加载失败
        """
        model_key = model_config.model_name
        
        with self._lock:
            # 检查缓存
            if not force_reload and model_key in self._model_cache:
                cached = self._model_cache[model_key]
                cached.touch()
                # 移动到末尾（最近使用）
                self._model_cache.move_to_end(model_key)
                self._stats["cache_hits"] += 1
                logger.info(f"Cache hit for model: {model_key}")
                return cached.loaded_model
            
            self._stats["cache_misses"] += 1
            
            # 检查是否需要腾出空间
            self._ensure_cache_space()
            
            # 检查内存
            self._check_memory_before_load(model_config)
            
            # 加载模型
            try:
                loaded_model = self._model_loader.load_model(
                    model_config,
                    fallback_to_cpu=self.fallback_to_cpu,
                    fallback_to_quantization=True
                )
                
                # 检查是否发生了回退
                if loaded_model.device == "cpu" and model_config.device != "cpu":
                    self._stats["fallbacks"] += 1
                    logger.info(f"Model {model_key} loaded on CPU (fallback)")
                
                # 添加到缓存
                self._model_cache[model_key] = CachedModel(
                    loaded_model=loaded_model,
                    last_accessed=datetime.now()
                )
                self._stats["loads"] += 1
                
                logger.info(f"Model loaded and cached: {model_key}")
                return loaded_model
                
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                raise
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功卸载
        """
        with self._lock:
            if model_name not in self._model_cache:
                logger.warning(f"Model not in cache: {model_name}")
                return False
            
            cached = self._model_cache.pop(model_name)
            
            # 释放模型资源
            try:
                del cached.loaded_model.model
                del cached.loaded_model.tokenizer
                del cached.loaded_model
                del cached
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            self._stats["unloads"] += 1
            logger.info(f"Model unloaded: {model_name}")
            return True
    
    def get_model(self, model_name: str) -> Optional[LoadedModel]:
        """
        获取已加载的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            LoadedModel或None
        """
        with self._lock:
            if model_name in self._model_cache:
                cached = self._model_cache[model_name]
                cached.touch()
                self._model_cache.move_to_end(model_name)
                return cached.loaded_model
            return None
    
    def is_loaded(self, model_name: str) -> bool:
        """检查模型是否已加载"""
        with self._lock:
            return model_name in self._model_cache
    
    def list_loaded_models(self) -> List[str]:
        """列出所有已加载的模型"""
        with self._lock:
            return list(self._model_cache.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型信息字典或None
        """
        with self._lock:
            if model_name not in self._model_cache:
                return None
            
            cached = self._model_cache[model_name]
            return {
                "model_name": model_name,
                "device": cached.loaded_model.device,
                "dtype": str(cached.loaded_model.dtype),
                "memory_footprint_gb": cached.loaded_model.memory_footprint / 1024**3,
                "last_accessed": cached.last_accessed.isoformat(),
                "access_count": cached.access_count,
                "config": {
                    "model_type": cached.loaded_model.config.model_type,
                    "quantization": cached.loaded_model.config.quantization,
                    "flash_attention": cached.loaded_model.config.flash_attention,
                }
            }
    
    def _ensure_cache_space(self) -> None:
        """确保缓存有空间"""
        while len(self._model_cache) >= self.max_cached_models:
            self._evict_lru_model()
    
    def _evict_lru_model(self) -> bool:
        """
        驱逐最少使用的模型
        
        Returns:
            是否成功驱逐
        """
        with self._lock:
            if not self._model_cache:
                return False
            
            # 获取最早的模型（LRU）
            lru_key = next(iter(self._model_cache))
            logger.info(f"Evicting LRU model: {lru_key}")
            return self.unload_model(lru_key)
    
    def _check_memory_before_load(self, model_config: ModelConfig) -> None:
        """加载前检查内存"""
        snapshot = self._memory_monitor.take_snapshot()
        
        # 如果GPU内存使用率过高，尝试清理
        if snapshot.gpu_percent > self.memory_threshold * 100:
            logger.warning(
                f"GPU memory usage high ({snapshot.gpu_percent:.1f}%), "
                "attempting cleanup"
            )
            self._memory_monitor.clear_gpu_cache()
            
            # 如果仍然过高，驱逐模型
            snapshot = self._memory_monitor.take_snapshot()
            if snapshot.gpu_percent > self.memory_threshold * 100:
                self._evict_lru_model()
    
    def switch_device(
        self,
        model_name: str,
        target_device: str
    ) -> bool:
        """
        切换模型设备
        
        Args:
            model_name: 模型名称
            target_device: 目标设备
            
        Returns:
            是否成功切换
        """
        with self._lock:
            if model_name not in self._model_cache:
                logger.warning(f"Model not loaded: {model_name}")
                return False
            
            cached = self._model_cache[model_name]
            current_device = cached.loaded_model.device
            
            if current_device == target_device:
                logger.info(f"Model already on {target_device}")
                return True
            
            try:
                # 移动模型
                model = cached.loaded_model.model
                model.to(target_device)
                cached.loaded_model.device = target_device
                
                # 清理旧设备缓存
                if current_device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
                
                logger.info(
                    f"Model {model_name} moved from {current_device} to {target_device}"
                )
                return True
                
            except Exception as e:
                logger.error(f"Failed to switch device: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                **self._stats,
                "cached_models": len(self._model_cache),
                "max_cached_models": self.max_cached_models,
            }
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        return self._memory_monitor.get_summary()
    
    def clear_cache(self) -> int:
        """
        清空模型缓存
        
        Returns:
            卸载的模型数量
        """
        with self._lock:
            count = len(self._model_cache)
            model_names = list(self._model_cache.keys())
            
            for name in model_names:
                self.unload_model(name)
            
            logger.info(f"Cleared {count} models from cache")
            return count
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self.clear_cache()
        except Exception:
            pass
