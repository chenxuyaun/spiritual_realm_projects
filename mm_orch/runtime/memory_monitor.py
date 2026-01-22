"""
内存监控器模块

提供GPU和CPU内存监控功能，支持阈值告警和内存使用追踪。
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from mm_orch.exceptions import OutOfMemoryError

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: datetime
    gpu_allocated: int = 0  # bytes
    gpu_reserved: int = 0  # bytes
    gpu_total: int = 0  # bytes
    cpu_used: int = 0  # bytes
    cpu_total: int = 0  # bytes
    cpu_percent: float = 0.0
    
    @property
    def gpu_free(self) -> int:
        """GPU可用内存"""
        return self.gpu_total - self.gpu_allocated
    
    @property
    def gpu_percent(self) -> float:
        """GPU内存使用百分比"""
        if self.gpu_total == 0:
            return 0.0
        return (self.gpu_allocated / self.gpu_total) * 100
    
    @property
    def cpu_free(self) -> int:
        """CPU可用内存"""
        return self.cpu_total - self.cpu_used
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu": {
                "allocated_gb": self.gpu_allocated / 1024**3,
                "reserved_gb": self.gpu_reserved / 1024**3,
                "total_gb": self.gpu_total / 1024**3,
                "free_gb": self.gpu_free / 1024**3,
                "percent": self.gpu_percent,
            },
            "cpu": {
                "used_gb": self.cpu_used / 1024**3,
                "total_gb": self.cpu_total / 1024**3,
                "free_gb": self.cpu_free / 1024**3,
                "percent": self.cpu_percent,
            }
        }


@dataclass
class MemoryAlert:
    """内存告警数据类"""
    timestamp: datetime
    alert_type: str  # "gpu_high", "cpu_high", "gpu_critical", "cpu_critical"
    message: str
    current_usage: float  # percentage
    threshold: float  # percentage


class MemoryMonitor:
    """
    内存监控器
    
    提供GPU和CPU内存监控功能：
    - 实时内存使用查询
    - 内存使用历史追踪
    - 阈值告警
    - 内存清理建议
    """
    
    def __init__(
        self,
        gpu_warning_threshold: float = 0.8,
        gpu_critical_threshold: float = 0.95,
        cpu_warning_threshold: float = 0.8,
        cpu_critical_threshold: float = 0.95,
        history_size: int = 100
    ):
        """
        初始化内存监控器
        
        Args:
            gpu_warning_threshold: GPU内存警告阈值（0-1）
            gpu_critical_threshold: GPU内存严重阈值（0-1）
            cpu_warning_threshold: CPU内存警告阈值（0-1）
            cpu_critical_threshold: CPU内存严重阈值（0-1）
            history_size: 历史记录大小
        """
        self.gpu_warning_threshold = gpu_warning_threshold
        self.gpu_critical_threshold = gpu_critical_threshold
        self.cpu_warning_threshold = cpu_warning_threshold
        self.cpu_critical_threshold = cpu_critical_threshold
        self.history_size = history_size
        
        self._history: List[MemorySnapshot] = []
        self._alerts: List[MemoryAlert] = []
        self._alert_callbacks: List[Callable[[MemoryAlert], None]] = []
        self._lock = threading.Lock()
        
        # 检查CUDA可用性
        self._cuda_available = torch.cuda.is_available()
        if not self._cuda_available:
            logger.info("CUDA not available, GPU monitoring disabled")
        
        # 检查psutil可用性
        if not HAS_PSUTIL:
            logger.warning("psutil not installed, CPU monitoring limited")
    
    def get_gpu_memory(self) -> Dict[str, int]:
        """
        获取GPU内存使用情况
        
        Returns:
            包含allocated, reserved, total的字典（字节）
        """
        if not self._cuda_available:
            return {"allocated": 0, "reserved": 0, "total": 0}
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                "allocated": allocated,
                "reserved": reserved,
                "total": total
            }
        except Exception as e:
            logger.error(f"Failed to get GPU memory: {e}")
            return {"allocated": 0, "reserved": 0, "total": 0}
    
    def get_cpu_memory(self) -> Dict[str, Any]:
        """
        获取CPU内存使用情况
        
        Returns:
            包含used, total, percent的字典
        """
        if not HAS_PSUTIL:
            return {"used": 0, "total": 0, "percent": 0.0}
        
        try:
            mem = psutil.virtual_memory()
            return {
                "used": mem.used,
                "total": mem.total,
                "percent": mem.percent
            }
        except Exception as e:
            logger.error(f"Failed to get CPU memory: {e}")
            return {"used": 0, "total": 0, "percent": 0.0}
    
    def take_snapshot(self) -> MemorySnapshot:
        """
        获取当前内存快照
        
        Returns:
            MemorySnapshot对象
        """
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=gpu_mem["allocated"],
            gpu_reserved=gpu_mem["reserved"],
            gpu_total=gpu_mem["total"],
            cpu_used=cpu_mem["used"],
            cpu_total=cpu_mem["total"],
            cpu_percent=cpu_mem["percent"]
        )
        
        # 添加到历史记录
        with self._lock:
            self._history.append(snapshot)
            if len(self._history) > self.history_size:
                self._history.pop(0)
        
        # 检查阈值
        self._check_thresholds(snapshot)
        
        return snapshot
    
    def _check_thresholds(self, snapshot: MemorySnapshot) -> None:
        """检查内存阈值并触发告警"""
        alerts = []
        
        # 检查GPU内存
        if snapshot.gpu_total > 0:
            gpu_usage = snapshot.gpu_allocated / snapshot.gpu_total
            
            if gpu_usage >= self.gpu_critical_threshold:
                alerts.append(MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="gpu_critical",
                    message=f"GPU memory critical: {gpu_usage*100:.1f}% used",
                    current_usage=gpu_usage * 100,
                    threshold=self.gpu_critical_threshold * 100
                ))
            elif gpu_usage >= self.gpu_warning_threshold:
                alerts.append(MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="gpu_high",
                    message=f"GPU memory high: {gpu_usage*100:.1f}% used",
                    current_usage=gpu_usage * 100,
                    threshold=self.gpu_warning_threshold * 100
                ))
        
        # 检查CPU内存
        if snapshot.cpu_total > 0:
            cpu_usage = snapshot.cpu_used / snapshot.cpu_total
            
            if cpu_usage >= self.cpu_critical_threshold:
                alerts.append(MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="cpu_critical",
                    message=f"CPU memory critical: {cpu_usage*100:.1f}% used",
                    current_usage=cpu_usage * 100,
                    threshold=self.cpu_critical_threshold * 100
                ))
            elif cpu_usage >= self.cpu_warning_threshold:
                alerts.append(MemoryAlert(
                    timestamp=snapshot.timestamp,
                    alert_type="cpu_high",
                    message=f"CPU memory high: {cpu_usage*100:.1f}% used",
                    current_usage=cpu_usage * 100,
                    threshold=self.cpu_warning_threshold * 100
                ))
        
        # 记录告警并触发回调
        for alert in alerts:
            logger.warning(alert.message)
            with self._lock:
                self._alerts.append(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(
        self,
        callback: Callable[[MemoryAlert], None]
    ) -> None:
        """
        注册告警回调函数
        
        Args:
            callback: 告警回调函数
        """
        self._alert_callbacks.append(callback)
    
    def get_history(self) -> List[MemorySnapshot]:
        """获取内存历史记录"""
        with self._lock:
            return list(self._history)
    
    def get_alerts(self) -> List[MemoryAlert]:
        """获取告警记录"""
        with self._lock:
            return list(self._alerts)
    
    def clear_alerts(self) -> None:
        """清除告警记录"""
        with self._lock:
            self._alerts.clear()
    
    def check_memory_available(
        self,
        required_gpu_bytes: int = 0,
        required_cpu_bytes: int = 0
    ) -> bool:
        """
        检查是否有足够的内存
        
        Args:
            required_gpu_bytes: 需要的GPU内存（字节）
            required_cpu_bytes: 需要的CPU内存（字节）
            
        Returns:
            是否有足够内存
        """
        snapshot = self.take_snapshot()
        
        if required_gpu_bytes > 0:
            if snapshot.gpu_free < required_gpu_bytes:
                return False
        
        if required_cpu_bytes > 0:
            if snapshot.cpu_free < required_cpu_bytes:
                return False
        
        return True
    
    def clear_gpu_cache(self) -> int:
        """
        清理GPU缓存
        
        Returns:
            释放的内存量（字节）
        """
        if not self._cuda_available:
            return 0
        
        before = torch.cuda.memory_allocated()
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        after = torch.cuda.memory_allocated()
        freed = before - after
        
        if freed > 0:
            logger.info(f"Freed {freed / 1024**2:.2f} MB GPU memory")
        
        return freed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取内存使用摘要
        
        Returns:
            内存使用摘要字典
        """
        snapshot = self.take_snapshot()
        return {
            "gpu": {
                "available": self._cuda_available,
                "allocated_gb": snapshot.gpu_allocated / 1024**3,
                "total_gb": snapshot.gpu_total / 1024**3,
                "free_gb": snapshot.gpu_free / 1024**3,
                "usage_percent": snapshot.gpu_percent,
            },
            "cpu": {
                "used_gb": snapshot.cpu_used / 1024**3,
                "total_gb": snapshot.cpu_total / 1024**3,
                "free_gb": snapshot.cpu_free / 1024**3,
                "usage_percent": snapshot.cpu_percent,
            },
            "alerts_count": len(self._alerts),
            "history_size": len(self._history),
        }
