"""
内存监控器单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from mm_orch.runtime.memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    MemoryAlert,
)


class TestMemorySnapshot:
    """MemorySnapshot测试"""
    
    def test_create_snapshot(self):
        """测试创建快照"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=1024 * 1024 * 1024,  # 1GB
            gpu_reserved=2 * 1024 * 1024 * 1024,  # 2GB
            gpu_total=16 * 1024 * 1024 * 1024,  # 16GB
            cpu_used=8 * 1024 * 1024 * 1024,  # 8GB
            cpu_total=32 * 1024 * 1024 * 1024,  # 32GB
            cpu_percent=25.0
        )
        
        assert snapshot.gpu_allocated == 1024 * 1024 * 1024
        assert snapshot.gpu_total == 16 * 1024 * 1024 * 1024
    
    def test_gpu_free(self):
        """测试GPU可用内存计算"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=4 * 1024**3,
            gpu_total=16 * 1024**3
        )
        
        assert snapshot.gpu_free == 12 * 1024**3
    
    def test_gpu_percent(self):
        """测试GPU使用百分比计算"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=4 * 1024**3,
            gpu_total=16 * 1024**3
        )
        
        assert snapshot.gpu_percent == 25.0
    
    def test_gpu_percent_zero_total(self):
        """测试GPU总量为0时的百分比"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=0,
            gpu_total=0
        )
        
        assert snapshot.gpu_percent == 0.0
    
    def test_cpu_free(self):
        """测试CPU可用内存计算"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            cpu_used=8 * 1024**3,
            cpu_total=32 * 1024**3
        )
        
        assert snapshot.cpu_free == 24 * 1024**3
    
    def test_to_dict(self):
        """测试转换为字典"""
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            gpu_allocated=1024**3,
            gpu_total=16 * 1024**3,
            cpu_used=8 * 1024**3,
            cpu_total=32 * 1024**3,
            cpu_percent=25.0
        )
        
        result = snapshot.to_dict()
        
        assert "timestamp" in result
        assert "gpu" in result
        assert "cpu" in result
        assert result["gpu"]["allocated_gb"] == 1.0
        assert result["cpu"]["used_gb"] == 8.0


class TestMemoryAlert:
    """MemoryAlert测试"""
    
    def test_create_alert(self):
        """测试创建告警"""
        alert = MemoryAlert(
            timestamp=datetime.now(),
            alert_type="gpu_high",
            message="GPU memory high: 85% used",
            current_usage=85.0,
            threshold=80.0
        )
        
        assert alert.alert_type == "gpu_high"
        assert alert.current_usage == 85.0
        assert alert.threshold == 80.0


class TestMemoryMonitor:
    """MemoryMonitor测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        monitor = MemoryMonitor()
        
        assert monitor.gpu_warning_threshold == 0.8
        assert monitor.gpu_critical_threshold == 0.95
        assert monitor.cpu_warning_threshold == 0.8
        assert monitor.cpu_critical_threshold == 0.95
        assert monitor.history_size == 100
    
    def test_init_custom_thresholds(self):
        """测试自定义阈值初始化"""
        monitor = MemoryMonitor(
            gpu_warning_threshold=0.7,
            gpu_critical_threshold=0.9,
            cpu_warning_threshold=0.75,
            cpu_critical_threshold=0.92
        )
        
        assert monitor.gpu_warning_threshold == 0.7
        assert monitor.gpu_critical_threshold == 0.9
    
    @patch("torch.cuda.is_available", return_value=False)
    def test_get_gpu_memory_no_cuda(self, mock_available):
        """测试无CUDA时获取GPU内存"""
        monitor = MemoryMonitor()
        result = monitor.get_gpu_memory()
        
        assert result["allocated"] == 0
        assert result["reserved"] == 0
        assert result["total"] == 0
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024**3)
    @patch("torch.cuda.memory_reserved", return_value=2 * 1024**3)
    def test_get_gpu_memory_with_cuda(self, mock_reserved, mock_allocated, mock_available):
        """测试有CUDA时获取GPU内存"""
        mock_props = Mock()
        mock_props.total_memory = 16 * 1024**3
        
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            monitor = MemoryMonitor()
            result = monitor.get_gpu_memory()
            
            assert result["allocated"] == 1024**3
            assert result["reserved"] == 2 * 1024**3
            assert result["total"] == 16 * 1024**3
    
    @patch("mm_orch.runtime.memory_monitor.HAS_PSUTIL", False)
    def test_get_cpu_memory_no_psutil(self):
        """测试无psutil时获取CPU内存"""
        monitor = MemoryMonitor()
        result = monitor.get_cpu_memory()
        
        assert result["used"] == 0
        assert result["total"] == 0
        assert result["percent"] == 0.0
    
    @patch("mm_orch.runtime.memory_monitor.HAS_PSUTIL", True)
    def test_get_cpu_memory_with_psutil(self):
        """测试有psutil时获取CPU内存"""
        mock_mem = Mock()
        mock_mem.used = 8 * 1024**3
        mock_mem.total = 32 * 1024**3
        mock_mem.percent = 25.0
        
        with patch("mm_orch.runtime.memory_monitor.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            
            monitor = MemoryMonitor()
            result = monitor.get_cpu_memory()
            
            assert result["used"] == 8 * 1024**3
            assert result["total"] == 32 * 1024**3
            assert result["percent"] == 25.0
    
    def test_take_snapshot(self):
        """测试获取快照"""
        monitor = MemoryMonitor()
        
        with patch.object(monitor, "get_gpu_memory", return_value={
            "allocated": 1024**3,
            "reserved": 2 * 1024**3,
            "total": 16 * 1024**3
        }):
            with patch.object(monitor, "get_cpu_memory", return_value={
                "used": 8 * 1024**3,
                "total": 32 * 1024**3,
                "percent": 25.0
            }):
                snapshot = monitor.take_snapshot()
                
                assert snapshot.gpu_allocated == 1024**3
                assert snapshot.cpu_used == 8 * 1024**3
    
    def test_history_tracking(self):
        """测试历史记录追踪"""
        monitor = MemoryMonitor(history_size=5)
        
        with patch.object(monitor, "get_gpu_memory", return_value={
            "allocated": 0, "reserved": 0, "total": 0
        }):
            with patch.object(monitor, "get_cpu_memory", return_value={
                "used": 0, "total": 0, "percent": 0.0
            }):
                # 添加多个快照
                for _ in range(7):
                    monitor.take_snapshot()
                
                history = monitor.get_history()
                assert len(history) == 5  # 限制为history_size
    
    def test_alert_callback(self):
        """测试告警回调"""
        monitor = MemoryMonitor(gpu_warning_threshold=0.5)
        callback_called = []
        
        def callback(alert):
            callback_called.append(alert)
        
        monitor.register_alert_callback(callback)
        
        # 模拟高内存使用
        with patch.object(monitor, "get_gpu_memory", return_value={
            "allocated": 10 * 1024**3,
            "reserved": 12 * 1024**3,
            "total": 16 * 1024**3  # 62.5% usage
        }):
            with patch.object(monitor, "get_cpu_memory", return_value={
                "used": 0, "total": 32 * 1024**3, "percent": 0.0
            }):
                monitor.take_snapshot()
        
        assert len(callback_called) == 1
        assert callback_called[0].alert_type == "gpu_high"
    
    def test_check_memory_available(self):
        """测试检查内存可用性"""
        monitor = MemoryMonitor()
        
        with patch.object(monitor, "take_snapshot") as mock_snapshot:
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                gpu_allocated=4 * 1024**3,
                gpu_total=16 * 1024**3,
                cpu_used=8 * 1024**3,
                cpu_total=32 * 1024**3
            )
            mock_snapshot.return_value = snapshot
            
            # 有足够内存
            assert monitor.check_memory_available(
                required_gpu_bytes=4 * 1024**3
            ) is True
            
            # 内存不足
            assert monitor.check_memory_available(
                required_gpu_bytes=20 * 1024**3
            ) is False
    
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    @patch("torch.cuda.memory_allocated")
    def test_clear_gpu_cache(self, mock_allocated, mock_empty, mock_available):
        """测试清理GPU缓存"""
        mock_allocated.side_effect = [2 * 1024**3, 1 * 1024**3]  # before, after
        
        monitor = MemoryMonitor()
        freed = monitor.clear_gpu_cache()
        
        mock_empty.assert_called_once()
        assert freed == 1 * 1024**3
    
    def test_get_summary(self):
        """测试获取摘要"""
        monitor = MemoryMonitor()
        
        with patch.object(monitor, "take_snapshot") as mock_snapshot:
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                gpu_allocated=4 * 1024**3,
                gpu_total=16 * 1024**3,
                cpu_used=8 * 1024**3,
                cpu_total=32 * 1024**3,
                cpu_percent=25.0
            )
            mock_snapshot.return_value = snapshot
            
            summary = monitor.get_summary()
            
            assert "gpu" in summary
            assert "cpu" in summary
            assert summary["gpu"]["allocated_gb"] == 4.0
            assert summary["cpu"]["used_gb"] == 8.0
    
    def test_clear_alerts(self):
        """测试清除告警"""
        monitor = MemoryMonitor()
        monitor._alerts.append(MemoryAlert(
            timestamp=datetime.now(),
            alert_type="test",
            message="test",
            current_usage=0,
            threshold=0
        ))
        
        assert len(monitor.get_alerts()) == 1
        
        monitor.clear_alerts()
        
        assert len(monitor.get_alerts()) == 0
