"""
内存基准测试模块单元测试
"""

import gc
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import torch

from mm_orch.benchmark.memory import MemoryBenchmark, MemoryResult


class TestMemoryResult:
    """MemoryResult测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = MemoryResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            model_load_gpu_mb=1000.0,
            model_load_cpu_mb=500.0,
        )
        
        assert result.test_name == "test"
        assert result.model_name == "gpt2"
        assert result.model_load_gpu_mb == 1000.0
        assert result.model_load_cpu_mb == 500.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = MemoryResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            model_load_gpu_mb=1000.0,
            inference_gpu_peak_mb=1500.0,
            quantization_type="8bit",
            memory_reduction_percent=50.0,
        )
        
        d = result.to_dict()
        
        assert d["test_name"] == "test"
        assert d["model_load"]["gpu_mb"] == 1000.0
        assert d["inference"]["gpu_peak_mb"] == 1500.0
        assert d["quantization"]["type"] == "8bit"
        assert d["quantization"]["memory_reduction_percent"] == 50.0
    
    def test_default_values(self):
        """测试默认值"""
        result = MemoryResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        assert result.model_load_gpu_mb == 0.0
        assert result.inference_gpu_peak_mb == 0.0
        assert result.quantization_type is None
        assert result.raw_measurements == {}


class TestMemoryBenchmark:
    """MemoryBenchmark测试"""
    
    def test_init(self):
        """测试初始化"""
        benchmark = MemoryBenchmark(
            gc_before_measure=True,
            track_allocations=True
        )
        
        assert benchmark.gc_before_measure is True
        assert benchmark.track_allocations is True
    
    def test_init_defaults(self):
        """测试默认初始化"""
        benchmark = MemoryBenchmark()
        
        assert benchmark.gc_before_measure is True
        assert benchmark.track_allocations is True
    
    def test_clear_memory(self):
        """测试清理内存"""
        benchmark = MemoryBenchmark(gc_before_measure=True)
        
        # 应该不抛出异常
        benchmark._clear_memory()
    
    def test_clear_memory_disabled(self):
        """测试禁用GC"""
        benchmark = MemoryBenchmark(gc_before_measure=False)
        
        # 应该不执行GC
        benchmark._clear_memory()
    
    def test_get_gpu_memory_no_cuda(self):
        """测试无CUDA时获取GPU内存"""
        benchmark = MemoryBenchmark()
        benchmark._cuda_available = False
        
        mem = benchmark._get_gpu_memory()
        
        assert mem["allocated"] == 0.0
        assert mem["reserved"] == 0.0
        assert mem.get("peak", 0.0) == 0.0
    
    @patch("mm_orch.benchmark.memory.HAS_PSUTIL", False)
    def test_get_cpu_memory_no_psutil(self):
        """测试无psutil时获取CPU内存"""
        benchmark = MemoryBenchmark()
        
        mem = benchmark._get_cpu_memory()
        
        assert mem["used"] == 0.0
        assert mem["percent"] == 0.0
    
    def test_measure_model_load_memory(self):
        """测试模型加载内存测量"""
        benchmark = MemoryBenchmark()
        
        mock_model = MagicMock()
        
        def mock_load():
            return mock_model
        
        result = benchmark.measure_model_load_memory(mock_load, "test_model")
        
        assert result.test_name == "model_load"
        assert result.model_name == "test_model"
        assert isinstance(result.timestamp, datetime)
        # 内存值可能为0（取决于环境）
        assert result.model_load_gpu_mb >= 0
        assert result.model_load_cpu_mb >= -1000  # CPU内存可能有波动
    
    def test_measure_inference_memory(self):
        """测试推理内存测量"""
        benchmark = MemoryBenchmark()
        
        def mock_generate(prompt):
            return "response"
        
        result = benchmark.measure_inference_memory(
            mock_generate, "test prompt", "test_model", num_runs=2
        )
        
        assert result.test_name == "inference_memory"
        assert result.model_name == "test_model"
        assert result.input_length == len("test prompt")
        assert "gpu_deltas" in result.raw_measurements
        assert len(result.raw_measurements["gpu_deltas"]) == 2
    
    def test_measure_inference_memory_single_run(self):
        """测试单次推理内存测量"""
        benchmark = MemoryBenchmark()
        
        def mock_generate(prompt):
            return "response"
        
        result = benchmark.measure_inference_memory(
            mock_generate, "test", "model", num_runs=1
        )
        
        assert len(result.raw_measurements["gpu_deltas"]) == 1
    
    def test_measure_kv_cache_memory(self):
        """测试KV缓存内存测量"""
        benchmark = MemoryBenchmark()
        
        def mock_generate(prompt, max_tokens):
            return "response" * max_tokens
        
        kv_memory = benchmark.measure_kv_cache_memory(
            mock_generate, "test", "model", [10, 50, 100]
        )
        
        assert 10 in kv_memory
        assert 50 in kv_memory
        assert 100 in kv_memory
    
    def test_compare_quantization_memory(self):
        """测试量化内存对比"""
        benchmark = MemoryBenchmark()
        
        def mock_load_fp32():
            return MagicMock()
        
        def mock_load_8bit():
            return MagicMock()
        
        load_fns = {
            "fp32": mock_load_fp32,
            "8bit": mock_load_8bit,
        }
        
        results = benchmark.compare_quantization_memory(load_fns, "test_model")
        
        assert "fp32" in results
        assert "8bit" in results
        assert results["fp32"].quantization_type == "fp32"
        assert results["8bit"].quantization_type == "8bit"
    
    def test_run_memory_suite(self):
        """测试运行内存测试套件"""
        benchmark = MemoryBenchmark()
        
        mock_model = MagicMock()
        
        def mock_load():
            return mock_model
        
        def mock_generate(prompt):
            return "response"
        
        result = benchmark.run_memory_suite(
            mock_load, mock_generate, "test prompt", "test_model"
        )
        
        assert result.test_name == "memory_suite"
        assert result.model_name == "test_model"
        assert result.input_length == len("test prompt")


class TestMemoryBenchmarkWithCuda:
    """带CUDA的MemoryBenchmark测试"""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_get_gpu_memory_with_cuda(self):
        """测试有CUDA时获取GPU内存"""
        benchmark = MemoryBenchmark()
        
        mem = benchmark._get_gpu_memory()
        
        assert "allocated" in mem
        assert "reserved" in mem
        assert "total" in mem
        assert mem["total"] > 0
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_reset_peak_memory(self):
        """测试重置峰值内存"""
        benchmark = MemoryBenchmark()
        
        # 应该不抛出异常
        benchmark._reset_peak_memory()


class TestMemoryBenchmarkIntegration:
    """MemoryBenchmark集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        benchmark = MemoryBenchmark(gc_before_measure=True)
        
        # 模拟模型加载
        model_data = []
        
        def mock_load():
            # 分配一些内存
            model_data.append([0] * 1000)
            return model_data
        
        def mock_generate(prompt):
            return "response"
        
        result = benchmark.run_memory_suite(
            mock_load, mock_generate, "test", "test_model"
        )
        
        assert result.test_name == "memory_suite"
        
        # 验证可以转换为字典
        d = result.to_dict()
        assert "model_load" in d
        assert "inference" in d
        assert "quantization" in d
