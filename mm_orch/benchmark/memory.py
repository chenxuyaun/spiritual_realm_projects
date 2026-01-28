"""
内存基准测试模块

提供模型加载内存和推理内存增长测量功能。
"""

import gc
import logging
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

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """内存测试结果"""

    test_name: str
    model_name: str
    timestamp: datetime

    # 模型加载内存
    model_load_gpu_mb: float = 0.0
    model_load_cpu_mb: float = 0.0

    # 推理内存增长
    inference_gpu_peak_mb: float = 0.0
    inference_gpu_delta_mb: float = 0.0
    inference_cpu_peak_mb: float = 0.0
    inference_cpu_delta_mb: float = 0.0

    # KV缓存内存（如果可测量）
    kv_cache_mb: float = 0.0

    # 量化对比
    quantization_type: Optional[str] = None
    memory_reduction_percent: float = 0.0

    # 测试配置
    input_length: int = 0
    output_length: int = 0
    gc_before_measure: bool = True

    # 原始测量数据
    raw_measurements: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "test_name": self.test_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "model_load": {
                "gpu_mb": self.model_load_gpu_mb,
                "cpu_mb": self.model_load_cpu_mb,
            },
            "inference": {
                "gpu_peak_mb": self.inference_gpu_peak_mb,
                "gpu_delta_mb": self.inference_gpu_delta_mb,
                "cpu_peak_mb": self.inference_cpu_peak_mb,
                "cpu_delta_mb": self.inference_cpu_delta_mb,
            },
            "kv_cache_mb": self.kv_cache_mb,
            "quantization": {
                "type": self.quantization_type,
                "memory_reduction_percent": self.memory_reduction_percent,
            },
            "config": {
                "input_length": self.input_length,
                "output_length": self.output_length,
                "gc_before_measure": self.gc_before_measure,
            },
        }


class MemoryBenchmark:
    """
    内存基准测试

    测量模型的内存使用：
    - 模型加载内存
    - 推理内存增长
    - KV缓存内存
    - 量化对比
    """

    def __init__(self, gc_before_measure: bool = True, track_allocations: bool = True):
        """
        初始化内存基准测试

        Args:
            gc_before_measure: 测量前是否执行GC
            track_allocations: 是否追踪内存分配
        """
        self.gc_before_measure = gc_before_measure
        self.track_allocations = track_allocations
        self._cuda_available = torch.cuda.is_available()

    def _clear_memory(self) -> None:
        """清理内存"""
        if self.gc_before_measure:
            gc.collect()
            if self._cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _get_gpu_memory(self) -> Dict[str, float]:
        """获取GPU内存使用（MB）"""
        if not self._cuda_available:
            return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}

        return {
            "allocated": torch.cuda.memory_allocated() / 1024**2,
            "reserved": torch.cuda.memory_reserved() / 1024**2,
            "peak": torch.cuda.max_memory_allocated() / 1024**2,
        }

    def _get_cpu_memory(self) -> Dict[str, float]:
        """获取CPU内存使用（MB）"""
        if not HAS_PSUTIL:
            return {"used": 0.0, "percent": 0.0}

        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "used": mem_info.rss / 1024**2,
            "percent": process.memory_percent(),
        }

    def _reset_peak_memory(self) -> None:
        """重置峰值内存统计"""
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats()

    def measure_model_load_memory(
        self, load_fn: Callable[[], Any], model_name: str
    ) -> MemoryResult:
        """
        测量模型加载内存

        Args:
            load_fn: 模型加载函数
            model_name: 模型名称

        Returns:
            MemoryResult: 内存测试结果
        """
        logger.info(f"Measuring model load memory for {model_name}")

        # 清理内存并记录基线
        self._clear_memory()
        self._reset_peak_memory()

        gpu_before = self._get_gpu_memory()
        cpu_before = self._get_cpu_memory()

        # 加载模型
        start_time = time.perf_counter()
        _ = load_fn()  # Model loaded but not used in this function
        load_time = time.perf_counter() - start_time

        # 同步GPU
        if self._cuda_available:
            torch.cuda.synchronize()

        gpu_after = self._get_gpu_memory()
        cpu_after = self._get_cpu_memory()

        result = MemoryResult(
            test_name="model_load",
            model_name=model_name,
            timestamp=datetime.now(),
            model_load_gpu_mb=gpu_after["allocated"] - gpu_before["allocated"],
            model_load_cpu_mb=cpu_after["used"] - cpu_before["used"],
            gc_before_measure=self.gc_before_measure,
        )

        logger.info(
            f"Model load memory: GPU={result.model_load_gpu_mb:.2f}MB, "
            f"CPU={result.model_load_cpu_mb:.2f}MB, "
            f"Time={load_time:.2f}s"
        )

        return result

    def measure_inference_memory(
        self, generate_fn: Callable[[str], Any], prompt: str, model_name: str, num_runs: int = 3
    ) -> MemoryResult:
        """
        测量推理内存增长

        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            model_name: 模型名称
            num_runs: 运行次数

        Returns:
            MemoryResult: 内存测试结果
        """
        logger.info(f"Measuring inference memory for {model_name}")

        gpu_deltas = []
        cpu_deltas = []
        gpu_peaks = []

        for i in range(num_runs):
            # 清理内存并记录基线
            self._clear_memory()
            self._reset_peak_memory()

            gpu_before = self._get_gpu_memory()
            cpu_before = self._get_cpu_memory()

            # 执行推理
            generate_fn(prompt)

            # 同步GPU
            if self._cuda_available:
                torch.cuda.synchronize()

            gpu_after = self._get_gpu_memory()
            cpu_after = self._get_cpu_memory()

            gpu_delta = gpu_after["allocated"] - gpu_before["allocated"]
            cpu_delta = cpu_after["used"] - cpu_before["used"]
            gpu_peak = gpu_after["peak"]

            gpu_deltas.append(gpu_delta)
            cpu_deltas.append(cpu_delta)
            gpu_peaks.append(gpu_peak)

            logger.debug(
                f"Run {i+1}: GPU delta={gpu_delta:.2f}MB, "
                f"CPU delta={cpu_delta:.2f}MB, GPU peak={gpu_peak:.2f}MB"
            )

        result = MemoryResult(
            test_name="inference_memory",
            model_name=model_name,
            timestamp=datetime.now(),
            inference_gpu_delta_mb=sum(gpu_deltas) / len(gpu_deltas),
            inference_cpu_delta_mb=sum(cpu_deltas) / len(cpu_deltas),
            inference_gpu_peak_mb=max(gpu_peaks) if gpu_peaks else 0.0,
            input_length=len(prompt),
            gc_before_measure=self.gc_before_measure,
            raw_measurements={
                "gpu_deltas": gpu_deltas,
                "cpu_deltas": cpu_deltas,
                "gpu_peaks": gpu_peaks,
            },
        )

        logger.info(
            f"Inference memory: GPU delta={result.inference_gpu_delta_mb:.2f}MB, "
            f"GPU peak={result.inference_gpu_peak_mb:.2f}MB"
        )

        return result

    def measure_kv_cache_memory(
        self,
        generate_fn: Callable[[str, int], Any],
        prompt: str,
        model_name: str,
        output_lengths: List[int],
    ) -> Dict[int, float]:
        """
        测量KV缓存内存增长

        Args:
            generate_fn: 生成函数 (prompt, max_tokens) -> result
            prompt: 输入提示词
            model_name: 模型名称
            output_lengths: 输出长度列表

        Returns:
            Dict[int, float]: 输出长度到KV缓存内存的映射
        """
        logger.info(f"Measuring KV cache memory for {model_name}")

        kv_cache_memory = {}

        for output_len in output_lengths:
            self._clear_memory()
            self._reset_peak_memory()

            gpu_before = self._get_gpu_memory()

            # 生成指定长度的输出
            generate_fn(prompt, output_len)

            if self._cuda_available:
                torch.cuda.synchronize()

            gpu_after = self._get_gpu_memory()

            # KV缓存内存近似为峰值内存减去基线
            kv_memory = gpu_after["peak"] - gpu_before["allocated"]
            kv_cache_memory[output_len] = kv_memory

            logger.debug(f"Output length {output_len}: KV cache ~{kv_memory:.2f}MB")

        return kv_cache_memory

    def compare_quantization_memory(
        self, load_fns: Dict[str, Callable[[], Any]], model_name: str
    ) -> Dict[str, MemoryResult]:
        """
        对比不同量化级别的内存使用

        Args:
            load_fns: 量化类型到加载函数的映射
            model_name: 模型名称

        Returns:
            Dict[str, MemoryResult]: 量化类型到内存结果的映射
        """
        logger.info(f"Comparing quantization memory for {model_name}")

        results = {}
        baseline_memory = None

        for quant_type, load_fn in load_fns.items():
            logger.info(f"Testing quantization: {quant_type}")

            result = self.measure_model_load_memory(load_fn, model_name)
            result.quantization_type = quant_type

            # 计算内存减少百分比
            if baseline_memory is None:
                baseline_memory = result.model_load_gpu_mb
            elif baseline_memory > 0:
                reduction = (baseline_memory - result.model_load_gpu_mb) / baseline_memory * 100
                result.memory_reduction_percent = reduction

            results[quant_type] = result

            # 卸载模型以测试下一个
            self._clear_memory()

        return results

    def run_memory_suite(
        self,
        load_fn: Callable[[], Any],
        generate_fn: Callable[[str], Any],
        prompt: str,
        model_name: str,
    ) -> MemoryResult:
        """
        运行完整内存测试套件

        Args:
            load_fn: 模型加载函数
            generate_fn: 生成函数
            prompt: 输入提示词
            model_name: 模型名称

        Returns:
            MemoryResult: 综合内存测试结果
        """
        logger.info(f"Running memory suite for {model_name}")

        # 测量模型加载内存
        load_result = self.measure_model_load_memory(load_fn, model_name)

        # 测量推理内存
        inference_result = self.measure_inference_memory(generate_fn, prompt, model_name)

        # 合并结果
        result = MemoryResult(
            test_name="memory_suite",
            model_name=model_name,
            timestamp=datetime.now(),
            model_load_gpu_mb=load_result.model_load_gpu_mb,
            model_load_cpu_mb=load_result.model_load_cpu_mb,
            inference_gpu_peak_mb=inference_result.inference_gpu_peak_mb,
            inference_gpu_delta_mb=inference_result.inference_gpu_delta_mb,
            inference_cpu_peak_mb=inference_result.inference_cpu_peak_mb,
            inference_cpu_delta_mb=inference_result.inference_cpu_delta_mb,
            input_length=len(prompt),
            gc_before_measure=self.gc_before_measure,
        )

        logger.info(
            f"Memory suite complete: Load GPU={result.model_load_gpu_mb:.2f}MB, "
            f"Inference peak={result.inference_gpu_peak_mb:.2f}MB"
        )

        return result
