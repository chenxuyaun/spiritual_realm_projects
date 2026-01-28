"""
吞吐量基准测试模块

提供单请求、并发和批处理吞吐量测量功能。
"""

import concurrent.futures
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ThroughputResult:
    """吞吐量测试结果"""

    test_name: str
    model_name: str
    timestamp: datetime

    # 单请求吞吐量
    single_requests_per_second: float = 0.0
    single_tokens_per_second: float = 0.0

    # 并发吞吐量
    concurrent_requests_per_second: float = 0.0
    concurrent_tokens_per_second: float = 0.0
    concurrent_level: int = 1

    # 批处理吞吐量
    batch_requests_per_second: float = 0.0
    batch_tokens_per_second: float = 0.0
    batch_size: int = 1

    # 延迟统计
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # 测试配置
    duration_seconds: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    failed_requests: int = 0

    # 原始测量数据
    raw_latencies: List[float] = field(default_factory=list)
    raw_tokens: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "test_name": self.test_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "single": {
                "requests_per_second": self.single_requests_per_second,
                "tokens_per_second": self.single_tokens_per_second,
            },
            "concurrent": {
                "requests_per_second": self.concurrent_requests_per_second,
                "tokens_per_second": self.concurrent_tokens_per_second,
                "level": self.concurrent_level,
            },
            "batch": {
                "requests_per_second": self.batch_requests_per_second,
                "tokens_per_second": self.batch_tokens_per_second,
                "size": self.batch_size,
            },
            "latency": {
                "mean_ms": self.latency_mean_ms,
                "p50_ms": self.latency_p50_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
            },
            "summary": {
                "duration_seconds": self.duration_seconds,
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "failed_requests": self.failed_requests,
            },
        }


class ThroughputBenchmark:
    """
    吞吐量基准测试

    测量模型的吞吐量：
    - 单请求吞吐量
    - 并发吞吐量
    - 批处理吞吐量
    """

    def __init__(self, duration_seconds: float = 60.0, warmup_requests: int = 3):
        """
        初始化吞吐量基准测试

        Args:
            duration_seconds: 测试持续时间（秒）
            warmup_requests: 预热请求数
        """
        self.duration_seconds = duration_seconds
        self.warmup_requests = warmup_requests

    def measure_single_throughput(
        self,
        generate_fn: Callable[[str], Any],
        prompt: str,
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        num_requests: int = 10,
    ) -> ThroughputResult:
        """
        测量单请求吞吐量

        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            num_requests: 请求数量

        Returns:
            ThroughputResult: 吞吐量测试结果
        """
        logger.info(f"Measuring single throughput for {model_name}")

        # 预热
        for _ in range(self.warmup_requests):
            generate_fn(prompt)

        latencies = []
        tokens = []

        start_time = time.perf_counter()

        for _ in range(num_requests):
            req_start = time.perf_counter()
            result = generate_fn(prompt)
            req_end = time.perf_counter()

            latencies.append(req_end - req_start)
            tokens.append(get_output_tokens(result))

        total_time = time.perf_counter() - start_time
        total_tokens = sum(tokens)

        result = ThroughputResult(
            test_name="single_throughput",
            model_name=model_name,
            timestamp=datetime.now(),
            single_requests_per_second=num_requests / total_time,
            single_tokens_per_second=total_tokens / total_time,
            duration_seconds=total_time,
            total_requests=num_requests,
            total_tokens=total_tokens,
            raw_latencies=latencies,
            raw_tokens=tokens,
        )

        # 计算延迟统计
        self._compute_latency_stats(result, latencies)

        logger.info(
            f"Single throughput: {result.single_requests_per_second:.2f} req/s, "
            f"{result.single_tokens_per_second:.2f} tokens/s"
        )

        return result

    def measure_concurrent_throughput(
        self,
        generate_fn: Callable[[str], Any],
        prompts: List[str],
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        num_concurrent: int = 4,
        max_requests: Optional[int] = None,
    ) -> ThroughputResult:
        """
        测量并发吞吐量

        Args:
            generate_fn: 生成函数
            prompts: 输入提示词列表
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            num_concurrent: 并发数
            max_requests: 最大请求数（可选）

        Returns:
            ThroughputResult: 吞吐量测试结果
        """
        logger.info(
            f"Measuring concurrent throughput for {model_name} " f"(concurrency={num_concurrent})"
        )

        # 预热
        for _ in range(self.warmup_requests):
            generate_fn(prompts[0])

        latencies = []
        tokens = []
        failed = 0
        lock = threading.Lock()

        def worker(prompt: str) -> None:
            nonlocal failed
            try:
                req_start = time.perf_counter()
                result = generate_fn(prompt)
                req_end = time.perf_counter()

                with lock:
                    latencies.append(req_end - req_start)
                    tokens.append(get_output_tokens(result))
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                with lock:
                    failed += 1

        # 准备请求
        if max_requests:
            request_prompts = (prompts * ((max_requests // len(prompts)) + 1))[:max_requests]
        else:
            request_prompts = prompts

        start_time = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(worker, p) for p in request_prompts]
            concurrent.futures.wait(futures)

        total_time = time.perf_counter() - start_time
        total_tokens = sum(tokens)
        num_requests = len(request_prompts)

        result = ThroughputResult(
            test_name="concurrent_throughput",
            model_name=model_name,
            timestamp=datetime.now(),
            concurrent_requests_per_second=num_requests / total_time,
            concurrent_tokens_per_second=total_tokens / total_time,
            concurrent_level=num_concurrent,
            duration_seconds=total_time,
            total_requests=num_requests,
            total_tokens=total_tokens,
            failed_requests=failed,
            raw_latencies=latencies,
            raw_tokens=tokens,
        )

        # 计算延迟统计
        self._compute_latency_stats(result, latencies)

        logger.info(
            f"Concurrent throughput: {result.concurrent_requests_per_second:.2f} req/s, "
            f"{result.concurrent_tokens_per_second:.2f} tokens/s"
        )

        return result

    def measure_batch_throughput(
        self,
        batch_generate_fn: Callable[[List[str]], List[Any]],
        prompts: List[str],
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        batch_size: int = 4,
    ) -> ThroughputResult:
        """
        测量批处理吞吐量

        Args:
            batch_generate_fn: 批量生成函数
            prompts: 输入提示词列表
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            batch_size: 批大小

        Returns:
            ThroughputResult: 吞吐量测试结果
        """
        logger.info(f"Measuring batch throughput for {model_name} " f"(batch_size={batch_size})")

        # 预热
        batch_generate_fn(prompts[: min(batch_size, len(prompts))])

        latencies = []
        tokens = []

        # 分批处理
        batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

        start_time = time.perf_counter()

        for batch in batches:
            batch_start = time.perf_counter()
            results = batch_generate_fn(batch)
            batch_end = time.perf_counter()

            batch_latency = batch_end - batch_start
            batch_tokens = sum(get_output_tokens(r) for r in results)

            # 每个请求的平均延迟
            for _ in batch:
                latencies.append(batch_latency / len(batch))
            tokens.append(batch_tokens)

        total_time = time.perf_counter() - start_time
        total_tokens = sum(tokens)
        num_requests = len(prompts)

        result = ThroughputResult(
            test_name="batch_throughput",
            model_name=model_name,
            timestamp=datetime.now(),
            batch_requests_per_second=num_requests / total_time,
            batch_tokens_per_second=total_tokens / total_time,
            batch_size=batch_size,
            duration_seconds=total_time,
            total_requests=num_requests,
            total_tokens=total_tokens,
            raw_latencies=latencies,
            raw_tokens=[sum(tokens)],
        )

        # 计算延迟统计
        self._compute_latency_stats(result, latencies)

        logger.info(
            f"Batch throughput: {result.batch_requests_per_second:.2f} req/s, "
            f"{result.batch_tokens_per_second:.2f} tokens/s"
        )

        return result

    def run_throughput_suite(
        self,
        generate_fn: Callable[[str], Any],
        batch_generate_fn: Callable[[List[str]], List[Any]],
        prompts: List[str],
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        concurrent_levels: List[int] = None,
        batch_sizes: List[int] = None,
    ) -> List[ThroughputResult]:
        """
        运行完整吞吐量测试套件

        Args:
            generate_fn: 生成函数
            batch_generate_fn: 批量生成函数
            prompts: 输入提示词列表
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            concurrent_levels: 并发级别列表
            batch_sizes: 批大小列表

        Returns:
            List[ThroughputResult]: 吞吐量测试结果列表
        """
        if concurrent_levels is None:
            concurrent_levels = [1, 2, 4, 8]
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]

        results = []

        # 单请求吞吐量
        single_result = self.measure_single_throughput(
            generate_fn, prompts[0], model_name, get_output_tokens
        )
        results.append(single_result)

        # 并发吞吐量
        for level in concurrent_levels:
            concurrent_result = self.measure_concurrent_throughput(
                generate_fn, prompts, model_name, get_output_tokens, num_concurrent=level
            )
            results.append(concurrent_result)

        # 批处理吞吐量
        for size in batch_sizes:
            batch_result = self.measure_batch_throughput(
                batch_generate_fn, prompts, model_name, get_output_tokens, batch_size=size
            )
            results.append(batch_result)

        return results

    def _compute_latency_stats(self, result: ThroughputResult, latencies: List[float]) -> None:
        """计算延迟统计"""
        if not latencies:
            return

        result.latency_mean_ms = statistics.mean(latencies) * 1000

        sorted_latencies = sorted(latencies)
        result.latency_p50_ms = self._percentile(sorted_latencies, 50) * 1000
        result.latency_p95_ms = self._percentile(sorted_latencies, 95) * 1000
        result.latency_p99_ms = self._percentile(sorted_latencies, 99) * 1000

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        if f == c:
            return sorted_data[f]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
