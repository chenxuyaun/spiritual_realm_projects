"""
延迟基准测试模块

提供TTFT、tokens/s和端到端延迟测量功能。
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """延迟测试结果"""
    test_name: str
    model_name: str
    timestamp: datetime
    
    # TTFT (Time To First Token)
    ttft_mean: float = 0.0  # seconds
    ttft_std: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    
    # Tokens per second
    tokens_per_second_mean: float = 0.0
    tokens_per_second_std: float = 0.0
    tokens_per_second_min: float = 0.0
    tokens_per_second_max: float = 0.0
    
    # End-to-end latency
    e2e_latency_mean: float = 0.0  # seconds
    e2e_latency_std: float = 0.0
    e2e_latency_min: float = 0.0
    e2e_latency_max: float = 0.0
    
    # Test configuration
    warmup_runs: int = 0
    test_runs: int = 0
    input_length: int = 0
    output_length: int = 0
    
    # Raw measurements
    raw_ttft: List[float] = field(default_factory=list)
    raw_tps: List[float] = field(default_factory=list)
    raw_e2e: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "test_name": self.test_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "ttft": {
                "mean_ms": self.ttft_mean * 1000,
                "std_ms": self.ttft_std * 1000,
                "min_ms": self.ttft_min * 1000,
                "max_ms": self.ttft_max * 1000,
                "p50_ms": self.ttft_p50 * 1000,
                "p95_ms": self.ttft_p95 * 1000,
                "p99_ms": self.ttft_p99 * 1000,
            },
            "tokens_per_second": {
                "mean": self.tokens_per_second_mean,
                "std": self.tokens_per_second_std,
                "min": self.tokens_per_second_min,
                "max": self.tokens_per_second_max,
            },
            "e2e_latency": {
                "mean_ms": self.e2e_latency_mean * 1000,
                "std_ms": self.e2e_latency_std * 1000,
                "min_ms": self.e2e_latency_min * 1000,
                "max_ms": self.e2e_latency_max * 1000,
            },
            "config": {
                "warmup_runs": self.warmup_runs,
                "test_runs": self.test_runs,
                "input_length": self.input_length,
                "output_length": self.output_length,
            },
        }


class LatencyBenchmark:
    """
    延迟基准测试
    
    测量模型推理的各种延迟指标：
    - TTFT (Time To First Token): 首token延迟
    - Tokens/s: 生成速度
    - End-to-end latency: 端到端延迟
    """
    
    def __init__(
        self,
        warmup_runs: int = 3,
        test_runs: int = 10
    ):
        """
        初始化延迟基准测试
        
        Args:
            warmup_runs: 预热运行次数
            test_runs: 测试运行次数
        """
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs
    
    def measure_ttft(
        self,
        generate_fn: Callable[[str], Any],
        prompt: str,
        stream_fn: Optional[Callable[[str], Any]] = None
    ) -> float:
        """
        测量首token延迟 (TTFT)
        
        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            stream_fn: 流式生成函数（可选，用于更精确测量）
            
        Returns:
            TTFT（秒）
        """
        if stream_fn is not None:
            # 使用流式生成测量真实TTFT
            start_time = time.perf_counter()
            stream = stream_fn(prompt)
            try:
                # 获取第一个token
                next(iter(stream))
                ttft = time.perf_counter() - start_time
                # 消费剩余的流
                for _ in stream:
                    pass
                return ttft
            except StopIteration:
                return time.perf_counter() - start_time
        else:
            # 使用非流式生成，TTFT近似为总时间
            start_time = time.perf_counter()
            generate_fn(prompt)
            return time.perf_counter() - start_time
    
    def measure_tokens_per_second(
        self,
        generate_fn: Callable[[str], Any],
        prompt: str,
        get_output_tokens: Callable[[Any], int]
    ) -> float:
        """
        测量生成速度 (tokens/s)
        
        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            get_output_tokens: 获取输出token数的函数
            
        Returns:
            tokens/s
        """
        start_time = time.perf_counter()
        result = generate_fn(prompt)
        elapsed = time.perf_counter() - start_time
        
        output_tokens = get_output_tokens(result)
        
        if elapsed > 0 and output_tokens > 0:
            return output_tokens / elapsed
        return 0.0
    
    def measure_e2e_latency(
        self,
        generate_fn: Callable[[str], Any],
        prompt: str
    ) -> float:
        """
        测量端到端延迟
        
        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            
        Returns:
            端到端延迟（秒）
        """
        start_time = time.perf_counter()
        generate_fn(prompt)
        return time.perf_counter() - start_time
    
    def run_latency_suite(
        self,
        generate_fn: Callable[[str], Any],
        prompt: str,
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        stream_fn: Optional[Callable[[str], Any]] = None,
        test_name: str = "latency_test"
    ) -> LatencyResult:
        """
        运行完整延迟测试套件
        
        Args:
            generate_fn: 生成函数
            prompt: 输入提示词
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            stream_fn: 流式生成函数（可选）
            test_name: 测试名称
            
        Returns:
            LatencyResult: 延迟测试结果
        """
        logger.info(f"Running latency suite: {test_name} for {model_name}")
        
        # 预热
        logger.info(f"Warming up ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            generate_fn(prompt)
        
        # 收集测量数据
        ttft_measurements = []
        tps_measurements = []
        e2e_measurements = []
        
        logger.info(f"Running tests ({self.test_runs} runs)...")
        for i in range(self.test_runs):
            # TTFT
            ttft = self.measure_ttft(generate_fn, prompt, stream_fn)
            ttft_measurements.append(ttft)
            
            # Tokens/s
            tps = self.measure_tokens_per_second(
                generate_fn, prompt, get_output_tokens
            )
            tps_measurements.append(tps)
            
            # E2E latency
            e2e = self.measure_e2e_latency(generate_fn, prompt)
            e2e_measurements.append(e2e)
            
            logger.debug(
                f"Run {i+1}: TTFT={ttft*1000:.2f}ms, "
                f"TPS={tps:.2f}, E2E={e2e*1000:.2f}ms"
            )
        
        # 计算统计数据
        result = LatencyResult(
            test_name=test_name,
            model_name=model_name,
            timestamp=datetime.now(),
            warmup_runs=self.warmup_runs,
            test_runs=self.test_runs,
            input_length=len(prompt),
            raw_ttft=ttft_measurements,
            raw_tps=tps_measurements,
            raw_e2e=e2e_measurements,
        )
        
        # TTFT统计
        if ttft_measurements:
            result.ttft_mean = statistics.mean(ttft_measurements)
            result.ttft_min = min(ttft_measurements)
            result.ttft_max = max(ttft_measurements)
            if len(ttft_measurements) > 1:
                result.ttft_std = statistics.stdev(ttft_measurements)
            sorted_ttft = sorted(ttft_measurements)
            result.ttft_p50 = self._percentile(sorted_ttft, 50)
            result.ttft_p95 = self._percentile(sorted_ttft, 95)
            result.ttft_p99 = self._percentile(sorted_ttft, 99)
        
        # TPS统计
        if tps_measurements:
            result.tokens_per_second_mean = statistics.mean(tps_measurements)
            result.tokens_per_second_min = min(tps_measurements)
            result.tokens_per_second_max = max(tps_measurements)
            if len(tps_measurements) > 1:
                result.tokens_per_second_std = statistics.stdev(tps_measurements)
        
        # E2E统计
        if e2e_measurements:
            result.e2e_latency_mean = statistics.mean(e2e_measurements)
            result.e2e_latency_min = min(e2e_measurements)
            result.e2e_latency_max = max(e2e_measurements)
            if len(e2e_measurements) > 1:
                result.e2e_latency_std = statistics.stdev(e2e_measurements)
        
        logger.info(
            f"Latency suite complete: TTFT={result.ttft_mean*1000:.2f}ms, "
            f"TPS={result.tokens_per_second_mean:.2f}, "
            f"E2E={result.e2e_latency_mean*1000:.2f}ms"
        )
        
        return result
    
    def run_multi_length_suite(
        self,
        generate_fn: Callable[[str, int], Any],
        base_prompt: str,
        model_name: str,
        get_output_tokens: Callable[[Any], int],
        input_lengths: List[int],
        output_lengths: List[int],
        stream_fn: Optional[Callable[[str, int], Any]] = None
    ) -> List[LatencyResult]:
        """
        运行多长度延迟测试套件
        
        Args:
            generate_fn: 生成函数 (prompt, max_tokens) -> result
            base_prompt: 基础提示词
            model_name: 模型名称
            get_output_tokens: 获取输出token数的函数
            input_lengths: 输入长度列表
            output_lengths: 输出长度列表
            stream_fn: 流式生成函数（可选）
            
        Returns:
            List[LatencyResult]: 延迟测试结果列表
        """
        results = []
        
        for input_len in input_lengths:
            for output_len in output_lengths:
                # 调整提示词长度
                prompt = self._adjust_prompt_length(base_prompt, input_len)
                
                # 创建带输出长度的生成函数
                def gen_fn(p):
                    return generate_fn(p, output_len)
                
                stream_fn_wrapped = None
                if stream_fn:
                    def stream_fn_wrapped(p):
                        return stream_fn(p, output_len)
                
                test_name = f"latency_in{input_len}_out{output_len}"
                
                result = self.run_latency_suite(
                    generate_fn=gen_fn,
                    prompt=prompt,
                    model_name=model_name,
                    get_output_tokens=get_output_tokens,
                    stream_fn=stream_fn_wrapped,
                    test_name=test_name
                )
                result.input_length = input_len
                result.output_length = output_len
                
                results.append(result)
        
        return results
    
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
    
    @staticmethod
    def _adjust_prompt_length(base_prompt: str, target_length: int) -> str:
        """调整提示词长度"""
        if len(base_prompt) >= target_length:
            return base_prompt[:target_length]
        
        # 重复提示词以达到目标长度
        repeat_count = (target_length // len(base_prompt)) + 1
        extended = (base_prompt + " ") * repeat_count
        return extended[:target_length]
