"""
吞吐量基准测试模块单元测试
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from mm_orch.benchmark.throughput import ThroughputBenchmark, ThroughputResult


class TestThroughputResult:
    """ThroughputResult测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = ThroughputResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            single_requests_per_second=10.0,
            single_tokens_per_second=500.0,
        )
        
        assert result.test_name == "test"
        assert result.model_name == "gpt2"
        assert result.single_requests_per_second == 10.0
        assert result.single_tokens_per_second == 500.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = ThroughputResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            single_requests_per_second=10.0,
            concurrent_requests_per_second=30.0,
            concurrent_level=4,
            batch_requests_per_second=40.0,
            batch_size=8,
            latency_mean_ms=100.0,
            latency_p95_ms=150.0,
        )
        
        d = result.to_dict()
        
        assert d["test_name"] == "test"
        assert d["single"]["requests_per_second"] == 10.0
        assert d["concurrent"]["requests_per_second"] == 30.0
        assert d["concurrent"]["level"] == 4
        assert d["batch"]["requests_per_second"] == 40.0
        assert d["batch"]["size"] == 8
        assert d["latency"]["mean_ms"] == 100.0
        assert d["latency"]["p95_ms"] == 150.0
    
    def test_default_values(self):
        """测试默认值"""
        result = ThroughputResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        assert result.single_requests_per_second == 0.0
        assert result.concurrent_level == 1
        assert result.batch_size == 1
        assert result.raw_latencies == []
        assert result.raw_tokens == []


class TestThroughputBenchmark:
    """ThroughputBenchmark测试"""
    
    def test_init(self):
        """测试初始化"""
        benchmark = ThroughputBenchmark(
            duration_seconds=30.0,
            warmup_requests=5
        )
        
        assert benchmark.duration_seconds == 30.0
        assert benchmark.warmup_requests == 5
    
    def test_init_defaults(self):
        """测试默认初始化"""
        benchmark = ThroughputBenchmark()
        
        assert benchmark.duration_seconds == 60.0
        assert benchmark.warmup_requests == 3
    
    def test_measure_single_throughput(self):
        """测试单请求吞吐量测量"""
        benchmark = ThroughputBenchmark(warmup_requests=1)
        
        def mock_generate(prompt):
            time.sleep(0.01)  # 10ms
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        result = benchmark.measure_single_throughput(
            mock_generate, "test", "model", get_tokens, num_requests=5
        )
        
        assert result.test_name == "single_throughput"
        assert result.model_name == "model"
        assert result.total_requests == 5
        assert result.total_tokens == 50  # 5 * 10
        assert result.single_requests_per_second > 0
        assert result.single_tokens_per_second > 0
        assert len(result.raw_latencies) == 5
    
    def test_measure_single_throughput_latency_stats(self):
        """测试单请求吞吐量延迟统计"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        def mock_generate(prompt):
            time.sleep(0.01)
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        result = benchmark.measure_single_throughput(
            mock_generate, "test", "model", get_tokens, num_requests=10
        )
        
        assert result.latency_mean_ms > 0
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms
    
    def test_measure_concurrent_throughput(self):
        """测试并发吞吐量测量"""
        benchmark = ThroughputBenchmark(warmup_requests=1)
        
        def mock_generate(prompt):
            time.sleep(0.01)
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test1", "test2", "test3", "test4"]
        
        result = benchmark.measure_concurrent_throughput(
            mock_generate, prompts, "model", get_tokens,
            num_concurrent=2
        )
        
        assert result.test_name == "concurrent_throughput"
        assert result.concurrent_level == 2
        assert result.total_requests == 4
        assert result.concurrent_requests_per_second > 0
        assert result.concurrent_tokens_per_second > 0
    
    def test_measure_concurrent_throughput_with_max_requests(self):
        """测试带最大请求数的并发吞吐量"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        def mock_generate(prompt):
            return {"tokens": 5}
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test1", "test2"]
        
        result = benchmark.measure_concurrent_throughput(
            mock_generate, prompts, "model", get_tokens,
            num_concurrent=2, max_requests=10
        )
        
        assert result.total_requests == 10
    
    def test_measure_concurrent_throughput_with_failures(self):
        """测试带失败的并发吞吐量"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        call_count = 0
        
        def mock_generate(prompt):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                raise Exception("Test error")
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test"] * 6
        
        result = benchmark.measure_concurrent_throughput(
            mock_generate, prompts, "model", get_tokens,
            num_concurrent=2
        )
        
        assert result.failed_requests > 0
        assert result.total_requests == 6
    
    def test_measure_batch_throughput(self):
        """测试批处理吞吐量测量"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        def mock_batch_generate(prompts):
            time.sleep(0.01)
            return [{"tokens": 10} for _ in prompts]
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test1", "test2", "test3", "test4"]
        
        result = benchmark.measure_batch_throughput(
            mock_batch_generate, prompts, "model", get_tokens,
            batch_size=2
        )
        
        assert result.test_name == "batch_throughput"
        assert result.batch_size == 2
        assert result.total_requests == 4
        assert result.batch_requests_per_second > 0
        assert result.batch_tokens_per_second > 0
    
    def test_measure_batch_throughput_uneven_batches(self):
        """测试不均匀批次的批处理吞吐量"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        def mock_batch_generate(prompts):
            return [{"tokens": 10} for _ in prompts]
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test"] * 5  # 5个请求，批大小2，最后一批只有1个
        
        result = benchmark.measure_batch_throughput(
            mock_batch_generate, prompts, "model", get_tokens,
            batch_size=2
        )
        
        assert result.total_requests == 5
    
    def test_run_throughput_suite(self):
        """测试运行吞吐量测试套件"""
        benchmark = ThroughputBenchmark(warmup_requests=0)
        
        def mock_generate(prompt):
            return {"tokens": 10}
        
        def mock_batch_generate(prompts):
            return [{"tokens": 10} for _ in prompts]
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test1", "test2", "test3", "test4"]
        
        results = benchmark.run_throughput_suite(
            mock_generate, mock_batch_generate, prompts, "model",
            get_tokens,
            concurrent_levels=[1, 2],
            batch_sizes=[1, 2]
        )
        
        # 1 single + 2 concurrent + 2 batch = 5 results
        assert len(results) == 5
        
        # 检查结果类型
        test_names = [r.test_name for r in results]
        assert "single_throughput" in test_names
        assert "concurrent_throughput" in test_names
        assert "batch_throughput" in test_names
    
    def test_percentile_calculation(self):
        """测试百分位数计算"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        p50 = ThroughputBenchmark._percentile(data, 50)
        p95 = ThroughputBenchmark._percentile(data, 95)
        p99 = ThroughputBenchmark._percentile(data, 99)
        
        assert 5.0 <= p50 <= 6.0
        assert p95 > p50
        assert p99 >= p95
    
    def test_percentile_empty_data(self):
        """测试空数据百分位数"""
        assert ThroughputBenchmark._percentile([], 50) == 0.0
    
    def test_compute_latency_stats(self):
        """测试延迟统计计算"""
        benchmark = ThroughputBenchmark()
        result = ThroughputResult(
            test_name="test",
            model_name="model",
            timestamp=datetime.now(),
        )
        
        latencies = [0.1, 0.2, 0.15, 0.12, 0.18]
        benchmark._compute_latency_stats(result, latencies)
        
        assert result.latency_mean_ms > 0
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms > 0
        assert result.latency_p99_ms > 0
    
    def test_compute_latency_stats_empty(self):
        """测试空延迟统计"""
        benchmark = ThroughputBenchmark()
        result = ThroughputResult(
            test_name="test",
            model_name="model",
            timestamp=datetime.now(),
        )
        
        benchmark._compute_latency_stats(result, [])
        
        assert result.latency_mean_ms == 0.0


class TestThroughputBenchmarkIntegration:
    """ThroughputBenchmark集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        benchmark = ThroughputBenchmark(
            duration_seconds=10.0,
            warmup_requests=1
        )
        
        def mock_generate(prompt):
            time.sleep(0.005)
            return {"text": "response", "tokens": 5}
        
        def mock_batch_generate(prompts):
            time.sleep(0.01)
            return [{"text": "response", "tokens": 5} for _ in prompts]
        
        def get_tokens(result):
            return result["tokens"]
        
        prompts = ["test1", "test2", "test3", "test4"]
        
        results = benchmark.run_throughput_suite(
            mock_generate, mock_batch_generate, prompts, "test_model",
            get_tokens,
            concurrent_levels=[1, 2],
            batch_sizes=[2]
        )
        
        assert len(results) >= 3
        
        # 验证所有结果可以转换为字典
        for result in results:
            d = result.to_dict()
            assert "single" in d
            assert "concurrent" in d
            assert "batch" in d
            assert "latency" in d
