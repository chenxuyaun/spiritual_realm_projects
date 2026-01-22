"""
延迟基准测试模块单元测试
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from mm_orch.benchmark.latency import LatencyBenchmark, LatencyResult


class TestLatencyResult:
    """LatencyResult测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = LatencyResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            ttft_mean=0.1,
            tokens_per_second_mean=50.0,
            e2e_latency_mean=1.0,
        )
        
        assert result.test_name == "test"
        assert result.model_name == "gpt2"
        assert result.ttft_mean == 0.1
        assert result.tokens_per_second_mean == 50.0
        assert result.e2e_latency_mean == 1.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = LatencyResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            ttft_mean=0.1,
            ttft_std=0.01,
            tokens_per_second_mean=50.0,
            e2e_latency_mean=1.0,
            warmup_runs=3,
            test_runs=10,
        )
        
        d = result.to_dict()
        
        assert d["test_name"] == "test"
        assert d["model_name"] == "gpt2"
        assert d["ttft"]["mean_ms"] == 100.0  # 0.1s = 100ms
        assert d["tokens_per_second"]["mean"] == 50.0
        assert d["config"]["warmup_runs"] == 3
        assert d["config"]["test_runs"] == 10
    
    def test_default_values(self):
        """测试默认值"""
        result = LatencyResult(
            test_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        assert result.ttft_mean == 0.0
        assert result.tokens_per_second_mean == 0.0
        assert result.raw_ttft == []
        assert result.raw_tps == []


class TestLatencyBenchmark:
    """LatencyBenchmark测试"""
    
    def test_init(self):
        """测试初始化"""
        benchmark = LatencyBenchmark(warmup_runs=5, test_runs=20)
        
        assert benchmark.warmup_runs == 5
        assert benchmark.test_runs == 20
    
    def test_init_defaults(self):
        """测试默认初始化"""
        benchmark = LatencyBenchmark()
        
        assert benchmark.warmup_runs == 3
        assert benchmark.test_runs == 10
    
    def test_measure_ttft_non_streaming(self):
        """测试非流式TTFT测量"""
        benchmark = LatencyBenchmark()
        
        def mock_generate(prompt):
            time.sleep(0.01)  # 10ms
            return "response"
        
        ttft = benchmark.measure_ttft(mock_generate, "test prompt")
        
        assert ttft >= 0.01
        assert ttft < 0.1  # 应该在合理范围内
    
    def test_measure_ttft_streaming(self):
        """测试流式TTFT测量"""
        benchmark = LatencyBenchmark()
        
        def mock_generate(prompt):
            return "response"
        
        def mock_stream(prompt):
            time.sleep(0.005)  # 5ms到第一个token
            yield "first"
            time.sleep(0.005)
            yield "second"
        
        ttft = benchmark.measure_ttft(mock_generate, "test", mock_stream)
        
        assert ttft >= 0.005
        assert ttft < 0.05
    
    def test_measure_tokens_per_second(self):
        """测试tokens/s测量"""
        benchmark = LatencyBenchmark()
        
        def mock_generate(prompt):
            time.sleep(0.1)  # 100ms
            return {"tokens": 50}
        
        def get_tokens(result):
            return result["tokens"]
        
        tps = benchmark.measure_tokens_per_second(
            mock_generate, "test", get_tokens
        )
        
        # 50 tokens / 0.1s = 500 tokens/s (大约)
        assert tps > 0
        assert tps < 1000  # 合理范围
    
    def test_measure_e2e_latency(self):
        """测试端到端延迟测量"""
        benchmark = LatencyBenchmark()
        
        def mock_generate(prompt):
            time.sleep(0.05)  # 50ms
            return "response"
        
        latency = benchmark.measure_e2e_latency(mock_generate, "test")
        
        assert latency >= 0.05
        assert latency < 0.1
    
    def test_run_latency_suite(self):
        """测试运行延迟测试套件"""
        benchmark = LatencyBenchmark(warmup_runs=1, test_runs=3)
        
        call_count = 0
        
        def mock_generate(prompt):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        result = benchmark.run_latency_suite(
            generate_fn=mock_generate,
            prompt="test prompt",
            model_name="test_model",
            get_output_tokens=get_tokens,
            test_name="test_suite"
        )
        
        assert result.test_name == "test_suite"
        assert result.model_name == "test_model"
        assert result.warmup_runs == 1
        assert result.test_runs == 3
        assert len(result.raw_ttft) == 3
        assert len(result.raw_tps) == 3
        assert len(result.raw_e2e) == 3
        assert result.ttft_mean > 0
        assert result.tokens_per_second_mean > 0
        # warmup(1) + ttft(3) + tps(3) + e2e(3) = 10 calls
        assert call_count >= 10
    
    def test_run_latency_suite_statistics(self):
        """测试延迟套件统计计算"""
        benchmark = LatencyBenchmark(warmup_runs=0, test_runs=5)
        
        def mock_generate(prompt):
            time.sleep(0.01)
            return {"tokens": 10}
        
        def get_tokens(result):
            return result["tokens"]
        
        result = benchmark.run_latency_suite(
            generate_fn=mock_generate,
            prompt="test",
            model_name="test",
            get_output_tokens=get_tokens,
        )
        
        # 检查统计数据
        assert result.ttft_min <= result.ttft_mean <= result.ttft_max
        assert result.ttft_p50 >= result.ttft_min
        assert result.ttft_p95 >= result.ttft_p50
        assert result.ttft_p99 >= result.ttft_p95
    
    def test_percentile_calculation(self):
        """测试百分位数计算"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        p50 = LatencyBenchmark._percentile(data, 50)
        p95 = LatencyBenchmark._percentile(data, 95)
        p99 = LatencyBenchmark._percentile(data, 99)
        
        assert 5.0 <= p50 <= 6.0
        assert p95 > p50
        assert p99 >= p95
    
    def test_percentile_empty_data(self):
        """测试空数据百分位数"""
        assert LatencyBenchmark._percentile([], 50) == 0.0
    
    def test_percentile_single_value(self):
        """测试单值百分位数"""
        assert LatencyBenchmark._percentile([5.0], 50) == 5.0
        assert LatencyBenchmark._percentile([5.0], 99) == 5.0
    
    def test_adjust_prompt_length_shorter(self):
        """测试缩短提示词"""
        prompt = "This is a test prompt"
        adjusted = LatencyBenchmark._adjust_prompt_length(prompt, 10)
        
        assert len(adjusted) == 10
        assert adjusted == "This is a "
    
    def test_adjust_prompt_length_longer(self):
        """测试延长提示词"""
        prompt = "Hi"
        adjusted = LatencyBenchmark._adjust_prompt_length(prompt, 10)
        
        assert len(adjusted) == 10
    
    def test_adjust_prompt_length_exact(self):
        """测试精确长度"""
        prompt = "12345"
        adjusted = LatencyBenchmark._adjust_prompt_length(prompt, 5)
        
        assert adjusted == prompt


class TestLatencyBenchmarkIntegration:
    """LatencyBenchmark集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        benchmark = LatencyBenchmark(warmup_runs=1, test_runs=2)
        
        def mock_generate(prompt):
            time.sleep(0.005)
            return {"text": "response", "tokens": 5}
        
        def mock_stream(prompt):
            time.sleep(0.002)
            yield "first"
            yield "second"
        
        def get_tokens(result):
            return result["tokens"]
        
        result = benchmark.run_latency_suite(
            generate_fn=mock_generate,
            prompt="test",
            model_name="test_model",
            get_output_tokens=get_tokens,
            stream_fn=mock_stream,
            test_name="integration_test"
        )
        
        assert result.test_name == "integration_test"
        assert result.ttft_mean > 0
        assert result.tokens_per_second_mean > 0
        assert result.e2e_latency_mean > 0
        
        # 验证可以转换为字典
        d = result.to_dict()
        assert "ttft" in d
        assert "tokens_per_second" in d
        assert "e2e_latency" in d
