"""
MetricsCollector单元测试

测试性能指标收集器的核心功能。
"""

import time
import pytest
import threading
from unittest.mock import patch

from mm_orch.metrics import (
    MetricsCollector,
    MetricType,
    MetricValue,
    TimerMetric,
    HistogramMetric,
    get_metrics_collector,
    reset_metrics_collector,
    create_metrics_collector,
    record_workflow_execution,
    record_model_inference,
    record_memory_usage,
    record_request,
    timer
)


class TestMetricsCollector:
    """MetricsCollector测试类"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        reset_metrics_collector()
    
    def teardown_method(self):
        """每个测试后清理"""
        reset_metrics_collector()
    
    # ============ 计数器测试 ============
    
    def test_counter_increment(self):
        """测试计数器增加"""
        collector = create_metrics_collector()
        
        result = collector.increment("test_counter")
        assert result == 1.0
        
        result = collector.increment("test_counter", 5.0)
        assert result == 6.0
    
    def test_counter_with_labels(self):
        """测试带标签的计数器"""
        collector = create_metrics_collector()
        
        collector.increment("requests", labels={"endpoint": "/api/query"})
        collector.increment("requests", labels={"endpoint": "/api/chat"})
        collector.increment("requests", labels={"endpoint": "/api/query"})
        
        assert collector.get_counter("requests", {"endpoint": "/api/query"}) == 2.0
        assert collector.get_counter("requests", {"endpoint": "/api/chat"}) == 1.0
    
    def test_counter_negative_value_raises(self):
        """测试计数器不允许负值"""
        collector = create_metrics_collector()
        
        with pytest.raises(ValueError, match="non-negative"):
            collector.increment("test_counter", -1.0)
    
    def test_get_counter_nonexistent(self):
        """测试获取不存在的计数器"""
        collector = create_metrics_collector()
        
        assert collector.get_counter("nonexistent") == 0.0
    
    # ============ 仪表盘测试 ============
    
    def test_gauge_set(self):
        """测试仪表盘设置"""
        collector = create_metrics_collector()
        
        collector.set_gauge("memory_usage", 1024.0)
        assert collector.get_gauge("memory_usage") == 1024.0
        
        collector.set_gauge("memory_usage", 2048.0)
        assert collector.get_gauge("memory_usage") == 2048.0
    
    def test_gauge_increment_decrement(self):
        """测试仪表盘增减"""
        collector = create_metrics_collector()
        
        collector.set_gauge("active_connections", 10.0)
        
        result = collector.increment_gauge("active_connections", 5.0)
        assert result == 15.0
        
        result = collector.decrement_gauge("active_connections", 3.0)
        assert result == 12.0
    
    def test_gauge_with_labels(self):
        """测试带标签的仪表盘"""
        collector = create_metrics_collector()
        
        collector.set_gauge("cpu_usage", 50.0, {"host": "server1"})
        collector.set_gauge("cpu_usage", 70.0, {"host": "server2"})
        
        assert collector.get_gauge("cpu_usage", {"host": "server1"}) == 50.0
        assert collector.get_gauge("cpu_usage", {"host": "server2"}) == 70.0
    
    # ============ 计时器测试 ============
    
    def test_timer_record(self):
        """测试计时器记录"""
        collector = create_metrics_collector()
        
        collector.record_time("response_time", 0.1)
        collector.record_time("response_time", 0.2)
        collector.record_time("response_time", 0.3)
        
        stats = collector.get_timer_stats("response_time")
        
        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert abs(stats["mean"] - 0.2) < 0.001
    
    def test_timer_context_manager(self):
        """测试计时器上下文管理器"""
        collector = create_metrics_collector()
        
        with collector.timer("operation_time"):
            time.sleep(0.05)
        
        stats = collector.get_timer_stats("operation_time")
        
        assert stats["count"] == 1
        assert stats["min"] >= 0.05
    
    def test_timer_with_labels(self):
        """测试带标签的计时器"""
        collector = create_metrics_collector()
        
        collector.record_time("workflow_time", 1.0, {"workflow": "search_qa"})
        collector.record_time("workflow_time", 2.0, {"workflow": "chat_generate"})
        
        search_stats = collector.get_timer_stats("workflow_time", {"workflow": "search_qa"})
        chat_stats = collector.get_timer_stats("workflow_time", {"workflow": "chat_generate"})
        
        assert search_stats["count"] == 1
        assert search_stats["mean"] == 1.0
        assert chat_stats["count"] == 1
        assert chat_stats["mean"] == 2.0
    
    def test_timer_statistics(self):
        """测试计时器统计计算"""
        collector = create_metrics_collector()
        
        # 记录100个值
        for i in range(100):
            collector.record_time("test_timer", float(i) / 100)
        
        stats = collector.get_timer_stats("test_timer")
        
        assert stats["count"] == 100
        assert stats["min"] == 0.0
        assert stats["max"] == 0.99
        assert 0.49 <= stats["mean"] <= 0.50
        assert stats["p95"] >= 0.94
        assert stats["p99"] >= 0.98
    
    # ============ 直方图测试 ============
    
    def test_histogram_record(self):
        """测试直方图记录"""
        collector = create_metrics_collector()
        
        collector.create_histogram("latency", [0.1, 0.5, 1.0, 5.0])
        
        collector.record_histogram("latency", 0.05)  # -> 0.1 bucket
        collector.record_histogram("latency", 0.3)   # -> 0.5 bucket
        collector.record_histogram("latency", 0.8)   # -> 1.0 bucket
        collector.record_histogram("latency", 2.0)   # -> 5.0 bucket
        collector.record_histogram("latency", 10.0)  # -> +Inf bucket
        
        dist = collector.get_histogram_distribution("latency")
        
        assert dist["count"] == 5
        # Each value falls into exactly one bucket (non-cumulative)
        assert dist["buckets"]["0.1"] == 1   # 0.05
        assert dist["buckets"]["0.5"] == 1   # 0.3
        assert dist["buckets"]["1.0"] == 1   # 0.8
        assert dist["buckets"]["5.0"] == 1   # 2.0
        assert dist["buckets"]["+Inf"] == 1  # 10.0
    
    def test_histogram_auto_create(self):
        """测试直方图自动创建"""
        collector = create_metrics_collector()
        
        # 不预先创建，直接记录
        collector.record_histogram("auto_histogram", 0.5)
        
        dist = collector.get_histogram_distribution("auto_histogram")
        assert dist["count"] == 1
    
    # ============ 预定义指标测试 ============
    
    def test_record_workflow_execution(self):
        """测试工作流执行记录"""
        collector = create_metrics_collector()
        
        collector.record_workflow_execution("search_qa", 1.5, True)
        collector.record_workflow_execution("search_qa", 2.0, False)
        
        # 检查计数器
        assert collector.get_counter(
            "workflow_execution_total",
            {"workflow": "search_qa"}
        ) == 2.0
        assert collector.get_counter(
            "workflow_success_total",
            {"workflow": "search_qa"}
        ) == 1.0
        assert collector.get_counter(
            "workflow_failure_total",
            {"workflow": "search_qa"}
        ) == 1.0
        
        # 检查计时器
        stats = collector.get_timer_stats(
            "workflow_execution_time",
            {"workflow": "search_qa"}
        )
        assert stats["count"] == 2
    
    def test_record_model_inference(self):
        """测试模型推理记录"""
        collector = create_metrics_collector()
        
        collector.record_model_inference(
            "qwen-chat",
            0.5,
            input_tokens=100,
            output_tokens=50
        )
        
        assert collector.get_counter(
            "model_inference_total",
            {"model": "qwen-chat"}
        ) == 1.0
        assert collector.get_counter(
            "model_input_tokens_total",
            {"model": "qwen-chat"}
        ) == 100.0
        assert collector.get_counter(
            "model_output_tokens_total",
            {"model": "qwen-chat"}
        ) == 50.0
    
    def test_record_memory_usage(self):
        """测试内存使用记录"""
        collector = create_metrics_collector()
        
        collector.record_memory_usage("model_cache", 1024 * 1024 * 100)
        
        assert collector.get_gauge(
            "memory_usage_bytes",
            {"component": "model_cache"}
        ) == 1024 * 1024 * 100
    
    def test_record_request(self):
        """测试API请求记录"""
        collector = create_metrics_collector()
        
        collector.record_request("/api/query", "POST", 200, 0.5)
        collector.record_request("/api/query", "POST", 500, 1.0)
        
        assert collector.get_counter(
            "http_requests_total",
            {"endpoint": "/api/query", "method": "POST", "status": "200"}
        ) == 1.0
        assert collector.get_counter(
            "http_requests_total",
            {"endpoint": "/api/query", "method": "POST", "status": "500"}
        ) == 1.0
    
    # ============ 查询接口测试 ============
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        collector = create_metrics_collector()
        
        collector.increment("counter1")
        collector.set_gauge("gauge1", 100.0)
        collector.record_time("timer1", 0.5)
        
        metrics = collector.get_all_metrics()
        
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "timers" in metrics
        assert "histograms" in metrics
        assert "uptime" in metrics
        
        assert "counter1" in metrics["counters"]
        assert "gauge1" in metrics["gauges"]
        assert "timer1" in metrics["timers"]
    
    def test_get_metric(self):
        """测试获取单个指标"""
        collector = create_metrics_collector()
        
        collector.increment("test_counter", 5.0)
        collector.set_gauge("test_gauge", 100.0)
        collector.record_time("test_timer", 0.5)
        
        assert collector.get_metric("test_counter") == 5.0
        assert collector.get_metric("test_gauge") == 100.0
        
        timer_stats = collector.get_metric("test_timer")
        assert timer_stats["count"] == 1
    
    def test_get_recent_values(self):
        """测试获取最近的指标值"""
        collector = create_metrics_collector()
        
        for i in range(10):
            collector.increment("test_counter")
        
        recent = collector.get_recent_values("test_counter", limit=5)
        
        assert len(recent) == 5
        assert all(v.name == "test_counter" for v in recent)
    
    def test_get_summary(self):
        """测试获取指标摘要"""
        collector = create_metrics_collector()
        
        collector.record_workflow_execution("search_qa", 1.0, True)
        collector.record_workflow_execution("search_qa", 2.0, True)
        collector.record_model_inference("qwen-chat", 0.5)
        
        summary = collector.get_summary()
        
        assert summary["total_requests"] == 2
        assert summary["success_rate"] == 1.0
        assert "search_qa" in summary["workflows"]
        assert "qwen-chat" in summary["models"]
    
    # ============ 重置测试 ============
    
    def test_reset(self):
        """测试重置指标"""
        collector = create_metrics_collector()
        
        collector.increment("counter1", 10.0)
        collector.set_gauge("gauge1", 100.0)
        
        collector.reset()
        
        assert collector.get_counter("counter1") == 0.0
        assert collector.get_gauge("gauge1") == 0.0
    
    # ============ 线程安全测试 ============
    
    def test_thread_safety(self):
        """测试线程安全"""
        collector = create_metrics_collector()
        num_threads = 10
        increments_per_thread = 100
        
        def increment_counter():
            for _ in range(increments_per_thread):
                collector.increment("thread_counter")
        
        threads = [
            threading.Thread(target=increment_counter)
            for _ in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        expected = num_threads * increments_per_thread
        assert collector.get_counter("thread_counter") == expected
    
    # ============ 历史裁剪测试 ============
    
    def test_history_trimming(self):
        """测试历史记录裁剪"""
        collector = create_metrics_collector(max_history=100)
        
        for i in range(200):
            collector.increment("test_counter")
        
        recent = collector.get_recent_values()
        assert len(recent) <= 100


class TestTimerMetric:
    """TimerMetric测试类"""
    
    def test_empty_statistics(self):
        """测试空计时器统计"""
        timer = TimerMetric(name="test")
        stats = timer.get_statistics()
        
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
    
    def test_single_value(self):
        """测试单值统计"""
        timer = TimerMetric(name="test")
        timer.record(1.0)
        
        stats = timer.get_statistics()
        
        assert stats["count"] == 1
        assert stats["min"] == 1.0
        assert stats["max"] == 1.0
        assert stats["mean"] == 1.0


class TestHistogramMetric:
    """HistogramMetric测试类"""
    
    def test_empty_distribution(self):
        """测试空直方图分布"""
        histogram = HistogramMetric(name="test", buckets=[1.0, 5.0, 10.0])
        dist = histogram.get_distribution()
        
        assert dist["count"] == 0
        assert dist["sum"] == 0.0


class TestGlobalFunctions:
    """全局函数测试类"""
    
    def setup_method(self):
        reset_metrics_collector()
    
    def teardown_method(self):
        reset_metrics_collector()
    
    def test_get_metrics_collector_singleton(self):
        """测试单例获取"""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_convenience_functions(self):
        """测试便捷函数"""
        record_workflow_execution("test_workflow", 1.0, True)
        record_model_inference("test_model", 0.5)
        record_memory_usage("test_component", 1024)
        record_request("/test", "GET", 200, 0.1)
        
        collector = get_metrics_collector()
        
        assert collector.get_counter(
            "workflow_execution_total",
            {"workflow": "test_workflow"}
        ) == 1.0
        assert collector.get_counter(
            "model_inference_total",
            {"model": "test_model"}
        ) == 1.0
    
    def test_timer_convenience_function(self):
        """测试计时器便捷函数"""
        with timer("test_operation"):
            time.sleep(0.01)
        
        collector = get_metrics_collector()
        stats = collector.get_timer_stats("test_operation")
        
        assert stats["count"] == 1
        assert stats["min"] >= 0.01
