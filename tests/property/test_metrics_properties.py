"""
性能监控属性测试

使用Hypothesis进行基于属性的测试，验证MetricsCollector的正确性属性。

属性33: 性能指标收集
对于任何工作流执行，系统应该收集并记录性能指标（如响应时间、模型推理时间），
且这些指标应该可以通过API查询。

验证需求: 12.4, 12.5
"""

import time
import pytest
from hypothesis import given, strategies as st, settings, assume

from mm_orch.metrics import (
    MetricsCollector,
    MetricType,
    MetricValue,
    TimerMetric,
    HistogramMetric,
    get_metrics_collector,
    reset_metrics_collector,
    create_metrics_collector
)


# ============ 测试策略定义 ============

# 指标名称策略
metric_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1,
    max_size=50
).filter(lambda x: x[0].isalpha() if x else False)

# 标签键策略
label_key_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1,
    max_size=20
).filter(lambda x: x[0].isalpha() if x else False)

# 标签值策略
label_value_strategy = st.text(min_size=1, max_size=50)

# 标签字典策略
labels_strategy = st.dictionaries(
    label_key_strategy,
    label_value_strategy,
    min_size=0,
    max_size=5
)

# 正数策略
positive_float_strategy = st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False)

# 非负数策略
non_negative_float_strategy = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)

# 工作流类型策略
workflow_type_strategy = st.sampled_from([
    "search_qa", "lesson_pack", "chat_generate", "rag_qa", "self_ask_search_qa"
])

# 模型名称策略
model_name_strategy = st.sampled_from([
    "qwen-chat", "gpt2", "t5-small", "minilm", "distilgpt2"
])


class TestMetricsCollectorProperties:
    """MetricsCollector属性测试类"""
    
    def setup_method(self):
        """每个测试前重置单例"""
        reset_metrics_collector()
    
    def teardown_method(self):
        """每个测试后清理"""
        reset_metrics_collector()
    
    # ============ 属性33: 性能指标收集 ============
    
    @given(
        workflow_type=workflow_type_strategy,
        duration=positive_float_strategy,
        success=st.booleans()
    )
    @settings(max_examples=100)
    def test_property33_workflow_metrics_recorded(
        self,
        workflow_type: str,
        duration: float,
        success: bool
    ):
        """
        Feature: muai-orchestration-system, Property 33: 性能指标收集
        
        对于任何工作流执行，系统应该收集并记录性能指标（如响应时间），
        且这些指标应该可以通过API查询。
        
        **Validates: Requirements 12.4, 12.5**
        """
        collector = create_metrics_collector()
        
        # 记录工作流执行
        collector.record_workflow_execution(workflow_type, duration, success)
        
        # 验证指标被记录
        labels = {"workflow": workflow_type}
        
        # 1. 执行计数应该增加
        execution_count = collector.get_counter("workflow_execution_total", labels)
        assert execution_count >= 1.0, "Workflow execution should be counted"
        
        # 2. 成功/失败计数应该正确
        if success:
            success_count = collector.get_counter("workflow_success_total", labels)
            assert success_count >= 1.0, "Success should be counted"
        else:
            failure_count = collector.get_counter("workflow_failure_total", labels)
            assert failure_count >= 1.0, "Failure should be counted"
        
        # 3. 执行时间应该被记录
        timer_stats = collector.get_timer_stats("workflow_execution_time", labels)
        assert timer_stats["count"] >= 1, "Execution time should be recorded"
        assert timer_stats["min"] > 0, "Execution time should be positive"
    
    @given(
        model_name=model_name_strategy,
        duration=positive_float_strategy,
        input_tokens=st.integers(min_value=1, max_value=10000),
        output_tokens=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=100)
    def test_property33_model_inference_metrics_recorded(
        self,
        model_name: str,
        duration: float,
        input_tokens: int,
        output_tokens: int
    ):
        """
        Feature: muai-orchestration-system, Property 33: 性能指标收集
        
        对于任何模型推理，系统应该收集并记录性能指标（如推理时间、token数），
        且这些指标应该可以通过API查询。
        
        **Validates: Requirements 12.4, 12.5**
        """
        collector = create_metrics_collector()
        
        # 记录模型推理
        collector.record_model_inference(model_name, duration, input_tokens, output_tokens)
        
        # 验证指标被记录
        labels = {"model": model_name}
        
        # 1. 推理计数应该增加
        inference_count = collector.get_counter("model_inference_total", labels)
        assert inference_count >= 1.0, "Model inference should be counted"
        
        # 2. Token计数应该正确
        input_token_count = collector.get_counter("model_input_tokens_total", labels)
        assert input_token_count >= input_tokens, "Input tokens should be counted"
        
        output_token_count = collector.get_counter("model_output_tokens_total", labels)
        assert output_token_count >= output_tokens, "Output tokens should be counted"
        
        # 3. 推理时间应该被记录
        timer_stats = collector.get_timer_stats("model_inference_time", labels)
        assert timer_stats["count"] >= 1, "Inference time should be recorded"
    
    @given(
        metric_name=metric_name_strategy,
        values=st.lists(positive_float_strategy, min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_property33_metrics_queryable_via_api(
        self,
        metric_name: str,
        values: list
    ):
        """
        Feature: muai-orchestration-system, Property 33: 性能指标收集
        
        对于任何记录的指标，这些指标应该可以通过API查询。
        
        **Validates: Requirements 12.4, 12.5**
        """
        collector = create_metrics_collector()
        
        # 记录多个时间值
        for value in values:
            collector.record_time(metric_name, value)
        
        # 验证可以通过API查询
        all_metrics = collector.get_all_metrics()
        
        # 1. 应该包含timers部分
        assert "timers" in all_metrics, "Metrics should include timers"
        
        # 2. 应该包含记录的指标
        assert metric_name in all_metrics["timers"], f"Timer {metric_name} should be queryable"
        
        # 3. 统计信息应该正确
        timer_data = all_metrics["timers"][metric_name]
        # 获取默认标签的统计
        stats = timer_data.get("", timer_data.get("default", {}))
        if stats:
            assert stats["count"] == len(values), "Count should match recorded values"
    
    # ============ 计数器属性测试 ============
    
    @given(
        metric_name=metric_name_strategy,
        increments=st.lists(non_negative_float_strategy, min_size=1, max_size=20)
    )
    @settings(max_examples=100)
    def test_counter_monotonically_increasing(
        self,
        metric_name: str,
        increments: list
    ):
        """
        计数器应该单调递增
        
        对于任何计数器，每次增加后的值应该大于或等于之前的值。
        """
        collector = create_metrics_collector()
        
        previous_value = 0.0
        for increment in increments:
            new_value = collector.increment(metric_name, increment)
            assert new_value >= previous_value, "Counter should be monotonically increasing"
            previous_value = new_value
        
        # 最终值应该等于所有增量的总和
        expected_total = sum(increments)
        actual_total = collector.get_counter(metric_name)
        assert abs(actual_total - expected_total) < 0.001, "Counter total should match sum of increments"
    
    @given(
        metric_name=metric_name_strategy,
        labels=labels_strategy,
        value=non_negative_float_strategy
    )
    @settings(max_examples=100)
    def test_counter_with_labels_isolation(
        self,
        metric_name: str,
        labels: dict,
        value: float
    ):
        """
        带标签的计数器应该相互隔离
        
        对于任何带标签的计数器，不同标签的计数器应该独立计数。
        """
        collector = create_metrics_collector()
        
        # 增加带标签的计数器
        collector.increment(metric_name, value, labels)
        
        # 验证带标签的计数器值
        labeled_value = collector.get_counter(metric_name, labels)
        assert abs(labeled_value - value) < 0.001, "Labeled counter should have correct value"
        
        # 验证无标签的计数器值（应该为0）
        unlabeled_value = collector.get_counter(metric_name)
        if labels:
            assert unlabeled_value == 0.0, "Unlabeled counter should be 0"
    
    # ============ 计时器属性测试 ============
    
    @given(
        metric_name=metric_name_strategy,
        durations=st.lists(positive_float_strategy, min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_timer_statistics_correctness(
        self,
        metric_name: str,
        durations: list
    ):
        """
        计时器统计应该正确
        
        对于任何计时器，统计信息（min, max, mean, count）应该正确计算。
        """
        collector = create_metrics_collector()
        
        for duration in durations:
            collector.record_time(metric_name, duration)
        
        stats = collector.get_timer_stats(metric_name)
        
        # 验证统计正确性
        assert stats["count"] == len(durations), "Count should match"
        assert abs(stats["min"] - min(durations)) < 0.001, "Min should be correct"
        assert abs(stats["max"] - max(durations)) < 0.001, "Max should be correct"
        
        expected_mean = sum(durations) / len(durations)
        assert abs(stats["mean"] - expected_mean) < 0.001, "Mean should be correct"
    
    @given(
        metric_name=metric_name_strategy,
        durations=st.lists(positive_float_strategy, min_size=10, max_size=100)
    )
    @settings(max_examples=50)
    def test_timer_percentiles_ordering(
        self,
        metric_name: str,
        durations: list
    ):
        """
        计时器百分位数应该有序
        
        对于任何计时器，p95 >= median >= min 且 p99 >= p95。
        """
        collector = create_metrics_collector()
        
        for duration in durations:
            collector.record_time(metric_name, duration)
        
        stats = collector.get_timer_stats(metric_name)
        
        # 验证百分位数顺序 (使用小容差处理浮点精度问题)
        epsilon = 1e-9
        assert stats["min"] <= stats["median"] + epsilon, "min <= median"
        assert stats["median"] <= stats["p95"] + epsilon, "median <= p95"
        assert stats["p95"] <= stats["p99"] + epsilon, "p95 <= p99"
        assert stats["p99"] <= stats["max"] + epsilon, "p99 <= max"
    
    # ============ 仪表盘属性测试 ============
    
    @given(
        metric_name=metric_name_strategy,
        values=st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=1, max_size=20)
    )
    @settings(max_examples=100)
    def test_gauge_last_value_wins(
        self,
        metric_name: str,
        values: list
    ):
        """
        仪表盘应该保留最后设置的值
        
        对于任何仪表盘，get_gauge应该返回最后一次set_gauge设置的值。
        """
        collector = create_metrics_collector()
        
        for value in values:
            collector.set_gauge(metric_name, value)
        
        # 最终值应该是最后设置的值
        final_value = collector.get_gauge(metric_name)
        assert abs(final_value - values[-1]) < 0.001, "Gauge should have last set value"
    
    # ============ 历史记录属性测试 ============
    
    @given(
        metric_name=metric_name_strategy,
        num_records=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=50)
    def test_recent_values_ordering(
        self,
        metric_name: str,
        num_records: int
    ):
        """
        最近的指标值应该按时间顺序排列
        
        对于任何指标，get_recent_values返回的值应该按时间戳升序排列。
        """
        collector = create_metrics_collector()
        
        for i in range(num_records):
            collector.increment(metric_name)
        
        recent = collector.get_recent_values(metric_name)
        
        # 验证时间戳顺序
        timestamps = [v.timestamp for v in recent]
        assert timestamps == sorted(timestamps), "Recent values should be ordered by timestamp"
    
    @given(
        max_history=st.integers(min_value=10, max_value=100),
        num_records=st.integers(min_value=1, max_value=200)
    )
    @settings(max_examples=50)
    def test_history_trimming_respects_limit(
        self,
        max_history: int,
        num_records: int
    ):
        """
        历史记录应该尊重最大限制
        
        对于任何max_history设置，历史记录数量不应超过该限制。
        """
        collector = create_metrics_collector(max_history=max_history)
        
        for i in range(num_records):
            collector.increment("test_counter")
        
        recent = collector.get_recent_values()
        
        assert len(recent) <= max_history, f"History should not exceed {max_history}"
    
    # ============ 摘要属性测试 ============
    
    @given(
        workflow_executions=st.lists(
            st.tuples(workflow_type_strategy, positive_float_strategy, st.booleans()),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=50)
    def test_summary_reflects_recorded_data(
        self,
        workflow_executions: list
    ):
        """
        摘要应该反映记录的数据
        
        对于任何记录的工作流执行，摘要中的统计应该正确反映这些数据。
        """
        collector = create_metrics_collector()
        
        # 记录工作流执行
        success_count = 0
        for workflow_type, duration, success in workflow_executions:
            collector.record_workflow_execution(workflow_type, duration, success)
            if success:
                success_count += 1
        
        summary = collector.get_summary()
        
        # 验证总请求数
        assert summary["total_requests"] == len(workflow_executions), "Total requests should match"
        
        # 验证成功率
        expected_success_rate = success_count / len(workflow_executions)
        assert abs(summary["success_rate"] - expected_success_rate) < 0.001, "Success rate should be correct"
    
    # ============ 线程安全属性测试 ============
    
    @given(
        num_threads=st.integers(min_value=2, max_value=10),
        increments_per_thread=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=20)
    def test_thread_safety_counter_consistency(
        self,
        num_threads: int,
        increments_per_thread: int
    ):
        """
        计数器应该在多线程环境下保持一致性
        
        对于任何并发增加操作，最终计数应该等于所有增量的总和。
        """
        import threading
        
        collector = create_metrics_collector()
        
        def increment_counter():
            for _ in range(increments_per_thread):
                collector.increment("concurrent_counter")
        
        threads = [
            threading.Thread(target=increment_counter)
            for _ in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        expected = num_threads * increments_per_thread
        actual = collector.get_counter("concurrent_counter")
        
        assert actual == expected, f"Expected {expected}, got {actual}"


class TestTimerMetricProperties:
    """TimerMetric属性测试类"""
    
    @given(
        values=st.lists(positive_float_strategy, min_size=0, max_size=100)
    )
    @settings(max_examples=100)
    def test_timer_metric_statistics_bounds(
        self,
        values: list
    ):
        """
        计时器统计应该在合理范围内
        
        对于任何值列表，统计信息应该满足：min <= mean <= max。
        """
        timer = TimerMetric(name="test")
        
        for value in values:
            timer.record(value)
        
        stats = timer.get_statistics()
        
        if values:
            assert stats["min"] <= stats["mean"] <= stats["max"], "min <= mean <= max"
            assert stats["total"] == pytest.approx(sum(values), rel=0.001), "Total should match sum"
        else:
            assert stats["count"] == 0, "Empty timer should have count 0"


class TestHistogramMetricProperties:
    """HistogramMetric属性测试类"""
    
    @given(
        buckets=st.lists(positive_float_strategy, min_size=1, max_size=10, unique=True),
        values=st.lists(positive_float_strategy, min_size=1, max_size=50)
    )
    @settings(max_examples=50)
    def test_histogram_bucket_coverage(
        self,
        buckets: list,
        values: list
    ):
        """
        直方图桶应该覆盖所有值
        
        对于任何值列表，所有桶的计数总和应该等于值的数量。
        """
        sorted_buckets = sorted(buckets)
        histogram = HistogramMetric(name="test", buckets=sorted_buckets)
        
        for value in values:
            histogram.record(value)
        
        dist = histogram.get_distribution()
        
        # 所有桶的计数总和应该等于值的数量
        total_in_buckets = sum(dist["buckets"].values())
        assert total_in_buckets == len(values), "All values should be in buckets"
        assert dist["count"] == len(values), "Count should match"
