"""
性能监控系统 - MetricsCollector

收集和管理系统性能指标，包括响应时间、模型推理时间、内存使用等。

需求: 12.4, 12.5
属性33: 性能指标收集
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from contextlib import contextmanager
import statistics


class MetricType(Enum):
    """指标类型枚举"""

    COUNTER = "counter"  # 计数器（只增不减）
    GAUGE = "gauge"  # 仪表盘（可增可减）
    HISTOGRAM = "histogram"  # 直方图（分布统计）
    TIMER = "timer"  # 计时器（时间测量）


@dataclass
class MetricValue:
    """单个指标值"""

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class TimerMetric:
    """计时器指标"""

    name: str
    values: List[float] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def record(self, duration: float) -> None:
        """记录一个时间值"""
        self.values.append(duration)

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "total": 0.0,
            }

        sorted_values = sorted(self.values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": min(sorted_values),
            "max": max(sorted_values),
            "mean": statistics.mean(sorted_values),
            "median": statistics.median(sorted_values),
            "p95": self._percentile(sorted_values, 95),
            "p99": self._percentile(sorted_values, 99),
            "total": sum(sorted_values),
        }

    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        if f == c:
            return sorted_values[f]
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


@dataclass
class HistogramMetric:
    """直方图指标"""

    name: str
    buckets: List[float]  # 桶边界
    values: List[float] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def record(self, value: float) -> None:
        """记录一个值"""
        self.values.append(value)

    def get_distribution(self) -> Dict[str, Any]:
        """获取分布信息"""
        if not self.values:
            return {"count": 0, "buckets": {str(b): 0 for b in self.buckets}, "sum": 0.0}

        bucket_counts = {str(b): 0 for b in self.buckets}
        bucket_counts["+Inf"] = 0

        for value in self.values:
            for bucket in self.buckets:
                if value <= bucket:
                    bucket_counts[str(bucket)] += 1
                    break
            else:
                bucket_counts["+Inf"] += 1

        return {"count": len(self.values), "buckets": bucket_counts, "sum": sum(self.values)}


class MetricsCollector:
    """
    性能指标收集器

    特性:
    - 支持多种指标类型（计数器、仪表盘、直方图、计时器）
    - 线程安全
    - 支持标签（labels）进行指标分组
    - 提供统计信息查询

    属性33: 性能指标收集
    对于任何工作流执行，系统应该收集并记录性能指标（如响应时间、模型推理时间），
    且这些指标应该可以通过API查询。

    需求:
    - 12.4: 收集性能指标（响应时间、模型推理时间、内存使用）
    - 12.5: 提供API接口查询系统状态和性能指标
    """

    def __init__(self, max_history: int = 10000):
        """
        初始化指标收集器

        Args:
            max_history: 每个指标保留的最大历史记录数
        """
        self._lock = threading.RLock()
        self._max_history = max_history

        # 计数器存储
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # 仪表盘存储
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # 计时器存储
        self._timers: Dict[str, TimerMetric] = {}

        # 直方图存储
        self._histograms: Dict[str, HistogramMetric] = {}

        # 最近的指标值（用于查询）
        self._recent_values: List[MetricValue] = []

        # 启动时间
        self._start_time = time.time()

    def _get_label_key(self, labels: Optional[Dict[str, str]] = None) -> str:
        """生成标签键"""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _trim_history(self) -> None:
        """裁剪历史记录"""
        if len(self._recent_values) > self._max_history:
            self._recent_values = self._recent_values[-self._max_history :]

    # ============ 计数器操作 ============

    def increment(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """
        增加计数器值

        Args:
            name: 指标名称
            value: 增加的值（必须为正数）
            labels: 标签字典

        Returns:
            增加后的值
        """
        if value < 0:
            raise ValueError("Counter increment value must be non-negative")

        with self._lock:
            label_key = self._get_label_key(labels)
            self._counters[name][label_key] += value
            new_value = self._counters[name][label_key]

            # 记录到历史
            self._recent_values.append(
                MetricValue(
                    name=name,
                    value=new_value,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=MetricType.COUNTER,
                )
            )
            self._trim_history()

            return new_value

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取计数器值"""
        with self._lock:
            label_key = self._get_label_key(labels)
            return self._counters[name].get(label_key, 0.0)

    # ============ 仪表盘操作 ============

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        设置仪表盘值

        Args:
            name: 指标名称
            value: 值
            labels: 标签字典
        """
        with self._lock:
            label_key = self._get_label_key(labels)
            self._gauges[name][label_key] = value

            # 记录到历史
            self._recent_values.append(
                MetricValue(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=MetricType.GAUGE,
                )
            )
            self._trim_history()

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """获取仪表盘值"""
        with self._lock:
            label_key = self._get_label_key(labels)
            return self._gauges[name].get(label_key, 0.0)

    def increment_gauge(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """增加仪表盘值"""
        with self._lock:
            label_key = self._get_label_key(labels)
            self._gauges[name][label_key] += value
            new_value = self._gauges[name][label_key]

            self._recent_values.append(
                MetricValue(
                    name=name,
                    value=new_value,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=MetricType.GAUGE,
                )
            )
            self._trim_history()

            return new_value

    def decrement_gauge(
        self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """减少仪表盘值"""
        return self.increment_gauge(name, -value, labels)

    # ============ 计时器操作 ============

    def record_time(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录时间值

        Args:
            name: 指标名称
            duration: 持续时间（秒）
            labels: 标签字典
        """
        with self._lock:
            label_key = self._get_label_key(labels)
            timer_key = f"{name}:{label_key}"

            if timer_key not in self._timers:
                self._timers[timer_key] = TimerMetric(name=name, labels=labels or {})

            self._timers[timer_key].record(duration)

            # 裁剪计时器历史
            if len(self._timers[timer_key].values) > self._max_history:
                self._timers[timer_key].values = self._timers[timer_key].values[
                    -self._max_history :
                ]

            # 记录到历史
            self._recent_values.append(
                MetricValue(
                    name=name,
                    value=duration,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=MetricType.TIMER,
                )
            )
            self._trim_history()

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        计时器上下文管理器

        用法:
            with metrics.timer("workflow_execution", {"workflow": "search_qa"}):
                # 执行代码
                pass
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_time(name, duration, labels)

    def get_timer_stats(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """获取计时器统计信息"""
        with self._lock:
            label_key = self._get_label_key(labels)
            timer_key = f"{name}:{label_key}"

            if timer_key in self._timers:
                return self._timers[timer_key].get_statistics()

            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "total": 0.0,
            }

    # ============ 直方图操作 ============

    def create_histogram(
        self, name: str, buckets: List[float], labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        创建直方图

        Args:
            name: 指标名称
            buckets: 桶边界列表
            labels: 标签字典
        """
        with self._lock:
            label_key = self._get_label_key(labels)
            histogram_key = f"{name}:{label_key}"

            self._histograms[histogram_key] = HistogramMetric(
                name=name, buckets=sorted(buckets), labels=labels or {}
            )

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        记录直方图值

        Args:
            name: 指标名称
            value: 值
            labels: 标签字典
        """
        with self._lock:
            label_key = self._get_label_key(labels)
            histogram_key = f"{name}:{label_key}"

            # 如果直方图不存在，创建默认桶
            if histogram_key not in self._histograms:
                default_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                self._histograms[histogram_key] = HistogramMetric(
                    name=name, buckets=default_buckets, labels=labels or {}
                )

            self._histograms[histogram_key].record(value)

            # 裁剪直方图历史
            if len(self._histograms[histogram_key].values) > self._max_history:
                self._histograms[histogram_key].values = self._histograms[histogram_key].values[
                    -self._max_history :
                ]

            # 记录到历史
            self._recent_values.append(
                MetricValue(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=MetricType.HISTOGRAM,
                )
            )
            self._trim_history()

    def get_histogram_distribution(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """获取直方图分布"""
        with self._lock:
            label_key = self._get_label_key(labels)
            histogram_key = f"{name}:{label_key}"

            if histogram_key in self._histograms:
                return self._histograms[histogram_key].get_distribution()

            return {"count": 0, "buckets": {}, "sum": 0.0}

    # ============ 预定义指标 ============

    def record_workflow_execution(self, workflow_type: str, duration: float, success: bool) -> None:
        """
        记录工作流执行指标

        Args:
            workflow_type: 工作流类型
            duration: 执行时间（秒）
            success: 是否成功
        """
        labels = {"workflow": workflow_type}

        # 记录执行时间
        self.record_time("workflow_execution_time", duration, labels)

        # 增加执行计数
        self.increment("workflow_execution_total", 1.0, labels)

        # 记录成功/失败
        if success:
            self.increment("workflow_success_total", 1.0, labels)
        else:
            self.increment("workflow_failure_total", 1.0, labels)

    def record_model_inference(
        self,
        model_name: str,
        duration: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """
        记录模型推理指标

        Args:
            model_name: 模型名称
            duration: 推理时间（秒）
            input_tokens: 输入token数
            output_tokens: 输出token数
        """
        labels = {"model": model_name}

        # 记录推理时间
        self.record_time("model_inference_time", duration, labels)

        # 增加推理计数
        self.increment("model_inference_total", 1.0, labels)

        # 记录token数
        if input_tokens is not None:
            self.increment("model_input_tokens_total", float(input_tokens), labels)
        if output_tokens is not None:
            self.increment("model_output_tokens_total", float(output_tokens), labels)

    def record_memory_usage(self, component: str, memory_bytes: int) -> None:
        """
        记录内存使用

        Args:
            component: 组件名称
            memory_bytes: 内存使用量（字节）
        """
        labels = {"component": component}
        self.set_gauge("memory_usage_bytes", float(memory_bytes), labels)

    def record_request(self, endpoint: str, method: str, status_code: int, duration: float) -> None:
        """
        记录API请求指标

        Args:
            endpoint: 端点路径
            method: HTTP方法
            status_code: 状态码
            duration: 响应时间（秒）
        """
        labels = {"endpoint": endpoint, "method": method, "status": str(status_code)}

        # 记录响应时间
        self.record_time("http_request_duration", duration, labels)

        # 增加请求计数
        self.increment("http_requests_total", 1.0, labels)

    # ============ 查询接口 ============

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标

        Returns:
            包含所有指标的字典
        """
        with self._lock:
            result = {
                "counters": {},
                "gauges": {},
                "timers": {},
                "histograms": {},
                "uptime": time.time() - self._start_time,
            }

            # 收集计数器
            for name, label_values in self._counters.items():
                result["counters"][name] = dict(label_values)

            # 收集仪表盘
            for name, label_values in self._gauges.items():
                result["gauges"][name] = dict(label_values)

            # 收集计时器统计
            for key, timer in self._timers.items():
                if timer.name not in result["timers"]:
                    result["timers"][timer.name] = {}
                label_key = self._get_label_key(timer.labels) or "default"
                result["timers"][timer.name][label_key] = timer.get_statistics()

            # 收集直方图分布
            for key, histogram in self._histograms.items():
                if histogram.name not in result["histograms"]:
                    result["histograms"][histogram.name] = {}
                label_key = self._get_label_key(histogram.labels) or "default"
                result["histograms"][histogram.name][label_key] = histogram.get_distribution()

            return result

    def get_metric(
        self,
        name: str,
        metric_type: Optional[MetricType] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        """
        获取单个指标

        Args:
            name: 指标名称
            metric_type: 指标类型（可选，自动检测）
            labels: 标签字典

        Returns:
            指标值或统计信息
        """
        with self._lock:
            label_key = self._get_label_key(labels)

            # 尝试从各类型中查找
            if metric_type == MetricType.COUNTER or metric_type is None:
                if name in self._counters:
                    return self._counters[name].get(label_key, 0.0)

            if metric_type == MetricType.GAUGE or metric_type is None:
                if name in self._gauges:
                    return self._gauges[name].get(label_key, 0.0)

            if metric_type == MetricType.TIMER or metric_type is None:
                timer_key = f"{name}:{label_key}"
                if timer_key in self._timers:
                    return self._timers[timer_key].get_statistics()

            if metric_type == MetricType.HISTOGRAM or metric_type is None:
                histogram_key = f"{name}:{label_key}"
                if histogram_key in self._histograms:
                    return self._histograms[histogram_key].get_distribution()

            return None

    def get_recent_values(self, name: Optional[str] = None, limit: int = 100) -> List[MetricValue]:
        """
        获取最近的指标值

        Args:
            name: 指标名称（可选，不指定则返回所有）
            limit: 返回数量限制

        Returns:
            MetricValue列表
        """
        with self._lock:
            if name:
                filtered = [v for v in self._recent_values if v.name == name]
            else:
                filtered = self._recent_values.copy()

            return filtered[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要

        Returns:
            摘要信息字典
        """
        with self._lock:
            # 计算工作流统计
            workflow_stats = {}
            for key, timer in self._timers.items():
                if timer.name == "workflow_execution_time":
                    workflow = timer.labels.get("workflow", "unknown")
                    stats = timer.get_statistics()
                    workflow_stats[workflow] = {
                        "count": stats["count"],
                        "avg_time": stats["mean"],
                        "p95_time": stats["p95"],
                    }

            # 计算模型统计
            model_stats = {}
            for key, timer in self._timers.items():
                if timer.name == "model_inference_time":
                    model = timer.labels.get("model", "unknown")
                    stats = timer.get_statistics()
                    model_stats[model] = {
                        "count": stats["count"],
                        "avg_time": stats["mean"],
                        "p95_time": stats["p95"],
                    }

            # 总体统计
            total_requests = sum(
                v for v in self._counters.get("workflow_execution_total", {}).values()
            )
            total_success = sum(
                v for v in self._counters.get("workflow_success_total", {}).values()
            )

            return {
                "uptime": time.time() - self._start_time,
                "total_requests": total_requests,
                "success_rate": total_success / total_requests if total_requests > 0 else 0.0,
                "workflows": workflow_stats,
                "models": model_stats,
                "counter_count": len(self._counters),
                "gauge_count": len(self._gauges),
                "timer_count": len(self._timers),
                "histogram_count": len(self._histograms),
            }

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
            self._histograms.clear()
            self._recent_values.clear()
            self._start_time = time.time()


# 全局单例实例
_metrics_instance: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector(max_history: int = 10000) -> MetricsCollector:
    """
    获取MetricsCollector单例实例

    Args:
        max_history: 最大历史记录数

    Returns:
        MetricsCollector实例
    """
    global _metrics_instance

    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = MetricsCollector(max_history=max_history)

    return _metrics_instance


def reset_metrics_collector() -> None:
    """重置MetricsCollector单例"""
    global _metrics_instance

    with _metrics_lock:
        if _metrics_instance is not None:
            _metrics_instance.reset()
        _metrics_instance = None


def create_metrics_collector(max_history: int = 10000) -> MetricsCollector:
    """
    创建新的MetricsCollector实例

    Args:
        max_history: 最大历史记录数

    Returns:
        新的MetricsCollector实例
    """
    return MetricsCollector(max_history=max_history)


# ============ 便捷函数 ============


def record_workflow_execution(workflow_type: str, duration: float, success: bool) -> None:
    """便捷函数：记录工作流执行"""
    get_metrics_collector().record_workflow_execution(workflow_type, duration, success)


def record_model_inference(
    model_name: str,
    duration: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """便捷函数：记录模型推理"""
    get_metrics_collector().record_model_inference(
        model_name, duration, input_tokens, output_tokens
    )


def record_memory_usage(component: str, memory_bytes: int) -> None:
    """便捷函数：记录内存使用"""
    get_metrics_collector().record_memory_usage(component, memory_bytes)


def record_request(endpoint: str, method: str, status_code: int, duration: float) -> None:
    """便捷函数：记录API请求"""
    get_metrics_collector().record_request(endpoint, method, status_code, duration)


@contextmanager
def timer(name: str, labels: Optional[Dict[str, str]] = None):
    """便捷函数：计时器上下文管理器"""
    with get_metrics_collector().timer(name, labels):
        yield
