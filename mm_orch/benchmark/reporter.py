"""
基准测试报告生成模块

提供JSON、CSV报告生成和系统信息收集功能。
"""

import csv
import json
import logging
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from mm_orch.benchmark.latency import LatencyResult
from mm_orch.benchmark.memory import MemoryResult
from mm_orch.benchmark.throughput import ThroughputResult

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """系统信息"""
    platform: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_count: int = 0
    cpu_memory_gb: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda": {
                "available": self.cuda_available,
                "version": self.cuda_version,
            },
            "gpu": {
                "name": self.gpu_name,
                "memory_gb": self.gpu_memory_gb,
            },
            "cpu": {
                "count": self.cpu_count,
                "memory_gb": self.cpu_memory_gb,
            },
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkReport:
    """基准测试报告"""
    report_name: str
    model_name: str
    timestamp: datetime
    system_info: Optional[SystemInfo] = None
    latency_results: List[LatencyResult] = field(default_factory=list)
    memory_results: List[MemoryResult] = field(default_factory=list)
    throughput_results: List[ThroughputResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "report_name": self.report_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info.to_dict() if self.system_info else None,
            "latency": [r.to_dict() for r in self.latency_results],
            "memory": [r.to_dict() for r in self.memory_results],
            "throughput": [r.to_dict() for r in self.throughput_results],
            "metadata": self.metadata,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取报告摘要"""
        summary = {
            "report_name": self.report_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
        }
        
        # 延迟摘要
        if self.latency_results:
            latest = self.latency_results[-1]
            summary["latency"] = {
                "ttft_mean_ms": latest.ttft_mean * 1000,
                "tokens_per_second": latest.tokens_per_second_mean,
                "e2e_latency_mean_ms": latest.e2e_latency_mean * 1000,
            }
        
        # 内存摘要
        if self.memory_results:
            latest = self.memory_results[-1]
            summary["memory"] = {
                "model_load_gpu_mb": latest.model_load_gpu_mb,
                "inference_gpu_peak_mb": latest.inference_gpu_peak_mb,
            }
        
        # 吞吐量摘要
        if self.throughput_results:
            latest = self.throughput_results[-1]
            summary["throughput"] = {
                "requests_per_second": max(
                    latest.single_requests_per_second,
                    latest.concurrent_requests_per_second,
                    latest.batch_requests_per_second
                ),
                "tokens_per_second": max(
                    latest.single_tokens_per_second,
                    latest.concurrent_tokens_per_second,
                    latest.batch_tokens_per_second
                ),
            }
        
        return summary


class BenchmarkReporter:
    """
    基准测试报告生成器
    
    支持生成JSON和CSV格式的报告，并收集系统信息。
    """
    
    def __init__(
        self,
        output_dir: str = "data/benchmarks",
        include_system_info: bool = True,
        timestamp_format: str = "%Y%m%d_%H%M%S"
    ):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
            include_system_info: 是否包含系统信息
            timestamp_format: 时间戳格式
        """
        self.output_dir = Path(output_dir)
        self.include_system_info = include_system_info
        self.timestamp_format = timestamp_format
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_system_info(self) -> SystemInfo:
        """
        收集系统信息
        
        Returns:
            SystemInfo: 系统信息
        """
        info = SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            timestamp=datetime.now().isoformat(),
        )
        
        # CUDA信息
        if torch.cuda.is_available():
            info.cuda_version = torch.version.cuda or ""
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
            )
        
        # CPU信息
        info.cpu_count = os.cpu_count() or 0
        
        if HAS_PSUTIL:
            info.cpu_memory_gb = psutil.virtual_memory().total / 1024**3
        
        return info
    
    def create_report(
        self,
        model_name: str,
        report_name: Optional[str] = None,
        latency_results: Optional[List[LatencyResult]] = None,
        memory_results: Optional[List[MemoryResult]] = None,
        throughput_results: Optional[List[ThroughputResult]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkReport:
        """
        创建基准测试报告
        
        Args:
            model_name: 模型名称
            report_name: 报告名称
            latency_results: 延迟测试结果
            memory_results: 内存测试结果
            throughput_results: 吞吐量测试结果
            metadata: 元数据
            
        Returns:
            BenchmarkReport: 基准测试报告
        """
        timestamp = datetime.now()
        
        if report_name is None:
            report_name = f"{model_name}_{timestamp.strftime(self.timestamp_format)}"
        
        report = BenchmarkReport(
            report_name=report_name,
            model_name=model_name,
            timestamp=timestamp,
            latency_results=latency_results or [],
            memory_results=memory_results or [],
            throughput_results=throughput_results or [],
            metadata=metadata or {},
        )
        
        if self.include_system_info:
            report.system_info = self.collect_system_info()
        
        return report
    
    def save_json(
        self,
        report: BenchmarkReport,
        filename: Optional[str] = None
    ) -> str:
        """
        保存JSON格式报告
        
        Args:
            report: 基准测试报告
            filename: 文件名（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"{report.report_name}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to {filepath}")
        return str(filepath)
    
    def save_csv(
        self,
        report: BenchmarkReport,
        filename: Optional[str] = None
    ) -> str:
        """
        保存CSV格式报告
        
        Args:
            report: 基准测试报告
            filename: 文件名（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"{report.report_name}.csv"
        
        filepath = self.output_dir / filename
        
        rows = []
        
        # 延迟结果
        for result in report.latency_results:
            rows.append({
                "type": "latency",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "ttft_mean_ms",
                "value": result.ttft_mean * 1000,
            })
            rows.append({
                "type": "latency",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "tokens_per_second",
                "value": result.tokens_per_second_mean,
            })
            rows.append({
                "type": "latency",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "e2e_latency_mean_ms",
                "value": result.e2e_latency_mean * 1000,
            })
        
        # 内存结果
        for result in report.memory_results:
            rows.append({
                "type": "memory",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "model_load_gpu_mb",
                "value": result.model_load_gpu_mb,
            })
            rows.append({
                "type": "memory",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "inference_gpu_peak_mb",
                "value": result.inference_gpu_peak_mb,
            })
        
        # 吞吐量结果
        for result in report.throughput_results:
            rows.append({
                "type": "throughput",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "requests_per_second",
                "value": max(
                    result.single_requests_per_second,
                    result.concurrent_requests_per_second,
                    result.batch_requests_per_second
                ),
            })
            rows.append({
                "type": "throughput",
                "test_name": result.test_name,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "metric": "tokens_per_second",
                "value": max(
                    result.single_tokens_per_second,
                    result.concurrent_tokens_per_second,
                    result.batch_tokens_per_second
                ),
            })
        
        if rows:
            fieldnames = ["type", "test_name", "model_name", "timestamp", "metric", "value"]
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"CSV report saved to {filepath}")
        return str(filepath)
    
    def save_report(
        self,
        report: BenchmarkReport,
        format: str = "json"
    ) -> str:
        """
        保存报告
        
        Args:
            report: 基准测试报告
            format: 格式 ("json" | "csv")
            
        Returns:
            str: 保存的文件路径
        """
        if format == "json":
            return self.save_json(report)
        elif format == "csv":
            return self.save_csv(report)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_report(self, filepath: str) -> BenchmarkReport:
        """
        加载JSON报告
        
        Args:
            filepath: 文件路径
            
        Returns:
            BenchmarkReport: 基准测试报告
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 重建报告对象
        report = BenchmarkReport(
            report_name=data["report_name"],
            model_name=data["model_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )
        
        # 重建系统信息
        if data.get("system_info"):
            si = data["system_info"]
            report.system_info = SystemInfo(
                platform=si.get("platform", ""),
                python_version=si.get("python_version", ""),
                torch_version=si.get("torch_version", ""),
                cuda_available=si.get("cuda", {}).get("available", False),
                cuda_version=si.get("cuda", {}).get("version", ""),
                gpu_name=si.get("gpu", {}).get("name", ""),
                gpu_memory_gb=si.get("gpu", {}).get("memory_gb", 0.0),
                cpu_count=si.get("cpu", {}).get("count", 0),
                cpu_memory_gb=si.get("cpu", {}).get("memory_gb", 0.0),
                timestamp=si.get("timestamp", ""),
            )
        
        # 注意：完整重建结果对象需要更多代码，这里简化处理
        # 实际使用时可以扩展此方法
        
        return report
    
    def compare_reports(
        self,
        reports: List[BenchmarkReport]
    ) -> Dict[str, Any]:
        """
        比较多个报告
        
        Args:
            reports: 报告列表
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        comparison = {
            "reports": [r.report_name for r in reports],
            "models": [r.model_name for r in reports],
            "latency": {},
            "memory": {},
            "throughput": {},
        }
        
        for report in reports:
            name = report.report_name
            
            # 延迟比较
            if report.latency_results:
                latest = report.latency_results[-1]
                comparison["latency"][name] = {
                    "ttft_mean_ms": latest.ttft_mean * 1000,
                    "tokens_per_second": latest.tokens_per_second_mean,
                }
            
            # 内存比较
            if report.memory_results:
                latest = report.memory_results[-1]
                comparison["memory"][name] = {
                    "model_load_gpu_mb": latest.model_load_gpu_mb,
                    "inference_gpu_peak_mb": latest.inference_gpu_peak_mb,
                }
            
            # 吞吐量比较
            if report.throughput_results:
                latest = report.throughput_results[-1]
                comparison["throughput"][name] = {
                    "requests_per_second": max(
                        latest.single_requests_per_second,
                        latest.concurrent_requests_per_second,
                        latest.batch_requests_per_second
                    ),
                }
        
        return comparison
