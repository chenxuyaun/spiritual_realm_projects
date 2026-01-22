"""
基准测试报告生成模块单元测试
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from mm_orch.benchmark.latency import LatencyResult
from mm_orch.benchmark.memory import MemoryResult
from mm_orch.benchmark.throughput import ThroughputResult
from mm_orch.benchmark.reporter import (
    BenchmarkReport,
    BenchmarkReporter,
    SystemInfo,
)


class TestSystemInfo:
    """SystemInfo测试"""
    
    def test_create_system_info(self):
        """测试创建系统信息"""
        info = SystemInfo(
            platform="Linux",
            python_version="3.9.0",
            torch_version="2.0.0",
            cuda_available=True,
            cuda_version="11.8",
            gpu_name="NVIDIA A100",
            gpu_memory_gb=40.0,
            cpu_count=32,
            cpu_memory_gb=128.0,
        )
        
        assert info.platform == "Linux"
        assert info.cuda_available is True
        assert info.gpu_memory_gb == 40.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        info = SystemInfo(
            platform="Linux",
            python_version="3.9.0",
            torch_version="2.0.0",
            cuda_available=True,
            gpu_name="NVIDIA A100",
            gpu_memory_gb=40.0,
            cpu_count=32,
            cpu_memory_gb=128.0,
        )
        
        d = info.to_dict()
        
        assert d["platform"] == "Linux"
        assert d["cuda"]["available"] is True
        assert d["gpu"]["name"] == "NVIDIA A100"
        assert d["cpu"]["count"] == 32
    
    def test_default_values(self):
        """测试默认值"""
        info = SystemInfo()
        
        assert info.platform == ""
        assert info.cuda_available is False
        assert info.gpu_memory_gb == 0.0


class TestBenchmarkReport:
    """BenchmarkReport测试"""
    
    def test_create_report(self):
        """测试创建报告"""
        report = BenchmarkReport(
            report_name="test_report",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        assert report.report_name == "test_report"
        assert report.model_name == "gpt2"
        assert report.latency_results == []
        assert report.memory_results == []
        assert report.throughput_results == []
    
    def test_to_dict(self):
        """测试转换为字典"""
        report = BenchmarkReport(
            report_name="test_report",
            model_name="gpt2",
            timestamp=datetime.now(),
            system_info=SystemInfo(platform="Linux"),
            metadata={"version": "1.0"},
        )
        
        d = report.to_dict()
        
        assert d["report_name"] == "test_report"
        assert d["model_name"] == "gpt2"
        assert d["system_info"]["platform"] == "Linux"
        assert d["metadata"]["version"] == "1.0"
    
    def test_to_dict_without_system_info(self):
        """测试无系统信息时转换为字典"""
        report = BenchmarkReport(
            report_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        d = report.to_dict()
        
        assert d["system_info"] is None
    
    def test_get_summary(self):
        """测试获取摘要"""
        report = BenchmarkReport(
            report_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
            latency_results=[
                LatencyResult(
                    test_name="latency",
                    model_name="gpt2",
                    timestamp=datetime.now(),
                    ttft_mean=0.1,
                    tokens_per_second_mean=50.0,
                    e2e_latency_mean=1.0,
                )
            ],
            memory_results=[
                MemoryResult(
                    test_name="memory",
                    model_name="gpt2",
                    timestamp=datetime.now(),
                    model_load_gpu_mb=1000.0,
                    inference_gpu_peak_mb=1500.0,
                )
            ],
            throughput_results=[
                ThroughputResult(
                    test_name="throughput",
                    model_name="gpt2",
                    timestamp=datetime.now(),
                    single_requests_per_second=10.0,
                    single_tokens_per_second=500.0,
                )
            ],
        )
        
        summary = report.get_summary()
        
        assert summary["report_name"] == "test"
        assert summary["latency"]["ttft_mean_ms"] == 100.0
        assert summary["memory"]["model_load_gpu_mb"] == 1000.0
        assert summary["throughput"]["requests_per_second"] == 10.0
    
    def test_get_summary_empty(self):
        """测试空报告摘要"""
        report = BenchmarkReport(
            report_name="test",
            model_name="gpt2",
            timestamp=datetime.now(),
        )
        
        summary = report.get_summary()
        
        assert "latency" not in summary
        assert "memory" not in summary
        assert "throughput" not in summary


class TestBenchmarkReporter:
    """BenchmarkReporter测试"""
    
    def test_init(self):
        """测试初始化"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(
                output_dir=tmpdir,
                include_system_info=True,
            )
            
            assert reporter.output_dir == Path(tmpdir)
            assert reporter.include_system_info is True
    
    def test_init_creates_directory(self):
        """测试初始化创建目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "benchmarks", "results")
            reporter = BenchmarkReporter(output_dir=output_dir)
            
            assert Path(output_dir).exists()
    
    def test_collect_system_info(self):
        """测试收集系统信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            info = reporter.collect_system_info()
            
            assert info.platform != ""
            assert info.python_version != ""
            assert info.torch_version != ""
            assert info.timestamp != ""
    
    def test_create_report(self):
        """测试创建报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(
                output_dir=tmpdir,
                include_system_info=True,
            )
            
            report = reporter.create_report(
                model_name="gpt2",
                report_name="test_report",
            )
            
            assert report.report_name == "test_report"
            assert report.model_name == "gpt2"
            assert report.system_info is not None
    
    def test_create_report_auto_name(self):
        """测试自动生成报告名称"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = reporter.create_report(model_name="gpt2")
            
            assert "gpt2" in report.report_name
    
    def test_create_report_without_system_info(self):
        """测试不包含系统信息的报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(
                output_dir=tmpdir,
                include_system_info=False,
            )
            
            report = reporter.create_report(model_name="gpt2")
            
            assert report.system_info is None
    
    def test_save_json(self):
        """测试保存JSON报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            filepath = reporter.save_json(report)
            
            assert os.path.exists(filepath)
            assert filepath.endswith(".json")
            
            # 验证JSON内容
            with open(filepath, "r") as f:
                data = json.load(f)
            
            assert data["report_name"] == "test"
            assert data["model_name"] == "gpt2"
    
    def test_save_json_custom_filename(self):
        """测试自定义文件名保存JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            filepath = reporter.save_json(report, filename="custom.json")
            
            assert filepath.endswith("custom.json")
    
    def test_save_csv(self):
        """测试保存CSV报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
                latency_results=[
                    LatencyResult(
                        test_name="latency",
                        model_name="gpt2",
                        timestamp=datetime.now(),
                        ttft_mean=0.1,
                        tokens_per_second_mean=50.0,
                    )
                ],
            )
            
            filepath = reporter.save_csv(report)
            
            assert os.path.exists(filepath)
            assert filepath.endswith(".csv")
            
            # 验证CSV内容
            with open(filepath, "r") as f:
                content = f.read()
            
            assert "latency" in content
            assert "ttft_mean_ms" in content
    
    def test_save_csv_empty_report(self):
        """测试保存空报告CSV"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            filepath = reporter.save_csv(report)
            
            # 空报告不会创建CSV文件（没有数据行）
            # 这是预期行为
            assert filepath.endswith(".csv")
    
    def test_save_report_json(self):
        """测试save_report方法（JSON）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            filepath = reporter.save_report(report, format="json")
            
            assert filepath.endswith(".json")
    
    def test_save_report_csv(self):
        """测试save_report方法（CSV）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            filepath = reporter.save_report(report, format="csv")
            
            assert filepath.endswith(".csv")
    
    def test_save_report_invalid_format(self):
        """测试无效格式"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
            )
            
            with pytest.raises(ValueError):
                reporter.save_report(report, format="invalid")
    
    def test_load_report(self):
        """测试加载报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            # 创建并保存报告
            original = BenchmarkReport(
                report_name="test",
                model_name="gpt2",
                timestamp=datetime.now(),
                metadata={"key": "value"},
            )
            filepath = reporter.save_json(original)
            
            # 加载报告
            loaded = reporter.load_report(filepath)
            
            assert loaded.report_name == "test"
            assert loaded.model_name == "gpt2"
            assert loaded.metadata["key"] == "value"
    
    def test_compare_reports(self):
        """测试比较报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(output_dir=tmpdir)
            
            report1 = BenchmarkReport(
                report_name="report1",
                model_name="gpt2",
                timestamp=datetime.now(),
                latency_results=[
                    LatencyResult(
                        test_name="latency",
                        model_name="gpt2",
                        timestamp=datetime.now(),
                        ttft_mean=0.1,
                    )
                ],
            )
            
            report2 = BenchmarkReport(
                report_name="report2",
                model_name="gpt2-medium",
                timestamp=datetime.now(),
                latency_results=[
                    LatencyResult(
                        test_name="latency",
                        model_name="gpt2-medium",
                        timestamp=datetime.now(),
                        ttft_mean=0.2,
                    )
                ],
            )
            
            comparison = reporter.compare_reports([report1, report2])
            
            assert "report1" in comparison["reports"]
            assert "report2" in comparison["reports"]
            assert "report1" in comparison["latency"]
            assert "report2" in comparison["latency"]


class TestBenchmarkReporterIntegration:
    """BenchmarkReporter集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReporter(
                output_dir=tmpdir,
                include_system_info=True,
            )
            
            # 创建带完整结果的报告
            report = reporter.create_report(
                model_name="gpt2",
                latency_results=[
                    LatencyResult(
                        test_name="latency",
                        model_name="gpt2",
                        timestamp=datetime.now(),
                        ttft_mean=0.1,
                        tokens_per_second_mean=50.0,
                        e2e_latency_mean=1.0,
                    )
                ],
                memory_results=[
                    MemoryResult(
                        test_name="memory",
                        model_name="gpt2",
                        timestamp=datetime.now(),
                        model_load_gpu_mb=500.0,
                        inference_gpu_peak_mb=600.0,
                    )
                ],
                throughput_results=[
                    ThroughputResult(
                        test_name="throughput",
                        model_name="gpt2",
                        timestamp=datetime.now(),
                        single_requests_per_second=10.0,
                        single_tokens_per_second=500.0,
                    )
                ],
                metadata={"test": True},
            )
            
            # 保存JSON
            json_path = reporter.save_json(report)
            assert os.path.exists(json_path)
            
            # 保存CSV
            csv_path = reporter.save_csv(report)
            assert os.path.exists(csv_path)
            
            # 加载并验证
            loaded = reporter.load_report(json_path)
            assert loaded.report_name == report.report_name
            
            # 获取摘要
            summary = report.get_summary()
            assert "latency" in summary
            assert "memory" in summary
            assert "throughput" in summary
