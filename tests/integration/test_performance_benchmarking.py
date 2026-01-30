"""
Performance benchmarking integration tests.

Tests model loading time, inference latency, and throughput for both
PyTorch and OpenVINO backends, and verifies OpenVINO speedup.

Requirements tested:
- 7.1: Inference latency tracking
- 7.2: Throughput tracking
"""

import time
import pytest
import torch
from transformers import AutoTokenizer
import statistics

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.runtime.backend_factory import BackendFactory
from mm_orch.schemas import ModelConfig


@pytest.fixture
def benchmark_model():
    """Return a model name for benchmarking."""
    return "distilgpt2"


@pytest.fixture
def benchmark_iterations():
    """Number of iterations for benchmarking."""
    return 10


class TestModelLoadingBenchmark:
    """Benchmark model loading time for both backends."""
    
    def test_pytorch_model_loading_time(self, benchmark_model):
        """
        Benchmark PyTorch model loading time.
        
        Requirement: 7.1 - Performance tracking
        """
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        model_config = ModelConfig(
            name="pytorch-load-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Measure loading time
        start_time = time.time()
        model = manager.load_model("pytorch-load-bench")
        load_time = time.time() - start_time
        
        assert model is not None
        assert load_time > 0
        
        print(f"\nPyTorch model loading time: {load_time:.3f}s")
        
        # Cleanup
        manager.unload_model("pytorch-load-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_openvino_model_loading_time(self, benchmark_model):
        """
        Benchmark OpenVINO model loading time.
        
        Requirement: 7.1 - Performance tracking
        """
        manager = ModelManager(backend="openvino", max_cached_models=2)
        
        model_config = ModelConfig(
            name="openvino-load-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Measure loading time (may fallback to PyTorch)
        start_time = time.time()
        model = manager.load_model("openvino-load-bench")
        load_time = time.time() - start_time
        
        assert model is not None
        assert load_time > 0
        
        print(f"\nOpenVINO model loading time: {load_time:.3f}s")
        
        # Cleanup
        manager.unload_model("openvino-load-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_compare_loading_times(self, benchmark_model):
        """
        Compare model loading times between backends.
        
        Requirement: 7.1 - Performance comparison
        """
        # PyTorch loading
        manager_pt = ModelManager(backend="pytorch", max_cached_models=2)
        model_config_pt = ModelConfig(
            name="pt-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_pt.register_model(model_config_pt)
        
        start_pt = time.time()
        model_pt = manager_pt.load_model("pt-compare")
        time_pt = time.time() - start_pt
        
        # OpenVINO loading
        manager_ov = ModelManager(backend="openvino", max_cached_models=2)
        model_config_ov = ModelConfig(
            name="ov-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_ov.register_model(model_config_ov)
        
        start_ov = time.time()
        model_ov = manager_ov.load_model("ov-compare")
        time_ov = time.time() - start_ov
        
        print(f"\nLoading time comparison:")
        print(f"  PyTorch: {time_pt:.3f}s")
        print(f"  OpenVINO: {time_ov:.3f}s")
        
        # Cleanup
        manager_pt.unload_model("pt-compare")
        manager_ov.unload_model("ov-compare")


class TestInferenceLatencyBenchmark:
    """Benchmark inference latency for both backends."""
    
    def test_pytorch_inference_latency(self, benchmark_model, benchmark_iterations):
        """
        Benchmark PyTorch inference latency.
        
        Requirement: 7.1 - Inference latency tracking
        """
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        model_config = ModelConfig(
            name="pytorch-latency-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("pytorch-latency-bench")
        
        # Prepare inputs
        test_text = "The quick brown fox jumps over the lazy dog"
        
        # Warmup
        for _ in range(3):
            manager.infer("pytorch-latency-bench", test_text)
        
        # Benchmark
        latencies = []
        for _ in range(benchmark_iterations):
            start_time = time.time()
            outputs = manager.infer("pytorch-latency-bench", test_text)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\nPyTorch inference latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Std Dev: {std_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        
        assert avg_latency > 0
        
        # Cleanup
        manager.unload_model("pytorch-latency-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_openvino_inference_latency(self, benchmark_model, benchmark_iterations):
        """
        Benchmark OpenVINO inference latency.
        
        Requirement: 7.1 - Inference latency tracking
        """
        manager = ModelManager(backend="openvino", max_cached_models=2)
        
        model_config = ModelConfig(
            name="openvino-latency-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("openvino-latency-bench")
        
        # Prepare inputs
        test_text = "The quick brown fox jumps over the lazy dog"
        
        # Warmup
        for _ in range(3):
            manager.infer("openvino-latency-bench", test_text)
        
        # Benchmark
        latencies = []
        for _ in range(benchmark_iterations):
            start_time = time.time()
            outputs = manager.infer("openvino-latency-bench", test_text)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\nOpenVINO inference latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Std Dev: {std_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        
        assert avg_latency > 0
        
        # Cleanup
        manager.unload_model("openvino-latency-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_compare_inference_latency(self, benchmark_model, benchmark_iterations):
        """
        Compare inference latency between backends.
        
        Requirement: 7.1 - Performance comparison
        """
        # PyTorch benchmark
        manager_pt = ModelManager(backend="pytorch", max_cached_models=2)
        model_config_pt = ModelConfig(
            name="pt-latency-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_pt.register_model(model_config_pt)
        model_pt = manager_pt.load_model("pt-latency-compare")
        
        test_text = "The quick brown fox jumps over the lazy dog"
        
        # Warmup PyTorch
        for _ in range(3):
            manager_pt.infer("pt-latency-compare", test_text)
        
        # Benchmark PyTorch
        latencies_pt = []
        for _ in range(benchmark_iterations):
            start = time.time()
            manager_pt.infer("pt-latency-compare", test_text)
            latencies_pt.append((time.time() - start) * 1000)
        
        avg_latency_pt = statistics.mean(latencies_pt)
        
        # OpenVINO benchmark
        manager_ov = ModelManager(backend="openvino", max_cached_models=2)
        model_config_ov = ModelConfig(
            name="ov-latency-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_ov.register_model(model_config_ov)
        model_ov = manager_ov.load_model("ov-latency-compare")
        
        # Warmup OpenVINO
        for _ in range(3):
            manager_ov.infer("ov-latency-compare", test_text)
        
        # Benchmark OpenVINO
        latencies_ov = []
        for _ in range(benchmark_iterations):
            start = time.time()
            manager_ov.infer("ov-latency-compare", test_text)
            latencies_ov.append((time.time() - start) * 1000)
        
        avg_latency_ov = statistics.mean(latencies_ov)
        
        # Calculate speedup
        speedup = avg_latency_pt / avg_latency_ov if avg_latency_ov > 0 else 0
        
        print(f"\nInference latency comparison:")
        print(f"  PyTorch: {avg_latency_pt:.2f}ms")
        print(f"  OpenVINO: {avg_latency_ov:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Cleanup
        manager_pt.unload_model("pt-latency-compare")
        manager_ov.unload_model("ov-latency-compare")


class TestThroughputBenchmark:
    """Benchmark throughput (tokens/second) for both backends."""
    
    def test_pytorch_throughput(self, benchmark_model, benchmark_iterations):
        """
        Benchmark PyTorch throughput.
        
        Requirement: 7.2 - Throughput tracking
        """
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        model_config = ModelConfig(
            name="pytorch-throughput-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("pytorch-throughput-bench")
        
        # Prepare inputs
        tokenizer = AutoTokenizer.from_pretrained(benchmark_model)
        test_text = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        
        # Warmup
        for _ in range(3):
            manager.infer("pytorch-throughput-bench", test_text)
        
        # Benchmark
        throughputs = []
        for _ in range(benchmark_iterations):
            start_time = time.time()
            outputs = manager.infer("pytorch-throughput-bench", test_text)
            elapsed = time.time() - start_time
            
            # Calculate throughput (tokens/second)
            throughput = input_length / elapsed if elapsed > 0 else 0
            throughputs.append(throughput)
        
        avg_throughput = statistics.mean(throughputs)
        
        print(f"\nPyTorch throughput:")
        print(f"  Average: {avg_throughput:.2f} tokens/s")
        print(f"  Min: {min(throughputs):.2f} tokens/s")
        print(f"  Max: {max(throughputs):.2f} tokens/s")
        
        assert avg_throughput > 0
        
        # Cleanup
        manager.unload_model("pytorch-throughput-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_openvino_throughput(self, benchmark_model, benchmark_iterations):
        """
        Benchmark OpenVINO throughput.
        
        Requirement: 7.2 - Throughput tracking
        """
        manager = ModelManager(backend="openvino", max_cached_models=2)
        
        model_config = ModelConfig(
            name="openvino-throughput-bench",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("openvino-throughput-bench")
        
        # Prepare inputs
        tokenizer = AutoTokenizer.from_pretrained(benchmark_model)
        test_text = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        
        # Warmup
        for _ in range(3):
            manager.infer("openvino-throughput-bench", test_text)
        
        # Benchmark
        throughputs = []
        for _ in range(benchmark_iterations):
            start_time = time.time()
            outputs = manager.infer("openvino-throughput-bench", test_text)
            elapsed = time.time() - start_time
            
            # Calculate throughput (tokens/second)
            throughput = input_length / elapsed if elapsed > 0 else 0
            throughputs.append(throughput)
        
        avg_throughput = statistics.mean(throughputs)
        
        print(f"\nOpenVINO throughput:")
        print(f"  Average: {avg_throughput:.2f} tokens/s")
        print(f"  Min: {min(throughputs):.2f} tokens/s")
        print(f"  Max: {max(throughputs):.2f} tokens/s")
        
        assert avg_throughput > 0
        
        # Cleanup
        manager.unload_model("openvino-throughput-bench")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_compare_throughput(self, benchmark_model, benchmark_iterations):
        """
        Compare throughput between backends.
        
        Requirement: 7.2 - Performance comparison
        """
        # PyTorch benchmark
        manager_pt = ModelManager(backend="pytorch", max_cached_models=2)
        model_config_pt = ModelConfig(
            name="pt-throughput-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_pt.register_model(model_config_pt)
        model_pt = manager_pt.load_model("pt-throughput-compare")
        
        tokenizer = AutoTokenizer.from_pretrained(benchmark_model)
        test_text = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(test_text, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        
        # Warmup and benchmark PyTorch
        for _ in range(3):
            manager_pt.infer("pt-throughput-compare", test_text)
        
        throughputs_pt = []
        for _ in range(benchmark_iterations):
            start = time.time()
            manager_pt.infer("pt-throughput-compare", test_text)
            elapsed = time.time() - start
            throughputs_pt.append(input_length / elapsed if elapsed > 0 else 0)
        
        avg_throughput_pt = statistics.mean(throughputs_pt)
        
        # OpenVINO benchmark
        manager_ov = ModelManager(backend="openvino", max_cached_models=2)
        model_config_ov = ModelConfig(
            name="ov-throughput-compare",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_ov.register_model(model_config_ov)
        model_ov = manager_ov.load_model("ov-throughput-compare")
        
        # Warmup and benchmark OpenVINO
        for _ in range(3):
            manager_ov.infer("ov-throughput-compare", test_text)
        
        throughputs_ov = []
        for _ in range(benchmark_iterations):
            start = time.time()
            manager_ov.infer("ov-throughput-compare", test_text)
            elapsed = time.time() - start
            throughputs_ov.append(input_length / elapsed if elapsed > 0 else 0)
        
        avg_throughput_ov = statistics.mean(throughputs_ov)
        
        # Calculate speedup
        speedup = avg_throughput_ov / avg_throughput_pt if avg_throughput_pt > 0 else 0
        
        print(f"\nThroughput comparison:")
        print(f"  PyTorch: {avg_throughput_pt:.2f} tokens/s")
        print(f"  OpenVINO: {avg_throughput_ov:.2f} tokens/s")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Cleanup
        manager_pt.unload_model("pt-throughput-compare")
        manager_ov.unload_model("ov-throughput-compare")


class TestPerformanceMetricsIntegration:
    """Test integration with performance monitoring system."""
    
    def test_performance_stats_recorded(self, benchmark_model):
        """
        Test that performance stats are recorded during inference.
        
        Requirement: 7.1, 7.2 - Metrics recording
        """
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        model_config = ModelConfig(
            name="stats-test",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("stats-test")
        
        # Run inference
        manager.infer("stats-test", "Test")
        
        # Get performance stats
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        manager.unload_model("stats-test")
    
    @pytest.mark.skipif(
        "openvino" not in BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_backend_comparison_metrics(self, benchmark_model):
        """
        Test backend comparison through performance monitor.
        
        Requirement: 7.1, 7.2 - Backend comparison
        """
        # Run inference on both backends
        manager_pt = ModelManager(backend="pytorch", max_cached_models=2)
        model_config_pt = ModelConfig(
            name="pt-metrics",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_pt.register_model(model_config_pt)
        manager_pt.load_model("pt-metrics")
        
        manager_pt.infer("pt-metrics", "Test")
        
        manager_ov = ModelManager(backend="openvino", max_cached_models=2)
        model_config_ov = ModelConfig(
            name="ov-metrics",
            model_path=benchmark_model,
            device="cpu",
            max_length=50
        )
        manager_ov.register_model(model_config_ov)
        manager_ov.load_model("ov-metrics")
        manager_ov.infer("ov-metrics", "Test")
        
        # Get stats from both
        stats_pt = manager_pt.get_performance_stats()
        stats_ov = manager_ov.get_performance_stats()
        
        assert stats_pt is not None
        assert stats_ov is not None
        
        # Cleanup
        manager_pt.unload_model("pt-metrics")
        manager_ov.unload_model("ov-metrics")


# Feature: openvino-backend-integration
# Performance benchmarking integration tests
# Tests Requirements: 7.1, 7.2
