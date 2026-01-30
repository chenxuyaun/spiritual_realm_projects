"""
Unit tests for PerformanceMonitor.

Tests specific examples and edge cases for performance monitoring functionality.
"""

import pytest
import time
from mm_orch.runtime.performance_monitor import PerformanceMonitor, InferenceMetrics


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor class."""
    
    def test_initialization(self):
        """Test PerformanceMonitor initializes correctly."""
        monitor = PerformanceMonitor()
        
        assert monitor is not None
        assert monitor._metrics is not None
        assert monitor._inference_records is not None
        assert monitor._backend_stats is not None
        assert len(monitor._metrics) == 0
        assert len(monitor._inference_records) == 0
    
    def test_record_single_inference(self):
        """Test recording a single inference operation."""
        monitor = PerformanceMonitor()
        
        monitor.record_inference(
            backend="pytorch",
            model_name="gpt2",
            latency=1.5,
            tokens=100
        )
        
        # Check metrics were recorded
        assert len(monitor._inference_records) == 1
        
        # Check record details
        record = monitor._inference_records[0]
        assert record.backend == "pytorch"
        assert record.model_name == "gpt2"
        assert record.latency_ms == 1500.0  # 1.5s = 1500ms
        assert record.tokens_generated == 100
        assert record.throughput_tokens_per_sec == pytest.approx(100 / 1.5)
    
    def test_get_backend_stats_single_inference(self):
        """Test getting backend statistics for single inference."""
        monitor = PerformanceMonitor()
        
        monitor.record_inference(
            backend="pytorch",
            model_name="gpt2",
            latency=2.0,
            tokens=200
        )
        
        stats = monitor.get_backend_stats("pytorch")
        
        assert stats["backend"] == "pytorch"
        assert stats["avg_latency"] == 2.0
        assert stats["p50_latency"] == 2.0
        assert stats["p95_latency"] == 2.0
        assert stats["avg_throughput"] == pytest.approx(100.0)  # 200 tokens / 2.0s
        assert stats["sample_count"] == 1
    
    def test_get_backend_stats_multiple_inferences(self):
        """Test getting backend statistics for multiple inferences."""
        monitor = PerformanceMonitor()
        
        # Record multiple inferences with different latencies
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        tokens = 100
        
        for latency in latencies:
            monitor.record_inference(
                backend="pytorch",
                model_name="gpt2",
                latency=latency,
                tokens=tokens
            )
        
        stats = monitor.get_backend_stats("pytorch")
        
        assert stats["backend"] == "pytorch"
        assert stats["avg_latency"] == pytest.approx(3.0)  # Mean of 1,2,3,4,5
        assert stats["p50_latency"] == pytest.approx(3.0)  # Median
        assert stats["p95_latency"] == pytest.approx(4.8)  # 95th percentile
        assert stats["sample_count"] == 5
    
    def test_get_backend_stats_empty(self):
        """Test getting stats for backend with no data."""
        monitor = PerformanceMonitor()
        
        stats = monitor.get_backend_stats("nonexistent")
        
        assert stats == {}
    
    def test_compare_backends_basic(self):
        """Test basic backend comparison."""
        monitor = PerformanceMonitor()
        
        # PyTorch: slower (2.0s for 100 tokens = 50 tok/s)
        monitor.record_inference(
            backend="pytorch",
            model_name="gpt2",
            latency=2.0,
            tokens=100
        )
        
        # OpenVINO: faster (1.0s for 100 tokens = 100 tok/s)
        monitor.record_inference(
            backend="openvino",
            model_name="gpt2",
            latency=1.0,
            tokens=100
        )
        
        comparison = monitor.compare_backends("pytorch", "openvino")
        
        assert "latency_improvement" in comparison
        assert "throughput_improvement" in comparison
        assert "pytorch_stats" in comparison
        assert "openvino_stats" in comparison
        
        # OpenVINO should be 2x faster (latency improvement = 2.0/1.0 = 2.0)
        assert comparison["latency_improvement"] == pytest.approx(2.0)
        
        # OpenVINO should have 2x throughput (100/50 = 2.0)
        assert comparison["throughput_improvement"] == pytest.approx(2.0)
    
    def test_compare_backends_missing_data(self):
        """Test backend comparison with missing data."""
        monitor = PerformanceMonitor()
        
        # Only record for one backend
        monitor.record_inference(
            backend="pytorch",
            model_name="gpt2",
            latency=1.0,
            tokens=100
        )
        
        # Comparison should return empty dict
        comparison = monitor.compare_backends("pytorch", "openvino")
        assert comparison == {}
    
    def test_get_model_stats(self):
        """Test getting model-specific statistics."""
        monitor = PerformanceMonitor()
        
        # Record for same model with different backends
        monitor.record_inference(
            backend="pytorch",
            model_name="gpt2",
            latency=2.0,
            tokens=100
        )
        
        monitor.record_inference(
            backend="openvino",
            model_name="gpt2",
            latency=1.0,
            tokens=100
        )
        
        # Get stats for model on specific backend
        pytorch_stats = monitor.get_model_stats("gpt2", "pytorch")
        assert pytorch_stats["model_name"] == "gpt2"
        assert pytorch_stats["backend"] == "pytorch"
        assert pytorch_stats["sample_count"] == 1
        assert pytorch_stats["avg_latency"] == 2.0
        
        openvino_stats = monitor.get_model_stats("gpt2", "openvino")
        assert openvino_stats["model_name"] == "gpt2"
        assert openvino_stats["backend"] == "openvino"
        assert openvino_stats["sample_count"] == 1
        assert openvino_stats["avg_latency"] == 1.0
        
        # Get stats for model across all backends
        all_stats = monitor.get_model_stats("gpt2", backend=None)
        assert all_stats["model_name"] == "gpt2"
        assert all_stats["sample_count"] == 2  # Both backends
    
    def test_get_model_stats_empty(self):
        """Test getting stats for non-existent model."""
        monitor = PerformanceMonitor()
        
        stats = monitor.get_model_stats("nonexistent")
        assert stats == {}
    
    def test_get_all_records(self):
        """Test retrieving all inference records."""
        monitor = PerformanceMonitor()
        
        # Record multiple inferences
        monitor.record_inference("pytorch", "gpt2", 1.0, 100)
        monitor.record_inference("openvino", "gpt2", 0.5, 100)
        monitor.record_inference("pytorch", "t5", 2.0, 200)
        
        records = monitor.get_all_records()
        
        assert len(records) == 3
        assert all(isinstance(r, InferenceMetrics) for r in records)
        
        # Verify records are copies (not references)
        records[0].backend = "modified"
        original_records = monitor.get_all_records()
        assert original_records[0].backend != "modified"
    
    def test_clear(self):
        """Test clearing all metrics."""
        monitor = PerformanceMonitor()
        
        # Record some data
        monitor.record_inference("pytorch", "gpt2", 1.0, 100)
        monitor.record_inference("openvino", "gpt2", 0.5, 100)
        
        # Verify data exists
        assert len(monitor._inference_records) == 2
        assert len(monitor._metrics) > 0
        
        # Clear
        monitor.clear()
        
        # Verify data is cleared
        assert len(monitor._inference_records) == 0
        assert len(monitor._metrics) == 0
        assert len(monitor._backend_stats) == 0
    
    def test_get_summary(self):
        """Test getting performance summary."""
        monitor = PerformanceMonitor()
        
        # Record for multiple backends
        monitor.record_inference("pytorch", "gpt2", 2.0, 100)
        monitor.record_inference("pytorch", "t5", 1.5, 150)
        monitor.record_inference("openvino", "gpt2", 1.0, 100)
        
        summary = monitor.get_summary()
        
        assert "total_inferences" in summary
        assert summary["total_inferences"] == 3
        
        assert "backends" in summary
        assert "pytorch" in summary["backends"]
        assert "openvino" in summary["backends"]
        
        # PyTorch should have 2 samples
        assert summary["backends"]["pytorch"]["sample_count"] == 2
        
        # OpenVINO should have 1 sample
        assert summary["backends"]["openvino"]["sample_count"] == 1
    
    def test_throughput_calculation(self):
        """Test throughput is calculated correctly."""
        monitor = PerformanceMonitor()
        
        # Test various token counts and latencies
        test_cases = [
            (1.0, 100, 100.0),   # 100 tokens in 1s = 100 tok/s
            (2.0, 100, 50.0),    # 100 tokens in 2s = 50 tok/s
            (0.5, 100, 200.0),   # 100 tokens in 0.5s = 200 tok/s
            (1.0, 50, 50.0),     # 50 tokens in 1s = 50 tok/s
        ]
        
        for latency, tokens, expected_throughput in test_cases:
            monitor.clear()
            monitor.record_inference("pytorch", "model", latency, tokens)
            
            stats = monitor.get_backend_stats("pytorch")
            assert stats["avg_throughput"] == pytest.approx(expected_throughput)
    
    def test_multiple_models_same_backend(self):
        """Test tracking multiple models on same backend."""
        monitor = PerformanceMonitor()
        
        # Record for different models on same backend
        monitor.record_inference("pytorch", "gpt2", 1.0, 100)
        monitor.record_inference("pytorch", "t5", 2.0, 200)
        monitor.record_inference("pytorch", "bert", 0.5, 50)
        
        # Backend stats should aggregate all models
        backend_stats = monitor.get_backend_stats("pytorch")
        assert backend_stats["sample_count"] == 3
        
        # Each model should have separate stats
        gpt2_stats = monitor.get_model_stats("gpt2", "pytorch")
        assert gpt2_stats["sample_count"] == 1
        assert gpt2_stats["avg_latency"] == 1.0
        
        t5_stats = monitor.get_model_stats("t5", "pytorch")
        assert t5_stats["sample_count"] == 1
        assert t5_stats["avg_latency"] == 2.0
        
        bert_stats = monitor.get_model_stats("bert", "pytorch")
        assert bert_stats["sample_count"] == 1
        assert bert_stats["avg_latency"] == 0.5
    
    def test_stats_caching(self):
        """Test that backend stats are cached and invalidated correctly."""
        monitor = PerformanceMonitor()
        
        # Record inference
        monitor.record_inference("pytorch", "gpt2", 1.0, 100)
        
        # Get stats (should cache)
        stats1 = monitor.get_backend_stats("pytorch")
        assert stats1["sample_count"] == 1
        
        # Get stats again (should use cache)
        stats2 = monitor.get_backend_stats("pytorch")
        assert stats2 == stats1
        
        # Record another inference (should invalidate cache)
        monitor.record_inference("pytorch", "gpt2", 2.0, 100)
        
        # Get stats (should recalculate)
        stats3 = monitor.get_backend_stats("pytorch")
        assert stats3["sample_count"] == 2
        assert stats3 != stats1
    
    def test_percentile_calculations(self):
        """Test percentile calculations are correct."""
        monitor = PerformanceMonitor()
        
        # Record inferences with known distribution
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for latency in latencies:
            monitor.record_inference("pytorch", "model", latency, 100)
        
        stats = monitor.get_backend_stats("pytorch")
        
        # p50 should be median (5.5)
        assert stats["p50_latency"] == pytest.approx(5.5)
        
        # p95 should be 95th percentile (9.55)
        assert stats["p95_latency"] == pytest.approx(9.55)
    
    def test_zero_latency_handling(self):
        """Test handling of edge case with very small latency."""
        monitor = PerformanceMonitor()
        
        # Record with very small latency (should not cause division by zero)
        monitor.record_inference("pytorch", "model", 0.001, 100)
        
        stats = monitor.get_backend_stats("pytorch")
        assert stats["avg_throughput"] > 0
        assert stats["avg_throughput"] == pytest.approx(100000.0)  # 100 / 0.001
    
    def test_timestamp_recording(self):
        """Test that timestamps are recorded correctly."""
        monitor = PerformanceMonitor()
        
        before = time.time()
        monitor.record_inference("pytorch", "model", 1.0, 100)
        after = time.time()
        
        records = monitor.get_all_records()
        assert len(records) == 1
        
        # Timestamp should be between before and after
        assert before <= records[0].timestamp <= after
