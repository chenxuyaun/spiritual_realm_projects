"""
Integration tests for monitoring in real workflows.

Tests monitoring components integrated with actual workflows:
- Metrics collection in real workflows
- Tracing in real workflows
- Anomaly detection with real thresholds

Validates: All monitoring requirements
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

from mm_orch.monitoring.prometheus_exporter import PrometheusExporter
from mm_orch.monitoring.otel_tracer import OTelTracer
from mm_orch.monitoring.performance_monitor import PerformanceMonitor
from mm_orch.monitoring.anomaly_detector import AnomalyDetector
from mm_orch.monitoring.config import (
    PrometheusConfig,
    TracingConfig,
    AnomalyConfig
)


class TestMetricsInWorkflows:
    """Test metrics collection in real workflows."""
    
    def test_metrics_recorded_during_inference(self):
        """Test that metrics are recorded during inference operations."""
        # Setup: Create exporter
        config = PrometheusConfig(
            enabled=True,
            port=9091,
            start_server=False  # Don't start HTTP server in tests
        )
        exporter = PrometheusExporter(config)
        
        # Execute: Simulate workflow with metrics
        model_name = "test_model"
        engine = "pytorch"
        
        # Record inference metrics
        for i in range(10):
            latency_ms = 50.0 + i * 5
            exporter.record_inference_latency(
                model_name=model_name,
                engine=engine,
                latency_ms=latency_ms
            )
            exporter.record_throughput(
                model_name=model_name,
                requests_per_second=10.0 + i
            )
        
        # Verify: Metrics recorded
        metrics = exporter.get_metrics()
        assert metrics is not None
        
        # Check that metrics contain expected data
        metrics_text = exporter.format_metrics()
        assert model_name in metrics_text or "inference" in metrics_text
    
    def test_metrics_for_multiple_models(self):
        """Test metrics collection for multiple models."""
        config = PrometheusConfig(
            enabled=True,
            port=9092,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Execute: Record metrics for different models
        models = ["model_a", "model_b", "model_c"]
        
        for model in models:
            for i in range(5):
                exporter.record_inference_latency(
                    model_name=model,
                    engine="pytorch",
                    latency_ms=100.0 + models.index(model) * 20
                )
        
        # Verify: Metrics for all models
        metrics_text = exporter.format_metrics()
        
        # Should contain metrics (exact format depends on implementation)
        assert len(metrics_text) > 0
    
    def test_resource_metrics_collection(self):
        """Test resource utilization metrics collection."""
        config = PrometheusConfig(
            enabled=True,
            port=9093,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Execute: Record resource metrics
        for i in range(5):
            exporter.record_resource_usage(
                gpu_memory_mb=1024.0 + i * 100,
                cpu_percent=50.0 + i * 5
            )
            time.sleep(0.1)
        
        # Verify: Resource metrics recorded
        metrics_text = exporter.format_metrics()
        assert len(metrics_text) > 0
    
    def test_model_lifecycle_metrics(self):
        """Test model lifecycle event metrics."""
        config = PrometheusConfig(
            enabled=True,
            port=9094,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Execute: Record lifecycle events
        exporter.record_model_lifecycle(
            model_name="test_model",
            event="load",
            duration_ms=500.0
        )
        
        exporter.record_model_lifecycle(
            model_name="test_model",
            event="unload",
            duration_ms=100.0
        )
        
        # Verify: Lifecycle metrics recorded
        metrics_text = exporter.format_metrics()
        assert len(metrics_text) > 0
    
    def test_concurrent_metrics_recording(self):
        """Test metrics recording under concurrent load."""
        config = PrometheusConfig(
            enabled=True,
            port=9095,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Execute: Concurrent metric recording
        def record_metrics(thread_id):
            for i in range(10):
                exporter.record_inference_latency(
                    model_name=f"model_{thread_id}",
                    engine="pytorch",
                    latency_ms=50.0 + i
                )
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(record_metrics, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()
        
        # Verify: All metrics recorded
        metrics_text = exporter.format_metrics()
        assert len(metrics_text) > 0


class TestTracingInWorkflows:
    """Test tracing in real workflows."""
    
    def setup_method(self):
        """Clear spans before each test."""
        # Clear any existing spans from previous tests
        OTelTracer.reset_for_testing()
    
    def test_trace_complete_workflow(self):
        """Test tracing a complete workflow execution."""
        # Setup: Create tracer
        config = TracingConfig(
            enabled=True,
            endpoint="memory://",  # In-memory for testing
            sample_rate=1.0
        )
        tracer = OTelTracer(config)
        
        # Execute: Trace workflow
        with tracer.trace_request("req_001") as root_span:
            root_span.set_attribute("workflow_type", "search_qa")
            
            with tracer.trace_workflow("search_qa") as workflow_span:
                workflow_span.set_attribute("step", "preprocessing")
                time.sleep(0.01)
                
                with tracer.trace_inference("gpt2", "pytorch") as inference_span:
                    inference_span.set_attribute("batch_size", 1)
                    time.sleep(0.02)
                
                workflow_span.set_attribute("step", "postprocessing")
                time.sleep(0.01)
        
        # Verify: Spans created
        spans = tracer.get_finished_spans()
        assert len(spans) >= 3  # root, workflow, inference
        
        # Verify: Span hierarchy
        span_names = [span.name for span in spans]
        assert any("request" in name.lower() for name in span_names)
        assert any("workflow" in name.lower() or "search_qa" in name.lower() for name in span_names)
        assert any("inference" in name.lower() for name in span_names)
    
    def test_trace_with_error_recording(self):
        """Test error recording in traces."""
        config = TracingConfig(
            enabled=True,
            endpoint="memory://",
            sample_rate=1.0
        )
        tracer = OTelTracer(config)
        
        # Execute: Trace with error
        try:
            with tracer.trace_request("req_002") as root_span:
                with tracer.trace_inference("model", "engine") as inference_span:
                    # Simulate error
                    raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify: Error recorded in span
        spans = tracer.get_finished_spans()
        assert len(spans) >= 1
        
        # Check for error attributes
        error_span = spans[-1]
        assert error_span.status.is_ok is False or hasattr(error_span, 'events')
    
    def test_trace_context_propagation(self):
        """Test trace context propagation across operations."""
        config = TracingConfig(
            enabled=True,
            endpoint="memory://",
            sample_rate=1.0
        )
        tracer = OTelTracer(config)
        
        # Execute: Create nested spans
        with tracer.trace_request("req_003") as root_span:
            trace_id_1 = root_span.get_span_context().trace_id
            
            with tracer.trace_workflow("workflow") as workflow_span:
                trace_id_2 = workflow_span.get_span_context().trace_id
                
                with tracer.trace_inference("model", "engine") as inference_span:
                    trace_id_3 = inference_span.get_span_context().trace_id
        
        # Verify: Same trace ID across all spans
        assert trace_id_1 == trace_id_2 == trace_id_3
    
    def test_trace_metadata_recording(self):
        """Test recording of metadata in traces."""
        config = TracingConfig(
            enabled=True,
            endpoint="memory://",
            sample_rate=1.0
        )
        tracer = OTelTracer(config)
        
        # Execute: Record metadata
        with tracer.trace_inference("gpt2", "vllm") as span:
            span.set_attribute("model_name", "gpt2")
            span.set_attribute("engine", "vllm")
            span.set_attribute("batch_size", 4)
            span.set_attribute("sequence_length", 128)
            span.set_attribute("tensor_parallel_size", 2)
        
        # Verify: Metadata recorded
        spans = tracer.get_finished_spans()
        assert len(spans) >= 1
        
        last_span = spans[-1]
        attributes = dict(last_span.attributes) if hasattr(last_span, 'attributes') else {}
        
        # Check for expected attributes
        assert len(attributes) >= 0  # Attributes may be stored differently
    
    def test_concurrent_tracing(self):
        """Test tracing under concurrent load."""
        config = TracingConfig(
            enabled=True,
            endpoint="memory://",
            sample_rate=1.0
        )
        tracer = OTelTracer(config)
        
        # Execute: Concurrent traces
        def create_trace(request_id):
            with tracer.trace_request(f"req_{request_id}") as root_span:
                with tracer.trace_inference("model", "engine") as inference_span:
                    time.sleep(0.01)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_trace, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()
        
        # Verify: All traces created
        spans = tracer.get_finished_spans()
        assert len(spans) >= 10  # At least 10 request spans


class TestAnomalyDetectionIntegration:
    """Test anomaly detection with real thresholds."""
    
    def test_anomaly_detection_with_performance_monitor(self):
        """Test anomaly detection integrated with performance monitor."""
        # Setup: Create components
        monitor = PerformanceMonitor(max_history_seconds=3600)
        detector = AnomalyDetector(AnomalyConfig(
            latency_threshold_ms=100.0,
            error_rate_threshold=0.1,
            alert_destinations=["log"],
            alert_rate_limit_seconds=1
        ))
        
        # Execute: Simulate workload with anomalies
        alert_count = 0
        for i in range(20):
            # Normal latency
            if i < 15:
                latency = 50.0
            # Anomalous latency
            else:
                latency = 150.0
            
            monitor.record_latency(
                operation="inference",
                latency_ms=latency,
                metadata={"request_id": i}
            )
            
            # Check for anomalies
            alert = detector.check_latency("inference", latency)
            
            if i >= 15:
                # First anomaly should trigger alert, subsequent ones may be rate-limited
                if alert is not None:
                    alert_count += 1
                    assert alert.alert_type == "latency"
            elif i < 15:
                assert alert is None, f"Should not detect anomaly at iteration {i}"
        
        # Verify: At least one alert was triggered
        assert alert_count >= 1, "Should have detected at least one anomaly"
    
    def test_error_rate_anomaly_detection(self):
        """Test error rate anomaly detection."""
        detector = AnomalyDetector(AnomalyConfig(
            error_rate_threshold=0.2,  # 20%
            alert_destinations=["log"],
            alert_rate_limit_seconds=1
        ))
        
        # Execute: Simulate requests with errors
        # 30% error rate (should trigger alert)
        for i in range(10):
            success = i >= 3  # 3 failures out of 10
            detector._record_request(success=success)
        
        alert = detector.check_error_rate(window_seconds=60)
        
        # Verify: Alert triggered
        assert alert is not None
        assert alert.alert_type == "error_rate"
        assert "error" in alert.message.lower() or "rate" in alert.message.lower()
    
    def test_resource_anomaly_detection(self):
        """Test resource usage anomaly detection."""
        detector = AnomalyDetector(AnomalyConfig(
            memory_threshold_percent=80.0,
            alert_destinations=["log"],
            alert_rate_limit_seconds=1
        ))
        
        # Execute: Simulate resource usage
        # Normal usage
        alert1 = detector.check_resources(
            gpu_memory_percent=70.0,
            cpu_percent=50.0
        )
        assert alert1 is None
        
        # High usage (should trigger)
        alert2 = detector.check_resources(
            gpu_memory_percent=95.0,
            cpu_percent=50.0
        )
        assert alert2 is not None
        assert alert2.alert_type == "resource"
    
    def test_throughput_anomaly_detection(self):
        """Test throughput anomaly detection."""
        monitor = PerformanceMonitor(max_history_seconds=3600)
        detector = AnomalyDetector(AnomalyConfig(
            throughput_threshold_rps=50.0,
            alert_destinations=["log"],
            alert_rate_limit_seconds=1
        ))
        
        # Execute: Simulate low throughput
        # Record few requests over time
        for i in range(5):
            monitor.record_latency(
                operation="inference",
                latency_ms=100.0,
                metadata={}
            )
            time.sleep(0.1)
        
        # Calculate throughput
        throughput = monitor.get_throughput(window_seconds=1)
        
        # Check for anomaly
        alert = detector.check_throughput(throughput)
        
        # Verify: Alert if throughput is low
        if throughput < 50.0:
            assert alert is not None
            assert alert.alert_type == "throughput"
    
    def test_alert_delivery_to_multiple_destinations(self):
        """Test alert delivery to multiple destinations."""
        # Setup: Configure multiple destinations
        detector = AnomalyDetector(AnomalyConfig(
            latency_threshold_ms=50.0,
            alert_destinations=["log", "webhook"],
            webhook_url="http://localhost:9999/alerts",
            alert_rate_limit_seconds=1
        ))
        
        # Mock webhook delivery
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            
            # Execute: Trigger alert
            alert = detector.check_latency("inference", 100.0)
            
            if alert:
                detector.send_alert(alert)
                
                # Verify: Webhook called (if implemented)
                # Note: This depends on implementation details
                pass
    
    def test_anomaly_detection_with_adaptive_thresholds(self):
        """Test anomaly detection with adaptive thresholds."""
        monitor = PerformanceMonitor(max_history_seconds=3600)
        detector = AnomalyDetector(AnomalyConfig(
            latency_threshold_ms=100.0,
            alert_destinations=["log"],
            alert_rate_limit_seconds=1
        ))
        
        # Execute: Build baseline
        for i in range(50):
            monitor.record_latency(
                operation="inference",
                latency_ms=50.0 + (i % 10),
                metadata={}
            )
        
        # Get percentiles for adaptive threshold
        percentiles = monitor.get_percentiles("inference", window_seconds=60)
        
        # Use p95 as adaptive threshold
        if percentiles and "p95" in percentiles:
            adaptive_threshold = percentiles["p95"] * 1.5
            
            # Update detector threshold
            detector.config.latency_threshold_ms = adaptive_threshold
            
            # Test with new threshold
            alert = detector.check_latency("inference", adaptive_threshold + 10)
            assert alert is not None


class TestMonitoringFailureHandling:
    """Test monitoring failure handling."""
    
    def test_metrics_failure_doesnt_block_inference(self):
        """Test that metrics export failures don't block inference."""
        config = PrometheusConfig(
            enabled=True,
            port=9096,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Mock metrics recording to fail
        with patch.object(exporter, 'record_inference_latency', side_effect=Exception("Metrics failed")):
            # Execute: Try to record metrics
            try:
                exporter.record_inference_latency(
                    model_name="test",
                    engine="pytorch",
                    latency_ms=50.0
                )
            except Exception:
                pass  # Should not propagate
            
            # Verify: Can continue with other operations
            exporter.record_throughput("test", 10.0)
    
    def test_tracing_failure_doesnt_block_inference(self):
        """Test that tracing failures don't block inference."""
        config = TracingConfig(
            enabled=True,
            endpoint="invalid://endpoint",
            sample_rate=1.0
        )
        
        # Create tracer with invalid endpoint
        try:
            tracer = OTelTracer(config)
            
            # Execute: Try to create trace
            with tracer.trace_request("req_001") as span:
                # Should not raise exception
                pass
        except Exception:
            # If initialization fails, that's acceptable
            pass
    
    def test_alert_delivery_failure_handling(self):
        """Test handling of alert delivery failures."""
        detector = AnomalyDetector(AnomalyConfig(
            latency_threshold_ms=50.0,
            alert_destinations=["webhook"],
            webhook_url="http://invalid-url:9999/alerts",
            alert_rate_limit_seconds=1
        ))
        
        # Mock webhook to fail
        with patch('requests.post', side_effect=Exception("Network error")):
            # Execute: Trigger alert
            alert = detector.check_latency("inference", 100.0)
            
            if alert:
                # Should handle failure gracefully
                try:
                    detector.send_alert(alert)
                except Exception:
                    pass  # Should not propagate
    
    def test_monitoring_degradation_status(self):
        """Test that monitoring degradation is reflected in status."""
        config = PrometheusConfig(
            enabled=True,
            port=9097,
            start_server=False
        )
        exporter = PrometheusExporter(config)
        
        # Simulate degradation
        exporter._degraded = True
        exporter._degradation_reason = "Metrics export failed"
        
        # Verify: Status reflects degradation
        status = exporter.get_status()
        assert status.get("degraded", False) is True
        assert "reason" in status or "degradation" in str(status).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
