"""
Property-based tests for PerformanceMonitor.

Feature: openvino-backend-integration
Property 10: Performance Metrics Recording

For any inference operation on any backend, the system should record latency,
throughput, and backend information in the performance monitoring system,
making these metrics accessible through the monitoring API.

Validates: Requirements 7.1, 7.2, 7.3, 7.4
"""

import pytest
from hypothesis import given, strategies as st, settings
from mm_orch.runtime.performance_monitor import PerformanceMonitor


# Strategies for generating test data
backend_strategy = st.sampled_from(["pytorch", "openvino"])
model_name_strategy = st.text(
    min_size=3, 
    max_size=50, 
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), 
        whitelist_characters="-_"
    )
).filter(lambda x: x not in ["latency", "throughput", "tokens"])  # Avoid metric keywords
latency_strategy = st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)
tokens_strategy = st.integers(min_value=1, max_value=1000)


@settings(max_examples=100, deadline=None)
@given(
    backend=backend_strategy,
    model_name=model_name_strategy,
    latency=latency_strategy,
    tokens=tokens_strategy
)
def test_property_10_metrics_recording(backend, model_name, latency, tokens):
    """
    Property 10: Performance Metrics Recording
    
    For any inference operation, the system should:
    1. Record latency for the backend
    2. Record throughput (tokens/second) for the backend
    3. Make metrics accessible through get_backend_stats
    4. Calculate correct throughput from latency and tokens
    
    Validates: Requirements 7.1, 7.2, 7.3, 7.4
    """
    monitor = PerformanceMonitor()
    
    # Record inference
    monitor.record_inference(
        backend=backend,
        model_name=model_name,
        latency=latency,
        tokens=tokens
    )
    
    # Verify metrics are accessible (Requirement 7.3)
    stats = monitor.get_backend_stats(backend)
    
    # Should have recorded the metrics
    assert stats is not None, "Stats should not be None"
    assert len(stats) > 0, "Stats should not be empty"
    assert "backend" in stats, "Stats should include backend name"
    assert stats["backend"] == backend, "Backend name should match"
    
    # Should have latency metrics (Requirement 7.1)
    assert "avg_latency" in stats, "Stats should include average latency"
    assert "p50_latency" in stats, "Stats should include median latency"
    assert "p95_latency" in stats, "Stats should include 95th percentile latency"
    assert stats["avg_latency"] > 0, "Average latency should be positive"
    assert stats["avg_latency"] == pytest.approx(latency, rel=1e-6), \
        "Average latency should match recorded latency for single sample"
    
    # Should have throughput metrics (Requirement 7.2)
    assert "avg_throughput" in stats, "Stats should include average throughput"
    assert stats["avg_throughput"] > 0, "Average throughput should be positive"
    
    # Throughput should be correctly calculated
    expected_throughput = tokens / latency
    assert stats["avg_throughput"] == pytest.approx(expected_throughput, rel=1e-6), \
        "Throughput should equal tokens divided by latency"
    
    # Should track sample count
    assert "sample_count" in stats, "Stats should include sample count"
    assert stats["sample_count"] == 1, "Sample count should be 1 for single inference"


@settings(max_examples=100, deadline=None)
@given(
    backend=backend_strategy,
    model_name=model_name_strategy,
    inferences=st.lists(
        st.tuples(latency_strategy, tokens_strategy),
        min_size=2,
        max_size=20
    )
)
def test_property_10_multiple_inferences(backend, model_name, inferences):
    """
    Property 10: Performance Metrics Recording (Multiple Inferences)
    
    For multiple inference operations, the system should:
    1. Aggregate metrics correctly
    2. Calculate accurate statistics (mean, percentiles)
    3. Track total sample count
    
    Validates: Requirements 7.1, 7.2, 7.3
    """
    monitor = PerformanceMonitor()
    
    # Record multiple inferences
    latencies = []
    throughputs = []
    for latency, tokens in inferences:
        monitor.record_inference(
            backend=backend,
            model_name=model_name,
            latency=latency,
            tokens=tokens
        )
        latencies.append(latency)
        throughputs.append(tokens / latency)
    
    # Get aggregated stats
    stats = monitor.get_backend_stats(backend)
    
    # Verify aggregation
    assert stats["sample_count"] == len(inferences), \
        "Sample count should match number of inferences"
    
    # Verify average latency is correct
    expected_avg_latency = sum(latencies) / len(latencies)
    assert stats["avg_latency"] == pytest.approx(expected_avg_latency, rel=1e-6), \
        "Average latency should be mean of all latencies"
    
    # Verify average throughput is correct
    expected_avg_throughput = sum(throughputs) / len(throughputs)
    assert stats["avg_throughput"] == pytest.approx(expected_avg_throughput, rel=1e-6), \
        "Average throughput should be mean of all throughputs"


@settings(max_examples=100, deadline=None)
@given(
    backend1=st.just("pytorch"),
    backend2=st.just("openvino"),
    model_name=model_name_strategy,
    pytorch_latency=latency_strategy,
    openvino_latency=latency_strategy,
    tokens=tokens_strategy
)
def test_property_10_backend_comparison(
    backend1, backend2, model_name, pytorch_latency, openvino_latency, tokens
):
    """
    Property 10: Performance Metrics Recording (Backend Comparison)
    
    When both backends are used, the system should:
    1. Track metrics separately for each backend
    2. Provide comparative metrics (Requirement 7.4)
    3. Calculate improvement ratios correctly
    
    Validates: Requirements 7.3, 7.4
    """
    monitor = PerformanceMonitor()
    
    # Record inference for both backends
    monitor.record_inference(
        backend=backend1,
        model_name=model_name,
        latency=pytorch_latency,
        tokens=tokens
    )
    
    monitor.record_inference(
        backend=backend2,
        model_name=model_name,
        latency=openvino_latency,
        tokens=tokens
    )
    
    # Get comparison (Requirement 7.4)
    comparison = monitor.compare_backends(backend1, backend2)
    
    # Should have comparison data
    assert comparison is not None, "Comparison should not be None"
    assert len(comparison) > 0, "Comparison should not be empty"
    
    # Should include improvement ratios
    assert "latency_improvement" in comparison, \
        "Comparison should include latency improvement"
    assert "throughput_improvement" in comparison, \
        "Comparison should include throughput improvement"
    
    # Should include stats for both backends
    assert f"{backend1}_stats" in comparison, \
        f"Comparison should include {backend1} stats"
    assert f"{backend2}_stats" in comparison, \
        f"Comparison should include {backend2} stats"
    
    # Verify improvement calculations
    expected_latency_improvement = pytorch_latency / openvino_latency
    assert comparison["latency_improvement"] == pytest.approx(
        expected_latency_improvement, rel=1e-6
    ), "Latency improvement should be ratio of latencies"
    
    pytorch_throughput = tokens / pytorch_latency
    openvino_throughput = tokens / openvino_latency
    expected_throughput_improvement = openvino_throughput / pytorch_throughput
    assert comparison["throughput_improvement"] == pytest.approx(
        expected_throughput_improvement, rel=1e-6
    ), "Throughput improvement should be ratio of throughputs"


@settings(max_examples=100, deadline=None)
@given(
    backend=backend_strategy,
    model_names=st.lists(
        model_name_strategy,
        min_size=2,
        max_size=5,
        unique=True
    ),
    latency=latency_strategy,
    tokens=tokens_strategy
)
def test_property_10_multiple_models(backend, model_names, latency, tokens):
    """
    Property 10: Performance Metrics Recording (Multiple Models)
    
    For multiple models on the same backend, the system should:
    1. Track metrics separately per model
    2. Aggregate correctly at backend level
    3. Provide model-specific statistics
    
    Validates: Requirements 7.1, 7.2, 7.3
    """
    monitor = PerformanceMonitor()
    
    # Record inference for each model
    for model_name in model_names:
        monitor.record_inference(
            backend=backend,
            model_name=model_name,
            latency=latency,
            tokens=tokens
        )
    
    # Backend stats should aggregate all models
    backend_stats = monitor.get_backend_stats(backend)
    assert backend_stats["sample_count"] == len(model_names), \
        "Backend sample count should include all models"
    
    # Each model should have its own stats
    for model_name in model_names:
        model_stats = monitor.get_model_stats(model_name, backend)
        assert model_stats is not None, f"Stats should exist for {model_name}"
        assert model_stats["sample_count"] == 1, \
            f"Model {model_name} should have 1 sample"
        assert model_stats["avg_latency"] == pytest.approx(latency, rel=1e-6), \
            f"Model {model_name} latency should match recorded value"


@settings(max_examples=50, deadline=None)
@given(
    backend=backend_strategy,
    model_name=model_name_strategy,
    latency=latency_strategy,
    tokens=tokens_strategy
)
def test_property_10_metrics_persistence(backend, model_name, latency, tokens):
    """
    Property 10: Performance Metrics Recording (Persistence)
    
    Recorded metrics should persist across multiple queries and should
    be accessible through different API methods.
    
    Validates: Requirements 7.3
    """
    monitor = PerformanceMonitor()
    
    # Record inference
    monitor.record_inference(
        backend=backend,
        model_name=model_name,
        latency=latency,
        tokens=tokens
    )
    
    # Metrics should be accessible through backend stats
    backend_stats = monitor.get_backend_stats(backend)
    assert backend_stats["sample_count"] == 1
    
    # Metrics should be accessible through model stats
    model_stats = monitor.get_model_stats(model_name, backend)
    assert model_stats["sample_count"] == 1
    
    # Metrics should be accessible through summary
    summary = monitor.get_summary()
    assert summary["total_inferences"] == 1
    assert backend in summary["backends"]
    
    # Metrics should be accessible through records
    records = monitor.get_all_records()
    assert len(records) == 1
    assert records[0].backend == backend
    assert records[0].model_name == model_name
    assert records[0].tokens_generated == tokens


def test_property_10_empty_backend():
    """
    Property 10: Performance Metrics Recording (Empty Backend)
    
    Querying stats for a backend with no recorded inferences should
    return an empty dictionary.
    
    Validates: Requirements 7.3
    """
    monitor = PerformanceMonitor()
    
    # Query non-existent backend
    stats = monitor.get_backend_stats("nonexistent")
    assert stats == {}, "Stats for non-existent backend should be empty dict"
    
    # Comparison with non-existent backend should return empty dict
    monitor.record_inference("pytorch", "model1", 1.0, 100)
    comparison = monitor.compare_backends("pytorch", "nonexistent")
    assert comparison == {}, "Comparison with non-existent backend should be empty dict"
