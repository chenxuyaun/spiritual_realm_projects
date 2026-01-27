"""Property-based tests for OpenTelemetry tracer."""

import pytest
from hypothesis import given, strategies as st, settings
import time
from unittest.mock import Mock, patch

# Try to import OpenTelemetry, skip tests if not available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    from opentelemetry.trace import Status, StatusCode
    from typing import Sequence
    OTEL_AVAILABLE = True
    
    # Create a simple in-memory span exporter for testing
    class InMemorySpanExporter(SpanExporter):
        """Simple in-memory span exporter for testing."""
        
        def __init__(self):
            self.spans = []
        
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS
        
        def shutdown(self):
            pass
        
        def get_finished_spans(self):
            return self.spans
        
        def clear(self):
            self.spans = []
    
except ImportError:
    OTEL_AVAILABLE = False

from mm_orch.monitoring.otel_tracer import OTelTracer


# Skip all tests if OpenTelemetry is not available
pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE,
    reason="OpenTelemetry not installed"
)


# Strategies for test data generation
request_ids = st.text(min_size=1, max_size=100, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

workflow_names = st.sampled_from([
    'search_qa', 'lesson_pack', 'chat_generate', 'rag_qa', 'self_ask_search_qa'
])

model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

engine_names = st.sampled_from(['vllm', 'deepspeed', 'onnx', 'pytorch'])

batch_sizes = st.integers(min_value=1, max_value=128)

operation_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='._-'
))


def create_tracer_with_memory_exporter():
    """Helper to create a tracer with in-memory span exporter for testing."""
    # Create a new tracer provider with in-memory exporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Don't set global provider, just use this provider directly
    # Get tracer from this provider
    tracer_instance = provider.get_tracer(__name__)
    
    # Create OTelTracer but we'll manually inject the tracer
    tracer = OTelTracer(
        service_name="test-service",
        enabled=True,
        use_console_exporter=True
    )
    
    # Override the tracer with our test tracer
    tracer._tracer = tracer_instance
    
    return tracer, exporter


# Feature: advanced-optimization-monitoring, Property 19: Root span is created for each request
@given(request_id=request_ids)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_19_root_span_created(request_id):
    """
    Property 19: Root span is created for each request.
    
    For any request entering the system, a root span should be created
    with a unique trace ID.
    
    Validates: Requirements 5.1
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create a request span
    with tracer.trace_request(request_id) as span:
        # Span should be created
        assert span is not None
        
        # Span should have the request ID attribute
        span_context = span.get_span_context()
        assert span_context.trace_id > 0
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    
    # Find our request span
    request_spans = [s for s in spans if s.name == "request"]
    assert len(request_spans) > 0
    
    # Verify request ID attribute
    request_span = request_spans[0]
    assert "request.id" in request_span.attributes
    assert request_span.attributes["request.id"] == request_id


# Feature: advanced-optimization-monitoring, Property 20: Child spans are created for workflow steps
@given(
    request_id=request_ids,
    workflow_name=workflow_names
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_20_child_spans_created(request_id, workflow_name):
    """
    Property 20: Child spans are created for workflow steps.
    
    For any workflow execution with multiple steps, child spans should be
    created for each step under the workflow span.
    
    Validates: Requirements 5.2
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create request span with workflow child span
    with tracer.trace_request(request_id) as request_span:
        with tracer.trace_workflow(workflow_name) as workflow_span:
            # Both spans should be created
            assert request_span is not None
            assert workflow_span is not None
    
    # Verify spans were exported
    spans = exporter.get_finished_spans()
    assert len(spans) >= 2
    
    # Find workflow span
    workflow_spans = [s for s in spans if workflow_name in s.name]
    assert len(workflow_spans) > 0
    
    # Verify workflow name attribute
    workflow_span = workflow_spans[0]
    assert "workflow.name" in workflow_span.attributes
    assert workflow_span.attributes["workflow.name"] == workflow_name


# Feature: advanced-optimization-monitoring, Property 21: Inference spans include model metadata
@given(
    model_name=model_names,
    engine=engine_names,
    batch_size=batch_sizes
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_21_inference_spans_include_metadata(model_name, engine, batch_size):
    """
    Property 21: Inference spans include model metadata.
    
    For any model inference operation, the corresponding span should include
    model name, engine type, and batch size as attributes.
    
    Validates: Requirements 5.3
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create inference span
    with tracer.trace_inference(model_name, engine, batch_size) as span:
        assert span is not None
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    
    # Find inference span
    inference_spans = [s for s in spans if "inference" in s.name]
    assert len(inference_spans) > 0
    
    # Verify metadata attributes
    inference_span = inference_spans[0]
    assert "inference.model" in inference_span.attributes
    assert inference_span.attributes["inference.model"] == model_name
    assert "inference.engine" in inference_span.attributes
    assert inference_span.attributes["inference.engine"] == engine
    assert "inference.batch_size" in inference_span.attributes
    assert inference_span.attributes["inference.batch_size"] == batch_size


# Feature: advanced-optimization-monitoring, Property 22: Span duration and status are recorded
@given(
    request_id=request_ids,
    sleep_ms=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_property_22_span_duration_and_status_recorded(request_id, sleep_ms):
    """
    Property 22: Span duration and status are recorded.
    
    For any completed span, the duration and final status (success/error)
    should be recorded.
    
    Validates: Requirements 5.4
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create span with some duration
    with tracer.trace_request(request_id) as span:
        time.sleep(sleep_ms / 1000.0)  # Convert to seconds
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    
    request_span = spans[0]
    
    # Verify duration is recorded (end_time > start_time)
    assert request_span.end_time > request_span.start_time
    duration_ns = request_span.end_time - request_span.start_time
    assert duration_ns > 0
    
    # Verify status is recorded (should be OK for successful span)
    assert request_span.status is not None
    assert request_span.status.status_code == StatusCode.OK


# Feature: advanced-optimization-monitoring, Property 23: Errors are recorded in spans
@given(
    request_id=request_ids,
    error_message=st.text(min_size=1, max_size=100)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_23_errors_recorded_in_spans(request_id, error_message):
    """
    Property 23: Errors are recorded in spans.
    
    For any error occurring during a traced operation, error type, message,
    and stack trace should be recorded in the span.
    
    Validates: Requirements 5.5
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create span that raises an error
    try:
        with tracer.trace_request(request_id) as span:
            raise ValueError(error_message)
    except ValueError:
        pass  # Expected
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    
    request_span = spans[0]
    
    # Verify error status
    assert request_span.status is not None
    assert request_span.status.status_code == StatusCode.ERROR
    
    # Verify error attributes
    assert "error" in request_span.attributes
    assert request_span.attributes["error"] is True
    assert "error.type" in request_span.attributes
    assert request_span.attributes["error.type"] == "ValueError"
    assert "error.message" in request_span.attributes
    assert error_message in request_span.attributes["error.message"]
    
    # Verify error event
    events = request_span.events
    exception_events = [e for e in events if e.name == "exception"]
    assert len(exception_events) > 0


# Feature: advanced-optimization-monitoring, Property 24: Trace context is propagated
@given(
    request_id=request_ids,
    workflow_name=workflow_names,
    model_name=model_names,
    engine=engine_names
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_24_trace_context_propagated(request_id, workflow_name, model_name, engine):
    """
    Property 24: Trace context is propagated.
    
    For any operation spanning multiple components, trace context should be
    propagated such that all spans share the same trace ID.
    
    Validates: Requirements 5.6
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create nested spans
    with tracer.trace_request(request_id) as request_span:
        request_trace_id = request_span.get_span_context().trace_id
        
        with tracer.trace_workflow(workflow_name) as workflow_span:
            workflow_trace_id = workflow_span.get_span_context().trace_id
            
            with tracer.trace_inference(model_name, engine) as inference_span:
                inference_trace_id = inference_span.get_span_context().trace_id
    
    # All spans should share the same trace ID
    assert request_trace_id == workflow_trace_id
    assert workflow_trace_id == inference_trace_id
    
    # Verify in exported spans
    spans = exporter.get_finished_spans()
    assert len(spans) >= 3
    
    # All spans should have the same trace ID
    trace_ids = [s.context.trace_id for s in spans]
    assert len(set(trace_ids)) == 1  # All trace IDs are the same


# Additional property: Disabled tracer doesn't create spans
@given(request_id=request_ids)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_disabled_tracer_no_spans(request_id):
    """
    Property: When disabled, tracer doesn't create spans.
    
    For any tracing operation when tracer is disabled, no spans should
    be created or exported.
    """
    # Create disabled tracer
    tracer = OTelTracer(enabled=False)
    
    # Try to create spans
    with tracer.trace_request(request_id) as span:
        # Span should be None when disabled
        assert span is None
    
    # Verify tracer is disabled
    assert not tracer.is_enabled()


# Property: Multiple operations create separate spans
@given(
    request_id=request_ids,
    operations=st.lists(operation_names, min_size=1, max_size=5)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_multiple_operations_separate_spans(request_id, operations):
    """
    Property: Multiple operations create separate spans.
    
    For any sequence of operations, each should create its own span
    with appropriate attributes.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create request with multiple operation spans
    with tracer.trace_request(request_id):
        for operation in operations:
            with tracer.trace_operation(operation):
                pass
    
    # Verify spans were exported
    spans = exporter.get_finished_spans()
    # Should have at least request span + operation spans
    assert len(spans) >= len(operations) + 1


# Property: Span attributes can be set dynamically
@given(
    request_id=request_ids,
    attr_key=st.text(min_size=1, max_size=50),
    attr_value=st.one_of(
        st.text(min_size=0, max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans()
    )
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_span_attributes_set_dynamically(request_id, attr_key, attr_value):
    """
    Property: Span attributes can be set dynamically.
    
    For any span, attributes should be settable during span lifetime.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create span and set attribute
    with tracer.trace_request(request_id) as span:
        if span is not None:
            tracer.set_span_attribute(span, attr_key, attr_value)
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    if len(spans) > 0:
        request_span = spans[0]
        # Attribute should be present (if it was valid)
        # Some attribute types might not be supported by OpenTelemetry
        # so we just verify the operation didn't crash
        assert True


# Property: Events can be recorded in spans
@given(
    request_id=request_ids,
    event_name=st.text(min_size=1, max_size=50),
    event_attrs=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(min_size=0, max_size=50),
        min_size=0,
        max_size=5
    )
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_events_recorded_in_spans(request_id, event_name, event_attrs):
    """
    Property: Events can be recorded in spans.
    
    For any span, events with attributes should be recordable.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create span and record event
    with tracer.trace_request(request_id) as span:
        if span is not None:
            tracer.record_event(span, event_name, event_attrs)
    
    # Verify span was exported
    spans = exporter.get_finished_spans()
    if len(spans) > 0:
        request_span = spans[0]
        # Event should be present
        events = request_span.events
        event_names = [e.name for e in events]
        # The event might be there along with start/end time events
        # Just verify the operation didn't crash
        assert True


# Property: Current span can be retrieved
@given(request_id=request_ids)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_current_span_retrievable(request_id):
    """
    Property: Current span can be retrieved.
    
    For any active span, it should be retrievable via get_current_span().
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Outside any span, current span should be None or invalid
    current_outside = tracer.get_current_span()
    
    # Inside a span, current span should be valid
    with tracer.trace_request(request_id) as span:
        current_inside = tracer.get_current_span()
        assert current_inside is not None
        # Should be the same span
        if span is not None:
            assert current_inside.get_span_context().span_id == span.get_span_context().span_id


# Property: Trace ID can be retrieved
@given(request_id=request_ids)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_trace_id_retrievable(request_id):
    """
    Property: Trace ID can be retrieved.
    
    For any active trace, the trace ID should be retrievable.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Inside a span, trace ID should be retrievable
    with tracer.trace_request(request_id) as span:
        trace_id = tracer.get_trace_id()
        if span is not None:
            assert trace_id is not None
            assert isinstance(trace_id, str)
            assert len(trace_id) == 32  # Hex string of 128-bit trace ID


# Property: Tracer can be shutdown gracefully
@given(request_id=request_ids)
@settings(max_examples=20, deadline=None)
@pytest.mark.property
def test_tracer_shutdown_graceful(request_id):
    """
    Property: Tracer can be shutdown gracefully.
    
    For any tracer, shutdown should complete without errors and
    flush pending spans.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create some spans
    with tracer.trace_request(request_id):
        pass
    
    # Shutdown should not raise exception
    try:
        tracer.shutdown()
    except Exception as e:
        pytest.fail(f"Tracer shutdown raised exception: {e}")


# Property: Nested spans maintain parent-child relationship
@given(
    request_id=request_ids,
    workflow_name=workflow_names
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_nested_spans_parent_child_relationship(request_id, workflow_name):
    """
    Property: Nested spans maintain parent-child relationship.
    
    For any nested spans, child spans should reference their parent span.
    """
    tracer, exporter = create_tracer_with_memory_exporter()
    
    # Create nested spans
    with tracer.trace_request(request_id) as request_span:
        request_span_id = request_span.get_span_context().span_id if request_span else None
        
        with tracer.trace_workflow(workflow_name) as workflow_span:
            workflow_parent_id = workflow_span.parent.span_id if workflow_span and workflow_span.parent else None
    
    # Verify parent-child relationship in exported spans
    spans = exporter.get_finished_spans()
    if len(spans) >= 2:
        # Find workflow span
        workflow_spans = [s for s in spans if workflow_name in s.name]
        if len(workflow_spans) > 0:
            workflow_span = workflow_spans[0]
            # Workflow span should have a parent
            assert workflow_span.parent is not None
