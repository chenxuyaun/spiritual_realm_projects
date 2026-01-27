"""OpenTelemetry distributed tracing for request flow analysis."""

import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Status, StatusCode, Span
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Define dummy types for type hints when OpenTelemetry is not available
    Span = Any

from mm_orch.logger import get_logger

logger = get_logger(__name__)


class OTelTracer:
    """
    OpenTelemetry tracer for distributed tracing.
    
    Provides context managers for creating spans at different levels
    (request, workflow, inference) and records span duration, status,
    and error information.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
    """
    
    def __init__(
        self,
        service_name: str = "muai-orchestration",
        endpoint: Optional[str] = None,
        enabled: bool = True,
        sample_rate: float = 1.0,
        use_console_exporter: bool = False
    ):
        """
        Initialize OpenTelemetry tracer.
        
        Args:
            service_name: Name of the service for trace identification
            endpoint: OTLP endpoint URL (e.g., "http://localhost:4317")
            enabled: Whether tracing is enabled (default: True)
            sample_rate: Sampling rate for traces (0.0 to 1.0, default: 1.0)
            use_console_exporter: Use console exporter for debugging (default: False)
            
        Requirements: 5.1, 5.7
        """
        self.service_name = service_name
        self.endpoint = endpoint
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._tracer = None
        
        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry not available, tracing disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )
            self.enabled = False
            return
        
        if not self.enabled:
            logger.info("OpenTelemetry tracing disabled by configuration")
            return
        
        # Initialize OpenTelemetry SDK
        self._init_tracer(use_console_exporter)
        logger.info(
            f"OTelTracer initialized for service '{service_name}' "
            f"with endpoint '{endpoint or 'console'}'"
        )
    
    def _init_tracer(self, use_console_exporter: bool = False):
        """
        Initialize OpenTelemetry tracer with configured exporter.
        
        Requirement 5.7: Configure exporter (OTLP, Jaeger, etc.)
        """
        try:
            # Create resource with service name
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Configure exporter
            if use_console_exporter:
                # Console exporter for debugging
                exporter = ConsoleSpanExporter()
                logger.info("Using ConsoleSpanExporter for tracing")
            elif self.endpoint:
                # OTLP exporter for production
                exporter = OTLPSpanExporter(endpoint=self.endpoint)
                logger.info(f"Using OTLPSpanExporter with endpoint: {self.endpoint}")
            else:
                # Default to console if no endpoint specified
                exporter = ConsoleSpanExporter()
                logger.warning(
                    "No OTLP endpoint specified, using ConsoleSpanExporter. "
                    "Set endpoint for production use."
                )
            
            # Add span processor
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            
            # Get tracer instance
            self._tracer = trace.get_tracer(__name__)
            
            logger.info("OpenTelemetry tracer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracer: {e}")
            logger.error(traceback.format_exc())
            self.enabled = False
    
    @contextmanager
    def trace_request(self, request_id: str, **attributes):
        """
        Create root span for a request.
        
        Args:
            request_id: Unique identifier for the request
            **attributes: Additional span attributes
            
        Yields:
            Span object for the request
            
        Requirement 5.1: Create root span for each request
        """
        if not self.enabled or not self._tracer:
            # Return a no-op context manager
            yield None
            return
        
        # Create root span
        with self._tracer.start_as_current_span(
            "request",
            attributes={
                "request.id": request_id,
                **attributes
            }
        ) as span:
            # Record start time
            span.set_attribute("request.start_time", datetime.utcnow().isoformat())
            
            try:
                yield span
                # Mark as successful if no exception
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                # Record error in span
                self._record_error_in_span(span, e)
                raise
            finally:
                # Record end time
                span.set_attribute("request.end_time", datetime.utcnow().isoformat())
    
    @contextmanager
    def trace_workflow(self, workflow_name: str, **attributes):
        """
        Create span for workflow execution.
        
        Args:
            workflow_name: Name of the workflow
            **attributes: Additional span attributes
            
        Yields:
            Span object for the workflow
            
        Requirement 5.2: Create child spans for workflow steps
        """
        if not self.enabled or not self._tracer:
            yield None
            return
        
        with self._tracer.start_as_current_span(
            f"workflow.{workflow_name}",
            attributes={
                "workflow.name": workflow_name,
                **attributes
            }
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                self._record_error_in_span(span, e)
                raise
    
    @contextmanager
    def trace_inference(
        self,
        model_name: str,
        engine: str,
        batch_size: Optional[int] = None,
        **attributes
    ):
        """
        Create span for model inference.
        
        Args:
            model_name: Name of the model
            engine: Inference engine used (vllm, deepspeed, onnx, pytorch)
            batch_size: Size of the batch (optional)
            **attributes: Additional span attributes
            
        Yields:
            Span object for the inference
            
        Requirement 5.3: Create inference spans with model metadata
        """
        if not self.enabled or not self._tracer:
            yield None
            return
        
        span_attributes = {
            "inference.model": model_name,
            "inference.engine": engine,
            **attributes
        }
        
        if batch_size is not None:
            span_attributes["inference.batch_size"] = batch_size
        
        with self._tracer.start_as_current_span(
            f"inference.{model_name}",
            attributes=span_attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                self._record_error_in_span(span, e)
                raise
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """
        Create span for a generic operation.
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional span attributes
            
        Yields:
            Span object for the operation
            
        Requirement 5.2: Create child spans for workflow steps
        """
        if not self.enabled or not self._tracer:
            yield None
            return
        
        with self._tracer.start_as_current_span(
            operation_name,
            attributes=attributes
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                self._record_error_in_span(span, e)
                raise
    
    def _record_error_in_span(self, span: Span, error: Exception):
        """
        Record error information in a span.
        
        Args:
            span: The span to record the error in
            error: The exception that occurred
            
        Requirement 5.5: Record error information in spans
        """
        if span is None:
            return
        
        try:
            # Set error status
            span.set_status(Status(StatusCode.ERROR, str(error)))
            
            # Record error attributes
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(error).__name__)
            span.set_attribute("error.message", str(error))
            span.set_attribute("error.stack_trace", traceback.format_exc())
            
            # Record as event
            span.add_event(
                "exception",
                attributes={
                    "exception.type": type(error).__name__,
                    "exception.message": str(error),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to record error in span: {e}")
    
    def record_event(
        self,
        span: Optional[Span],
        event_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Record an event in a span.
        
        Args:
            span: The span to record the event in
            event_name: Name of the event
            attributes: Event attributes (optional)
        """
        if not self.enabled or span is None:
            return
        
        try:
            span.add_event(event_name, attributes=attributes or {})
        except Exception as e:
            logger.error(f"Failed to record event in span: {e}")
    
    def set_span_attribute(
        self,
        span: Optional[Span],
        key: str,
        value: Any
    ):
        """
        Set an attribute on a span.
        
        Args:
            span: The span to set the attribute on
            key: Attribute key
            value: Attribute value
            
        Requirement 5.4: Record span duration and status
        """
        if not self.enabled or span is None:
            return
        
        try:
            span.set_attribute(key, value)
        except Exception as e:
            logger.error(f"Failed to set span attribute: {e}")
    
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.enabled
    
    def get_current_span(self) -> Optional[Span]:
        """
        Get the current active span.
        
        Returns:
            Current span or None if tracing is disabled
            
        Requirement 5.6: Propagate trace context across component boundaries
        """
        if not self.enabled or not OTEL_AVAILABLE:
            return None
        
        try:
            return trace.get_current_span()
        except Exception as e:
            logger.error(f"Failed to get current span: {e}")
            return None
    
    def get_trace_id(self) -> Optional[str]:
        """
        Get the current trace ID.
        
        Returns:
            Trace ID as hex string or None if no active trace
            
        Requirement 5.6: Propagate trace context across component boundaries
        """
        span = self.get_current_span()
        if span is None:
            return None
        
        try:
            trace_id = span.get_span_context().trace_id
            return format(trace_id, '032x') if trace_id else None
        except Exception as e:
            logger.error(f"Failed to get trace ID: {e}")
            return None
    
    def shutdown(self):
        """
        Shutdown the tracer and flush pending spans.
        
        Should be called before application exit to ensure all spans are exported.
        """
        if not self.enabled or not OTEL_AVAILABLE:
            return
        
        try:
            # Get the tracer provider and shutdown
            provider = trace.get_tracer_provider()
            if hasattr(provider, 'shutdown'):
                provider.shutdown()
                logger.info("OpenTelemetry tracer shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down tracer: {e}")
