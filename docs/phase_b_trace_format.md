# Phase B Trace Format Documentation

## Overview

Traces provide comprehensive execution logs in JSONL format for observability, debugging, and router training.

## Trace Structure

### WorkflowTrace

```python
@dataclass
class WorkflowTrace:
    request_id: str  # Unique request identifier
    conversation_id: Optional[str]  # For multi-turn conversations
    question: str  # User input
    chosen_workflow: str  # Selected workflow name
    router_version: str  # Router version used (v1/v2/v3)
    mode: str  # Execution mode (default/chat)
    turn_index: Optional[int]  # Turn number in conversation
    steps: List[StepTrace]  # Step execution traces
    urls_used: List[str]  # URLs accessed
    final_answer: str  # Generated response
    quality_signals: Dict[str, Any]  # Quality metrics
    cost_stats: Dict[str, float]  # Cost metrics
    timestamp: float  # Unix timestamp
    success: bool  # Overall success status
```

### StepTrace

```python
@dataclass
class StepTrace:
    step_name: str  # Step identifier
    start_time: float  # Unix timestamp
    end_time: float  # Unix timestamp
    latency_ms: float  # Execution time
    vram_peak_mb: int  # Peak VRAM usage
    model_loads: int  # Number of models loaded
    success: bool  # Step success status
    error: Optional[str]  # Error message if failed
    error_type: Optional[str]  # Exception type
    stack_trace: Optional[str]  # Full stack trace
```

## Example Trace

```json
{
  "request_id": "req-20260127-001",
  "conversation_id": "conv-123",
  "question": "What is Python programming?",
  "chosen_workflow": "search_qa",
  "router_version": "v3",
  "mode": "chat",
  "turn_index": 5,
  "steps": [
    {
      "step_name": "web_search",
      "start_time": 1706313600.0,
      "end_time": 1706313601.5,
      "latency_ms": 1500.0,
      "vram_peak_mb": 0,
      "model_loads": 0,
      "success": true,
      "error": null,
      "error_type": null,
      "stack_trace": null
    },
    {
      "step_name": "fetch_url",
      "start_time": 1706313601.5,
      "end_time": 1706313603.0,
      "latency_ms": 1500.0,
      "vram_peak_mb": 0,
      "model_loads": 0,
      "success": true,
      "error": null,
      "error_type": null,
      "stack_trace": null
    },
    {
      "step_name": "summarize",
      "start_time": 1706313603.0,
      "end_time": 1706313605.0,
      "latency_ms": 2000.0,
      "vram_peak_mb": 500,
      "model_loads": 1,
      "success": true,
      "error": null,
      "error_type": null,
      "stack_trace": null
    },
    {
      "step_name": "answer_generate",
      "start_time": 1706313605.0,
      "end_time": 1706313607.0,
      "latency_ms": 2000.0,
      "vram_peak_mb": 800,
      "model_loads": 1,
      "success": true,
      "error": null,
      "error_type": null,
      "stack_trace": null
    }
  ],
  "urls_used": [
    "https://python.org/tutorial",
    "https://python.org/features"
  ],
  "final_answer": "Python is a high-level programming language...",
  "quality_signals": {
    "citation_count": 2,
    "answer_length": 250,
    "has_search": true,
    "has_citations": true,
    "has_structure": false,
    "failure_occurred": false
  },
  "cost_stats": {
    "total_latency_ms": 7000.0,
    "peak_vram_mb": 800,
    "total_model_loads": 2
  },
  "timestamp": 1706313600.0,
  "success": true
}
```

## Querying Traces

### Load Traces

```python
from mm_orch.observability.trace_query import load_traces

traces = load_traces("data/traces/traces.jsonl")
```

### Filter by Workflow

```python
from mm_orch.observability.trace_query import filter_by_workflow

search_qa_traces = filter_by_workflow(traces, "search_qa")
```

### Filter by Success

```python
from mm_orch.observability.trace_query import filter_by_success

successful = filter_by_success(traces, success=True)
failed = filter_by_success(traces, success=False)
```

### Aggregate Statistics

```python
from mm_orch.observability.trace_query import (
    aggregate_latency_stats,
    aggregate_quality_signals
)

latency_stats = aggregate_latency_stats(traces)
# Returns: {"avg": 5000.0, "min": 1000.0, "max": 10000.0}

quality_stats = aggregate_quality_signals(traces)
# Returns: {"avg_citations": 2.5, "avg_answer_length": 300}
```

## Use Cases

### 1. Router Training

Extract questions and workflows:

```python
training_data = [
    (trace.question, trace.chosen_workflow)
    for trace in traces
    if trace.success
]
```

### 2. Performance Analysis

Identify slow workflows:

```python
workflow_latencies = {}
for trace in traces:
    workflow = trace.chosen_workflow
    latency = sum(s.latency_ms for s in trace.steps)
    
    if workflow not in workflow_latencies:
        workflow_latencies[workflow] = []
    workflow_latencies[workflow].append(latency)

for workflow, latencies in workflow_latencies.items():
    avg = sum(latencies) / len(latencies)
    print(f"{workflow}: {avg:.0f}ms")
```

### 3. Error Analysis

Find common failures:

```python
from collections import Counter

error_types = Counter()
for trace in traces:
    if not trace.success:
        for step in trace.steps:
            if not step.success and step.error_type:
                error_types[step.error_type] += 1

print("Most common errors:")
for error_type, count in error_types.most_common(5):
    print(f"  {error_type}: {count}")
```

### 4. Quality Monitoring

Track quality over time:

```python
import matplotlib.pyplot as plt

timestamps = [t.timestamp for t in traces]
citations = [t.quality_signals.get("citation_count", 0) for t in traces]

plt.plot(timestamps, citations)
plt.xlabel("Time")
plt.ylabel("Citation Count")
plt.title("Quality Over Time")
plt.show()
```

## Best Practices

1. **Rotate trace files**: Prevent files from growing too large
2. **Index by timestamp**: Enable time-range queries
3. **Compress old traces**: Save storage space
4. **Backup regularly**: Traces are valuable training data
5. **Monitor file size**: Set up alerts for large files

## See Also

- [Quality Signals Guide](./phase_b_quality_signals.md)
- [Cost Tracking Guide](./phase_b_cost_tracking.md)
- [Router Training Guide](./phase_b_router_training_guide.md)
