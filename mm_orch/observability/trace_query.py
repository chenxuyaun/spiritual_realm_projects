"""Trace query and analysis tools."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from mm_orch.observability.tracer import WorkflowTrace, StepTrace


def load_traces(trace_file: str) -> List[WorkflowTrace]:
    """
    Load all traces from a JSONL file.

    Args:
        trace_file: Path to JSONL trace file

    Returns:
        List of WorkflowTrace objects
    """
    traces = []
    trace_path = Path(trace_file)

    if not trace_path.exists():
        return traces

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Reconstruct StepTrace objects
                steps = [StepTrace(**step_data) for step_data in data.get("steps", [])]
                data["steps"] = steps
                traces.append(WorkflowTrace(**data))
            except Exception as e:
                print(f"Failed to parse trace line: {e}")
                continue

    return traces


def filter_by_workflow(traces: List[WorkflowTrace], workflow_name: str) -> List[WorkflowTrace]:
    """
    Filter traces by workflow name.

    Args:
        traces: List of traces to filter
        workflow_name: Name of workflow to match

    Returns:
        Filtered list of traces
    """
    return [t for t in traces if t.chosen_workflow == workflow_name]


def filter_by_date_range(
    traces: List[WorkflowTrace],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[WorkflowTrace]:
    """
    Filter traces by date range.

    Args:
        traces: List of traces to filter
        start_date: Start of date range (inclusive), None for no lower bound
        end_date: End of date range (inclusive), None for no upper bound

    Returns:
        Filtered list of traces
    """
    filtered = []

    for trace in traces:
        trace_dt = datetime.fromtimestamp(trace.timestamp)

        if start_date and trace_dt < start_date:
            continue
        if end_date and trace_dt > end_date:
            continue

        filtered.append(trace)

    return filtered


def filter_by_success(traces: List[WorkflowTrace], success: bool = True) -> List[WorkflowTrace]:
    """
    Filter traces by success status.

    Args:
        traces: List of traces to filter
        success: True for successful traces, False for failed traces

    Returns:
        Filtered list of traces
    """
    return [t for t in traces if t.success == success]


def filter_by_mode(traces: List[WorkflowTrace], mode: str) -> List[WorkflowTrace]:
    """
    Filter traces by execution mode.

    Args:
        traces: List of traces to filter
        mode: Mode to match (e.g., "chat", "default")

    Returns:
        Filtered list of traces
    """
    return [t for t in traces if t.mode == mode]


def aggregate_latency_stats(traces: List[WorkflowTrace]) -> Dict[str, float]:
    """
    Calculate latency statistics across traces.

    Args:
        traces: List of traces to analyze

    Returns:
        Dictionary with min, max, avg, median latency in ms
    """
    if not traces:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}

    latencies = []
    for trace in traces:
        total_latency = sum(step.latency_ms for step in trace.steps)
        latencies.append(total_latency)

    latencies.sort()

    return {
        "min": min(latencies),
        "max": max(latencies),
        "avg": sum(latencies) / len(latencies),
        "median": latencies[len(latencies) // 2],
    }


def aggregate_vram_stats(traces: List[WorkflowTrace]) -> Dict[str, float]:
    """
    Calculate VRAM usage statistics across traces.

    Args:
        traces: List of traces to analyze

    Returns:
        Dictionary with min, max, avg, median VRAM in MB
    """
    if not traces:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}

    vram_peaks = []
    for trace in traces:
        max_vram = max((step.vram_peak_mb for step in trace.steps), default=0)
        vram_peaks.append(max_vram)

    vram_peaks.sort()

    return {
        "min": min(vram_peaks),
        "max": max(vram_peaks),
        "avg": sum(vram_peaks) / len(vram_peaks),
        "median": vram_peaks[len(vram_peaks) // 2],
    }


def aggregate_success_rate(traces: List[WorkflowTrace]) -> float:
    """
    Calculate success rate across traces.

    Args:
        traces: List of traces to analyze

    Returns:
        Success rate as a float between 0 and 1
    """
    if not traces:
        return 0.0

    successful = sum(1 for t in traces if t.success)
    return successful / len(traces)


def aggregate_quality_signals(traces: List[WorkflowTrace]) -> Dict[str, Any]:
    """
    Aggregate quality signals across traces.

    Args:
        traces: List of traces to analyze

    Returns:
        Dictionary with aggregated quality metrics
    """
    if not traces:
        return {
            "avg_citation_count": 0,
            "avg_answer_length": 0,
            "has_search_rate": 0,
            "has_citations_rate": 0,
            "failure_rate": 0,
        }

    total_citations = 0
    total_answer_length = 0
    has_search_count = 0
    has_citations_count = 0
    failure_count = 0

    for trace in traces:
        signals = trace.quality_signals
        total_citations += signals.get("citation_count", 0)
        total_answer_length += signals.get("answer_length", 0)
        has_search_count += 1 if signals.get("has_search", False) else 0
        has_citations_count += 1 if signals.get("has_citations", False) else 0
        failure_count += 1 if signals.get("failure_occurred", False) else 0

    n = len(traces)

    return {
        "avg_citation_count": total_citations / n,
        "avg_answer_length": total_answer_length / n,
        "has_search_rate": has_search_count / n,
        "has_citations_rate": has_citations_count / n,
        "failure_rate": failure_count / n,
    }


def group_by_workflow(traces: List[WorkflowTrace]) -> Dict[str, List[WorkflowTrace]]:
    """
    Group traces by workflow name.

    Args:
        traces: List of traces to group

    Returns:
        Dictionary mapping workflow names to lists of traces
    """
    grouped: Dict[str, List[WorkflowTrace]] = {}

    for trace in traces:
        workflow = trace.chosen_workflow
        if workflow not in grouped:
            grouped[workflow] = []
        grouped[workflow].append(trace)

    return grouped


def find_failed_steps(traces: List[WorkflowTrace]) -> List[Dict[str, Any]]:
    """
    Find all failed steps across traces.

    Args:
        traces: List of traces to analyze

    Returns:
        List of dictionaries with failed step information
    """
    failed_steps = []

    for trace in traces:
        for step in trace.steps:
            if not step.success:
                failed_steps.append(
                    {
                        "request_id": trace.request_id,
                        "workflow": trace.chosen_workflow,
                        "step_name": step.step_name,
                        "error": step.error,
                        "error_type": step.error_type,
                        "timestamp": trace.timestamp,
                    }
                )

    return failed_steps


def custom_filter(
    traces: List[WorkflowTrace], predicate: Callable[[WorkflowTrace], bool]
) -> List[WorkflowTrace]:
    """
    Filter traces using a custom predicate function.

    Args:
        traces: List of traces to filter
        predicate: Function that returns True for traces to keep

    Returns:
        Filtered list of traces
    """
    return [t for t in traces if predicate(t)]
