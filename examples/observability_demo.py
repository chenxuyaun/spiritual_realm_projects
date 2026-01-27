#!/usr/bin/env python3
"""Demo script showing observability components in action."""

import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.observability import (
    Tracer, StepTrace, WorkflowTrace,
    QualitySignals, CostStatsManager
)
from mm_orch.observability import trace_query


def create_sample_trace(request_id: str, workflow: str, success: bool = True) -> WorkflowTrace:
    """Create a sample workflow trace."""
    return WorkflowTrace(
        request_id=request_id,
        conversation_id=None,
        question=f"Sample question for {workflow}",
        chosen_workflow=workflow,
        router_version="v1",
        mode="default",
        turn_index=None,
        steps=[
            StepTrace(
                step_name="web_search",
                start_time=1000.0,
                end_time=1001.5,
                latency_ms=1500.0,
                vram_peak_mb=150,
                model_loads=0,
                success=True
            ),
            StepTrace(
                step_name="summarize",
                start_time=1001.5,
                end_time=1003.0,
                latency_ms=1500.0,
                vram_peak_mb=300,
                model_loads=1,
                success=success
            )
        ],
        urls_used=["http://example.com/article1", "http://example.com/article2"],
        final_answer="This is a sample answer with citations.",
        quality_signals={},
        cost_stats={},
        timestamp=1000.0,
        success=success
    )


def demo_tracer():
    """Demonstrate tracer functionality."""
    print("=" * 60)
    print("TRACER DEMO")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "traces.jsonl"
        tracer = Tracer(str(trace_file))
        
        # Create and write sample traces
        print("\n1. Writing traces...")
        for i in range(3):
            trace = create_sample_trace(
                f"req-{i}",
                "search_qa" if i % 2 == 0 else "rag_qa",
                success=(i != 2)  # Last one fails
            )
            tracer.write_workflow_trace(trace)
            print(f"   ✓ Wrote trace {i+1}: {trace.request_id}")
        
        # Load traces
        print("\n2. Loading traces...")
        traces = trace_query.load_traces(str(trace_file))
        print(f"   ✓ Loaded {len(traces)} traces")
        
        # Show trace details
        print("\n3. Trace details:")
        for trace in traces:
            status = "✓" if trace.success else "✗"
            print(f"   {status} {trace.request_id}: {trace.chosen_workflow}")
            print(f"      Steps: {len(trace.steps)}, URLs: {len(trace.urls_used)}")


def demo_quality_signals():
    """Demonstrate quality signals calculation."""
    print("\n" + "=" * 60)
    print("QUALITY SIGNALS DEMO")
    print("=" * 60)
    
    # Create sample trace
    trace = create_sample_trace("req-quality", "search_qa", success=True)
    
    # Create sample state
    state = {
        "question": "What is Python?",
        "final_answer": "Python is a high-level programming language known for its simplicity.",
        "citations": ["http://example.com/python", "http://example.com/programming"]
    }
    
    # Calculate quality signals
    print("\n1. Calculating quality signals...")
    signals = QualitySignals.from_trace(trace, state)
    
    print("\n2. Quality metrics:")
    print(f"   Citation count: {signals.citation_count}")
    print(f"   Answer length: {signals.answer_length} characters")
    print(f"   Has search: {signals.has_search}")
    print(f"   Has citations: {signals.has_citations}")
    print(f"   Has structure: {signals.has_structure}")
    print(f"   Failure occurred: {signals.failure_occurred}")


def demo_cost_stats():
    """Demonstrate cost statistics aggregation."""
    print("\n" + "=" * 60)
    print("COST STATISTICS DEMO")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "costs.json"
        manager = CostStatsManager(str(stats_file))
        
        # Update with multiple traces
        print("\n1. Updating cost statistics...")
        workflows = ["search_qa", "search_qa", "rag_qa", "search_qa"]
        for i, workflow in enumerate(workflows):
            trace = create_sample_trace(f"req-{i}", workflow)
            manager.update(trace)
            print(f"   ✓ Updated stats for {workflow}")
        
        # Show statistics
        print("\n2. Cost statistics:")
        for workflow_name in ["search_qa", "rag_qa"]:
            stats = manager.get(workflow_name)
            print(f"\n   {workflow_name}:")
            print(f"      Executions: {stats.execution_count}")
            print(f"      Avg latency: {stats.avg_latency_ms:.1f}ms")
            print(f"      Avg VRAM: {stats.avg_vram_mb:.1f}MB")
            print(f"      Avg model loads: {stats.avg_model_loads:.1f}")
            print(f"      Success rate: {stats.success_rate:.1%}")
        
        # Save and reload
        print("\n3. Persisting statistics...")
        manager.save()
        print(f"   ✓ Saved to {stats_file}")
        
        manager2 = CostStatsManager(str(stats_file))
        print(f"   ✓ Reloaded {len(manager2.get_all_stats())} workflow stats")


def demo_trace_query():
    """Demonstrate trace query tools."""
    print("\n" + "=" * 60)
    print("TRACE QUERY DEMO")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "traces.jsonl"
        tracer = Tracer(str(trace_file))
        
        # Create diverse traces
        print("\n1. Creating test traces...")
        traces_data = [
            ("req-1", "search_qa", True),
            ("req-2", "search_qa", True),
            ("req-3", "rag_qa", True),
            ("req-4", "search_qa", False),
            ("req-5", "lesson_pack", True),
        ]
        
        for req_id, workflow, success in traces_data:
            trace = create_sample_trace(req_id, workflow, success)
            tracer.write_workflow_trace(trace)
        
        print(f"   ✓ Created {len(traces_data)} traces")
        
        # Load and query
        print("\n2. Querying traces...")
        all_traces = trace_query.load_traces(str(trace_file))
        print(f"   Total traces: {len(all_traces)}")
        
        # Filter by workflow
        search_traces = trace_query.filter_by_workflow(all_traces, "search_qa")
        print(f"   search_qa traces: {len(search_traces)}")
        
        # Filter by success
        successful = trace_query.filter_by_success(all_traces, success=True)
        failed = trace_query.filter_by_success(all_traces, success=False)
        print(f"   Successful: {len(successful)}, Failed: {len(failed)}")
        
        # Aggregate statistics
        print("\n3. Aggregate statistics:")
        latency_stats = trace_query.aggregate_latency_stats(all_traces)
        print(f"   Latency - Min: {latency_stats['min']:.1f}ms, "
              f"Max: {latency_stats['max']:.1f}ms, "
              f"Avg: {latency_stats['avg']:.1f}ms")
        
        vram_stats = trace_query.aggregate_vram_stats(all_traces)
        print(f"   VRAM - Min: {vram_stats['min']:.1f}MB, "
              f"Max: {vram_stats['max']:.1f}MB, "
              f"Avg: {vram_stats['avg']:.1f}MB")
        
        success_rate = trace_query.aggregate_success_rate(all_traces)
        print(f"   Success rate: {success_rate:.1%}")
        
        # Group by workflow
        print("\n4. Traces by workflow:")
        grouped = trace_query.group_by_workflow(all_traces)
        for workflow, traces in grouped.items():
            print(f"   {workflow}: {len(traces)} traces")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("OBSERVABILITY COMPONENTS DEMO")
    print("=" * 60)
    
    demo_tracer()
    demo_quality_signals()
    demo_cost_stats()
    demo_trace_query()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nAll observability components demonstrated successfully!")
    print("See docs/phase_b4_implementation_summary.md for details.\n")


if __name__ == "__main__":
    main()
