"""Integration tests for observability components."""

import pytest
import tempfile
import json
from pathlib import Path

from mm_orch.observability.tracer import Tracer, StepTrace, WorkflowTrace
from mm_orch.observability.quality_signals import QualitySignals
from mm_orch.observability.cost_stats import WorkflowCostStats, CostStatsManager
from mm_orch.observability.trace_query import (
    load_traces, filter_by_workflow, filter_by_success,
    aggregate_latency_stats, aggregate_quality_signals
)


class TestObservabilityIntegration:
    """Test observability components working together."""
    
    def test_tracer_writes_and_loads(self):
        """Test that tracer writes traces that can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"
            tracer = Tracer(str(trace_file))
            
            # Create a sample trace
            trace = WorkflowTrace(
                request_id="test-123",
                conversation_id=None,
                question="What is Python?",
                chosen_workflow="search_qa",
                router_version="v1",
                mode="default",
                turn_index=None,
                steps=[
                    StepTrace(
                        step_name="web_search",
                        start_time=1000.0,
                        end_time=1001.0,
                        latency_ms=1000.0,
                        vram_peak_mb=100,
                        model_loads=0,
                        success=True
                    )
                ],
                urls_used=["http://example.com"],
                final_answer="Python is a programming language",
                quality_signals={"citation_count": 1},
                cost_stats={"latency": 1000.0},
                timestamp=1000.0,
                success=True
            )
            
            # Write trace
            tracer.write_workflow_trace(trace)
            
            # Load traces
            loaded_traces = load_traces(str(trace_file))
            
            assert len(loaded_traces) == 1
            assert loaded_traces[0].request_id == "test-123"
            assert loaded_traces[0].question == "What is Python?"
    
    def test_quality_signals_from_trace(self):
        """Test quality signals calculation from trace."""
        trace = WorkflowTrace(
            request_id="test-456",
            conversation_id=None,
            question="Explain AI",
            chosen_workflow="search_qa",
            router_version="v1",
            mode="default",
            turn_index=None,
            steps=[
                StepTrace(
                    step_name="web_search",
                    start_time=1000.0,
                    end_time=1001.0,
                    latency_ms=1000.0,
                    vram_peak_mb=100,
                    model_loads=0,
                    success=True
                ),
                StepTrace(
                    step_name="summarize",
                    start_time=1001.0,
                    end_time=1002.0,
                    latency_ms=1000.0,
                    vram_peak_mb=200,
                    model_loads=1,
                    success=True
                )
            ],
            urls_used=["http://example.com"],
            final_answer="AI is artificial intelligence",
            quality_signals={},
            cost_stats={},
            timestamp=1000.0,
            success=True
        )
        
        state = {
            "question": "Explain AI",
            "final_answer": "AI is artificial intelligence",
            "citations": ["http://example.com"]
        }
        
        signals = QualitySignals.from_trace(trace, state)
        
        assert signals.citation_count == 1
        assert signals.answer_length == len("AI is artificial intelligence")
        assert signals.has_search is True
        assert signals.has_citations is True
        assert signals.failure_occurred is False
    
    def test_cost_stats_update_and_persistence(self):
        """Test cost statistics update and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_file = Path(tmpdir) / "costs.json"
            manager = CostStatsManager(str(stats_file))
            
            # Create sample traces
            trace1 = WorkflowTrace(
                request_id="test-1",
                conversation_id=None,
                question="Q1",
                chosen_workflow="search_qa",
                router_version="v1",
                mode="default",
                turn_index=None,
                steps=[
                    StepTrace(
                        step_name="step1",
                        start_time=1000.0,
                        end_time=1001.0,
                        latency_ms=1000.0,
                        vram_peak_mb=100,
                        model_loads=1,
                        success=True
                    )
                ],
                urls_used=[],
                final_answer="Answer 1",
                quality_signals={},
                cost_stats={},
                timestamp=1000.0,
                success=True
            )
            
            trace2 = WorkflowTrace(
                request_id="test-2",
                conversation_id=None,
                question="Q2",
                chosen_workflow="search_qa",
                router_version="v1",
                mode="default",
                turn_index=None,
                steps=[
                    StepTrace(
                        step_name="step1",
                        start_time=2000.0,
                        end_time=2002.0,
                        latency_ms=2000.0,
                        vram_peak_mb=200,
                        model_loads=2,
                        success=True
                    )
                ],
                urls_used=[],
                final_answer="Answer 2",
                quality_signals={},
                cost_stats={},
                timestamp=2000.0,
                success=True
            )
            
            # Update stats
            manager.update(trace1)
            manager.update(trace2)
            
            # Check stats
            stats = manager.get("search_qa")
            assert stats.execution_count == 2
            assert stats.avg_latency_ms == 1500.0  # (1000 + 2000) / 2
            assert stats.avg_vram_mb == 150.0  # (100 + 200) / 2
            assert stats.avg_model_loads == 1.5  # (1 + 2) / 2
            
            # Save and reload
            manager.save()
            
            manager2 = CostStatsManager(str(stats_file))
            stats2 = manager2.get("search_qa")
            
            assert stats2.execution_count == 2
            assert stats2.avg_latency_ms == 1500.0
    
    def test_trace_query_filtering(self):
        """Test trace query filtering functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_file = Path(tmpdir) / "traces.jsonl"
            tracer = Tracer(str(trace_file))
            
            # Create multiple traces
            traces = [
                WorkflowTrace(
                    request_id=f"test-{i}",
                    conversation_id=None,
                    question=f"Question {i}",
                    chosen_workflow="search_qa" if i % 2 == 0 else "rag_qa",
                    router_version="v1",
                    mode="default",
                    turn_index=None,
                    steps=[
                        StepTrace(
                            step_name="step1",
                            start_time=1000.0 + i,
                            end_time=1001.0 + i,
                            latency_ms=1000.0,
                            vram_peak_mb=100,
                            model_loads=0,
                            success=(i % 3 != 0)  # Every 3rd fails
                        )
                    ],
                    urls_used=[],
                    final_answer=f"Answer {i}",
                    quality_signals={},
                    cost_stats={},
                    timestamp=1000.0 + i,
                    success=(i % 3 != 0)
                )
                for i in range(10)
            ]
            
            # Write all traces
            for trace in traces:
                tracer.write_workflow_trace(trace)
            
            # Load and filter
            loaded = load_traces(str(trace_file))
            assert len(loaded) == 10
            
            # Filter by workflow
            search_qa_traces = filter_by_workflow(loaded, "search_qa")
            assert len(search_qa_traces) == 5
            
            # Filter by success
            successful = filter_by_success(loaded, success=True)
            assert len(successful) == 6  # 0,3,6,9 fail (4 failures), so 6 succeed
            
            # Aggregate stats
            latency_stats = aggregate_latency_stats(loaded)
            assert latency_stats["avg"] == 1000.0
            assert latency_stats["min"] == 1000.0
            assert latency_stats["max"] == 1000.0
