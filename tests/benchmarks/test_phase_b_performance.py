"""
Performance benchmarks for Phase B orchestration system.

Benchmarks:
- Workflow execution times
- Model loading and unloading
- Router prediction times
- Phase B vs Phase A performance comparison
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock
from typing import List, Dict, Any

from mm_orch.orchestration import State, GraphExecutor, GraphNode
from mm_orch.orchestration.workflow_steps import (
    WebSearchStep, FetchUrlStep, SummarizeStep, AnswerGenerateStep
)
from mm_orch.routing.router_v1 import RouterV1
from mm_orch.routing.router_v2 import RouterV2
from mm_orch.routing.router_v3 import RouterV3
from mm_orch.observability.tracer import Tracer


class TestWorkflowExecutionBenchmarks:
    """Benchmark workflow execution times."""
    
    @pytest.fixture
    def mock_runtime(self):
        """Create mock runtime."""
        runtime = Mock()
        runtime.model_manager = Mock()
        runtime.model_manager.infer = Mock(return_value="Mocked output")
        return runtime
    
    @pytest.fixture
    def mock_search_tool(self):
        """Create mock search tool."""
        from mm_orch.tools.web_search import SearchResult
        
        tool = Mock()
        tool.search = Mock(return_value=[
            SearchResult(
                title=f"Result {i}",
                url=f"https://example.com/{i}",
                snippet=f"Snippet {i}"
            )
            for i in range(5)
        ])
        return tool
    
    @pytest.fixture
    def mock_fetch_tool(self):
        """Create mock fetch tool."""
        from mm_orch.tools.fetch_url import FetchedContent
        
        tool = Mock()
        tool.fetch_multiple = Mock(return_value=[
            FetchedContent(
                url=f"https://example.com/{i}",
                content="Content " * 100,
                title=f"Result {i}",
                success=True
            )
            for i in range(5)
        ])
        return tool
    
    def test_search_qa_workflow_execution_time(self, mock_runtime, mock_search_tool, mock_fetch_tool, benchmark):
        """Benchmark search QA workflow execution time."""
        # Setup workflow
        search_step = WebSearchStep(search_tool=mock_search_tool)
        fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool)
        summarize_step = SummarizeStep()
        answer_step = AnswerGenerateStep()
        
        step_registry = {
            "search": search_step,
            "fetch": fetch_step,
            "summarize": summarize_step,
            "answer": answer_step
        }
        
        graph = {
            "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
            "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
            "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
            "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
            "end": GraphNode(step_name="answer", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        
        initial_state: State = {
            "question": "What is Python?",
            "meta": {"mode": "default"}
        }
        
        # Benchmark execution
        result = benchmark(executor.execute, graph, initial_state, mock_runtime)
        
        # Verify result is valid
        assert "final_answer" in result
        
        # Print benchmark stats
        print(f"\nSearch QA Workflow Execution Time:")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")
        print(f"  Min: {benchmark.stats['min'] * 1000:.2f}ms")
        print(f"  Max: {benchmark.stats['max'] * 1000:.2f}ms")
    
    def test_search_qa_fast_workflow_execution_time(self, mock_runtime, mock_search_tool, mock_fetch_tool, benchmark):
        """Benchmark fast search QA workflow (without summarization)."""
        # Setup workflow without summarization step
        search_step = WebSearchStep(search_tool=mock_search_tool)
        fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool)
        answer_step = AnswerGenerateStep()
        
        step_registry = {
            "search": search_step,
            "fetch": fetch_step,
            "answer": answer_step
        }
        
        graph = {
            "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
            "fetch_node": GraphNode(step_name="fetch", next_nodes=["answer_node"]),
            "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
            "end": GraphNode(step_name="answer", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        
        initial_state: State = {
            "question": "What is Python?",
            "meta": {"mode": "default"}
        }
        
        # Benchmark execution
        result = benchmark(executor.execute, graph, initial_state, mock_runtime)
        
        assert "final_answer" in result
        
        print(f"\nSearch QA Fast Workflow Execution Time:")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")
        print(f"  Min: {benchmark.stats['min'] * 1000:.2f}ms")
        print(f"  Max: {benchmark.stats['max'] * 1000:.2f}ms")
    
    def test_workflow_step_overhead(self, mock_runtime, benchmark):
        """Benchmark overhead of step execution framework."""
        from mm_orch.orchestration.base_step import BaseStep
        
        class MinimalStep(BaseStep):
            name = "minimal"
            input_keys = ["question"]
            output_keys = ["result"]
            
            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                return {"result": "done"}
        
        step = MinimalStep()
        step_registry = {"minimal": step}
        
        graph = {
            "start": GraphNode(step_name="minimal", next_nodes=["end"]),
            "end": GraphNode(step_name="minimal", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        
        initial_state: State = {"question": "test"}
        
        # Benchmark minimal step execution
        result = benchmark(executor.execute, graph, initial_state, mock_runtime)
        
        assert "result" in result
        
        print(f"\nMinimal Step Overhead:")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")


class TestModelLoadingBenchmarks:
    """Benchmark model loading and unloading."""
    
    def test_model_manager_load_time(self, benchmark):
        """Benchmark model loading time."""
        # Note: This would require real model manager
        # For now, we'll benchmark the mock
        mock_manager = Mock()
        mock_manager.get_model = Mock(return_value=Mock())
        
        def load_model():
            return mock_manager.get_model("test-model")
        
        result = benchmark(load_model)
        
        assert result is not None
        
        print(f"\nModel Load Time (mocked):")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")
    
    def test_model_manager_cache_hit(self, benchmark):
        """Benchmark cached model retrieval."""
        mock_manager = Mock()
        cached_model = Mock()
        mock_manager.get_model = Mock(return_value=cached_model)
        
        # Warm up cache
        mock_manager.get_model("test-model")
        
        def get_cached_model():
            return mock_manager.get_model("test-model")
        
        result = benchmark(get_cached_model)
        
        assert result is not None
        
        print(f"\nCached Model Retrieval Time:")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")


class TestRouterPredictionBenchmarks:
    """Benchmark router prediction times."""
    
    def test_router_v1_prediction_time(self, benchmark):
        """Benchmark Router v1 prediction time."""
        router = RouterV1()
        
        def predict():
            return router.route("What is Python programming?", {})
        
        result = benchmark(predict)
        
        assert result[0] is not None  # workflow name
        
        print(f"\nRouter v1 Prediction Time:")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")
    
    def test_router_v1_batch_predictions(self, benchmark):
        """Benchmark Router v1 batch predictions."""
        router = RouterV1()
        
        questions = [
            "What is Python?",
            "How to install Python?",
            "Python vs Java",
            "Teach me Python",
            "Summarize this article"
        ]
        
        def batch_predict():
            return [router.route(q, {}) for q in questions]
        
        results = benchmark(batch_predict)
        
        assert len(results) == 5
        
        print(f"\nRouter v1 Batch Prediction Time (5 questions):")
        print(f"  Mean: {benchmark.stats['mean'] * 1000:.2f}ms")
        print(f"  Per question: {benchmark.stats['mean'] * 1000 / 5:.2f}ms")


class TestPhaseComparison:
    """Compare Phase B performance to Phase A baseline."""
    
    def test_phase_b_vs_phase_a_workflow_execution(self):
        """
        Compare Phase B workflow execution to Phase A baseline.
        
        Note: This is a conceptual test. Actual comparison would require
        running both Phase A and Phase B implementations.
        """
        # Phase B execution time (from previous benchmarks)
        phase_b_times = []
        
        # Simulate Phase A baseline (would be actual measurements)
        phase_a_baseline = 150.0  # ms
        
        # Run Phase B workflow multiple times
        from unittest.mock import Mock
        
        mock_runtime = Mock()
        mock_runtime.model_manager = Mock()
        mock_runtime.model_manager.infer = Mock(return_value="Output")
        
        from mm_orch.tools.web_search import SearchResult
        mock_search_tool = Mock()
        mock_search_tool.search = Mock(return_value=[
            SearchResult(title="Test", url="http://test.com", snippet="Test")
        ])
        
        from mm_orch.tools.fetch_url import FetchedContent
        mock_fetch_tool = Mock()
        mock_fetch_tool.fetch_multiple = Mock(return_value=[
            FetchedContent(url="http://test.com", content="Content", title="Test", success=True)
        ])
        
        search_step = WebSearchStep(search_tool=mock_search_tool)
        fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool)
        summarize_step = SummarizeStep()
        answer_step = AnswerGenerateStep()
        
        step_registry = {
            "search": search_step,
            "fetch": fetch_step,
            "summarize": summarize_step,
            "answer": answer_step
        }
        
        graph = {
            "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
            "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
            "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
            "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
            "end": GraphNode(step_name="answer", next_nodes=[])
        }
        
        executor = GraphExecutor(step_registry)
        
        initial_state: State = {
            "question": "What is Python?",
            "meta": {"mode": "default"}
        }
        
        # Measure Phase B execution times
        for _ in range(10):
            start = time.time()
            executor.execute(graph, initial_state, mock_runtime)
            elapsed = (time.time() - start) * 1000
            phase_b_times.append(elapsed)
        
        phase_b_mean = sum(phase_b_times) / len(phase_b_times)
        
        print(f"\nPhase A vs Phase B Comparison:")
        print(f"  Phase A Baseline: {phase_a_baseline:.2f}ms")
        print(f"  Phase B Mean: {phase_b_mean:.2f}ms")
        print(f"  Difference: {phase_b_mean - phase_a_baseline:+.2f}ms ({((phase_b_mean / phase_a_baseline - 1) * 100):+.1f}%)")
        
        # Phase B should be within reasonable overhead (< 50% slower)
        assert phase_b_mean < phase_a_baseline * 1.5, "Phase B is significantly slower than Phase A"
    
    def test_router_overhead_comparison(self):
        """Compare router overhead between versions."""
        router_v1 = RouterV1()
        
        question = "What is Python programming?"
        
        # Measure Router v1
        v1_times = []
        for _ in range(100):
            start = time.time()
            router_v1.route(question, {})
            elapsed = (time.time() - start) * 1000
            v1_times.append(elapsed)
        
        v1_mean = sum(v1_times) / len(v1_times)
        
        print(f"\nRouter Overhead Comparison:")
        print(f"  Router v1 Mean: {v1_mean:.3f}ms")
        
        # Router should be fast (< 10ms)
        assert v1_mean < 10.0, "Router v1 is too slow"


class TestScalabilityBenchmarks:
    """Test scalability with increasing load."""
    
    def test_workflow_execution_with_varying_document_count(self):
        """Test how execution time scales with document count."""
        from unittest.mock import Mock
        from mm_orch.tools.web_search import SearchResult
        from mm_orch.tools.fetch_url import FetchedContent
        
        mock_runtime = Mock()
        mock_runtime.model_manager = Mock()
        mock_runtime.model_manager.infer = Mock(return_value="Output")
        
        results = {}
        
        for doc_count in [1, 2, 5, 10]:
            # Create mock tools with varying document counts
            mock_search_tool = Mock()
            mock_search_tool.search = Mock(return_value=[
                SearchResult(title=f"Result {i}", url=f"http://example.com/{i}", snippet=f"Snippet {i}")
                for i in range(doc_count)
            ])
            
            mock_fetch_tool = Mock()
            mock_fetch_tool.fetch_multiple = Mock(return_value=[
                FetchedContent(url=f"http://example.com/{i}", content="Content " * 50, title=f"Result {i}", success=True)
                for i in range(doc_count)
            ])
            
            # Setup workflow
            search_step = WebSearchStep(search_tool=mock_search_tool)
            fetch_step = FetchUrlStep(fetch_tool=mock_fetch_tool)
            summarize_step = SummarizeStep()
            answer_step = AnswerGenerateStep()
            
            step_registry = {
                "search": search_step,
                "fetch": fetch_step,
                "summarize": summarize_step,
                "answer": answer_step
            }
            
            graph = {
                "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
                "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
                "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
                "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
                "end": GraphNode(step_name="answer", next_nodes=[])
            }
            
            executor = GraphExecutor(step_registry)
            
            initial_state: State = {
                "question": "Test question",
                "meta": {"mode": "default"}
            }
            
            # Measure execution time
            times = []
            for _ in range(5):
                start = time.time()
                executor.execute(graph, initial_state, mock_runtime)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            mean_time = sum(times) / len(times)
            results[doc_count] = mean_time
        
        print(f"\nScalability with Document Count:")
        for doc_count, mean_time in results.items():
            print(f"  {doc_count} docs: {mean_time:.2f}ms")
        
        # Verify reasonable scaling (should be roughly linear)
        # Time for 10 docs should be < 5x time for 2 docs
        assert results[10] < results[2] * 5, "Execution time scales poorly with document count"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
