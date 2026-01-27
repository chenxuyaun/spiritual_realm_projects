"""
Integration tests for Phase A/B compatibility.

Tests that actual Phase A workflows (SearchQA, LessonPack, ChatGenerate, RAGQA)
work correctly in Phase B environment using the compatibility layer.
"""

import pytest
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.workflows.lesson_pack import LessonPackWorkflow
from mm_orch.workflows.chat_generate import ChatGenerateWorkflow
from mm_orch.workflows.rag_qa import RAGQAWorkflow
from mm_orch.orchestration.legacy_adapter import (
    LegacyWorkflowAdapter,
    register_legacy_workflows
)
from mm_orch.orchestration.compatibility import (
    LegacyRuntime,
    LegacyWorkflowExecutor
)
from mm_orch.orchestration.state import State
from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode, SimpleTracer


class TestSearchQACompatibility:
    """Test SearchQA workflow compatibility."""
    
    def test_search_qa_as_step(self):
        """Test SearchQA workflow wrapped as a Phase B step."""
        workflow = SearchQAWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "What is Python?",
            "meta": {"max_results": 3}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        # Verify step execution
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] in ["success", "partial"]
        
        # SearchQA should produce citations if successful
        if result_state["meta"]["workflow_status"] == "success":
            assert "citations" in result_state or "workflow_result" in result_state
    
    def test_search_qa_in_graph(self):
        """Test SearchQA workflow in a Phase B graph."""
        workflow = SearchQAWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        # Create simple graph with just the search_qa step
        graph = {
            "start": GraphNode(step_name="search_qa", next_nodes=[])
        }
        
        step_registry = {"search_qa": adapter}
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer)
        
        initial_state: State = {
            "question": "What is Python?",
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        final_state = executor.execute(graph, initial_state, runtime)
        
        # Verify execution
        assert "final_answer" in final_state
        assert len(tracer.get_traces()) > 0


class TestLessonPackCompatibility:
    """Test LessonPack workflow compatibility."""
    
    def test_lesson_pack_as_step(self):
        """Test LessonPack workflow wrapped as a Phase B step."""
        workflow = LessonPackWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "lesson_topic": "Python basics",
            "meta": {"difficulty": "beginner", "num_exercises": 2}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        # Verify step execution
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] in ["success", "partial"]


class TestChatGenerateCompatibility:
    """Test ChatGenerate workflow compatibility."""
    
    def test_chat_generate_as_step(self):
        """Test ChatGenerate workflow wrapped as a Phase B step."""
        workflow = ChatGenerateWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "Hello, how are you?",
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        # Verify step execution
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] in ["success", "partial"]
        
        # Chat should create a conversation_id
        if result_state["meta"]["workflow_status"] == "success":
            assert "conversation_id" in result_state or "workflow_result" in result_state


class TestRAGQACompatibility:
    """Test RAGQA workflow compatibility."""
    
    def test_rag_qa_as_step(self):
        """Test RAGQA workflow wrapped as a Phase B step."""
        workflow = RAGQAWorkflow()
        adapter = LegacyWorkflowAdapter(workflow)
        
        state: State = {
            "question": "What is in the knowledge base?",
            "meta": {"top_k": 3}
        }
        
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        # Verify step execution
        assert "final_answer" in result_state
        # RAG may fail if vector DB is empty, which is expected
        assert result_state["meta"]["workflow_status"] in ["success", "partial", "failed"]


class TestWorkflowRegistry:
    """Test registering multiple workflows."""
    
    def test_register_all_workflows(self):
        """Test registering all Phase A workflows as Phase B steps."""
        workflows = [
            SearchQAWorkflow(),
            LessonPackWorkflow(),
            ChatGenerateWorkflow(),
            RAGQAWorkflow()
        ]
        
        step_registry = {}
        register_legacy_workflows(step_registry, workflows)
        
        # Verify all workflows are registered
        assert "SearchQA" in step_registry
        assert "LessonPack" in step_registry
        assert "ChatGenerate" in step_registry
        assert "RAGQA" in step_registry
        
        # Verify they are all adapters
        for step in step_registry.values():
            assert isinstance(step, LegacyWorkflowAdapter)


class TestLegacyWorkflowExecutor:
    """Test the LegacyWorkflowExecutor with real workflows."""
    
    def test_executor_with_search_qa(self):
        """Test executor with SearchQA workflow."""
        executor = LegacyWorkflowExecutor()
        workflow = SearchQAWorkflow()
        
        parameters = {"query": "What is Python?", "max_results": 3}
        
        result_state = executor.execute(workflow, parameters=parameters)
        
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] in ["success", "partial"]
    
    def test_executor_with_lesson_pack(self):
        """Test executor with LessonPack workflow."""
        executor = LegacyWorkflowExecutor()
        workflow = LessonPackWorkflow()
        
        parameters = {
            "topic": "Python basics",
            "difficulty": "beginner",
            "num_exercises": 2
        }
        
        result_state = executor.execute(workflow, parameters=parameters)
        
        assert "final_answer" in result_state
        assert result_state["meta"]["workflow_status"] in ["success", "partial"]
    
    def test_executor_phase_a_mode(self):
        """Test executor returning Phase A WorkflowResult."""
        executor = LegacyWorkflowExecutor()
        workflow = SearchQAWorkflow()
        
        parameters = {"query": "What is Python?"}
        
        result = executor.execute_with_result(workflow, parameters)
        
        assert result.status in ["success", "partial", "failed"]
        # Result should have the expected structure
        assert hasattr(result, 'result')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'status')


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_mixed_phase_workflow(self):
        """Test a graph mixing Phase A and Phase B steps."""
        # This would be a more complex test where we have both
        # native Phase B steps and adapted Phase A workflows
        # in the same graph
        
        # For now, just verify we can create such a setup
        search_qa = SearchQAWorkflow()
        search_qa_adapter = LegacyWorkflowAdapter(search_qa)
        
        step_registry = {
            "search_qa": search_qa_adapter
        }
        
        # Create a simple graph
        graph = {
            "start": GraphNode(step_name="search_qa", next_nodes=[])
        }
        
        tracer = SimpleTracer()
        executor = GraphExecutor(step_registry, tracer)
        
        initial_state: State = {
            "question": "Test question",
            "meta": {}
        }
        
        runtime = LegacyRuntime()
        final_state = executor.execute(graph, initial_state, runtime)
        
        # Verify execution completed
        assert "final_answer" in final_state
        assert len(tracer.get_traces()) > 0
    
    def test_backward_compatibility_maintained(self):
        """Test that Phase A code still works without modification."""
        # This test verifies that existing Phase A code continues to work
        
        # Phase A style - direct workflow execution
        workflow = SearchQAWorkflow()
        parameters = {"query": "What is Python?", "max_results": 3}
        result = workflow.run(parameters)
        
        # Should work exactly as before
        assert result.status in ["success", "partial", "failed"]
        assert hasattr(result, 'result')
        assert hasattr(result, 'metadata')
        
        # Phase B style - using adapter
        adapter = LegacyWorkflowAdapter(workflow)
        state: State = {
            "question": "What is Python?",
            "meta": {"max_results": 3}
        }
        runtime = LegacyRuntime()
        result_state = adapter.run(state, runtime)
        
        # Both should produce equivalent results
        assert result.status == result_state["meta"]["workflow_status"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
