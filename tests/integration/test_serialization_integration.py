"""
Integration tests for State serialization with other components.

Tests verify that serialization works correctly with:
- Tracer component
- Workflow execution
- Real-world State objects from workflows
"""

import json
import tempfile
from pathlib import Path

from mm_orch.orchestration.state import State
from mm_orch.orchestration.serialization import (
    state_to_json,
    json_to_state,
    state_to_dict,
)
from mm_orch.observability.tracer import Tracer, WorkflowTrace, StepTrace


class TestTracerIntegration:
    """Test serialization integration with Tracer."""
    
    def test_state_in_workflow_trace(self):
        """Test that State can be serialized as part of workflow trace."""
        # Create a State
        state: State = {
            "question": "What is Python?",
            "final_answer": "Python is a programming language.",
            "citations": ["https://python.org"],
            "meta": {"mode": "default", "workflow": "search_qa"}
        }
        
        # Serialize State
        state_json = state_to_json(state)
        
        # Verify it can be included in trace metadata
        assert isinstance(state_json, str)
        assert "question" in state_json
        
        # Deserialize and verify
        restored = json_to_state(state_json)
        assert restored["question"] == state["question"]
    
    def test_workflow_trace_with_state_metadata(self):
        """Test WorkflowTrace can include serialized State."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "traces.jsonl"
            tracer = Tracer(str(trace_path))
            
            # Create workflow trace
            step_trace = StepTrace(
                step_name="test_step",
                start_time=1000.0,
                end_time=1001.0,
                latency_ms=1000.0,
                success=True
            )
            
            workflow_trace = WorkflowTrace(
                request_id="req-123",
                conversation_id=None,
                question="Test question",
                chosen_workflow="search_qa",
                router_version="v3",
                mode="default",
                turn_index=None,
                steps=[step_trace],
                urls_used=["https://example.com"],
                final_answer="Test answer",
                quality_signals={"citation_count": 1},
                cost_stats={"latency_ms": 1000.0},
                timestamp=1000.0,
                success=True
            )
            
            # Write trace
            tracer.write_workflow_trace(workflow_trace)
            
            # Read and verify
            with open(trace_path, 'r', encoding='utf-8') as f:
                line = f.readline()
                trace_data = json.loads(line)
                
                assert trace_data["question"] == "Test question"
                assert trace_data["chosen_workflow"] == "search_qa"
                assert len(trace_data["steps"]) == 1


class TestWorkflowStateRoundTrip:
    """Test serialization of States from actual workflows."""
    
    def test_search_qa_state_round_trip(self):
        """Test serialization of typical search_qa State."""
        state: State = {
            "question": "What is machine learning?",
            "search_results": [
                {
                    "title": "ML Overview",
                    "url": "https://example.com/ml",
                    "snippet": "Machine learning is..."
                }
            ],
            "docs": {
                "https://example.com/ml": "Full content about ML..."
            },
            "summaries": {
                "https://example.com/ml": "ML is a subset of AI..."
            },
            "final_answer": "Machine learning is a field of AI...",
            "citations": ["https://example.com/ml"],
            "meta": {
                "mode": "default",
                "router_version": "v3",
                "workflow": "search_qa"
            }
        }
        
        # Serialize
        json_str = state_to_json(state)
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify all fields
        assert restored["question"] == state["question"]
        assert len(restored["search_results"]) == 1
        assert restored["search_results"][0]["title"] == "ML Overview"
        assert "https://example.com/ml" in restored["docs"]
        assert "https://example.com/ml" in restored["summaries"]
        assert restored["final_answer"] == state["final_answer"]
        assert restored["citations"] == state["citations"]
        assert restored["meta"]["workflow"] == "search_qa"
    
    def test_lesson_pack_state_round_trip(self):
        """Test serialization of typical lesson_pack State."""
        state: State = {
            "lesson_topic": "Python Basics",
            "lesson_objectives": [
                "Understand Python syntax",
                "Write simple programs"
            ],
            "lesson_outline": [
                "Introduction",
                "Variables",
                "Functions"
            ],
            "board_plan": [
                "Title: Python Basics",
                "Example: print('Hello')"
            ],
            "lesson_explain_structured": {
                "topic": "Python Basics",
                "grade": "High School",
                "sections": [
                    {
                        "name": "Introduction",
                        "teacher_say": "Today we'll learn Python...",
                        "examples": ["print('Hello')", "x = 5"]
                    }
                ]
            },
            "teaching_text": "Python is a programming language...",
            "exercises": [
                {
                    "question": "What does print() do?",
                    "answer": "It outputs text to the console."
                }
            ],
            "meta": {
                "mode": "default",
                "workflow": "lesson_pack"
            }
        }
        
        # Serialize
        json_str = state_to_json(state)
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify all fields
        assert restored["lesson_topic"] == "Python Basics"
        assert len(restored["lesson_objectives"]) == 2
        assert len(restored["lesson_outline"]) == 3
        assert len(restored["board_plan"]) == 2
        assert "sections" in restored["lesson_explain_structured"]
        assert len(restored["lesson_explain_structured"]["sections"]) == 1
        assert len(restored["exercises"]) == 1
    
    def test_rag_qa_state_round_trip(self):
        """Test serialization of typical rag_qa State."""
        state: State = {
            "question": "What is the refund policy?",
            "kb_sources": [
                {
                    "doc_id": "policy-001",
                    "content": "Refunds are available within 30 days...",
                    "score": 0.95,
                    "metadata": {"section": "refunds"}
                },
                {
                    "doc_id": "policy-002",
                    "content": "Contact support for refund requests...",
                    "score": 0.87,
                    "metadata": {"section": "support"}
                }
            ],
            "memory_context": "User previously asked about shipping...",
            "final_answer": "According to our policy, refunds are available...",
            "citations": ["policy-001", "policy-002"],
            "meta": {
                "mode": "default",
                "workflow": "rag_qa"
            }
        }
        
        # Serialize
        json_str = state_to_json(state)
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify all fields
        assert restored["question"] == state["question"]
        assert len(restored["kb_sources"]) == 2
        assert restored["kb_sources"][0]["score"] == 0.95
        assert restored["kb_sources"][1]["metadata"]["section"] == "support"
        assert "shipping" in restored["memory_context"]
        assert restored["final_answer"] == state["final_answer"]
    
    def test_chat_generate_state_round_trip(self):
        """Test serialization of typical chat_generate State."""
        state: State = {
            "question": "Tell me a joke",
            "conversation_id": "conv-abc123",
            "turn_index": 5,
            "memory_context": "Previous turns: User asked about weather...",
            "final_answer": "Why did the Python programmer...",
            "meta": {
                "mode": "chat",
                "workflow": "chat_generate",
                "router_version": "v3"
            }
        }
        
        # Serialize
        json_str = state_to_json(state)
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify all fields
        assert restored["question"] == "Tell me a joke"
        assert restored["conversation_id"] == "conv-abc123"
        assert restored["turn_index"] == 5
        assert "weather" in restored["memory_context"]
        assert restored["meta"]["mode"] == "chat"


class TestStateDictConversion:
    """Test state_to_dict for intermediate processing."""
    
    def test_state_to_dict_preserves_structure(self):
        """Test that state_to_dict preserves nested structure."""
        state: State = {
            "question": "Test",
            "meta": {
                "nested": {
                    "deep": {
                        "value": 42
                    }
                }
            }
        }
        
        result = state_to_dict(state)
        
        assert isinstance(result, dict)
        assert result["meta"]["nested"]["deep"]["value"] == 42
    
    def test_state_to_dict_with_lists(self):
        """Test state_to_dict with list fields."""
        state: State = {
            "citations": ["url1", "url2", "url3"],
            "lesson_objectives": ["obj1", "obj2"]
        }
        
        result = state_to_dict(state)
        
        assert len(result["citations"]) == 3
        assert len(result["lesson_objectives"]) == 2
        assert result["citations"][1] == "url2"


class TestSerializationPerformance:
    """Test serialization performance with large States."""
    
    def test_large_state_serialization(self):
        """Test serialization of State with many fields."""
        # Create a large State
        state: State = {
            "question": "Test question",
            "search_results": [
                {
                    "title": f"Result {i}",
                    "url": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}" * 10
                }
                for i in range(50)
            ],
            "docs": {
                f"https://example.com/{i}": f"Content {i}" * 100
                for i in range(20)
            },
            "summaries": {
                f"https://example.com/{i}": f"Summary {i}" * 50
                for i in range(20)
            },
            "meta": {
                "large_list": list(range(1000)),
                "large_dict": {f"key{i}": f"value{i}" for i in range(100)}
            }
        }
        
        # Serialize (should complete without error)
        json_str = state_to_json(state)
        
        # Verify it's valid JSON
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify key fields
        assert len(restored["search_results"]) == 50
        assert len(restored["docs"]) == 20
        assert len(restored["meta"]["large_list"]) == 1000
