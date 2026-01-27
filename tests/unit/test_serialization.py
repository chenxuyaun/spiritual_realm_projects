"""
Unit tests for State serialization utilities.

Tests cover:
- Basic serialization and deserialization
- Nested structures (lists of dicts, dicts of lists)
- Optional fields and None values
- Empty collections
- Error handling and descriptive messages
- Edge cases

Requirements:
    - 24.1: State serialization preserving all field values
    - 24.2: State deserialization producing equivalent State
    - 24.3: Handle nested structures and optional fields
    - 24.4: Descriptive error messages
"""

import json
import pytest
from datetime import datetime
from dataclasses import dataclass

from mm_orch.orchestration.state import State
from mm_orch.orchestration.serialization import (
    state_to_json,
    json_to_state,
    state_to_dict,
    dict_to_state,
    validate_state_serializable,
    serialize_state_safely,
    StateSerializationError,
    StateDeserializationError,
)


class TestBasicSerialization:
    """Test basic serialization and deserialization."""
    
    def test_simple_state_round_trip(self):
        """Test serialization and deserialization of simple State."""
        state: State = {
            "question": "What is Python?",
            "final_answer": "Python is a programming language.",
            "meta": {"mode": "default"}
        }
        
        # Serialize
        json_str = state_to_json(state)
        
        # Deserialize
        restored = json_to_state(json_str)
        
        # Verify
        assert restored["question"] == state["question"]
        assert restored["final_answer"] == state["final_answer"]
        assert restored["meta"] == state["meta"]
    
    def test_empty_state(self):
        """Test serialization of empty State."""
        state: State = {}
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored == {}
    
    def test_state_with_all_core_fields(self):
        """Test State with all core fields populated."""
        state: State = {
            "question": "Test question",
            "search_results": [
                {"title": "Result 1", "url": "https://example.com", "snippet": "..."}
            ],
            "docs": {"https://example.com": "Content here"},
            "summaries": {"https://example.com": "Summary here"},
            "final_answer": "Answer here",
            "citations": ["https://example.com"],
            "meta": {"mode": "default", "router_version": "v3"}
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["question"] == state["question"]
        assert restored["search_results"] == state["search_results"]
        assert restored["docs"] == state["docs"]
        assert restored["summaries"] == state["summaries"]
        assert restored["final_answer"] == state["final_answer"]
        assert restored["citations"] == state["citations"]
        assert restored["meta"] == state["meta"]


class TestNestedStructures:
    """Test handling of nested structures."""
    
    def test_nested_dicts(self):
        """Test deeply nested dictionaries."""
        state: State = {
            "meta": {
                "level1": {
                    "level2": {
                        "level3": "deep value"
                    }
                }
            }
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["meta"]["level1"]["level2"]["level3"] == "deep value"
    
    def test_list_of_dicts(self):
        """Test list containing dictionaries."""
        state: State = {
            "search_results": [
                {"title": "Result 1", "url": "url1"},
                {"title": "Result 2", "url": "url2"},
                {"title": "Result 3", "url": "url3"}
            ]
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert len(restored["search_results"]) == 3
        assert restored["search_results"][0]["title"] == "Result 1"
        assert restored["search_results"][2]["url"] == "url3"
    
    def test_dict_of_lists(self):
        """Test dictionary containing lists."""
        state: State = {
            "lesson_objectives": ["Objective 1", "Objective 2"],
            "lesson_outline": ["Section 1", "Section 2", "Section 3"]
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["lesson_objectives"] == ["Objective 1", "Objective 2"]
        assert len(restored["lesson_outline"]) == 3
    
    def test_mixed_nested_structures(self):
        """Test complex nested structures with mixed types."""
        state: State = {
            "exercises": [
                {
                    "question": "Q1",
                    "options": ["A", "B", "C"],
                    "metadata": {"difficulty": "easy"}
                },
                {
                    "question": "Q2",
                    "options": ["X", "Y"],
                    "metadata": {"difficulty": "hard"}
                }
            ]
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert len(restored["exercises"]) == 2
        assert restored["exercises"][0]["options"] == ["A", "B", "C"]
        assert restored["exercises"][1]["metadata"]["difficulty"] == "hard"


class TestOptionalFieldsAndNone:
    """Test handling of optional fields and None values."""
    
    def test_none_values_preserved(self):
        """Test that None values are preserved."""
        state: State = {
            "question": "Test",
            "final_answer": None,  # type: ignore
            "meta": {"key": None}
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["final_answer"] is None
        assert restored["meta"]["key"] is None
    
    def test_missing_optional_fields(self):
        """Test State with only some fields populated."""
        state: State = {
            "question": "Test question"
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert "question" in restored
        assert "final_answer" not in restored
        assert "search_results" not in restored
    
    def test_empty_collections(self):
        """Test that empty collections are preserved."""
        state: State = {
            "question": "Test",
            "search_results": [],
            "docs": {},
            "citations": [],
            "meta": {}
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["search_results"] == []
        assert restored["docs"] == {}
        assert restored["citations"] == []
        assert restored["meta"] == {}


class TestLessonPackFields:
    """Test serialization of lesson pack specific fields."""
    
    def test_lesson_pack_state(self):
        """Test State with lesson pack fields."""
        state: State = {
            "lesson_topic": "Python Basics",
            "lesson_objectives": ["Learn syntax", "Write programs"],
            "lesson_outline": ["Introduction", "Variables", "Functions"],
            "board_plan": ["Title", "Examples", "Exercises"],
            "lesson_explain_structured": {
                "topic": "Python",
                "sections": [
                    {"name": "Intro", "content": "..."}
                ]
            },
            "teaching_text": "Today we will learn...",
            "exercises": [
                {"question": "Q1", "answer": "A1"}
            ]
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["lesson_topic"] == "Python Basics"
        assert len(restored["lesson_objectives"]) == 2
        assert len(restored["lesson_outline"]) == 3
        assert restored["lesson_explain_structured"]["topic"] == "Python"
        assert len(restored["exercises"]) == 1


class TestRAGFields:
    """Test serialization of RAG specific fields."""
    
    def test_rag_state(self):
        """Test State with RAG fields."""
        state: State = {
            "question": "What is the policy?",
            "kb_sources": [
                {"doc_id": "doc1", "content": "...", "score": 0.95},
                {"doc_id": "doc2", "content": "...", "score": 0.87}
            ],
            "memory_context": "Previous conversation context..."
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert len(restored["kb_sources"]) == 2
        assert restored["kb_sources"][0]["score"] == 0.95
        assert "Previous conversation" in restored["memory_context"]


class TestConversationFields:
    """Test serialization of conversation fields."""
    
    def test_conversation_state(self):
        """Test State with conversation fields."""
        state: State = {
            "question": "Hello",
            "conversation_id": "conv-123",
            "turn_index": 5,
            "meta": {"mode": "chat"}
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["conversation_id"] == "conv-123"
        assert restored["turn_index"] == 5
        assert restored["meta"]["mode"] == "chat"


class TestPrettyPrinting:
    """Test JSON formatting options."""
    
    def test_compact_json(self):
        """Test compact JSON output (no indentation)."""
        state: State = {
            "question": "Test",
            "meta": {"key": "value"}
        }
        
        json_str = state_to_json(state, indent=None)
        
        # Compact JSON should not have newlines
        assert "\n" not in json_str
    
    def test_pretty_json(self):
        """Test pretty-printed JSON output."""
        state: State = {
            "question": "Test",
            "meta": {"key": "value"}
        }
        
        json_str = state_to_json(state, indent=2)
        
        # Pretty JSON should have newlines and indentation
        assert "\n" in json_str
        assert "  " in json_str


class TestErrorHandling:
    """Test error handling and descriptive messages."""
    
    def test_invalid_json_string(self):
        """Test deserialization of invalid JSON."""
        invalid_json = "{ invalid json }"
        
        with pytest.raises(StateDeserializationError) as exc_info:
            json_to_state(invalid_json)
        
        # Error message should mention JSON and provide details
        assert "JSON" in str(exc_info.value)
    
    def test_non_dict_json(self):
        """Test deserialization of JSON that's not an object."""
        json_array = '["not", "a", "dict"]'
        
        with pytest.raises(StateDeserializationError) as exc_info:
            json_to_state(json_array)
        
        # Error message should mention expected type
        assert "dict" in str(exc_info.value).lower()
    
    def test_dict_to_state_with_non_dict(self):
        """Test dict_to_state with non-dict input."""
        with pytest.raises(StateDeserializationError) as exc_info:
            dict_to_state("not a dict")  # type: ignore
        
        assert "dict" in str(exc_info.value).lower()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_state_to_dict(self):
        """Test state_to_dict conversion."""
        state: State = {
            "question": "Test",
            "meta": {"key": "value"}
        }
        
        result = state_to_dict(state)
        
        assert isinstance(result, dict)
        assert result["question"] == "Test"
        assert result["meta"]["key"] == "value"
    
    def test_dict_to_state(self):
        """Test dict_to_state conversion."""
        data = {
            "question": "Test",
            "meta": {"key": "value"}
        }
        
        state = dict_to_state(data)
        
        assert state["question"] == "Test"
        assert state["meta"]["key"] == "value"
    
    def test_validate_state_serializable_valid(self):
        """Test validation of valid State."""
        state: State = {
            "question": "Test",
            "meta": {}
        }
        
        is_valid, error = validate_state_serializable(state)
        
        assert is_valid
        assert error is None
    
    def test_serialize_state_safely_success(self):
        """Test safe serialization with valid State."""
        state: State = {
            "question": "Test"
        }
        
        json_str = serialize_state_safely(state)
        
        assert "question" in json_str
        assert "Test" in json_str
    
    def test_serialize_state_safely_with_fallback(self):
        """Test safe serialization returns fallback on error."""
        # Create a state that would fail serialization
        # (though in practice, most States should serialize fine)
        state: State = {
            "question": "Test"
        }
        
        # This should succeed, but test the fallback parameter
        json_str = serialize_state_safely(state, fallback='{"error": true}')
        
        # Should get actual serialization, not fallback
        assert "question" in json_str


class TestUnicodeAndSpecialCharacters:
    """Test handling of Unicode and special characters."""
    
    def test_unicode_characters(self):
        """Test State with Unicode characters."""
        state: State = {
            "question": "什么是Python？",
            "final_answer": "Python是一种编程语言。",
            "meta": {"language": "中文"}
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["question"] == "什么是Python？"
        assert restored["final_answer"] == "Python是一种编程语言。"
        assert restored["meta"]["language"] == "中文"
    
    def test_special_characters(self):
        """Test State with special characters."""
        state: State = {
            "question": 'Test with "quotes" and \\backslashes\\',
            "final_answer": "Line 1\nLine 2\tTabbed"
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["question"] == state["question"]
        assert restored["final_answer"] == state["final_answer"]
    
    def test_emoji_characters(self):
        """Test State with emoji characters."""
        state: State = {
            "question": "What is Python? 🐍",
            "final_answer": "Python is great! 👍"
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert "🐍" in restored["question"]
        assert "👍" in restored["final_answer"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_strings(self):
        """Test State with very long string values."""
        long_text = "x" * 10000
        state: State = {
            "question": long_text,
            "final_answer": long_text
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert len(restored["question"]) == 10000
        assert len(restored["final_answer"]) == 10000
    
    def test_deeply_nested_structures(self):
        """Test very deeply nested structures."""
        # Create a deeply nested dict
        nested = {"level": 0}
        current = nested
        for i in range(1, 20):
            current["next"] = {"level": i}
            current = current["next"]
        
        state: State = {
            "meta": nested
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        # Navigate to the deepest level
        current = restored["meta"]
        for _ in range(19):
            current = current["next"]
        assert current["level"] == 19
    
    def test_large_lists(self):
        """Test State with large lists."""
        state: State = {
            "search_results": [
                {"title": f"Result {i}", "url": f"url{i}"}
                for i in range(100)
            ]
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert len(restored["search_results"]) == 100
        assert restored["search_results"][99]["title"] == "Result 99"
    
    def test_numeric_edge_cases(self):
        """Test State with various numeric values."""
        state: State = {
            "meta": {
                "zero": 0,
                "negative": -42,
                "float": 3.14159,
                "large": 999999999999,
                "scientific": 1.23e-10
            }
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["meta"]["zero"] == 0
        assert restored["meta"]["negative"] == -42
        assert abs(restored["meta"]["float"] - 3.14159) < 0.00001
        assert restored["meta"]["large"] == 999999999999
        assert abs(restored["meta"]["scientific"] - 1.23e-10) < 1e-15
    
    def test_boolean_values(self):
        """Test State with boolean values."""
        state: State = {
            "meta": {
                "flag1": True,
                "flag2": False,
                "nested": {"enabled": True}
            }
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["meta"]["flag1"] is True
        assert restored["meta"]["flag2"] is False
        assert restored["meta"]["nested"]["enabled"] is True


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_search_qa_workflow_state(self):
        """Test typical State from search_qa workflow."""
        state: State = {
            "question": "What is machine learning?",
            "search_results": [
                {
                    "title": "Machine Learning - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Machine_learning",
                    "snippet": "Machine learning is a field of study..."
                }
            ],
            "docs": {
                "https://en.wikipedia.org/wiki/Machine_learning": "Full content here..."
            },
            "summaries": {
                "https://en.wikipedia.org/wiki/Machine_learning": "ML is a subset of AI..."
            },
            "final_answer": "Machine learning is a field of artificial intelligence...",
            "citations": ["https://en.wikipedia.org/wiki/Machine_learning"],
            "meta": {
                "mode": "default",
                "router_version": "v3",
                "workflow": "search_qa"
            }
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        # Verify all fields preserved
        assert restored["question"] == state["question"]
        assert len(restored["search_results"]) == 1
        assert len(restored["docs"]) == 1
        assert len(restored["summaries"]) == 1
        assert restored["final_answer"] == state["final_answer"]
        assert restored["citations"] == state["citations"]
        assert restored["meta"]["workflow"] == "search_qa"
    
    def test_lesson_pack_workflow_state(self):
        """Test typical State from lesson_pack workflow."""
        state: State = {
            "lesson_topic": "Introduction to Python",
            "lesson_objectives": [
                "Understand Python syntax",
                "Write simple programs"
            ],
            "lesson_outline": [
                "Introduction",
                "Variables and Data Types",
                "Control Flow",
                "Functions"
            ],
            "lesson_explain_structured": {
                "topic": "Introduction to Python",
                "grade": "High School",
                "sections": [
                    {
                        "name": "Introduction",
                        "teacher_say": "Today we'll learn Python...",
                        "examples": ["print('Hello')", "x = 5"]
                    }
                ]
            },
            "meta": {
                "mode": "default",
                "workflow": "lesson_pack"
            }
        }
        
        json_str = state_to_json(state)
        restored = json_to_state(json_str)
        
        assert restored["lesson_topic"] == "Introduction to Python"
        assert len(restored["lesson_objectives"]) == 2
        assert len(restored["lesson_outline"]) == 4
        assert "sections" in restored["lesson_explain_structured"]
