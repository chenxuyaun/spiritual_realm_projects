"""
Integration test for Lesson Pack Structured Output MVP.

This test verifies the minimal viable product implementation:
1. Structured JSON field is added to context
2. JSON parsing works with valid input
3. Fallback to plain text works
4. Backward compatibility is maintained
"""

import pytest
import json
from unittest.mock import Mock, MagicMock

from mm_orch.workflows.lesson_pack import LessonPackWorkflow, LessonPackContext


class TestLessonPackStructuredMVP:
    """Test Lesson Pack structured output MVP functionality."""
    
    def test_context_has_structured_field(self):
        """Test that LessonPackContext has lesson_explain_structured field."""
        ctx = LessonPackContext(topic="Python Basics")
        
        assert hasattr(ctx, "lesson_explain_structured"), "Context should have structured field"
        assert ctx.lesson_explain_structured is None, "Should default to None"
    
    def test_json_parsing_with_valid_input(self):
        """Test JSON parsing with valid structured input."""
        workflow = LessonPackWorkflow()
        
        valid_json = json.dumps({
            "topic": "Python Basics",
            "sections": [
                {"name": "Introduction", "content": "Python is a programming language"},
                {"name": "Main Content", "content": "Variables and data types"}
            ]
        }, ensure_ascii=False)
        
        result = workflow._parse_structured_explanation(valid_json)
        
        assert result is not None, "Should parse valid JSON"
        assert "topic" in result
        assert "sections" in result
        assert len(result["sections"]) == 2
        assert result["sections"][0]["name"] == "Introduction"
    
    def test_json_parsing_with_markdown_blocks(self):
        """Test JSON parsing removes markdown code blocks."""
        workflow = LessonPackWorkflow()
        
        json_with_markdown = """```json
{
  "topic": "Test",
  "sections": [
    {"name": "Section 1", "content": "Content 1"}
  ]
}
```"""
        
        result = workflow._parse_structured_explanation(json_with_markdown)
        
        assert result is not None, "Should parse JSON with markdown blocks"
        assert result["topic"] == "Test"
    
    def test_json_parsing_with_invalid_input(self):
        """Test JSON parsing returns None for invalid input."""
        workflow = LessonPackWorkflow()
        
        # Invalid JSON
        result = workflow._parse_structured_explanation("This is not JSON")
        assert result is None, "Should return None for invalid JSON"
        
        # Missing required fields
        invalid_structure = json.dumps({"topic": "Test"})
        result = workflow._parse_structured_explanation(invalid_structure)
        assert result is None, "Should return None for missing sections"
        
        # Invalid section structure
        invalid_sections = json.dumps({
            "topic": "Test",
            "sections": [{"name": "Section 1"}]  # Missing content
        })
        result = workflow._parse_structured_explanation(invalid_sections)
        assert result is None, "Should return None for invalid section structure"
    
    def test_text_rendering_from_structured(self):
        """Test rendering structured JSON to plain text."""
        workflow = LessonPackWorkflow()
        
        structured = {
            "topic": "Python Basics",
            "sections": [
                {"name": "导入", "content": "今天我们学习Python"},
                {"name": "新授", "content": "Python是一种编程语言"}
            ]
        }
        
        text = workflow._render_structured_to_text(structured, "zh")
        
        assert "# Python Basics" in text
        assert "## 导入" in text
        assert "## 新授" in text
        assert "今天我们学习Python" in text
        assert "Python是一种编程语言" in text
    
    def test_workflow_backward_compatibility(self):
        """Test that workflow maintains backward compatibility."""
        # Create workflow with mock model manager
        model_manager = Mock()
        model_manager.infer = Mock(return_value="Generated plan content")
        
        workflow = LessonPackWorkflow(
            model_manager=model_manager,
            use_real_models=False
        )
        
        # Execute workflow
        parameters = {
            "topic": "Python Basics",
            "difficulty": "beginner",
            "num_exercises": 2
        }
        
        result = workflow.execute(parameters)
        
        # Verify backward compatibility
        assert result.result is not None, "Should return result"
        assert "topic" in result.result
        assert "plan" in result.result
        assert "explanation" in result.result, "Should have explanation field (backward compat)"
        assert "exercises" in result.result
        
        # Verify metadata
        assert "has_structured_output" in result.metadata
    
    def test_workflow_with_structured_output(self):
        """Test workflow with structured output generation."""
        # Create workflow with mock inference engine
        inference_engine = Mock()
        
        # Mock successful JSON generation
        mock_result = Mock()
        mock_result.text = json.dumps({
            "topic": "Python Basics",
            "sections": [
                {"name": "Introduction", "content": "Python is great"},
                {"name": "Details", "content": "Variables and types"}
            ]
        }, ensure_ascii=False)
        mock_result.tokens_per_second = 50.0
        
        inference_engine.generate = Mock(return_value=mock_result)
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        # Execute workflow
        parameters = {
            "topic": "Python Basics",
            "difficulty": "beginner",
            "num_exercises": 2
        }
        
        result = workflow.execute(parameters)
        
        # Verify structured output
        assert result.metadata["has_structured_output"] is True
        assert "lesson_explain_structured" in result.metadata
        assert result.metadata["lesson_explain_structured"]["topic"] == "Python Basics"
        assert len(result.metadata["lesson_explain_structured"]["sections"]) == 2
        
        # Verify backward compatibility - explanation field still exists
        assert result.result["explanation"] != ""
        assert "# Python Basics" in result.result["explanation"]
    
    def test_workflow_fallback_to_plain_text(self):
        """Test workflow falls back to plain text when JSON parsing fails."""
        # Create workflow with mock inference engine that returns invalid JSON
        inference_engine = Mock()
        
        mock_result = Mock()
        mock_result.text = "This is plain text, not JSON"
        mock_result.tokens_per_second = 50.0
        
        inference_engine.generate = Mock(return_value=mock_result)
        
        # Mock model manager for fallback
        model_manager = Mock()
        model_manager.infer = Mock(return_value="Fallback plain text explanation")
        
        workflow = LessonPackWorkflow(
            model_manager=model_manager,
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        # Execute workflow
        parameters = {
            "topic": "Python Basics",
            "difficulty": "beginner",
            "num_exercises": 2
        }
        
        result = workflow.execute(parameters)
        
        # Verify fallback behavior
        assert result.metadata["has_structured_output"] is False
        assert "lesson_explain_structured" not in result.metadata or result.metadata.get("lesson_explain_structured") is None
        
        # Verify explanation still exists (backward compat)
        assert result.result["explanation"] != ""


@pytest.mark.integration
class TestLessonPackStructuredIntegration:
    """Integration tests for full workflow execution."""
    
    def test_end_to_end_with_mock_models(self):
        """Test end-to-end workflow with mock models."""
        model_manager = Mock()
        model_manager.infer = Mock(side_effect=[
            "# Teaching Plan\n\n1. Introduction\n2. Main content",  # Plan
            "Detailed explanation content",  # Explanation
            "Question 1: What is Python?\nAnswer 1: A programming language"  # Exercises
        ])
        
        workflow = LessonPackWorkflow(
            model_manager=model_manager,
            use_real_models=False
        )
        
        result = workflow.execute({
            "topic": "Python Programming",
            "difficulty": "intermediate",
            "num_exercises": 1
        })
        
        assert result.status == "success"
        assert result.result is not None
        assert result.result["topic"] == "Python Programming"
        assert result.result["plan"] != ""
        assert result.result["explanation"] != ""
        assert len(result.result["exercises"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
