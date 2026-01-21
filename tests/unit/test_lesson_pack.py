"""
Unit tests for LessonPack Workflow.

Tests the LessonPackWorkflow implementation including:
- Parameter validation
- Plan generation
- Explanation generation
- Exercise generation
- Result structure validation
"""

import pytest
from unittest.mock import Mock, patch

from mm_orch.workflows.lesson_pack import (
    LessonPackWorkflow,
    LessonPackContext,
    LessonPackStep
)
from mm_orch.schemas import WorkflowResult, WorkflowType, LessonPack
from mm_orch.exceptions import ValidationError


class TestLessonPackWorkflow:
    """Tests for LessonPackWorkflow class."""
    
    def test_workflow_type(self):
        """Test that workflow has correct type."""
        workflow = LessonPackWorkflow()
        assert workflow.workflow_type == WorkflowType.LESSON_PACK
        assert workflow.name == "LessonPack"
    
    def test_required_parameters(self):
        """Test required parameters list."""
        workflow = LessonPackWorkflow()
        required = workflow.get_required_parameters()
        assert "topic" in required
    
    def test_optional_parameters(self):
        """Test optional parameters with defaults."""
        workflow = LessonPackWorkflow()
        optional = workflow.get_optional_parameters()
        assert "difficulty" in optional
        assert "num_exercises" in optional
        assert "language" in optional
        assert "include_examples" in optional
    
    def test_validate_parameters_valid(self):
        """Test parameter validation with valid parameters."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Python Programming"}
        assert workflow.validate_parameters(params) is True
    
    def test_validate_parameters_empty_topic(self):
        """Test parameter validation with empty topic."""
        workflow = LessonPackWorkflow()
        params = {"topic": ""}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_missing_topic(self):
        """Test parameter validation with missing topic."""
        workflow = LessonPackWorkflow()
        params = {}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_num_exercises(self):
        """Test parameter validation with invalid num_exercises."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "num_exercises": 0}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_difficulty(self):
        """Test parameter validation with invalid difficulty."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "difficulty": "expert"}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)

    def test_execute_basic(self):
        """Test basic workflow execution without model manager."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Python基础"}
        
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.status in ["success", "partial"]
        assert result.result is not None
        assert "plan" in result.result
        assert "explanation" in result.result
        assert "exercises" in result.result
    
    def test_execute_with_all_parameters(self):
        """Test workflow execution with all parameters."""
        workflow = LessonPackWorkflow()
        params = {
            "topic": "Machine Learning",
            "difficulty": "advanced",
            "num_exercises": 5,
            "language": "en",
            "include_examples": True
        }
        
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.result is not None
        assert result.result["topic"] == "Machine Learning"
    
    def test_execute_chinese_language(self):
        """Test workflow execution with Chinese language."""
        workflow = LessonPackWorkflow()
        params = {
            "topic": "数据结构",
            "language": "zh",
            "num_exercises": 3
        }
        
        result = workflow.execute(params)
        
        assert result.status in ["success", "partial"]
        assert result.result is not None
    
    def test_execute_english_language(self):
        """Test workflow execution with English language."""
        workflow = LessonPackWorkflow()
        params = {
            "topic": "Data Structures",
            "language": "en",
            "num_exercises": 3
        }
        
        result = workflow.execute(params)
        
        assert result.status in ["success", "partial"]
        assert result.result is not None
    
    def test_result_structure_completeness(self):
        """
        Test that result has complete structure.
        
        Property 6: 教学包结构完整性
        """
        workflow = LessonPackWorkflow()
        params = {"topic": "Test Topic"}
        
        result = workflow.execute(params)
        
        assert result.result is not None
        # Verify all required fields are present
        assert "plan" in result.result
        assert "explanation" in result.result
        assert "exercises" in result.result
        assert "topic" in result.result
        
        # Verify exercises structure
        exercises = result.result["exercises"]
        assert isinstance(exercises, list)
        for exercise in exercises:
            assert "question" in exercise
            assert "answer" in exercise
    
    def test_metadata_contains_workflow_info(self):
        """Test that metadata contains workflow information."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test"}
        
        result = workflow.execute(params)
        
        assert "workflow" in result.metadata
        assert result.metadata["workflow"] == "LessonPack"
        assert "topic" in result.metadata
        assert "steps" in result.metadata
    
    def test_run_method_with_validation(self):
        """Test the run method which includes validation."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test Topic"}
        
        result = workflow.run(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestLessonPackContext:
    """Tests for LessonPackContext dataclass."""
    
    def test_context_initialization(self):
        """Test context initialization with defaults."""
        ctx = LessonPackContext(topic="Test")
        
        assert ctx.topic == "Test"
        assert ctx.difficulty == "intermediate"
        assert ctx.num_exercises == 3
        assert ctx.language == "zh"
        assert ctx.plan == ""
        assert ctx.explanation == ""
        assert ctx.exercises == []
        assert ctx.steps == []
    
    def test_context_add_step(self):
        """Test adding steps to context."""
        ctx = LessonPackContext(topic="Test")
        step = LessonPackStep(name="test_step", success=True, duration=1.0)
        
        ctx.add_step(step)
        
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == "test_step"


class TestLessonPackStep:
    """Tests for LessonPackStep dataclass."""
    
    def test_step_initialization(self):
        """Test step initialization."""
        step = LessonPackStep(name="generate_plan", success=True)
        
        assert step.name == "generate_plan"
        assert step.success is True
        assert step.duration == 0.0
        assert step.error is None
    
    def test_step_with_error(self):
        """Test step with error."""
        step = LessonPackStep(
            name="generate_plan",
            success=False,
            error="Generation failed"
        )
        
        assert step.success is False
        assert step.error == "Generation failed"


class TestLessonPackWithMockModel:
    """Tests for LessonPackWorkflow with mocked model manager."""
    
    def test_execute_with_model_manager(self):
        """Test execution with model manager."""
        mock_model_manager = Mock()
        mock_model_manager.infer.return_value = "Generated content"
        
        workflow = LessonPackWorkflow(model_manager=mock_model_manager)
        params = {"topic": "Test Topic"}
        
        result = workflow.execute(params)
        
        assert result.status in ["success", "partial"]
        # Model manager should be called for generation
        assert mock_model_manager.infer.called
    
    def test_execute_model_failure_fallback(self):
        """Test fallback when model fails."""
        mock_model_manager = Mock()
        mock_model_manager.infer.side_effect = Exception("Model error")
        
        workflow = LessonPackWorkflow(model_manager=mock_model_manager)
        params = {"topic": "Test Topic"}
        
        result = workflow.execute(params)
        
        # Should still produce result using template fallback
        assert result.status in ["success", "partial"]
        assert result.result is not None


class TestLessonPackHelperMethods:
    """Tests for helper methods in LessonPackWorkflow."""
    
    def test_translate_difficulty_chinese(self):
        """Test difficulty translation to Chinese."""
        workflow = LessonPackWorkflow()
        
        assert workflow._translate_difficulty("beginner", "zh") == "初级"
        assert workflow._translate_difficulty("intermediate", "zh") == "中级"
        assert workflow._translate_difficulty("advanced", "zh") == "高级"
    
    def test_translate_difficulty_english(self):
        """Test difficulty translation to English (no change)."""
        workflow = LessonPackWorkflow()
        
        assert workflow._translate_difficulty("beginner", "en") == "beginner"
        assert workflow._translate_difficulty("intermediate", "en") == "intermediate"
        assert workflow._translate_difficulty("advanced", "en") == "advanced"
    
    def test_clean_generated_text(self):
        """Test cleaning generated text."""
        workflow = LessonPackWorkflow()
        
        text = "讲解内容: This is the content"
        cleaned = workflow._clean_generated_text(text, "讲解内容:")
        assert cleaned == "This is the content"
    
    def test_clean_generated_text_empty(self):
        """Test cleaning empty text."""
        workflow = LessonPackWorkflow()
        
        cleaned = workflow._clean_generated_text("", "prefix:")
        assert cleaned == ""
    
    def test_parse_exercises_chinese(self):
        """Test parsing Chinese exercises."""
        workflow = LessonPackWorkflow()
        
        text = """
        问题1: 什么是Python？
        答案1: Python是一种编程语言。
        
        问题2: Python有什么特点？
        答案2: Python简单易学。
        """
        
        exercises = workflow._parse_exercises(text, "zh")
        
        assert len(exercises) >= 1
        assert "question" in exercises[0]
        assert "answer" in exercises[0]
    
    def test_parse_exercises_english(self):
        """Test parsing English exercises."""
        workflow = LessonPackWorkflow()
        
        text = """
        Question 1: What is Python?
        Answer 1: Python is a programming language.
        
        Question 2: What are Python's features?
        Answer 2: Python is easy to learn.
        """
        
        exercises = workflow._parse_exercises(text, "en")
        
        assert len(exercises) >= 1
        assert "question" in exercises[0]
        assert "answer" in exercises[0]
    
    def test_generate_template_exercises(self):
        """Test template exercise generation."""
        workflow = LessonPackWorkflow()
        
        exercises = workflow._generate_template_exercises(
            topic="Python",
            difficulty="intermediate",
            num_exercises=3,
            language="zh"
        )
        
        assert len(exercises) == 3
        for ex in exercises:
            assert "question" in ex
            assert "answer" in ex
    
    def test_create_lesson_pack(self):
        """Test creating LessonPack from context."""
        workflow = LessonPackWorkflow()
        
        ctx = LessonPackContext(topic="Test")
        ctx.plan = "Test plan"
        ctx.explanation = "Test explanation"
        ctx.exercises = [{"question": "Q1", "answer": "A1"}]
        
        lesson_pack = workflow.create_lesson_pack(ctx)
        
        assert lesson_pack is not None
        assert isinstance(lesson_pack, LessonPack)
        assert lesson_pack.topic == "Test"
        assert lesson_pack.plan == "Test plan"
        assert lesson_pack.explanation == "Test explanation"
        assert len(lesson_pack.exercises) == 1
    
    def test_create_lesson_pack_incomplete(self):
        """Test creating LessonPack with incomplete context."""
        workflow = LessonPackWorkflow()
        
        ctx = LessonPackContext(topic="Test")
        ctx.plan = "Test plan"
        # Missing explanation and exercises
        
        lesson_pack = workflow.create_lesson_pack(ctx)
        
        assert lesson_pack is None


class TestLessonPackDifficultyLevels:
    """Tests for different difficulty levels."""
    
    def test_beginner_difficulty(self):
        """Test workflow with beginner difficulty."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "difficulty": "beginner"}
        
        result = workflow.execute(params)
        
        assert result.metadata["difficulty"] == "beginner"
    
    def test_intermediate_difficulty(self):
        """Test workflow with intermediate difficulty."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "difficulty": "intermediate"}
        
        result = workflow.execute(params)
        
        assert result.metadata["difficulty"] == "intermediate"
    
    def test_advanced_difficulty(self):
        """Test workflow with advanced difficulty."""
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "difficulty": "advanced"}
        
        result = workflow.execute(params)
        
        assert result.metadata["difficulty"] == "advanced"
