"""
Property-based tests for LessonPack Workflow.

This module contains property-based tests using Hypothesis to verify
the correctness properties of the LessonPackWorkflow implementation.

Properties tested:
- Property 6: 教学包结构完整性 (LessonPack Structure Completeness)

**Validates: Requirements 3.1, 3.2, 3.3, 3.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any

from mm_orch.workflows.lesson_pack import (
    LessonPackWorkflow,
    LessonPackContext,
    LessonPackStep
)
from mm_orch.schemas import WorkflowResult, WorkflowType, LessonPack


# Configure Hypothesis settings for property tests
settings.register_profile("default", max_examples=100)
settings.load_profile("default")


# Custom strategies for generating test data
topic_strategy = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' '
    )
).filter(lambda x: x.strip())

difficulty_strategy = st.sampled_from(["beginner", "intermediate", "advanced"])

language_strategy = st.sampled_from(["zh", "en"])

num_exercises_strategy = st.integers(min_value=1, max_value=10)


class TestLessonPackStructureCompleteness:
    """
    Property 6: 教学包结构完整性
    
    对于任何课题输入，LessonPack工作流返回的结果应该包含plan、explanation
    和exercises三个字段，且exercises应该是一个列表，每个元素包含question
    和answer字段。
    
    **Validates: Requirements 3.1, 3.2, 3.3, 3.5**
    """
    
    @given(topic=topic_strategy)
    @settings(max_examples=100)
    def test_result_contains_required_fields(self, topic: str):
        """
        Feature: muai-orchestration-system, Property 6: 教学包结构完整性
        
        For any topic input, the LessonPack workflow result should contain
        plan, explanation, and exercises fields.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.5**
        """
        assume(topic.strip())  # Ensure non-empty topic
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic}
        
        result = workflow.execute(params)
        
        # Result should be a WorkflowResult
        assert isinstance(result, WorkflowResult)
        
        # For successful or partial results, check structure
        if result.status in ["success", "partial"] and result.result is not None:
            assert "plan" in result.result, "Result must contain 'plan' field"
            assert "explanation" in result.result, "Result must contain 'explanation' field"
            assert "exercises" in result.result, "Result must contain 'exercises' field"
            assert "topic" in result.result, "Result must contain 'topic' field"
    
    @given(topic=topic_strategy, num_exercises=num_exercises_strategy)
    @settings(max_examples=100)
    def test_exercises_structure(self, topic: str, num_exercises: int):
        """
        Feature: muai-orchestration-system, Property 6: 教学包结构完整性
        
        For any topic input, exercises should be a list where each element
        contains 'question' and 'answer' fields.
        
        **Validates: Requirements 3.3, 3.5**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic, "num_exercises": num_exercises}
        
        result = workflow.execute(params)
        
        if result.status in ["success", "partial"] and result.result is not None:
            exercises = result.result.get("exercises", [])
            
            # Exercises should be a list
            assert isinstance(exercises, list), "Exercises must be a list"
            
            # Each exercise should have question and answer
            for i, exercise in enumerate(exercises):
                assert isinstance(exercise, dict), f"Exercise {i} must be a dict"
                assert "question" in exercise, f"Exercise {i} must have 'question'"
                assert "answer" in exercise, f"Exercise {i} must have 'answer'"

    @given(
        topic=topic_strategy,
        difficulty=difficulty_strategy,
        language=language_strategy
    )
    @settings(max_examples=100)
    def test_result_with_various_parameters(
        self, 
        topic: str, 
        difficulty: str, 
        language: str
    ):
        """
        Feature: muai-orchestration-system, Property 6: 教学包结构完整性
        
        For any combination of valid parameters, the result structure
        should remain consistent.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.5**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {
            "topic": topic,
            "difficulty": difficulty,
            "language": language
        }
        
        result = workflow.execute(params)
        
        # Result should always be a WorkflowResult
        assert isinstance(result, WorkflowResult)
        
        # Status should be valid
        assert result.status in ["success", "partial", "failed"]
        
        # Metadata should contain workflow info
        assert "workflow" in result.metadata
        assert result.metadata["workflow"] == "LessonPack"
        
        # If successful, verify structure
        if result.status == "success":
            assert result.result is not None
            assert "plan" in result.result
            assert "explanation" in result.result
            assert "exercises" in result.result
            assert len(result.result["plan"]) > 0
            assert len(result.result["explanation"]) > 0
            assert len(result.result["exercises"]) > 0


class TestLessonPackWorkflowProperties:
    """Additional property tests for LessonPackWorkflow."""
    
    @given(topic=topic_strategy)
    @settings(max_examples=100)
    def test_workflow_type_consistency(self, topic: str):
        """
        For any execution, the workflow type should remain consistent.
        
        **Validates: Requirements 3.1**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        
        assert workflow.workflow_type == WorkflowType.LESSON_PACK
        assert workflow.name == "LessonPack"
    
    @given(topic=topic_strategy)
    @settings(max_examples=100)
    def test_metadata_completeness(self, topic: str):
        """
        For any execution, metadata should contain required information.
        
        **Validates: Requirements 3.5**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic}
        
        result = workflow.execute(params)
        
        # Metadata should always be present
        assert result.metadata is not None
        assert isinstance(result.metadata, dict)
        
        # Required metadata fields
        assert "workflow" in result.metadata
        assert "topic" in result.metadata
        assert "steps" in result.metadata
        
        # Steps should be a list
        assert isinstance(result.metadata["steps"], list)
    
    @given(topic=topic_strategy, num_exercises=num_exercises_strategy)
    @settings(max_examples=100)
    def test_num_exercises_respected(self, topic: str, num_exercises: int):
        """
        The number of exercises generated should not exceed the requested amount.
        
        **Validates: Requirements 3.3**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic, "num_exercises": num_exercises}
        
        result = workflow.execute(params)
        
        if result.status in ["success", "partial"] and result.result is not None:
            exercises = result.result.get("exercises", [])
            # Should not exceed requested number
            assert len(exercises) <= num_exercises


class TestLessonPackContextProperties:
    """Property tests for LessonPackContext."""
    
    @given(
        topic=topic_strategy,
        difficulty=difficulty_strategy,
        num_exercises=num_exercises_strategy,
        language=language_strategy
    )
    @settings(max_examples=100)
    def test_context_initialization_properties(
        self,
        topic: str,
        difficulty: str,
        num_exercises: int,
        language: str
    ):
        """
        Context should properly store all initialization parameters.
        """
        assume(topic.strip())
        
        ctx = LessonPackContext(
            topic=topic,
            difficulty=difficulty,
            num_exercises=num_exercises,
            language=language
        )
        
        assert ctx.topic == topic
        assert ctx.difficulty == difficulty
        assert ctx.num_exercises == num_exercises
        assert ctx.language == language
        assert ctx.plan == ""
        assert ctx.explanation == ""
        assert ctx.exercises == []
        assert ctx.steps == []
    
    @given(
        step_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        success=st.booleans(),
        duration=st.floats(min_value=0, max_value=1000)
    )
    @settings(max_examples=100)
    def test_context_step_tracking(
        self,
        step_name: str,
        success: bool,
        duration: float
    ):
        """
        Context should properly track execution steps.
        """
        ctx = LessonPackContext(topic="Test")
        step = LessonPackStep(name=step_name, success=success, duration=duration)
        
        ctx.add_step(step)
        
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == step_name
        assert ctx.steps[0].success == success
        assert ctx.steps[0].duration == duration


class TestLessonPackValidationProperties:
    """Property tests for parameter validation."""
    
    @given(topic=topic_strategy)
    @settings(max_examples=100)
    def test_valid_topic_passes_validation(self, topic: str):
        """
        Any non-empty topic should pass validation.
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic}
        
        # Should not raise
        assert workflow.validate_parameters(params) is True
    
    @given(difficulty=difficulty_strategy)
    @settings(max_examples=100)
    def test_valid_difficulty_passes_validation(self, difficulty: str):
        """
        Any valid difficulty level should pass validation.
        """
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "difficulty": difficulty}
        
        assert workflow.validate_parameters(params) is True
    
    @given(num_exercises=st.integers(min_value=1, max_value=100))
    @settings(max_examples=100)
    def test_valid_num_exercises_passes_validation(self, num_exercises: int):
        """
        Any positive integer for num_exercises should pass validation.
        """
        workflow = LessonPackWorkflow()
        params = {"topic": "Test", "num_exercises": num_exercises}
        
        assert workflow.validate_parameters(params) is True


class TestLessonPackOutputProperties:
    """Property tests for output consistency."""
    
    @given(topic=topic_strategy, language=language_strategy)
    @settings(max_examples=100)
    def test_plan_is_non_empty_on_success(self, topic: str, language: str):
        """
        On successful execution, plan should be non-empty.
        
        **Validates: Requirements 3.1**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic, "language": language}
        
        result = workflow.execute(params)
        
        if result.status == "success":
            assert result.result is not None
            assert len(result.result["plan"]) > 0
    
    @given(topic=topic_strategy, language=language_strategy)
    @settings(max_examples=100)
    def test_explanation_is_non_empty_on_success(self, topic: str, language: str):
        """
        On successful execution, explanation should be non-empty.
        
        **Validates: Requirements 3.2**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic, "language": language}
        
        result = workflow.execute(params)
        
        if result.status == "success":
            assert result.result is not None
            assert len(result.result["explanation"]) > 0
    
    @given(topic=topic_strategy, num_exercises=num_exercises_strategy)
    @settings(max_examples=100)
    def test_exercises_have_content_on_success(self, topic: str, num_exercises: int):
        """
        On successful execution, exercises should have content.
        
        **Validates: Requirements 3.3**
        """
        assume(topic.strip())
        
        workflow = LessonPackWorkflow()
        params = {"topic": topic, "num_exercises": num_exercises}
        
        result = workflow.execute(params)
        
        if result.status == "success":
            assert result.result is not None
            exercises = result.result["exercises"]
            assert len(exercises) > 0
            
            for exercise in exercises:
                assert len(exercise["question"]) > 0
                assert len(exercise["answer"]) > 0
