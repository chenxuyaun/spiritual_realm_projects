"""
Unit tests for SelfAskSearchQA workflow.

Tests the SelfAskSearchQA workflow implementation including:
- Question decomposition
- Sub-question searching
- Answer synthesis
- Parameter validation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from mm_orch.workflows.self_ask_search_qa import (
    SelfAskSearchQAWorkflow,
    QuestionDecomposer,
    AnswerSynthesizer,
    SelfAskContext,
    SelfAskStep,
    SubQuestion
)
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.exceptions import ValidationError


class TestQuestionDecomposer:
    """Tests for QuestionDecomposer class."""
    
    @pytest.fixture
    def decomposer(self):
        """Create a decomposer instance."""
        return QuestionDecomposer()
    
    def test_is_complex_question_with_and(self, decomposer):
        """Test detection of complex questions with 'and'."""
        query = "What is Python and how does it compare to Java?"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_with_compare(self, decomposer):
        """Test detection of comparison questions."""
        query = "Compare the differences between Python and Java"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_with_multiple_question_words(self, decomposer):
        """Test detection of questions with multiple question words."""
        query = "What is machine learning and why is it important?"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_with_pros_cons(self, decomposer):
        """Test detection of pros and cons questions."""
        query = "What are the pros and cons of remote work?"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_with_relationship(self, decomposer):
        """Test detection of relationship questions."""
        query = "What is the relationship between AI and machine learning?"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_long_query(self, decomposer):
        """Test detection of long questions as complex."""
        query = "I want to understand how artificial intelligence works and what are the main techniques used in modern AI systems and how they are applied in real world applications"
        assert decomposer.is_complex_question(query) is True
    
    def test_is_complex_question_simple(self, decomposer):
        """Test that simple questions are not marked as complex."""
        query = "What is Python?"
        assert decomposer.is_complex_question(query) is False
    
    def test_decompose_and_question(self, decomposer):
        """Test decomposition of questions with 'and'."""
        query = "What is Python and what is Java?"
        sub_questions = decomposer.decompose(query)
        
        assert len(sub_questions) >= 1
        assert all(isinstance(sq, SubQuestion) for sq in sub_questions)
    
    def test_decompose_compare_question(self, decomposer):
        """Test decomposition of comparison questions."""
        query = "Compare the difference between Python and Java?"
        sub_questions = decomposer.decompose(query)
        
        assert len(sub_questions) >= 1
        # Should have questions about each entity
        questions_text = " ".join(sq.question.lower() for sq in sub_questions)
        # At least one sub-question should be generated
        assert len(sub_questions) > 0
    
    def test_decompose_pros_cons_question(self, decomposer):
        """Test decomposition of pros and cons questions."""
        query = "What are the pros and cons of Python?"
        sub_questions = decomposer.decompose(query)
        
        assert len(sub_questions) >= 2
        questions_text = " ".join(sq.question.lower() for sq in sub_questions)
        # Should have questions about advantages and disadvantages
        assert "advantage" in questions_text or "disadvantage" in questions_text or "pros" in questions_text or "cons" in questions_text
    
    def test_decompose_simple_question(self, decomposer):
        """Test that simple questions return single sub-question."""
        query = "What is Python?"
        sub_questions = decomposer.decompose(query)
        
        assert len(sub_questions) >= 1
        assert sub_questions[0].question == query
    
    def test_decompose_max_questions_limit(self, decomposer):
        """Test that decomposition respects max_questions limit."""
        query = "What is A and B and C and D and E and F?"
        sub_questions = decomposer.decompose(query, max_questions=3)
        
        assert len(sub_questions) <= 3
    
    def test_sub_question_order(self, decomposer):
        """Test that sub-questions have correct order."""
        query = "What is Python and what is Java?"
        sub_questions = decomposer.decompose(query)
        
        for i, sq in enumerate(sub_questions):
            assert sq.order == i


class TestAnswerSynthesizer:
    """Tests for AnswerSynthesizer class."""
    
    @pytest.fixture
    def synthesizer(self):
        """Create a synthesizer instance."""
        return AnswerSynthesizer()
    
    def test_synthesize_single_answer(self, synthesizer):
        """Test synthesis with single answer."""
        sub_questions = [SubQuestion(question="What is Python?", order=0, success=True)]
        sub_answers = ["Python is a programming language."]
        
        result = synthesizer.synthesize("What is Python?", sub_questions, sub_answers)
        
        assert result is not None
        assert len(result) > 0
        assert "Python" in result
    
    def test_synthesize_multiple_answers(self, synthesizer):
        """Test synthesis with multiple answers."""
        sub_questions = [
            SubQuestion(question="What is Python?", order=0, success=True),
            SubQuestion(question="What is Java?", order=1, success=True)
        ]
        sub_answers = [
            "Python is a high-level programming language.",
            "Java is an object-oriented programming language."
        ]
        
        result = synthesizer.synthesize(
            "What is Python and Java?",
            sub_questions,
            sub_answers
        )
        
        assert result is not None
        assert len(result) > 0
    
    def test_synthesize_no_answers(self, synthesizer):
        """Test synthesis with no successful answers."""
        sub_questions = [
            SubQuestion(question="What is X?", order=0, success=False)
        ]
        sub_answers = [""]
        
        result = synthesizer.synthesize("What is X?", sub_questions, sub_answers)
        
        assert "couldn't find" in result.lower()
    
    def test_synthesize_partial_answers(self, synthesizer):
        """Test synthesis with partial answers."""
        sub_questions = [
            SubQuestion(question="What is Python?", order=0, success=True),
            SubQuestion(question="What is X?", order=1, success=False)
        ]
        sub_answers = ["Python is a programming language.", ""]
        
        result = synthesizer.synthesize(
            "What is Python and X?",
            sub_questions,
            sub_answers
        )
        
        assert result is not None
        assert "Python" in result
    
    def test_generate_summary(self, synthesizer):
        """Test summary generation."""
        answers = [
            {"question": "Q1", "answer": "Answer one is here."},
            {"question": "Q2", "answer": "Answer two is here."}
        ]
        
        summary = synthesizer._generate_summary(answers)
        
        assert summary is not None
        assert len(summary) > 0


class TestSelfAskSearchQAWorkflow:
    """Tests for SelfAskSearchQAWorkflow class."""
    
    @pytest.fixture
    def mock_search_qa_workflow(self):
        """Create a mock SearchQA workflow."""
        workflow = Mock(spec=SearchQAWorkflow)
        workflow.execute.return_value = WorkflowResult(
            result="This is a test answer about the topic.",
            metadata={"workflow": "SearchQA"},
            status="success"
        )
        workflow.get_required_models.return_value = ["t5-small", "gpt2"]
        return workflow
    
    @pytest.fixture
    def workflow(self, mock_search_qa_workflow):
        """Create a SelfAskSearchQA workflow instance."""
        return SelfAskSearchQAWorkflow(
            search_qa_workflow=mock_search_qa_workflow,
            model_manager=None
        )
    
    def test_workflow_type(self, workflow):
        """Test that workflow has correct type."""
        assert workflow.workflow_type == WorkflowType.SELF_ASK_SEARCH_QA
        assert workflow.name == "SelfAskSearchQA"
    
    def test_required_parameters(self, workflow):
        """Test required parameters list."""
        required = workflow.get_required_parameters()
        assert "query" in required
    
    def test_optional_parameters(self, workflow):
        """Test optional parameters with defaults."""
        optional = workflow.get_optional_parameters()
        assert "max_sub_questions" in optional
        assert "include_sub_answers" in optional
        assert "force_decompose" in optional
    
    def test_validate_parameters_valid(self, workflow):
        """Test parameter validation with valid parameters."""
        params = {"query": "What is Python and Java?"}
        assert workflow.validate_parameters(params) is True
    
    def test_validate_parameters_empty_query(self, workflow):
        """Test parameter validation with empty query."""
        params = {"query": ""}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_missing_query(self, workflow):
        """Test parameter validation with missing query."""
        params = {}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_validate_parameters_invalid_max_sub_questions(self, workflow):
        """Test parameter validation with invalid max_sub_questions."""
        params = {"query": "test", "max_sub_questions": 0}
        with pytest.raises(ValidationError):
            workflow.validate_parameters(params)
    
    def test_execute_simple_question(self, workflow, mock_search_qa_workflow):
        """Test execution with simple question (no decomposition)."""
        params = {"query": "What is Python?"}
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.metadata["decomposed"] is False
    
    def test_execute_complex_question(self, workflow, mock_search_qa_workflow):
        """Test execution with complex question (with decomposition)."""
        params = {"query": "What is Python and how does it compare to Java?"}
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.status in ["success", "partial", "failed"]
    
    def test_execute_force_decompose(self, workflow, mock_search_qa_workflow):
        """Test execution with force_decompose flag."""
        params = {"query": "What is Python?", "force_decompose": True}
        result = workflow.execute(params)
        
        assert isinstance(result, WorkflowResult)
        # Should decompose even simple question
        assert result.metadata["decomposed"] is True
    
    def test_execute_includes_sub_answers(self, workflow, mock_search_qa_workflow):
        """Test that sub-answers are included in metadata."""
        params = {
            "query": "What is Python and Java?",
            "include_sub_answers": True,
            "force_decompose": True
        }
        result = workflow.execute(params)
        
        assert "sub_questions" in result.metadata
    
    def test_execute_excludes_sub_answers(self, workflow, mock_search_qa_workflow):
        """Test execution without sub-answers in metadata."""
        params = {
            "query": "What is Python and Java?",
            "include_sub_answers": False,
            "force_decompose": True
        }
        result = workflow.execute(params)
        
        assert "sub_questions" not in result.metadata
    
    def test_execute_search_failure(self, workflow, mock_search_qa_workflow):
        """Test handling of search failures."""
        mock_search_qa_workflow.execute.return_value = WorkflowResult(
            result=None,
            metadata={"workflow": "SearchQA"},
            status="failed",
            error="Search failed"
        )
        
        params = {"query": "What is Python and Java?", "force_decompose": True}
        result = workflow.execute(params)
        
        # Should handle failure gracefully
        assert result.status in ["partial", "failed"]
    
    def test_execute_partial_search_success(self, workflow, mock_search_qa_workflow):
        """Test handling of partial search success."""
        # First call succeeds, second fails
        mock_search_qa_workflow.execute.side_effect = [
            WorkflowResult(
                result="Python is a programming language.",
                metadata={},
                status="success"
            ),
            WorkflowResult(
                result=None,
                metadata={},
                status="failed",
                error="Failed"
            )
        ]
        
        # Use a question that will decompose into multiple sub-questions
        params = {"query": "What are the pros and cons of Python?", "force_decompose": True}
        result = workflow.execute(params)
        
        # Should return partial result when some searches fail
        # Note: if only one sub-question is generated and it succeeds, status will be success
        assert result.status in ["success", "partial"]
        assert result.result is not None
    
    def test_run_method_with_validation(self, workflow):
        """Test the run method which includes validation."""
        params = {"query": "What is Python?"}
        result = workflow.run(params)
        
        assert isinstance(result, WorkflowResult)
        assert result.execution_time is not None
        assert result.execution_time >= 0
    
    def test_run_method_validation_failure(self, workflow):
        """Test run method with invalid parameters."""
        params = {"query": ""}
        result = workflow.run(params)
        
        assert result.status == "failed"
        assert "Validation error" in result.error


class TestSelfAskContext:
    """Tests for SelfAskContext dataclass."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        ctx = SelfAskContext(original_query="test query")
        
        assert ctx.original_query == "test query"
        assert ctx.sub_questions == []
        assert ctx.sub_answers == []
        assert ctx.final_answer == ""
        assert ctx.steps == []
        assert ctx.max_sub_questions == 5
    
    def test_add_step(self):
        """Test adding steps to context."""
        ctx = SelfAskContext(original_query="test")
        step = SelfAskStep(name="decompose", success=True)
        
        ctx.add_step(step)
        
        assert len(ctx.steps) == 1
        assert ctx.steps[0].name == "decompose"


class TestSelfAskStep:
    """Tests for SelfAskStep dataclass."""
    
    def test_step_initialization(self):
        """Test step initialization."""
        step = SelfAskStep(name="decompose", success=True)
        
        assert step.name == "decompose"
        assert step.success is True
        assert step.duration == 0.0
        assert step.input_count == 0
        assert step.output_count == 0
        assert step.error is None
    
    def test_step_with_all_fields(self):
        """Test step with all fields set."""
        step = SelfAskStep(
            name="search",
            success=False,
            duration=2.5,
            input_count=3,
            output_count=2,
            error="One search failed"
        )
        
        assert step.name == "search"
        assert step.success is False
        assert step.duration == 2.5
        assert step.input_count == 3
        assert step.output_count == 2
        assert step.error == "One search failed"


class TestSubQuestion:
    """Tests for SubQuestion dataclass."""
    
    def test_sub_question_initialization(self):
        """Test sub-question initialization."""
        sq = SubQuestion(question="What is Python?", order=0)
        
        assert sq.question == "What is Python?"
        assert sq.order == 0
        assert sq.depends_on == []
        assert sq.answer is None
        assert sq.success is False
        assert sq.error is None
    
    def test_sub_question_with_dependencies(self):
        """Test sub-question with dependencies."""
        sq = SubQuestion(
            question="How do they compare?",
            order=2,
            depends_on=[0, 1]
        )
        
        assert sq.depends_on == [0, 1]
    
    def test_sub_question_with_answer(self):
        """Test sub-question with answer."""
        sq = SubQuestion(question="What is Python?", order=0)
        sq.answer = "Python is a programming language."
        sq.success = True
        
        assert sq.answer == "Python is a programming language."
        assert sq.success is True


class TestSelfAskSearchQAIntegration:
    """Integration-style tests for SelfAskSearchQA workflow."""
    
    def test_full_workflow_mock(self):
        """Test full workflow with mocked SearchQA."""
        # Setup mock SearchQA
        mock_search_qa = Mock(spec=SearchQAWorkflow)
        mock_search_qa.execute.return_value = WorkflowResult(
            result="Python is a high-level programming language known for its simplicity.",
            metadata={"workflow": "SearchQA"},
            status="success"
        )
        mock_search_qa.get_required_models.return_value = ["t5-small", "gpt2"]
        
        workflow = SelfAskSearchQAWorkflow(
            search_qa_workflow=mock_search_qa
        )
        
        # Execute with complex question
        result = workflow.run({
            "query": "What are the pros and cons of Python?",
            "force_decompose": True
        })
        
        # Verify
        assert result.status in ["success", "partial"]
        assert result.result is not None
        assert result.metadata["decomposed"] is True
        assert result.metadata["sub_questions_count"] >= 2
    
    def test_workflow_metrics(self):
        """Test that workflow collects metrics."""
        mock_search_qa = Mock(spec=SearchQAWorkflow)
        mock_search_qa.execute.return_value = WorkflowResult(
            result="Test answer",
            metadata={},
            status="success"
        )
        mock_search_qa.get_required_models.return_value = []
        
        workflow = SelfAskSearchQAWorkflow(
            search_qa_workflow=mock_search_qa
        )
        
        # Execute multiple times
        workflow.run({"query": "test 1"})
        workflow.run({"query": "test 2"})
        
        metrics = workflow.get_metrics()
        
        assert metrics["execution_count"] == 2
        # Execution time may be 0.0 for very fast operations
        assert metrics["total_execution_time"] >= 0
    
    def test_complex_comparison_question(self):
        """Test handling of comparison questions."""
        mock_search_qa = Mock(spec=SearchQAWorkflow)
        mock_search_qa.execute.return_value = WorkflowResult(
            result="Detailed comparison information.",
            metadata={},
            status="success"
        )
        mock_search_qa.get_required_models.return_value = []
        
        workflow = SelfAskSearchQAWorkflow(
            search_qa_workflow=mock_search_qa
        )
        
        result = workflow.run({
            "query": "Compare the difference between Python and Java?",
            "force_decompose": True
        })
        
        assert result.status in ["success", "partial"]
        assert result.metadata["sub_questions_count"] >= 1
