"""
Unit tests for BaseWorkflow abstract class.

Tests parameter validation, workflow execution flow, error handling,
and metrics collection.
"""

import pytest
from typing import Any, Dict, List

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.exceptions import ValidationError, WorkflowError


class ConcreteWorkflow(BaseWorkflow):
    """Concrete implementation of BaseWorkflow for testing."""
    
    workflow_type = WorkflowType.SEARCH_QA
    name = "TestWorkflow"
    description = "A test workflow implementation"
    
    def __init__(self, should_fail: bool = False, partial_result: bool = False):
        super().__init__()
        self.should_fail = should_fail
        self.partial_result = partial_result
        self.execute_called = False
        self.last_parameters = None
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        self.execute_called = True
        self.last_parameters = parameters
        
        if self.should_fail:
            raise WorkflowError("Simulated workflow failure")
        
        status = "partial" if self.partial_result else "success"
        return WorkflowResult(
            result={"answer": f"Result for query: {parameters.get('query', '')}"},
            metadata={"workflow": self.name, "params": parameters},
            status=status,
            error="Partial result due to some issues" if self.partial_result else None
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        if "query" not in parameters:
            raise ValidationError("Missing required parameter: query")
        if not isinstance(parameters["query"], str):
            raise ValidationError("Parameter 'query' must be a string")
        if not parameters["query"].strip():
            raise ValidationError("Parameter 'query' cannot be empty")
        return True
    
    def get_required_models(self) -> List[str]:
        return ["qwen-chat", "t5-summarizer"]
    
    def get_required_parameters(self) -> List[str]:
        return ["query"]
    
    def get_optional_parameters(self) -> Dict[str, Any]:
        return {"max_results": 5, "language": "en"}


class TestBaseWorkflowInstantiation:
    """Tests for BaseWorkflow instantiation and attributes."""
    
    def test_cannot_instantiate_abstract_class(self):
        """BaseWorkflow cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseWorkflow()
    
    def test_concrete_workflow_instantiation(self):
        """Concrete workflow can be instantiated."""
        workflow = ConcreteWorkflow()
        assert workflow is not None
        assert workflow.name == "TestWorkflow"
        assert workflow.workflow_type == WorkflowType.SEARCH_QA
    
    def test_workflow_initial_metrics(self):
        """Workflow starts with zero execution metrics."""
        workflow = ConcreteWorkflow()
        metrics = workflow.get_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["total_execution_time"] == 0.0
        assert metrics["average_execution_time"] == 0.0


class TestParameterValidation:
    """Tests for parameter validation."""
    
    def test_valid_parameters(self):
        """Valid parameters pass validation."""
        workflow = ConcreteWorkflow()
        assert workflow.validate_parameters({"query": "test question"}) is True
    
    def test_missing_required_parameter(self):
        """Missing required parameter raises ValidationError."""
        workflow = ConcreteWorkflow()
        with pytest.raises(ValidationError) as exc_info:
            workflow.validate_parameters({})
        assert "query" in str(exc_info.value)
    
    def test_invalid_parameter_type(self):
        """Invalid parameter type raises ValidationError."""
        workflow = ConcreteWorkflow()
        with pytest.raises(ValidationError) as exc_info:
            workflow.validate_parameters({"query": 123})
        assert "string" in str(exc_info.value)
    
    def test_empty_query_parameter(self):
        """Empty query parameter raises ValidationError."""
        workflow = ConcreteWorkflow()
        with pytest.raises(ValidationError) as exc_info:
            workflow.validate_parameters({"query": "   "})
        assert "empty" in str(exc_info.value)
    
    def test_get_required_parameters(self):
        """get_required_parameters returns correct list."""
        workflow = ConcreteWorkflow()
        required = workflow.get_required_parameters()
        assert "query" in required
    
    def test_get_optional_parameters(self):
        """get_optional_parameters returns correct defaults."""
        workflow = ConcreteWorkflow()
        optional = workflow.get_optional_parameters()
        assert optional["max_results"] == 5
        assert optional["language"] == "en"


class TestWorkflowExecution:
    """Tests for workflow execution."""
    
    def test_successful_execution(self):
        """Successful execution returns success result."""
        workflow = ConcreteWorkflow()
        result = workflow.run({"query": "What is Python?"})
        
        assert result.status == "success"
        assert result.result is not None
        assert "Python" in result.result["answer"]
        assert result.error is None
        assert result.execution_time is not None
        assert result.execution_time >= 0
    
    def test_execute_method_called(self):
        """Execute method is called during run."""
        workflow = ConcreteWorkflow()
        workflow.run({"query": "test"})
        
        assert workflow.execute_called is True
        assert workflow.last_parameters is not None
    
    def test_parameters_merged_with_defaults(self):
        """Optional parameters are merged with defaults."""
        workflow = ConcreteWorkflow()
        workflow.run({"query": "test"})
        
        # Check that defaults were merged
        assert workflow.last_parameters["max_results"] == 5
        assert workflow.last_parameters["language"] == "en"
    
    def test_user_parameters_override_defaults(self):
        """User-provided parameters override defaults."""
        workflow = ConcreteWorkflow()
        workflow.run({"query": "test", "max_results": 10, "language": "zh"})
        
        assert workflow.last_parameters["max_results"] == 10
        assert workflow.last_parameters["language"] == "zh"
    
    def test_partial_result(self):
        """Partial result returns partial status."""
        workflow = ConcreteWorkflow(partial_result=True)
        result = workflow.run({"query": "test"})
        
        assert result.status == "partial"
        assert result.result is not None
        assert result.error is not None


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""
    
    def test_validation_error_returns_failed_result(self):
        """Validation error returns failed result."""
        workflow = ConcreteWorkflow()
        result = workflow.run({})  # Missing required 'query'
        
        assert result.status == "failed"
        assert result.result is None
        assert "Validation error" in result.error
        assert result.execution_time is not None
    
    def test_workflow_error_returns_failed_result(self):
        """WorkflowError returns failed result."""
        workflow = ConcreteWorkflow(should_fail=True)
        result = workflow.run({"query": "test"})
        
        assert result.status == "failed"
        assert result.result is None
        assert "Workflow error" in result.error
        assert "Simulated workflow failure" in result.error
    
    def test_unexpected_error_returns_failed_result(self):
        """Unexpected error returns failed result with error type."""
        class BrokenWorkflow(ConcreteWorkflow):
            def execute(self, parameters):
                raise RuntimeError("Unexpected crash")
        
        workflow = BrokenWorkflow()
        result = workflow.run({"query": "test"})
        
        assert result.status == "failed"
        assert result.result is None
        assert "RuntimeError" in result.error
        assert "Unexpected crash" in result.error


class TestWorkflowMetrics:
    """Tests for workflow metrics collection."""
    
    def test_metrics_updated_after_execution(self):
        """Metrics are updated after successful execution."""
        workflow = ConcreteWorkflow()
        workflow.run({"query": "test1"})
        workflow.run({"query": "test2"})
        
        metrics = workflow.get_metrics()
        assert metrics["execution_count"] == 2
        # Execution time may be 0.0 on fast systems, just check it's non-negative
        assert metrics["total_execution_time"] >= 0
        assert metrics["average_execution_time"] >= 0
    
    def test_metrics_include_workflow_info(self):
        """Metrics include workflow name and type."""
        workflow = ConcreteWorkflow()
        metrics = workflow.get_metrics()
        
        assert metrics["workflow"] == "TestWorkflow"
        assert metrics["workflow_type"] == "search_qa"
    
    def test_average_execution_time_calculation(self):
        """Average execution time is calculated correctly."""
        workflow = ConcreteWorkflow()
        workflow.run({"query": "test1"})
        workflow.run({"query": "test2"})
        
        metrics = workflow.get_metrics()
        expected_avg = metrics["total_execution_time"] / metrics["execution_count"]
        assert abs(metrics["average_execution_time"] - expected_avg) < 0.001


class TestWorkflowModels:
    """Tests for workflow model requirements."""
    
    def test_get_required_models(self):
        """get_required_models returns model list."""
        workflow = ConcreteWorkflow()
        models = workflow.get_required_models()
        
        assert isinstance(models, list)
        assert "qwen-chat" in models
        assert "t5-summarizer" in models
    
    def test_default_required_models_empty(self):
        """Default get_required_models returns empty list."""
        class MinimalWorkflow(BaseWorkflow):
            workflow_type = WorkflowType.CHAT_GENERATE
            
            def execute(self, parameters):
                return WorkflowResult(result="ok", metadata={}, status="success")
            
            def validate_parameters(self, parameters):
                return True
        
        workflow = MinimalWorkflow()
        assert workflow.get_required_models() == []


class TestWorkflowRepr:
    """Tests for workflow string representation."""
    
    def test_repr(self):
        """__repr__ returns informative string."""
        workflow = ConcreteWorkflow()
        repr_str = repr(workflow)
        
        assert "ConcreteWorkflow" in repr_str
        assert "TestWorkflow" in repr_str
        assert "SEARCH_QA" in repr_str
