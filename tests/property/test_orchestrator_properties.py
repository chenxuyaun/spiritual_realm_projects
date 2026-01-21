"""
Property-based tests for WorkflowOrchestrator.

Tests verify:
- Property 4: 结果结构完整性 (Result structure completeness)

Requirements verified:
- 1.2: 按照预定义的步骤序列执行工作流
- 1.3: 记录错误信息并返回可理解的错误响应
- 1.4: 返回结构化的执行结果
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, Optional

from mm_orch.orchestrator import (
    WorkflowOrchestrator,
    create_orchestrator,
    ExecutionContext
)
from mm_orch.schemas import (
    UserRequest,
    WorkflowResult,
    WorkflowSelection,
    WorkflowType
)
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.router import Router
from mm_orch.exceptions import ValidationError


# Test strategies
workflow_type_strategy = st.sampled_from(list(WorkflowType))

query_strategy = st.text(
    min_size=1,
    max_size=200,
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' ?!.,;:'
    )
).filter(lambda x: x.strip())

parameters_strategy = st.fixed_dictionaries({
    "query": query_strategy
}).map(lambda d: {**d, "topic": d["query"], "message": d["query"]})


class MockWorkflow(BaseWorkflow):
    """Mock workflow for testing."""
    
    workflow_type = WorkflowType.SEARCH_QA
    name = "MockWorkflow"
    description = "Mock workflow for testing"
    
    def __init__(self, should_succeed: bool = True, result_value: Any = "test result"):
        super().__init__()
        self.should_succeed = should_succeed
        self.result_value = result_value
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        if self.should_succeed:
            return WorkflowResult(
                result=self.result_value,
                metadata={"mock": True},
                status="success"
            )
        else:
            return WorkflowResult(
                result=None,
                metadata={"mock": True},
                status="failed",
                error="Mock failure"
            )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return True
    
    def get_required_parameters(self):
        return ["query"]


@pytest.fixture
def orchestrator():
    """Create a fresh orchestrator for each test."""
    return create_orchestrator(auto_register_workflows=True)


@pytest.fixture
def mock_orchestrator():
    """Create an orchestrator with mock workflows."""
    orch = create_orchestrator(auto_register_workflows=False)
    mock_wf = MockWorkflow()
    orch.register_workflow(mock_wf)
    return orch


class TestProperty4ResultStructureCompleteness:
    """
    Property 4: 结果结构完整性
    
    对于任何成功执行的工作流，返回的WorkflowResult对象应该包含
    非空的result字段、metadata字典和status='success'。
    
    验证需求: 1.4
    """
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_successful_result_has_complete_structure(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        对于任何成功执行的工作流，返回的WorkflowResult应该包含完整结构。
        
        **Validates: Requirements 1.4**
        """
        # Create orchestrator with mock workflow that always succeeds
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True, result_value=f"Result for: {query}")
        orch.register_workflow(mock_wf)
        
        # Execute workflow
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # Verify structure completeness for successful results
        if result.status == "success":
            # result field must be non-None
            assert result.result is not None, "Successful result must have non-None result"
            
            # metadata must be a dictionary
            assert isinstance(result.metadata, dict), "metadata must be a dictionary"
            
            # status must be 'success'
            assert result.status == "success", "status must be 'success'"
            
            # metadata should contain workflow_type
            assert "workflow_type" in result.metadata, "metadata should contain workflow_type"
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_result_always_has_metadata_dict(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        对于任何工作流执行结果，metadata必须是字典类型。
        
        **Validates: Requirements 1.4**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # metadata must always be a dictionary
        assert isinstance(result.metadata, dict), "metadata must always be a dictionary"
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_result_has_valid_status(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        对于任何工作流执行结果，status必须是有效值之一。
        
        **Validates: Requirements 1.4**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # status must be one of valid values
        valid_statuses = {"success", "partial", "failed"}
        assert result.status in valid_statuses, f"status must be one of {valid_statuses}"


class TestErrorHandlingProperties:
    """
    Tests for error handling in orchestrator.
    
    验证需求: 1.3 - 记录错误信息并返回可理解的错误响应
    """
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_failed_result_has_error_field(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        对于失败的工作流执行，返回的WorkflowResult应该包含error字段。
        
        **Validates: Requirements 1.3**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=False)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # Failed results should have error field
        if result.status == "failed":
            assert result.error is not None, "Failed result must have error field"
            assert isinstance(result.error, str), "error must be a string"
            assert len(result.error) > 0, "error must not be empty"
    
    def test_unregistered_workflow_returns_error(self):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        对于未注册的工作流类型，应该返回错误结果。
        
        **Validates: Requirements 1.3**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test query"}
        )
        
        # Should return failed status
        assert result.status == "failed", "Unregistered workflow should return failed status"
        assert result.error is not None, "Should have error message"
        assert "not registered" in result.error.lower(), "Error should mention workflow not registered"


class TestWorkflowRegistrationProperties:
    """
    Tests for workflow registration functionality.
    """
    
    @given(workflow_type=workflow_type_strategy)
    @settings(max_examples=20, deadline=None)
    def test_registered_workflow_can_be_retrieved(self, workflow_type: WorkflowType):
        """
        Feature: muai-orchestration-system
        
        注册的工作流应该可以被检索。
        
        **Validates: Requirements 1.2**
        """
        orch = create_orchestrator(auto_register_workflows=True)
        
        # All default workflows should be registered
        workflow = orch.get_workflow(workflow_type)
        assert workflow is not None, f"Workflow {workflow_type.value} should be registered"
        assert isinstance(workflow, BaseWorkflow), "Retrieved workflow should be BaseWorkflow instance"
    
    def test_all_workflow_types_registered_by_default(self):
        """
        Feature: muai-orchestration-system
        
        默认情况下所有工作流类型都应该被注册。
        
        **Validates: Requirements 1.2**
        """
        orch = create_orchestrator(auto_register_workflows=True)
        
        registered = orch.get_registered_workflows()
        
        # All workflow types should be registered
        for wf_type in WorkflowType:
            assert wf_type in registered, f"Workflow {wf_type.value} should be registered by default"


class TestExecutionMetadataProperties:
    """
    Tests for execution metadata in results.
    """
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_result_contains_execution_time(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        执行结果应该包含执行时间。
        
        **Validates: Requirements 1.4**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # execution_time should be set
        assert result.execution_time is not None, "execution_time should be set"
        assert result.execution_time >= 0, "execution_time should be non-negative"
    
    @given(query=query_strategy)
    @settings(max_examples=50, deadline=None)
    def test_result_metadata_contains_workflow_type(self, query: str):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        执行结果的metadata应该包含workflow_type。
        
        **Validates: Requirements 1.4**
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        result = orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": query}
        )
        
        # metadata should contain workflow_type
        assert "workflow_type" in result.metadata, "metadata should contain workflow_type"
        assert result.metadata["workflow_type"] == WorkflowType.SEARCH_QA.value


class TestStatisticsProperties:
    """
    Tests for orchestrator statistics tracking.
    """
    
    def test_statistics_track_executions(self):
        """
        Feature: muai-orchestration-system
        
        统计信息应该正确跟踪执行次数。
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        # Execute multiple times
        for i in range(5):
            orch.execute_workflow(
                workflow_type=WorkflowType.SEARCH_QA,
                parameters={"query": f"test query {i}"}
            )
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 5, "Should track 5 executions"
        assert stats["success_count"] == 5, "Should track 5 successes"
        assert stats["failure_count"] == 0, "Should track 0 failures"
    
    def test_statistics_track_failures(self):
        """
        Feature: muai-orchestration-system
        
        统计信息应该正确跟踪失败次数。
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=False)
        orch.register_workflow(mock_wf)
        
        # Execute multiple times
        for i in range(3):
            orch.execute_workflow(
                workflow_type=WorkflowType.SEARCH_QA,
                parameters={"query": f"test query {i}"}
            )
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 3, "Should track 3 executions"
        assert stats["failure_count"] == 3, "Should track 3 failures"
    
    def test_reset_statistics(self):
        """
        Feature: muai-orchestration-system
        
        重置统计信息应该清除所有计数。
        """
        orch = create_orchestrator(auto_register_workflows=False)
        mock_wf = MockWorkflow(should_succeed=True)
        orch.register_workflow(mock_wf)
        
        # Execute some workflows
        orch.execute_workflow(
            workflow_type=WorkflowType.SEARCH_QA,
            parameters={"query": "test"}
        )
        
        # Reset
        orch.reset_statistics()
        
        stats = orch.get_statistics()
        
        assert stats["execution_count"] == 0, "execution_count should be 0 after reset"
        assert stats["success_count"] == 0, "success_count should be 0 after reset"
        assert stats["failure_count"] == 0, "failure_count should be 0 after reset"
