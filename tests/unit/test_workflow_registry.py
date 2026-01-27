"""
Unit tests for Workflow Registry.

Tests the WorkflowRegistry class for registering, retrieving,
and validating workflow definitions.
"""

import pytest
from mm_orch.orchestration.graph_executor import GraphNode
from mm_orch.registries.workflow_registry import (
    WorkflowRegistry,
    WorkflowDefinition,
    get_workflow_registry,
    reset_workflow_registry
)
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.state import State


class DummyStep(BaseStep):
    """Dummy step for testing."""
    
    name = "dummy"
    input_keys = []
    output_keys = []
    
    def execute(self, state: State, runtime) -> dict:
        return {}


class TestWorkflowRegistry:
    """Tests for WorkflowRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create step registry with dummy steps
        self.step_registry = {
            "step1": DummyStep(),
            "step2": DummyStep(),
            "step3": DummyStep(),
            "end": DummyStep()
        }
        
        # Create workflow registry
        self.registry = WorkflowRegistry(self.step_registry)
        
        # Create sample workflow definition
        self.sample_workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["node2"]),
                "node2": GraphNode(step_name="step2", next_nodes=["node3"]),
                "node3": GraphNode(step_name="step3", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["test"],
            tags=["test", "sample"]
        )
    
    def test_register_workflow(self):
        """Test registering a workflow."""
        self.registry.register(self.sample_workflow)
        
        # Verify workflow is registered
        assert self.registry.has("test_workflow")
        assert "test_workflow" in self.registry.list_all()
    
    def test_register_duplicate_workflow(self):
        """Test that registering duplicate workflow name raises error."""
        self.registry.register(self.sample_workflow)
        
        # Try to register again
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(self.sample_workflow)
    
    def test_get_workflow(self):
        """Test retrieving a workflow by name."""
        self.registry.register(self.sample_workflow)
        
        # Retrieve workflow
        workflow = self.registry.get("test_workflow")
        
        assert workflow.name == "test_workflow"
        assert workflow.description == "Test workflow"
        assert len(workflow.graph) == 4
        assert workflow.required_capabilities == ["test"]
        assert "test" in workflow.tags
    
    def test_get_nonexistent_workflow(self):
        """Test that getting non-existent workflow raises error."""
        with pytest.raises(KeyError, match="not found"):
            self.registry.get("nonexistent")
    
    def test_list_all_workflows(self):
        """Test listing all registered workflows."""
        # Register multiple workflows
        self.registry.register(self.sample_workflow)
        
        workflow2 = WorkflowDefinition(
            name="workflow2",
            description="Second workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        self.registry.register(workflow2)
        
        # List all
        workflows = self.registry.list_all()
        
        assert len(workflows) == 2
        assert "test_workflow" in workflows
        assert "workflow2" in workflows
    
    def test_has_workflow(self):
        """Test checking if workflow exists."""
        assert not self.registry.has("test_workflow")
        
        self.registry.register(self.sample_workflow)
        
        assert self.registry.has("test_workflow")
        assert not self.registry.has("nonexistent")
    
    def test_unregister_workflow(self):
        """Test unregistering a workflow."""
        self.registry.register(self.sample_workflow)
        assert self.registry.has("test_workflow")
        
        self.registry.unregister("test_workflow")
        
        assert not self.registry.has("test_workflow")
        assert "test_workflow" not in self.registry.list_all()
    
    def test_unregister_nonexistent_workflow(self):
        """Test that unregistering non-existent workflow raises error."""
        with pytest.raises(KeyError, match="not found"):
            self.registry.unregister("nonexistent")
    
    def test_get_by_capability(self):
        """Test finding workflows by capability."""
        self.registry.register(self.sample_workflow)
        
        workflow2 = WorkflowDefinition(
            name="workflow2",
            description="Second workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["test", "other"]
        )
        self.registry.register(workflow2)
        
        # Find by capability
        test_workflows = self.registry.get_by_capability("test")
        other_workflows = self.registry.get_by_capability("other")
        
        assert len(test_workflows) == 2
        assert len(other_workflows) == 1
        assert other_workflows[0].name == "workflow2"
    
    def test_get_by_tag(self):
        """Test finding workflows by tag."""
        self.registry.register(self.sample_workflow)
        
        workflow2 = WorkflowDefinition(
            name="workflow2",
            description="Second workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            tags=["fast"]
        )
        self.registry.register(workflow2)
        
        # Find by tag
        test_workflows = self.registry.get_by_tag("test")
        fast_workflows = self.registry.get_by_tag("fast")
        
        assert len(test_workflows) == 1
        assert test_workflows[0].name == "test_workflow"
        assert len(fast_workflows) == 1
        assert fast_workflows[0].name == "workflow2"


class TestWorkflowValidation:
    """Tests for workflow validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.step_registry = {
            "step1": DummyStep(),
            "step2": DummyStep(),
            "end": DummyStep()
        }
        self.registry = WorkflowRegistry(self.step_registry)
    
    def test_validate_missing_start_node(self):
        """Test that workflow without start node is rejected."""
        workflow = WorkflowDefinition(
            name="invalid",
            description="Invalid workflow",
            graph={
                "node1": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        with pytest.raises(ValueError, match="must have a 'start' node"):
            self.registry.register(workflow)
    
    def test_validate_nonexistent_step(self):
        """Test that workflow with non-existent step is rejected."""
        workflow = WorkflowDefinition(
            name="invalid",
            description="Invalid workflow",
            graph={
                "start": GraphNode(step_name="nonexistent_step", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        with pytest.raises(KeyError, match="non-existent step"):
            self.registry.register(workflow)
    
    def test_validate_dangling_next_node(self):
        """Test that workflow with dangling next_node reference is rejected."""
        workflow = WorkflowDefinition(
            name="invalid",
            description="Invalid workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["nonexistent_node"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        with pytest.raises(ValueError, match="non-existent next node"):
            self.registry.register(workflow)
    
    def test_validate_valid_workflow(self):
        """Test that valid workflow passes validation."""
        workflow = WorkflowDefinition(
            name="valid",
            description="Valid workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["node2"]),
                "node2": GraphNode(step_name="step2", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Should not raise
        self.registry.register(workflow)
        assert self.registry.has("valid")
    
    def test_validate_end_reference(self):
        """Test that 'end' reference in next_nodes is allowed."""
        workflow = WorkflowDefinition(
            name="valid",
            description="Valid workflow with end reference",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                # Note: 'end' node doesn't need to be in graph
            }
        )
        
        # Should not raise
        self.registry.register(workflow)
        assert self.registry.has("valid")


class TestGlobalRegistry:
    """Tests for global registry singleton."""
    
    def setup_method(self):
        """Reset global registry before each test."""
        reset_workflow_registry()
    
    def teardown_method(self):
        """Reset global registry after each test."""
        reset_workflow_registry()
    
    def test_get_workflow_registry_first_call(self):
        """Test getting global registry on first call."""
        step_registry = {"step1": DummyStep()}
        
        registry = get_workflow_registry(step_registry)
        
        assert isinstance(registry, WorkflowRegistry)
        assert registry.step_registry == step_registry
    
    def test_get_workflow_registry_without_step_registry(self):
        """Test that first call without step_registry raises error."""
        with pytest.raises(ValueError, match="step_registry must be provided"):
            get_workflow_registry()
    
    def test_get_workflow_registry_subsequent_calls(self):
        """Test that subsequent calls return same instance."""
        step_registry = {"step1": DummyStep()}
        
        registry1 = get_workflow_registry(step_registry)
        registry2 = get_workflow_registry()
        
        assert registry1 is registry2
    
    def test_reset_workflow_registry(self):
        """Test resetting global registry."""
        step_registry = {"step1": DummyStep()}
        
        registry1 = get_workflow_registry(step_registry)
        reset_workflow_registry()
        registry2 = get_workflow_registry(step_registry)
        
        assert registry1 is not registry2


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""
    
    def test_create_workflow_definition(self):
        """Test creating a workflow definition."""
        workflow = WorkflowDefinition(
            name="test",
            description="Test workflow",
            graph={
                "start": GraphNode(step_name="step1", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["test"],
            tags=["test"],
            metadata={"key": "value"}
        )
        
        assert workflow.name == "test"
        assert workflow.description == "Test workflow"
        assert len(workflow.graph) == 2
        assert workflow.required_capabilities == ["test"]
        assert workflow.tags == ["test"]
        assert workflow.metadata == {"key": "value"}
    
    def test_workflow_definition_defaults(self):
        """Test workflow definition with default values."""
        workflow = WorkflowDefinition(
            name="test",
            description="Test workflow",
            graph={}
        )
        
        assert workflow.required_capabilities == []
        assert workflow.tags == []
        assert workflow.metadata == {}
