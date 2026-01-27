"""
Property-based tests for Workflow Registry.

Tests universal properties that must hold for all workflow registrations
and step validation using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from mm_orch.orchestration.graph_executor import GraphNode
from mm_orch.registries.workflow_registry import (
    WorkflowRegistry,
    WorkflowDefinition
)
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration.state import State


# Strategy for generating valid workflow names
# Use only printable ASCII characters excluding backslash and quotes to avoid string representation issues
workflow_names = st.text(
    alphabet=st.characters(
        min_codepoint=32,  # Space
        max_codepoint=126,  # Tilde (printable ASCII range)
        blacklist_characters='\\\'"'  # Exclude backslash and quotes to avoid escaping issues
    ),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "" and x not in ["start", "end"])

# Strategy for generating step names
step_names = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters='\\\'"'
    ),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "" and x not in ["start", "end"])

# Strategy for generating node names
node_names = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters='\\\'"'
    ),
    min_size=1,
    max_size=30
).filter(lambda x: x.strip() != "")

# Strategy for generating descriptions
descriptions = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters='\\\'"'
    ),
    min_size=1,
    max_size=100
)

# Strategy for generating capability lists
capabilities = st.lists(
    st.text(
        alphabet=st.characters(
            min_codepoint=32,
            max_codepoint=126,
            blacklist_characters='\\\'"'
        ),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.strip() != ""),
    min_size=0,
    max_size=5,
    unique=True
)


class DummyStep(BaseStep):
    """Dummy step for testing."""
    
    name = "dummy"
    input_keys = []
    output_keys = []
    
    def execute(self, state: State, runtime) -> dict:
        return {}


def create_step_registry(step_names_list):
    """Create a step registry with given step names."""
    registry = {}
    for name in step_names_list:
        step = DummyStep()
        step.name = name
        registry[name] = step
    # Always include 'end' step
    end_step = DummyStep()
    end_step.name = "end"
    registry["end"] = end_step
    return registry


def create_simple_workflow_graph(step_name):
    """Create a simple valid workflow graph with one step."""
    return {
        "start": GraphNode(step_name=step_name, next_nodes=["end"]),
        "end": GraphNode(step_name="end", next_nodes=[])
    }


class TestWorkflowStepValidation:
    """
    Property 9: Workflow Step Validation
    
    For any workflow registration attempt, if the workflow graph references
    a step name that does not exist in the Step_Registry, the Workflow_Registry
    must reject the registration with an error identifying the missing step.
    
    Validates: Requirements 8.3
    """

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        valid_step_names=st.lists(
            step_names,
            min_size=1,
            max_size=10,
            unique=True
        ),
        invalid_step_name=step_names,
        desc=descriptions
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None
    )
    def test_workflow_rejects_nonexistent_step(
        self, workflow_name, valid_step_names, invalid_step_name, desc
    ):
        """
        Property 9: Workflow Step Validation
        
        Verifies that workflow registration fails when the graph references
        a step that doesn't exist in the step registry.
        """
        # Ensure invalid step is not in valid steps
        assume(invalid_step_name not in valid_step_names)
        
        # Create step registry with valid steps
        step_registry = create_step_registry(valid_step_names)
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow with invalid step reference
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph={
                "start": GraphNode(step_name=invalid_step_name, next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Registration must fail with KeyError
        with pytest.raises(KeyError) as exc_info:
            registry.register(workflow)
        
        # Error message must identify the missing step
        error_message = str(exc_info.value)
        assert invalid_step_name in error_message, \
            f"Error message must identify missing step '{invalid_step_name}'"
        assert "non-existent step" in error_message.lower() or \
               "not found" in error_message.lower() or \
               "does not exist" in error_message.lower(), \
            "Error message must indicate step doesn't exist"

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        valid_step_names=st.lists(
            step_names,
            min_size=2,
            max_size=10,
            unique=True
        ),
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_accepts_all_valid_steps(
        self, workflow_name, valid_step_names, desc
    ):
        """
        Property 9: Workflow Step Validation (Valid Case)
        
        Verifies that workflow registration succeeds when all referenced
        steps exist in the step registry.
        """
        # Create step registry with valid steps
        step_registry = create_step_registry(valid_step_names)
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow using only valid steps
        # Build a linear chain through all steps
        graph = {}
        graph["start"] = GraphNode(
            step_name=valid_step_names[0],
            next_nodes=["node1"] if len(valid_step_names) > 1 else ["end"]
        )
        
        for i in range(1, len(valid_step_names)):
            node_name = f"node{i}"
            next_node = f"node{i+1}" if i < len(valid_step_names) - 1 else "end"
            graph[node_name] = GraphNode(
                step_name=valid_step_names[i],
                next_nodes=[next_node]
            )
        
        graph["end"] = GraphNode(step_name="end", next_nodes=[])
        
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph=graph
        )
        
        # Registration must succeed
        registry.register(workflow)
        
        # Workflow must be retrievable
        assert registry.has(workflow_name)
        retrieved = registry.get(workflow_name)
        assert retrieved.name == workflow_name

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        valid_step_names=st.lists(
            step_names,
            min_size=3,
            max_size=10,
            unique=True
        ),
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_validates_all_nodes(
        self, workflow_name, valid_step_names, desc
    ):
        """
        Property 9: Workflow Step Validation (All Nodes)
        
        Verifies that validation checks ALL nodes in the graph,
        not just the start node.
        """
        # Create step registry with some valid steps
        step_registry = create_step_registry(valid_step_names[:2])
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow where middle node references invalid step
        # valid_step_names[2] is NOT in step_registry
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph={
                "start": GraphNode(
                    step_name=valid_step_names[0],
                    next_nodes=["middle"]
                ),
                "middle": GraphNode(
                    step_name=valid_step_names[2],  # Invalid!
                    next_nodes=["end"]
                ),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Registration must fail
        with pytest.raises(KeyError) as exc_info:
            registry.register(workflow)
        
        # Error must identify the invalid step
        error_message = str(exc_info.value)
        assert valid_step_names[2] in error_message

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        valid_step_names=st.lists(
            step_names,
            min_size=1,
            max_size=10,
            unique=True
        ),
        invalid_step_names=st.lists(
            step_names,
            min_size=1,
            max_size=5,
            unique=True
        ),
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_rejects_any_invalid_step(
        self, workflow_name, valid_step_names, invalid_step_names, desc
    ):
        """
        Property 9: Workflow Step Validation (Multiple Invalid)
        
        Verifies that if ANY step in the workflow is invalid,
        registration fails.
        """
        # Ensure invalid steps are not in valid steps
        assume(not any(name in valid_step_names for name in invalid_step_names))
        
        # Create step registry with valid steps
        step_registry = create_step_registry(valid_step_names)
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow mixing valid and invalid steps
        # Use first valid step for start, first invalid for next node
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph={
                "start": GraphNode(
                    step_name=valid_step_names[0],
                    next_nodes=["invalid_node"]
                ),
                "invalid_node": GraphNode(
                    step_name=invalid_step_names[0],  # Invalid!
                    next_nodes=["end"]
                ),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Registration must fail
        with pytest.raises(KeyError):
            registry.register(workflow)

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        step_name=step_names,
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_validation_error_is_descriptive(
        self, workflow_name, step_name, desc
    ):
        """
        Property 9: Workflow Step Validation (Error Quality)
        
        Verifies that validation errors provide helpful information
        including the workflow name, node name, and missing step name.
        """
        # Create empty step registry (only 'end' step)
        step_registry = create_step_registry([])
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow with invalid step
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph={
                "start": GraphNode(step_name=step_name, next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            }
        )
        
        # Registration must fail with descriptive error
        with pytest.raises(KeyError) as exc_info:
            registry.register(workflow)
        
        error_message = str(exc_info.value)
        
        # Error should mention workflow name
        assert workflow_name in error_message, \
            "Error must identify which workflow failed validation"
        
        # Error should mention the missing step
        assert step_name in error_message, \
            "Error must identify which step is missing"
        
        # Error should mention it's a step validation issue
        assert any(keyword in error_message.lower() for keyword in [
            "step", "non-existent", "not found", "does not exist", "missing"
        ]), "Error must indicate this is a step validation issue"

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflows=st.lists(
            st.tuples(workflow_names, step_names, descriptions),
            min_size=1,
            max_size=10,
            unique_by=lambda x: x[0]  # Unique by workflow name
        )
    )
    @settings(max_examples=100)
    def test_workflow_validation_consistency(self, workflows):
        """
        Property 9: Workflow Step Validation (Consistency)
        
        Verifies that validation is consistent - the same workflow
        definition always produces the same validation result.
        """
        # Separate workflows into those with valid and invalid steps
        valid_step_names = [step_name for _, step_name, _ in workflows[:len(workflows)//2]]
        
        # Create step registry
        step_registry = create_step_registry(valid_step_names)
        registry = WorkflowRegistry(step_registry)
        
        # Test each workflow multiple times
        for workflow_name, step_name, desc in workflows:
            workflow = WorkflowDefinition(
                name=workflow_name,
                description=desc,
                graph=create_simple_workflow_graph(step_name)
            )
            
            # Determine if step is valid
            is_valid = step_name in valid_step_names
            
            if is_valid:
                # Should succeed consistently
                registry.register(workflow)
                assert registry.has(workflow_name)
                # Unregister for next iteration
                registry.unregister(workflow_name)
            else:
                # Should fail consistently
                with pytest.raises(KeyError):
                    registry.register(workflow)
                assert not registry.has(workflow_name)

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        valid_step_names=st.lists(
            step_names,
            min_size=1,
            max_size=10,
            unique=True
        ),
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_validation_before_registration(
        self, workflow_name, valid_step_names, desc
    ):
        """
        Property 9: Workflow Step Validation (Timing)
        
        Verifies that validation occurs BEFORE the workflow is added
        to the registry, so failed validations don't pollute the registry.
        """
        # Create step registry with some steps
        step_registry = create_step_registry(valid_step_names[:1])
        registry = WorkflowRegistry(step_registry)
        
        # Get initial workflow count
        initial_count = len(registry.list_all())
        
        # Try to register workflow with invalid step
        # valid_step_names[1] is NOT in step_registry (if it exists)
        if len(valid_step_names) > 1:
            workflow = WorkflowDefinition(
                name=workflow_name,
                description=desc,
                graph={
                    "start": GraphNode(
                        step_name=valid_step_names[1],  # Invalid!
                        next_nodes=["end"]
                    ),
                    "end": GraphNode(step_name="end", next_nodes=[])
                }
            )
            
            # Registration must fail
            with pytest.raises(KeyError):
                registry.register(workflow)
            
            # Workflow must NOT be in registry
            assert not registry.has(workflow_name)
            assert len(registry.list_all()) == initial_count, \
                "Failed registration must not add workflow to registry"

    # Feature: extensible-orchestration-phase-b, Property 9: Workflow Step Validation
    @given(
        workflow_name=workflow_names,
        step_name=step_names,
        desc=descriptions
    )
    @settings(max_examples=100)
    def test_workflow_validation_with_end_step(
        self, workflow_name, step_name, desc
    ):
        """
        Property 9: Workflow Step Validation (End Step)
        
        Verifies that the 'end' step is always considered valid,
        even if not explicitly in the step registry.
        """
        # Create step registry with one step
        step_registry = create_step_registry([step_name])
        registry = WorkflowRegistry(step_registry)
        
        # Create workflow that references 'end' in next_nodes
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=desc,
            graph={
                "start": GraphNode(step_name=step_name, next_nodes=["end"])
                # Note: 'end' node not in graph, but referenced in next_nodes
            }
        )
        
        # Registration should succeed - 'end' is a special case
        registry.register(workflow)
        assert registry.has(workflow_name)
