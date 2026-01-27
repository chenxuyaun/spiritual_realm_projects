"""
Workflow Registry - Central registry for workflow definitions.

This module provides the WorkflowRegistry that manages workflow definitions,
validates workflow graphs, and enables workflow discovery by name.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from mm_orch.orchestration.graph_executor import GraphNode
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class WorkflowDefinition:
    """
    Definition of a workflow graph.
    
    Attributes:
        name: Unique workflow identifier (e.g., "search_qa", "lesson_pack")
        description: Human-readable description of workflow purpose
        graph: Dictionary mapping node names to GraphNode instances
        required_capabilities: List of capabilities needed (e.g., ["search", "summarize"])
        tags: Optional tags for categorization (e.g., ["qa", "fast"])
        metadata: Optional additional metadata
    
    Example:
        definition = WorkflowDefinition(
            name="search_qa",
            description="Search-based question answering",
            graph={
                "start": GraphNode(step_name="web_search", next_nodes=["fetch"]),
                "fetch": GraphNode(step_name="fetch_url", next_nodes=["summarize"]),
                "summarize": GraphNode(step_name="summarize", next_nodes=["answer"]),
                "answer": GraphNode(step_name="answer_generate", next_nodes=["end"]),
                "end": GraphNode(step_name="end", next_nodes=[])
            },
            required_capabilities=["search", "fetch", "summarize", "generate"]
        )
    """
    
    name: str
    description: str
    graph: Dict[str, GraphNode]
    required_capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowRegistry:
    """
    Registry for workflow definitions.
    
    The registry:
    1. Stores workflow definitions by name
    2. Validates workflow graphs during registration
    3. Provides methods to retrieve and list workflows
    4. Ensures all referenced steps exist in the step registry
    
    Example:
        # Create registry
        registry = WorkflowRegistry(step_registry)
        
        # Register workflow
        registry.register(workflow_definition)
        
        # Retrieve workflow
        workflow = registry.get("search_qa")
        
        # List all workflows
        names = registry.list_all()
    """
    
    def __init__(self, step_registry: Dict[str, Any]):
        """
        Initialize workflow registry.
        
        Args:
            step_registry: Dictionary mapping step names to Step instances
                          Used to validate that workflow steps exist
        """
        self.step_registry = step_registry
        self._workflows: Dict[str, WorkflowDefinition] = {}
        
        logger.info("WorkflowRegistry initialized")
    
    def register(self, definition: WorkflowDefinition) -> None:
        """
        Register a workflow definition with validation.
        
        Validates:
        - Workflow name is unique
        - Graph structure is valid (has start node, no dangling references)
        - All referenced steps exist in step registry
        
        Args:
            definition: WorkflowDefinition to register
        
        Raises:
            ValueError: If workflow name already exists or graph is invalid
            KeyError: If referenced steps don't exist in step registry
        """
        # Check for duplicate name
        if definition.name in self._workflows:
            raise ValueError(
                f"Workflow '{definition.name}' is already registered. "
                f"Use a different name or unregister the existing workflow first."
            )
        
        # Validate graph structure
        self._validate_workflow(definition)
        
        # Register workflow
        self._workflows[definition.name] = definition
        
        logger.info(
            f"Registered workflow '{definition.name}': {definition.description} "
            f"({len(definition.graph)} nodes, {len(definition.required_capabilities)} capabilities)"
        )
    
    def get(self, name: str) -> WorkflowDefinition:
        """
        Retrieve workflow definition by name.
        
        Args:
            name: Workflow name
        
        Returns:
            WorkflowDefinition for the requested workflow
        
        Raises:
            KeyError: If workflow name not found
        """
        if name not in self._workflows:
            available = ", ".join(self._workflows.keys()) if self._workflows else "none"
            raise KeyError(
                f"Workflow '{name}' not found in registry. "
                f"Available workflows: {available}"
            )
        
        return self._workflows[name]
    
    def list_all(self) -> List[str]:
        """
        List all registered workflow names.
        
        Returns:
            List of workflow names in registration order
        """
        return list(self._workflows.keys())
    
    def has(self, name: str) -> bool:
        """
        Check if a workflow is registered.
        
        Args:
            name: Workflow name to check
        
        Returns:
            True if workflow exists, False otherwise
        """
        return name in self._workflows
    
    def unregister(self, name: str) -> None:
        """
        Remove a workflow from the registry.
        
        Args:
            name: Workflow name to remove
        
        Raises:
            KeyError: If workflow name not found
        """
        if name not in self._workflows:
            raise KeyError(f"Workflow '{name}' not found in registry")
        
        del self._workflows[name]
        logger.info(f"Unregistered workflow '{name}'")
    
    def get_by_capability(self, capability: str) -> List[WorkflowDefinition]:
        """
        Find workflows that require a specific capability.
        
        Args:
            capability: Capability to search for (e.g., "search", "summarize")
        
        Returns:
            List of WorkflowDefinitions that require the capability
        """
        return [
            definition
            for definition in self._workflows.values()
            if capability in definition.required_capabilities
        ]
    
    def get_by_tag(self, tag: str) -> List[WorkflowDefinition]:
        """
        Find workflows with a specific tag.
        
        Args:
            tag: Tag to search for (e.g., "fast", "qa")
        
        Returns:
            List of WorkflowDefinitions with the tag
        """
        return [
            definition
            for definition in self._workflows.values()
            if tag in definition.tags
        ]
    
    def _validate_workflow(self, definition: WorkflowDefinition) -> None:
        """
        Validate workflow definition.
        
        Checks:
        - Graph has a "start" node
        - All referenced next_nodes exist in graph or are "end"
        - All referenced steps exist in step registry
        - No dangling node references
        
        Args:
            definition: WorkflowDefinition to validate
        
        Raises:
            ValueError: If graph structure is invalid
            KeyError: If referenced steps don't exist
        """
        graph = definition.graph
        
        # Check for start node
        if "start" not in graph:
            raise ValueError(
                f"Workflow '{definition.name}' graph must have a 'start' node"
            )
        
        # Check all nodes and their references
        for node_name, node in graph.items():
            # Check step exists in registry
            if node.step_name not in self.step_registry:
                available_steps = ", ".join(list(self.step_registry.keys())[:10])
                if len(self.step_registry) > 10:
                    available_steps += ", ..."
                
                raise KeyError(
                    f"Workflow '{definition.name}' node '{node_name}' references "
                    f"non-existent step '{node.step_name}'. "
                    f"Available steps: {available_steps}"
                )
            
            # Check all next_nodes exist
            for next_node_name in node.next_nodes:
                if next_node_name != "end" and next_node_name not in graph:
                    raise ValueError(
                        f"Workflow '{definition.name}' node '{node_name}' references "
                        f"non-existent next node '{next_node_name}'"
                    )
        
        logger.debug(f"Validated workflow '{definition.name}' graph structure")


# Global registry instance
_global_workflow_registry: Optional[WorkflowRegistry] = None


def get_workflow_registry(step_registry: Optional[Dict[str, Any]] = None) -> WorkflowRegistry:
    """
    Get or create the global workflow registry instance.
    
    Args:
        step_registry: Step registry to use (required on first call)
    
    Returns:
        Global WorkflowRegistry instance
    
    Raises:
        ValueError: If step_registry not provided on first call
    """
    global _global_workflow_registry
    
    if _global_workflow_registry is None:
        if step_registry is None:
            raise ValueError(
                "step_registry must be provided when creating workflow registry"
            )
        _global_workflow_registry = WorkflowRegistry(step_registry)
    
    return _global_workflow_registry


def reset_workflow_registry() -> None:
    """
    Reset the global workflow registry instance.
    
    Useful for testing to ensure clean state.
    """
    global _global_workflow_registry
    _global_workflow_registry = None
