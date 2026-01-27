# Phase B Registry System Guide

## Overview

The Registry System provides centralized discovery and management for tools, models, and workflows in Phase B. Registries enable dynamic component lookup, capability-based queries, and metadata-driven decisions.

## Tool Registry

### Registration

```python
from mm_orch.registries.tool_registry import ToolRegistry, ToolMetadata

registry = ToolRegistry()

# Register a tool
metadata = ToolMetadata(
    name="web_search",
    capabilities=["search", "web"],
    description="DuckDuckGo web search",
    parameters={"max_results": 10}
)

registry.register("web_search", search_tool, metadata)
```

### Retrieval

```python
# Get by name
tool = registry.get("web_search")

# Find by capability
search_tools = registry.find_by_capability("search")
```

## Model Registry

### Registration

```python
from mm_orch.registries.model_registry import ModelRegistry, ModelMetadata

registry = ModelRegistry()

metadata = ModelMetadata(
    name="t5-small",
    capabilities=["summarize", "generate"],
    expected_vram_mb=500,
    supports_quant=True,
    preferred_device_policy="gpu_on_demand",
    model_path="t5-small"
)

registry.register(metadata)
```

### Querying

```python
# Get by name
model_meta = registry.get("t5-small")

# Find by capability
summarizers = registry.find_by_capability("summarize")

# Check VRAM requirements
total_vram = sum(m.expected_vram_mb for m in summarizers)
```

## Workflow Registry

### Registration

```python
from mm_orch.registries.workflow_registry import WorkflowRegistry, WorkflowDefinition
from mm_orch.orchestration import GraphNode

registry = WorkflowRegistry(step_registry)

definition = WorkflowDefinition(
    name="search_qa",
    description="Search-based Q&A workflow",
    graph={
        "start": GraphNode(step_name="search", next_nodes=["fetch"]),
        "fetch": GraphNode(step_name="fetch", next_nodes=["answer"]),
        "answer": GraphNode(step_name="answer", next_nodes=["end"]),
        "end": GraphNode(step_name="answer", next_nodes=[])
    },
    required_capabilities=["search", "generate"]
)

registry.register(definition)
```

### Retrieval

```python
# Get workflow
workflow = registry.get("search_qa")

# List all workflows
all_workflows = registry.list_all()

# Execute workflow
executor = GraphExecutor(step_registry)
result = executor.execute(workflow.graph, initial_state, runtime)
```

## Best Practices

1. **Register at startup**: Register all components during system initialization
2. **Validate metadata**: Ensure all required fields are present
3. **Use capabilities**: Query by capability for flexibility
4. **Document parameters**: Include parameter schemas in metadata

## See Also

- [Step API Guide](./phase_b_step_api_guide.md)
- [Router Training Guide](./phase_b_router_training_guide.md)
