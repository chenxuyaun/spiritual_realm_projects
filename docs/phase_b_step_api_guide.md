# Phase B Step API Guide

## Overview

The Step API provides a unified interface for creating composable workflow components in the Phase B orchestration system. Steps are atomic execution units that transform State objects through well-defined input/output contracts.

## Step Interface

Every step must implement the Step protocol:

```python
from typing import Protocol, List, Any

class Step(Protocol):
    """Unified interface for all workflow steps."""
    
    name: str  # Unique identifier for the step
    input_keys: List[str]  # Required State fields
    output_keys: List[str]  # Fields this step produces
    
    def run(self, state: State, runtime: Runtime) -> State:
        """Execute step logic and return updated state."""
        ...
```

## Creating Steps

### Method 1: Class-Based Steps

For complex steps with initialization logic:

```python
from mm_orch.orchestration.base_step import BaseStep
from mm_orch.orchestration import State
from typing import Dict, Any

class MyCustomStep(BaseStep):
    name = "my_custom_step"
    input_keys = ["question", "context"]
    output_keys = ["result"]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """Implement your step logic here."""
        question = state["question"]
        context = state.get("context", "")
        
        # Your processing logic
        result = self.process(question, context)
        
        return {"result": result}
    
    def process(self, question: str, context: str) -> str:
        """Helper method for processing."""
        return f"Processed: {question} with {context}"
```

### Method 2: Function-Based Steps

For simple steps without state:

```python
from dataclasses import dataclass
from mm_orch.orchestration import State

@dataclass
class SimpleFunctionStep:
    name: str = "simple_step"
    input_keys: List[str] = field(default_factory=lambda: ["input"])
    output_keys: List[str] = field(default_factory=lambda: ["output"])
    
    def run(self, state: State, runtime: Any) -> State:
        input_value = state["input"]
        output_value = input_value.upper()
        
        return {**state, "output": output_value}
```

## Step Best Practices

### 1. Input Validation

Always validate required inputs:

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    # BaseStep automatically validates input_keys exist
    # Add custom validation as needed
    question = state["question"]
    
    if not question or len(question) < 3:
        raise ValueError("Question must be at least 3 characters")
    
    # Process...
```

### 2. Partial State Updates

Return only the fields you're updating:

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    # Don't copy entire state - just return new fields
    return {
        "result": "my result",
        "metadata": {"processed_at": time.time()}
    }
```

### 3. Error Handling

Let exceptions propagate - the Graph Executor will handle them:

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    try:
        result = self.risky_operation()
        return {"result": result}
    except SpecificError as e:
        # Log if needed, then re-raise
        logger.error(f"Operation failed: {e}")
        raise
```

### 4. Cost Tracking

Use the cost tracker for resource monitoring:

```python
from mm_orch.orchestration.cost_tracker import CostTracker

def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    tracker = CostTracker()
    
    with tracker.track_step(self.name):
        # Your processing
        result = self.process()
    
    # Cost metrics automatically recorded in state.meta
    return {"result": result}
```

## Built-in Steps

### WebSearchStep

Performs web search using DuckDuckGo:

```python
from mm_orch.orchestration.workflow_steps import WebSearchStep

step = WebSearchStep(search_tool=search_tool, max_results=5)
```

**Input**: `question`  
**Output**: `search_results` (list of SearchResult objects)

### FetchUrlStep

Fetches content from URLs:

```python
from mm_orch.orchestration.workflow_steps import FetchUrlStep

step = FetchUrlStep(fetch_tool=fetch_tool, max_content_length=5000)
```

**Input**: `search_results`  
**Output**: `docs` (dict mapping URL to content)

### SummarizeStep

Summarizes documents:

```python
from mm_orch.orchestration.workflow_steps import SummarizeStep

step = SummarizeStep(model_name="t5-small", max_summary_length=200)
```

**Input**: `docs`  
**Output**: `summaries` (dict mapping URL to summary)

### AnswerGenerateStep

Generates final answer:

```python
from mm_orch.orchestration.workflow_steps import AnswerGenerateStep

step = AnswerGenerateStep(model_name="gpt2", max_context_length=1000)
```

**Input**: `question`, `summaries`  
**Output**: `final_answer`, `citations`

## Registering Steps

Steps must be registered before use:

```python
from mm_orch.registries.step_registry import StepRegistry

registry = StepRegistry()

# Register built-in steps
registry.register("search", WebSearchStep(search_tool))
registry.register("fetch", FetchUrlStep(fetch_tool))

# Register custom steps
registry.register("my_step", MyCustomStep(config))
```

## Testing Steps

### Unit Testing

Test steps in isolation:

```python
def test_my_custom_step():
    step = MyCustomStep(config={})
    
    initial_state = {
        "question": "Test question",
        "context": "Test context"
    }
    
    result_state = step.run(initial_state, runtime=None)
    
    assert "result" in result_state
    assert result_state["result"].startswith("Processed:")
```

### Integration Testing

Test steps in workflows:

```python
def test_step_in_workflow():
    executor = GraphExecutor(step_registry)
    
    graph = {
        "start": GraphNode(step_name="my_step", next_nodes=["end"]),
        "end": GraphNode(step_name="my_step", next_nodes=[])
    }
    
    final_state = executor.execute(graph, initial_state, runtime)
    
    assert "result" in final_state
```

## Common Patterns

### Conditional Processing

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    if state.get("use_cache"):
        return {"result": self.get_cached_result()}
    else:
        return {"result": self.compute_result()}
```

### Batch Processing

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    items = state["items"]
    results = []
    
    for item in items:
        result = self.process_item(item)
        results.append(result)
    
    return {"results": results}
```

### Model Inference

```python
def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
    model = runtime.model_manager.get_model(self.model_name)
    
    input_text = state["input"]
    output = model.generate(input_text)
    
    return {"output": output}
```

## Next Steps

- Learn about [Graph Executor](./phase_b_graph_executor_guide.md)
- Explore [Workflow Registry](./phase_b_workflow_registry_guide.md)
- Read [Router Training](./phase_b_router_training_guide.md)
