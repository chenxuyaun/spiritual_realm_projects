# Phase A to Phase B Migration Guide

## Overview

This guide helps you migrate from Phase A (fixed pipeline) to Phase B (extensible orchestration) while maintaining backward compatibility.

## Migration Strategy

Phase B is designed for **gradual migration**. You can:
1. Run Phase A and Phase B side-by-side
2. Migrate workflows one at a time
3. Fall back to Phase A if issues arise

## Quick Start

### Enable Phase B

```python
from mm_orch.orchestration.phase_b_orchestrator import get_phase_b_orchestrator

# Initialize Phase B orchestrator
orchestrator = get_phase_b_orchestrator()

# Process requests (automatically uses Phase B if available)
result = orchestrator.process_request(user_request)
```

### CLI Usage

```bash
# Use Phase B
python -m mm_orch.main --phase-b "What is Python?"

# Use Phase A (default)
python -m mm_orch.main "What is Python?"
```

## Step-by-Step Migration

### Step 1: Understand Phase B Architecture

**Phase A**: Fixed pipeline with hardcoded steps  
**Phase B**: Graph-based execution with registries

Key differences:
- Steps are now composable units
- Workflows are defined as graphs
- Routers can be trained
- Comprehensive tracing

### Step 2: Convert Workflows to Graphs

**Phase A Workflow:**

```python
def search_qa_workflow(question):
    results = web_search(question)
    docs = fetch_urls(results)
    summaries = summarize(docs)
    answer = generate_answer(question, summaries)
    return answer
```

**Phase B Workflow:**

```python
from mm_orch.orchestration import GraphNode
from mm_orch.registries.workflow_registry import WorkflowDefinition

definition = WorkflowDefinition(
    name="search_qa",
    description="Search-based Q&A",
    graph={
        "start": GraphNode(step_name="search", next_nodes=["fetch"]),
        "fetch": GraphNode(step_name="fetch", next_nodes=["summarize"]),
        "summarize": GraphNode(step_name="summarize", next_nodes=["answer"]),
        "answer": GraphNode(step_name="answer", next_nodes=["end"]),
        "end": GraphNode(step_name="answer", next_nodes=[])
    },
    required_capabilities=["search", "generate"]
)
```

### Step 3: Register Components

**Create registries:**

```python
from mm_orch.registries import ToolRegistry, ModelRegistry, WorkflowRegistry

# Tool registry
tool_registry = ToolRegistry()
tool_registry.register("web_search", search_tool, metadata)

# Model registry
model_registry = ModelRegistry()
model_registry.register(model_metadata)

# Workflow registry
workflow_registry = WorkflowRegistry(step_registry)
workflow_registry.register(workflow_definition)
```

### Step 4: Update Step Implementations

**Phase A Step:**

```python
def summarize_step(docs):
    summaries = {}
    for url, content in docs.items():
        summary = model.summarize(content)
        summaries[url] = summary
    return summaries
```

**Phase B Step:**

```python
from mm_orch.orchestration.base_step import BaseStep

class SummarizeStep(BaseStep):
    name = "summarize"
    input_keys = ["docs"]
    output_keys = ["summaries"]
    
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        docs = state["docs"]
        summaries = {}
        
        for url, content in docs.items():
            model = runtime.model_manager.get_model(self.model_name)
            summary = model.summarize(content)
            summaries[url] = summary
        
        return {"summaries": summaries}
```

### Step 5: Enable Tracing

```python
from mm_orch.observability.tracer import Tracer

tracer = Tracer("data/traces/traces.jsonl")

# Tracer is automatically used by GraphExecutor
executor = GraphExecutor(step_registry, tracer=tracer)
```

### Step 6: Train Routers (Optional)

```bash
# Collect traces
python -m mm_orch.main --phase-b --trace "Question 1"
python -m mm_orch.main --phase-b --trace "Question 2"

# Train router
python scripts/train_router_v2.py \
    --traces data/traces/traces.jsonl \
    --output models/router_v2
```

## Compatibility Layer

Phase B includes a compatibility layer for Phase A code:

```python
from mm_orch.orchestration.compatibility import LegacyWorkflowAdapter

# Wrap Phase A workflow
adapter = LegacyWorkflowAdapter(legacy_workflow_function)

# Use as Phase B step
step_registry.register("legacy_workflow", adapter)
```

## Configuration Changes

### Phase A Config

```yaml
# config/system.yaml
workflows:
  - search_qa
  - rag_qa
  - lesson_pack
```

### Phase B Config

```yaml
# config/phase_b.yaml
registries:
  tool_registry:
    enabled: true
  model_registry:
    enabled: true
  workflow_registry:
    enabled: true

router:
  version: v3
  model_path: models/router_v3

tracer:
  enabled: true
  output_path: data/traces/traces.jsonl
```

## Troubleshooting

### Issue: Phase B Not Loading

**Symptom**: System falls back to Phase A

**Solution**:
1. Check Phase B components are installed
2. Verify configuration files exist
3. Check logs for initialization errors

```python
from mm_orch.orchestration.phase_b_orchestrator import get_phase_b_orchestrator

orchestrator = get_phase_b_orchestrator()
stats = orchestrator.get_statistics()
print(stats)  # Check which components are available
```

### Issue: Workflow Not Found

**Symptom**: `WorkflowNotFoundError`

**Solution**:
1. Verify workflow is registered
2. Check workflow name spelling
3. Ensure step registry has all required steps

```python
workflow_registry = get_workflow_registry()
print(workflow_registry.list_all())  # List registered workflows
```

### Issue: Step Execution Fails

**Symptom**: `RuntimeError: Step 'X' failed`

**Solution**:
1. Check step input_keys are in State
2. Verify runtime has required components
3. Review trace for error details

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Issue: Router Predictions Incorrect

**Symptom**: Wrong workflow selected

**Solution**:
1. Collect more training data
2. Retrain router with balanced dataset
3. Fall back to Router v1 (rule-based)

```python
# Use specific router version
from mm_orch.routing.router_v1 import RouterV1
router = RouterV1()  # Always works, no training needed
```

## Performance Considerations

### Phase B Overhead

Phase B adds minimal overhead (~1-2ms per workflow):
- Step validation: <0.5ms
- State copying: <0.5ms
- Tracing: <1ms

### Optimization Tips

1. **Reuse registries**: Create once, use many times
2. **Cache models**: Use model manager's caching
3. **Batch operations**: Process multiple requests together
4. **Disable tracing**: For production, trace sampling only

```python
# Sample 10% of requests
if random.random() < 0.1:
    executor = GraphExecutor(step_registry, tracer=tracer)
else:
    executor = GraphExecutor(step_registry)  # No tracing
```

## Testing Migration

### Unit Tests

```python
def test_phase_b_workflow():
    """Test Phase B workflow produces same output as Phase A."""
    question = "What is Python?"
    
    # Phase A
    phase_a_result = legacy_search_qa(question)
    
    # Phase B
    orchestrator = get_phase_b_orchestrator()
    phase_b_result = orchestrator.process_request(UserRequest(query=question))
    
    # Compare
    assert phase_b_result.answer == phase_a_result
```

### Integration Tests

```python
def test_backward_compatibility():
    """Test Phase A code works with Phase B."""
    from mm_orch.orchestration.compatibility import run_legacy_workflow
    
    result = run_legacy_workflow("search_qa", {"question": "Test"})
    assert result is not None
```

## Rollback Plan

If you need to rollback to Phase A:

1. **Disable Phase B flag**:
```python
orchestrator = get_phase_b_orchestrator()
orchestrator.using_phase_b = False
```

2. **Remove Phase B config**:
```bash
rm config/phase_b.yaml
```

3. **Use Phase A directly**:
```python
from mm_orch.orchestrator import Orchestrator  # Phase A
orchestrator = Orchestrator()
```

## Migration Checklist

- [ ] Read Phase B documentation
- [ ] Convert one workflow to graph format
- [ ] Test converted workflow
- [ ] Register workflow in registry
- [ ] Enable tracing
- [ ] Collect execution traces
- [ ] Train router (optional)
- [ ] Test with real traffic
- [ ] Monitor performance
- [ ] Migrate remaining workflows
- [ ] Update documentation
- [ ] Train team on Phase B

## Next Steps

After migration:
1. Explore new workflow variants (search_qa_fast, search_qa_strict_citations)
2. Train custom routers for your use cases
3. Implement custom steps for domain-specific logic
4. Set up monitoring dashboards using traces
5. Optimize workflows based on cost statistics

## Support

For migration help:
- Check [Phase B documentation](./phase_b_step_api_guide.md)
- Review [example workflows](../examples/phase_b_orchestration_demo.py)
- Open an issue on GitHub
- Contact the development team

## FAQ

**Q: Can I use Phase A and Phase B together?**  
A: Yes, the compatibility layer allows mixing both.

**Q: Do I need to retrain routers?**  
A: No, Router v1 (rule-based) works without training.

**Q: Will Phase A be deprecated?**  
A: Phase A will be maintained for backward compatibility.

**Q: How long does migration take?**  
A: Simple workflows: 1-2 hours. Complex workflows: 1-2 days.

**Q: What if I find a bug?**  
A: Report it and fall back to Phase A while we fix it.
