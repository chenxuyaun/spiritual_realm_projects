# Phase B Compatibility Layer

## Overview

The compatibility layer ensures that Phase A workflows (BaseWorkflow subclasses) continue to work seamlessly in the Phase B environment (Step-based graph execution). This maintains backward compatibility while enabling gradual migration to the new architecture.

## Architecture

### Key Components

1. **LegacyWorkflowAdapter** (`mm_orch/orchestration/legacy_adapter.py`)
   - Wraps Phase A BaseWorkflow as Phase B Step
   - Handles parameter mapping between State and workflow parameters
   - Injects runtime dependencies automatically
   - Converts WorkflowResult to State updates

2. **Compatibility Helpers** (`mm_orch/orchestration/compatibility.py`)
   - LegacyRuntime: Runtime context for legacy workflows
   - LegacyWorkflowExecutor: High-level executor for legacy workflows
   - Conversion functions for State ↔ Parameters

3. **Automatic Parameter Mapping**
   - `question` (State) → `query` (workflow parameter)
   - `lesson_topic` (State) → `topic` (workflow parameter)
   - `conversation_id` (State) → `session_id` (workflow parameter)

## Usage Examples

### Basic Adapter Usage

```python
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.orchestration.legacy_adapter import LegacyWorkflowAdapter
from mm_orch.orchestration.compatibility import LegacyRuntime

# Create workflow
workflow = SearchQAWorkflow()

# Wrap as Phase B step
adapter = LegacyWorkflowAdapter(workflow)

# Execute with Phase B State
state = {
    "question": "What is Python?",
    "meta": {"max_results": 5}
}

runtime = LegacyRuntime()
result_state = adapter.run(state, runtime)

print(result_state["final_answer"])
```

### Using in Graph Executor

```python
from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode

# Register adapted workflow as step
step_registry = {
    "search_qa": LegacyWorkflowAdapter(SearchQAWorkflow())
}

# Define graph
graph = {
    "start": GraphNode(step_name="search_qa", next_nodes=[])
}

# Execute
executor = GraphExecutor(step_registry)
final_state = executor.execute(graph, initial_state, runtime)
```

### Registering Multiple Workflows

```python
from mm_orch.orchestration.legacy_adapter import register_legacy_workflows
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.workflows.lesson_pack import LessonPackWorkflow

workflows = [
    SearchQAWorkflow(),
    LessonPackWorkflow()
]

step_registry = {}
register_legacy_workflows(step_registry, workflows)

# Now step_registry contains:
# {"SearchQA": <adapter>, "LessonPack": <adapter>}
```

### Using LegacyWorkflowExecutor

```python
from mm_orch.orchestration.compatibility import LegacyWorkflowExecutor

executor = LegacyWorkflowExecutor()

# Execute with Phase A parameters
result_state = executor.execute(
    workflow=SearchQAWorkflow(),
    parameters={"query": "What is Python?"}
)

# Or execute with Phase B state
result_state = executor.execute(
    workflow=SearchQAWorkflow(),
    state={"question": "What is Python?", "meta": {}}
)

# Or get Phase A WorkflowResult
result = executor.execute_with_result(
    workflow=SearchQAWorkflow(),
    parameters={"query": "What is Python?"}
)
```

## Parameter Mapping

The adapter automatically maps common State fields to workflow parameters:

| State Field | Workflow Parameter | Workflows |
|-------------|-------------------|-----------|
| `question` | `query` | SearchQA, RAGQA |
| `question` | `message` | ChatGenerate |
| `lesson_topic` | `topic` | LessonPack |
| `conversation_id` | `session_id` | ChatGenerate |

Additional parameters from `state["meta"]` are passed through to the workflow.

## Runtime Dependency Injection

The adapter automatically injects runtime dependencies into workflows:

- `model_manager` - For model inference
- `real_model_manager` - For actual LLM inference
- `inference_engine` - For generation
- `conversation_manager` - For chat formatting
- `search_tool` - For web search
- `fetch_tool` - For URL fetching
- `vector_db` - For RAG workflows
- `chat_storage` - For chat history

## State Conversion

### WorkflowResult → State

The adapter converts WorkflowResult to State updates:

```python
# WorkflowResult
result = WorkflowResult(
    result="Answer text",
    metadata={
        "workflow": "SearchQA",
        "sources": [{"url": "https://example.com"}]
    },
    status="success",
    execution_time=1.5
)

# Converted to State updates
{
    "final_answer": "Answer text",
    "citations": ["https://example.com"],
    "workflow_result": {...},  # Full result for reference
    "meta": {
        "workflow_name": "SearchQA",
        "workflow_status": "success",
        "workflow_execution_time": 1.5,
        "workflow_metadata": {...}
    }
}
```

### Workflow-Specific Conversions

- **SearchQA**: Extracts `citations` from sources
- **LessonPack**: Extracts `lesson_outline`, `teaching_text`, `exercises`
- **RAGQA**: Extracts `kb_sources` and `citations`
- **ChatGenerate**: Preserves `conversation_id`

## Testing

### Unit Tests

Located in `tests/unit/test_legacy_compatibility.py`:

- Adapter creation and execution
- Parameter mapping
- Runtime injection
- Error handling
- State preservation
- Conversion functions

### Integration Tests

Located in `tests/integration/test_phase_a_phase_b_compatibility.py`:

- Real workflow execution (SearchQA, LessonPack, ChatGenerate, RAGQA)
- Graph execution with adapted workflows
- Workflow registration
- End-to-end compatibility
- Backward compatibility verification

All tests pass, confirming Phase A workflows work correctly in Phase B.

## Migration Path

### Phase 1: Use Adapters (Current)

```python
# Existing Phase A code continues to work
workflow = SearchQAWorkflow()
result = workflow.run({"query": "What is Python?"})

# New Phase B code uses adapters
adapter = LegacyWorkflowAdapter(workflow)
state = adapter.run({"question": "What is Python?"}, runtime)
```

### Phase 2: Gradual Migration

Convert workflows to native Phase B steps over time:

```python
# Old: Phase A workflow
class SearchQAWorkflow(BaseWorkflow):
    def execute(self, parameters):
        ...

# New: Phase B step
class SearchQAStep(BaseStep):
    name = "search_qa"
    input_keys = ["question"]
    output_keys = ["final_answer"]
    
    def execute(self, state, runtime):
        ...
```

### Phase 3: Full Phase B

Eventually all workflows become native Phase B steps, and adapters are removed.

## Benefits

1. **Zero Breaking Changes**: Existing Phase A code works without modification
2. **Gradual Migration**: Teams can migrate at their own pace
3. **Mixed Execution**: Phase A and Phase B components work together
4. **Full Feature Parity**: All Phase A features available in Phase B
5. **Comprehensive Testing**: Both unit and integration tests ensure compatibility

## Limitations

1. **Performance Overhead**: Adapter adds minimal overhead for parameter conversion
2. **State Mapping**: Some workflow-specific fields may need manual mapping
3. **Deprecation Path**: Eventually adapters should be removed in favor of native steps

## Conclusion

The compatibility layer successfully bridges Phase A and Phase B, ensuring:
- ✅ All Phase A workflows work in Phase B environment
- ✅ No code changes required for existing workflows
- ✅ Smooth migration path to native Phase B steps
- ✅ Full test coverage with 29 passing tests (18 unit + 11 integration)

This enables teams to adopt Phase B incrementally while maintaining full backward compatibility.
