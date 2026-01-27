# Design Document: Extensible Orchestration Phase B

## Overview

Phase B transforms the MuAI orchestration system from a fixed pipeline architecture into an extensible, observable, and intelligent framework. The design introduces three core abstractions: a unified Step API for composable operations, a State-driven execution model for data flow, and a Graph Executor for flexible workflow composition. These foundations enable registry-based component management, comprehensive observability through structured tracing, and machine learning-driven routing that learns from execution feedback.

The architecture follows a layered approach: registries provide discovery and metadata, the graph executor orchestrates step execution with tracing, and routers evolve from rule-based (v1) to classifier-based (v2) to cost-aware (v3) decision making. This design maintains backward compatibility with Phase A while enabling future extensions like multi-agent collaboration and adaptive workflow optimization.

## Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│              (CLI, Chat REPL, API Endpoints)                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Routing Layer                           │
│         (Router v1/v2/v3, Mode Detection, Bandit)           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│          (Graph Executor, Workflow Registry, Trace)          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Step Layer                              │
│    (WebSearch, Fetch, Summarize, Generate, LessonPlan)      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  (Tool Registry, Model Registry, Model Manager, Vector DB)  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Steps handle logic, State carries data, Graph Executor manages flow
2. **Registry Pattern**: Centralized discovery for tools, models, and workflows
3. **Observability First**: Every execution produces structured traces for analysis and learning
4. **Progressive Enhancement**: Routers evolve from rules to ML without breaking existing code
5. **Resource Awareness**: Cost tracking and model lifecycle management optimize GPU usage

## Components and Interfaces

### Step Interface

```python
from typing import Protocol, List, Any
from dataclasses import dataclass

class Step(Protocol):
    """Unified interface for all workflow steps."""
    
    name: str
    input_keys: List[str]
    output_keys: List[str]
    
    def run(self, state: State, runtime: Runtime) -> State:
        """Execute step logic and return updated state."""
        ...
```

**Implementation Patterns:**

- **Function-based**: Simple steps can be functions decorated with metadata
- **Class-based**: Complex steps inherit from BaseStep with lifecycle hooks
- **Validation**: Steps validate required input_keys exist in State before execution
- **Idempotency**: Steps should be idempotent when possible for retry safety

### State Object

```python
from typing import TypedDict, Dict, List, Any, Optional

class State(TypedDict, total=False):
    """Workflow execution state."""
    
    # Core fields
    question: str
    search_results: List[Dict[str, str]]
    docs: Dict[str, str]  # url -> content
    summaries: Dict[str, str]  # url -> summary
    final_answer: str
    citations: List[str]
    
    # Lesson pack fields
    lesson_topic: str
    lesson_objectives: List[str]
    lesson_outline: List[str]
    board_plan: List[str]
    lesson_explain_structured: Dict[str, Any]
    teaching_text: str
    exercises: List[Dict[str, str]]
    
    # RAG fields
    kb_sources: List[Dict[str, Any]]
    memory_context: str
    
    # Metadata
    meta: Dict[str, Any]
```

**State Management:**

- **Immutability**: Steps create new State dicts rather than mutating in place
- **Partial Updates**: Only modified keys are updated, preserving other fields
- **Metadata**: `meta` dict stores execution context (mode, turn_index, router_version)
- **Serialization**: State must be JSON-serializable for trace logging

### Graph Executor

```python
@dataclass
class GraphNode:
    """Node in workflow graph."""
    step_name: str
    next_nodes: List[str]
    condition: Optional[Callable[[State], bool]] = None

class GraphExecutor:
    """Executes workflow graphs with tracing."""
    
    def __init__(self, step_registry: StepRegistry, tracer: Tracer):
        self.step_registry = step_registry
        self.tracer = tracer
    
    def execute(self, graph: Dict[str, GraphNode], 
                initial_state: State, 
                runtime: Runtime) -> State:
        """Execute graph from start node to completion."""
        current = "start"
        state = initial_state
        
        while current != "end":
            node = graph[current]
            step = self.step_registry.get(node.step_name)
            
            # Trace before execution
            trace_id = self.tracer.start_step(node.step_name, state)
            
            try:
                state = step.run(state, runtime)
                self.tracer.end_step(trace_id, state, success=True)
            except Exception as e:
                self.tracer.end_step(trace_id, state, success=False, error=e)
                raise
            
            # Select next node
            current = self._select_next(node, state)
        
        return state
    
    def _select_next(self, node: GraphNode, state: State) -> str:
        """Select next node based on conditions."""
        for next_name in node.next_nodes:
            next_node = self.graph[next_name]
            if next_node.condition is None or next_node.condition(state):
                return next_name
        return "end"
```

**Graph Features:**

- **Linear Chains**: Most workflows are simple sequences
- **Conditional Branching**: Nodes can have conditions for dynamic routing
- **Error Handling**: Exceptions are caught, traced, and propagated
- **Cycle Detection**: Graph validation prevents infinite loops

### Tool Registry

```python
@dataclass
class ToolMetadata:
    """Metadata for registered tools."""
    name: str
    capabilities: List[str]
    description: str
    parameters: Dict[str, Any]

class ToolRegistry:
    """Registry for external tools."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
    
    def register(self, name: str, tool: Callable, metadata: ToolMetadata):
        """Register a tool with metadata."""
        self._tools[name] = tool
        self._metadata[name] = metadata
    
    def get(self, name: str) -> Callable:
        """Retrieve tool by name."""
        return self._tools[name]
    
    def find_by_capability(self, capability: str) -> List[str]:
        """Find tools with specific capability."""
        return [name for name, meta in self._metadata.items()
                if capability in meta.capabilities]
```

**Registered Tools:**

- **web_search**: DuckDuckGo search via ddgs library
- **fetch_url**: Content extraction via trafilatura
- **calculator**: Basic arithmetic operations
- **translator**: Text translation (future)

### Model Registry

```python
@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    name: str
    capabilities: List[str]  # ["summarize", "generate", "qa", "embed"]
    expected_vram_mb: int
    supports_quant: bool
    preferred_device_policy: str  # "gpu_on_demand", "cpu_only", "gpu_resident"
    model_path: str
    quantization_config: Optional[Dict[str, Any]] = None

class ModelRegistry:
    """Registry for ML models."""
    
    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
    
    def register(self, metadata: ModelMetadata):
        """Register model with metadata."""
        self._validate_metadata(metadata)
        self._models[metadata.name] = metadata
    
    def get(self, name: str) -> ModelMetadata:
        """Retrieve model metadata by name."""
        return self._models[name]
    
    def find_by_capability(self, capability: str) -> List[ModelMetadata]:
        """Find models with specific capability."""
        return [meta for meta in self._models.values()
                if capability in meta.capabilities]
    
    def _validate_metadata(self, metadata: ModelMetadata):
        """Validate required fields are present."""
        required = ["name", "capabilities", "expected_vram_mb", 
                   "supports_quant", "preferred_device_policy"]
        for field in required:
            if not getattr(metadata, field):
                raise ValueError(f"Missing required field: {field}")
```

**Model Lifecycle:**

- **Lazy Loading**: Models load on first use, not at registration
- **Usage Tracking**: Each load/use increments counter
- **Short-term Residency**: Models stay loaded for 30s after last use
- **Quantization**: 8-bit/4-bit loading via bitsandbytes for memory efficiency

### Enhanced Model Manager

```python
@dataclass
class ModelUsageStats:
    """Usage statistics for a loaded model."""
    load_count: int = 0
    last_used: float = 0.0
    total_inference_time: float = 0.0
    peak_vram_mb: int = 0

class EnhancedModelManager:
    """Manages model lifecycle with cost tracking."""
    
    def __init__(self, registry: ModelRegistry, residency_seconds: int = 30):
        self.registry = registry
        self.residency_seconds = residency_seconds
        self._loaded: Dict[str, Any] = {}  # name -> model
        self._stats: Dict[str, ModelUsageStats] = {}
    
    def get_model(self, name: str, quantization: Optional[str] = None):
        """Load or retrieve cached model."""
        if name in self._loaded:
            self._stats[name].last_used = time.time()
            return self._loaded[name]
        
        # Load model
        metadata = self.registry.get(name)
        model = self._load_model(metadata, quantization)
        
        self._loaded[name] = model
        self._stats[name] = ModelUsageStats(
            load_count=1,
            last_used=time.time()
        )
        
        return model
    
    def cleanup_stale(self):
        """Unload models not used recently."""
        now = time.time()
        to_unload = []
        
        for name, stats in self._stats.items():
            if now - stats.last_used > self.residency_seconds:
                to_unload.append(name)
        
        for name in to_unload:
            self._unload_model(name)
    
    def _load_model(self, metadata: ModelMetadata, quantization: Optional[str]):
        """Load model with optional quantization."""
        if quantization and metadata.supports_quant:
            return self._load_quantized(metadata, quantization)
        else:
            return self._load_standard(metadata)
    
    def get_stats(self, name: str) -> ModelUsageStats:
        """Retrieve usage statistics for model."""
        return self._stats.get(name, ModelUsageStats())
```

**Quantization Support:**

- **8-bit**: Uses bitsandbytes LLM.int8() for ~50% memory reduction
- **4-bit**: Uses bitsandbytes 4-bit quantization for ~75% memory reduction
- **Configuration**: Quantization config stored in ModelMetadata
- **Fallback**: If quantization fails, falls back to standard loading

### Workflow Registry

```python
@dataclass
class WorkflowDefinition:
    """Definition of a workflow graph."""
    name: str
    description: str
    graph: Dict[str, GraphNode]
    required_capabilities: List[str]

class WorkflowRegistry:
    """Registry for workflow definitions."""
    
    def __init__(self, step_registry: StepRegistry):
        self.step_registry = step_registry
        self._workflows: Dict[str, WorkflowDefinition] = {}
    
    def register(self, definition: WorkflowDefinition):
        """Register workflow with validation."""
        self._validate_workflow(definition)
        self._workflows[definition.name] = definition
    
    def get(self, name: str) -> WorkflowDefinition:
        """Retrieve workflow definition."""
        return self._workflows[name]
    
    def list_all(self) -> List[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())
    
    def _validate_workflow(self, definition: WorkflowDefinition):
        """Validate all steps exist in registry."""
        for node in definition.graph.values():
            if not self.step_registry.has(node.step_name):
                raise ValueError(f"Unknown step: {node.step_name}")
```

**Registered Workflows:**

1. **search_qa**: Original search → fetch → summarize → answer
2. **search_qa_fast**: Search → fetch top 2 → direct answer (no summarization)
3. **search_qa_strict_citations**: Search → fetch → summarize → answer with citation validation
4. **summarize_url**: Fetch single URL → summarize
5. **rag_qa**: Embed query → retrieve docs → answer
6. **lesson_pack**: Plan → explain → exercises
7. **chat_generate**: Context-aware generation
8. **self_ask_search_qa**: Decompose → sub-QA → aggregate

### Tracer

```python
@dataclass
class StepTrace:
    """Trace record for a single step."""
    step_name: str
    start_time: float
    end_time: float
    latency_ms: float
    vram_peak_mb: int
    model_loads: int
    success: bool
    error: Optional[str] = None

@dataclass
class WorkflowTrace:
    """Complete trace for workflow execution."""
    request_id: str
    conversation_id: Optional[str]
    question: str
    chosen_workflow: str
    router_version: str
    mode: str
    turn_index: Optional[int]
    steps: List[StepTrace]
    urls_used: List[str]
    final_answer: str
    quality_signals: Dict[str, Any]
    cost_stats: Dict[str, float]
    timestamp: float

class Tracer:
    """Manages execution tracing."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self._active_traces: Dict[str, StepTrace] = {}
    
    def start_step(self, step_name: str, state: State) -> str:
        """Begin tracing a step execution."""
        trace_id = f"{step_name}_{time.time()}"
        self._active_traces[trace_id] = StepTrace(
            step_name=step_name,
            start_time=time.time(),
            end_time=0.0,
            latency_ms=0.0,
            vram_peak_mb=self._get_vram_usage(),
            model_loads=0,
            success=False
        )
        return trace_id
    
    def end_step(self, trace_id: str, state: State, 
                 success: bool, error: Optional[Exception] = None):
        """Complete step trace."""
        trace = self._active_traces[trace_id]
        trace.end_time = time.time()
        trace.latency_ms = (trace.end_time - trace.start_time) * 1000
        trace.success = success
        if error:
            trace.error = str(error)
    
    def write_workflow_trace(self, trace: WorkflowTrace):
        """Write complete workflow trace to JSONL."""
        with open(self.output_path, 'a') as f:
            f.write(json.dumps(asdict(trace)) + '\n')
```

**Trace Features:**

- **JSONL Format**: One JSON object per line for easy streaming
- **Hierarchical**: Workflow trace contains step traces
- **Cost Tracking**: Each step records latency, VRAM, model loads
- **Quality Signals**: Automatic metrics for training routers
- **Queryable**: Traces can be filtered by workflow, date, success status

### Router Evolution

#### Router v1: Rule-Based

```python
class RouterV1:
    """Rule-based router with confidence scores."""
    
    def __init__(self):
        self.rules = [
            (r"最新|今天|新闻", "search_qa", 0.9),
            (r"总结|摘要", "summarize_url", 0.8),
            (r"讲解|教学|课程", "lesson_pack", 0.85),
            (r"知识库|文档|规范", "rag_qa", 0.8),
        ]
    
    def route(self, question: str, state: State) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return (workflow, confidence, candidates)."""
        candidates = []
        
        for pattern, workflow, confidence in self.rules:
            if re.search(pattern, question):
                candidates.append((workflow, confidence))
        
        if not candidates:
            candidates.append(("search_qa_fast", 0.5))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0], candidates[0][1], candidates
```

#### Router v2: Classifier-Based

```python
class RouterV2:
    """Trained classifier for intent classification."""
    
    def __init__(self, vectorizer_path: str, clf_path: str):
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(clf_path)
        self.workflow_names = self.clf.classes_
    
    def route(self, question: str, state: State) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predict workflow using trained classifier."""
        X = self.vectorizer.transform([question])
        probs = self.clf.predict_proba(X)[0]
        
        candidates = list(zip(self.workflow_names, probs))
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0], candidates[0][1], candidates
```

**Training Process:**

1. Extract questions and chosen workflows from traces
2. Use TF-IDF vectorization on question text
3. Train LogisticRegression or DistilBERT classifier
4. Save vectorizer and model for deployment

#### Router v3: Cost-Aware

```python
class RouterV3:
    """Cost-aware router with mode features."""
    
    def __init__(self, vectorizer_path: str, clf_path: str, costs_path: str):
        self.vectorizer = joblib.load(vectorizer_path)
        self.clf = joblib.load(clf_path)
        self.workflow_costs = json.load(open(costs_path))
        self.workflow_names = self.clf.classes_
        self.lambda_cost = 0.1  # Cost weight
    
    def route(self, question: str, state: State) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Route with cost-aware scoring."""
        # Extract features
        mode = state.get("meta", {}).get("mode", "default")
        mode_is_chat = 1 if mode == "chat" else 0
        
        # Vectorize text
        X_text = self.vectorizer.transform([question])
        
        # Add mode feature
        X = np.hstack([X_text.toarray(), np.array([[mode_is_chat]])])
        
        # Get quality predictions
        quality_probs = self.clf.predict_proba(X)[0]
        
        # Calculate cost-aware scores
        scores = []
        for workflow, quality in zip(self.workflow_names, quality_probs):
            cost = self.workflow_costs.get(workflow, 1.0)
            score = quality - self.lambda_cost * cost
            scores.append((workflow, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0], scores[0][1], scores
```

**Mode Features:**

- **One-hot Encoding**: mode_chat encoded as binary feature
- **Training**: Mode extracted from trace metadata during training
- **Inference**: Mode read from State.meta at prediction time
- **Effect**: Chat mode increases preference for chat_generate and lesson_pack

## Data Models

### Lesson Section Structure

```python
@dataclass
class LessonSection:
    """Structured section of teaching content."""
    name: str  # "导入", "新授", "练习", "小结"
    teacher_say: str
    student_may_say: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    tips: Optional[str] = None

@dataclass
class StructuredLesson:
    """Complete structured lesson plan."""
    topic: str
    grade: str
    sections: List[LessonSection]
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize to JSON."""
        return {
            "topic": self.topic,
            "grade": self.grade,
            "sections": [asdict(s) for s in self.sections]
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "StructuredLesson":
        """Deserialize from JSON."""
        sections = [LessonSection(**s) for s in data["sections"]]
        return cls(
            topic=data["topic"],
            grade=data["grade"],
            sections=sections
        )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate lesson structure."""
        errors = []
        
        if len(self.sections) < 3:
            errors.append("Lesson must have at least 3 sections")
        
        has_examples = any(len(s.examples) > 0 for s in self.sections)
        has_questions = any(len(s.questions) > 0 for s in self.sections)
        
        if not (has_examples or has_questions):
            errors.append("Lesson must contain examples or questions")
        
        return len(errors) == 0, errors
```

### Cost Statistics

```python
@dataclass
class WorkflowCostStats:
    """Aggregated cost statistics for a workflow."""
    workflow_name: str
    execution_count: int
    avg_latency_ms: float
    avg_vram_mb: float
    avg_model_loads: float
    success_rate: float
    
    def update(self, trace: WorkflowTrace):
        """Update statistics with new trace."""
        n = self.execution_count
        total_latency = sum(s.latency_ms for s in trace.steps)
        max_vram = max(s.vram_peak_mb for s in trace.steps)
        total_loads = sum(s.model_loads for s in trace.steps)
        
        self.avg_latency_ms = (self.avg_latency_ms * n + total_latency) / (n + 1)
        self.avg_vram_mb = (self.avg_vram_mb * n + max_vram) / (n + 1)
        self.avg_model_loads = (self.avg_model_loads * n + total_loads) / (n + 1)
        
        success = all(s.success for s in trace.steps)
        self.success_rate = (self.success_rate * n + (1 if success else 0)) / (n + 1)
        
        self.execution_count += 1
```

### Quality Signals

```python
@dataclass
class QualitySignals:
    """Automatic quality metrics for execution."""
    citation_count: int
    answer_length: int
    has_search: bool
    has_citations: bool
    has_structure: bool  # For lessons: has sections
    failure_occurred: bool
    
    @classmethod
    def from_trace(cls, trace: WorkflowTrace, state: State) -> "QualitySignals":
        """Calculate quality signals from trace and state."""
        return cls(
            citation_count=len(state.get("citations", [])),
            answer_length=len(state.get("final_answer", "")),
            has_search=any(s.step_name == "web_search" for s in trace.steps),
            has_citations=len(state.get("citations", [])) > 0,
            has_structure="lesson_explain_structured" in state,
            failure_occurred=any(not s.success for s in trace.steps)
        )
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection Analysis

After analyzing all acceptance criteria, several redundancies were identified and consolidated:

- **Step interface properties** (1.1-1.3) can be combined into a single property about Step protocol compliance
- **State field preservation** (2.3) and **State serialization round-trip** (24.1-24.2) are related but test different aspects
- **Registry retrieval properties** (4.4, 5.3, 8.4) follow the same pattern and can share test infrastructure
- **Cost tracking properties** (7.1-7.4) can be validated together as a cost calculation invariant
- **Trace field properties** (10.2, 10.3, 23.2) can be combined into comprehensive trace completeness
- **Router probability properties** (15.3) is a mathematical invariant that applies to all classifier-based routers

### Core System Properties

**Property 1: Step Protocol Compliance**

*For any* Step implementation, it must have name, input_keys, and output_keys attributes, and its run method must accept State and Runtime parameters and return a State object.

**Validates: Requirements 1.1, 1.2, 1.3**

**Property 2: State Field Preservation**

*For any* State object and any Step execution, all fields not explicitly updated by the Step must remain unchanged in the returned State.

**Validates: Requirements 2.3**

**Property 3: Graph Linear Execution Order**

*For any* linear workflow graph (no branches), the Graph_Executor must execute steps in the exact order defined by the graph's next_nodes relationships.

**Validates: Requirements 3.2**

**Property 4: Graph Conditional Branching**

*For any* workflow graph with conditional branches, the Graph_Executor must select the next node based on the condition evaluation against the current State, and only one branch must be taken per decision point.

**Validates: Requirements 3.3**

**Property 5: Universal Step Tracing**

*For any* Step execution within a workflow, the Graph_Executor must create a trace record before execution begins and complete it after execution ends, regardless of success or failure.

**Validates: Requirements 3.4, 23.1**

### Registry Properties

**Property 6: Tool Metadata Persistence**

*For any* tool registered in Tool_Registry, retrieving the tool by name must return the same callable, and the stored metadata must contain the name and capabilities provided during registration.

**Validates: Requirements 4.3, 4.4**

**Property 7: Model Metadata Validation**

*For any* model registration attempt, if any required metadata field (name, capabilities, expected_vram_mb, supports_quant, preferred_device_policy) is missing, the Model_Registry must reject the registration with a descriptive error.

**Validates: Requirements 5.2, 5.4**

**Property 8: Capability Query Correctness**

*For any* capability string and any registry (Tool_Registry or Model_Registry), querying by that capability must return only items whose capabilities list contains that capability, and must return all such items.

**Validates: Requirements 4.4, 5.3**

**Property 9: Workflow Step Validation**

*For any* workflow registration attempt, if the workflow graph references a step name that does not exist in the Step_Registry, the Workflow_Registry must reject the registration with an error identifying the missing step.

**Validates: Requirements 8.3**

### Model Lifecycle Properties

**Property 10: Usage Counter Increment**

*For any* model load operation, the model's usage counter must increase by exactly one, regardless of whether the model was already loaded or newly loaded.

**Validates: Requirements 6.1**

**Property 11: Residency Timeout Eligibility**

*For any* loaded model, if the time since last_used exceeds the residency_seconds threshold, the model must be marked as eligible for unloading during the next cleanup cycle.

**Validates: Requirements 6.2**

**Property 12: Model Priority Ordering**

*For any* set of models competing for GPU memory, the system must order them by a priority function combining usage patterns and device_policy, and models with higher priority must be retained when memory is constrained.

**Validates: Requirements 6.4**

### Cost Tracking Properties

**Property 13: Cost Calculation Invariant**

*For any* completed Step execution, the recorded cost metrics (latency_ms, vram_peak_mb, model_loads) must all be non-negative, and the normalized cost must equal the weighted sum: w1×latency + w2×vram + w3×loads, where weights are positive constants.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

**Property 14: Cost Statistics Convergence**

*For any* workflow with multiple executions, the running average statistics (avg_latency_ms, avg_vram_mb, avg_model_loads) must converge toward the true mean as execution_count increases, following the incremental average formula.

**Validates: Requirements 12.1, 12.2, 12.3**

### Trace Properties

**Property 15: Trace Completeness**

*For any* workflow execution, the generated trace must include all required fields (request_id, question, chosen_workflow, steps, urls_used, final_answer, quality_signals, cost_stats, timestamp), and each step trace must include (step_name, latency_ms, vram_peak_mb, model_loads, success).

**Validates: Requirements 10.2, 10.3, 23.2**

**Property 16: Trace Append Behavior**

*For any* sequence of workflow executions writing to the same trace file, each execution must append its trace without overwriting previous traces, and the file must contain exactly N lines after N executions.

**Validates: Requirements 10.4**

**Property 17: Exception Capture Completeness**

*For any* Step execution that raises an exception, the trace must capture the exception type, message, and stack trace, and the step's success field must be False.

**Validates: Requirements 23.3**

### Quality Signal Properties

**Property 18: Quality Signal Calculation**

*For any* completed workflow execution, the quality signals must accurately reflect the State contents: citation_count must equal len(citations), answer_length must equal len(final_answer), has_search must be True if any step is web_search, and has_structure must be True if lesson_explain_structured exists in State.

**Validates: Requirements 11.1, 11.2, 11.4**

**Property 19: Failure Rate Tracking**

*For any* workflow execution that contains at least one failed step, the quality_signals.failure_occurred must be True, and this must contribute to the workflow's failure rate statistic.

**Validates: Requirements 11.3**

### Router Properties

**Property 20: Router Candidate Ranking**

*For any* router (v1, v2, or v3) and any question, the returned candidate list must be sorted in descending order by score/confidence, with the top candidate matching the returned workflow name.

**Validates: Requirements 14.3**

**Property 21: Probability Distribution Validity**

*For any* Router v2 or v3 prediction, the probability distribution over workflows must sum to 1.0 (within floating-point tolerance), and all individual probabilities must be in the range [0, 1].

**Validates: Requirements 15.3**

**Property 22: Cost-Aware Scoring Formula**

*For any* Router v3 evaluation, the score for each workflow must equal quality_prob - lambda_cost × normalized_cost, where quality_prob comes from the classifier and normalized_cost comes from the cost statistics.

**Validates: Requirements 16.2**

**Property 23: Mode Feature Encoding**

*For any* State with mode="chat", Router v3 must encode this as a one-hot feature with value 1 for mode_chat, and for any other mode value, the feature must be 0.

**Validates: Requirements 16.3, 21.2**

**Property 24: Mode-Specific Preference**

*For any* question evaluated by Router v3 with mode="chat", the scores for chat_generate and lesson_pack workflows must be higher than they would be with mode="default", all else being equal.

**Validates: Requirements 16.4**

### Workflow-Specific Properties

**Property 25: Summarize URL Output Fields**

*For any* successful execution of the summarize_url workflow, the final State must contain both a summary field (non-empty string) and the source URL in citations.

**Validates: Requirements 9.4**

**Property 26: Strict Citations Validation**

*For any* execution of search_qa_strict_citations workflow, if the final_answer contains key points, each key point must be followed by a citation reference in the format [N] where N is a number, and validation must fail if any key point lacks a citation.

**Validates: Requirements 9.5**

**Property 27: Lesson JSON Structure**

*For any* successful lesson_pack execution that produces JSON output, the JSON must parse successfully and contain topic (string), grade (string), and sections (array), and each section must contain at minimum name and teacher_say fields.

**Validates: Requirements 18.1, 18.2**

**Property 28: Lesson Minimum Sections**

*For any* lesson validation, if the sections array contains fewer than 3 sections, validation must return False with an error message indicating insufficient sections.

**Validates: Requirements 18.3, 19.1**

**Property 29: Lesson Content Requirement**

*For any* lesson validation, if no section contains examples and no section contains questions, validation must return False with an error message indicating missing content.

**Validates: Requirements 19.2**

**Property 30: Lesson Completeness Score**

*For any* lesson structure, the completeness score must increase monotonically as more required fields (examples, questions, key_points, student_may_say) are populated across sections.

**Validates: Requirements 19.4**

### Serialization Properties

**Property 31: State Serialization Round Trip**

*For any* valid State object, serializing to JSON and then deserializing must produce an equivalent State object with all field values preserved.

**Validates: Requirements 24.1, 24.2**

**Property 32: Nested Structure Preservation**

*For any* State containing nested structures (lists of dicts, dicts of lists), serialization and deserialization must preserve the nested structure exactly, including empty collections and None values.

**Validates: Requirements 24.3**

### Backward Compatibility Properties

**Property 33: Legacy Workflow Execution**

*For any* workflow that existed in Phase A, executing it in Phase B with the same input parameters must produce equivalent outputs and must not raise errors due to interface changes.

**Validates: Requirements 22.1, 22.2**

**Property 34: API Contract Stability**

*For any* public API method signature from Phase A, the same signature must exist in Phase B with the same parameter names, types, and return types.

**Validates: Requirements 22.3**

**Property 35: Configuration Fallback**

*For any* Phase B component that requires configuration, if the configuration is missing or invalid, the component must fall back to Phase A behavior or safe defaults without crashing.

**Validates: Requirements 22.4**

## Error Handling

### Error Categories

1. **Validation Errors**: Invalid input, missing required fields, malformed data
2. **Resource Errors**: GPU OOM, model loading failures, network timeouts
3. **Execution Errors**: Step failures, workflow exceptions, timeout exceeded
4. **Configuration Errors**: Missing config files, invalid registry entries

### Error Handling Strategy

**Step-Level Error Handling:**

- Each Step execution is wrapped in try-except by Graph_Executor
- Exceptions are caught, logged to trace, and propagated
- Step can implement custom error recovery in its run method
- Partial State updates are preserved even on failure

**Workflow-Level Error Handling:**

- Workflow execution failures are recorded in trace with full context
- Quality signals include failure_occurred flag
- Failed workflows contribute to failure rate statistics
- Router can learn to avoid workflows with high failure rates

**Resource Error Handling:**

- GPU OOM triggers automatic model unloading and retry
- Model loading failures fall back to CPU or quantized versions
- Network timeouts use exponential backoff with max retries
- VRAM monitoring prevents overcommitment

**Graceful Degradation:**

- If Router v3 fails to load, fall back to Router v2
- If Router v2 fails to load, fall back to Router v1
- If all routers fail, use default workflow (search_qa_fast)
- If trace writing fails, log to stderr but continue execution

### Error Recovery Patterns

**Retry with Backoff:**

```python
def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except RetryableError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
```

**Circuit Breaker:**

```python
class CircuitBreaker:
    """Prevent repeated calls to failing components."""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpen()
        
        try:
            result = func()
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs using Hypothesis
- Both approaches are complementary and necessary

### Unit Testing Focus

Unit tests should focus on:

- Specific workflow examples (e.g., "search for Python tutorials")
- Integration points between components (e.g., Router → Graph_Executor)
- Edge cases (empty State, missing fields, invalid JSON)
- Error conditions (network failures, GPU OOM, invalid config)

Avoid writing too many unit tests for cases that property tests can cover through randomization.

### Property-Based Testing Configuration

**Library**: Hypothesis for Python

**Configuration**:
- Minimum 100 iterations per property test
- Each test must reference its design document property in a comment
- Tag format: `# Feature: extensible-orchestration-phase-b, Property N: <property_text>`

**Example Property Test:**

```python
from hypothesis import given, strategies as st
import hypothesis

# Feature: extensible-orchestration-phase-b, Property 2: State Field Preservation
@given(
    initial_state=st.fixed_dictionaries({
        "question": st.text(),
        "meta": st.dictionaries(st.text(), st.text())
    }),
    updated_field=st.sampled_from(["question", "final_answer"]),
    new_value=st.text()
)
@hypothesis.settings(max_examples=100)
def test_state_field_preservation(initial_state, updated_field, new_value):
    """Property 2: State fields not updated by Step must be preserved."""
    step = SimpleUpdateStep(field_to_update=updated_field)
    runtime = MockRuntime()
    
    result_state = step.run(initial_state, runtime)
    
    # Check that updated field changed
    assert result_state[updated_field] == new_value
    
    # Check that other fields preserved
    for key, value in initial_state.items():
        if key != updated_field:
            assert result_state[key] == value
```

### Test Organization

**Directory Structure:**

```
tests/
├── unit/
│   ├── test_step_interface.py
│   ├── test_graph_executor.py
│   ├── test_registries.py
│   ├── test_model_manager.py
│   ├── test_tracer.py
│   ├── test_routers.py
│   └── test_workflows.py
├── property/
│   ├── test_step_properties.py
│   ├── test_state_properties.py
│   ├── test_graph_properties.py
│   ├── test_registry_properties.py
│   ├── test_cost_properties.py
│   ├── test_trace_properties.py
│   ├── test_router_properties.py
│   └── test_serialization_properties.py
├── integration/
│   ├── test_end_to_end_workflows.py
│   ├── test_router_training_pipeline.py
│   └── test_backward_compatibility.py
└── fixtures/
    ├── sample_traces.jsonl
    ├── sample_workflows.json
    └── test_configs.yaml
```

### Regression Test Harness

The regression test harness validates that changes don't break existing functionality:

**Test Dataset**: `tests/fixtures/regression_cases.jsonl`

Each test case includes:
- question: Input question
- expected_workflow: Expected router selection
- expected_fields: Required fields in output State
- quality_thresholds: Minimum quality signal values

**Execution**:

```bash
python scripts/run_regression_tests.py --dataset tests/fixtures/regression_cases.jsonl
```

**Report Output**:

- Pass/fail status for each test case
- Quality metrics comparison (current vs. baseline)
- Cost statistics comparison
- Workflow selection accuracy
- Overall regression score

### Testing Priorities

**Phase B1 (Architecture Refactoring)**:
1. Property 1: Step Protocol Compliance
2. Property 2: State Field Preservation
3. Property 3: Graph Linear Execution Order
4. Property 5: Universal Step Tracing

**Phase B2 (Registries & Model Management)**:
1. Property 7: Model Metadata Validation
2. Property 8: Capability Query Correctness
3. Property 10: Usage Counter Increment
4. Property 13: Cost Calculation Invariant

**Phase B3 (Workflow Extensions)**:
1. Property 25: Summarize URL Output Fields
2. Property 26: Strict Citations Validation
3. Property 33: Legacy Workflow Execution

**Phase B4 (Observability)**:
1. Property 15: Trace Completeness
2. Property 16: Trace Append Behavior
3. Property 18: Quality Signal Calculation

**Phase C (Trainable Router)**:
1. Property 21: Probability Distribution Validity
2. Property 22: Cost-Aware Scoring Formula
3. Property 24: Mode-Specific Preference

**Phase D (Structured Lessons)**:
1. Property 27: Lesson JSON Structure
2. Property 28: Lesson Minimum Sections
3. Property 31: State Serialization Round Trip
```

