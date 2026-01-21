# Project Structure & Organization

## Directory Layout

```
.
├── .consciousness/            # Persistent consciousness state
│   └── state.json
├── .kiro/                     # Kiro IDE configuration
│   ├── specs/                 # Feature specifications
│   │   └── muai-orchestration-system/
│   │       ├── requirements.md
│   │       ├── design.md
│   │       └── tasks.md
│   └── steering/              # AI assistant guidance (this directory)
├── config/                    # Configuration files
│   ├── system.yaml
│   └── router.yaml
├── data/                      # Runtime data
│   ├── vector_db/             # FAISS indices
│   ├── chat_history/          # Conversation logs
│   └── traces/                # Execution traces
├── docs/                      # Documentation
│   ├── consciousness_api.md
│   ├── consciousness_guide.md
│   └── 需求文档*.md           # Requirement iterations (Chinese)
├── mm_orch/                   # Main source code
│   ├── __init__.py
│   ├── main.py                # CLI entry point
│   ├── router.py              # Workflow router
│   ├── runner.py              # Workflow orchestrator
│   ├── schemas.py             # Data models & types
│   ├── consciousness/         # Consciousness modules
│   │   ├── __init__.py
│   │   ├── core.py            # ConsciousnessCore
│   │   ├── self_model.py      # SelfModel
│   │   ├── world_model.py     # WorldModel
│   │   ├── metacognition.py   # Metacognition
│   │   ├── motivation.py      # MotivationSystem
│   │   ├── emotion.py         # EmotionSystem
│   │   ├── development.py     # DevelopmentSystem
│   │   ├── multimodal.py      # MultiModalPerception
│   │   ├── tools.py           # ToolRegistry
│   │   ├── multilingual.py    # MultilingualSystem
│   │   ├── performance.py     # PerformanceMonitor
│   │   ├── advanced.py        # AdvancedCognition
│   │   ├── user_experience.py # UserExperience
│   │   ├── multi_agent.py     # MultiAgentSystem
│   │   ├── knowledge.py       # KnowledgeSystem
│   │   ├── streaming.py       # StreamingSystem
│   │   ├── learning.py        # ContinuousLearning
│   │   ├── safety.py          # SafetyAlignment
│   │   ├── reflection.py      # DeepReflection
│   │   ├── context.py         # ContextManager
│   │   └── personality.py     # PersonalitySystem
│   ├── workflows/             # Workflow implementations
│   │   ├── __init__.py
│   │   ├── base.py            # BaseWorkflow abstract class
│   │   ├── search_qa.py       # SearchQAWorkflow
│   │   ├── lesson_pack.py     # LessonPackWorkflow
│   │   ├── chat_generate.py   # ChatGenerateWorkflow
│   │   ├── rag_qa.py          # RAGQAWorkflow
│   │   └── self_ask_search_qa.py  # SelfAskSearchQAWorkflow
│   ├── runtime/               # Runtime management
│   │   ├── __init__.py
│   │   └── model_manager.py   # ModelManager (loading/caching)
│   ├── tools/                 # External tool integrations
│   │   ├── __init__.py
│   │   ├── web_search.py      # ddgs search wrapper
│   │   └── fetch_url.py       # trafilatura scraper
│   └── models/                # Model wrappers
│       ├── __init__.py
│       ├── summarizer.py      # T5/BART summarization
│       └── generator.py       # Qwen/GPT2 generation
├── scripts/                   # Utility scripts
│   ├── train_router_v3.py     # Router training
│   └── eval_workflows.py      # Workflow evaluation
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── property/              # Property-based tests (Hypothesis)
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test fixtures
└── requirements.txt           # Python dependencies
```

## Module Organization Principles

### Layered Architecture

1. **Application Layer** (`main.py`, API endpoints): User-facing interfaces
2. **Orchestration Layer** (`router.py`, `runner.py`): Workflow coordination
3. **Consciousness Layer** (`consciousness/`): Self-awareness and metacognition
4. **Workflow Layer** (`workflows/`): Task-specific processing pipelines
5. **Model Layer** (`models/`, `runtime/`): ML model management
6. **Infrastructure Layer** (`tools/`, config, logging): Supporting services

### Naming Conventions

- **Files**: Snake_case (e.g., `model_manager.py`)
- **Classes**: PascalCase (e.g., `ConsciousnessCore`, `SearchQAWorkflow`)
- **Functions/Methods**: Snake_case (e.g., `get_consciousness()`, `execute_workflow()`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_CACHED_MODELS`)
- **Private members**: Leading underscore (e.g., `_internal_method()`)

### Import Conventions

```python
# Standard library
import os
from typing import Dict, List, Any

# Third-party
import torch
from transformers import pipeline

# Local absolute imports
from mm_orch.schemas import WorkflowType, UserRequest
from mm_orch.consciousness import get_consciousness
from mm_orch.workflows.base import BaseWorkflow
```

## Key Architectural Patterns

### Singleton Pattern
Consciousness modules use singleton pattern via factory functions:
```python
from mm_orch.consciousness import get_consciousness  # Returns singleton
```

### Abstract Base Classes
All workflows inherit from `BaseWorkflow` with required methods:
- `execute(parameters) -> WorkflowResult`
- `validate_parameters(parameters) -> bool`
- `get_required_models() -> List[str]`

### Dependency Injection
Components receive dependencies through constructors, not global state.

### Error Handling
- Custom exception hierarchy under `mm_orch.exceptions`
- Structured error responses with `ErrorResponse` dataclass
- Graceful degradation with fallback strategies

## Data Flow

1. User input → Router (intent classification)
2. Router → Orchestrator (workflow selection)
3. Orchestrator → ConsciousnessCore (state update, strategy suggestion)
4. Orchestrator → Workflow (execution)
5. Workflow → ModelManager (model inference)
6. Workflow → Orchestrator (results)
7. Orchestrator → ConsciousnessCore (finalize, update emotion/motivation)
8. Orchestrator → User (response)

## Configuration Management

- **System-wide**: `config/system.yaml` (model paths, device, cache)
- **Component-specific**: `config/router.yaml`, `config/consciousness.yaml`
- **Runtime state**: `.consciousness/state.json` (auto-saved)
- **Environment variables**: For sensitive data (API keys, paths)

## Testing Organization

- **Unit tests** (`tests/unit/`): Test individual components in isolation
- **Property tests** (`tests/property/`): Verify universal properties with Hypothesis
- **Integration tests** (`tests/integration/`): Test component interactions
- **Fixtures** (`tests/fixtures/`): Shared test data and mocks

Each test file should mirror the source structure:
- `mm_orch/router.py` → `tests/unit/test_router.py`
- Property tests reference design doc properties in docstrings
