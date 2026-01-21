# MuAI多模型编排系统

MuAI Multi-Model Orchestration System - A general-purpose AI system with consciousness modules, multi-workflow orchestration, and teaching capabilities.

## Features

- **Multi-Workflow Orchestration**: Automatic routing to appropriate workflows (search_qa, lesson_pack, chat_generate, rag_qa, self_ask_search_qa)
- **Consciousness Modules**: Self-awareness, emotion processing, metacognition, motivation, and developmental stages
- **Teaching Assistant**: Generates structured lesson plans, explanations, and exercises
- **Search-Enhanced Q&A**: Web search → content extraction → summarization → answer generation
- **RAG Knowledge Base**: Vector-based document retrieval with FAISS
- **Multi-turn Conversations**: Context-aware dialogue with persistent history

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended: T4 or A100) or CPU

### Setup

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
mm_orch/
├── consciousness/     # Consciousness modules
├── workflows/         # Workflow implementations
├── runtime/           # Model management
├── tools/             # External tool integrations
├── models/            # Model wrappers
├── router.py          # Workflow routing
├── runner.py          # Workflow orchestration
└── schemas.py         # Data models

tests/
├── unit/              # Unit tests
├── property/          # Property-based tests
├── integration/       # Integration tests
└── fixtures/          # Test fixtures

config/                # Configuration files
data/                  # Runtime data
scripts/               # Utility scripts
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mm_orch tests/

# Run only property-based tests
pytest tests/property/

# Run specific test file
pytest tests/unit/test_router.py
```

### Code Quality

```bash
# Format code
black mm_orch/ tests/

# Lint code
flake8 mm_orch/ tests/

# Type checking
mypy mm_orch/
```

## Configuration

Configuration files are located in the `config/` directory:
- `system.yaml`: Model paths, device settings, cache limits
- `router.yaml`: Routing rules and thresholds

## License

TBD

## Contributing

TBD
