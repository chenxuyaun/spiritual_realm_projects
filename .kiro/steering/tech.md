# Technology Stack

## Core Framework

- **Language**: Python 3.8+
- **ML Framework**: PyTorch
- **Model Library**: HuggingFace Transformers
- **Acceleration**: accelerate, bitsandbytes (for 8bit/4bit quantization)

## Models

- **Chat/Generation**: Qwen Chat, GPT2/DistilGPT2
- **Summarization**: T5-small, BART
- **Embeddings**: MiniLM (sentence-transformers)
- **Multimodal**: LLaVA (via Ollama, optional)

## External Tools & Libraries

- **Web Search**: ddgs (DuckDuckGo Search)
- **Web Scraping**: trafilatura (content extraction)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **HTTP Client**: httpx, requests
- **Testing**: pytest, hypothesis (property-based testing)

## Infrastructure

- **Configuration**: YAML/JSON config files
- **Logging**: Structured logging (JSON format)
- **Persistence**: File system + JSON for state, FAISS indices for vectors
- **API**: RESTful API with JSON request/response

## Hardware Requirements

- **Recommended**: NVIDIA T4 (15GB) or A100 GPU
- **Fallback**: CPU execution with model quantization
- **Memory Management**: Lazy loading, LRU cache (max 3 models), automatic GPU/CPU switching

## Project Structure

```
mm_orch/
├── main.py                    # CLI entry point
├── router.py                  # Workflow routing
├── runner.py                  # Workflow orchestration
├── schemas.py                 # Data models
├── consciousness/             # Consciousness modules
│   ├── core.py
│   ├── self_model.py
│   ├── world_model.py
│   ├── metacognition.py
│   ├── motivation.py
│   ├── emotion.py
│   └── development.py
├── workflows/                 # Workflow implementations
│   ├── base.py
│   ├── search_qa.py
│   ├── lesson_pack.py
│   ├── chat_generate.py
│   ├── rag_qa.py
│   └── self_ask_search_qa.py
├── runtime/
│   └── model_manager.py       # Model loading/caching
├── tools/
│   ├── web_search.py
│   └── fetch_url.py
└── models/
    ├── summarizer.py
    └── generator.py
```

## Common Commands

### Installation
```bash
pip install torch transformers accelerate bitsandbytes
pip install ddgs trafilatura faiss-cpu sentence-transformers
pip install httpx pytest hypothesis
```

### Running
```bash
# CLI mode
python -m mm_orch.main "your question here"

# With specific workflow
python -m mm_orch.main --workflow search_qa "question"

# Chat mode
python -m mm_orch.main --mode chat
```

### Testing
```bash
# Run all tests
pytest tests/

# Run property-based tests only
pytest tests/property/

# Run with coverage
pytest --cov=mm_orch tests/
```

### Development
```bash
# Check consciousness status
python -c "from mm_orch.consciousness import get_consciousness; print(get_consciousness().get_status_summary())"

# Save consciousness state
python -c "from mm_orch.consciousness import save_consciousness; save_consciousness(force=True)"
```

## Configuration Files

- **System Config**: `config/system.yaml` - model paths, device settings, cache limits
- **Router Config**: `config/router.yaml` - routing rules and thresholds
- **Consciousness State**: `.consciousness/state.json` - persistent consciousness state
- **Vector DB**: `data/vector_db/` - FAISS indices and metadata
