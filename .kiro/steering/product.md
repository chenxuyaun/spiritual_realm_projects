# Product Overview

MuAI多模型编排系统 (MuAI Multi-Model Orchestration System) is a general-purpose AI system that integrates multiple pre-trained models to provide intelligent capabilities including search-based Q&A, teaching content generation, conversational AI, and knowledge base question answering.

## Core Capabilities

- **Multi-Workflow Orchestration**: Automatically routes user requests to appropriate workflows (search_qa, lesson_pack, chat_generate, rag_qa, self_ask_search_qa)
- **Consciousness Modules**: Self-awareness, emotion processing, metacognition, motivation, and developmental stages
- **Teaching Assistant**: Generates structured lesson plans, explanations, and exercises for educational content
- **Search-Enhanced Q&A**: Web search → content extraction → summarization → answer generation pipeline
- **RAG Knowledge Base**: Vector-based document retrieval with FAISS for domain-specific Q&A
- **Multi-turn Conversations**: Context-aware dialogue with persistent conversation history

## Target Users

- Educators requiring automated teaching content generation
- Users seeking AI-powered question answering with web search capabilities
- Developers building AI systems with consciousness-like capabilities
- Organizations needing intelligent document Q&A systems

## Key Design Principles

1. **Modular Architecture**: Decoupled components with clear interfaces
2. **Consciousness-Driven**: Integrated self-model, world model, and metacognition throughout processing
3. **Resource Optimization**: Lazy model loading, LRU caching, GPU/CPU hybrid deployment
4. **Progressive Development**: Staged capability unlocking from infant to adult stages
5. **Graceful Degradation**: Fallback strategies for errors and resource constraints
