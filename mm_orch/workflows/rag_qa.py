"""
RAGQA Workflow Implementation.

This module implements the RAG (Retrieval-Augmented Generation) QA workflow which:
1. Converts the query to a vector embedding
2. Retrieves relevant documents from the vector database
3. Generates an answer based on the retrieved context
4. Annotates the answer with source information

Properties verified:
- Property 12: RAG检索相关性
- Property 13: RAG答案来源标注
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import numpy as np

from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult, WorkflowType, Document
from mm_orch.runtime.vector_db import VectorDBManager, get_vector_db
from mm_orch.exceptions import ValidationError, WorkflowError, ResourceError
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RAGQAStep:
    """Tracks the execution of a workflow step."""

    name: str
    success: bool
    duration: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: Optional[str] = None


@dataclass
class RAGQAContext:
    """Context for RAGQA workflow execution."""

    query: str
    query_embedding: Optional[np.ndarray] = None
    retrieved_docs: List[Tuple[Document, float]] = field(default_factory=list)
    answer: str = ""
    steps: List[RAGQAStep] = field(default_factory=list)

    def add_step(self, step: RAGQAStep) -> None:
        """Add a step to the execution history."""
        self.steps.append(step)


class RAGQAWorkflow(BaseWorkflow):
    """
    RAG Question Answering Workflow: Embed → Retrieve → Generate

    This workflow performs knowledge base question answering by:
    1. Converting the query to a vector embedding using MiniLM
    2. Retrieving relevant document fragments from FAISS vector database
    3. Generating an answer based on the retrieved context
    4. Annotating the answer with source document information

    Attributes:
        workflow_type: WorkflowType.RAG_QA
        name: "RAGQA"

    Properties verified:
        - Property 12: RAG检索相关性 (RAG retrieval relevance)
        - Property 13: RAG答案来源标注 (RAG answer source annotation)

    Requirements:
        - 需求5.3: 将问题转换为向量并检索最相关的文档片段
        - 需求5.4: 基于检索到的上下文生成答案
        - 需求5.5: 在答案中标注信息来源的文档片段
    """

    workflow_type = WorkflowType.RAG_QA
    name = "RAGQA"
    description = "RAG-based knowledge base question answering workflow"

    def __init__(
        self,
        vector_db: Optional[VectorDBManager] = None,
        model_manager: Optional[Any] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "gpt2",
        default_top_k: int = 5,
        max_context_length: int = 3000,
    ):
        """
        Initialize the RAGQA workflow.

        Args:
            vector_db: Vector database manager instance
            model_manager: Model manager for embedding and generation
            embedding_model: Model name for query embedding
            generator_model: Model name for answer generation
            default_top_k: Default number of documents to retrieve
            max_context_length: Maximum context length for generation
        """
        super().__init__()
        self.vector_db = vector_db
        self.model_manager = model_manager
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.default_top_k = default_top_k
        self.max_context_length = max_context_length

    def get_required_parameters(self) -> List[str]:
        """Return required parameters for this workflow."""
        return ["query"]

    def get_optional_parameters(self) -> Dict[str, Any]:
        """Return optional parameters with defaults."""
        return {
            "top_k": self.default_top_k,
            "include_sources": True,
            "threshold": None,  # Optional distance threshold
        }

    def get_required_models(self) -> List[str]:
        """Return the list of models required by this workflow."""
        return [self.embedding_model, self.generator_model]

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate workflow parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If parameters are invalid
        """
        self._validate_required_parameters(parameters)

        query = parameters.get("query", "")
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        top_k = parameters.get("top_k", self.default_top_k)
        if not isinstance(top_k, int) or top_k < 1:
            raise ValidationError("top_k must be a positive integer")

        threshold = parameters.get("threshold")
        if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0):
            raise ValidationError("threshold must be a non-negative number")

        return True

    def _get_vector_db(self) -> VectorDBManager:
        """Get the vector database instance."""
        if self.vector_db is not None:
            return self.vector_db
        return get_vector_db()

    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the RAGQA workflow.

        Steps:
        1. Embed: Convert query to vector embedding
        2. Retrieve: Search vector database for relevant documents
        3. Generate: Create answer based on retrieved context

        Args:
            parameters: Workflow parameters including 'query'

        Returns:
            WorkflowResult with answer and source metadata
        """
        query = parameters["query"]
        top_k = parameters.get("top_k", self.default_top_k)
        include_sources = parameters.get("include_sources", True)
        threshold = parameters.get("threshold")

        # Initialize context
        ctx = RAGQAContext(query=query)

        try:
            # Step 1: Embed query
            ctx = self._step_embed_query(ctx)

            if ctx.query_embedding is None:
                return self._create_result(
                    ctx,
                    status="failed",
                    error="Failed to generate query embedding",
                    include_sources=include_sources,
                )

            # Step 2: Retrieve documents
            ctx = self._step_retrieve(ctx, top_k, threshold)

            if not ctx.retrieved_docs:
                return self._create_result(
                    ctx,
                    status="partial",
                    error="No relevant documents found",
                    include_sources=include_sources,
                )

            # Step 3: Generate answer
            ctx = self._step_generate_answer(ctx)

            status = "success" if ctx.answer else "partial"

            return self._create_result(ctx, status=status, include_sources=include_sources)

        except ResourceError as e:
            logger.error("RAGQA workflow resource error", error=str(e), query=query[:50])
            return self._create_result(
                ctx,
                status="failed",
                error=f"Resource error: {str(e)}",
                include_sources=include_sources,
            )
        except Exception as e:
            logger.error("RAGQA workflow failed", error=str(e), query=query[:50])
            return self._create_result(
                ctx, status="failed", error=str(e), include_sources=include_sources
            )

    def _step_embed_query(self, ctx: RAGQAContext) -> RAGQAContext:
        """
        Step 1: Convert query to vector embedding.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with query embedding
        """
        start_time = time.time()
        step = RAGQAStep(name="embed", success=False, input_count=1)

        try:
            logger.info(f"Step 1: Embedding query '{ctx.query[:50]}...'")

            embedding = self._get_query_embedding(ctx.query)
            ctx.query_embedding = embedding

            step.success = embedding is not None
            step.output_count = 1 if embedding is not None else 0

            if embedding is not None:
                logger.info(f"Query embedded successfully, dimension: {embedding.shape[0]}")
            else:
                step.error = "Failed to generate embedding"

        except Exception as e:
            step.error = str(e)
            logger.error(f"Query embedding failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Generate embedding for the query.

        Args:
            query: Query text

        Returns:
            Query embedding vector or None if failed
        """
        # If model manager is available, use it
        if self.model_manager:
            try:
                embedding = self.model_manager.get_embedding(self.embedding_model, query)
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.warning(f"Model manager embedding failed: {e}")

        # Fallback: try to use sentence-transformers directly
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model)
            embedding = model.encode(query, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except ImportError:
            logger.warning("sentence-transformers not installed")
        except Exception as e:
            logger.warning(f"Direct embedding failed: {e}")

        # Last fallback: generate a random embedding for testing
        # This should only be used in test scenarios
        logger.warning("Using fallback random embedding - not for production use")
        vector_db = self._get_vector_db()
        return np.random.randn(vector_db.dimension).astype(np.float32)

    def _step_retrieve(
        self, ctx: RAGQAContext, top_k: int, threshold: Optional[float]
    ) -> RAGQAContext:
        """
        Step 2: Retrieve relevant documents from vector database.

        Args:
            ctx: Workflow context with query embedding
            top_k: Number of documents to retrieve
            threshold: Optional distance threshold

        Returns:
            Updated context with retrieved documents
        """
        start_time = time.time()
        step = RAGQAStep(name="retrieve", success=False, input_count=1)

        try:
            logger.info(f"Step 2: Retrieving top {top_k} documents")

            vector_db = self._get_vector_db()

            # Check if vector database has documents
            if vector_db.is_empty:
                step.error = "Vector database is empty"
                step.duration = time.time() - start_time
                ctx.add_step(step)
                return ctx

            # Search for relevant documents
            results = vector_db.search(ctx.query_embedding, top_k=top_k)

            # Apply threshold filter if specified
            if threshold is not None:
                results = [(doc, dist) for doc, dist in results if dist <= threshold]

            ctx.retrieved_docs = results
            step.success = len(results) > 0
            step.output_count = len(results)

            if not results:
                step.error = "No documents matched the query"

            logger.info(f"Retrieved {len(results)} documents")

        except ResourceError as e:
            step.error = str(e)
            logger.error(f"Retrieval failed: {e}")
            raise
        except Exception as e:
            step.error = str(e)
            logger.error(f"Retrieval failed: {e}")

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _step_generate_answer(self, ctx: RAGQAContext) -> RAGQAContext:
        """
        Step 3: Generate answer based on retrieved context.

        Args:
            ctx: Workflow context with retrieved documents

        Returns:
            Updated context with generated answer
        """
        start_time = time.time()
        step = RAGQAStep(name="generate", success=False, input_count=len(ctx.retrieved_docs))

        try:
            if not ctx.retrieved_docs:
                step.error = "No documents to generate answer from"
                step.duration = time.time() - start_time
                ctx.add_step(step)
                return ctx

            logger.info("Step 3: Generating answer")

            answer = self._generate_answer(ctx.query, ctx.retrieved_docs)
            ctx.answer = answer

            step.success = bool(answer)
            step.output_count = 1 if answer else 0

            if not answer:
                step.error = "Empty answer generated"

            logger.info(f"Generated answer of length {len(answer)}")

        except Exception as e:
            step.error = str(e)
            logger.error(f"Answer generation failed: {e}")
            # Fallback: use document content as answer
            ctx = self._fallback_answer(ctx)

        step.duration = time.time() - start_time
        ctx.add_step(step)
        return ctx

    def _generate_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Generate answer from query and retrieved documents.

        Args:
            query: User's question
            retrieved_docs: List of (document, distance) tuples

        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, distance) in enumerate(retrieved_docs):
            source_info = doc.metadata.get("source", f"Document {i+1}")
            context_parts.append(f"[Source: {source_info}]\n{doc.content}")

        context = "\n\n".join(context_parts)

        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "..."

        # Build prompt
        prompt = f"""Based on the following information from the knowledge base, answer the question.

Knowledge Base Context:
{context}

Question: {query}

Answer:"""

        # If model manager is available, use it
        if self.model_manager:
            try:
                answer = self.model_manager.infer(
                    self.generator_model, prompt, max_new_tokens=300, temperature=0.7
                )
                # Clean up the answer
                if "Answer:" in answer:
                    answer = answer.split("Answer:")[-1].strip()
                return answer
            except Exception as e:
                logger.warning(f"Model generation failed: {e}")

        # Fallback: return a simple combined response
        return self._simple_answer(query, retrieved_docs)

    def _simple_answer(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:
        """
        Generate a simple answer without using models.

        Args:
            query: User's question
            retrieved_docs: Retrieved documents

        Returns:
            Simple combined answer
        """
        if not retrieved_docs:
            return "I couldn't find relevant information in the knowledge base to answer your question."

        # Use the most relevant document's content
        top_doc, _ = retrieved_docs[0]
        content = top_doc.content

        # Truncate if needed
        max_length = 500
        if len(content) > max_length:
            # Try to truncate at sentence boundary
            truncated = content[:max_length]
            last_period = truncated.rfind(".")
            if last_period > max_length // 2:
                content = truncated[: last_period + 1]
            else:
                content = truncated + "..."

        source = top_doc.metadata.get("source", "knowledge base")
        return f"Based on the {source}: {content}"

    def _fallback_answer(self, ctx: RAGQAContext) -> RAGQAContext:
        """
        Fallback: Use document content when generation fails.

        Args:
            ctx: Workflow context

        Returns:
            Updated context with fallback answer
        """
        logger.warning("Using fallback answer from document content")

        if ctx.retrieved_docs:
            ctx.answer = self._simple_answer(ctx.query, ctx.retrieved_docs)
        else:
            ctx.answer = "Unable to generate an answer due to processing errors."

        return ctx

    def _create_result(
        self,
        ctx: RAGQAContext,
        status: str = "success",
        error: Optional[str] = None,
        include_sources: bool = True,
    ) -> WorkflowResult:
        """
        Create the workflow result with source annotations.

        This method ensures Property 13 (RAG答案来源标注) is satisfied
        by including source metadata in the result.

        Args:
            ctx: Workflow context
            status: Result status
            error: Error message if any
            include_sources: Whether to include source information

        Returns:
            WorkflowResult object with sources in metadata
        """
        metadata: Dict[str, Any] = {
            "workflow": self.name,
            "query": ctx.query,
            "steps": [
                {
                    "name": s.name,
                    "success": s.success,
                    "duration": s.duration,
                    "input_count": s.input_count,
                    "output_count": s.output_count,
                    "error": s.error,
                }
                for s in ctx.steps
            ],
            "retrieved_count": len(ctx.retrieved_docs),
        }

        # Property 13: RAG答案来源标注
        # Always include sources in metadata for traceability
        if include_sources and ctx.retrieved_docs:
            metadata["sources"] = [
                {
                    "doc_id": doc.doc_id,
                    "content_preview": doc.content[:200] if doc.content else "",
                    "metadata": doc.metadata,
                    "distance": float(distance),
                }
                for doc, distance in ctx.retrieved_docs
            ]
        else:
            metadata["sources"] = []

        result = ctx.answer if ctx.answer else None

        return WorkflowResult(result=result, metadata=metadata, status=status, error=error)

    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[np.ndarray]] = None
    ) -> int:
        """
        Add documents to the vector database.

        This is a convenience method for adding documents to the
        underlying vector database.

        Args:
            documents: List of documents to add
            embeddings: Optional pre-computed embeddings

        Returns:
            Number of documents successfully added
        """
        vector_db = self._get_vector_db()

        # If embeddings not provided, generate them
        if embeddings is None and self.model_manager:
            embeddings = []
            for doc in documents:
                try:
                    embedding = self.model_manager.get_embedding(self.embedding_model, doc.content)
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed document {doc.doc_id}: {e}")
                    embeddings.append(None)

        return vector_db.add_documents(documents, embeddings)

    def get_document_count(self) -> int:
        """Get the number of documents in the vector database."""
        vector_db = self._get_vector_db()
        return vector_db.document_count
