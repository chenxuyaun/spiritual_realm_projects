"""
Unit tests for RAGQA Workflow.

Tests the RAG (Retrieval-Augmented Generation) QA workflow including:
- Query embedding
- Document retrieval
- Answer generation
- Source annotation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from mm_orch.workflows.rag_qa import RAGQAWorkflow, RAGQAContext, RAGQAStep
from mm_orch.schemas import WorkflowResult, WorkflowType, Document
from mm_orch.runtime.vector_db import VectorDBManager
from mm_orch.exceptions import ValidationError


class TestRAGQAWorkflowInit:
    """Tests for RAGQAWorkflow initialization."""
    
    def test_default_initialization(self):
        """Test workflow initializes with default values."""
        workflow = RAGQAWorkflow()
        
        assert workflow.workflow_type == WorkflowType.RAG_QA
        assert workflow.name == "RAGQA"
        assert workflow.default_top_k == 5
        assert workflow.max_context_length == 3000
    
    def test_custom_initialization(self):
        """Test workflow initializes with custom values."""
        mock_vector_db = Mock(spec=VectorDBManager)
        mock_model_manager = Mock()
        
        workflow = RAGQAWorkflow(
            vector_db=mock_vector_db,
            model_manager=mock_model_manager,
            embedding_model="custom-embedding",
            generator_model="custom-generator",
            default_top_k=10,
            max_context_length=5000
        )
        
        assert workflow.vector_db == mock_vector_db
        assert workflow.model_manager == mock_model_manager
        assert workflow.embedding_model == "custom-embedding"
        assert workflow.generator_model == "custom-generator"
        assert workflow.default_top_k == 10
        assert workflow.max_context_length == 5000


class TestRAGQAWorkflowValidation:
    """Tests for parameter validation."""
    
    def test_validate_valid_parameters(self):
        """Test validation passes for valid parameters."""
        workflow = RAGQAWorkflow()
        
        assert workflow.validate_parameters({"query": "What is Python?"})
        assert workflow.validate_parameters({
            "query": "What is Python?",
            "top_k": 3,
            "include_sources": True
        })
    
    def test_validate_empty_query_raises(self):
        """Test validation fails for empty query."""
        workflow = RAGQAWorkflow()
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": ""})
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": "   "})
    
    def test_validate_missing_query_raises(self):
        """Test validation fails for missing query."""
        workflow = RAGQAWorkflow()
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({})
    
    def test_validate_invalid_top_k_raises(self):
        """Test validation fails for invalid top_k."""
        workflow = RAGQAWorkflow()
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": "test", "top_k": 0})
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": "test", "top_k": -1})
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": "test", "top_k": "five"})
    
    def test_validate_invalid_threshold_raises(self):
        """Test validation fails for invalid threshold."""
        workflow = RAGQAWorkflow()
        
        with pytest.raises(ValidationError):
            workflow.validate_parameters({"query": "test", "threshold": -1})


class TestRAGQAWorkflowExecution:
    """Tests for workflow execution."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        return mock_db
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                content="Python is a programming language.",
                metadata={"source": "doc1.txt"},
                doc_id="doc1"
            ),
            Document(
                content="Python was created by Guido van Rossum.",
                metadata={"source": "doc2.txt"},
                doc_id="doc2"
            )
        ]
    
    def test_execute_success(self, mock_vector_db, sample_documents):
        """Test successful workflow execution."""
        # Setup mock
        mock_vector_db.search.return_value = [
            (sample_documents[0], 0.5),
            (sample_documents[1], 0.7)
        ]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation to avoid loading real models
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({"query": "What is Python?"})
        
        assert result.status in ["success", "partial"]
        assert result.metadata["workflow"] == "RAGQA"
        assert "sources" in result.metadata
        assert len(result.metadata["sources"]) == 2
    
    def test_execute_empty_database(self, mock_vector_db):
        """Test execution with empty database."""
        mock_vector_db.is_empty = True
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({"query": "What is Python?"})
        
        assert result.status == "partial"
        assert "empty" in result.error.lower() or "no" in result.error.lower()
    
    def test_execute_no_results(self, mock_vector_db):
        """Test execution when no documents match."""
        mock_vector_db.search.return_value = []
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({"query": "What is Python?"})
        
        assert result.status == "partial"
        assert result.metadata["retrieved_count"] == 0
    
    def test_execute_with_threshold(self, mock_vector_db, sample_documents):
        """Test execution with distance threshold."""
        mock_vector_db.search.return_value = [
            (sample_documents[0], 0.3),
            (sample_documents[1], 0.8)  # This should be filtered out
        ]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({
                "query": "What is Python?",
                "threshold": 0.5
            })
        
        # Only one document should pass the threshold
        assert result.metadata["retrieved_count"] == 1
    
    def test_execute_with_custom_top_k(self, mock_vector_db, sample_documents):
        """Test execution with custom top_k."""
        mock_vector_db.search.return_value = [(sample_documents[0], 0.5)]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({
                "query": "What is Python?",
                "top_k": 1
            })
        
        mock_vector_db.search.assert_called_once()
        call_args = mock_vector_db.search.call_args
        assert call_args[1]["top_k"] == 1


class TestRAGQASourceAnnotation:
    """Tests for source annotation (Property 13)."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        return mock_db
    
    def test_sources_included_in_metadata(self, mock_vector_db):
        """Test that sources are included in result metadata."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt", "page": 1},
            doc_id="test-doc"
        )
        mock_vector_db.search.return_value = [(doc, 0.5)]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({"query": "test query"})
        
        assert "sources" in result.metadata
        assert len(result.metadata["sources"]) == 1
        
        source = result.metadata["sources"][0]
        assert source["doc_id"] == "test-doc"
        assert "content_preview" in source
        assert source["metadata"]["source"] == "test.txt"
        assert "distance" in source
    
    def test_sources_excluded_when_disabled(self, mock_vector_db):
        """Test that sources can be excluded from metadata."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt"},
            doc_id="test-doc"
        )
        mock_vector_db.search.return_value = [(doc, 0.5)]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({
                "query": "test query",
                "include_sources": False
            })
        
        # Sources should be empty list when disabled
        assert result.metadata["sources"] == []
    
    def test_multiple_sources_annotated(self, mock_vector_db):
        """Test that multiple sources are properly annotated."""
        docs = [
            Document(
                content=f"Content {i}",
                metadata={"source": f"doc{i}.txt"},
                doc_id=f"doc{i}"
            )
            for i in range(3)
        ]
        mock_vector_db.search.return_value = [
            (docs[0], 0.3),
            (docs[1], 0.5),
            (docs[2], 0.7)
        ]
        
        workflow = RAGQAWorkflow(vector_db=mock_vector_db)
        
        # Mock the embedding generation
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384, dtype=np.float32)
            result = workflow.run({"query": "test query"})
        
        assert len(result.metadata["sources"]) == 3
        
        # Verify each source has required fields
        for i, source in enumerate(result.metadata["sources"]):
            assert source["doc_id"] == f"doc{i}"
            assert "content_preview" in source
            assert "metadata" in source
            assert "distance" in source


class TestRAGQAContext:
    """Tests for RAGQAContext dataclass."""
    
    def test_context_initialization(self):
        """Test context initializes correctly."""
        ctx = RAGQAContext(query="test query")
        
        assert ctx.query == "test query"
        assert ctx.query_embedding is None
        assert ctx.retrieved_docs == []
        assert ctx.answer == ""
        assert ctx.steps == []
    
    def test_add_step(self):
        """Test adding steps to context."""
        ctx = RAGQAContext(query="test")
        
        step1 = RAGQAStep(name="embed", success=True, duration=0.1)
        step2 = RAGQAStep(name="retrieve", success=True, duration=0.2)
        
        ctx.add_step(step1)
        ctx.add_step(step2)
        
        assert len(ctx.steps) == 2
        assert ctx.steps[0].name == "embed"
        assert ctx.steps[1].name == "retrieve"


class TestRAGQAStep:
    """Tests for RAGQAStep dataclass."""
    
    def test_step_initialization(self):
        """Test step initializes correctly."""
        step = RAGQAStep(name="test", success=True)
        
        assert step.name == "test"
        assert step.success is True
        assert step.duration == 0.0
        assert step.input_count == 0
        assert step.output_count == 0
        assert step.error is None
    
    def test_step_with_error(self):
        """Test step with error."""
        step = RAGQAStep(
            name="test",
            success=False,
            error="Something went wrong"
        )
        
        assert step.success is False
        assert step.error == "Something went wrong"


class TestRAGQAWorkflowHelpers:
    """Tests for helper methods."""
    
    def test_get_required_parameters(self):
        """Test required parameters list."""
        workflow = RAGQAWorkflow()
        
        required = workflow.get_required_parameters()
        
        assert "query" in required
    
    def test_get_optional_parameters(self):
        """Test optional parameters with defaults."""
        workflow = RAGQAWorkflow(default_top_k=10)
        
        optional = workflow.get_optional_parameters()
        
        assert "top_k" in optional
        assert optional["top_k"] == 10
        assert "include_sources" in optional
        assert optional["include_sources"] is True
        assert "threshold" in optional
    
    def test_get_required_models(self):
        """Test required models list."""
        workflow = RAGQAWorkflow(
            embedding_model="test-embed",
            generator_model="test-gen"
        )
        
        models = workflow.get_required_models()
        
        assert "test-embed" in models
        assert "test-gen" in models


class TestRAGQAWorkflowIntegration:
    """Integration-style tests with real VectorDBManager."""
    
    @pytest.fixture
    def vector_db_with_docs(self):
        """Create a vector database with sample documents."""
        db = VectorDBManager(dimension=4)  # Small dimension for testing
        db.create_index()
        
        # Add sample documents with embeddings
        docs = [
            Document(
                content="Python is a high-level programming language.",
                metadata={"source": "python_intro.txt"},
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            ),
            Document(
                content="Machine learning uses algorithms to learn from data.",
                metadata={"source": "ml_basics.txt"},
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            ),
            Document(
                content="Deep learning is a subset of machine learning.",
                metadata={"source": "dl_intro.txt"},
                embedding=np.array([0.0, 0.8, 0.2, 0.0], dtype=np.float32)
            )
        ]
        
        db.add_documents(docs)
        return db
    
    def test_full_workflow_with_real_db(self, vector_db_with_docs):
        """Test full workflow with real vector database."""
        workflow = RAGQAWorkflow(vector_db=vector_db_with_docs)
        
        # Mock the embedding generation to return a known vector
        with patch.object(workflow, '_get_query_embedding') as mock_embed:
            # Query vector similar to Python document
            mock_embed.return_value = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
            
            result = workflow.run({"query": "What is Python?"})
        
        assert result.status in ["success", "partial"]
        assert result.metadata["retrieved_count"] > 0
        assert "sources" in result.metadata
        
        # The Python document should be most relevant
        if result.metadata["sources"]:
            top_source = result.metadata["sources"][0]
            assert "python" in top_source["metadata"]["source"].lower()
