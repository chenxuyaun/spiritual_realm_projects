"""
Unit tests for VectorDBManager.

Tests specific examples and edge cases for the vector database functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from mm_orch.runtime.vector_db import (
    VectorDBManager,
    IndexMetadata,
    get_vector_db,
    configure_vector_db,
)
from mm_orch.schemas import Document
from mm_orch.exceptions import ResourceError, StorageError, ValidationError


# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for storage tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def vector_db():
    """Create a VectorDBManager instance for testing."""
    return VectorDBManager(dimension=64, index_type="flat")


def generate_embedding(dimension: int = 64) -> np.ndarray:
    """Generate a random embedding vector."""
    return np.random.randn(dimension).astype(np.float32)


class TestIndexMetadata:
    """Tests for IndexMetadata dataclass."""
    
    def test_create_metadata(self):
        """Test creating index metadata."""
        metadata = IndexMetadata(dimension=128, index_type="flat")
        
        assert metadata.dimension == 128
        assert metadata.index_type == "flat"
        assert metadata.document_count == 0
        assert metadata.created_at > 0
        assert metadata.updated_at > 0
    
    def test_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = IndexMetadata(dimension=64, index_type="ivf", document_count=10)
        
        data = metadata.to_dict()
        
        assert data["dimension"] == 64
        assert data["index_type"] == "ivf"
        assert data["document_count"] == 10
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "dimension": 256,
            "index_type": "hnsw",
            "created_at": 1000.0,
            "updated_at": 2000.0,
            "document_count": 50
        }
        
        metadata = IndexMetadata.from_dict(data)
        
        assert metadata.dimension == 256
        assert metadata.index_type == "hnsw"
        assert metadata.created_at == 1000.0
        assert metadata.updated_at == 2000.0
        assert metadata.document_count == 50


class TestVectorDBManagerInit:
    """Tests for VectorDBManager initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        manager = VectorDBManager()
        
        assert manager.dimension == 384
        assert manager.index_type == "flat"
        assert manager.storage_path is None
        assert manager._index is None
        assert manager.document_count == 0
    
    def test_custom_initialization(self, temp_storage_path):
        """Test initialization with custom parameters."""
        manager = VectorDBManager(
            dimension=128,
            index_type="ivf",
            storage_path=temp_storage_path
        )
        
        assert manager.dimension == 128
        assert manager.index_type == "ivf"
        assert manager.storage_path == Path(temp_storage_path)
    
    def test_repr(self, vector_db):
        """Test string representation."""
        repr_str = repr(vector_db)
        
        assert "VectorDBManager" in repr_str
        assert "dimension=64" in repr_str
        assert "index_type=flat" in repr_str


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerCreateIndex:
    """Tests for index creation."""
    
    def test_create_flat_index(self):
        """Test creating a flat index."""
        manager = VectorDBManager(dimension=64, index_type="flat")
        manager.create_index()
        
        assert manager._index is not None
        assert manager._metadata is not None
        assert manager._metadata.dimension == 64
        assert manager._metadata.index_type == "flat"
    
    def test_create_ivf_index(self):
        """Test creating an IVF index."""
        manager = VectorDBManager(dimension=64, index_type="ivf")
        manager.create_index()
        
        assert manager._index is not None
        assert manager.index_type == "ivf"
    
    def test_create_hnsw_index(self):
        """Test creating an HNSW index."""
        manager = VectorDBManager(dimension=64, index_type="hnsw")
        manager.create_index()
        
        assert manager._index is not None
        assert manager.index_type == "hnsw"
    
    def test_create_index_with_override(self):
        """Test creating index with dimension/type override."""
        manager = VectorDBManager(dimension=64, index_type="flat")
        manager.create_index(dimension=128, index_type="hnsw")
        
        assert manager.dimension == 128
        assert manager.index_type == "hnsw"
    
    def test_invalid_index_type_raises_error(self):
        """Test that invalid index type raises ValidationError."""
        manager = VectorDBManager()
        
        with pytest.raises(ValidationError):
            manager.create_index(index_type="invalid_type")


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerAddDocuments:
    """Tests for adding documents."""
    
    def test_add_single_document(self, vector_db):
        """Test adding a single document."""
        vector_db.create_index()
        
        doc = Document(
            content="Test document",
            metadata={"source": "test"},
            embedding=generate_embedding(64)
        )
        
        success = vector_db.add_document(doc)
        
        assert success
        assert vector_db.document_count == 1
        assert vector_db.index_size == 1
    
    def test_add_multiple_documents(self, vector_db):
        """Test adding multiple documents."""
        vector_db.create_index()
        
        documents = [
            Document(
                content=f"Document {i}",
                metadata={"index": i},
                embedding=generate_embedding(64)
            )
            for i in range(5)
        ]
        
        count = vector_db.add_documents(documents)
        
        assert count == 5
        assert vector_db.document_count == 5
        assert vector_db.index_size == 5
    
    def test_add_documents_with_external_embeddings(self, vector_db):
        """Test adding documents with externally provided embeddings."""
        vector_db.create_index()
        
        documents = [
            Document(content=f"Doc {i}", metadata={})
            for i in range(3)
        ]
        embeddings = [generate_embedding(64) for _ in range(3)]
        
        count = vector_db.add_documents(documents, embeddings=embeddings)
        
        assert count == 3
    
    def test_add_document_without_embedding_fails(self, vector_db):
        """Test that adding document without embedding fails."""
        vector_db.create_index()
        
        doc = Document(content="No embedding", metadata={})
        
        success = vector_db.add_document(doc)
        
        assert not success
        assert vector_db.document_count == 0
    
    def test_add_document_with_wrong_dimension_skipped(self, vector_db):
        """Test that documents with wrong dimension are skipped."""
        vector_db.create_index()
        
        doc = Document(
            content="Wrong dimension",
            metadata={},
            embedding=generate_embedding(128)  # Wrong dimension
        )
        
        success = vector_db.add_document(doc)
        
        assert not success
        assert vector_db.document_count == 0
    
    def test_add_empty_list_returns_zero(self, vector_db):
        """Test that adding empty list returns 0."""
        vector_db.create_index()
        
        count = vector_db.add_documents([])
        
        assert count == 0


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerSearch:
    """Tests for search functionality."""
    
    def test_search_returns_results(self, vector_db):
        """Test basic search functionality."""
        vector_db.create_index()
        
        # Add documents
        for i in range(5):
            doc = Document(
                content=f"Document {i}",
                metadata={"index": i},
                embedding=generate_embedding(64)
            )
            vector_db.add_document(doc)
        
        # Search
        query = generate_embedding(64)
        results = vector_db.search(query, top_k=3)
        
        assert len(results) == 3
        for doc, distance in results:
            assert isinstance(doc, Document)
            assert isinstance(distance, float)
    
    def test_search_with_top_k_larger_than_index(self, vector_db):
        """Test search when top_k is larger than index size."""
        vector_db.create_index()
        
        # Add only 2 documents
        for i in range(2):
            doc = Document(
                content=f"Document {i}",
                metadata={},
                embedding=generate_embedding(64)
            )
            vector_db.add_document(doc)
        
        # Search with top_k=10
        query = generate_embedding(64)
        results = vector_db.search(query, top_k=10)
        
        assert len(results) == 2  # Only 2 documents available
    
    def test_search_empty_index_returns_empty(self, vector_db):
        """Test search on empty index returns empty list."""
        vector_db.create_index()
        
        query = generate_embedding(64)
        results = vector_db.search(query, top_k=5)
        
        assert len(results) == 0
    
    def test_search_without_index_raises_error(self, vector_db):
        """Test search without creating index raises error."""
        query = generate_embedding(64)
        
        with pytest.raises(ResourceError):
            vector_db.search(query, top_k=5)
    
    def test_search_with_wrong_dimension_raises_error(self, vector_db):
        """Test search with wrong query dimension raises error."""
        vector_db.create_index()
        
        doc = Document(
            content="Test",
            metadata={},
            embedding=generate_embedding(64)
        )
        vector_db.add_document(doc)
        
        wrong_query = generate_embedding(128)  # Wrong dimension
        
        with pytest.raises(ValidationError):
            vector_db.search(wrong_query, top_k=1)
    
    def test_search_by_text_returns_documents(self, vector_db):
        """Test search_by_text returns document list."""
        vector_db.create_index()
        
        for i in range(3):
            doc = Document(
                content=f"Document {i}",
                metadata={},
                embedding=generate_embedding(64)
            )
            vector_db.add_document(doc)
        
        query = generate_embedding(64)
        results = vector_db.search_by_text(query, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerPersistence:
    """Tests for save/load functionality."""
    
    def test_save_and_load_index(self, temp_storage_path):
        """Test saving and loading index."""
        # Create and populate
        manager = VectorDBManager(dimension=64, storage_path=temp_storage_path)
        manager.create_index()
        
        for i in range(5):
            doc = Document(
                content=f"Document {i}",
                metadata={"index": i},
                embedding=generate_embedding(64)
            )
            manager.add_document(doc)
        
        # Save
        manager.save_index()
        
        # Load in new manager
        new_manager = VectorDBManager(storage_path=temp_storage_path)
        new_manager.load_index()
        
        assert new_manager.document_count == 5
        assert new_manager.dimension == 64
    
    def test_save_without_index_raises_error(self, temp_storage_path):
        """Test saving without index raises error."""
        manager = VectorDBManager(storage_path=temp_storage_path)
        
        with pytest.raises(ResourceError):
            manager.save_index()
    
    def test_save_without_path_raises_error(self):
        """Test saving without storage path raises error."""
        manager = VectorDBManager(dimension=64)
        manager.create_index()
        
        with pytest.raises(StorageError):
            manager.save_index()
    
    def test_load_nonexistent_path_raises_error(self, temp_storage_path):
        """Test loading from nonexistent path raises error."""
        manager = VectorDBManager(storage_path=temp_storage_path + "/nonexistent")
        
        with pytest.raises(StorageError):
            manager.load_index()
    
    def test_load_preserves_document_content(self, temp_storage_path):
        """Test that loaded documents have correct content."""
        # Create and save
        manager = VectorDBManager(dimension=64, storage_path=temp_storage_path)
        manager.create_index()
        
        doc = Document(
            content="Specific content to verify",
            metadata={"key": "value"},
            embedding=generate_embedding(64)
        )
        manager.add_document(doc)
        doc_id = doc.doc_id
        
        manager.save_index()
        
        # Load and verify
        new_manager = VectorDBManager(storage_path=temp_storage_path)
        new_manager.load_index()
        
        loaded_doc = new_manager.get_document(doc_id)
        
        assert loaded_doc is not None
        assert loaded_doc.content == "Specific content to verify"
        assert loaded_doc.metadata == {"key": "value"}


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerDocumentManagement:
    """Tests for document management."""
    
    def test_get_document_by_id(self, vector_db):
        """Test retrieving document by ID."""
        vector_db.create_index()
        
        doc = Document(
            content="Test content",
            metadata={"test": True},
            embedding=generate_embedding(64)
        )
        vector_db.add_document(doc)
        
        retrieved = vector_db.get_document(doc.doc_id)
        
        assert retrieved is not None
        assert retrieved.content == "Test content"
        assert retrieved.doc_id == doc.doc_id
    
    def test_get_nonexistent_document_returns_none(self, vector_db):
        """Test getting nonexistent document returns None."""
        vector_db.create_index()
        
        result = vector_db.get_document("nonexistent-id")
        
        assert result is None
    
    def test_remove_document(self, vector_db):
        """Test removing document from storage."""
        vector_db.create_index()
        
        doc = Document(
            content="To be removed",
            metadata={},
            embedding=generate_embedding(64)
        )
        vector_db.add_document(doc)
        
        assert vector_db.get_document(doc.doc_id) is not None
        
        success = vector_db.remove_document(doc.doc_id)
        
        assert success
        assert vector_db.get_document(doc.doc_id) is None
    
    def test_remove_nonexistent_document_returns_false(self, vector_db):
        """Test removing nonexistent document returns False."""
        vector_db.create_index()
        
        success = vector_db.remove_document("nonexistent-id")
        
        assert not success


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerUtilities:
    """Tests for utility methods."""
    
    def test_clear_removes_all(self, vector_db):
        """Test clear removes all data."""
        vector_db.create_index()
        
        for i in range(3):
            doc = Document(
                content=f"Doc {i}",
                metadata={},
                embedding=generate_embedding(64)
            )
            vector_db.add_document(doc)
        
        assert vector_db.document_count == 3
        
        vector_db.clear()
        
        assert vector_db.document_count == 0
        assert vector_db.is_empty
        assert vector_db._index is None
    
    def test_is_empty_property(self, vector_db):
        """Test is_empty property."""
        assert vector_db.is_empty
        
        vector_db.create_index()
        assert vector_db.is_empty
        
        doc = Document(
            content="Test",
            metadata={},
            embedding=generate_embedding(64)
        )
        vector_db.add_document(doc)
        
        assert not vector_db.is_empty
    
    def test_get_index_info(self, vector_db):
        """Test get_index_info returns correct information."""
        vector_db.create_index()
        
        info = vector_db.get_index_info()
        
        assert info["dimension"] == 64
        assert info["index_type"] == "flat"
        assert info["index_created"] == True
        assert info["document_count"] == 0
        assert info["faiss_available"] == True
    
    def test_rebuild_index(self, vector_db):
        """Test rebuilding index."""
        vector_db.create_index()
        
        # Add documents
        for i in range(5):
            doc = Document(
                content=f"Doc {i}",
                metadata={},
                embedding=generate_embedding(64)
            )
            vector_db.add_document(doc)
        
        # Remove one from storage
        doc_ids = list(vector_db._documents.keys())
        vector_db.remove_document(doc_ids[0])
        
        # Rebuild
        vector_db.rebuild_index()
        
        # Should have 4 documents now
        assert vector_db.document_count == 4


class TestGlobalVectorDB:
    """Tests for global vector DB functions."""
    
    def test_get_vector_db_returns_singleton(self):
        """Test get_vector_db returns singleton instance."""
        # Reset global
        import mm_orch.runtime.vector_db as vdb_module
        vdb_module._global_vector_db = None
        
        db1 = get_vector_db()
        db2 = get_vector_db()
        
        assert db1 is db2
    
    def test_configure_vector_db_creates_new_instance(self):
        """Test configure_vector_db creates new instance."""
        import mm_orch.runtime.vector_db as vdb_module
        vdb_module._global_vector_db = None
        
        db1 = get_vector_db(dimension=64)
        db2 = configure_vector_db(dimension=128)
        
        assert db1 is not db2
        assert db2.dimension == 128
