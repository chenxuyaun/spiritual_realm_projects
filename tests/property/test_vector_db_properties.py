"""
Property-based tests for VectorDBManager.

Tests the correctness properties defined in the design document:
- Property 11: 文档向量化完整性
- Property 12: RAG检索相关性
- Property 37: 向量库持久化往返

**Validates: Requirements 5.1, 5.2, 5.3, 14.2**
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
import numpy as np

from mm_orch.runtime.vector_db import VectorDBManager, IndexMetadata
from mm_orch.schemas import Document
from mm_orch.exceptions import ResourceError, StorageError, ValidationError


# Check if FAISS is available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# Strategies for generating test data
dimension_strategy = st.integers(min_value=8, max_value=128)

index_type_strategy = st.sampled_from(["flat", "ivf", "hnsw"])

document_content_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Z')),
    min_size=10,
    max_size=500
).filter(lambda x: len(x.strip()) >= 10)

metadata_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip() == x),
    values=st.text(min_size=1, max_size=50),
    min_size=0,
    max_size=5
)

top_k_strategy = st.integers(min_value=1, max_value=20)


def generate_random_embedding(dimension: int) -> np.ndarray:
    """Generate a random normalized embedding vector."""
    vec = np.random.randn(dimension).astype(np.float32)
    # Normalize to unit length for better similarity search
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def create_document_with_embedding(content: str, metadata: dict, dimension: int) -> Document:
    """Create a document with a random embedding."""
    embedding = generate_random_embedding(dimension)
    return Document(content=content, metadata=metadata, embedding=embedding)


@pytest.fixture
def temp_storage_path():
    """Create a temporary directory for storage tests."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestProperty11DocumentVectorizationCompleteness:
    """
    Property 11: 文档向量化完整性
    
    对于任何上传的文档，系统应该将其分块并为每个块生成向量嵌入，
    且所有向量应该被添加到FAISS索引中，索引的大小应该增加相应的块数量。
    
    **Validates: Requirements 5.1, 5.2**
    """
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_documents_added_to_index(self, dimension, num_documents):
        """
        Feature: muai-orchestration-system, Property 11: 文档向量化完整性
        
        对于任何数量的文档，添加后索引大小应该等于文档数量。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Create documents with embeddings
        documents = []
        for i in range(num_documents):
            doc = Document(
                content=f"Test document content {i}",
                metadata={"index": i},
                embedding=generate_random_embedding(dimension)
            )
            documents.append(doc)
        
        # Add documents
        added_count = manager.add_documents(documents)
        
        # All documents should be added
        assert added_count == num_documents
        assert manager.index_size == num_documents
        assert manager.document_count == num_documents
    
    @given(
        dimension=dimension_strategy,
        content=document_content_strategy,
        metadata=metadata_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_single_document_added_correctly(self, dimension, content, metadata):
        """
        Feature: muai-orchestration-system, Property 11: 文档向量化完整性
        
        对于任何单个文档，添加后应该可以通过ID检索。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        embedding = generate_random_embedding(dimension)
        doc = Document(content=content, metadata=metadata, embedding=embedding)
        
        # Add document
        success = manager.add_document(doc)
        
        assert success
        assert manager.document_count == 1
        
        # Should be retrievable by ID
        retrieved = manager.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.content == content
        assert retrieved.doc_id == doc.doc_id
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_index_size_equals_document_count(self, dimension, num_documents):
        """
        Feature: muai-orchestration-system, Property 11: 文档向量化完整性
        
        索引大小应该始终等于成功添加的文档数量。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        for i in range(num_documents):
            doc = Document(
                content=f"Document {i}",
                metadata={},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
            
            # Invariant: index size equals document count
            assert manager.index_size == i + 1
            assert manager.document_count == i + 1


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestProperty12RAGRetrievalRelevance:
    """
    Property 12: RAG检索相关性
    
    对于任何RAG问答请求，检索到的文档片段数量应该等于请求的top_k值
    （或索引中的文档总数，如果少于top_k），且每个片段应该包含content和metadata字段。
    
    **Validates: Requirements 5.3**
    """
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=20),
        top_k=top_k_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_search_returns_correct_count(self, dimension, num_documents, top_k):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        检索结果数量应该等于min(top_k, 文档总数)。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Add documents
        for i in range(num_documents):
            doc = Document(
                content=f"Document content {i}",
                metadata={"index": i},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
        
        # Search
        query_vector = generate_random_embedding(dimension)
        results = manager.search(query_vector, top_k=top_k)
        
        # Result count should be min(top_k, num_documents)
        expected_count = min(top_k, num_documents)
        assert len(results) == expected_count
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=10),
        top_k=top_k_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_search_results_have_required_fields(self, dimension, num_documents, top_k):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        每个检索结果应该包含content和metadata字段。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Add documents
        for i in range(num_documents):
            doc = Document(
                content=f"Document content {i}",
                metadata={"index": i, "source": f"source_{i}"},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
        
        # Search
        query_vector = generate_random_embedding(dimension)
        results = manager.search(query_vector, top_k=top_k)
        
        # Each result should have required fields
        for doc, distance in results:
            assert hasattr(doc, 'content')
            assert hasattr(doc, 'metadata')
            assert doc.content is not None
            assert doc.metadata is not None
            assert isinstance(distance, float)
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_search_results_ordered_by_distance(self, dimension, num_documents):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        检索结果应该按距离升序排列。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Add documents
        for i in range(num_documents):
            doc = Document(
                content=f"Document {i}",
                metadata={},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
        
        # Search
        query_vector = generate_random_embedding(dimension)
        results = manager.search(query_vector, top_k=num_documents)
        
        # Results should be ordered by distance (ascending)
        distances = [dist for _, dist in results]
        assert distances == sorted(distances)
    
    @given(dimension=dimension_strategy)
    @settings(max_examples=50, deadline=None)
    def test_similar_query_returns_similar_document(self, dimension):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        使用文档自身的嵌入作为查询时，该文档应该是最相关的结果。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Create a specific document
        target_embedding = generate_random_embedding(dimension)
        target_doc = Document(
            content="Target document",
            metadata={"is_target": True},
            embedding=target_embedding
        )
        
        # Add target and other documents
        manager.add_document(target_doc)
        for i in range(5):
            other_doc = Document(
                content=f"Other document {i}",
                metadata={"is_target": False},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(other_doc)
        
        # Search with target's embedding
        results = manager.search(target_embedding, top_k=1)
        
        # Target should be the most similar
        assert len(results) == 1
        found_doc, distance = results[0]
        assert found_doc.doc_id == target_doc.doc_id
        # Distance should be very small (nearly 0 for exact match)
        assert distance < 0.01


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestProperty37VectorDBPersistenceRoundTrip:
    """
    Property 37: 向量库持久化往返
    
    对于任何向量库索引，保存到磁盘后再加载，加载后的索引应该包含相同数量的向量，
    且对相同查询的检索结果应该一致。
    
    **Validates: Requirements 14.2**
    """
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=15)
    )
    @settings(max_examples=50, deadline=None)
    def test_save_load_preserves_document_count(self, dimension, num_documents):
        """
        Feature: muai-orchestration-system, Property 37: 向量库持久化往返
        
        保存后加载的索引应该包含相同数量的文档。
        """
        temp_storage_path = tempfile.mkdtemp()
        try:
            # Create and populate index
            manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            manager.create_index()
            
            for i in range(num_documents):
                doc = Document(
                    content=f"Document {i}",
                    metadata={"index": i},
                    embedding=generate_random_embedding(dimension)
                )
                manager.add_document(doc)
            
            original_count = manager.document_count
            original_index_size = manager.index_size
            
            # Save
            manager.save_index()
            
            # Create new manager and load
            new_manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            new_manager.load_index()
            
            # Document count should be preserved
            assert new_manager.document_count == original_count
            assert new_manager.index_size == original_index_size
        finally:
            shutil.rmtree(temp_storage_path, ignore_errors=True)
    
    @given(
        dimension=dimension_strategy,
        num_documents=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50, deadline=None)
    def test_save_load_preserves_search_results(self, dimension, num_documents):
        """
        Feature: muai-orchestration-system, Property 37: 向量库持久化往返
        
        保存后加载的索引对相同查询应该返回相同的结果。
        """
        temp_storage_path = tempfile.mkdtemp()
        try:
            # Create and populate index
            manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            manager.create_index()
            
            for i in range(num_documents):
                doc = Document(
                    content=f"Document {i}",
                    metadata={"index": i},
                    embedding=generate_random_embedding(dimension)
                )
                manager.add_document(doc)
            
            # Generate a fixed query vector
            np.random.seed(42)
            query_vector = generate_random_embedding(dimension)
            
            # Search before save
            results_before = manager.search(query_vector, top_k=min(5, num_documents))
            doc_ids_before = [doc.doc_id for doc, _ in results_before]
            
            # Save
            manager.save_index()
            
            # Create new manager and load
            new_manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            new_manager.load_index()
            
            # Search after load
            results_after = new_manager.search(query_vector, top_k=min(5, num_documents))
            doc_ids_after = [doc.doc_id for doc, _ in results_after]
            
            # Results should be the same
            assert doc_ids_before == doc_ids_after
        finally:
            shutil.rmtree(temp_storage_path, ignore_errors=True)
    
    @given(
        dimension=dimension_strategy,
        content=document_content_strategy,
        metadata=metadata_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_save_load_preserves_document_content(self, dimension, content, metadata):
        """
        Feature: muai-orchestration-system, Property 37: 向量库持久化往返
        
        保存后加载的文档内容和元数据应该保持不变。
        """
        temp_storage_path = tempfile.mkdtemp()
        try:
            # Create and add document
            manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            manager.create_index()
            
            doc = Document(
                content=content,
                metadata=metadata,
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
            doc_id = doc.doc_id
            
            # Save
            manager.save_index()
            
            # Create new manager and load
            new_manager = VectorDBManager(dimension=dimension, storage_path=temp_storage_path)
            new_manager.load_index()
            
            # Retrieve document
            loaded_doc = new_manager.get_document(doc_id)
            
            assert loaded_doc is not None
            assert loaded_doc.content == content
            assert loaded_doc.metadata == metadata
            assert loaded_doc.doc_id == doc_id
        finally:
            shutil.rmtree(temp_storage_path, ignore_errors=True)
    
    @given(
        dimension=dimension_strategy,
        index_type=index_type_strategy
    )
    @settings(max_examples=30, deadline=None)
    def test_save_load_preserves_index_metadata(self, dimension, index_type):
        """
        Feature: muai-orchestration-system, Property 37: 向量库持久化往返
        
        保存后加载的索引元数据（维度、类型）应该保持不变。
        """
        temp_storage_path = tempfile.mkdtemp()
        try:
            # Create index with specific settings
            manager = VectorDBManager(
                dimension=dimension,
                index_type=index_type,
                storage_path=temp_storage_path
            )
            manager.create_index()
            
            # Add at least one document (required for IVF training)
            for i in range(10):  # IVF needs some data for training
                doc = Document(
                    content=f"Document {i}",
                    metadata={},
                    embedding=generate_random_embedding(dimension)
                )
                manager.add_document(doc)
            
            # Save
            manager.save_index()
            
            # Create new manager and load
            new_manager = VectorDBManager(storage_path=temp_storage_path)
            new_manager.load_index()
            
            # Metadata should be preserved
            assert new_manager.dimension == dimension
            assert new_manager.index_type == index_type
        finally:
            shutil.rmtree(temp_storage_path, ignore_errors=True)


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestVectorDBManagerInvariants:
    """
    Additional invariant tests for VectorDBManager.
    """
    
    @given(dimension=dimension_strategy)
    @settings(max_examples=50, deadline=None)
    def test_empty_index_returns_empty_results(self, dimension):
        """
        空索引的搜索应该返回空结果。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        query_vector = generate_random_embedding(dimension)
        results = manager.search(query_vector, top_k=10)
        
        assert len(results) == 0
    
    @given(
        dimension=dimension_strategy,
        wrong_dimension=dimension_strategy
    )
    @settings(max_examples=50, deadline=None)
    def test_dimension_mismatch_raises_error(self, dimension, wrong_dimension):
        """
        维度不匹配的查询应该抛出ValidationError。
        """
        assume(dimension != wrong_dimension)
        
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Add a document
        doc = Document(
            content="Test",
            metadata={},
            embedding=generate_random_embedding(dimension)
        )
        manager.add_document(doc)
        
        # Query with wrong dimension
        wrong_query = generate_random_embedding(wrong_dimension)
        
        with pytest.raises(ValidationError):
            manager.search(wrong_query, top_k=1)
    
    @given(dimension=dimension_strategy)
    @settings(max_examples=50, deadline=None)
    def test_clear_removes_all_data(self, dimension):
        """
        清空操作应该移除所有数据。
        """
        manager = VectorDBManager(dimension=dimension)
        manager.create_index()
        
        # Add documents
        for i in range(5):
            doc = Document(
                content=f"Document {i}",
                metadata={},
                embedding=generate_random_embedding(dimension)
            )
            manager.add_document(doc)
        
        assert manager.document_count > 0
        
        # Clear
        manager.clear()
        
        assert manager.document_count == 0
        assert manager.is_empty
    
    @given(
        dimension=dimension_strategy,
        index_type=index_type_strategy
    )
    @settings(max_examples=30, deadline=None)
    def test_index_info_consistency(self, dimension, index_type):
        """
        索引信息应该与实际状态一致。
        """
        manager = VectorDBManager(dimension=dimension, index_type=index_type)
        manager.create_index()
        
        info = manager.get_index_info()
        
        assert info["dimension"] == dimension
        assert info["index_type"] == index_type
        assert info["index_created"] == True
        assert info["document_count"] == 0
        assert info["faiss_available"] == True


@pytest.mark.skipif(FAISS_AVAILABLE, reason="Test for when FAISS is not installed")
class TestVectorDBWithoutFAISS:
    """
    Tests for VectorDBManager behavior when FAISS is not available.
    """
    
    def test_create_index_raises_resource_error(self):
        """
        没有FAISS时创建索引应该抛出ResourceError。
        """
        manager = VectorDBManager()
        manager._faiss_available = False
        
        with pytest.raises(ResourceError):
            manager.create_index()
    
    def test_add_documents_raises_resource_error(self):
        """
        没有FAISS时添加文档应该抛出ResourceError。
        """
        manager = VectorDBManager()
        manager._faiss_available = False
        
        doc = Document(content="Test", metadata={})
        
        with pytest.raises(ResourceError):
            manager.add_documents([doc])
