"""
Property-based tests for RAGQA Workflow.

This module contains property-based tests using Hypothesis to verify
the correctness properties of the RAGQA workflow.

Properties tested:
- Property 12: RAG检索相关性 (RAG retrieval relevance)
- Property 13: RAG答案来源标注 (RAG answer source annotation)
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch

from mm_orch.workflows.rag_qa import RAGQAWorkflow, RAGQAContext
from mm_orch.schemas import WorkflowResult, WorkflowType, Document
from mm_orch.runtime.vector_db import VectorDBManager
from mm_orch.exceptions import ValidationError


# Strategies for generating test data
query_strategy = st.text(min_size=1, max_size=200).filter(lambda x: x.strip())

top_k_strategy = st.integers(min_value=1, max_value=20)

document_content_strategy = st.text(min_size=10, max_size=500).filter(lambda x: x.strip())

document_metadata_strategy = st.fixed_dictionaries({
    "source": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
}).map(lambda d: {**d, "source": d["source"] or "unknown"})


@st.composite
def document_strategy(draw):
    """Strategy for generating Document objects."""
    content = draw(document_content_strategy)
    metadata = draw(document_metadata_strategy)
    doc_id = draw(st.uuids().map(str))
    
    # Generate a random embedding
    embedding = np.random.randn(384).astype(np.float32)
    
    return Document(
        content=content,
        metadata=metadata,
        doc_id=doc_id,
        embedding=embedding
    )


@st.composite
def document_list_strategy(draw, min_size=1, max_size=10):
    """Strategy for generating lists of documents."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    return [draw(document_strategy()) for _ in range(size)]


class TestProperty12RAGRetrievalRelevance:
    """
    Property 12: RAG检索相关性
    
    对于任何RAG问答请求，检索到的文档片段数量应该等于请求的top_k值
    （或索引中的文档总数，如果少于top_k），且每个片段应该包含content和metadata字段。
    
    **Validates: Requirements 5.3**
    """
    
    @given(
        query=query_strategy,
        top_k=top_k_strategy
    )
    @settings(max_examples=10, deadline=5000)
    def test_retrieval_count_matches_top_k_or_total(self, query, top_k):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        检索到的文档数量应该等于min(top_k, 索引中的文档总数)
        """
        # Create mock vector database
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        
        # Generate random number of documents in the database
        total_docs = np.random.randint(1, 15)
        expected_count = min(top_k, total_docs)
        
        # Create mock documents
        mock_docs = [
            (
                Document(
                    content=f"Document content {i}",
                    metadata={"source": f"doc{i}.txt"},
                    doc_id=f"doc{i}"
                ),
                float(i) * 0.1  # distance
            )
            for i in range(expected_count)
        ]
        
        mock_db.search.return_value = mock_docs
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query, "top_k": top_k})
        
        # Property: retrieved count should match expected
        assert result.metadata["retrieved_count"] == expected_count
        
        # Property: each document should have content and metadata
        if result.metadata.get("sources"):
            for source in result.metadata["sources"]:
                assert "content_preview" in source
                assert "metadata" in source
    
    @given(documents=document_list_strategy(min_size=1, max_size=5))
    @settings(max_examples=10, deadline=5000)
    def test_retrieved_documents_have_required_fields(self, documents):
        """
        Feature: muai-orchestration-system, Property 12: RAG检索相关性
        
        每个检索到的文档片段应该包含content和metadata字段
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = [(doc, 0.5) for doc in documents]
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": "test query"})
        
        # Property: each source should have required fields
        for source in result.metadata.get("sources", []):
            assert "content_preview" in source
            assert "metadata" in source
            assert "doc_id" in source
            assert "distance" in source


class TestProperty13RAGSourceAnnotation:
    """
    Property 13: RAG答案来源标注
    
    对于任何RAG问答的结果，返回的WorkflowResult的metadata字段应该包含'sources'键，
    其值应该是一个列表，包含用于生成答案的文档片段的元数据。
    
    **Validates: Requirements 5.4, 5.5**
    """
    
    @given(
        query=query_strategy,
        documents=document_list_strategy(min_size=1, max_size=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_sources_key_always_present(self, query, documents):
        """
        Feature: muai-orchestration-system, Property 13: RAG答案来源标注
        
        返回的WorkflowResult的metadata字段应该包含'sources'键
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = [(doc, 0.5) for doc in documents]
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: sources key must always be present in metadata
        assert "sources" in result.metadata
    
    @given(
        query=query_strategy,
        documents=document_list_strategy(min_size=1, max_size=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_sources_is_list(self, query, documents):
        """
        Feature: muai-orchestration-system, Property 13: RAG答案来源标注
        
        sources的值应该是一个列表
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = [(doc, 0.5) for doc in documents]
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: sources must be a list
        assert isinstance(result.metadata["sources"], list)
    
    @given(
        query=query_strategy,
        documents=document_list_strategy(min_size=1, max_size=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_sources_contain_document_metadata(self, query, documents):
        """
        Feature: muai-orchestration-system, Property 13: RAG答案来源标注
        
        sources列表应该包含用于生成答案的文档片段的元数据
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = [(doc, 0.5) for doc in documents]
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        sources = result.metadata["sources"]
        
        # Property: number of sources should match retrieved documents
        assert len(sources) == len(documents)
        
        # Property: each source should contain the document's metadata
        for i, source in enumerate(sources):
            assert source["doc_id"] == documents[i].doc_id
            assert source["metadata"] == documents[i].metadata
    
    @given(query=query_strategy)
    @settings(max_examples=10, deadline=5000)
    def test_sources_empty_when_no_documents(self, query):
        """
        Feature: muai-orchestration-system, Property 13: RAG答案来源标注
        
        当没有检索到文档时，sources应该是空列表
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = []
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: sources should be empty list when no documents retrieved
        assert result.metadata["sources"] == []
    
    @given(
        query=query_strategy,
        documents=document_list_strategy(min_size=1, max_size=3)
    )
    @settings(max_examples=10, deadline=5000)
    def test_sources_excluded_when_disabled(self, query, documents):
        """
        Feature: muai-orchestration-system, Property 13: RAG答案来源标注
        
        当include_sources=False时，sources应该是空列表
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = [(doc, 0.5) for doc in documents]
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({
            "query": query,
            "include_sources": False
        })
        
        # Property: sources should be empty when disabled
        assert result.metadata["sources"] == []


class TestRAGQAWorkflowResultProperties:
    """Additional property tests for workflow result structure."""
    
    @given(query=query_strategy)
    @settings(max_examples=10, deadline=5000)
    def test_result_always_has_valid_status(self, query):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        工作流结果应该总是有有效的状态
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = []
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: status must be one of valid values
        assert result.status in ["success", "partial", "failed"]
    
    @given(query=query_strategy)
    @settings(max_examples=10, deadline=5000)
    def test_result_always_has_metadata(self, query):
        """
        Feature: muai-orchestration-system, Property 4: 结果结构完整性
        
        工作流结果应该总是包含metadata字典
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = []
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: metadata must be a dictionary
        assert isinstance(result.metadata, dict)
        
        # Property: metadata must contain workflow name
        assert "workflow" in result.metadata
        assert result.metadata["workflow"] == "RAGQA"
    
    @given(query=query_strategy)
    @settings(max_examples=10, deadline=5000)
    def test_result_contains_step_information(self, query):
        """
        Feature: muai-orchestration-system, Property 2: 工作流执行步骤顺序
        
        工作流结果应该包含执行步骤信息
        """
        mock_db = Mock(spec=VectorDBManager)
        mock_db.dimension = 384
        mock_db.is_empty = False
        mock_db.search.return_value = []
        
        # Create mock model manager to avoid loading real models
        mock_model_manager = Mock()
        mock_model_manager.get_embedding.return_value = np.random.randn(384).astype(np.float32)
        mock_model_manager.infer.return_value = "Test answer"
        
        workflow = RAGQAWorkflow(vector_db=mock_db, model_manager=mock_model_manager)
        
        result = workflow.run({"query": query})
        
        # Property: steps must be present in metadata
        assert "steps" in result.metadata
        assert isinstance(result.metadata["steps"], list)
        
        # Property: each step should have required fields
        for step in result.metadata["steps"]:
            assert "name" in step
            assert "success" in step
            assert "duration" in step


class TestRAGQAValidationProperties:
    """Property tests for parameter validation."""
    
    @given(query=query_strategy, top_k=top_k_strategy)
    @settings(max_examples=10, deadline=5000)
    def test_valid_parameters_always_pass_validation(self, query, top_k):
        """
        Feature: muai-orchestration-system
        
        有效的参数应该总是通过验证
        """
        workflow = RAGQAWorkflow()
        
        # Property: valid parameters should pass validation
        assert workflow.validate_parameters({
            "query": query,
            "top_k": top_k
        })
    
    @given(
        query=query_strategy,
        threshold=st.floats(min_value=0.0, max_value=100.0)
    )
    @settings(max_examples=10, deadline=5000)
    def test_valid_threshold_passes_validation(self, query, threshold):
        """
        Feature: muai-orchestration-system
        
        有效的threshold参数应该通过验证
        """
        assume(not np.isnan(threshold) and not np.isinf(threshold))
        
        workflow = RAGQAWorkflow()
        
        # Property: valid threshold should pass validation
        assert workflow.validate_parameters({
            "query": query,
            "threshold": threshold
        })
    
    @given(invalid_top_k=st.integers(max_value=0))
    @settings(max_examples=10, deadline=5000)
    def test_invalid_top_k_fails_validation(self, invalid_top_k):
        """
        Feature: muai-orchestration-system
        
        无效的top_k参数应该验证失败
        """
        workflow = RAGQAWorkflow()
        
        # Property: invalid top_k should fail validation
        with pytest.raises(ValidationError):
            workflow.validate_parameters({
                "query": "test",
                "top_k": invalid_top_k
            })
    
    @given(invalid_threshold=st.floats(max_value=-0.01))
    @settings(max_examples=10, deadline=5000)
    def test_negative_threshold_fails_validation(self, invalid_threshold):
        """
        Feature: muai-orchestration-system
        
        负数的threshold参数应该验证失败
        """
        assume(not np.isnan(invalid_threshold))
        
        workflow = RAGQAWorkflow()
        
        # Property: negative threshold should fail validation
        with pytest.raises(ValidationError):
            workflow.validate_parameters({
                "query": "test",
                "threshold": invalid_threshold
            })
