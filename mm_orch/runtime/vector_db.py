"""
向量库管理器

负责FAISS向量库的创建、索引和检索操作。
支持索引持久化、文档添加和向量检索。
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from mm_orch.exceptions import ResourceError, StorageError, ValidationError
from mm_orch.logger import get_logger
from mm_orch.schemas import Document


@dataclass
class IndexMetadata:
    """索引元数据"""
    dimension: int
    index_type: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    document_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "document_count": self.document_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """从字典创建"""
        return cls(
            dimension=data["dimension"],
            index_type=data["index_type"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            document_count=data.get("document_count", 0)
        )


class VectorDBManager:
    """
    向量库管理器
    
    功能:
    - 创建FAISS向量索引
    - 添加文档和向量
    - 向量相似度检索
    - 索引持久化和加载
    
    属性:
    - 属性11: 文档向量化完整性
    - 属性12: RAG检索相关性
    - 属性37: 向量库持久化往返
    """
    
    # 支持的索引类型
    SUPPORTED_INDEX_TYPES = {"flat", "ivf", "hnsw"}
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        storage_path: Optional[str] = None
    ):
        """
        初始化向量库管理器
        
        Args:
            dimension: 向量维度（默认384，适用于MiniLM）
            index_type: 索引类型 ('flat', 'ivf', 'hnsw')
            storage_path: 持久化存储路径
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None
        
        # FAISS索引
        self._index = None
        
        # 文档存储（ID到文档的映射）
        self._documents: Dict[str, Document] = {}
        
        # ID到索引位置的映射
        self._id_to_idx: Dict[str, int] = {}
        
        # 索引位置到ID的映射
        self._idx_to_id: List[str] = []
        
        # 索引元数据
        self._metadata: Optional[IndexMetadata] = None
        
        # 日志记录器
        self._logger = get_logger()
        
        # 检查FAISS是否可用
        self._faiss_available = self._check_faiss_available()
    
    def _check_faiss_available(self) -> bool:
        """检查FAISS是否可用"""
        try:
            import faiss
            return True
        except ImportError:
            self._logger.warning(
                "FAISS not installed. Vector database functionality will be limited.",
                context={"install_hint": "pip install faiss-cpu"}
            )
            return False
    
    def _validate_index_type(self, index_type: str) -> None:
        """验证索引类型"""
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValidationError(
                f"Unsupported index type: {index_type}",
                context={"supported_types": list(self.SUPPORTED_INDEX_TYPES)}
            )
    
    def create_index(
        self,
        dimension: Optional[int] = None,
        index_type: Optional[str] = None
    ) -> None:
        """
        创建向量索引
        
        Args:
            dimension: 向量维度（可选，使用初始化时的值）
            index_type: 索引类型（可选，使用初始化时的值）
            
        Raises:
            ResourceError: FAISS不可用时抛出
            ValidationError: 索引类型无效时抛出
        """
        if not self._faiss_available:
            raise ResourceError(
                "FAISS is not available",
                context={"install_hint": "pip install faiss-cpu"}
            )
        
        import faiss
        
        dim = dimension or self.dimension
        idx_type = index_type or self.index_type
        
        self._validate_index_type(idx_type)
        
        self._logger.info(
            f"Creating vector index",
            context={"dimension": dim, "index_type": idx_type}
        )
        
        # 根据索引类型创建索引
        if idx_type == "flat":
            # 精确搜索，适合小规模数据
            self._index = faiss.IndexFlatL2(dim)
        elif idx_type == "ivf":
            # IVF索引，适合中等规模数据
            # 使用较小的聚类中心数量以支持小数据集
            # 实际训练时会根据数据量调整
            quantizer = faiss.IndexFlatL2(dim)
            # 默认使用10个聚类中心，训练时会根据数据量调整
            self._index = faiss.IndexIVFFlat(quantizer, dim, 10)
            self._ivf_nlist = 10  # 记录当前聚类数
        elif idx_type == "hnsw":
            # HNSW索引，适合大规模数据
            self._index = faiss.IndexHNSWFlat(dim, 32)
        
        # 更新维度和类型
        self.dimension = dim
        self.index_type = idx_type
        
        # 创建元数据
        self._metadata = IndexMetadata(
            dimension=dim,
            index_type=idx_type
        )
        
        # 清空文档存储
        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        
        self._logger.info(
            f"Vector index created successfully",
            context={"dimension": dim, "index_type": idx_type}
        )

    def _ensure_index_exists(self) -> None:
        """确保索引已创建"""
        if self._index is None:
            self.create_index()
    
    def _get_embedding(self, document: Document) -> np.ndarray:
        """
        获取文档的向量嵌入
        
        Args:
            document: 文档对象
            
        Returns:
            向量嵌入
            
        Raises:
            ValidationError: 文档没有嵌入且无法生成时抛出
        """
        if document.embedding is not None:
            embedding = document.embedding
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            return embedding.astype(np.float32)
        
        raise ValidationError(
            "Document has no embedding",
            context={"doc_id": document.doc_id}
        )
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> int:
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表
            embeddings: 可选的向量嵌入列表（如果文档没有嵌入）
            
        Returns:
            成功添加的文档数量
            
        Raises:
            ResourceError: FAISS不可用时抛出
            ValidationError: 文档或嵌入无效时抛出
        """
        if not self._faiss_available:
            raise ResourceError(
                "FAISS is not available",
                context={"install_hint": "pip install faiss-cpu"}
            )
        
        if not documents:
            return 0
        
        self._ensure_index_exists()
        
        # 准备向量
        vectors = []
        valid_documents = []
        
        for i, doc in enumerate(documents):
            try:
                # 优先使用提供的嵌入
                if embeddings is not None and i < len(embeddings):
                    embedding = embeddings[i]
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                    # 更新文档的嵌入
                    doc.embedding = embedding
                else:
                    embedding = self._get_embedding(doc)
                
                # 验证维度
                if embedding.shape[0] != self.dimension:
                    self._logger.warning(
                        f"Embedding dimension mismatch for document {doc.doc_id}",
                        context={
                            "expected": self.dimension,
                            "got": embedding.shape[0]
                        }
                    )
                    continue
                
                vectors.append(embedding.reshape(1, -1))
                valid_documents.append(doc)
                
            except ValidationError as e:
                self._logger.warning(
                    f"Skipping document without embedding: {doc.doc_id}",
                    context={"error": str(e)}
                )
                continue
        
        if not vectors:
            return 0
        
        # 合并向量
        vectors_array = np.vstack(vectors).astype(np.float32)
        
        # 如果是IVF索引且未训练，先训练
        if self.index_type == "ivf" and not self._index.is_trained:
            import faiss
            
            # IVF需要至少nlist个训练样本
            nlist = getattr(self, '_ivf_nlist', 10)
            n_vectors = vectors_array.shape[0]
            
            if n_vectors < nlist:
                # 如果数据量太少，重新创建一个更小的IVF索引
                new_nlist = max(1, n_vectors)
                self._logger.info(
                    f"Adjusting IVF nlist from {nlist} to {new_nlist} due to small dataset",
                    context={"n_vectors": n_vectors}
                )
                quantizer = faiss.IndexFlatL2(self.dimension)
                self._index = faiss.IndexIVFFlat(quantizer, self.dimension, new_nlist)
                self._ivf_nlist = new_nlist
            
            self._logger.info("Training IVF index...")
            self._index.train(vectors_array)
        
        # 添加到索引
        start_idx = len(self._idx_to_id)
        self._index.add(vectors_array)
        
        # 更新文档存储和映射
        for i, doc in enumerate(valid_documents):
            idx = start_idx + i
            self._documents[doc.doc_id] = doc
            self._id_to_idx[doc.doc_id] = idx
            self._idx_to_id.append(doc.doc_id)
        
        # 更新元数据
        if self._metadata:
            self._metadata.document_count = len(self._documents)
            self._metadata.updated_at = time.time()
        
        self._logger.info(
            f"Added {len(valid_documents)} documents to vector index",
            context={"total_documents": len(self._documents)}
        )
        
        return len(valid_documents)
    
    def add_document(
        self,
        document: Document,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        添加单个文档到向量库
        
        Args:
            document: 文档对象
            embedding: 可选的向量嵌入
            
        Returns:
            是否成功添加
        """
        embeddings = [embedding] if embedding is not None else None
        return self.add_documents([document], embeddings) > 0

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        检索最相关的文档
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最大文档数量
            
        Returns:
            (文档, 距离)元组列表，按距离升序排列
            
        Raises:
            ResourceError: FAISS不可用或索引未创建时抛出
            ValidationError: 查询向量维度不匹配时抛出
        """
        if not self._faiss_available:
            raise ResourceError(
                "FAISS is not available",
                context={"install_hint": "pip install faiss-cpu"}
            )
        
        if self._index is None:
            raise ResourceError(
                "Vector index not created",
                context={"hint": "Call create_index() first"}
            )
        
        # 确保查询向量是正确的格式
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        query_vector = query_vector.astype(np.float32)
        
        # 验证维度
        if query_vector.shape[0] != self.dimension:
            raise ValidationError(
                "Query vector dimension mismatch",
                context={
                    "expected": self.dimension,
                    "got": query_vector.shape[0]
                }
            )
        
        # 重塑为2D数组
        query_vector = query_vector.reshape(1, -1)
        
        # 限制top_k不超过索引中的文档数量
        actual_k = min(top_k, self._index.ntotal)
        
        if actual_k == 0:
            return []
        
        # 执行搜索
        distances, indices = self._index.search(query_vector, actual_k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._idx_to_id):
                continue
            
            doc_id = self._idx_to_id[idx]
            if doc_id in self._documents:
                doc = self._documents[doc_id]
                distance = float(distances[0][i])
                results.append((doc, distance))
        
        self._logger.debug(
            f"Search completed",
            context={"top_k": top_k, "results_count": len(results)}
        )
        
        return results
    
    def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Document]:
        """
        根据查询嵌入检索文档（简化接口）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回的最大文档数量
            threshold: 可选的距离阈值，超过此值的结果将被过滤
            
        Returns:
            文档列表
        """
        results = self.search(query_embedding, top_k)
        
        if threshold is not None:
            results = [(doc, dist) for doc, dist in results if dist <= threshold]
        
        return [doc for doc, _ in results]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        根据ID获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档对象，如果不存在则返回None
        """
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """
        从向量库中移除文档
        
        注意：FAISS不支持直接删除，此方法仅从文档存储中移除
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否成功移除
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            # 注意：向量仍在索引中，但不会在结果中返回
            self._logger.info(
                f"Document removed from storage: {doc_id}",
                context={"note": "Vector still in index until rebuild"}
            )
            return True
        return False

    def save_index(self, path: Optional[str] = None) -> None:
        """
        持久化索引到磁盘
        
        Args:
            path: 保存路径（可选，使用初始化时的storage_path）
            
        Raises:
            StorageError: 保存失败时抛出
            ResourceError: 索引未创建时抛出
        """
        if not self._faiss_available:
            raise ResourceError(
                "FAISS is not available",
                context={"install_hint": "pip install faiss-cpu"}
            )
        
        if self._index is None:
            raise ResourceError(
                "No index to save",
                context={"hint": "Create and populate index first"}
            )
        
        import faiss
        
        save_path = Path(path) if path else self.storage_path
        if save_path is None:
            raise StorageError(
                "No storage path specified",
                context={"hint": "Provide path or set storage_path"}
            )
        
        # 确保目录存在
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存FAISS索引
            index_file = save_path / "index.faiss"
            faiss.write_index(self._index, str(index_file))
            
            # 保存元数据
            metadata_file = save_path / "metadata.json"
            if self._metadata:
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self._metadata.to_dict(), f, indent=2)
            
            # 保存文档数据
            documents_file = save_path / "documents.json"
            documents_data = {}
            for doc_id, doc in self._documents.items():
                doc_dict = {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "doc_id": doc.doc_id
                }
                # 不保存嵌入向量（太大），需要时重新生成
                documents_data[doc_id] = doc_dict
            
            with open(documents_file, 'w', encoding='utf-8') as f:
                json.dump(documents_data, f, indent=2, ensure_ascii=False)
            
            # 保存ID映射
            mapping_file = save_path / "id_mapping.json"
            mapping_data = {
                "id_to_idx": self._id_to_idx,
                "idx_to_id": self._idx_to_id
            }
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
            
            self._logger.info(
                f"Index saved successfully",
                context={
                    "path": str(save_path),
                    "document_count": len(self._documents)
                }
            )
            
        except Exception as e:
            raise StorageError(
                f"Failed to save index: {e}",
                context={"path": str(save_path)}
            )
    
    def load_index(self, path: Optional[str] = None) -> None:
        """
        从磁盘加载索引
        
        Args:
            path: 加载路径（可选，使用初始化时的storage_path）
            
        Raises:
            StorageError: 加载失败时抛出
            ResourceError: FAISS不可用时抛出
        """
        if not self._faiss_available:
            raise ResourceError(
                "FAISS is not available",
                context={"install_hint": "pip install faiss-cpu"}
            )
        
        import faiss
        
        load_path = Path(path) if path else self.storage_path
        if load_path is None:
            raise StorageError(
                "No storage path specified",
                context={"hint": "Provide path or set storage_path"}
            )
        
        if not load_path.exists():
            raise StorageError(
                f"Storage path does not exist: {load_path}",
                context={"path": str(load_path)}
            )
        
        try:
            # 加载FAISS索引
            index_file = load_path / "index.faiss"
            if not index_file.exists():
                raise StorageError(
                    f"Index file not found: {index_file}",
                    context={"path": str(load_path)}
                )
            
            self._index = faiss.read_index(str(index_file))
            
            # 加载元数据
            metadata_file = load_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    self._metadata = IndexMetadata.from_dict(metadata_dict)
                    self.dimension = self._metadata.dimension
                    self.index_type = self._metadata.index_type
            
            # 加载文档数据
            documents_file = load_path / "documents.json"
            if documents_file.exists():
                with open(documents_file, 'r', encoding='utf-8') as f:
                    documents_data = json.load(f)
                    self._documents = {}
                    for doc_id, doc_dict in documents_data.items():
                        self._documents[doc_id] = Document(
                            content=doc_dict["content"],
                            metadata=doc_dict["metadata"],
                            doc_id=doc_dict.get("doc_id", doc_id)
                        )
            
            # 加载ID映射
            mapping_file = load_path / "id_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    self._id_to_idx = {k: int(v) for k, v in mapping_data["id_to_idx"].items()}
                    self._idx_to_id = mapping_data["idx_to_id"]
            
            self._logger.info(
                f"Index loaded successfully",
                context={
                    "path": str(load_path),
                    "document_count": len(self._documents),
                    "index_size": self._index.ntotal
                }
            )
            
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to load index: {e}",
                context={"path": str(load_path)}
            )

    def rebuild_index(self) -> None:
        """
        重建索引（用于删除文档后清理）
        
        这会创建一个新索引，只包含当前存储中的文档
        """
        if not self._documents:
            self.create_index()
            return
        
        # 收集所有有嵌入的文档
        docs_with_embeddings = [
            doc for doc in self._documents.values()
            if doc.embedding is not None
        ]
        
        if not docs_with_embeddings:
            self.create_index()
            return
        
        # 保存当前配置
        dim = self.dimension
        idx_type = self.index_type
        
        # 重新创建索引
        self.create_index(dimension=dim, index_type=idx_type)
        
        # 重新添加文档
        self.add_documents(docs_with_embeddings)
        
        self._logger.info(
            f"Index rebuilt",
            context={"document_count": len(self._documents)}
        )
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        获取索引信息
        
        Returns:
            索引统计信息
        """
        info = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "faiss_available": self._faiss_available,
            "index_created": self._index is not None,
            "document_count": len(self._documents),
            "storage_path": str(self.storage_path) if self.storage_path else None
        }
        
        if self._index is not None:
            info["index_size"] = self._index.ntotal
            info["is_trained"] = getattr(self._index, 'is_trained', True)
        
        if self._metadata:
            info["metadata"] = self._metadata.to_dict()
        
        return info
    
    @property
    def document_count(self) -> int:
        """获取文档数量"""
        return len(self._documents)
    
    @property
    def index_size(self) -> int:
        """获取索引中的向量数量"""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    @property
    def is_empty(self) -> bool:
        """检查索引是否为空"""
        return self._index is None or self._index.ntotal == 0
    
    def clear(self) -> None:
        """清空索引和文档存储"""
        self._index = None
        self._documents.clear()
        self._id_to_idx.clear()
        self._idx_to_id.clear()
        self._metadata = None
        
        self._logger.info("Vector index cleared")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"VectorDBManager(dimension={self.dimension}, "
            f"index_type={self.index_type}, "
            f"documents={len(self._documents)})"
        )


# 全局向量库管理器实例
_global_vector_db: Optional[VectorDBManager] = None


def get_vector_db(
    dimension: int = 384,
    index_type: str = "flat",
    storage_path: Optional[str] = None
) -> VectorDBManager:
    """
    获取全局向量库管理器实例
    
    Args:
        dimension: 向量维度
        index_type: 索引类型
        storage_path: 存储路径
        
    Returns:
        VectorDBManager实例
    """
    global _global_vector_db
    
    if _global_vector_db is None:
        _global_vector_db = VectorDBManager(
            dimension=dimension,
            index_type=index_type,
            storage_path=storage_path
        )
    
    return _global_vector_db


def configure_vector_db(
    dimension: int = 384,
    index_type: str = "flat",
    storage_path: Optional[str] = None
) -> VectorDBManager:
    """
    配置全局向量库管理器
    
    Args:
        dimension: 向量维度
        index_type: 索引类型
        storage_path: 存储路径
        
    Returns:
        配置后的VectorDBManager实例
    """
    global _global_vector_db
    _global_vector_db = VectorDBManager(
        dimension=dimension,
        index_type=index_type,
        storage_path=storage_path
    )
    return _global_vector_db
