"""
API请求和响应数据模型

使用Pydantic定义API的请求和响应结构。

需求: 13.2 - 使用JSON格式进行请求和响应数据交换
属性34: API JSON格式
属性35: API错误响应格式
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import time


class WorkflowTypeEnum(str, Enum):
    """工作流类型枚举"""
    SEARCH_QA = "search_qa"
    LESSON_PACK = "lesson_pack"
    CHAT_GENERATE = "chat_generate"
    RAG_QA = "rag_qa"
    SELF_ASK_SEARCH_QA = "self_ask_search_qa"


# ============ 请求模型 ============

class QueryRequest(BaseModel):
    """
    通用查询请求
    
    用于POST /api/query端点，系统自动路由到合适的工作流
    """
    query: str = Field(..., min_length=1, max_length=10000, description="用户查询文本")
    context: Optional[Dict[str, Any]] = Field(default=None, description="附加上下文信息")
    session_id: Optional[str] = Field(default=None, description="会话ID（用于多轮对话）")
    preferences: Optional[Dict[str, Any]] = Field(default=None, description="用户偏好设置")
    workflow_type: Optional[WorkflowTypeEnum] = Field(
        default=None, 
        description="指定工作流类型（可选，不指定则自动路由）"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "什么是机器学习？",
                "context": {"domain": "technology"},
                "session_id": None,
                "preferences": {"language": "zh"},
                "workflow_type": None
            }
        }


class ChatRequest(BaseModel):
    """
    对话请求
    
    用于POST /api/chat端点
    """
    message: str = Field(..., min_length=1, max_length=10000, description="用户消息")
    session_id: Optional[str] = Field(default=None, description="会话ID，不提供则创建新会话")
    context: Optional[Dict[str, Any]] = Field(default=None, description="附加上下文")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "你好，请介绍一下你自己",
                "session_id": None,
                "context": None
            }
        }


class RAGUploadRequest(BaseModel):
    """
    RAG文档上传请求
    
    用于POST /api/rag/upload端点
    """
    content: str = Field(..., min_length=1, description="文档内容")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="文档元数据（来源、标题等）"
    )
    chunk_size: Optional[int] = Field(
        default=500, 
        ge=100, 
        le=2000, 
        description="文档分块大小"
    )
    chunk_overlap: Optional[int] = Field(
        default=50, 
        ge=0, 
        le=500, 
        description="分块重叠大小"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "这是一篇关于人工智能的文档...",
                "metadata": {"source": "wiki", "title": "人工智能简介"},
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }


class RAGQueryRequest(BaseModel):
    """
    RAG问答请求
    
    用于POST /api/rag/query端点
    """
    query: str = Field(..., min_length=1, max_length=5000, description="查询问题")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="检索文档数量")
    threshold: Optional[float] = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="相似度阈值"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "什么是深度学习？",
                "top_k": 5,
                "threshold": 0.5
            }
        }


class LessonPackRequest(BaseModel):
    """
    教学包生成请求
    
    用于POST /api/lesson端点
    """
    topic: str = Field(..., min_length=1, max_length=500, description="课题")
    level: Optional[str] = Field(
        default="intermediate", 
        description="难度级别（beginner/intermediate/advanced）"
    )
    num_exercises: Optional[int] = Field(
        default=3, 
        ge=1, 
        le=10, 
        description="练习题数量"
    )

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = {'beginner', 'intermediate', 'advanced'}
        if v not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Python基础语法",
                "level": "beginner",
                "num_exercises": 5
            }
        }


# ============ 响应模型 ============

class ErrorDetail(BaseModel):
    """错误详情"""
    code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(default=None, description="详细信息")


class ErrorResponse(BaseModel):
    """
    错误响应
    
    属性35: API错误响应格式
    对于任何格式错误的API请求，响应应该包含清晰的错误信息
    """
    success: bool = Field(default=False, description="请求是否成功")
    error: ErrorDetail = Field(..., description="错误详情")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Query cannot be empty",
                    "details": {"field": "query"}
                },
                "timestamp": 1700000000.0,
                "request_id": "req_123456"
            }
        }


class WorkflowResultResponse(BaseModel):
    """工作流执行结果响应"""
    success: bool = Field(..., description="执行是否成功")
    result: Optional[Any] = Field(default=None, description="执行结果")
    status: str = Field(..., description="状态（success/partial/failed）")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    error: Optional[str] = Field(default=None, description="错误信息")
    execution_time: Optional[float] = Field(default=None, description="执行时间（秒）")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "result": "机器学习是人工智能的一个分支...",
                "status": "success",
                "metadata": {"workflow_type": "search_qa", "execution_time": 1.5},
                "error": None,
                "execution_time": 1.5,
                "timestamp": 1700000000.0,
                "request_id": "req_123456"
            }
        }


class ChatResponse(BaseModel):
    """对话响应"""
    success: bool = Field(..., description="执行是否成功")
    response: Optional[str] = Field(default=None, description="助手回复")
    session_id: str = Field(..., description="会话ID")
    message_id: Optional[str] = Field(default=None, description="消息ID")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "response": "你好！我是MuAI助手...",
                "session_id": "sess_abc123",
                "message_id": "msg_xyz789",
                "timestamp": 1700000000.0,
                "request_id": "req_123456",
                "error": None
            }
        }


class RAGUploadResponse(BaseModel):
    """RAG文档上传响应"""
    success: bool = Field(..., description="上传是否成功")
    document_id: Optional[str] = Field(default=None, description="文档ID")
    chunks_count: Optional[int] = Field(default=None, description="分块数量")
    message: str = Field(..., description="结果消息")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_abc123",
                "chunks_count": 10,
                "message": "Document uploaded successfully",
                "timestamp": 1700000000.0,
                "request_id": "req_123456"
            }
        }


class RAGQueryResponse(BaseModel):
    """RAG问答响应"""
    success: bool = Field(..., description="查询是否成功")
    answer: Optional[str] = Field(default=None, description="生成的答案")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="来源文档")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "深度学习是机器学习的一个子领域...",
                "sources": [
                    {"doc_id": "doc_1", "content": "...", "score": 0.95}
                ],
                "timestamp": 1700000000.0,
                "request_id": "req_123456",
                "error": None
            }
        }


class LessonPackResponse(BaseModel):
    """教学包响应"""
    success: bool = Field(..., description="生成是否成功")
    topic: Optional[str] = Field(default=None, description="课题")
    plan: Optional[str] = Field(default=None, description="教学计划")
    explanation: Optional[str] = Field(default=None, description="讲解内容")
    exercises: List[Dict[str, str]] = Field(default_factory=list, description="练习题")
    timestamp: float = Field(default_factory=time.time, description="时间戳")
    request_id: Optional[str] = Field(default=None, description="请求ID")
    error: Optional[str] = Field(default=None, description="错误信息")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "topic": "Python基础语法",
                "plan": "1. 变量和数据类型\n2. 控制流...",
                "explanation": "Python是一种解释型语言...",
                "exercises": [
                    {"question": "什么是变量？", "answer": "变量是..."}
                ],
                "timestamp": 1700000000.0,
                "request_id": "req_123456",
                "error": None
            }
        }


class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    status: str = Field(..., description="系统状态（healthy/degraded/unhealthy）")
    version: str = Field(..., description="系统版本")
    uptime: float = Field(..., description="运行时间（秒）")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, 
        description="组件状态"
    )
    metrics: Dict[str, Any] = Field(default_factory=dict, description="性能指标")
    timestamp: float = Field(default_factory=time.time, description="时间戳")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "uptime": 3600.0,
                "components": {
                    "orchestrator": {"status": "healthy", "workflows": 5},
                    "consciousness": {"status": "healthy", "stage": "adult"},
                    "vector_db": {"status": "healthy", "documents": 100}
                },
                "metrics": {
                    "total_requests": 1000,
                    "success_rate": 0.95,
                    "avg_response_time": 0.5
                },
                "timestamp": 1700000000.0
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="健康状态")
    timestamp: float = Field(default_factory=time.time, description="时间戳")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": 1700000000.0
            }
        }
