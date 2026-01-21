"""
API路由定义

实现核心API端点。

需求: 13.1, 12.5
- POST /api/query - 通用查询接口（自动路由）
- POST /api/chat - 对话接口
- POST /api/rag/upload - 文档上传
- POST /api/rag/query - RAG问答
- GET /api/status - 系统状态查询
"""

import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from mm_orch import __version__
from mm_orch.schemas import (
    UserRequest,
    WorkflowType,
    Document
)
from mm_orch.orchestrator import get_orchestrator, WorkflowOrchestrator
from mm_orch.consciousness import get_consciousness
from mm_orch.runtime.vector_db import get_vector_db, VectorDBManager
from mm_orch.logger import get_logger
from mm_orch.api.schemas import (
    QueryRequest,
    ChatRequest,
    RAGUploadRequest,
    RAGQueryRequest,
    LessonPackRequest,
    WorkflowResultResponse,
    ChatResponse,
    RAGUploadResponse,
    RAGQueryResponse,
    LessonPackResponse,
    SystemStatusResponse,
    HealthResponse,
    ErrorResponse
)
from mm_orch.api.auth import verify_api_key, optional_api_key
from mm_orch.api.app import get_uptime


logger = get_logger(__name__)


# 创建路由器
router = APIRouter(tags=["MuAI API"])


def get_request_id(request: Request) -> str:
    """获取请求ID"""
    return getattr(request.state, 'request_id', str(uuid.uuid4()))


# ============ 健康检查端点 ============

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查API服务是否正常运行"
)
async def health_check():
    """
    健康检查端点
    
    不需要认证，用于负载均衡器和监控系统
    """
    return HealthResponse(
        status="ok",
        timestamp=time.time()
    )


# ============ 系统状态端点 ============

@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="系统状态",
    description="获取系统运行状态和性能指标",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"}
    }
)
async def get_system_status(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    获取系统状态
    
    需求: 12.5 - 提供API接口查询系统状态和性能指标
    """
    request_id = get_request_id(request)
    
    try:
        orchestrator = get_orchestrator()
        consciousness = get_consciousness()
        vector_db = get_vector_db()
        
        # 获取编排器统计
        orch_stats = orchestrator.get_statistics()
        
        # 获取意识状态
        consciousness_status = consciousness.get_status_summary()
        
        # 获取向量库状态
        vector_db_status = {
            "status": "healthy",
            "documents": vector_db.get_document_count() if vector_db else 0
        }
        
        # 组件状态
        components = {
            "orchestrator": {
                "status": "healthy",
                "workflows": orch_stats.get("registered_workflows", 0)
            },
            "consciousness": {
                "status": "healthy",
                "stage": consciousness_status.get("development_stage", "unknown")
            },
            "vector_db": vector_db_status
        }
        
        # 性能指标
        metrics = {
            "total_requests": orch_stats.get("execution_count", 0),
            "success_rate": orch_stats.get("success_rate", 0.0),
            "avg_response_time": orch_stats.get("average_execution_time", 0.0)
        }
        
        return SystemStatusResponse(
            status="healthy",
            version=__version__,
            uptime=get_uptime(),
            components=components,
            metrics=metrics,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(
            "Failed to get system status",
            request_id=request_id,
            error=str(e)
        )
        return SystemStatusResponse(
            status="degraded",
            version=__version__,
            uptime=get_uptime(),
            components={},
            metrics={},
            timestamp=time.time()
        )


# ============ 通用查询端点 ============

@router.post(
    "/query",
    response_model=WorkflowResultResponse,
    summary="通用查询",
    description="提交查询请求，系统自动路由到合适的工作流",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def query(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    通用查询接口
    
    需求: 13.1 - 提供RESTful API接口用于所有核心功能
    
    支持自动路由或指定工作流类型
    """
    request_id = get_request_id(request)
    
    logger.info(
        "Query request received",
        request_id=request_id,
        query_length=len(query_request.query),
        workflow_type=query_request.workflow_type
    )
    
    try:
        orchestrator = get_orchestrator()
        
        # 如果指定了工作流类型，直接执行
        if query_request.workflow_type:
            workflow_type = WorkflowType(query_request.workflow_type.value)
            parameters = {
                "query": query_request.query,
                "context": query_request.context,
                "session_id": query_request.session_id,
                **(query_request.preferences or {})
            }
            result = orchestrator.execute_workflow(workflow_type, parameters)
        else:
            # 自动路由
            user_request = UserRequest(
                query=query_request.query,
                context=query_request.context,
                session_id=query_request.session_id,
                preferences=query_request.preferences
            )
            result = orchestrator.process_request(user_request)
        
        return WorkflowResultResponse(
            success=result.status == "success",
            result=result.result,
            status=result.status,
            metadata=result.metadata,
            error=result.error,
            execution_time=result.execution_time,
            timestamp=time.time(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "Query execution failed",
            request_id=request_id,
            error=str(e)
        )
        return WorkflowResultResponse(
            success=False,
            result=None,
            status="failed",
            metadata={},
            error=str(e),
            execution_time=None,
            timestamp=time.time(),
            request_id=request_id
        )


# ============ 对话端点 ============

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="对话",
    description="发送消息进行对话，支持多轮对话",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    对话接口
    
    需求: 4.1-4.5 - 多轮对话系统
    """
    request_id = get_request_id(request)
    
    logger.info(
        "Chat request received",
        request_id=request_id,
        session_id=chat_request.session_id,
        message_length=len(chat_request.message)
    )
    
    try:
        orchestrator = get_orchestrator()
        
        # 构建参数
        parameters = {
            "query": chat_request.message,
            "message": chat_request.message,
            "session_id": chat_request.session_id,
            "context": chat_request.context
        }
        
        # 执行对话工作流
        result = orchestrator.execute_workflow(
            WorkflowType.CHAT_GENERATE,
            parameters
        )
        
        # 从结果中提取会话ID
        session_id = chat_request.session_id
        if result.metadata and "session_id" in result.metadata:
            session_id = result.metadata["session_id"]
        elif not session_id:
            session_id = str(uuid.uuid4())
        
        return ChatResponse(
            success=result.status == "success",
            response=result.result if result.status == "success" else None,
            session_id=session_id,
            message_id=result.metadata.get("message_id") if result.metadata else None,
            timestamp=time.time(),
            request_id=request_id,
            error=result.error
        )
        
    except Exception as e:
        logger.error(
            "Chat execution failed",
            request_id=request_id,
            error=str(e)
        )
        return ChatResponse(
            success=False,
            response=None,
            session_id=chat_request.session_id or str(uuid.uuid4()),
            message_id=None,
            timestamp=time.time(),
            request_id=request_id,
            error=str(e)
        )


# ============ RAG端点 ============

@router.post(
    "/rag/upload",
    response_model=RAGUploadResponse,
    summary="上传文档",
    description="上传文档到RAG知识库",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def rag_upload(
    request: Request,
    upload_request: RAGUploadRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    RAG文档上传接口
    
    需求: 5.1, 5.2 - 文档向量化和存储
    """
    request_id = get_request_id(request)
    
    logger.info(
        "RAG upload request received",
        request_id=request_id,
        content_length=len(upload_request.content)
    )
    
    try:
        vector_db = get_vector_db()
        
        # 创建文档
        doc_id = str(uuid.uuid4())
        document = Document(
            content=upload_request.content,
            metadata=upload_request.metadata or {},
            doc_id=doc_id
        )
        
        # 添加到向量库
        chunks_count = vector_db.add_documents(
            [document],
            chunk_size=upload_request.chunk_size,
            chunk_overlap=upload_request.chunk_overlap
        )
        
        return RAGUploadResponse(
            success=True,
            document_id=doc_id,
            chunks_count=chunks_count,
            message="Document uploaded successfully",
            timestamp=time.time(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "RAG upload failed",
            request_id=request_id,
            error=str(e)
        )
        return RAGUploadResponse(
            success=False,
            document_id=None,
            chunks_count=None,
            message=f"Upload failed: {str(e)}",
            timestamp=time.time(),
            request_id=request_id
        )


@router.post(
    "/rag/query",
    response_model=RAGQueryResponse,
    summary="RAG问答",
    description="基于知识库进行问答",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def rag_query(
    request: Request,
    query_request: RAGQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    RAG问答接口
    
    需求: 5.3, 5.4, 5.5 - RAG检索和答案生成
    """
    request_id = get_request_id(request)
    
    logger.info(
        "RAG query request received",
        request_id=request_id,
        query_length=len(query_request.query),
        top_k=query_request.top_k
    )
    
    try:
        orchestrator = get_orchestrator()
        
        # 构建参数
        parameters = {
            "query": query_request.query,
            "top_k": query_request.top_k,
            "threshold": query_request.threshold
        }
        
        # 执行RAG工作流
        result = orchestrator.execute_workflow(
            WorkflowType.RAG_QA,
            parameters
        )
        
        # 提取来源信息
        sources = []
        if result.metadata and "sources" in result.metadata:
            sources = result.metadata["sources"]
        
        return RAGQueryResponse(
            success=result.status == "success",
            answer=result.result if result.status == "success" else None,
            sources=sources,
            timestamp=time.time(),
            request_id=request_id,
            error=result.error
        )
        
    except Exception as e:
        logger.error(
            "RAG query failed",
            request_id=request_id,
            error=str(e)
        )
        return RAGQueryResponse(
            success=False,
            answer=None,
            sources=[],
            timestamp=time.time(),
            request_id=request_id,
            error=str(e)
        )


# ============ 教学包端点 ============

@router.post(
    "/lesson",
    response_model=LessonPackResponse,
    summary="生成教学包",
    description="根据课题生成完整的教学内容",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def generate_lesson(
    request: Request,
    lesson_request: LessonPackRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    教学包生成接口
    
    需求: 3.1-3.5 - 教学工作流
    """
    request_id = get_request_id(request)
    
    logger.info(
        "Lesson pack request received",
        request_id=request_id,
        topic=lesson_request.topic,
        level=lesson_request.level
    )
    
    try:
        orchestrator = get_orchestrator()
        
        # 构建参数
        parameters = {
            "query": lesson_request.topic,
            "topic": lesson_request.topic,
            "level": lesson_request.level,
            "num_exercises": lesson_request.num_exercises
        }
        
        # 执行教学包工作流
        result = orchestrator.execute_workflow(
            WorkflowType.LESSON_PACK,
            parameters
        )
        
        # 解析结果
        if result.status == "success" and isinstance(result.result, dict):
            return LessonPackResponse(
                success=True,
                topic=lesson_request.topic,
                plan=result.result.get("plan", ""),
                explanation=result.result.get("explanation", ""),
                exercises=result.result.get("exercises", []),
                timestamp=time.time(),
                request_id=request_id,
                error=None
            )
        else:
            return LessonPackResponse(
                success=False,
                topic=lesson_request.topic,
                plan=None,
                explanation=None,
                exercises=[],
                timestamp=time.time(),
                request_id=request_id,
                error=result.error or "Failed to generate lesson pack"
            )
        
    except Exception as e:
        logger.error(
            "Lesson pack generation failed",
            request_id=request_id,
            error=str(e)
        )
        return LessonPackResponse(
            success=False,
            topic=lesson_request.topic,
            plan=None,
            explanation=None,
            exercises=[],
            timestamp=time.time(),
            request_id=request_id,
            error=str(e)
        )


# ============ 搜索问答端点 ============

@router.post(
    "/search",
    response_model=WorkflowResultResponse,
    summary="搜索问答",
    description="通过网络搜索回答问题",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"},
        422: {"model": ErrorResponse, "description": "请求验证失败"}
    }
)
async def search_qa(
    request: Request,
    query_request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    搜索问答接口
    
    需求: 2.1-2.5 - 搜索问答工作流
    """
    request_id = get_request_id(request)
    
    logger.info(
        "Search QA request received",
        request_id=request_id,
        query_length=len(query_request.query)
    )
    
    try:
        orchestrator = get_orchestrator()
        
        # 构建参数
        parameters = {
            "query": query_request.query,
            "context": query_request.context,
            **(query_request.preferences or {})
        }
        
        # 执行搜索问答工作流
        result = orchestrator.execute_workflow(
            WorkflowType.SEARCH_QA,
            parameters
        )
        
        return WorkflowResultResponse(
            success=result.status == "success",
            result=result.result,
            status=result.status,
            metadata=result.metadata,
            error=result.error,
            execution_time=result.execution_time,
            timestamp=time.time(),
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(
            "Search QA failed",
            request_id=request_id,
            error=str(e)
        )
        return WorkflowResultResponse(
            success=False,
            result=None,
            status="failed",
            metadata={},
            error=str(e),
            execution_time=None,
            timestamp=time.time(),
            request_id=request_id
        )


# ============ 意识状态端点 ============

@router.get(
    "/consciousness",
    summary="意识状态",
    description="获取意识模块状态",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"}
    }
)
async def get_consciousness_status(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    获取意识模块状态
    
    需求: 6.1-6.5 - 意识核心系统
    """
    request_id = get_request_id(request)
    
    try:
        consciousness = get_consciousness()
        status_summary = consciousness.get_status_summary()
        
        return JSONResponse(
            content={
                "success": True,
                "consciousness": status_summary,
                "timestamp": time.time(),
                "request_id": request_id
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to get consciousness status",
            request_id=request_id,
            error=str(e)
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "CONSCIOUSNESS_ERROR",
                    "message": str(e)
                },
                "timestamp": time.time(),
                "request_id": request_id
            }
        )


# ============ 工作流指标端点 ============

@router.get(
    "/metrics",
    summary="性能指标",
    description="获取工作流执行指标",
    responses={
        401: {"model": ErrorResponse, "description": "未提供API密钥"},
        403: {"model": ErrorResponse, "description": "无效的API密钥"}
    }
)
async def get_metrics(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    获取性能指标
    
    需求: 12.4, 12.5 - 性能指标收集和查询
    """
    request_id = get_request_id(request)
    
    try:
        orchestrator = get_orchestrator()
        
        # 获取编排器统计
        orch_stats = orchestrator.get_statistics()
        
        # 获取工作流指标
        workflow_metrics = orchestrator.get_workflow_metrics()
        
        return JSONResponse(
            content={
                "success": True,
                "orchestrator": orch_stats,
                "workflows": workflow_metrics,
                "timestamp": time.time(),
                "request_id": request_id
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to get metrics",
            request_id=request_id,
            error=str(e)
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "METRICS_ERROR",
                    "message": str(e)
                },
                "timestamp": time.time(),
                "request_id": request_id
            }
        )
