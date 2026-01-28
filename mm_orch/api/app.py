"""
FastAPI应用主模块

创建和配置FastAPI应用实例。

需求: 13.1, 13.2, 13.3
属性34: API JSON格式
属性35: API错误响应格式
"""

import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from mm_orch import __version__
from mm_orch.logger import get_logger
from mm_orch.exceptions import (
    MuAIError,
    ValidationError,
    ResourceError,
    NetworkError,
    WorkflowError,
)
from mm_orch.api.auth import APIKeyAuth, set_auth_manager


logger = get_logger(__name__)


# 应用启动时间
_start_time: float = 0.0


def get_uptime() -> float:
    """获取应用运行时间"""
    global _start_time
    if _start_time == 0:
        return 0.0
    return time.time() - _start_time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    处理启动和关闭事件
    """
    global _start_time

    # 启动时
    _start_time = time.time()
    logger.info("MuAI API server starting", version=__version__)

    yield

    # 关闭时
    logger.info("MuAI API server shutting down", uptime=get_uptime())


def create_app(
    title: str = "MuAI多模型编排系统API",
    description: str = "MuAI Multi-Model Orchestration System REST API",
    version: str = __version__,
    auth_enabled: bool = True,
    api_keys: Optional[set] = None,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    创建FastAPI应用实例

    Args:
        title: API标题
        description: API描述
        version: API版本
        auth_enabled: 是否启用认证
        api_keys: 初始API密钥集合
        cors_origins: CORS允许的源列表

    Returns:
        配置好的FastAPI应用实例
    """
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 配置认证
    auth_manager = APIKeyAuth(api_keys=api_keys, enabled=auth_enabled)
    set_auth_manager(auth_manager)

    # 配置CORS
    if cors_origins is None:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 添加请求ID中间件
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """为每个请求添加唯一ID"""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        logger.debug(
            "Request processed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time=process_time,
        )

        return response

    # 注册异常处理器
    _register_exception_handlers(app)

    # 注册路由
    _register_routes(app)

    logger.info("FastAPI app created", title=title, version=version, auth_enabled=auth_enabled)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """
    注册异常处理器

    属性35: API错误响应格式
    对于任何格式错误的API请求，响应应该包含清晰的错误信息（error字段）、
    适当的HTTP状态码（4xx或5xx），且响应体应该是有效的JSON。
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """处理请求验证错误"""
        request_id = getattr(request.state, "request_id", None)

        errors = []
        for error in exc.errors():
            errors.append(
                {
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                }
            )

        logger.warning("Request validation error", request_id=request_id, errors=errors)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"errors": errors},
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(ValidationError)
    async def muai_validation_exception_handler(request: Request, exc: ValidationError):
        """处理MuAI验证错误"""
        request_id = getattr(request.state, "request_id", None)

        logger.warning("MuAI validation error", request_id=request_id, error=str(exc))

        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": str(exc),
                    "details": exc.context if hasattr(exc, "context") else None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(ResourceError)
    async def resource_exception_handler(request: Request, exc: ResourceError):
        """处理资源错误"""
        request_id = getattr(request.state, "request_id", None)

        logger.error("Resource error", request_id=request_id, error=str(exc))

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "error": {
                    "code": "RESOURCE_ERROR",
                    "message": str(exc),
                    "details": exc.context if hasattr(exc, "context") else None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(NetworkError)
    async def network_exception_handler(request: Request, exc: NetworkError):
        """处理网络错误"""
        request_id = getattr(request.state, "request_id", None)

        logger.error("Network error", request_id=request_id, error=str(exc))

        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "success": False,
                "error": {
                    "code": "NETWORK_ERROR",
                    "message": str(exc),
                    "details": exc.context if hasattr(exc, "context") else None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(WorkflowError)
    async def workflow_exception_handler(request: Request, exc: WorkflowError):
        """处理工作流错误"""
        request_id = getattr(request.state, "request_id", None)

        logger.error("Workflow error", request_id=request_id, error=str(exc))

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "WORKFLOW_ERROR",
                    "message": str(exc),
                    "details": exc.context if hasattr(exc, "context") else None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(MuAIError)
    async def muai_exception_handler(request: Request, exc: MuAIError):
        """处理通用MuAI错误"""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            "MuAI error", request_id=request_id, error_type=type(exc).__name__, error=str(exc)
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": type(exc).__name__.upper(),
                    "message": str(exc),
                    "details": exc.context if hasattr(exc, "context") else None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理未捕获的异常"""
        request_id = getattr(request.state, "request_id", None)

        logger.error(
            "Unhandled exception",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": None,
                },
                "timestamp": time.time(),
                "request_id": request_id,
            },
        )


def _register_routes(app: FastAPI) -> None:
    """注册API路由"""
    from mm_orch.api.routes import router

    app.include_router(router, prefix="/api")


# 全局应用实例
_app_instance: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """获取全局应用实例"""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


def reset_app() -> None:
    """重置全局应用实例"""
    global _app_instance
    _app_instance = None
