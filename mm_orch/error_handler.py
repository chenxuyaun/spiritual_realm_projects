"""
错误处理系统

提供统一的错误处理接口和策略，包括错误分类、日志记录和降级策略。
"""

import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from mm_orch.exceptions import (
    MuAIError,
    ValidationError,
    ResourceError,
    NetworkError,
    ModelError,
    SystemError as MuAISystemError,
)
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ErrorResponse:
    """错误响应数据结构"""
    error_type: str
    error_message: str
    user_message: str
    context: Dict[str, Any]
    recoverable: bool = False
    suggested_action: Optional[str] = None


class ErrorHandler:
    """统一错误处理器"""
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorResponse:
        """
        统一错误处理入口
        
        策略:
        1. 记录错误到日志系统（包含堆栈和上下文）
        2. 根据错误类型选择处理策略
        3. 返回用户友好的错误信息
        
        Args:
            error: 捕获的异常
            context: 错误发生时的上下文信息
            
        Returns:
            ErrorResponse: 结构化的错误响应
        """
        context = context or {}
        
        # 记录错误到日志系统
        logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # 根据错误类型分类处理
        if isinstance(error, ValidationError):
            return self._handle_validation_error(error, context)
        elif isinstance(error, ResourceError):
            return self._handle_resource_error(error, context)
        elif isinstance(error, NetworkError):
            return self._handle_network_error(error, context)
        elif isinstance(error, ModelError):
            return self._handle_model_error(error, context)
        elif isinstance(error, MuAISystemError):
            return self._handle_system_error(error, context)
        else:
            return self._handle_unknown_error(error, context)
    
    def _handle_validation_error(
        self, 
        error: ValidationError, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理输入验证错误"""
        return ErrorResponse(
            error_type="ValidationError",
            error_message=str(error),
            user_message=f"输入验证失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="请检查输入参数的格式和类型"
        )
    
    def _handle_resource_error(
        self, 
        error: ResourceError, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理资源错误"""
        return ErrorResponse(
            error_type="ResourceError",
            error_message=str(error),
            user_message=f"资源不足或加载失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将尝试使用降级策略（如CPU模式或量化模型）"
        )
    
    def _handle_network_error(
        self, 
        error: NetworkError, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理网络错误"""
        return ErrorResponse(
            error_type="NetworkError",
            error_message=str(error),
            user_message=f"网络请求失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将自动重试，或使用本地知识库"
        )
    
    def _handle_model_error(
        self, 
        error: ModelError, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理模型推理错误"""
        return ErrorResponse(
            error_type="ModelError",
            error_message=str(error),
            user_message=f"模型推理失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将尝试使用备用模型"
        )
    
    def _handle_system_error(
        self, 
        error: MuAISystemError, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理系统错误"""
        return ErrorResponse(
            error_type="SystemError",
            error_message=str(error),
            user_message=f"系统错误: {error.message}",
            context=context,
            recoverable=False,
            suggested_action="请联系系统管理员"
        )
    
    def _handle_unknown_error(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理未知错误"""
        return ErrorResponse(
            error_type="UnknownError",
            error_message=str(error),
            user_message="发生未知错误，请稍后重试",
            context=context,
            recoverable=False,
            suggested_action="如果问题持续，请联系技术支持"
        )


# 全局错误处理器实例
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例（单例模式）"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
