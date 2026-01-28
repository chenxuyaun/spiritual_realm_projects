"""
错误处理系统

提供统一的错误处理接口和策略，包括错误分类、日志记录和降级策略。
支持严重错误的安全关闭机制（信号处理、状态保存）。

需求: 15.1, 15.3, 15.4, 15.5
"""

import atexit
import signal
import sys
import traceback
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mm_orch.exceptions import (
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
        self, error: Exception, context: Optional[Dict[str, Any]] = None
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
            context=context,
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
        self, error: ValidationError, context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理输入验证错误"""
        return ErrorResponse(
            error_type="ValidationError",
            error_message=str(error),
            user_message=f"输入验证失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="请检查输入参数的格式和类型",
        )

    def _handle_resource_error(
        self, error: ResourceError, context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理资源错误"""
        return ErrorResponse(
            error_type="ResourceError",
            error_message=str(error),
            user_message=f"资源不足或加载失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将尝试使用降级策略（如CPU模式或量化模型）",
        )

    def _handle_network_error(self, error: NetworkError, context: Dict[str, Any]) -> ErrorResponse:
        """处理网络错误"""
        return ErrorResponse(
            error_type="NetworkError",
            error_message=str(error),
            user_message=f"网络请求失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将自动重试，或使用本地知识库",
        )

    def _handle_model_error(self, error: ModelError, context: Dict[str, Any]) -> ErrorResponse:
        """处理模型推理错误"""
        return ErrorResponse(
            error_type="ModelError",
            error_message=str(error),
            user_message=f"模型推理失败: {error.message}",
            context=context,
            recoverable=True,
            suggested_action="系统将尝试使用备用模型",
        )

    def _handle_system_error(
        self, error: MuAISystemError, context: Dict[str, Any]
    ) -> ErrorResponse:
        """处理系统错误"""
        return ErrorResponse(
            error_type="SystemError",
            error_message=str(error),
            user_message=f"系统错误: {error.message}",
            context=context,
            recoverable=False,
            suggested_action="请联系系统管理员",
        )

    def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> ErrorResponse:
        """处理未知错误"""
        return ErrorResponse(
            error_type="UnknownError",
            error_message=str(error),
            user_message="发生未知错误，请稍后重试",
            context=context,
            recoverable=False,
            suggested_action="如果问题持续，请联系技术支持",
        )


# 全局错误处理器实例
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例（单例模式）"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


@dataclass
class ShutdownResult:
    """关闭操作的结果"""

    success: bool
    saved_components: List[str] = field(default_factory=list)
    failed_components: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    signal_received: Optional[str] = None


class GracefulShutdown:
    """
    安全关闭管理器

    实现需求15.5: 当发生严重错误时，系统应该保存当前状态并安全关闭。

    功能:
    - 注册信号处理器（SIGTERM、SIGINT）
    - 管理关闭前的状态保存回调
    - 确保关闭操作只执行一次
    - 支持自定义清理操作

    属性41: 严重错误安全关闭
    """

    def __init__(self):
        """初始化安全关闭管理器"""
        self._shutdown_callbacks: List[Callable[[], bool]] = []
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._is_shutting_down: bool = False
        self._shutdown_complete: bool = False
        self._lock = threading.Lock()
        self._last_result: Optional[ShutdownResult] = None
        self._signals_registered: bool = False
        self._original_handlers: Dict[int, Any] = {}

        logger.debug("GracefulShutdown manager initialized")

    def register_save_callback(
        self, callback: Callable[[], bool], name: Optional[str] = None
    ) -> None:
        """
        注册状态保存回调函数

        回调函数应该返回True表示保存成功，False表示失败。

        Args:
            callback: 保存状态的回调函数
            name: 可选的回调名称（用于日志）
        """
        # 包装回调以添加名称
        if name:
            callback._callback_name = name  # type: ignore
        self._shutdown_callbacks.append(callback)
        logger.debug(f"Registered save callback: {name or 'unnamed'}")

    def register_cleanup_callback(
        self, callback: Callable[[], None], name: Optional[str] = None
    ) -> None:
        """
        注册清理回调函数

        清理回调在保存回调之后执行，用于释放资源。

        Args:
            callback: 清理资源的回调函数
            name: 可选的回调名称（用于日志）
        """
        if name:
            callback._callback_name = name  # type: ignore
        self._cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {name or 'unnamed'}")

    def register_signal_handlers(self) -> None:
        """
        注册信号处理器

        处理SIGTERM和SIGINT信号，触发安全关闭流程。
        在Windows上，只处理SIGINT（Ctrl+C）。
        """
        if self._signals_registered:
            return

        try:
            # 保存原始处理器
            self._original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._signal_handler)

            # SIGTERM在Windows上可能不可用
            if hasattr(signal, "SIGTERM"):
                self._original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGTERM, self._signal_handler)

            # 注册atexit处理器
            atexit.register(self._atexit_handler)

            self._signals_registered = True
            logger.info("Signal handlers registered for graceful shutdown")

        except Exception as e:
            logger.warning(f"Failed to register signal handlers: {e}")

    def unregister_signal_handlers(self) -> None:
        """
        取消注册信号处理器

        恢复原始的信号处理器。主要用于测试。
        """
        if not self._signals_registered:
            return

        try:
            # 恢复原始处理器
            for sig, handler in self._original_handlers.items():
                if handler is not None:
                    signal.signal(sig, handler)

            self._original_handlers.clear()
            self._signals_registered = False
            logger.debug("Signal handlers unregistered")

        except Exception as e:
            logger.warning(f"Failed to unregister signal handlers: {e}")

    def _signal_handler(self, signum: int, frame) -> None:
        """
        信号处理函数

        Args:
            signum: 信号编号
            frame: 当前栈帧
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(f"Received signal {signal_name}, initiating graceful shutdown")

        result = self.shutdown(signal_name=signal_name)

        if not result.success:
            logger.error(f"Graceful shutdown had errors: {result.errors}")

        # 退出程序
        sys.exit(0)

    def _atexit_handler(self) -> None:
        """
        atexit处理函数

        在程序正常退出时执行，确保状态被保存。
        """
        if not self._shutdown_complete:
            logger.debug("atexit handler triggered, performing shutdown")
            self.shutdown(signal_name="atexit")

    def shutdown(self, signal_name: Optional[str] = None) -> ShutdownResult:
        """
        执行安全关闭流程

        Property 41: 严重错误安全关闭
        - 保存当前的意识状态和关键数据
        - 执行清理操作
        - 确保关闭操作只执行一次

        Args:
            signal_name: 触发关闭的信号名称

        Returns:
            ShutdownResult: 关闭操作的结果
        """
        with self._lock:
            # 防止重复关闭
            if self._is_shutting_down:
                logger.debug("Shutdown already in progress, skipping")
                return self._last_result or ShutdownResult(
                    success=True, signal_received=signal_name
                )

            if self._shutdown_complete:
                logger.debug("Shutdown already complete, skipping")
                return self._last_result or ShutdownResult(
                    success=True, signal_received=signal_name
                )

            self._is_shutting_down = True

        logger.info(f"Starting graceful shutdown (signal: {signal_name or 'manual'})")

        result = ShutdownResult(success=True, signal_received=signal_name)

        # 执行保存回调
        for callback in self._shutdown_callbacks:
            callback_name = getattr(callback, "_callback_name", "unnamed")
            try:
                logger.debug(f"Executing save callback: {callback_name}")
                save_success = callback()
                if save_success:
                    result.saved_components.append(callback_name)
                    logger.info(f"Successfully saved: {callback_name}")
                else:
                    result.failed_components.append(callback_name)
                    result.errors.append(f"Save failed for {callback_name}")
                    logger.warning(f"Failed to save: {callback_name}")
            except Exception as e:
                result.failed_components.append(callback_name)
                result.errors.append(f"Exception in {callback_name}: {str(e)}")
                logger.error(f"Exception during save callback {callback_name}: {e}")

        # 执行清理回调
        for callback in self._cleanup_callbacks:
            callback_name = getattr(callback, "_callback_name", "unnamed")
            try:
                logger.debug(f"Executing cleanup callback: {callback_name}")
                callback()
                logger.debug(f"Cleanup complete: {callback_name}")
            except Exception as e:
                result.errors.append(f"Cleanup exception in {callback_name}: {str(e)}")
                logger.error(f"Exception during cleanup callback {callback_name}: {e}")

        # 更新状态
        if result.failed_components:
            result.success = False

        with self._lock:
            self._shutdown_complete = True
            self._is_shutting_down = False
            self._last_result = result

        logger.info(
            f"Graceful shutdown complete. "
            f"Saved: {len(result.saved_components)}, "
            f"Failed: {len(result.failed_components)}"
        )

        return result

    def is_shutting_down(self) -> bool:
        """检查是否正在关闭"""
        return self._is_shutting_down

    def is_shutdown_complete(self) -> bool:
        """检查关闭是否已完成"""
        return self._shutdown_complete

    def get_last_result(self) -> Optional[ShutdownResult]:
        """获取最后一次关闭操作的结果"""
        return self._last_result

    def reset(self) -> None:
        """
        重置关闭管理器状态

        主要用于测试。
        """
        with self._lock:
            self._is_shutting_down = False
            self._shutdown_complete = False
            self._last_result = None
            self._shutdown_callbacks.clear()
            self._cleanup_callbacks.clear()
        logger.debug("GracefulShutdown manager reset")


# 全局安全关闭管理器实例
_graceful_shutdown: Optional[GracefulShutdown] = None


def get_graceful_shutdown() -> GracefulShutdown:
    """获取全局安全关闭管理器实例（单例模式）"""
    global _graceful_shutdown
    if _graceful_shutdown is None:
        _graceful_shutdown = GracefulShutdown()
    return _graceful_shutdown


def setup_graceful_shutdown() -> GracefulShutdown:
    """
    设置安全关闭机制

    这是一个便捷函数，用于初始化安全关闭管理器并注册信号处理器。
    通常在应用程序启动时调用。

    Returns:
        GracefulShutdown: 配置好的安全关闭管理器
    """
    shutdown_manager = get_graceful_shutdown()
    shutdown_manager.register_signal_handlers()

    # 注册意识状态保存回调
    try:
        from mm_orch.consciousness.core import save_consciousness

        shutdown_manager.register_save_callback(
            lambda: save_consciousness(force=True), name="consciousness_state"
        )
    except ImportError:
        logger.debug("Consciousness module not available for shutdown callback")

    # 注册持久化模块的保存回调
    try:
        from mm_orch.storage.persistence import get_consciousness_persistence

        persistence = get_consciousness_persistence()
        shutdown_manager.register_save_callback(
            lambda: persistence.save_state(force=True), name="persistence_state"
        )
        shutdown_manager.register_cleanup_callback(
            persistence.stop_auto_save, name="stop_auto_save"
        )
    except ImportError:
        logger.debug("Persistence module not available for shutdown callback")

    logger.info("Graceful shutdown mechanism configured")
    return shutdown_manager


def reset_graceful_shutdown() -> None:
    """
    重置安全关闭管理器

    主要用于测试。
    """
    global _graceful_shutdown
    if _graceful_shutdown:
        _graceful_shutdown.unregister_signal_handlers()
        _graceful_shutdown.reset()
    _graceful_shutdown = None
