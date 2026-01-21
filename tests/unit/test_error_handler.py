"""
错误处理系统的单元测试

测试特定的错误场景和边缘情况。
"""

import pytest
import time

from mm_orch.exceptions import (
    ValidationError,
    ResourceError,
    NetworkError,
    ModelError,
    SystemError as MuAISystemError,
    WorkflowError,
    ConsciousnessError,
    StorageError,
)
from mm_orch.error_handler import ErrorHandler, ErrorResponse, get_error_handler
from mm_orch.retry import RetryPolicy, retry, retry_with_policy


class TestErrorHandler:
    """测试ErrorHandler类"""
    
    def test_handle_validation_error(self):
        """测试处理ValidationError"""
        handler = ErrorHandler()
        error = ValidationError("Invalid parameter type")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "ValidationError"
        assert "Invalid parameter type" in response.error_message
        assert response.recoverable is True
        assert "输入参数" in response.suggested_action
    
    def test_handle_resource_error(self):
        """测试处理ResourceError"""
        handler = ErrorHandler()
        error = ResourceError("GPU memory insufficient")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "ResourceError"
        assert "GPU memory insufficient" in response.error_message
        assert response.recoverable is True
    
    def test_handle_network_error(self):
        """测试处理NetworkError"""
        handler = ErrorHandler()
        error = NetworkError("Connection timeout")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "NetworkError"
        assert "Connection timeout" in response.error_message
        assert response.recoverable is True
    
    def test_handle_model_error(self):
        """测试处理ModelError"""
        handler = ErrorHandler()
        error = ModelError("Model inference failed")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "ModelError"
        assert "Model inference failed" in response.error_message
        assert response.recoverable is True
    
    def test_handle_system_error(self):
        """测试处理SystemError"""
        handler = ErrorHandler()
        error = MuAISystemError("Configuration missing")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "SystemError"
        assert "Configuration missing" in response.error_message
        assert response.recoverable is False
    
    def test_handle_unknown_error(self):
        """测试处理未知错误类型"""
        handler = ErrorHandler()
        error = ValueError("Some unknown error")
        
        response = handler.handle_error(error)
        
        assert response.error_type == "UnknownError"
        assert "Some unknown error" in response.error_message
        assert response.recoverable is False
    
    def test_error_with_context(self):
        """测试带上下文的错误处理"""
        handler = ErrorHandler()
        error = ValidationError("Invalid input")
        context = {
            "user_id": "12345",
            "request_id": "abc-def",
            "timestamp": 1234567890
        }
        
        response = handler.handle_error(error, context)
        
        assert response.context == context
        assert "user_id" in response.context
    
    def test_get_error_handler_singleton(self):
        """测试单例模式"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        assert handler1 is handler2


class TestRetryPolicy:
    """测试RetryPolicy类"""
    
    def test_default_retry_policy(self):
        """测试默认重试策略"""
        policy = RetryPolicy()
        
        assert policy.max_retries == 3
        assert policy.backoff_factor == 2.0
        assert NetworkError in policy.retry_on
        assert TimeoutError in policy.retry_on
    
    def test_custom_retry_policy(self):
        """测试自定义重试策略"""
        policy = RetryPolicy(
            max_retries=5,
            backoff_factor=1.5,
            retry_on=[NetworkError]
        )
        
        assert policy.max_retries == 5
        assert policy.backoff_factor == 1.5
        assert policy.retry_on == [NetworkError]
    
    def test_should_retry_with_retryable_error(self):
        """测试可重试的错误"""
        policy = RetryPolicy(max_retries=3)
        error = NetworkError("Timeout")
        
        assert policy.should_retry(error, 0) is True
        assert policy.should_retry(error, 1) is True
        assert policy.should_retry(error, 2) is True
        assert policy.should_retry(error, 3) is False
    
    def test_should_retry_with_non_retryable_error(self):
        """测试不可重试的错误"""
        policy = RetryPolicy(max_retries=3)
        error = ValueError("Invalid value")
        
        assert policy.should_retry(error, 0) is False
        assert policy.should_retry(error, 1) is False
    
    def test_get_delay_exponential_backoff(self):
        """测试指数退避延迟计算"""
        policy = RetryPolicy(backoff_factor=2.0)
        
        assert policy.get_delay(0) == 1.0  # 2^0
        assert policy.get_delay(1) == 2.0  # 2^1
        assert policy.get_delay(2) == 4.0  # 2^2
        assert policy.get_delay(3) == 8.0  # 2^3


class TestRetryDecorator:
    """测试retry装饰器"""
    
    def test_retry_success_on_first_attempt(self):
        """测试第一次尝试就成功"""
        call_count = [0]
        
        @retry(max_retries=3)
        def successful_function():
            call_count[0] += 1
            return "success"
        
        result = successful_function()
        
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_success_after_failures(self):
        """测试重试后成功"""
        call_count = [0]
        
        @retry(max_retries=3, backoff_factor=0.1)
        def eventually_successful_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise NetworkError("Temporary failure")
            return "success"
        
        result = eventually_successful_function()
        
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        call_count = [0]
        
        @retry(max_retries=2, backoff_factor=0.1)
        def always_failing_function():
            call_count[0] += 1
            raise NetworkError("Permanent failure")
        
        with pytest.raises(NetworkError):
            always_failing_function()
        
        # 初始调用 + 2次重试 = 3次调用
        assert call_count[0] == 3
    
    def test_retry_non_retryable_error(self):
        """测试不可重试的错误立即抛出"""
        call_count = [0]
        
        @retry(max_retries=3)
        def function_with_non_retryable_error():
            call_count[0] += 1
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            function_with_non_retryable_error()
        
        # 只调用一次，不重试
        assert call_count[0] == 1
    
    def test_retry_with_custom_policy(self):
        """测试使用自定义策略的重试"""
        call_count = [0]
        custom_policy = RetryPolicy(
            max_retries=5,
            backoff_factor=0.1,
            retry_on=[NetworkError]
        )
        
        @retry_with_policy(custom_policy)
        def function_with_custom_policy():
            call_count[0] += 1
            raise NetworkError("Test error")
        
        with pytest.raises(NetworkError):
            function_with_custom_policy()
        
        # 初始调用 + 5次重试 = 6次调用
        assert call_count[0] == 6


class TestCustomExceptions:
    """测试自定义异常类"""
    
    def test_validation_error_with_context(self):
        """测试带上下文的ValidationError"""
        context = {"field": "username", "value": ""}
        error = ValidationError("Username cannot be empty", context)
        
        assert error.message == "Username cannot be empty"
        assert error.context == context
    
    def test_resource_error_inheritance(self):
        """测试ResourceError继承关系"""
        error = ResourceError("Out of memory")
        
        assert isinstance(error, Exception)
        assert hasattr(error, 'message')
        assert hasattr(error, 'context')
    
    def test_workflow_error(self):
        """测试WorkflowError"""
        error = WorkflowError("Workflow step failed")
        
        assert error.message == "Workflow step failed"
        assert isinstance(error.context, dict)
    
    def test_consciousness_error(self):
        """测试ConsciousnessError"""
        error = ConsciousnessError("Consciousness module initialization failed")
        
        assert "initialization failed" in error.message
    
    def test_storage_error(self):
        """测试StorageError"""
        error = StorageError("Failed to save state")
        
        assert "save state" in error.message


# ============================================================================
# 安全关闭机制测试
# ============================================================================

from mm_orch.error_handler import (
    GracefulShutdown,
    ShutdownResult,
    get_graceful_shutdown,
    reset_graceful_shutdown,
    setup_graceful_shutdown,
)


class TestGracefulShutdown:
    """测试GracefulShutdown类"""
    
    @pytest.fixture(autouse=True)
    def reset_shutdown(self):
        """每个测试前后重置关闭管理器"""
        reset_graceful_shutdown()
        yield
        reset_graceful_shutdown()
    
    def test_register_save_callback(self):
        """测试注册保存回调"""
        manager = GracefulShutdown()
        
        def save_callback():
            return True
        
        manager.register_save_callback(save_callback, name="test_save")
        
        # 验证回调被注册
        assert len(manager._shutdown_callbacks) == 1
    
    def test_register_cleanup_callback(self):
        """测试注册清理回调"""
        manager = GracefulShutdown()
        
        def cleanup_callback():
            pass
        
        manager.register_cleanup_callback(cleanup_callback, name="test_cleanup")
        
        # 验证回调被注册
        assert len(manager._cleanup_callbacks) == 1
    
    def test_shutdown_executes_all_callbacks(self):
        """测试关闭时执行所有回调"""
        manager = GracefulShutdown()
        
        executed = []
        
        def callback_1():
            executed.append(1)
            return True
        
        def callback_2():
            executed.append(2)
            return True
        
        manager.register_save_callback(callback_1, name="cb1")
        manager.register_save_callback(callback_2, name="cb2")
        
        result = manager.shutdown()
        
        assert executed == [1, 2]
        assert result.success is True
        assert len(result.saved_components) == 2
    
    def test_shutdown_handles_failed_callback(self):
        """测试关闭时处理失败的回调"""
        manager = GracefulShutdown()
        
        def failing_callback():
            return False
        
        manager.register_save_callback(failing_callback, name="failing")
        
        result = manager.shutdown()
        
        assert result.success is False
        assert "failing" in result.failed_components
    
    def test_shutdown_handles_exception_in_callback(self):
        """测试关闭时处理回调中的异常"""
        manager = GracefulShutdown()
        
        def exception_callback():
            raise RuntimeError("Test exception")
        
        manager.register_save_callback(exception_callback, name="exception")
        
        result = manager.shutdown()
        
        assert result.success is False
        assert "exception" in result.failed_components
        assert any("Test exception" in err for err in result.errors)
    
    def test_shutdown_only_runs_once(self):
        """测试关闭只执行一次"""
        manager = GracefulShutdown()
        
        call_count = [0]
        
        def counting_callback():
            call_count[0] += 1
            return True
        
        manager.register_save_callback(counting_callback, name="counter")
        
        manager.shutdown()
        manager.shutdown()
        manager.shutdown()
        
        assert call_count[0] == 1
    
    def test_is_shutting_down_state(self):
        """测试关闭状态跟踪"""
        manager = GracefulShutdown()
        
        assert manager.is_shutting_down() is False
        assert manager.is_shutdown_complete() is False
        
        manager.shutdown()
        
        assert manager.is_shutting_down() is False
        assert manager.is_shutdown_complete() is True
    
    def test_get_last_result(self):
        """测试获取最后一次关闭结果"""
        manager = GracefulShutdown()
        
        assert manager.get_last_result() is None
        
        manager.register_save_callback(lambda: True, name="test")
        manager.shutdown(signal_name="SIGTERM")
        
        result = manager.get_last_result()
        assert result is not None
        assert result.signal_received == "SIGTERM"
    
    def test_reset_clears_state(self):
        """测试重置清除状态"""
        manager = GracefulShutdown()
        
        manager.register_save_callback(lambda: True, name="test")
        manager.shutdown()
        
        assert manager.is_shutdown_complete() is True
        
        manager.reset()
        
        assert manager.is_shutdown_complete() is False
        assert manager.get_last_result() is None
        assert len(manager._shutdown_callbacks) == 0
    
    def test_cleanup_callbacks_run_after_save(self):
        """测试清理回调在保存回调之后运行"""
        manager = GracefulShutdown()
        
        order = []
        
        def save_cb():
            order.append("save")
            return True
        
        def cleanup_cb():
            order.append("cleanup")
        
        manager.register_save_callback(save_cb, name="save")
        manager.register_cleanup_callback(cleanup_cb, name="cleanup")
        
        manager.shutdown()
        
        assert order == ["save", "cleanup"]
    
    def test_cleanup_runs_even_if_save_fails(self):
        """测试即使保存失败，清理也会运行"""
        manager = GracefulShutdown()
        
        cleanup_ran = [False]
        
        def failing_save():
            return False
        
        def cleanup_cb():
            cleanup_ran[0] = True
        
        manager.register_save_callback(failing_save, name="save")
        manager.register_cleanup_callback(cleanup_cb, name="cleanup")
        
        manager.shutdown()
        
        assert cleanup_ran[0] is True


class TestGracefulShutdownSingleton:
    """测试安全关闭管理器的单例模式"""
    
    @pytest.fixture(autouse=True)
    def reset_shutdown(self):
        """每个测试前后重置关闭管理器"""
        reset_graceful_shutdown()
        yield
        reset_graceful_shutdown()
    
    def test_get_graceful_shutdown_returns_singleton(self):
        """测试get_graceful_shutdown返回单例"""
        manager1 = get_graceful_shutdown()
        manager2 = get_graceful_shutdown()
        
        assert manager1 is manager2
    
    def test_reset_graceful_shutdown_clears_singleton(self):
        """测试reset_graceful_shutdown清除单例"""
        manager1 = get_graceful_shutdown()
        reset_graceful_shutdown()
        manager2 = get_graceful_shutdown()
        
        assert manager1 is not manager2


class TestShutdownResult:
    """测试ShutdownResult数据类"""
    
    def test_default_values(self):
        """测试默认值"""
        result = ShutdownResult(success=True)
        
        assert result.success is True
        assert result.saved_components == []
        assert result.failed_components == []
        assert result.errors == []
        assert result.signal_received is None
    
    def test_with_all_fields(self):
        """测试所有字段"""
        result = ShutdownResult(
            success=False,
            saved_components=["comp1", "comp2"],
            failed_components=["comp3"],
            errors=["Error 1"],
            signal_received="SIGTERM"
        )
        
        assert result.success is False
        assert result.saved_components == ["comp1", "comp2"]
        assert result.failed_components == ["comp3"]
        assert result.errors == ["Error 1"]
        assert result.signal_received == "SIGTERM"
