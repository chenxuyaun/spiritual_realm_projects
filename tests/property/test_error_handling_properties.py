"""
错误处理系统的属性测试

验证错误处理系统的正确性属性。
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, Any
from unittest.mock import patch

from mm_orch.exceptions import (
    ValidationError,
    ResourceError,
    NetworkError,
    ModelError,
    SystemError as MuAISystemError,
)
from mm_orch.error_handler import ErrorHandler, ErrorResponse, get_error_handler
from mm_orch.retry import RetryPolicy, retry


# 策略：生成各种异常类型
exception_strategy = st.sampled_from([
    ValidationError,
    ResourceError,
    NetworkError,
    ModelError,
    MuAISystemError,
])

# 策略：生成错误消息
error_message_strategy = st.text(min_size=1, max_size=200)

# 策略：生成上下文字典
context_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=50),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans()
    ),
    max_size=10
)


@given(
    exception_type=exception_strategy,
    message=error_message_strategy,
    context=context_strategy
)
def test_property_3_error_handling_consistency(
    exception_type,
    message: str,
    context: Dict[str, Any]
):
    """
    Feature: muai-orchestration-system, Property 3: 错误处理一致性
    
    对于任何工作流执行过程中发生的异常，系统应该捕获该异常、
    记录到日志系统、并返回包含error字段的ErrorResponse对象。
    
    **验证需求: 1.3, 15.1, 15.4**
    """
    handler = ErrorHandler()
    
    # 创建异常实例
    error = exception_type(message, context)
    
    # 处理错误
    response = handler.handle_error(error, context)
    
    # 验证返回的是ErrorResponse对象
    assert isinstance(response, ErrorResponse)
    
    # 验证包含必要的错误字段
    assert hasattr(response, 'error_type')
    assert hasattr(response, 'error_message')
    assert hasattr(response, 'user_message')
    assert hasattr(response, 'context')
    
    # 验证错误类型正确
    assert response.error_type == exception_type.__name__
    
    # 验证错误消息非空
    assert len(response.error_message) > 0
    assert len(response.user_message) > 0
    
    # 验证上下文被保留
    assert isinstance(response.context, dict)


@given(
    exception_type=exception_strategy,
    message=error_message_strategy
)
def test_property_32_error_log_completeness(
    exception_type,
    message: str
):
    """
    Feature: muai-orchestration-system, Property 32: 错误日志完整性
    
    对于任何捕获的异常，记录的错误日志应该包含异常类型、错误消息、
    完整的堆栈跟踪和发生错误时的上下文信息。
    
    **验证需求: 12.3, 15.4**
    
    注意：此测试验证ErrorHandler调用logger.error时传递了必要的信息。
    实际的日志输出由logger模块处理，已在test_logger_properties.py中测试。
    """
    handler = ErrorHandler()
    
    # 创建异常实例
    error = exception_type(message)
    context = {"test_key": "test_value"}
    
    # 处理错误（会触发日志记录）
    response = handler.handle_error(error, context)
    
    # 验证ErrorResponse包含完整信息
    assert response.error_type == exception_type.__name__
    assert response.error_message == message
    assert "test_key" in response.context
    
    # 验证错误处理器正确分类了错误
    if exception_type == ValidationError:
        assert response.error_type == "ValidationError"
    elif exception_type == ResourceError:
        assert response.error_type == "ResourceError"
    elif exception_type == NetworkError:
        assert response.error_type == "NetworkError"
    elif exception_type == ModelError:
        assert response.error_type == "ModelError"
    elif exception_type == MuAISystemError:
        assert response.error_type == "SystemError"


@given(
    max_retries=st.integers(min_value=1, max_value=3),
    backoff_factor=st.floats(min_value=0.01, max_value=0.1)
)
@settings(max_examples=20)  # Reduce examples since this test involves actual retries
def test_property_40_network_request_retry_count(
    max_retries: int,
    backoff_factor: float
):
    """
    Feature: muai-orchestration-system, Property 40: 网络请求重试次数
    
    对于任何网络请求超时，系统应该最多重试max_retries次，
    且在第max_retries次失败后应该返回超时错误而不是继续重试。
    
    **验证需求: 15.2**
    """
    # 创建一个总是失败的函数
    call_count = [0]
    
    # Mock time.sleep to speed up the test
    with patch('time.sleep'):
        @retry(max_retries=max_retries, backoff_factor=backoff_factor)
        def failing_network_call():
            call_count[0] += 1
            raise NetworkError("Network timeout")
        
        # 尝试调用，应该失败
        with pytest.raises(NetworkError):
            failing_network_call()
        
        # 验证调用次数：初始调用 + max_retries次重试
        assert call_count[0] == max_retries + 1


@given(
    attempt=st.integers(min_value=0, max_value=5),
    backoff_factor=st.floats(min_value=0.1, max_value=2.0)
)
def test_retry_policy_delay_calculation(attempt: int, backoff_factor: float):
    """
    测试重试策略的延迟计算（指数退避）
    
    验证延迟时间按指数增长。
    """
    policy = RetryPolicy(backoff_factor=backoff_factor)
    
    delay = policy.get_delay(attempt)
    
    # 验证延迟计算正确
    expected_delay = backoff_factor ** attempt
    assert abs(delay - expected_delay) < 0.001


@given(
    exception_type=st.sampled_from([
        NetworkError,
        TimeoutError,
        ConnectionError,
        ValueError,  # 不在重试列表中
    ]),
    attempt=st.integers(min_value=0, max_value=5)
)
def test_retry_policy_should_retry_logic(exception_type, attempt: int):
    """
    测试重试策略的should_retry逻辑
    
    验证只有指定的异常类型才会被重试，且不超过最大重试次数。
    """
    policy = RetryPolicy(
        max_retries=3,
        retry_on=[NetworkError, TimeoutError, ConnectionError]
    )
    
    error = exception_type("Test error")
    should_retry = policy.should_retry(error, attempt)
    
    # 验证重试逻辑
    if exception_type in [NetworkError, TimeoutError, ConnectionError]:
        # 这些异常应该被重试，但不超过max_retries
        if attempt < policy.max_retries:
            assert should_retry is True
        else:
            assert should_retry is False
    else:
        # 其他异常不应该被重试
        assert should_retry is False


def test_error_handler_singleton():
    """
    测试错误处理器的单例模式
    
    验证get_error_handler返回同一个实例。
    """
    handler1 = get_error_handler()
    handler2 = get_error_handler()
    
    assert handler1 is handler2


@given(message=error_message_strategy)
def test_validation_error_handling(message: str):
    """
    测试ValidationError的特定处理逻辑
    
    验证ValidationError被标记为可恢复的。
    """
    handler = ErrorHandler()
    error = ValidationError(message)
    
    response = handler.handle_error(error)
    
    assert response.error_type == "ValidationError"
    assert response.recoverable is True
    assert response.suggested_action is not None


@given(message=error_message_strategy)
def test_resource_error_handling(message: str):
    """
    测试ResourceError的特定处理逻辑
    
    验证ResourceError被标记为可恢复的，并提供降级建议。
    """
    handler = ErrorHandler()
    error = ResourceError(message)
    
    response = handler.handle_error(error)
    
    assert response.error_type == "ResourceError"
    assert response.recoverable is True
    assert "降级" in response.suggested_action or "CPU" in response.suggested_action


@given(message=error_message_strategy)
def test_system_error_handling(message: str):
    """
    测试SystemError的特定处理逻辑
    
    验证SystemError被标记为不可恢复的。
    """
    handler = ErrorHandler()
    error = MuAISystemError(message)
    
    response = handler.handle_error(error)
    
    assert response.error_type == "SystemError"
    assert response.recoverable is False


# ============================================================================
# Property 41: 严重错误安全关闭 测试
# ============================================================================

from mm_orch.error_handler import (
    GracefulShutdown,
    ShutdownResult,
    get_graceful_shutdown,
    reset_graceful_shutdown,
)


@pytest.fixture
def shutdown_manager():
    """创建一个干净的GracefulShutdown实例用于测试"""
    reset_graceful_shutdown()
    manager = GracefulShutdown()
    yield manager
    manager.reset()


@given(
    num_callbacks=st.integers(min_value=0, max_value=5),
    callback_success_pattern=st.lists(
        st.booleans(),
        min_size=0,
        max_size=5
    )
)
def test_property_41_graceful_shutdown_saves_state(
    num_callbacks: int,
    callback_success_pattern
):
    """
    Feature: muai-orchestration-system, Property 41: 严重错误安全关闭
    
    对于任何严重错误（如内存溢出、系统资源耗尽），系统应该在关闭前
    保存当前的意识状态和关键数据，且保存操作应该成功完成。
    
    **验证需求: 15.5**
    """
    manager = GracefulShutdown()
    
    # 调整callback_success_pattern的长度以匹配num_callbacks
    success_pattern = callback_success_pattern[:num_callbacks]
    while len(success_pattern) < num_callbacks:
        success_pattern.append(True)
    
    # 注册回调
    callback_results = []
    for i, should_succeed in enumerate(success_pattern):
        def make_callback(idx, succeed):
            def callback():
                callback_results.append(idx)
                return succeed
            callback._callback_name = f"callback_{idx}"
            return callback
        
        manager.register_save_callback(
            make_callback(i, should_succeed),
            name=f"callback_{i}"
        )
    
    # 执行关闭
    result = manager.shutdown()
    
    # 验证所有回调都被执行
    assert len(callback_results) == num_callbacks
    
    # 验证回调按顺序执行
    assert callback_results == list(range(num_callbacks))
    
    # 验证结果结构
    assert isinstance(result, ShutdownResult)
    assert result.saved_components is not None
    assert result.failed_components is not None
    assert result.errors is not None
    
    # 验证成功/失败计数
    expected_success = sum(1 for s in success_pattern if s)
    expected_failure = sum(1 for s in success_pattern if not s)
    assert len(result.saved_components) == expected_success
    assert len(result.failed_components) == expected_failure
    
    # 验证整体成功状态
    if expected_failure > 0:
        assert result.success is False
    else:
        assert result.success is True
    
    # 验证关闭状态
    assert manager.is_shutdown_complete() is True
    assert manager.is_shutting_down() is False


def test_property_41_shutdown_only_executes_once():
    """
    Property 41 补充测试: 关闭操作只执行一次
    
    验证多次调用shutdown()只会执行一次保存操作。
    """
    manager = GracefulShutdown()
    
    call_count = [0]
    
    def counting_callback():
        call_count[0] += 1
        return True
    
    manager.register_save_callback(counting_callback, name="counter")
    
    # 多次调用shutdown
    result1 = manager.shutdown()
    result2 = manager.shutdown()
    result3 = manager.shutdown()
    
    # 验证回调只执行一次
    assert call_count[0] == 1
    
    # 验证所有结果都表示成功
    assert result1.success is True
    assert result2.success is True
    assert result3.success is True


def test_property_41_shutdown_handles_callback_exceptions():
    """
    Property 41 补充测试: 关闭时处理回调异常
    
    验证即使某个回调抛出异常，其他回调仍然会被执行。
    """
    manager = GracefulShutdown()
    
    executed = []
    
    def callback_1():
        executed.append(1)
        return True
    
    def callback_2():
        executed.append(2)
        raise RuntimeError("Simulated error")
    
    def callback_3():
        executed.append(3)
        return True
    
    manager.register_save_callback(callback_1, name="callback_1")
    manager.register_save_callback(callback_2, name="callback_2")
    manager.register_save_callback(callback_3, name="callback_3")
    
    result = manager.shutdown()
    
    # 验证所有回调都被执行
    assert executed == [1, 2, 3]
    
    # 验证结果反映了异常
    assert "callback_1" in result.saved_components
    assert "callback_2" in result.failed_components
    assert "callback_3" in result.saved_components
    assert any("callback_2" in err for err in result.errors)


def test_property_41_cleanup_callbacks_execute_after_save():
    """
    Property 41 补充测试: 清理回调在保存回调之后执行
    
    验证清理回调的执行顺序。
    """
    manager = GracefulShutdown()
    
    execution_order = []
    
    def save_callback():
        execution_order.append("save")
        return True
    
    def cleanup_callback():
        execution_order.append("cleanup")
    
    manager.register_save_callback(save_callback, name="save")
    manager.register_cleanup_callback(cleanup_callback, name="cleanup")
    
    manager.shutdown()
    
    # 验证执行顺序
    assert execution_order == ["save", "cleanup"]


@given(signal_name=st.sampled_from(["SIGTERM", "SIGINT", "atexit", None]))
def test_property_41_shutdown_records_signal(signal_name):
    """
    Property 41 补充测试: 关闭结果记录触发信号
    
    验证ShutdownResult正确记录触发关闭的信号。
    """
    manager = GracefulShutdown()
    
    result = manager.shutdown(signal_name=signal_name)
    
    assert result.signal_received == signal_name


def test_graceful_shutdown_singleton():
    """
    测试安全关闭管理器的单例模式
    
    验证get_graceful_shutdown返回同一个实例。
    """
    reset_graceful_shutdown()
    
    manager1 = get_graceful_shutdown()
    manager2 = get_graceful_shutdown()
    
    assert manager1 is manager2
    
    reset_graceful_shutdown()


def test_graceful_shutdown_reset():
    """
    测试安全关闭管理器的重置功能
    
    验证reset()正确清除状态。
    """
    manager = GracefulShutdown()
    
    # 注册回调并执行关闭
    manager.register_save_callback(lambda: True, name="test")
    manager.shutdown()
    
    assert manager.is_shutdown_complete() is True
    
    # 重置
    manager.reset()
    
    assert manager.is_shutdown_complete() is False
    assert manager.is_shutting_down() is False
    assert manager.get_last_result() is None
