"""
重试机制

提供可配置的重试策略，支持指数退避和自定义重试条件。
"""

import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Type, TypeVar

from mm_orch.exceptions import NetworkError
from mm_orch.logger import get_logger


logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """重试策略配置"""
    max_retries: int = 3
    backoff_factor: float = 2.0  # 指数退避因子
    retry_on: List[Type[Exception]] = None  # 需要重试的异常类型
    
    def __post_init__(self):
        if self.retry_on is None:
            self.retry_on = [NetworkError, TimeoutError, ConnectionError]
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        判断是否应该重试
        
        Args:
            error: 捕获的异常
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            bool: 是否应该重试
        """
        return (
            any(isinstance(error, exc_type) for exc_type in self.retry_on) and
            attempt < self.max_retries
        )
    
    def get_delay(self, attempt: int) -> float:
        """
        计算重试延迟（指数退避）
        
        Args:
            attempt: 当前尝试次数（从0开始）
            
        Returns:
            float: 延迟秒数
        """
        return self.backoff_factor ** attempt


def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on: Optional[List[Type[Exception]]] = None
) -> Callable:
    """
    重试装饰器
    
    使用示例:
        @retry(max_retries=3, backoff_factor=2.0)
        def fetch_url(url: str) -> str:
            # 可能失败的网络请求
            ...
    
    Args:
        max_retries: 最大重试次数
        backoff_factor: 指数退避因子
        retry_on: 需要重试的异常类型列表
        
    Returns:
        装饰后的函数
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        retry_on=retry_on
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_error = None
            
            while attempt <= policy.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if not policy.should_retry(e, attempt):
                        # 不应该重试，直接抛出异常
                        logger.warning(
                            f"Error not retryable or max retries reached",
                            function=func.__name__,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            attempt=attempt
                        )
                        raise
                    
                    # 计算延迟并等待
                    delay = policy.get_delay(attempt)
                    logger.info(
                        f"Retrying after error",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        attempt=attempt + 1,
                        max_retries=policy.max_retries,
                        delay=delay
                    )
                    time.sleep(delay)
                    attempt += 1
            
            # 达到最大重试次数，抛出最后一个错误
            if last_error:
                raise last_error
            
        return wrapper
    return decorator


def retry_with_policy(policy: RetryPolicy) -> Callable:
    """
    使用自定义策略的重试装饰器
    
    使用示例:
        custom_policy = RetryPolicy(max_retries=5, backoff_factor=1.5)
        
        @retry_with_policy(custom_policy)
        def my_function():
            ...
    
    Args:
        policy: 重试策略对象
        
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            last_error = None
            
            while attempt <= policy.max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if not policy.should_retry(e, attempt):
                        logger.warning(
                            f"Error not retryable or max retries reached",
                            function=func.__name__,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            attempt=attempt
                        )
                        raise
                    
                    delay = policy.get_delay(attempt)
                    logger.info(
                        f"Retrying after error",
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        attempt=attempt + 1,
                        max_retries=policy.max_retries,
                        delay=delay
                    )
                    time.sleep(delay)
                    attempt += 1
            
            if last_error:
                raise last_error
            
        return wrapper
    return decorator
