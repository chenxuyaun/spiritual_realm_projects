"""
API认证模块

实现API密钥认证机制保护敏感操作。

需求: 13.4 - 实现API认证机制保护敏感操作
属性36: API认证保护
"""

import os
import secrets
import hashlib
from typing import Optional, Set
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from mm_orch.logger import get_logger


logger = get_logger(__name__)


# API密钥请求头名称
API_KEY_HEADER_NAME = "X-API-Key"

# API密钥请求头安全方案
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def get_api_key_header():
    """获取API密钥请求头安全方案"""
    return api_key_header


class APIKeyAuth:
    """
    API密钥认证管理器
    
    属性36: API认证保护
    对于任何需要认证的API端点，未提供有效认证凭据的请求应该被拒绝，
    返回401或403状态码。
    
    支持:
    - 从环境变量加载API密钥
    - 从配置文件加载API密钥
    - 动态添加/移除API密钥
    - API密钥哈希存储（安全）
    """
    
    def __init__(
        self,
        api_keys: Optional[Set[str]] = None,
        env_var_name: str = "MUAI_API_KEYS",
        enabled: bool = True
    ):
        """
        初始化API密钥认证管理器
        
        Args:
            api_keys: 初始API密钥集合
            env_var_name: 环境变量名称（逗号分隔的密钥列表）
            enabled: 是否启用认证
        """
        self._enabled = enabled
        self._api_key_hashes: Set[str] = set()
        
        # 从参数加载密钥
        if api_keys:
            for key in api_keys:
                self.add_api_key(key)
        
        # 从环境变量加载密钥
        env_keys = os.environ.get(env_var_name, "")
        if env_keys:
            for key in env_keys.split(","):
                key = key.strip()
                if key:
                    self.add_api_key(key)
        
        logger.info(
            "APIKeyAuth initialized",
            enabled=self._enabled,
            num_keys=len(self._api_key_hashes)
        )
    
    @property
    def enabled(self) -> bool:
        """认证是否启用"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        """设置认证启用状态"""
        self._enabled = value
        logger.info("APIKeyAuth enabled status changed", enabled=value)
    
    def _hash_key(self, api_key: str) -> str:
        """
        对API密钥进行哈希
        
        Args:
            api_key: 原始API密钥
            
        Returns:
            哈希后的密钥
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def add_api_key(self, api_key: str) -> None:
        """
        添加API密钥
        
        Args:
            api_key: API密钥
        """
        if not api_key or len(api_key) < 16:
            logger.warning("Attempted to add invalid API key (too short)")
            return
        
        key_hash = self._hash_key(api_key)
        self._api_key_hashes.add(key_hash)
        logger.debug("API key added", key_hash_prefix=key_hash[:8])
    
    def remove_api_key(self, api_key: str) -> bool:
        """
        移除API密钥
        
        Args:
            api_key: API密钥
            
        Returns:
            是否成功移除
        """
        key_hash = self._hash_key(api_key)
        if key_hash in self._api_key_hashes:
            self._api_key_hashes.remove(key_hash)
            logger.debug("API key removed", key_hash_prefix=key_hash[:8])
            return True
        return False
    
    def validate_api_key(self, api_key: Optional[str]) -> bool:
        """
        验证API密钥
        
        Args:
            api_key: API密钥
            
        Returns:
            密钥是否有效
        """
        if not self._enabled:
            return True
        
        if not api_key:
            return False
        
        key_hash = self._hash_key(api_key)
        return key_hash in self._api_key_hashes
    
    def generate_api_key(self, length: int = 32) -> str:
        """
        生成新的API密钥
        
        Args:
            length: 密钥长度（字节数）
            
        Returns:
            新生成的API密钥
        """
        api_key = secrets.token_urlsafe(length)
        self.add_api_key(api_key)
        return api_key
    
    def get_num_keys(self) -> int:
        """获取已注册的API密钥数量"""
        return len(self._api_key_hashes)


# 全局认证管理器实例
_auth_manager: Optional[APIKeyAuth] = None


def get_auth_manager() -> APIKeyAuth:
    """获取全局认证管理器实例"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = APIKeyAuth()
    return _auth_manager


def set_auth_manager(manager: APIKeyAuth) -> None:
    """设置全局认证管理器实例"""
    global _auth_manager
    _auth_manager = manager


def reset_auth_manager() -> None:
    """重置全局认证管理器"""
    global _auth_manager
    _auth_manager = None


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    验证API密钥的依赖函数
    
    用于FastAPI的Depends注入
    
    Args:
        api_key: 从请求头获取的API密钥
        
    Returns:
        验证通过的API密钥
        
    Raises:
        HTTPException: 如果认证失败
    """
    auth_manager = get_auth_manager()
    
    # 如果认证未启用，直接通过
    if not auth_manager.enabled:
        return api_key or ""
    
    # 检查是否提供了API密钥
    if not api_key:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_API_KEY",
                "message": "API key is required",
                "header": API_KEY_HEADER_NAME
            }
        )
    
    # 验证API密钥
    if not auth_manager.validate_api_key(api_key):
        logger.warning(
            "Invalid API key attempt",
            key_prefix=api_key[:8] if len(api_key) >= 8 else "***"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "INVALID_API_KEY",
                "message": "Invalid API key"
            }
        )
    
    return api_key


async def optional_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[str]:
    """
    可选的API密钥验证
    
    用于不强制要求认证但可以接受认证的端点
    
    Args:
        api_key: 从请求头获取的API密钥
        
    Returns:
        API密钥（如果提供且有效）或None
    """
    auth_manager = get_auth_manager()
    
    if not api_key:
        return None
    
    if auth_manager.validate_api_key(api_key):
        return api_key
    
    return None
