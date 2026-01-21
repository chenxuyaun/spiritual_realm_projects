"""
REST API接口模块

提供RESTful API接口用于所有核心功能。

需求: 13.1, 13.2, 13.3, 13.4
属性34: API JSON格式
属性35: API错误响应格式
属性36: API认证保护
"""

from mm_orch.api.app import create_app, get_app
from mm_orch.api.auth import APIKeyAuth, get_api_key_header

__all__ = [
    "create_app",
    "get_app",
    "APIKeyAuth",
    "get_api_key_header"
]
