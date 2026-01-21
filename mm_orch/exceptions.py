"""
错误类型和异常层次结构

定义系统中使用的所有自定义异常类型，用于错误分类和处理。
"""

from typing import Any, Dict, Optional


class MuAIError(Exception):
    """MuAI系统的基础异常类"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ValidationError(MuAIError):
    """输入验证错误：用户输入格式错误、参数缺失、类型不匹配"""
    pass


class ResourceError(MuAIError):
    """资源错误：模型加载失败、GPU内存不足、磁盘空间不足"""
    pass


class NetworkError(MuAIError):
    """网络错误：搜索超时、网页抓取失败、API调用失败"""
    pass


class ModelError(MuAIError):
    """模型推理错误：模型输出格式错误、推理超时、模型崩溃"""
    pass


class SystemError(MuAIError):
    """系统错误：未捕获的异常、配置错误、依赖缺失"""
    pass


class WorkflowError(MuAIError):
    """工作流执行错误：工作流步骤失败、参数验证失败"""
    pass


class ConsciousnessError(MuAIError):
    """意识模块错误：意识状态更新失败、模块初始化失败"""
    pass


class StorageError(MuAIError):
    """存储错误：数据持久化失败、文件读写错误"""
    pass
