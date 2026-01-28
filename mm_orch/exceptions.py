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


class OrchestrationError(MuAIError):
    """编排错误：工作流编排失败、路由失败、协调失败"""

    pass


# Real Model Integration Exceptions (Phase 1)


class ModelLoadError(ResourceError):
    """模型加载错误：HuggingFace模型加载失败、权重下载失败、设备分配失败"""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.model_name = model_name
        if model_name and context is None:
            self.context["model_name"] = model_name


class InferenceError(ModelError):
    """推理错误：模型推理失败、生成超时、输出格式错误"""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        input_length: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.model_name = model_name
        self.input_length = input_length
        if model_name:
            self.context["model_name"] = model_name
        if input_length:
            self.context["input_length"] = input_length


class OutOfMemoryError(ResourceError):
    """内存不足错误：GPU/CPU内存不足、无法分配张量"""

    def __init__(
        self,
        message: str,
        device: Optional[str] = None,
        required_memory: Optional[int] = None,
        available_memory: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.device = device
        self.required_memory = required_memory
        self.available_memory = available_memory
        if device:
            self.context["device"] = device
        if required_memory:
            self.context["required_memory"] = required_memory
        if available_memory:
            self.context["available_memory"] = available_memory


class QuantizationError(ModelError):
    """量化错误：量化配置错误、量化加载失败、不支持的量化类型"""

    def __init__(
        self,
        message: str,
        quantization_type: Optional[str] = None,
        model_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.quantization_type = quantization_type
        self.model_name = model_name
        if quantization_type:
            self.context["quantization_type"] = quantization_type
        if model_name:
            self.context["model_name"] = model_name
