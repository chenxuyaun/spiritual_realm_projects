"""
结构化日志系统

提供JSON格式的结构化日志记录，支持可配置的日志级别和上下文信息。
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    结构化日志记录器

    特性:
    - JSON格式输出
    - 可配置的日志级别
    - 支持上下文信息
    - 标准字段: timestamp, level, message, context
    """

    def __init__(self, name: str = "mm_orch", level: str = "INFO", output_stream=None):
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            output_stream: 输出流，默认为sys.stdout
        """
        self.name = name
        self.level = self._parse_level(level)
        self.output_stream = output_stream or sys.stdout

        # 日志级别数值映射
        self._level_values = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50,
        }

    def _parse_level(self, level: str) -> LogLevel:
        """解析日志级别字符串"""
        try:
            return LogLevel[level.upper()]
        except KeyError:
            return LogLevel.INFO

    def _should_log(self, level: LogLevel) -> bool:
        """判断是否应该记录该级别的日志"""
        return self._level_values[level] >= self._level_values[self.level]

    def _format_log(
        self, level: LogLevel, message: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        格式化日志为JSON字符串

        Args:
            level: 日志级别
            message: 日志消息
            context: 上下文信息字典
            **kwargs: 额外的字段

        Returns:
            JSON格式的日志字符串
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "logger": self.name,
            "message": message,
        }

        # 添加上下文信息
        if context is not None:
            log_entry["context"] = context

        # 添加额外字段
        for key, value in kwargs.items():
            if key not in log_entry:
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)

    def _write_log(self, log_str: str) -> None:
        """写入日志到输出流"""
        self.output_stream.write(log_str + "\n")
        self.output_stream.flush()

    def log(
        self, level: str, message: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        通用日志记录方法

        Args:
            level: 日志级别字符串
            message: 日志消息
            context: 上下文信息
            **kwargs: 额外字段
        """
        log_level = self._parse_level(level)
        if self._should_log(log_level):
            log_str = self._format_log(log_level, message, context, **kwargs)
            self._write_log(log_str)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """记录DEBUG级别日志"""
        if self._should_log(LogLevel.DEBUG):
            log_str = self._format_log(LogLevel.DEBUG, message, context, **kwargs)
            self._write_log(log_str)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """记录INFO级别日志"""
        if self._should_log(LogLevel.INFO):
            log_str = self._format_log(LogLevel.INFO, message, context, **kwargs)
            self._write_log(log_str)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """记录WARNING级别日志"""
        if self._should_log(LogLevel.WARNING):
            log_str = self._format_log(LogLevel.WARNING, message, context, **kwargs)
            self._write_log(log_str)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """记录ERROR级别日志"""
        if self._should_log(LogLevel.ERROR):
            log_str = self._format_log(LogLevel.ERROR, message, context, **kwargs)
            self._write_log(log_str)

    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """记录CRITICAL级别日志"""
        if self._should_log(LogLevel.CRITICAL):
            log_str = self._format_log(LogLevel.CRITICAL, message, context, **kwargs)
            self._write_log(log_str)

    def set_level(self, level: str) -> None:
        """
        设置日志级别

        Args:
            level: 日志级别字符串
        """
        self.level = self._parse_level(level)


# 全局日志记录器实例
_global_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "mm_orch", level: Optional[str] = None, output_stream=None
) -> StructuredLogger:
    """
    获取日志记录器实例

    Args:
        name: 日志记录器名称
        level: 日志级别（可选）
        output_stream: 输出流（可选）

    Returns:
        StructuredLogger实例
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = StructuredLogger(
            name=name, level=level or "INFO", output_stream=output_stream
        )
    elif level is not None:
        _global_logger.set_level(level)

    return _global_logger


def configure_logger(
    level: str = "INFO", name: str = "mm_orch", output_stream=None
) -> StructuredLogger:
    """
    配置全局日志记录器

    Args:
        level: 日志级别
        name: 日志记录器名称
        output_stream: 输出流

    Returns:
        配置后的StructuredLogger实例
    """
    global _global_logger
    _global_logger = StructuredLogger(name=name, level=level, output_stream=output_stream)
    return _global_logger
