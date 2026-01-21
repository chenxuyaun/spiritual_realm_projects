"""
单元测试：日志系统

测试结构化日志记录器的基本功能和边缘情况。
"""

import json
import io
from mm_orch.logger import (
    StructuredLogger,
    LogLevel,
    get_logger,
    configure_logger
)


class TestStructuredLogger:
    """测试StructuredLogger类"""
    
    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        output = io.StringIO()
        logger = StructuredLogger(name="test", level="INFO", output_stream=output)
        
        assert logger.name == "test"
        assert logger.level == LogLevel.INFO
        assert logger.output_stream == output
    
    def test_log_level_parsing(self):
        """测试日志级别解析"""
        output = io.StringIO()
        
        # 测试各种级别
        for level_str in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger = StructuredLogger(level=level_str, output_stream=output)
            assert logger.level == LogLevel[level_str]
        
        # 测试小写
        logger = StructuredLogger(level="info", output_stream=output)
        assert logger.level == LogLevel.INFO
        
        # 测试无效级别（应该回退到INFO）
        logger = StructuredLogger(level="INVALID", output_stream=output)
        assert logger.level == LogLevel.INFO
    
    def test_json_format_output(self):
        """测试JSON格式输出"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        logger.info("Test message")
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        # 验证必需字段
        assert "timestamp" in log_entry
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
        assert "logger" in log_entry
    
    def test_log_with_context(self):
        """测试带上下文的日志"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        context = {"user_id": "123", "action": "login"}
        logger.info("User action", context=context)
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        assert "context" in log_entry
        assert log_entry["context"]["user_id"] == "123"
        assert log_entry["context"]["action"] == "login"
    
    def test_log_with_extra_fields(self):
        """测试带额外字段的日志"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        logger.info("Test", request_id="req-123", duration=1.5)
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        assert log_entry["request_id"] == "req-123"
        assert log_entry["duration"] == 1.5
    
    def test_log_level_filtering(self):
        """测试日志级别过滤"""
        output = io.StringIO()
        logger = StructuredLogger(level="WARNING", output_stream=output)
        
        # 这些不应该被记录
        logger.debug("Debug message")
        logger.info("Info message")
        
        # 这些应该被记录
        logger.warning("Warning message")
        logger.error("Error message")
        
        log_output = output.getvalue()
        lines = [line for line in log_output.strip().split("\n") if line]
        
        assert len(lines) == 2
        
        # 验证只有WARNING和ERROR被记录
        for line in lines:
            log_entry = json.loads(line)
            assert log_entry["level"] in ["WARNING", "ERROR"]
    
    def test_all_log_levels(self):
        """测试所有日志级别方法"""
        output = io.StringIO()
        logger = StructuredLogger(level="DEBUG", output_stream=output)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        log_output = output.getvalue()
        lines = [line for line in log_output.strip().split("\n") if line]
        
        assert len(lines) == 5
        
        levels = [json.loads(line)["level"] for line in lines]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_set_level(self):
        """测试动态设置日志级别"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        logger.debug("Should not appear")
        logger.info("Should appear")
        
        # 改变级别
        logger.set_level("DEBUG")
        logger.debug("Should appear now")
        
        log_output = output.getvalue()
        lines = [line for line in log_output.strip().split("\n") if line]
        
        assert len(lines) == 2
        messages = [json.loads(line)["message"] for line in lines]
        assert "Should appear" in messages
        assert "Should appear now" in messages
    
    def test_generic_log_method(self):
        """测试通用log方法"""
        output = io.StringIO()
        logger = StructuredLogger(level="DEBUG", output_stream=output)
        
        logger.log("INFO", "Test message")
        logger.log("error", "Error message")  # 测试小写
        
        log_output = output.getvalue()
        lines = [line for line in log_output.strip().split("\n") if line]
        
        assert len(lines) == 2
        levels = [json.loads(line)["level"] for line in lines]
        assert "INFO" in levels
        assert "ERROR" in levels
    
    def test_empty_message(self):
        """测试空消息"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        logger.info("")
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        assert log_entry["message"] == ""
    
    def test_unicode_message(self):
        """测试Unicode消息"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        logger.info("测试中文消息")
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        assert log_entry["message"] == "测试中文消息"
    
    def test_complex_context(self):
        """测试复杂的上下文对象"""
        output = io.StringIO()
        logger = StructuredLogger(level="INFO", output_stream=output)
        
        context = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
            "boolean": True,
            "null": None
        }
        logger.info("Complex context", context=context)
        
        log_output = output.getvalue().strip()
        log_entry = json.loads(log_output)
        
        assert log_entry["context"]["nested"]["key"] == "value"
        assert log_entry["context"]["list"] == [1, 2, 3]
        assert log_entry["context"]["number"] == 42
        assert log_entry["context"]["boolean"] is True
        assert log_entry["context"]["null"] is None


class TestGlobalLogger:
    """测试全局日志记录器函数"""
    
    def test_get_logger_singleton(self):
        """测试get_logger返回单例"""
        # 注意：这个测试可能受到其他测试的影响
        # 在实际使用中，应该在测试之间重置全局状态
        logger1 = get_logger()
        logger2 = get_logger()
        
        # 应该返回相同的实例
        assert logger1 is logger2
    
    def test_configure_logger(self):
        """测试configure_logger"""
        output = io.StringIO()
        logger = configure_logger(level="DEBUG", name="test", output_stream=output)
        
        assert logger.name == "test"
        assert logger.level == LogLevel.DEBUG
        
        logger.debug("Test message")
        log_output = output.getvalue().strip()
        assert len(log_output) > 0
