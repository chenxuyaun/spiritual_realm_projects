"""
基于属性的测试：日志系统

验证日志系统的正确性属性。
"""

import json
import io
from hypothesis import given, strategies as st
from mm_orch.logger import StructuredLogger, LogLevel


# 策略定义
log_level_strategy = st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
message_strategy = st.text(min_size=0, max_size=1000)
context_strategy = st.none() | st.dictionaries(
    st.text(min_size=1, max_size=50),
    st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans(),
        st.none()
    ),
    max_size=10
)


@given(
    level=log_level_strategy,
    message=message_strategy,
    context=context_strategy
)
def test_property_30_log_structured_format(level, message, context):
    """
    Feature: muai-orchestration-system, Property 30: 日志结构化格式
    
    对于任何系统日志记录，日志输出应该是结构化的（如JSON格式），
    包含timestamp、level、message、context等标准字段，
    且应该可以被解析为字典对象。
    
    **验证需求: 12.1, 9.5, 11.5**
    """
    output = io.StringIO()
    logger = StructuredLogger(level="DEBUG", output_stream=output)
    
    # 根据级别调用相应的日志方法
    log_method = getattr(logger, level.lower())
    log_method(message, context=context)
    
    log_output = output.getvalue().strip()
    
    # 属性1: 输出应该是有效的JSON
    try:
        log_entry = json.loads(log_output)
    except json.JSONDecodeError:
        assert False, f"Log output is not valid JSON: {log_output}"
    
    # 属性2: 必须包含标准字段
    assert "timestamp" in log_entry, "Missing timestamp field"
    assert "level" in log_entry, "Missing level field"
    assert "message" in log_entry, "Missing message field"
    assert "logger" in log_entry, "Missing logger field"
    
    # 属性3: level字段应该匹配输入
    assert log_entry["level"] == level.upper()
    
    # 属性4: message字段应该匹配输入
    assert log_entry["message"] == message
    
    # 属性5: 如果提供了context，应该包含context字段
    if context is not None:
        assert "context" in log_entry, "Missing context field when context provided"
        assert log_entry["context"] == context
    
    # 属性6: timestamp应该是ISO格式字符串
    assert isinstance(log_entry["timestamp"], str)
    assert "T" in log_entry["timestamp"]  # ISO格式包含T分隔符


@given(
    configured_level=log_level_strategy,
    log_level=log_level_strategy,
    message=message_strategy
)
def test_property_31_log_level_filtering(configured_level, log_level, message):
    """
    Feature: muai-orchestration-system, Property 31: 日志级别过滤
    
    对于任何日志级别配置，当设置为特定级别（如WARNING）时，
    低于该级别的日志（如DEBUG、INFO）不应该被输出，
    而高于或等于该级别的日志应该被输出。
    
    **验证需求: 12.2**
    """
    output = io.StringIO()
    logger = StructuredLogger(level=configured_level, output_stream=output)
    
    # 日志级别数值映射
    level_values = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    
    # 记录日志
    log_method = getattr(logger, log_level.lower())
    log_method(message)
    
    log_output = output.getvalue().strip()
    
    configured_value = level_values[configured_level.upper()]
    log_value = level_values[log_level.upper()]
    
    # 属性: 只有当日志级别 >= 配置级别时才应该输出
    if log_value >= configured_value:
        # 应该有输出
        assert len(log_output) > 0, \
            f"Expected log output for {log_level} when configured level is {configured_level}"
        
        # 验证输出的级别正确
        log_entry = json.loads(log_output)
        assert log_entry["level"] == log_level.upper()
    else:
        # 不应该有输出
        assert len(log_output) == 0, \
            f"Unexpected log output for {log_level} when configured level is {configured_level}"


@given(
    initial_level=log_level_strategy,
    new_level=log_level_strategy,
    message=message_strategy
)
def test_property_dynamic_level_change(initial_level, new_level, message):
    """
    Feature: muai-orchestration-system, Property: 动态日志级别变更
    
    对于任何日志记录器，当动态改变日志级别后，
    新的日志级别过滤规则应该立即生效。
    """
    output = io.StringIO()
    logger = StructuredLogger(level=initial_level, output_stream=output)
    
    # 改变日志级别
    logger.set_level(new_level)
    
    # 验证新级别生效
    assert logger.level == LogLevel[new_level.upper()]
    
    # 记录一条INFO日志
    logger.info(message)
    
    log_output = output.getvalue().strip()
    
    # 日志级别数值
    level_values = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50
    }
    
    new_level_value = level_values[new_level.upper()]
    info_level_value = level_values["INFO"]
    
    # 属性: 根据新级别判断是否应该输出
    if info_level_value >= new_level_value:
        assert len(log_output) > 0
    else:
        assert len(log_output) == 0


@given(
    level=log_level_strategy,
    messages=st.lists(message_strategy, min_size=1, max_size=10)
)
def test_property_multiple_log_entries(level, messages):
    """
    Feature: muai-orchestration-system, Property: 多条日志记录独立性
    
    对于任何日志记录器，记录多条日志时，
    每条日志应该是独立的JSON对象，可以单独解析。
    """
    output = io.StringIO()
    logger = StructuredLogger(level="DEBUG", output_stream=output)
    
    log_method = getattr(logger, level.lower())
    
    # 记录多条日志
    for message in messages:
        log_method(message)
    
    log_output = output.getvalue()
    lines = [line for line in log_output.strip().split("\n") if line]
    
    # 属性1: 日志条数应该等于消息数
    assert len(lines) == len(messages)
    
    # 属性2: 每条日志都应该是有效的JSON
    for i, line in enumerate(lines):
        try:
            log_entry = json.loads(line)
        except json.JSONDecodeError:
            assert False, f"Line {i} is not valid JSON: {line}"
        
        # 属性3: 每条日志的message应该对应输入
        assert log_entry["message"] == messages[i]
        assert log_entry["level"] == level.upper()


@given(
    level=log_level_strategy,
    message=message_strategy,
    extra_fields=st.dictionaries(
        st.text(min_size=1, max_size=20).filter(
            lambda x: x not in ["timestamp", "level", "message", "logger", "context"]
        ),
        st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False)),
        max_size=5
    )
)
def test_property_extra_fields_preserved(level, message, extra_fields):
    """
    Feature: muai-orchestration-system, Property: 额外字段保留
    
    对于任何日志记录，提供的额外字段应该被保留在输出中，
    且不应该覆盖标准字段。
    """
    output = io.StringIO()
    logger = StructuredLogger(level="DEBUG", output_stream=output)
    
    log_method = getattr(logger, level.lower())
    log_method(message, **extra_fields)
    
    log_output = output.getvalue().strip()
    log_entry = json.loads(log_output)
    
    # 属性1: 标准字段应该存在
    assert "timestamp" in log_entry
    assert "level" in log_entry
    assert "message" in log_entry
    assert "logger" in log_entry
    
    # 属性2: 额外字段应该被保留
    for key, value in extra_fields.items():
        assert key in log_entry, f"Extra field {key} not found in log entry"
        assert log_entry[key] == value, f"Extra field {key} value mismatch"
    
    # 属性3: 标准字段不应该被覆盖
    assert log_entry["level"] == level.upper()
    assert log_entry["message"] == message
