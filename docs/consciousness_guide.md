# MuAI 意识模块使用指南

## 安装依赖

```bash
pip install httpx
```

## 基础使用

### 1. 获取意识核心

```python
from mm_orch.consciousness import get_consciousness

core = get_consciousness()
```

### 2. 完整交互流程

```python
# Step 1: 感知用户输入
core.perceive("用户的问题", user_id="user123")

# Step 2: 思考准备
context = core.think("用户的问题")

# 检查是否应该回答
if context["should_respond"]:
    # 获取语气指导
    tone = context["tone_instruction"]
    
    # 生成回复...
    response = "AI的回答"
    
    # Step 3: 反思和学习
    core.finalize_interaction(
        query="用户的问题",
        response=response,
        success=True
    )
```

## 常用功能

### 查看AI状态

```python
# 简短摘要
print(core.get_status_summary())

# 详细内省
print(core.introspect())

# 自我介绍
print(core.who_am_i())
```

### 能力评估

```python
from mm_orch.consciousness import get_self_model

model = get_self_model()
result = model.can_i_do("写一个排序算法")

print(f"能做: {result['can_do']}")
print(f"置信度: {result['confidence']:.0%}")
```

### 情感状态

```python
from mm_orch.consciousness import get_emotion_system

emotion = get_emotion_system()

# 处理输入的情感
emotion.process_input("太棒了！")

# 获取当前心境
print(emotion.state.get_mood())  # e.g., "happy"
```

### 多语言检测

```python
from mm_orch.consciousness import get_multilingual

ml = get_multilingual()
info = ml.detect_and_adapt("Hello, how are you?")

print(info["detected_language"])  # "en"
```

### 执行代码

```python
from mm_orch.consciousness import get_tool_registry

tools = get_tool_registry()

# 安全执行Python代码
result = tools.run_code("print(sum(range(10)))")
print(result.output)  # "45"
```

## 性能监控

```python
from mm_orch.consciousness import timed, get_performance_monitor

@timed("my_function")
def process_data():
    # 处理逻辑
    pass

# 查看性能报告
monitor = get_performance_monitor()
print(monitor.get_report())
```

## 状态持久化

```python
from mm_orch.consciousness import save_consciousness, load_consciousness

# 手动保存（强制立即保存）
save_consciousness(force=True)

# 状态自动保存到: .consciousness/state.json
```

## 最佳实践

1. **使用 `get_consciousness()` 获取单例** - 不要手动创建实例
2. **完成交互后调用 `finalize_interaction`** - 确保学习和状态更新
3. **定期检查 `get_status_summary()`** - 监控AI状态
4. **使用 `@timed` 装饰器监控关键函数** - 追踪性能

## 故障排除

### 意识模块未加载
```python
# 检查模块是否可用
try:
    from mm_orch.consciousness import get_consciousness
    core = get_consciousness(auto_load=False)
    print("意识模块正常")
except ImportError as e:
    print(f"导入失败: {e}")
```

### 图片感知不可用
```python
from mm_orch.consciousness import get_multimodal_perception

mm = get_multimodal_perception()
if mm.is_image_available():
    print("图片感知可用")
else:
    print("需要安装LLaVA模型: ollama pull llava")
```
