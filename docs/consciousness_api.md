# MuAI Consciousness Module API 文档

## 快速开始

```python
from mm_orch.consciousness import get_consciousness
core = get_consciousness()
print(core.get_status_summary())
```

---

## 核心模块

| 模块 | 获取函数 | 描述 |
|------|----------|------|
| ConsciousnessCore | `get_consciousness()` | 意识核心 |
| SelfModel | `get_self_model()` | 自我模型 |
| WorldModel | `get_world_model()` | 世界模型 |
| Metacognition | `get_metacognition()` | 元认知 |
| MotivationSystem | `get_motivation_system()` | 动机系统 |
| EmotionSystem | `get_emotion_system()` | 情感系统 |
| DevelopmentSystem | `get_development_system()` | 发展阶段 |

---

## 功能模块

| 模块 | 获取函数 | 描述 |
|------|----------|------|
| MultiModalPerception | `get_multimodal_perception()` | 多模态感知 |
| ToolRegistry | `get_tool_registry()` | 工具执行 |
| MultilingualSystem | `get_multilingual()` | 多语言支持 |
| PerformanceMonitor | `get_performance_monitor()` | 性能监控 |
| AdvancedCognition | `get_advanced_cognition()` | 高级认知 |
| UserExperience | `get_user_experience()` | 用户体验 |

---

## 高级模块

| 模块 | 获取函数 | 描述 |
|------|----------|------|
| MultiAgentSystem | `get_multi_agent_system()` | 多智能体协作 |
| KnowledgeSystem | `get_knowledge_system()` | 知识图谱 |
| StreamingSystem | `get_streaming_system()` | 流式推理 |
| ContinuousLearning | `get_continuous_learning()` | 持续学习 |
| SafetyAlignment | `get_safety_system()` | 安全对齐 |
| DeepReflection | `get_deep_reflection()` | 深度反思 |
| ContextManager | `get_context_manager()` | 上下文感知 |
| PersonalitySystem | `get_personality_system()` | 人格系统 |

---

## 常用示例

### 意识核心

```python
from mm_orch.consciousness import get_consciousness

core = get_consciousness()
core.perceive("用户输入", user_id="user1")
ctx = core.think("问题")
core.finalize_interaction("问题", "回答", success=True)
```

### 多智能体

```python
from mm_orch.consciousness import get_multi_agent_system, AgentRole

mas = get_multi_agent_system()
agent_id = mas.create_agent("Worker", AgentRole.WORKER, ["coding"])
mas.delegate("任务", capability="coding")
```

### 知识图谱

```python
from mm_orch.consciousness import get_knowledge_system

kg = get_knowledge_system()
kg.learn("Python", "是", "编程语言")
print(kg.query("Python"))
```

### 持续学习

```python
from mm_orch.consciousness import get_continuous_learning

cl = get_continuous_learning()
cl.record_interaction("问题", "回答", success=True)
```

### 安全对齐

```python
from mm_orch.consciousness import get_safety_system

safety = get_safety_system()
can_respond, reason = safety.should_respond("请求")
```

### 上下文感知

```python
from mm_orch.consciousness import get_context_manager

ctx = get_context_manager()
result = ctx.process_input("用户输入")
# result['situation'].task_type -> 'question'/'request'/...
```

### 人格系统

```python
from mm_orch.consciousness import get_personality_system

personality = get_personality_system()
personality.prepare_for_interaction({"is_formal": True})
modifiers = personality.get_response_modifiers()
```

---

## 持久化

```python
from mm_orch.consciousness import save_consciousness, load_consciousness

save_consciousness(force=True)  # 手动保存
# 自动加载: get_consciousness(auto_load=True)
```

保存位置: `.consciousness/state.json`
