# 真实模型集成指南

本指南介绍如何在MuAI编排系统中使用真实的LLM模型进行推理。

## 概述

MuAI系统支持两种模型运行模式：
1. **Mock模式**（默认）：使用模拟响应，适合开发和测试
2. **真实模型模式**：使用实际的LLM模型进行推理

## 支持的模型

### Qwen系列
- `qwen-7b-chat`: Qwen-7B-Chat 全精度
- `qwen-7b-chat-int4`: Qwen-7B-Chat GPTQ 4-bit量化

### GPT-2系列
- `gpt2`: GPT-2 基础版
- `gpt2-medium`: GPT-2 中等版本

## 快速开始

### 1. 查看可用模型

```bash
# 列出所有配置的模型
python -m mm_orch.main --model-info

# 查看特定模型详情
python -m mm_orch.main --model-info gpt2
```

### 2. 使用真实模型

```bash
# 使用GPT-2进行查询
python -m mm_orch.main --real-models --model gpt2 "你好，请介绍一下你自己"

# 使用Qwen-Chat（需要GPU）
python -m mm_orch.main --real-models --model qwen-7b-chat "什么是机器学习？"
```

### 3. 交互式对话

```bash
python -m mm_orch.main --real-models --model gpt2 --mode chat
```

## 模型配置

模型配置文件位于 `config/models.yaml`：

```yaml
models:
  gpt2:
    model_name: "gpt2"
    model_type: "gpt2"
    device: "auto"
    dtype: "fp32"
    quantization: null
    trust_remote_code: false
    flash_attention: false
    max_context_length: 1024

  qwen-7b-chat:
    model_name: "Qwen/Qwen-7B-Chat"
    model_type: "qwen-chat"
    device: "auto"
    dtype: "bf16"
    quantization: null
    trust_remote_code: true
    flash_attention: true
    max_context_length: 8192
```

### 配置项说明

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `model_name` | HuggingFace模型名称 | 任意有效的模型ID |
| `model_type` | 模型类型 | `qwen-chat`, `gpt2`, `chatml`, `llama` |
| `device` | 运行设备 | `auto`, `cuda`, `cpu` |
| `dtype` | 数据类型 | `auto`, `fp32`, `fp16`, `bf16` |
| `quantization` | 量化方式 | `null`, `8bit`, `4bit`, `gptq` |
| `trust_remote_code` | 信任远程代码 | `true`, `false` |
| `flash_attention` | 启用FlashAttention | `true`, `false` |
| `max_context_length` | 最大上下文长度 | 整数 |

## 量化支持

### 8-bit量化

使用bitsandbytes进行8-bit量化，可显著减少内存占用：

```yaml
models:
  gpt2-8bit:
    model_name: "gpt2-medium"
    quantization: "8bit"
    device: "cuda"
```

### 4-bit量化

使用bitsandbytes NF4量化：

```yaml
models:
  qwen-4bit:
    model_name: "Qwen/Qwen-7B-Chat"
    quantization: "4bit"
    device: "cuda"
```

### GPTQ量化

使用预量化的GPTQ模型：

```yaml
models:
  qwen-gptq:
    model_name: "Qwen/Qwen-7B-Chat-Int4"
    quantization: "gptq"
    device: "cuda"
```

## Python API使用

### 基本使用

```python
from mm_orch.runtime.real_model_manager import RealModelManager
from mm_orch.runtime.model_loader import ModelConfig
from mm_orch.runtime.inference_engine import InferenceEngine, GenerationConfig

# 创建模型管理器
manager = RealModelManager(max_cached_models=3)

# 配置模型
config = ModelConfig(
    model_name="gpt2",
    model_type="gpt2",
    device="auto",
    dtype="fp32"
)

# 加载模型
loaded_model = manager.load_model(config)

# 创建推理引擎
engine = InferenceEngine(
    model=loaded_model.model,
    tokenizer=loaded_model.tokenizer,
    device=loaded_model.device
)

# 生成文本
gen_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)

result = engine.generate("Hello, how are you?", config=gen_config)
print(result.text)
```

### 流式生成

```python
# 流式生成
for chunk in engine.generate_stream("Tell me a story", config=gen_config):
    print(chunk, end="", flush=True)
```

### 批量生成

```python
# 批量生成
prompts = ["Hello!", "What is AI?", "Tell me a joke"]
results = engine.batch_generate(prompts, config=gen_config)

for result in results:
    print(f"Output: {result.text}")
    print(f"Tokens/s: {result.tokens_per_second:.1f}")
```

### 多轮对话

```python
from mm_orch.runtime.conversation import ConversationManager

# 创建对话管理器
conv = ConversationManager(
    model_type="qwen-chat",
    max_history_tokens=4096,
    system_prompt="你是一个有帮助的AI助手。"
)

# 添加对话
conv.add_user_input("你好！")
prompt = conv.build_prompt()

# 生成回复
result = engine.generate(prompt, config=gen_config)
conv.add_assistant_response(result.text)

# 继续对话
conv.add_user_input("你能做什么？")
prompt = conv.build_prompt()
result = engine.generate(prompt, config=gen_config)
```

## 工作流集成

### SearchQA工作流

```python
from mm_orch.workflows.search_qa import SearchQAWorkflow

workflow = SearchQAWorkflow(
    inference_engine=engine,
    use_real_models=True,
    language="zh"
)

result = workflow.execute({"query": "什么是深度学习？"})
print(result.result)
```

### LessonPack工作流

```python
from mm_orch.workflows.lesson_pack import LessonPackWorkflow

workflow = LessonPackWorkflow(
    inference_engine=engine,
    use_real_models=True
)

result = workflow.execute({
    "topic": "Python基础",
    "difficulty": "beginner",
    "num_exercises": 3,
    "language": "zh"
})

print(result.result["plan"])
print(result.result["explanation"])
```

### ChatGenerate工作流

```python
from mm_orch.workflows.chat_generate import ChatGenerateWorkflow
from mm_orch.runtime.conversation import ConversationManager

conv_manager = ConversationManager(model_type="gpt2")

workflow = ChatGenerateWorkflow(
    inference_engine=engine,
    conversation_manager=conv_manager,
    use_real_models=True,
    model_type="gpt2"
)

result = workflow.execute({
    "message": "你好！",
    "session_id": "my-session"
})

print(result.result)
```

## 内存管理

### LRU缓存

系统使用LRU（最近最少使用）策略管理模型缓存：

```python
manager = RealModelManager(
    max_cached_models=3,  # 最多缓存3个模型
    memory_threshold=0.8  # GPU内存使用超过80%时触发卸载
)
```

### 手动卸载模型

```python
# 卸载特定模型
manager.unload_model("gpt2")

# 清空所有缓存
manager.clear_cache()
```

### 设备切换

```python
# 将模型从GPU移到CPU
manager.switch_device("gpt2", "cpu")

# 将模型从CPU移到GPU
manager.switch_device("gpt2", "cuda")
```

## 性能优化

### FlashAttention

对于支持的模型，启用FlashAttention可显著提升性能：

```yaml
models:
  qwen-7b-chat:
    flash_attention: true
```

### 批处理

使用批处理可提高吞吐量：

```python
# 批量处理多个请求
results = engine.batch_generate(prompts, config=gen_config)
```

### 量化

使用量化可减少内存占用并可能提升速度：

| 量化方式 | 内存节省 | 速度影响 | 质量影响 |
|----------|----------|----------|----------|
| 8-bit | ~50% | 略慢 | 轻微 |
| 4-bit | ~75% | 略慢 | 中等 |
| GPTQ | ~75% | 相当 | 轻微 |

## 故障排除

### GPU内存不足

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

解决方案：
1. 减少 `max_cached_models`
2. 启用量化
3. 使用更小的模型
4. 切换到CPU

### 模型加载失败

```
ModelLoadError: Failed to load model
```

解决方案：
1. 检查模型名称是否正确
2. 确保网络连接正常
3. 检查磁盘空间
4. 尝试手动下载模型

### FlashAttention不可用

```
FlashAttention not available, falling back to standard attention
```

这是正常的警告，系统会自动回退到标准注意力机制。如需启用FlashAttention：
1. 确保使用NVIDIA GPU
2. 安装flash-attn包
3. 使用支持的模型

## 参考资料

- [HuggingFace Transformers文档](https://huggingface.co/docs/transformers)
- [bitsandbytes量化指南](https://github.com/TimDettmers/bitsandbytes)
- [FlashAttention项目](https://github.com/Dao-AILab/flash-attention)
