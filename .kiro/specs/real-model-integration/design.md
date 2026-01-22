# 真实模型集成与验证 - 设计文档

## 1. 架构概述

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   CLI       │  │   API       │  │   Workflows             │  │
│  │   main.py   │  │   routes.py │  │   search_qa/lesson_pack │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Management Layer                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    RealModelManager                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   ││
│  │  │ ModelLoader  │  │ ModelCache   │  │ QuantizationMgr  │   ││
│  │  │ - HF Loader  │  │ - LRU Cache  │  │ - 8bit/4bit      │   ││
│  │  │ - Device Mgr │  │ - Memory Mon │  │ - GPTQ Support   │   ││
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Inference Engine Layer                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    InferenceEngine                           ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   ││
│  │  │ TextGen      │  │ BatchInfer   │  │ FlashAttn        │   ││
│  │  │ - generate() │  │ - batch_gen  │  │ - auto_detect    │   ││
│  │  │ - stream()   │  │ - queue_mgr  │  │ - fallback       │   ││
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Benchmark & Validation Layer                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ LatencyBenchmark │  │ MemoryBenchmark  │  │ E2EValidator   │ │
│  │ - TTFT           │  │ - peak_memory    │  │ - SearchQA     │ │
│  │ - tokens/s       │  │ - kv_cache       │  │ - LessonPack   │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 组件职责

| 组件 | 职责 | 依赖 |
|------|------|------|
| RealModelManager | 管理真实模型的加载、缓存和卸载 | transformers, torch |
| ModelLoader | 加载HuggingFace模型，处理设备和精度 | AutoModelForCausalLM |
| ModelCache | LRU缓存策略，内存监控 | - |
| QuantizationMgr | 量化模型加载和管理 | bitsandbytes, auto-gptq |
| InferenceEngine | 执行模型推理，支持批处理 | - |
| FlashAttn | FlashAttention检测和启用 | flash-attn |
| LatencyBenchmark | 延迟性能测试 | time, statistics |
| MemoryBenchmark | 内存占用测试 | torch.cuda, psutil |
| E2EValidator | 端到端功能验证 | workflows |

## 2. 详细设计

### 2.1 RealModelManager

扩展现有ModelManager，支持真实模型加载。

```python
@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str                    # HuggingFace模型名称
    model_type: str                    # "qwen-chat" | "gpt2" | "custom"
    device: str = "auto"               # "auto" | "cuda" | "cpu"
    dtype: str = "auto"                # "auto" | "fp32" | "fp16" | "bf16"
    quantization: Optional[str] = None # None | "8bit" | "4bit" | "gptq"
    trust_remote_code: bool = False    # Qwen等自定义模型需要
    max_memory: Optional[Dict] = None  # 多GPU内存分配
    flash_attention: bool = True       # 启用FlashAttention

class RealModelManager:
    """真实模型管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model_cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._max_cached_models = config.get("max_cached_models", 3)
        self._memory_monitor = MemoryMonitor()
    
    def load_model(self, model_config: ModelConfig) -> LoadedModel:
        """加载模型，支持缓存和量化"""
        pass
    
    def unload_model(self, model_name: str) -> bool:
        """卸载模型释放内存"""
        pass
    
    def get_model(self, model_name: str) -> Optional[LoadedModel]:
        """获取已加载的模型"""
        pass
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> str:
        """执行文本生成"""
        pass
```

### 2.2 量化支持

```python
class QuantizationManager:
    """量化管理器"""
    
    @staticmethod
    def get_quantization_config(quant_type: str) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if quant_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        elif quant_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        return None
    
    @staticmethod
    def load_gptq_model(model_name: str, device: str) -> PreTrainedModel:
        """加载GPTQ量化模型"""
        pass
```

### 2.3 推理引擎

```python
class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._flash_attention_enabled = self._detect_flash_attention()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> GenerationResult:
        """单次生成"""
        pass
    
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """流式生成"""
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[GenerationResult]:
        """批量生成"""
        pass
    
    def _detect_flash_attention(self) -> bool:
        """检测FlashAttention可用性"""
        pass
```

### 2.4 对话管理

```python
class ConversationManager:
    """对话管理器 - 支持多轮对话"""
    
    def __init__(self, model_type: str, max_history_tokens: int = 4096):
        self.model_type = model_type
        self.max_history_tokens = max_history_tokens
        self.history: List[Dict[str, str]] = []
    
    def add_turn(self, role: str, content: str):
        """添加对话轮次"""
        pass
    
    def build_prompt(self, system_prompt: Optional[str] = None) -> str:
        """构建完整提示词"""
        if self.model_type == "qwen-chat":
            return self._build_qwen_prompt(system_prompt)
        elif self.model_type == "gpt2":
            return self._build_gpt2_prompt(system_prompt)
        pass
    
    def truncate_history(self, tokenizer: PreTrainedTokenizer):
        """截断历史以适应上下文窗口"""
        pass
```

### 2.5 性能基准测试

```python
@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    test_type: str
    metrics: Dict[str, float]
    timestamp: datetime
    config: Dict[str, Any]

class LatencyBenchmark:
    """延迟基准测试"""
    
    def measure_ttft(self, engine: InferenceEngine, prompt: str) -> float:
        """测量首token延迟"""
        pass
    
    def measure_tokens_per_second(
        self, 
        engine: InferenceEngine, 
        prompt: str,
        num_tokens: int = 100
    ) -> float:
        """测量生成速度"""
        pass
    
    def run_latency_suite(
        self,
        engine: InferenceEngine,
        test_prompts: List[str]
    ) -> BenchmarkResult:
        """运行完整延迟测试套件"""
        pass

class MemoryBenchmark:
    """内存基准测试"""
    
    def measure_model_memory(self, model: PreTrainedModel) -> Dict[str, int]:
        """测量模型内存占用"""
        pass
    
    def measure_inference_memory(
        self,
        engine: InferenceEngine,
        prompt: str,
        num_tokens: int
    ) -> Dict[str, int]:
        """测量推理内存增长"""
        pass

class ThroughputBenchmark:
    """吞吐量基准测试"""
    
    def measure_single_throughput(
        self,
        engine: InferenceEngine,
        prompt: str,
        num_tokens: int
    ) -> float:
        """测量单请求吞吐量"""
        pass
    
    def measure_concurrent_throughput(
        self,
        engine: InferenceEngine,
        prompts: List[str],
        num_concurrent: int
    ) -> float:
        """测量并发吞吐量"""
        pass
```

### 2.6 端到端验证器

```python
class E2EValidator:
    """端到端功能验证器"""
    
    def __init__(self, model_manager: RealModelManager):
        self.model_manager = model_manager
    
    def validate_search_qa(
        self,
        model_name: str,
        test_cases: List[SearchQATestCase]
    ) -> ValidationResult:
        """验证SearchQA场景"""
        pass
    
    def validate_lesson_pack(
        self,
        model_name: str,
        test_cases: List[LessonPackTestCase]
    ) -> ValidationResult:
        """验证LessonPack场景"""
        pass
    
    def validate_multi_turn(
        self,
        model_name: str,
        conversation: List[Dict[str, str]]
    ) -> ValidationResult:
        """验证多轮对话"""
        pass
```

## 3. 配置设计

### 3.1 模型配置文件

```yaml
# config/models.yaml
models:
  qwen-7b-chat:
    model_name: "Qwen/Qwen-7B-Chat"
    model_type: "qwen-chat"
    device: "auto"
    dtype: "bf16"
    quantization: null
    trust_remote_code: true
    flash_attention: true
    max_context_length: 8192
    
  qwen-7b-chat-int4:
    model_name: "Qwen/Qwen-7B-Chat-Int4"
    model_type: "qwen-chat"
    device: "auto"
    dtype: "fp16"
    quantization: "gptq"
    trust_remote_code: true
    flash_attention: true
    max_context_length: 8192
    
  gpt2:
    model_name: "gpt2"
    model_type: "gpt2"
    device: "auto"
    dtype: "fp32"
    quantization: null
    trust_remote_code: false
    flash_attention: false
    max_context_length: 1024
    
  gpt2-medium:
    model_name: "gpt2-medium"
    model_type: "gpt2"
    device: "auto"
    dtype: "fp16"
    quantization: "8bit"
    trust_remote_code: false
    flash_attention: false
    max_context_length: 1024

cache:
  max_cached_models: 3
  eviction_policy: "lru"
  memory_threshold: 0.8  # 80% GPU内存使用率触发卸载

inference:
  default_max_new_tokens: 512
  default_temperature: 0.7
  default_top_p: 0.9
  batch_size: 4
  timeout_seconds: 120
```

### 3.2 基准测试配置

```yaml
# config/benchmark.yaml
benchmark:
  latency:
    warmup_runs: 3
    test_runs: 10
    input_lengths: [128, 512, 1024, 2048]
    output_lengths: [64, 128, 256, 512]
    
  memory:
    measure_peak: true
    measure_kv_cache: true
    gc_before_measure: true
    
  throughput:
    concurrent_requests: [1, 2, 4, 8, 16]
    tokens_per_request: 100
    duration_seconds: 60
    
  report:
    output_dir: "data/benchmarks"
    format: "json"  # json | csv | html
    include_system_info: true
```

## 4. 正确性属性

### 4.1 模型加载属性

| ID | 属性名称 | 描述 | 验证方法 |
|----|----------|------|----------|
| P1.1 | 加载完整性 | 加载的模型能够执行推理 | 加载后执行简单生成测试 |
| P1.2 | 设备一致性 | 模型在指定设备上运行 | 检查model.device |
| P1.3 | 量化正确性 | 量化模型输出与全精度相似 | 对比输出相似度>0.9 |
| P1.4 | 缓存一致性 | 缓存的模型与新加载的行为一致 | 对比相同输入的输出 |

### 4.2 推理属性

| ID | 属性名称 | 描述 | 验证方法 |
|----|----------|------|----------|
| P2.1 | 输出确定性 | 相同输入+种子产生相同输出 | 多次运行对比 |
| P2.2 | 长度约束 | 输出不超过max_new_tokens | 检查输出token数 |
| P2.3 | 批处理等价性 | 批处理结果与单独处理一致 | 对比批处理和单独结果 |
| P2.4 | 流式完整性 | 流式输出拼接等于完整输出 | 对比流式和非流式结果 |

### 4.3 对话属性

| ID | 属性名称 | 描述 | 验证方法 |
|----|----------|------|----------|
| P3.1 | 历史保持 | 多轮对话保持上下文 | 检查后续回答引用前文 |
| P3.2 | 截断安全 | 截断不破坏对话结构 | 验证截断后仍可正常对话 |
| P3.3 | 角色一致 | 模型保持指定角色 | 检查回答风格一致性 |

### 4.4 性能属性

| ID | 属性名称 | 描述 | 验证方法 |
|----|----------|------|----------|
| P4.1 | 延迟稳定性 | 延迟波动<20% | 统计标准差 |
| P4.2 | 内存有界 | 内存不超过配置阈值 | 监控峰值内存 |
| P4.3 | 吞吐线性 | 批处理吞吐近似线性增长 | 对比不同批大小 |

## 5. 错误处理

### 5.1 错误类型

```python
class ModelLoadError(Exception):
    """模型加载错误"""
    pass

class InferenceError(Exception):
    """推理错误"""
    pass

class OutOfMemoryError(Exception):
    """内存不足错误"""
    pass

class QuantizationError(Exception):
    """量化错误"""
    pass
```

### 5.2 回退策略

| 错误场景 | 回退策略 |
|----------|----------|
| GPU内存不足 | 尝试量化加载 → 切换CPU |
| 模型下载失败 | 使用本地缓存 → 切换备用模型 |
| FlashAttention不可用 | 使用标准注意力 |
| 量化加载失败 | 回退全精度加载 |
| 推理超时 | 返回部分结果 + 错误标记 |

## 6. 测试策略

### 6.1 单元测试

- 模型加载器测试（mock模型）
- 量化配置测试
- 缓存管理测试
- 对话管理测试

### 6.2 集成测试

- 真实模型加载测试（小模型如gpt2）
- 端到端工作流测试
- 多模型切换测试

### 6.3 属性测试

- 使用Hypothesis测试正确性属性
- 生成随机输入验证输出约束
- 压力测试内存和性能边界

### 6.4 基准测试

- 延迟基准（不同输入/输出长度）
- 内存基准（不同量化级别）
- 吞吐基准（不同并发级别）
