# 性能基准测试指南

本指南介绍如何使用MuAI系统的性能基准测试功能来评估模型性能。

## 概述

MuAI基准测试系统提供三种核心测试：
1. **延迟测试 (Latency)**: 测量首token时间(TTFT)和生成速度(tokens/s)
2. **内存测试 (Memory)**: 测量模型加载和推理时的内存占用
3. **吞吐量测试 (Throughput)**: 测量单请求和并发场景下的处理能力

## 快速开始

### 命令行使用

```bash
# 运行完整基准测试
python -m mm_orch.main --benchmark --model gpt2

# 指定输出目录
python -m mm_orch.main --benchmark --model gpt2 --benchmark-output ./results

# 指定报告格式
python -m mm_orch.main --benchmark --model gpt2 --benchmark-format csv
```

### API使用

```bash
# 通过API运行基准测试
curl -X POST http://localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt2", "test_types": ["latency", "memory"]}'
```

## 配置文件

基准测试配置位于 `config/benchmark.yaml`：

```yaml
benchmark:
  latency:
    warmup_runs: 3        # 预热运行次数
    test_runs: 10         # 测试运行次数
    input_lengths: [128, 512, 1024, 2048]
    output_lengths: [64, 128, 256, 512]
    
  memory:
    measure_peak: true    # 测量峰值内存
    measure_kv_cache: true
    gc_before_measure: true
    
  throughput:
    concurrent_requests: [1, 2, 4, 8, 16]
    tokens_per_request: 100
    duration_seconds: 60
    batch_sizes: [1, 2, 4, 8]
    
  report:
    output_dir: "data/benchmarks"
    format: "json"        # json | csv
    include_system_info: true
```

## 延迟测试

### 测量指标

| 指标 | 说明 | 单位 |
|------|------|------|
| TTFT | 首token生成时间 | 毫秒 (ms) |
| tokens/s | 每秒生成token数 | tokens/second |
| E2E Latency | 端到端延迟 | 毫秒 (ms) |

### Python API

```python
from mm_orch.benchmark.latency import LatencyBenchmark

benchmark = LatencyBenchmark(
    warmup_runs=3,
    test_runs=10
)

# 运行测试
results = benchmark.run(
    engine=inference_engine,
    prompts=["Hello, how are you?"],
    output_length=100
)

print(f"TTFT: {results.ttft_ms:.2f} ms")
print(f"Tokens/s: {results.tokens_per_second:.1f}")
print(f"E2E Latency: {results.e2e_latency_ms:.2f} ms")
```

### 不同输入长度测试

```python
# 测试不同输入长度的性能
input_lengths = [128, 512, 1024, 2048]

for length in input_lengths:
    prompt = "x " * (length // 2)  # 生成指定长度的输入
    results = benchmark.run(engine, [prompt], output_length=100)
    print(f"Input {length}: {results.tokens_per_second:.1f} tokens/s")
```

## 内存测试

### 测量指标

| 指标 | 说明 | 单位 |
|------|------|------|
| Model Load Memory | 模型加载内存 | GB |
| Inference Memory | 推理时内存增长 | GB |
| Peak Memory | 峰值内存使用 | GB |
| KV Cache Size | KV缓存大小 | GB |

### Python API

```python
from mm_orch.benchmark.memory import MemoryBenchmark

benchmark = MemoryBenchmark(
    measure_peak=True,
    gc_before_measure=True
)

# 测量模型加载内存
load_result = benchmark.measure_model_load(model_manager, model_config)
print(f"Model Load: {load_result.memory_gb:.2f} GB")

# 测量推理内存
inference_result = benchmark.measure_inference(
    engine=inference_engine,
    prompt="Hello",
    output_length=100
)
print(f"Inference Memory: {inference_result.memory_delta_gb:.3f} GB")
print(f"Peak Memory: {inference_result.peak_memory_gb:.2f} GB")
```

### 量化对比测试

```python
# 对比不同量化方式的内存占用
configs = [
    ModelConfig(model_name="gpt2-medium", quantization=None),
    ModelConfig(model_name="gpt2-medium", quantization="8bit"),
    ModelConfig(model_name="gpt2-medium", quantization="4bit"),
]

for config in configs:
    result = benchmark.measure_model_load(manager, config)
    print(f"{config.quantization or 'fp32'}: {result.memory_gb:.2f} GB")
```

## 吞吐量测试

### 测量指标

| 指标 | 说明 | 单位 |
|------|------|------|
| Requests/s | 每秒处理请求数 | requests/second |
| Total Tokens/s | 总token生成速度 | tokens/second |
| Avg Latency | 平均请求延迟 | 毫秒 (ms) |
| P99 Latency | 99分位延迟 | 毫秒 (ms) |

### Python API

```python
from mm_orch.benchmark.throughput import ThroughputBenchmark

benchmark = ThroughputBenchmark(
    duration_seconds=60,
    tokens_per_request=100
)

# 单请求吞吐量
single_result = benchmark.measure_single_request(
    engine=inference_engine,
    prompts=test_prompts
)
print(f"Single Request: {single_result.tokens_per_second:.1f} tokens/s")

# 并发吞吐量
concurrent_result = benchmark.measure_concurrent(
    engine=inference_engine,
    prompts=test_prompts,
    num_concurrent=4
)
print(f"Concurrent (4): {concurrent_result.requests_per_second:.2f} req/s")

# 批处理吞吐量
batch_result = benchmark.measure_batch(
    engine=inference_engine,
    prompts=test_prompts,
    batch_size=8
)
print(f"Batch (8): {batch_result.total_tokens_per_second:.1f} tokens/s")
```

## 报告生成

### JSON报告

```python
from mm_orch.benchmark.reporter import BenchmarkReporter

reporter = BenchmarkReporter(
    output_dir="data/benchmarks",
    include_system_info=True
)

# 生成JSON报告
report_path = reporter.generate_json_report(
    model_name="gpt2",
    latency_results=latency_results,
    memory_results=memory_results,
    throughput_results=throughput_results
)
print(f"Report saved to: {report_path}")
```

### CSV报告

```python
# 生成CSV报告（适合数据分析）
csv_path = reporter.generate_csv_report(
    model_name="gpt2",
    results=all_results
)
```

### 报告内容示例

```json
{
  "model_name": "gpt2",
  "timestamp": "2024-01-15T10:30:00Z",
  "system_info": {
    "gpu": "NVIDIA T4",
    "gpu_memory_gb": 15.0,
    "cpu": "Intel Xeon",
    "ram_gb": 32.0,
    "python_version": "3.10.0",
    "torch_version": "2.1.0"
  },
  "latency": {
    "ttft_ms": 45.2,
    "tokens_per_second": 35.8,
    "e2e_latency_ms": 2850.5
  },
  "memory": {
    "model_load_gb": 1.2,
    "inference_delta_gb": 0.15,
    "peak_gb": 1.45
  },
  "throughput": {
    "single_request_tps": 35.8,
    "concurrent_4_rps": 12.5,
    "batch_8_tps": 180.2
  }
}
```

## 性能目标

配置文件中可以设置性能目标用于验证：

```yaml
targets:
  qwen-7b-chat:
    min_tokens_per_second: 30
    max_ttft_ms: 500
    max_memory_gb: 16
  gpt2:
    min_tokens_per_second: 5
    max_ttft_ms: 200
    max_memory_gb: 2
```

### 验证性能目标

```python
from mm_orch.benchmark.reporter import validate_targets

# 验证结果是否满足目标
validation = validate_targets(
    model_name="gpt2",
    results=benchmark_results,
    targets=config["targets"]["gpt2"]
)

if validation.passed:
    print("All targets met!")
else:
    for failure in validation.failures:
        print(f"Failed: {failure.metric} = {failure.actual} (target: {failure.target})")
```

## 测试提示词

配置文件包含预定义的测试提示词：

```yaml
test_prompts:
  short:
    - "What is the capital of France?"
    - "Explain quantum computing briefly."
  medium:
    - "Explain the difference between machine learning and deep learning."
  long:
    - "Write a comprehensive guide on building a REST API..."
```

### 使用测试提示词

```python
import yaml

with open("config/benchmark.yaml") as f:
    config = yaml.safe_load(f)

prompts = config["test_prompts"]["medium"]
results = benchmark.run(engine, prompts, output_length=100)
```

## 最佳实践

### 1. 预热运行

始终进行预热运行以获得稳定结果：

```python
benchmark = LatencyBenchmark(
    warmup_runs=3,  # 至少3次预热
    test_runs=10
)
```

### 2. 多次测试取平均

```python
# 运行多次测试
all_results = []
for _ in range(5):
    result = benchmark.run(engine, prompts, output_length=100)
    all_results.append(result)

# 计算平均值
avg_tps = sum(r.tokens_per_second for r in all_results) / len(all_results)
```

### 3. 控制变量

测试时保持其他条件一致：
- 相同的输入长度
- 相同的输出长度
- 相同的生成参数
- 相同的系统负载

### 4. 记录系统信息

```python
reporter = BenchmarkReporter(
    include_system_info=True,  # 记录GPU、CPU、内存等信息
    include_model_info=True    # 记录模型配置
)
```

## 故障排除

### 结果不稳定

可能原因：
1. 预热不足 - 增加 `warmup_runs`
2. 系统负载波动 - 确保测试时系统空闲
3. GPU温度过高 - 等待GPU冷却

### 内存测量不准确

```python
benchmark = MemoryBenchmark(
    gc_before_measure=True,  # 测量前进行垃圾回收
    measure_peak=True        # 测量峰值而非瞬时值
)
```

### 并发测试失败

确保有足够的GPU内存支持并发请求：

```python
# 从小的并发数开始
concurrent_requests = [1, 2, 4]  # 而不是 [1, 2, 4, 8, 16]
```

## 参考资料

- [PyTorch性能调优指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [HuggingFace基准测试](https://huggingface.co/docs/transformers/benchmarks)
- [NVIDIA GPU性能分析](https://developer.nvidia.com/nsight-systems)
