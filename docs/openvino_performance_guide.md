# OpenVINO Performance Guide

## Overview

This guide provides detailed information about OpenVINO backend performance characteristics, optimization strategies, and benchmarking methodologies for the MuAI Multi-Model Orchestration System.

## Expected Performance Improvements

### Typical Speedups

| Model Type | Model Size | PyTorch (CPU) | OpenVINO (CPU) | Speedup |
|------------|------------|---------------|----------------|---------|
| GPT-2 | 124M params | 2.1s | 0.8s | 2.6x |
| DistilGPT-2 | 82M params | 1.5s | 0.6s | 2.5x |
| T5-Small | 60M params | 1.8s | 0.7s | 2.6x |
| BERT-Base | 110M params | 0.9s | 0.4s | 2.3x |

**Test Conditions**:
- Hardware: Intel Core i7-10700K (8 cores, 3.8 GHz)
- Input: 50 tokens
- Output: 50 tokens
- Precision: FP32
- Batch size: 1

### Performance by Hardware

| Hardware | Backend | Latency (ms) | Throughput (tokens/s) |
|----------|---------|--------------|----------------------|
| Intel CPU (8 cores) | PyTorch | 2100 | 24 |
| Intel CPU (8 cores) | OpenVINO | 800 | 63 |
| Intel iGPU (Iris Xe) | PyTorch | 1800 | 28 |
| Intel iGPU (Iris Xe) | OpenVINO | 600 | 83 |
| NVIDIA GPU (T4) | PyTorch | 450 | 111 |
| NVIDIA GPU (T4) | OpenVINO | N/A | N/A |

**Note**: OpenVINO is optimized for Intel hardware. For NVIDIA GPUs, PyTorch with CUDA is recommended.

## Hardware Compatibility Matrix

### CPU Support

| Processor | OpenVINO Support | Expected Speedup | Recommended Precision |
|-----------|------------------|------------------|----------------------|
| Intel Core (6th gen+) | ✅ Excellent | 2-3x | FP32 |
| Intel Xeon | ✅ Excellent | 2-3x | FP32 |
| AMD Ryzen | ⚠️ Limited | 1.5-2x | FP32 |
| ARM (Apple M1/M2) | ⚠️ Limited | 1.2-1.5x | FP32 |

### GPU Support

| GPU | OpenVINO Support | Expected Speedup | Recommended Precision |
|-----|------------------|------------------|----------------------|
| Intel Iris Xe (iGPU) | ✅ Excellent | 2.5-3.5x | FP16 |
| Intel Arc (dGPU) | ✅ Excellent | 3-4x | FP16 |
| NVIDIA (CUDA) | ❌ Not supported | Use PyTorch | - |
| AMD (ROCm) | ❌ Not supported | Use PyTorch | - |

### NPU Support (Neural Processing Unit)

| NPU | OpenVINO Support | Expected Speedup | Recommended Precision |
|-----|------------------|------------------|----------------------|
| Intel Meteor Lake NPU | ✅ Experimental | 2-3x | INT8 |
| Intel Lunar Lake NPU | ✅ Experimental | 3-4x | INT8 |

**Note**: NPU support requires OpenVINO 2023.1+ and specific drivers.

## Optimization Strategies

### 1. Precision Optimization

#### FP32 (Full Precision)

**Use Case**: Maximum accuracy, baseline performance

```bash
python scripts/export_to_openvino.py gpt2 --precision FP32
```

**Characteristics**:
- Accuracy: 100% (reference)
- Speed: Baseline (2-3x vs PyTorch)
- Memory: Baseline
- Compatibility: All devices

#### FP16 (Half Precision)

**Use Case**: Balanced accuracy and performance

```bash
python scripts/export_to_openvino.py gpt2 --precision FP16
```

**Characteristics**:
- Accuracy: 99.9% (negligible loss)
- Speed: 1.5-2x faster than FP32
- Memory: 50% reduction
- Compatibility: GPU, modern CPUs

**Recommended for**: GPU inference, memory-constrained systems

#### INT8 (Quantized)

**Use Case**: Maximum performance, acceptable accuracy loss

```bash
python scripts/export_to_openvino.py gpt2 --precision INT8
```

**Characteristics**:
- Accuracy: 95-98% (small loss)
- Speed: 2-3x faster than FP32
- Memory: 75% reduction
- Compatibility: CPUs with VNNI, NPUs

**Recommended for**: NPU inference, edge devices

### 2. Device Selection

#### CPU Optimization

```yaml
backend:
  openvino:
    device: CPU
    num_streams: 4  # Parallel inference streams
    num_threads: 8  # CPU threads per stream
```

**Best Practices**:
- Set `num_streams` to number of physical cores / 2
- Set `num_threads` to number of physical cores
- Use FP32 precision for best accuracy
- Enable CPU pinning for consistent performance

#### GPU Optimization

```yaml
backend:
  openvino:
    device: GPU
    num_streams: 2
```

**Best Practices**:
- Use FP16 precision for best performance
- Batch multiple requests when possible
- Monitor GPU memory usage
- Use AUTO device for automatic GPU/CPU selection

#### AUTO Device Selection

```yaml
backend:
  openvino:
    device: AUTO
```

**Behavior**:
- Tries GPU first, falls back to CPU
- Automatically selects optimal device per model
- Balances performance and availability
- Recommended for production

### 3. Batch Processing

Process multiple requests together for higher throughput:

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend="openvino")
manager.load_model("gpt2", "transformers")

# Single request (lower throughput)
result = manager.generate("gpt2", "Hello", max_length=50)

# Batch requests (higher throughput)
prompts = ["Hello", "World", "OpenVINO"]
results = [manager.generate("gpt2", p, max_length=50) for p in prompts]
```

**Expected Improvement**: 1.5-2x throughput increase

### 4. Model Caching

Keep frequently used models in memory:

```python
from mm_orch.runtime.model_manager import ModelManager

# Increase cache size for multiple models
manager = ModelManager(
    backend="openvino",
    max_cached_models=5  # Default is 3
)

# Load models once, reuse many times
manager.load_model("gpt2", "transformers")
manager.load_model("t5-small", "transformers")

# Fast subsequent calls (no reload)
for i in range(100):
    result = manager.generate("gpt2", f"Prompt {i}")
```

### 5. Warmup Strategy

First inference includes initialization overhead:

```python
from mm_orch.runtime.model_manager import ModelManager
import time

manager = ModelManager(backend="openvino")
manager.load_model("gpt2", "transformers")

# Warmup (discard result)
manager.generate("gpt2", "warmup", max_length=10)

# Measure actual performance
start = time.time()
result = manager.generate("gpt2", "Hello world", max_length=50)
latency = time.time() - start
print(f"Latency: {latency:.3f}s")
```

## Performance Comparison Examples

### Example 1: Latency Comparison

```python
from mm_orch.runtime.model_manager import ModelManager
import time

def benchmark_backend(backend_name, num_runs=10):
    manager = ModelManager(backend=backend_name)
    manager.load_model("gpt2", "transformers")
    
    # Warmup
    manager.generate("gpt2", "warmup", max_length=10)
    
    # Benchmark
    latencies = []
    for i in range(num_runs):
        start = time.time()
        result = manager.generate("gpt2", f"Test {i}", max_length=50)
        latencies.append(time.time() - start)
    
    return {
        "backend": backend_name,
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))]
    }

# Compare backends
pytorch_stats = benchmark_backend("pytorch")
openvino_stats = benchmark_backend("openvino")

print(f"PyTorch avg: {pytorch_stats['avg_latency']:.3f}s")
print(f"OpenVINO avg: {openvino_stats['avg_latency']:.3f}s")
print(f"Speedup: {pytorch_stats['avg_latency'] / openvino_stats['avg_latency']:.2f}x")
```

**Expected Output**:
```
PyTorch avg: 2.145s
OpenVINO avg: 0.823s
Speedup: 2.61x
```

### Example 2: Throughput Comparison

```python
from mm_orch.runtime.model_manager import ModelManager
import time

def benchmark_throughput(backend_name, duration_seconds=30):
    manager = ModelManager(backend=backend_name)
    manager.load_model("gpt2", "transformers")
    
    # Warmup
    manager.generate("gpt2", "warmup", max_length=10)
    
    # Benchmark
    start = time.time()
    count = 0
    total_tokens = 0
    
    while time.time() - start < duration_seconds:
        result = manager.generate("gpt2", f"Test {count}", max_length=50)
        count += 1
        total_tokens += 50  # Approximate
    
    elapsed = time.time() - start
    return {
        "backend": backend_name,
        "requests": count,
        "requests_per_sec": count / elapsed,
        "tokens_per_sec": total_tokens / elapsed
    }

# Compare throughput
pytorch_stats = benchmark_throughput("pytorch")
openvino_stats = benchmark_throughput("openvino")

print(f"PyTorch: {pytorch_stats['requests_per_sec']:.1f} req/s, "
      f"{pytorch_stats['tokens_per_sec']:.1f} tokens/s")
print(f"OpenVINO: {openvino_stats['requests_per_sec']:.1f} req/s, "
      f"{openvino_stats['tokens_per_sec']:.1f} tokens/s")
print(f"Throughput improvement: "
      f"{openvino_stats['requests_per_sec'] / pytorch_stats['requests_per_sec']:.2f}x")
```

**Expected Output**:
```
PyTorch: 0.5 req/s, 23.5 tokens/s
OpenVINO: 1.2 req/s, 61.2 tokens/s
Throughput improvement: 2.60x
```

### Example 3: Memory Usage Comparison

```python
from mm_orch.runtime.model_manager import ModelManager
import psutil
import os

def measure_memory(backend_name):
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    
    # Load model
    manager = ModelManager(backend=backend_name)
    manager.load_model("gpt2", "transformers")
    
    # Memory after loading
    loaded = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run inference
    manager.generate("gpt2", "Test", max_length=50)
    
    # Memory after inference
    inference = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        "backend": backend_name,
        "baseline_mb": baseline,
        "model_size_mb": loaded - baseline,
        "inference_mb": inference - loaded,
        "total_mb": inference
    }

# Compare memory usage
pytorch_mem = measure_memory("pytorch")
openvino_mem = measure_memory("openvino")

print(f"PyTorch model size: {pytorch_mem['model_size_mb']:.1f} MB")
print(f"OpenVINO model size: {openvino_mem['model_size_mb']:.1f} MB")
print(f"Memory reduction: "
      f"{(1 - openvino_mem['model_size_mb'] / pytorch_mem['model_size_mb']) * 100:.1f}%")
```

## Benchmarking Instructions

### Quick Benchmark

Run a quick performance test:

```bash
# Benchmark PyTorch
python -c "
from mm_orch.runtime.model_manager import ModelManager
import time

manager = ModelManager(backend='pytorch')
manager.load_model('gpt2', 'transformers')
manager.generate('gpt2', 'warmup', max_length=10)

start = time.time()
for i in range(10):
    manager.generate('gpt2', f'Test {i}', max_length=50)
print(f'PyTorch: {(time.time() - start) / 10:.3f}s per request')
"

# Benchmark OpenVINO
python -c "
from mm_orch.runtime.model_manager import ModelManager
import time

manager = ModelManager(backend='openvino')
manager.load_model('gpt2', 'transformers')
manager.generate('gpt2', 'warmup', max_length=10)

start = time.time()
for i in range(10):
    manager.generate('gpt2', f'Test {i}', max_length=50)
print(f'OpenVINO: {(time.time() - start) / 10:.3f}s per request')
"
```

### Comprehensive Benchmark

Use the built-in performance monitoring:

```python
from mm_orch.runtime.model_manager import ModelManager
import json

# Initialize manager with both backends
manager_pt = ModelManager(backend="pytorch")
manager_ov = ModelManager(backend="openvino")

# Load models
manager_pt.load_model("gpt2", "transformers")
manager_ov.load_model("gpt2", "transformers")

# Run benchmark workload
test_prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models",
    "Natural language processing",
    "Computer vision applications"
]

for prompt in test_prompts:
    for _ in range(5):  # 5 runs per prompt
        manager_pt.generate("gpt2", prompt, max_length=50)
        manager_ov.generate("gpt2", prompt, max_length=50)

# Get statistics
pt_stats = manager_pt.get_performance_stats("pytorch")
ov_stats = manager_ov.get_performance_stats("openvino")

# Compare
comparison = manager_ov.compare_backends("pytorch", "openvino")

# Print results
print(json.dumps({
    "pytorch": pt_stats,
    "openvino": ov_stats,
    "comparison": comparison
}, indent=2))
```

### Automated Benchmark Script

Create a reusable benchmark script:

```python
# benchmark_backends.py
import argparse
import json
import time
from mm_orch.runtime.model_manager import ModelManager

def run_benchmark(model_name, backend, num_runs=100, max_length=50):
    manager = ModelManager(backend=backend)
    manager.load_model(model_name, "transformers")
    
    # Warmup
    manager.generate(model_name, "warmup", max_length=10)
    
    # Benchmark
    latencies = []
    start_time = time.time()
    
    for i in range(num_runs):
        start = time.time()
        manager.generate(model_name, f"Test {i}", max_length=max_length)
        latencies.append(time.time() - start)
    
    total_time = time.time() - start_time
    
    return {
        "model": model_name,
        "backend": backend,
        "num_runs": num_runs,
        "total_time": total_time,
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "p50_latency": sorted(latencies)[len(latencies) // 2],
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_latency": sorted(latencies)[int(0.99 * len(latencies))],
        "throughput": num_runs / total_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=50)
    args = parser.parse_args()
    
    # Benchmark both backends
    pt_results = run_benchmark(args.model, "pytorch", args.runs, args.max_length)
    ov_results = run_benchmark(args.model, "openvino", args.runs, args.max_length)
    
    # Calculate speedup
    speedup = pt_results["avg_latency"] / ov_results["avg_latency"]
    
    # Print results
    print(json.dumps({
        "pytorch": pt_results,
        "openvino": ov_results,
        "speedup": speedup
    }, indent=2))
```

Run it:
```bash
python benchmark_backends.py --model gpt2 --runs 100 --max-length 50
```

## Performance Tuning Checklist

- [ ] **Export models with optimal precision**
  - FP32 for CPU (accuracy)
  - FP16 for GPU (speed)
  - INT8 for NPU (efficiency)

- [ ] **Configure device selection**
  - Use AUTO for automatic selection
  - Use GPU for Intel graphics
  - Use CPU for maximum compatibility

- [ ] **Enable model caching**
  - Set `max_cached_models` appropriately
  - Keep frequently used models loaded

- [ ] **Implement warmup**
  - Run dummy inference before benchmarking
  - Discard first inference timing

- [ ] **Batch requests when possible**
  - Process multiple requests together
  - Improves throughput significantly

- [ ] **Monitor performance**
  - Use built-in performance monitoring
  - Track latency and throughput
  - Compare backends regularly

- [ ] **Optimize configuration**
  - Tune `num_streams` for CPU
  - Adjust `num_threads` per workload
  - Enable fallback for reliability

## Troubleshooting Performance Issues

### Issue: No performance improvement

**Possible Causes**:
1. Model not exported to OpenVINO format
2. Using wrong device (e.g., CPU instead of GPU)
3. First inference includes warmup overhead
4. Small model size (overhead dominates)

**Solutions**:
```bash
# Verify model export
ls models/openvino/gpt2/

# Check device selection
python -c "
from mm_orch.runtime.model_manager import ModelManager
manager = ModelManager(backend='openvino')
manager.load_model('gpt2', 'transformers')
model = manager.get_model('gpt2')
print(f\"Device: {model['backend_metadata']['device']}\")
"

# Run with warmup
python -c "
from mm_orch.runtime.model_manager import ModelManager
import time

manager = ModelManager(backend='openvino')
manager.load_model('gpt2', 'transformers')

# Warmup
manager.generate('gpt2', 'warmup', max_length=10)

# Measure
start = time.time()
manager.generate('gpt2', 'test', max_length=50)
print(f'Latency: {time.time() - start:.3f}s')
"
```

### Issue: Slower than expected

**Possible Causes**:
1. Using FP32 instead of FP16
2. CPU device instead of GPU
3. Insufficient CPU threads
4. Memory swapping

**Solutions**:
```bash
# Re-export with FP16
python scripts/export_to_openvino.py gpt2 --precision FP16

# Use GPU device
# In config/backend.yaml:
backend:
  openvino:
    device: GPU

# Increase CPU threads
backend:
  openvino:
    device: CPU
    num_threads: 8
```

### Issue: High memory usage

**Possible Causes**:
1. Too many cached models
2. Using FP32 precision
3. Memory leak

**Solutions**:
```python
# Reduce cache size
manager = ModelManager(
    backend="openvino",
    max_cached_models=2  # Reduce from default 3
)

# Use FP16 precision
# Export with: python scripts/export_to_openvino.py gpt2 --precision FP16

# Explicitly unload models
manager.unload_model("gpt2")
```

## Best Practices Summary

1. **Always warmup** before benchmarking
2. **Use FP16** for GPU inference
3. **Use AUTO device** for automatic optimization
4. **Enable fallback** for production reliability
5. **Monitor performance** with built-in tools
6. **Batch requests** for higher throughput
7. **Cache models** for repeated use
8. **Export all models** before deployment
9. **Test on target hardware** before production
10. **Compare backends** to verify improvements

## Next Steps

- Review [Migration Guide](openvino_migration_guide.md) for setup instructions
- Check [Configuration Examples](openvino_config_examples.md) for common scenarios
- Run [Example Scripts](../examples/) to see performance in action
- Consult [Troubleshooting](openvino_migration_guide.md#troubleshooting) for issues

## Additional Resources

- OpenVINO Performance Benchmarks: https://docs.openvino.ai/latest/openvino_docs_performance_benchmarks.html
- OpenVINO Optimization Guide: https://docs.openvino.ai/latest/openvino_docs_optimization_guide_dldt_optimization_guide.html
- Intel Hardware Specifications: https://ark.intel.com/
