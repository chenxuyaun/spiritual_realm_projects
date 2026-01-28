# Optimization and Monitoring Configuration Guide

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Optimization Configuration](#optimization-configuration)
4. [Monitoring Configuration](#monitoring-configuration)
5. [Environment Variables](#environment-variables)
6. [Configuration Examples](#configuration-examples)
7. [Best Practices](#best-practices)

## Overview

The MuAI Multi-Model Orchestration System provides advanced optimization and monitoring capabilities through a flexible configuration system. All features are optional and can be enabled/disabled independently.

### Configuration Priority

Configuration values are resolved in the following order (highest to lowest priority):

1. **Environment Variables** - Runtime overrides
2. **Configuration Files** - YAML configuration
3. **Default Values** - Built-in defaults

### Configuration Files

- **Primary**: `config/optimization.yaml` - Optimization and monitoring settings
- **Example**: `config/optimization.example.yaml` - Template with all options documented
- **System**: `config/system.yaml` - Can include optimization config inline

## Configuration File Structure

```yaml
optimization:
  enabled: true
  engine_preference: [vllm, deepspeed, onnx, pytorch]
  fallback_on_error: true
  vllm: {...}
  deepspeed: {...}
  onnx: {...}
  batcher: {...}
  cache: {...}
  tuner: {...}

monitoring:
  enabled: true
  prometheus: {...}
  tracing: {...}
  anomaly: {...}
  server: {...}
```

## Optimization Configuration

### Global Optimization Settings

```yaml
optimization:
  # Enable/disable all optimization features
  enabled: true
  
  # Engine preference order (first available is used)
  engine_preference:
    - vllm        # Highest priority
    - deepspeed
    - onnx
    - pytorch     # Fallback
  
  # Automatically fallback to next engine on error
  fallback_on_error: true
```

**Parameters:**

- `enabled` (bool): Master switch for optimization features
  - Default: `true`
  - Set to `false` to use standard PyTorch inference only

- `engine_preference` (list): Order of inference engines to try
  - Default: `[vllm, deepspeed, onnx, pytorch]`
  - System uses first available engine from this list

- `fallback_on_error` (bool): Enable automatic fallback on engine failure
  - Default: `true`
  - When `false`, engine errors will propagate to caller

### vLLM Configuration

vLLM provides high-throughput LLM inference with continuous batching and PagedAttention.

```yaml
optimization:
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: auto
    max_model_len: null
    gpu_memory_utilization: 0.9
    swap_space: 4
```

**Parameters:**

- `enabled` (bool): Enable vLLM engine
  - Default: `true`
  - Requires: `pip install vllm`

- `tensor_parallel_size` (int): Number of GPUs for tensor parallelism
  - Default: `1`
  - Recommended: Set to number of available GPUs for large models
  - Example: `4` for 4-GPU setup

- `dtype` (str): Data type for model weights
  - Default: `auto`
  - Options: `auto`, `fp16`, `fp32`, `bf16`
  - `auto`: Automatically select based on GPU capability
  - `fp16`: Half precision (recommended for most GPUs)
  - `bf16`: Brain Float 16 (requires A100 or newer)

- `max_model_len` (int|null): Maximum sequence length
  - Default: `null` (use model's default)
  - Set to limit memory usage for long sequences

- `gpu_memory_utilization` (float): Fraction of GPU memory to use
  - Default: `0.9`
  - Range: `0.0` to `1.0`
  - Higher values allow larger batches but risk OOM
  - Recommended: `0.85-0.95`

- `swap_space` (int): CPU swap space in GB
  - Default: `4`
  - Used when GPU memory is insufficient

**Use Cases:**

- **High-throughput serving**: Large batch sizes with continuous batching
- **Multi-GPU inference**: Distribute large models across GPUs
- **Production deployments**: Optimized for serving many concurrent requests

### DeepSpeed Configuration

DeepSpeed provides inference optimizations for large models with parallelism support.

```yaml
optimization:
  deepspeed:
    enabled: true
    tensor_parallel: 1
    pipeline_parallel: 1
    dtype: fp16
    replace_with_kernel_inject: true
```

**Parameters:**

- `enabled` (bool): Enable DeepSpeed engine
  - Default: `true`
  - Requires: `pip install deepspeed`

- `tensor_parallel` (int): Tensor parallelism degree
  - Default: `1`
  - Distributes model layers across GPUs
  - Example: `2` for 2-way tensor parallelism

- `pipeline_parallel` (int): Pipeline parallelism degree
  - Default: `1`
  - Distributes model stages across GPUs
  - Example: `4` for 4-stage pipeline

- `dtype` (str): Data type for inference
  - Default: `fp16`
  - Options: `fp16`, `fp32`, `bf16`

- `replace_with_kernel_inject` (bool): Use DeepSpeed optimized kernels
  - Default: `true`
  - Enables kernel fusion and other optimizations

**Use Cases:**

- **Large model inference**: Models that don't fit on single GPU
- **Multi-GPU setups**: Efficient parallelism strategies
- **Memory-constrained environments**: ZeRO-Inference for memory efficiency

### ONNX Runtime Configuration

ONNX Runtime provides cross-platform accelerated inference.

```yaml
optimization:
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
    optimization_level: all
    enable_quantization: false
```

**Parameters:**

- `enabled` (bool): Enable ONNX Runtime engine
  - Default: `true`
  - Requires: `pip install onnxruntime-gpu` or `onnxruntime`

- `execution_providers` (list): Execution providers in priority order
  - Default: `[CUDAExecutionProvider, CPUExecutionProvider]`
  - Options:
    - `CUDAExecutionProvider`: NVIDIA GPU acceleration
    - `TensorrtExecutionProvider`: TensorRT optimization
    - `CPUExecutionProvider`: CPU fallback
    - `OpenVINOExecutionProvider`: Intel hardware

- `optimization_level` (str): Graph optimization level
  - Default: `all`
  - Options: `none`, `basic`, `extended`, `all`
  - `all`: Maximum optimizations (recommended)

- `enable_quantization` (bool): Apply dynamic quantization
  - Default: `false`
  - Reduces model size and improves speed
  - May slightly reduce accuracy

**Use Cases:**

- **Cross-platform deployment**: Consistent performance across hardware
- **CPU inference**: Optimized CPU execution when GPU unavailable
- **Model optimization**: Graph-level optimizations and quantization

### Dynamic Batching Configuration

Dynamic batching groups multiple requests for efficient batch processing.

```yaml
optimization:
  batcher:
    enabled: true
    max_batch_size: 32
    batch_timeout_ms: 50
    adaptive_batching: true
    min_batch_size: 1
```

**Parameters:**

- `enabled` (bool): Enable dynamic batching
  - Default: `true`
  - Disable for single-request processing

- `max_batch_size` (int): Maximum requests per batch
  - Default: `32`
  - Recommended by GPU memory:
    - 16GB GPU: `16-32`
    - 32GB GPU: `32-64`
    - 80GB GPU: `64-128`

- `batch_timeout_ms` (int): Maximum wait time for batch formation
  - Default: `50` milliseconds
  - Lower values: Reduced latency, lower throughput
  - Higher values: Higher latency, higher throughput
  - Recommended: `20-100ms`

- `adaptive_batching` (bool): Dynamically adjust batch size
  - Default: `true`
  - Adapts to system load and latency targets

- `min_batch_size` (int): Minimum requests to trigger batching
  - Default: `1`
  - Set higher to ensure minimum batch efficiency

**Use Cases:**

- **High-throughput serving**: Maximize GPU utilization
- **Variable load**: Adaptive batching handles traffic spikes
- **Latency-sensitive**: Tune timeout for latency requirements

### KV Cache Configuration

KV cache stores transformer key-value pairs to reduce redundant computation.

```yaml
optimization:
  cache:
    enabled: true
    max_memory_mb: 4096
    eviction_policy: lru
```

**Parameters:**

- `enabled` (bool): Enable KV caching
  - Default: `true`
  - Recommended for multi-turn conversations

- `max_memory_mb` (int): Maximum cache memory in MB
  - Default: `4096` (4GB)
  - Recommended: 10-20% of GPU memory
  - Example: 8GB for 40GB GPU

- `eviction_policy` (str): Cache eviction strategy
  - Default: `lru` (Least Recently Used)
  - Options: `lru`, `fifo`

**Use Cases:**

- **Multi-turn conversations**: Reuse context from previous turns
- **Repeated queries**: Cache common prefixes
- **Long contexts**: Avoid recomputing long prompts

### Auto-Tuning Configuration

Auto-tuner automatically adjusts performance parameters based on workload.

```yaml
optimization:
  tuner:
    enabled: false
    observation_window_seconds: 300
    tuning_interval_seconds: 60
    enable_batch_size_tuning: true
    enable_timeout_tuning: true
    enable_cache_size_tuning: true
```

**Parameters:**

- `enabled` (bool): Enable auto-tuning
  - Default: `false`
  - **Warning**: Test thoroughly before enabling in production

- `observation_window_seconds` (int): Performance data collection window
  - Default: `300` (5 minutes)
  - Longer windows provide more stable tuning

- `tuning_interval_seconds` (int): Frequency of tuning adjustments
  - Default: `60` (1 minute)
  - Balance between responsiveness and stability

- `enable_batch_size_tuning` (bool): Auto-adjust batch sizes
  - Default: `true`

- `enable_timeout_tuning` (bool): Auto-adjust timeouts
  - Default: `true`

- `enable_cache_size_tuning` (bool): Auto-adjust cache sizes
  - Default: `true`

**Use Cases:**

- **Variable workloads**: Adapt to changing traffic patterns
- **Performance optimization**: Automatic parameter tuning
- **Reduced manual tuning**: System learns optimal settings

## Monitoring Configuration

### Prometheus Metrics

Export performance metrics in Prometheus format.

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
    path: "/metrics"
```

**Parameters:**

- `enabled` (bool): Enable Prometheus exporter
  - Default: `true`

- `port` (int): HTTP port for metrics endpoint
  - Default: `9090`
  - Ensure port is not in use

- `host` (str): Bind address
  - Default: `"0.0.0.0"` (all interfaces)
  - Use `"127.0.0.1"` for localhost only

- `path` (str): Metrics endpoint path
  - Default: `"/metrics"`

**Exposed Metrics:**

- `inference_latency_seconds{model, engine}`: Inference latency histogram
- `inference_requests_total{model, engine, status}`: Request counter
- `throughput_requests_per_second{model}`: Current throughput gauge
- `gpu_memory_used_bytes{gpu_id}`: GPU memory usage
- `cpu_usage_percent`: CPU utilization
- `kv_cache_hit_rate{model}`: Cache hit rate
- `batch_size{model}`: Batch size histogram

### OpenTelemetry Tracing

Distributed tracing for request flow analysis.

```yaml
monitoring:
  tracing:
    enabled: true
    endpoint: "http://localhost:4317"
    sample_rate: 1.0
    service_name: "muai-orchestration"
    export_timeout_ms: 30000
```

**Parameters:**

- `enabled` (bool): Enable distributed tracing
  - Default: `true`

- `endpoint` (str): OTLP collector endpoint
  - Default: `"http://localhost:4317"`
  - Supports Jaeger, Zipkin, etc.

- `sample_rate` (float): Fraction of requests to trace
  - Default: `1.0` (100%)
  - Range: `0.0` to `1.0`
  - Use `0.1` for 10% sampling in high-traffic scenarios

- `service_name` (str): Service identifier in traces
  - Default: `"muai-orchestration"`

- `export_timeout_ms` (int): Trace export timeout
  - Default: `30000` (30 seconds)

**Trace Spans:**

- Root span: Full request lifecycle
- Workflow spans: Individual workflow execution
- Inference spans: Model inference operations
- Tool spans: External tool calls

### Anomaly Detection

Detect performance anomalies and trigger alerts.

```yaml
monitoring:
  anomaly:
    enabled: true
    latency_threshold_ms: 1000.0
    error_rate_threshold: 0.05
    memory_threshold_percent: 90.0
    throughput_threshold_rps: 1.0
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
    webhook_url: null
```

**Parameters:**

- `enabled` (bool): Enable anomaly detection
  - Default: `true`

- `latency_threshold_ms` (float): Latency alert threshold
  - Default: `1000.0` (1 second)
  - Triggers alert when exceeded

- `error_rate_threshold` (float): Error rate alert threshold
  - Default: `0.05` (5%)
  - Range: `0.0` to `1.0`

- `memory_threshold_percent` (float): Memory usage alert threshold
  - Default: `90.0` (90%)
  - Range: `0` to `100`

- `throughput_threshold_rps` (float): Minimum throughput threshold
  - Default: `1.0` (1 request/second)
  - Triggers alert when throughput drops below

- `alert_rate_limit_seconds` (int): Minimum time between alerts
  - Default: `300` (5 minutes)
  - Prevents alert storms

- `alert_destinations` (list): Where to send alerts
  - Options: `log`, `webhook`, `alertmanager`
  - Default: `[log]`

- `webhook_url` (str|null): Webhook URL for alerts
  - Required when `webhook` in destinations

### Inference Server

Long-running server mode for production deployments.

```yaml
monitoring:
  server:
    enabled: false
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 100
    preload_models:
      - qwen_chat
      - t5_summarizer
    graceful_shutdown_timeout: 30
    health_check_path: "/health"
    readiness_check_path: "/ready"
```

**Parameters:**

- `enabled` (bool): Enable server mode
  - Default: `false`
  - Enable for production deployments

- `host` (str): Bind address
  - Default: `"0.0.0.0"`

- `port` (int): HTTP port
  - Default: `8000`

- `queue_capacity` (int): Maximum queued requests
  - Default: `100`
  - Requests exceeding capacity are rejected

- `preload_models` (list): Models to load at startup
  - Default: `[]`
  - Reduces first-request latency

- `graceful_shutdown_timeout` (int): Shutdown timeout in seconds
  - Default: `30`
  - Time to complete pending requests before shutdown

- `health_check_path` (str): Health check endpoint
  - Default: `"/health"`

- `readiness_check_path` (str): Readiness check endpoint
  - Default: `"/ready"`

## Environment Variables

All configuration options can be overridden with environment variables.

### Optimization Variables

```bash
# Global optimization
export MUAI_OPT_ENABLED=true

# vLLM
export MUAI_OPT_VLLM_ENABLED=true
export MUAI_OPT_VLLM_TENSOR_PARALLEL=2
export MUAI_OPT_VLLM_DTYPE=fp16
export MUAI_OPT_VLLM_GPU_MEMORY=0.9

# DeepSpeed
export MUAI_OPT_DEEPSPEED_ENABLED=true
export MUAI_OPT_DEEPSPEED_TENSOR_PARALLEL=2

# ONNX
export MUAI_OPT_ONNX_ENABLED=true

# Batching
export MUAI_OPT_BATCHER_ENABLED=true
export MUAI_OPT_BATCHER_MAX_SIZE=32
export MUAI_OPT_BATCHER_TIMEOUT=50

# Cache
export MUAI_OPT_CACHE_ENABLED=true
export MUAI_OPT_CACHE_MAX_MEMORY=4096

# Auto-tuning
export MUAI_OPT_TUNER_ENABLED=false
```

### Monitoring Variables

```bash
# Global monitoring
export MUAI_MON_ENABLED=true

# Prometheus
export MUAI_MON_PROMETHEUS_ENABLED=true
export MUAI_MON_PROMETHEUS_PORT=9090

# Tracing
export MUAI_MON_TRACING_ENABLED=true
export MUAI_MON_TRACING_ENDPOINT=http://localhost:4317
export MUAI_MON_TRACING_SAMPLE_RATE=1.0

# Anomaly detection
export MUAI_MON_ANOMALY_ENABLED=true
export MUAI_MON_ANOMALY_LATENCY_THRESHOLD=1000.0

# Server
export MUAI_MON_SERVER_ENABLED=false
export MUAI_MON_SERVER_PORT=8000
export MUAI_MON_SERVER_QUEUE_CAPACITY=100
export MUAI_MON_SERVER_PRELOAD_MODELS=qwen_chat,t5_summarizer
```

## Configuration Examples

See [Configuration Examples](./optimization_configuration_examples.md) for complete scenario-based configurations.

## Best Practices

### 1. Start with Defaults

Begin with the example configuration and adjust based on your needs:

```bash
cp config/optimization.example.yaml config/optimization.yaml
```

### 2. Enable Features Incrementally

Don't enable all features at once. Test each feature individually:

1. Start with basic optimization (vLLM or DeepSpeed)
2. Add monitoring (Prometheus)
3. Enable batching
4. Add caching
5. Enable auto-tuning (after thorough testing)

### 3. Monitor Resource Usage

Always monitor GPU memory, CPU, and system resources:

- Use Prometheus metrics to track resource usage
- Set appropriate memory thresholds
- Configure alerts for resource exhaustion

### 4. Tune for Your Workload

Different workloads require different configurations:

- **High throughput**: Large batch sizes, longer timeouts
- **Low latency**: Small batch sizes, short timeouts
- **Mixed workload**: Enable adaptive batching

### 5. Test Before Production

Always test configuration changes in a staging environment:

- Verify performance improvements
- Check for memory leaks
- Validate error handling
- Test failover scenarios

### 6. Use Environment Variables for Deployment

Use environment variables for deployment-specific settings:

- Different ports per environment
- Environment-specific endpoints
- Feature flags for gradual rollout

### 7. Document Your Configuration

Maintain documentation for your specific configuration:

- Why certain values were chosen
- Performance benchmarks
- Known issues and workarounds

### 8. Regular Review

Periodically review and update configuration:

- Check for new optimization features
- Update thresholds based on actual usage
- Remove unused features

### 9. Security Considerations

- Bind monitoring endpoints to localhost in production
- Use authentication for metrics endpoints
- Secure webhook URLs
- Limit queue capacity to prevent DoS

### 10. Backup Configuration

Version control your configuration files:

```bash
git add config/optimization.yaml
git commit -m "Update optimization configuration"
```

## Troubleshooting

### Common Issues

**Issue**: vLLM fails to initialize
- **Solution**: Check GPU availability and CUDA version
- **Check**: `nvidia-smi` and `pip show vllm`

**Issue**: High memory usage
- **Solution**: Reduce `max_batch_size` or `gpu_memory_utilization`
- **Check**: Prometheus `gpu_memory_used_bytes` metric

**Issue**: High latency
- **Solution**: Reduce `batch_timeout_ms` or disable batching
- **Check**: Prometheus `inference_latency_seconds` metric

**Issue**: Metrics not appearing
- **Solution**: Verify Prometheus endpoint is accessible
- **Check**: `curl http://localhost:9090/metrics`

**Issue**: Traces not exported
- **Solution**: Verify OTLP collector is running
- **Check**: Collector logs and endpoint connectivity

## Next Steps

- [Configuration Examples](./optimization_configuration_examples.md) - Scenario-based configurations
- [Migration Guide](./optimization_migration_guide.md) - Migrate from existing system
- [Deployment Guide](./optimization_deployment_guide.md) - Deploy in different environments
- [API Reference](./api_reference.md) - Programmatic configuration
