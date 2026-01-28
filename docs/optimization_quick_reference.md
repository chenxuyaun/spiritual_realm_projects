# Quick Reference: Optimization and Monitoring

## Configuration Quick Reference

### Enable/Disable Features

```yaml
# Enable all features
optimization:
  enabled: true
monitoring:
  enabled: true

# Disable all features
optimization:
  enabled: false
monitoring:
  enabled: false
```

### Engine Selection

```yaml
# vLLM only
optimization:
  engine_preference: [vllm, pytorch]

# DeepSpeed only
optimization:
  engine_preference: [deepspeed, pytorch]

# ONNX only
optimization:
  engine_preference: [onnx, pytorch]

# PyTorch only (no optimization)
optimization:
  engine_preference: [pytorch]
```

### Common Configurations

#### Development
```yaml
optimization:
  enabled: true
  engine_preference: [pytorch]
  batcher:
    enabled: false
  cache:
    enabled: true
    max_memory_mb: 1024
monitoring:
  prometheus:
    enabled: true
    port: 9090
    host: "127.0.0.1"
  tracing:
    enabled: false
  anomaly:
    enabled: false
```

#### Production (Single GPU)
```yaml
optimization:
  enabled: true
  engine_preference: [vllm, pytorch]
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: fp16
    gpu_memory_utilization: 0.90
  batcher:
    enabled: true
    max_batch_size: 16
    batch_timeout_ms: 50
  cache:
    enabled: true
    max_memory_mb: 2048
monitoring:
  prometheus:
    enabled: true
    port: 9090
  tracing:
    enabled: true
    sample_rate: 0.1
  anomaly:
    enabled: true
    latency_threshold_ms: 1000.0
  server:
    enabled: true
    port: 8000
```

#### Production (Multi-GPU)
```yaml
optimization:
  enabled: true
  engine_preference: [vllm, deepspeed, pytorch]
  vllm:
    enabled: true
    tensor_parallel_size: 4
    dtype: fp16
    gpu_memory_utilization: 0.85
  batcher:
    enabled: true
    max_batch_size: 64
    batch_timeout_ms: 100
  cache:
    enabled: true
    max_memory_mb: 8192
  tuner:
    enabled: true
monitoring:
  prometheus:
    enabled: true
  tracing:
    enabled: true
    sample_rate: 0.05
  anomaly:
    enabled: true
  server:
    enabled: true
    queue_capacity: 200
```

## Environment Variables Quick Reference

### Optimization

```bash
# Global
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

### Monitoring

```bash
# Global
export MUAI_MON_ENABLED=true

# Prometheus
export MUAI_MON_PROMETHEUS_ENABLED=true
export MUAI_MON_PROMETHEUS_PORT=9090

# Tracing
export MUAI_MON_TRACING_ENABLED=true
export MUAI_MON_TRACING_ENDPOINT=http://localhost:4317
export MUAI_MON_TRACING_SAMPLE_RATE=0.1

# Anomaly
export MUAI_MON_ANOMALY_ENABLED=true
export MUAI_MON_ANOMALY_LATENCY_THRESHOLD=1000.0

# Server
export MUAI_MON_SERVER_ENABLED=true
export MUAI_MON_SERVER_PORT=8000
export MUAI_MON_SERVER_QUEUE_CAPACITY=100
export MUAI_MON_SERVER_PRELOAD_MODELS=qwen_chat,t5_summarizer
```

## Command Reference

### Installation

```bash
# Basic installation
pip install -r requirements.txt

# With vLLM
pip install vllm

# With DeepSpeed
pip install deepspeed

# With ONNX Runtime (GPU)
pip install onnxruntime-gpu

# With ONNX Runtime (CPU)
pip install onnxruntime

# With monitoring
pip install prometheus-client opentelemetry-api opentelemetry-sdk
```

### Running

```bash
# CLI mode
python -m mm_orch.main "query"

# Server mode
python -m mm_orch.main --server

# With specific config
python -m mm_orch.main --config config/optimization.yaml --server

# With environment variables
MUAI_OPT_VLLM_ENABLED=true python -m mm_orch.main --server
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:8000/health

# Readiness endpoint
curl http://localhost:8000/ready

# Metrics endpoint
curl http://localhost:9090/metrics

# Check specific metric
curl http://localhost:9090/metrics | grep inference_latency
```

### Docker

```bash
# Build
docker build -t muai-orchestration .

# Run
docker run -d --gpus all -p 8000:8000 -p 9090:9090 muai-orchestration

# Run with config
docker run -d --gpus all \
  -p 8000:8000 -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  muai-orchestration

# Run with environment variables
docker run -d --gpus all \
  -p 8000:8000 -p 9090:9090 \
  -e MUAI_OPT_VLLM_ENABLED=true \
  -e MUAI_MON_PROMETHEUS_ENABLED=true \
  muai-orchestration

# Docker Compose
docker-compose up -d

# View logs
docker logs -f muai
```

### Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check status
kubectl get all -n muai-system

# Check pods
kubectl get pods -n muai-system

# View logs
kubectl logs -f deployment/muai-orchestration -n muai-system

# Port forward
kubectl port-forward svc/muai-orchestration 8000:80 -n muai-system

# Scale
kubectl scale deployment muai-orchestration --replicas=5 -n muai-system

# Delete
kubectl delete -f k8s/
```

## Metrics Reference

### Inference Metrics

```promql
# Request rate
rate(inference_requests_total[5m])

# Latency (p50, p95, p99)
histogram_quantile(0.50, rate(inference_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))

# Error rate
rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m])

# Throughput
rate(inference_requests_total{status="success"}[5m])
```

### Resource Metrics

```promql
# GPU memory usage (GB)
gpu_memory_used_bytes / 1024 / 1024 / 1024

# GPU memory utilization (%)
gpu_memory_used_bytes / gpu_memory_total_bytes * 100

# CPU usage
cpu_usage_percent

# Cache hit rate
kv_cache_hit_rate
```

### Batch Metrics

```promql
# Average batch size
avg(batch_size)

# Batch size distribution
histogram_quantile(0.95, rate(batch_size_bucket[5m]))
```

## Troubleshooting Quick Reference

### Check System Status

```bash
# GPU availability
nvidia-smi

# Python packages
pip list | grep -E "vllm|deepspeed|onnx|prometheus|opentelemetry"

# Process status
ps aux | grep mm_orch

# Port usage
lsof -i :8000
lsof -i :9090

# Disk space
df -h

# Memory usage
free -h
```

### Common Fixes

```bash
# Restart service
systemctl restart muai-orchestration

# Clear cache
rm -rf /root/.cache/huggingface/*

# Reset configuration
cp config/optimization.example.yaml config/optimization.yaml

# Disable optimization
export MUAI_OPT_ENABLED=false

# Disable monitoring
export MUAI_MON_ENABLED=false

# Reduce memory usage
export MUAI_OPT_VLLM_GPU_MEMORY=0.7
export MUAI_OPT_BATCHER_MAX_SIZE=8
```

### Log Analysis

```bash
# View recent logs
tail -f logs/muai.log

# Search for errors
grep ERROR logs/muai.log

# Search for warnings
grep WARN logs/muai.log

# Filter by component
grep "optimization" logs/muai.log
grep "monitoring" logs/muai.log

# Count errors
grep -c ERROR logs/muai.log
```

## Performance Tuning Quick Reference

### For High Throughput

```yaml
optimization:
  batcher:
    max_batch_size: 128
    batch_timeout_ms: 200
  vllm:
    gpu_memory_utilization: 0.95
```

### For Low Latency

```yaml
optimization:
  batcher:
    max_batch_size: 8
    batch_timeout_ms: 10
  vllm:
    gpu_memory_utilization: 0.85
```

### For Memory Efficiency

```yaml
optimization:
  vllm:
    gpu_memory_utilization: 0.70
  batcher:
    max_batch_size: 16
  cache:
    max_memory_mb: 1024
```

### For Multi-Turn Conversations

```yaml
optimization:
  cache:
    enabled: true
    max_memory_mb: 4096
    eviction_policy: lru
```

## Alert Thresholds Reference

### Conservative (Low Alert Volume)

```yaml
monitoring:
  anomaly:
    latency_threshold_ms: 2000.0
    error_rate_threshold: 0.10
    memory_threshold_percent: 95.0
    throughput_threshold_rps: 0.5
```

### Moderate (Balanced)

```yaml
monitoring:
  anomaly:
    latency_threshold_ms: 1000.0
    error_rate_threshold: 0.05
    memory_threshold_percent: 90.0
    throughput_threshold_rps: 1.0
```

### Aggressive (High Alert Volume)

```yaml
monitoring:
  anomaly:
    latency_threshold_ms: 500.0
    error_rate_threshold: 0.02
    memory_threshold_percent: 85.0
    throughput_threshold_rps: 2.0
```

## Resource Recommendations

### By GPU Type

**NVIDIA T4 (15GB)**
```yaml
vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
batcher:
  max_batch_size: 16
cache:
  max_memory_mb: 2048
```

**NVIDIA A10 (24GB)**
```yaml
vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
batcher:
  max_batch_size: 32
cache:
  max_memory_mb: 4096
```

**NVIDIA A100 (40GB)**
```yaml
vllm:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
batcher:
  max_batch_size: 64
cache:
  max_memory_mb: 8192
```

**Multi-GPU (4x A100)**
```yaml
vllm:
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.85
batcher:
  max_batch_size: 128
cache:
  max_memory_mb: 16384
```

## Links

- [Full Configuration Guide](./optimization_configuration_guide.md)
- [Configuration Examples](./optimization_configuration_examples.md)
- [Migration Guide](./optimization_migration_guide.md)
- [Deployment Guide](./optimization_deployment_guide.md)
- [Main README](./optimization_README.md)
