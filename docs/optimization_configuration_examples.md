# Optimization Configuration Examples

This document provides complete configuration examples for common deployment scenarios.

## Table of Contents

1. [Development Environment](#development-environment)
2. [Single GPU Production](#single-gpu-production)
3. [Multi-GPU Production](#multi-gpu-production)
4. [CPU-Only Deployment](#cpu-only-deployment)
5. [High-Throughput Serving](#high-throughput-serving)
6. [Low-Latency Serving](#low-latency-serving)
7. [Kubernetes Deployment](#kubernetes-deployment)
8. [Docker Deployment](#docker-deployment)

## Development Environment

Minimal configuration for local development and testing.

```yaml
# config/optimization.yaml - Development
optimization:
  enabled: true
  engine_preference: [pytorch]  # Use standard PyTorch only
  fallback_on_error: true
  
  vllm:
    enabled: false  # Disable for faster startup
  
  deepspeed:
    enabled: false
  
  onnx:
    enabled: false
  
  batcher:
    enabled: false  # Process requests individually
    max_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: 1024  # 1GB cache
    eviction_policy: lru
  
  tuner:
    enabled: false  # No auto-tuning in dev

monitoring:
  enabled: true
  collection_interval_seconds: 60
  
  prometheus:
    enabled: true
    port: 9090
    host: "127.0.0.1"  # Localhost only
  
  tracing:
    enabled: false  # Disable tracing in dev
  
  anomaly:
    enabled: false  # Disable alerts in dev
  
  server:
    enabled: false  # Use CLI mode
```

**Usage:**

```bash
# Run with development config
python -m mm_orch.main "What is machine learning?"

# Check metrics
curl http://localhost:9090/metrics
```

## Single GPU Production

Optimized configuration for single GPU production deployment.

```yaml
# config/optimization.yaml - Single GPU Production
optimization:
  enabled: true
  engine_preference: [vllm, onnx, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: fp16
    max_model_len: null
    gpu_memory_utilization: 0.90
    swap_space: 4
  
  deepspeed:
    enabled: false  # Not needed for single GPU
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: 16  # Adjust based on GPU memory
    batch_timeout_ms: 50
    adaptive_batching: true
    min_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: 2048  # 2GB cache
    eviction_policy: lru
  
  tuner:
    enabled: false  # Enable after testing
    observation_window_seconds: 300
    tuning_interval_seconds: 60

monitoring:
  enabled: true
  collection_interval_seconds: 30
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://localhost:4317"
    sample_rate: 0.1  # 10% sampling
    service_name: "muai-prod-single"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 2000.0
    error_rate_threshold: 0.05
    memory_threshold_percent: 85.0
    throughput_threshold_rps: 0.5
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
      - webhook
    webhook_url: "https://your-webhook-url.com/alerts"
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 50
    preload_models:
      - qwen_chat
      - t5_summarizer
    graceful_shutdown_timeout: 30
```

**Deployment:**

```bash
# Start server
python -m mm_orch.main --server

# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:9090/metrics
```

## Multi-GPU Production

Configuration for multi-GPU production deployment with tensor parallelism.

```yaml
# config/optimization.yaml - Multi-GPU Production (4 GPUs)
optimization:
  enabled: true
  engine_preference: [vllm, deepspeed, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: 4  # Use all 4 GPUs
    dtype: fp16
    max_model_len: null
    gpu_memory_utilization: 0.85  # Conservative for stability
    swap_space: 8
  
  deepspeed:
    enabled: true
    tensor_parallel: 4
    pipeline_parallel: 1
    dtype: fp16
    replace_with_kernel_inject: true
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - TensorrtExecutionProvider
      - CPUExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: 64  # Larger batches for multi-GPU
    batch_timeout_ms: 100
    adaptive_batching: true
    min_batch_size: 4
  
  cache:
    enabled: true
    max_memory_mb: 8192  # 8GB cache
    eviction_policy: lru
  
  tuner:
    enabled: true  # Enable auto-tuning
    observation_window_seconds: 300
    tuning_interval_seconds: 120
    enable_batch_size_tuning: true
    enable_timeout_tuning: true
    enable_cache_size_tuning: true

monitoring:
  enabled: true
  collection_interval_seconds: 15  # More frequent collection
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://jaeger:4317"
    sample_rate: 0.05  # 5% sampling for high traffic
    service_name: "muai-prod-multi"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 1500.0
    error_rate_threshold: 0.03
    memory_threshold_percent: 90.0
    throughput_threshold_rps: 5.0
    alert_rate_limit_seconds: 180
    alert_destinations:
      - log
      - webhook
      - alertmanager
    webhook_url: "https://your-webhook-url.com/alerts"
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 200
    preload_models:
      - qwen_chat
      - t5_summarizer
      - minilm_embedder
    graceful_shutdown_timeout: 60
```

**Environment Variables:**

```bash
# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start server
python -m mm_orch.main --server
```

## CPU-Only Deployment

Configuration for CPU-only environments (no GPU available).

```yaml
# config/optimization.yaml - CPU Only
optimization:
  enabled: true
  engine_preference: [onnx, pytorch]  # ONNX optimized for CPU
  fallback_on_error: true
  
  vllm:
    enabled: false  # Requires GPU
  
  deepspeed:
    enabled: false  # Requires GPU
  
  onnx:
    enabled: true
    execution_providers:
      - CPUExecutionProvider
      - OpenVINOExecutionProvider  # If available
    optimization_level: all
    enable_quantization: true  # Quantization helps on CPU
  
  batcher:
    enabled: true
    max_batch_size: 4  # Smaller batches for CPU
    batch_timeout_ms: 200
    adaptive_batching: true
    min_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: 2048
    eviction_policy: lru
  
  tuner:
    enabled: false

monitoring:
  enabled: true
  collection_interval_seconds: 60
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://localhost:4317"
    sample_rate: 1.0
    service_name: "muai-cpu"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 5000.0  # Higher threshold for CPU
    error_rate_threshold: 0.05
    memory_threshold_percent: 80.0
    throughput_threshold_rps: 0.1
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 20  # Smaller queue for CPU
    preload_models:
      - t5_summarizer  # Smaller models only
    graceful_shutdown_timeout: 30
```

## High-Throughput Serving

Optimized for maximum throughput with acceptable latency.

```yaml
# config/optimization.yaml - High Throughput
optimization:
  enabled: true
  engine_preference: [vllm, deepspeed, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: 2
    dtype: fp16
    max_model_len: null
    gpu_memory_utilization: 0.95  # Maximize GPU usage
    swap_space: 8
  
  deepspeed:
    enabled: true
    tensor_parallel: 2
    pipeline_parallel: 1
    dtype: fp16
    replace_with_kernel_inject: true
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - TensorrtExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: 128  # Very large batches
    batch_timeout_ms: 200  # Longer timeout for batch formation
    adaptive_batching: true
    min_batch_size: 8  # Prefer larger batches
  
  cache:
    enabled: true
    max_memory_mb: 4096
    eviction_policy: lru
  
  tuner:
    enabled: true
    observation_window_seconds: 600
    tuning_interval_seconds: 180
    enable_batch_size_tuning: true
    enable_timeout_tuning: true
    enable_cache_size_tuning: true

monitoring:
  enabled: true
  collection_interval_seconds: 30
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://localhost:4317"
    sample_rate: 0.01  # 1% sampling for very high traffic
    service_name: "muai-high-throughput"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 3000.0  # Accept higher latency
    error_rate_threshold: 0.02
    memory_threshold_percent: 95.0
    throughput_threshold_rps: 10.0  # High throughput target
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
      - alertmanager
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 500  # Large queue
    preload_models:
      - qwen_chat
      - t5_summarizer
    graceful_shutdown_timeout: 120
```

## Low-Latency Serving

Optimized for minimum latency with acceptable throughput.

```yaml
# config/optimization.yaml - Low Latency
optimization:
  enabled: true
  engine_preference: [vllm, onnx, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: fp16
    max_model_len: 2048  # Limit sequence length
    gpu_memory_utilization: 0.85
    swap_space: 2
  
  deepspeed:
    enabled: false  # Skip for lower latency
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - TensorrtExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: 8  # Small batches
    batch_timeout_ms: 10  # Very short timeout
    adaptive_batching: true
    min_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: 2048
    eviction_policy: lru
  
  tuner:
    enabled: false  # Disable for predictable latency

monitoring:
  enabled: true
  collection_interval_seconds: 15
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://localhost:4317"
    sample_rate: 0.1
    service_name: "muai-low-latency"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 500.0  # Strict latency requirement
    error_rate_threshold: 0.05
    memory_threshold_percent: 85.0
    throughput_threshold_rps: 1.0
    alert_rate_limit_seconds: 180
    alert_destinations:
      - log
      - webhook
    webhook_url: "https://your-webhook-url.com/alerts"
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 50
    preload_models:
      - qwen_chat
    graceful_shutdown_timeout: 30
```

## Kubernetes Deployment

Configuration for Kubernetes deployment with service discovery.

```yaml
# config/optimization.yaml - Kubernetes
optimization:
  enabled: true
  engine_preference: [vllm, deepspeed, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: ${TENSOR_PARALLEL_SIZE:1}  # From env
    dtype: fp16
    max_model_len: null
    gpu_memory_utilization: 0.90
    swap_space: 4
  
  deepspeed:
    enabled: true
    tensor_parallel: ${TENSOR_PARALLEL_SIZE:1}
    pipeline_parallel: 1
    dtype: fp16
    replace_with_kernel_inject: true
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: ${MAX_BATCH_SIZE:32}
    batch_timeout_ms: ${BATCH_TIMEOUT_MS:50}
    adaptive_batching: true
    min_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: ${CACHE_MEMORY_MB:4096}
    eviction_policy: lru
  
  tuner:
    enabled: ${AUTO_TUNING_ENABLED:false}
    observation_window_seconds: 300
    tuning_interval_seconds: 60

monitoring:
  enabled: true
  collection_interval_seconds: 30
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"  # Accessible to Prometheus scraper
  
  tracing:
    enabled: true
    endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:http://jaeger-collector:4317}
    sample_rate: ${TRACE_SAMPLE_RATE:0.1}
    service_name: ${SERVICE_NAME:muai-orchestration}
  
  anomaly:
    enabled: true
    latency_threshold_ms: ${LATENCY_THRESHOLD_MS:1000.0}
    error_rate_threshold: 0.05
    memory_threshold_percent: 90.0
    throughput_threshold_rps: 1.0
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
      - alertmanager
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: ${QUEUE_CAPACITY:100}
    preload_models: ${PRELOAD_MODELS:qwen_chat,t5_summarizer}
    graceful_shutdown_timeout: 30
```

**Kubernetes Deployment YAML:**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: muai-orchestration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: muai-orchestration
  template:
    metadata:
      labels:
        app: muai-orchestration
    spec:
      containers:
      - name: muai
        image: muai-orchestration:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: TENSOR_PARALLEL_SIZE
          value: "2"
        - name: MAX_BATCH_SIZE
          value: "32"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:4317"
        - name: SERVICE_NAME
          value: "muai-orchestration"
        resources:
          requests:
            nvidia.com/gpu: 2
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 2
            memory: "32Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: muai-orchestration
spec:
  selector:
    app: muai-orchestration
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

## Docker Deployment

Configuration for Docker container deployment.

```yaml
# config/optimization.yaml - Docker
optimization:
  enabled: true
  engine_preference: [vllm, pytorch]
  fallback_on_error: true
  
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: fp16
    max_model_len: null
    gpu_memory_utilization: 0.90
    swap_space: 4
  
  deepspeed:
    enabled: false
  
  onnx:
    enabled: true
    execution_providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
    optimization_level: all
    enable_quantization: false
  
  batcher:
    enabled: true
    max_batch_size: 16
    batch_timeout_ms: 50
    adaptive_batching: true
    min_batch_size: 1
  
  cache:
    enabled: true
    max_memory_mb: 2048
    eviction_policy: lru
  
  tuner:
    enabled: false

monitoring:
  enabled: true
  collection_interval_seconds: 30
  
  prometheus:
    enabled: true
    port: 9090
    host: "0.0.0.0"
  
  tracing:
    enabled: true
    endpoint: "http://jaeger:4317"
    sample_rate: 0.1
    service_name: "muai-docker"
  
  anomaly:
    enabled: true
    latency_threshold_ms: 1000.0
    error_rate_threshold: 0.05
    memory_threshold_percent: 85.0
    throughput_threshold_rps: 1.0
    alert_rate_limit_seconds: 300
    alert_destinations:
      - log
  
  server:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    queue_capacity: 50
    preload_models:
      - qwen_chat
      - t5_summarizer
    graceful_shutdown_timeout: 30
```

**Dockerfile:**

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "-m", "mm_orch.main", "--server"]
```

**Docker Compose:**

```yaml
# docker-compose.yml
version: '3.8'

services:
  muai:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - model-cache:/root/.cache/huggingface
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MUAI_OPT_VLLM_ENABLED=true
      - MUAI_MON_PROMETHEUS_ENABLED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - muai-network
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    networks:
      - muai-network
  
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
    networks:
      - muai-network

volumes:
  model-cache:
  prometheus-data:

networks:
  muai-network:
    driver: bridge
```

**Run with Docker:**

```bash
# Build image
docker build -t muai-orchestration .

# Run container
docker run -d \
  --name muai \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  muai-orchestration

# Or use docker-compose
docker-compose up -d

# Check logs
docker logs -f muai

# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:9090/metrics
```

## Next Steps

- [Configuration Guide](./optimization_configuration_guide.md) - Detailed parameter documentation
- [Migration Guide](./optimization_migration_guide.md) - Migrate from existing system
- [Deployment Guide](./optimization_deployment_guide.md) - Production deployment best practices
