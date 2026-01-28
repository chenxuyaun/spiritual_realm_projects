# Deployment Guide: Optimization and Monitoring

This guide covers deploying the MuAI Multi-Model Orchestration System with optimization and monitoring features in various environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployments](#cloud-deployments)
7. [Monitoring Setup](#monitoring-setup)
8. [Production Best Practices](#production-best-practices)

## Deployment Overview

### Deployment Architectures

**Single Instance**
- Simple deployment on single server
- Suitable for: Development, small-scale production
- Resources: 1 GPU, 16GB RAM minimum

**Multi-Instance with Load Balancer**
- Multiple instances behind load balancer
- Suitable for: Medium-scale production
- Resources: Multiple GPUs, horizontal scaling

**Kubernetes Cluster**
- Container orchestration with auto-scaling
- Suitable for: Large-scale production
- Resources: GPU node pools, auto-scaling

### Component Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer / Ingress         │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼─────────┐
│  MuAI Instance │  │  MuAI Instance │
│  (Port 8000)   │  │  (Port 8000)   │
│  Metrics: 9090 │  │  Metrics: 9090 │
└───────┬────────┘  └──────┬─────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼──────────┐
        │   Prometheus       │
        │   (Port 9091)      │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │   Grafana          │
        │   (Port 3000)      │
        └────────────────────┘
                  │
        ┌─────────▼──────────┐
        │   Jaeger/Zipkin    │
        │   (Port 16686)     │
        └────────────────────┘
```

## Prerequisites

### Hardware Requirements

**Minimum (Development)**
- CPU: 4 cores
- RAM: 16GB
- GPU: NVIDIA T4 (15GB VRAM) or equivalent
- Storage: 100GB SSD

**Recommended (Production)**
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA A100 (40GB VRAM) or multiple T4s
- Storage: 500GB+ SSD

### Software Requirements

**Operating System**
- Ubuntu 20.04+ (recommended)
- CentOS 8+
- Windows Server 2019+ (with WSL2)

**CUDA and Drivers**
- NVIDIA Driver: 525.x or newer
- CUDA: 11.8 or 12.1
- cuDNN: 8.x

**Python**
- Python 3.8, 3.9, 3.10, or 3.11
- pip 21.0+
- virtualenv or conda

**Container Runtime (for Docker/K8s)**
- Docker 20.10+
- NVIDIA Container Toolkit
- Kubernetes 1.24+ (for K8s deployment)

### Network Requirements

**Ports**
- 8000: HTTP API
- 9090: Prometheus metrics
- 4317: OpenTelemetry OTLP (gRPC)
- 4318: OpenTelemetry OTLP (HTTP)

**Firewall Rules**
```bash
# Allow HTTP API
sudo ufw allow 8000/tcp

# Allow Prometheus (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 9090

# Allow OTLP (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 4317
```

## Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/muai-orchestration.git
cd muai-orchestration

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install optimization dependencies (optional)
pip install vllm deepspeed onnxruntime-gpu

# Install monitoring dependencies
pip install prometheus-client opentelemetry-api opentelemetry-sdk

# Copy configuration
cp config/optimization.example.yaml config/optimization.yaml

# Edit configuration
nano config/optimization.yaml
```

### Running Locally

```bash
# Activate environment
source .venv/bin/activate

# Run CLI mode
python -m mm_orch.main "What is machine learning?"

# Run server mode
python -m mm_orch.main --server

# Check health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:9090/metrics
```

### Development with Hot Reload

```bash
# Install development dependencies
pip install watchdog

# Run with auto-reload
watchmedo auto-restart \
  --directory=./mm_orch \
  --pattern=*.py \
  --recursive \
  -- python -m mm_orch.main --server
```

## Docker Deployment

### Build Docker Image

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/vector_db data/chat_history logs

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python3", "-m", "mm_orch.main", "--server"]
```

```bash
# Build image
docker build -t muai-orchestration:latest .

# Tag for registry
docker tag muai-orchestration:latest your-registry/muai-orchestration:v1.0.0

# Push to registry
docker push your-registry/muai-orchestration:v1.0.0
```

### Run Docker Container

```bash
# Run with GPU
docker run -d \
  --name muai \
  --gpus all \
  -p 8000:8000 \
  -p 9090:9090 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e MUAI_OPT_VLLM_ENABLED=true \
  -e MUAI_MON_PROMETHEUS_ENABLED=true \
  --restart unless-stopped \
  muai-orchestration:latest

# Check logs
docker logs -f muai

# Check health
curl http://localhost:8000/health

# Stop container
docker stop muai

# Remove container
docker rm muai
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  muai:
    image: muai-orchestration:latest
    container_name: muai
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
      - model-cache:/root/.cache/huggingface
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MUAI_OPT_VLLM_ENABLED=true
      - MUAI_MON_PROMETHEUS_ENABLED=true
      - MUAI_MON_TRACING_ENDPOINT=http://jaeger:4317
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - muai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    networks:
      - muai-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - muai-network
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    networks:
      - muai-network
    restart: unless-stopped

volumes:
  model-cache:
  prometheus-data:
  grafana-data:

networks:
  muai-network:
    driver: bridge
```

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f muai

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### Create Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: muai-system
  labels:
    name: muai-system
```

```bash
kubectl apply -f k8s/namespace.yaml
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: muai-config
  namespace: muai-system
data:
  optimization.yaml: |
    optimization:
      enabled: true
      engine_preference: [vllm, pytorch]
      vllm:
        enabled: true
        tensor_parallel_size: 2
        dtype: fp16
        gpu_memory_utilization: 0.90
      batcher:
        enabled: true
        max_batch_size: 32
        batch_timeout_ms: 50
      cache:
        enabled: true
        max_memory_mb: 4096
    monitoring:
      enabled: true
      prometheus:
        enabled: true
        port: 9090
      tracing:
        enabled: true
        endpoint: "http://jaeger-collector:4317"
        sample_rate: 0.1
      server:
        enabled: true
        port: 8000
        queue_capacity: 100
```

```bash
kubectl apply -f k8s/configmap.yaml
```

### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: muai-orchestration
  namespace: muai-system
  labels:
    app: muai-orchestration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: muai-orchestration
  template:
    metadata:
      labels:
        app: muai-orchestration
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: muai
        image: your-registry/muai-orchestration:v1.0.0
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        - name: MUAI_OPT_VLLM_TENSOR_PARALLEL
          value: "2"
        - name: MUAI_MON_TRACING_ENDPOINT
          value: "http://jaeger-collector:4317"
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: data
          mountPath: /app/data
        - name: model-cache
          mountPath: /root/.cache/huggingface
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
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: config
        configMap:
          name: muai-config
      - name: data
        persistentVolumeClaim:
          claimName: muai-data-pvc
      - name: model-cache
        persistentVolumeClaim:
          claimName: muai-model-cache-pvc
      nodeSelector:
        accelerator: nvidia-tesla-t4
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

```bash
kubectl apply -f k8s/deployment.yaml
```


### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: muai-orchestration
  namespace: muai-system
  labels:
    app: muai-orchestration
spec:
  type: LoadBalancer
  selector:
    app: muai-orchestration
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
```

```bash
kubectl apply -f k8s/service.yaml
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: muai-orchestration-hpa
  namespace: muai-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: muai-orchestration
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

```bash
kubectl apply -f k8s/hpa.yaml
```

### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: muai-orchestration-ingress
  namespace: muai-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - muai.example.com
    secretName: muai-tls
  rules:
  - host: muai.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: muai-orchestration
            port:
              number: 80
```

```bash
kubectl apply -f k8s/ingress.yaml
```

### Deploy All

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get all -n muai-system

# Check pods
kubectl get pods -n muai-system -w

# Check logs
kubectl logs -f deployment/muai-orchestration -n muai-system

# Get service URL
kubectl get svc muai-orchestration -n muai-system

# Test health endpoint
curl http://<EXTERNAL-IP>/health
```

## Cloud Deployments

### AWS EKS

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name muai-cluster \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.2xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -f k8s/

# Get load balancer URL
kubectl get svc muai-orchestration -n muai-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

### Google GKE

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create muai-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5

# Install NVIDIA driver
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# Deploy application
kubectl apply -f k8s/

# Get load balancer IP
kubectl get svc muai-orchestration -n muai-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Azure AKS

```bash
# Create AKS cluster with GPU nodes
az aks create \
  --resource-group muai-rg \
  --name muai-cluster \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 5 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group muai-rg --name muai-cluster

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy application
kubectl apply -f k8s/

# Get load balancer IP
kubectl get svc muai-orchestration -n muai-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'muai-orchestration'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - muai-system
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "MuAI Orchestration Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(inference_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_used_bytes / 1024 / 1024 / 1024"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "kv_cache_hit_rate"
          }
        ]
      }
    ]
  }
}
```

### Jaeger Setup

```yaml
# k8s/jaeger.yaml
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: muai-system
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch:9200
  collector:
    maxReplicas: 5
    resources:
      limits:
        cpu: 1
        memory: 1Gi
  query:
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
```

## Production Best Practices

### Security

**1. Network Security**
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: muai-network-policy
  namespace: muai-system
spec:
  podSelector:
    matchLabels:
      app: muai-orchestration
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 4317  # OTLP
```

**2. RBAC**
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: muai-sa
  namespace: muai-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: muai-role
  namespace: muai-system
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: muai-rolebinding
  namespace: muai-system
subjects:
- kind: ServiceAccount
  name: muai-sa
  namespace: muai-system
roleRef:
  kind: Role
  name: muai-role
  apiGroup: rbac.authorization.k8s.io
```

**3. Secrets Management**
```bash
# Create secrets
kubectl create secret generic muai-secrets \
  --from-literal=webhook-url=https://your-webhook.com \
  --from-literal=api-key=your-api-key \
  -n muai-system

# Use in deployment
env:
- name: WEBHOOK_URL
  valueFrom:
    secretKeyRef:
      name: muai-secrets
      key: webhook-url
```

### High Availability

**1. Pod Disruption Budget**
```yaml
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: muai-pdb
  namespace: muai-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: muai-orchestration
```

**2. Anti-Affinity**
```yaml
# Add to deployment spec
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - muai-orchestration
        topologyKey: kubernetes.io/hostname
```

### Resource Management

**1. Resource Quotas**
```yaml
# k8s/resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: muai-quota
  namespace: muai-system
spec:
  hard:
    requests.cpu: "32"
    requests.memory: 128Gi
    requests.nvidia.com/gpu: "8"
    limits.cpu: "64"
    limits.memory: 256Gi
    limits.nvidia.com/gpu: "8"
```

**2. Limit Ranges**
```yaml
# k8s/limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: muai-limits
  namespace: muai-system
spec:
  limits:
  - max:
      cpu: "8"
      memory: 32Gi
    min:
      cpu: "1"
      memory: 4Gi
    type: Container
```

### Backup and Recovery

**1. Backup Configuration**
```bash
# Backup ConfigMaps and Secrets
kubectl get configmap -n muai-system -o yaml > backup/configmaps.yaml
kubectl get secret -n muai-system -o yaml > backup/secrets.yaml

# Backup PVCs
kubectl get pvc -n muai-system -o yaml > backup/pvcs.yaml
```

**2. Disaster Recovery**
```bash
# Restore from backup
kubectl apply -f backup/configmaps.yaml
kubectl apply -f backup/secrets.yaml
kubectl apply -f backup/pvcs.yaml

# Redeploy application
kubectl apply -f k8s/
```

### Logging

**1. Centralized Logging**
```yaml
# k8s/fluentd.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: muai-system
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/muai-*.log
      pos_file /var/log/fluentd-muai.pos
      tag muai.*
      <parse>
        @type json
      </parse>
    </source>
    
    <match muai.**>
      @type elasticsearch
      host elasticsearch
      port 9200
      logstash_format true
      logstash_prefix muai
    </match>
```

### Performance Tuning

**1. GPU Optimization**
```bash
# Set GPU clock speeds
nvidia-smi -pm 1
nvidia-smi -ac 5001,1590

# Monitor GPU usage
nvidia-smi dmon -s pucvmet
```

**2. Kernel Parameters**
```bash
# /etc/sysctl.conf
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535
vm.swappiness = 10
```

### Monitoring and Alerting

**1. Alert Rules**
```yaml
# prometheus-rules.yml
groups:
- name: muai_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High inference latency"
      description: "P95 latency is {{ $value }}s"
  
  - alert: HighErrorRate
    expr: rate(inference_requests_total{status="error"}[5m]) / rate(inference_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate"
      description: "Error rate is {{ $value | humanizePercentage }}"
  
  - alert: GPUMemoryHigh
    expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage high"
      description: "GPU {{ $labels.gpu_id }} memory is {{ $value | humanizePercentage }}"
```

## Troubleshooting

### Common Issues

**Pod Not Starting**
```bash
# Check pod status
kubectl describe pod <pod-name> -n muai-system

# Check logs
kubectl logs <pod-name> -n muai-system

# Check events
kubectl get events -n muai-system --sort-by='.lastTimestamp'
```

**GPU Not Available**
```bash
# Check GPU nodes
kubectl get nodes -o json | jq '.items[].status.capacity'

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Check GPU allocation
kubectl describe node <node-name> | grep nvidia.com/gpu
```

**High Memory Usage**
```bash
# Check memory usage
kubectl top pods -n muai-system

# Adjust memory limits
kubectl set resources deployment muai-orchestration \
  --limits=memory=32Gi \
  -n muai-system
```

## Next Steps

- [Configuration Guide](./optimization_configuration_guide.md)
- [Configuration Examples](./optimization_configuration_examples.md)
- [Migration Guide](./optimization_migration_guide.md)
- [API Reference](./api_reference.md)
