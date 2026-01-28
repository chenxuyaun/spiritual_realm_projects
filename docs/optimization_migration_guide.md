# Migration Guide: Adding Optimization and Monitoring

This guide helps you migrate from the existing MuAI system to the enhanced version with optimization and monitoring features.

## Table of Contents

1. [Overview](#overview)
2. [Backward Compatibility](#backward-compatibility)
3. [Migration Steps](#migration-steps)
4. [Code Changes](#code-changes)
5. [Configuration Migration](#configuration-migration)
6. [Testing Migration](#testing-migration)
7. [Rollback Plan](#rollback-plan)
8. [Troubleshooting](#troubleshooting)

## Overview

The optimization and monitoring features are designed to be **fully backward compatible** with the existing system. You can adopt features incrementally without breaking existing functionality.

### What's New

- **Optimization Engines**: vLLM, DeepSpeed, ONNX Runtime
- **Dynamic Batching**: Automatic request batching
- **KV Caching**: Transformer key-value caching
- **Monitoring**: Prometheus metrics and OpenTelemetry tracing
- **Anomaly Detection**: Automatic performance alerts
- **Server Mode**: Long-running inference server
- **Auto-Tuning**: Automatic parameter optimization

### Migration Philosophy

1. **Zero Breaking Changes**: Existing code continues to work
2. **Opt-In Features**: All new features are optional
3. **Graceful Degradation**: System works without optimization libraries
4. **Incremental Adoption**: Enable features one at a time

## Backward Compatibility

### Existing Code Works Unchanged

All existing code continues to work without modifications:

```python
# Existing code - still works!
from mm_orch.runtime.model_manager import ModelManager
from mm_orch.workflows.search_qa import SearchQAWorkflow

# No changes needed
model_manager = ModelManager()
workflow = SearchQAWorkflow(model_manager)
result = workflow.execute({"query": "What is AI?"})
```

### Optional Optimization

Optimization features are opt-in:

```python
# With optimization (optional)
from mm_orch.optimization import OptimizationManager, load_optimization_config

config = load_optimization_config("config/optimization.yaml")
opt_manager = OptimizationManager(config)

# Pass to ModelManager (optional parameter)
model_manager = ModelManager(optimization_manager=opt_manager)
```

### Configuration Compatibility

Existing configuration files continue to work:

```yaml
# config/system.yaml - existing config still works
models:
  qwen_chat:
    path: "Qwen/Qwen-7B-Chat"
    device: "cuda"
    load_in_8bit: false

# New optimization config is separate (optional)
# config/optimization.yaml
```

## Migration Steps

### Step 1: Install Optional Dependencies

Install optimization libraries as needed:

```bash
# Install all optimization dependencies
pip install vllm deepspeed onnxruntime-gpu

# Or install selectively
pip install vllm  # For vLLM only
pip install deepspeed  # For DeepSpeed only
pip install onnxruntime-gpu  # For ONNX Runtime

# Monitoring dependencies (recommended)
pip install prometheus-client opentelemetry-api opentelemetry-sdk
```

**Note**: System works without these dependencies - they're truly optional.

### Step 2: Add Configuration File

Create optimization configuration:

```bash
# Copy example configuration
cp config/optimization.example.yaml config/optimization.yaml

# Edit as needed
nano config/optimization.yaml
```

Start with minimal configuration:

```yaml
# config/optimization.yaml - minimal
optimization:
  enabled: true
  engine_preference: [pytorch]  # Start with existing engine
  
monitoring:
  enabled: true
  prometheus:
    enabled: true
    port: 9090
```

### Step 3: Update Code (Optional)

Update code to use optimization features:

#### Option A: Minimal Changes (Recommended)

Use environment variables to enable features:

```bash
# Enable vLLM via environment variable
export MUAI_OPT_VLLM_ENABLED=true

# Run existing code - no changes needed!
python -m mm_orch.main "What is AI?"
```

#### Option B: Explicit Configuration

Update initialization code:

```python
# Before (still works)
from mm_orch.runtime.model_manager import ModelManager

model_manager = ModelManager()

# After (with optimization)
from mm_orch.runtime.model_manager import ModelManager
from mm_orch.optimization import OptimizationManager, load_optimization_config

# Load optimization config
opt_config = load_optimization_config("config/optimization.yaml")
opt_manager = OptimizationManager(opt_config)

# Pass to ModelManager
model_manager = ModelManager(optimization_manager=opt_manager)
```

### Step 4: Enable Monitoring

Add monitoring to your application:

```python
# Before
from mm_orch.orchestrator import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.execute(request)

# After (with monitoring)
from mm_orch.orchestrator import Orchestrator
from mm_orch.monitoring import PrometheusExporter, OTelTracer

# Initialize monitoring
prometheus = PrometheusExporter(port=9090)
prometheus.start_server()

tracer = OTelTracer(endpoint="http://localhost:4317")

# Pass to orchestrator
orchestrator = Orchestrator(
    prometheus_exporter=prometheus,
    tracer=tracer
)

result = orchestrator.execute(request)
```

### Step 5: Test Incrementally

Test each feature individually:

```bash
# 1. Test with PyTorch only (baseline)
python -m mm_orch.main "test query"

# 2. Enable Prometheus metrics
export MUAI_MON_PROMETHEUS_ENABLED=true
python -m mm_orch.main "test query"
curl http://localhost:9090/metrics

# 3. Enable vLLM (if installed)
export MUAI_OPT_VLLM_ENABLED=true
python -m mm_orch.main "test query"

# 4. Enable batching
export MUAI_OPT_BATCHER_ENABLED=true
python -m mm_orch.main "test query"

# 5. Enable caching
export MUAI_OPT_CACHE_ENABLED=true
python -m mm_orch.main "test query"
```

### Step 6: Deploy to Production

Deploy with gradual rollout:

1. **Staging Environment**: Test full configuration
2. **Canary Deployment**: Deploy to small percentage of traffic
3. **Monitor Metrics**: Watch for issues
4. **Full Rollout**: Deploy to all instances

## Code Changes

### ModelManager Integration

The `ModelManager` class now accepts an optional `OptimizationManager`:

```python
# mm_orch/runtime/model_manager.py

class ModelManager:
    def __init__(
        self,
        config: Optional[Dict] = None,
        optimization_manager: Optional[OptimizationManager] = None  # NEW
    ):
        self.config = config or {}
        self.optimization_manager = optimization_manager  # NEW
        self.models = {}
        self.cache = LRUCache(max_size=3)
    
    def infer(self, model_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Use optimization if available
        if self.optimization_manager:
            result = self.optimization_manager.infer(model_name, inputs)
            return result.outputs
        
        # Fallback to standard inference
        model = self.load_model(model_name)
        return model(**inputs)
```

**Migration**: No changes required - `optimization_manager` parameter is optional.

### Workflow Integration

Workflows can optionally use optimization:

```python
# mm_orch/workflows/base.py

class BaseWorkflow:
    def __init__(
        self,
        model_manager: ModelManager,
        tracer: Optional[OTelTracer] = None  # NEW
    ):
        self.model_manager = model_manager
        self.tracer = tracer  # NEW
    
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        # Use tracing if available
        if self.tracer:
            with self.tracer.trace_workflow(self.__class__.__name__):
                return self._execute_impl(parameters)
        
        # Standard execution
        return self._execute_impl(parameters)
```

**Migration**: No changes required - `tracer` parameter is optional.

### Orchestrator Integration

The `Orchestrator` accepts optional monitoring components:

```python
# mm_orch/orchestrator.py

class Orchestrator:
    def __init__(
        self,
        router: Router,
        model_manager: ModelManager,
        prometheus_exporter: Optional[PrometheusExporter] = None,  # NEW
        tracer: Optional[OTelTracer] = None,  # NEW
        performance_monitor: Optional[PerformanceMonitor] = None,  # NEW
        anomaly_detector: Optional[AnomalyDetector] = None  # NEW
    ):
        self.router = router
        self.model_manager = model_manager
        self.prometheus = prometheus_exporter  # NEW
        self.tracer = tracer  # NEW
        self.perf_monitor = performance_monitor  # NEW
        self.anomaly_detector = anomaly_detector  # NEW
```

**Migration**: No changes required - all monitoring parameters are optional.

## Configuration Migration

### Merging Configurations

You can merge optimization config into existing `system.yaml`:

```yaml
# config/system.yaml - merged configuration

# Existing configuration
models:
  qwen_chat:
    path: "Qwen/Qwen-7B-Chat"
    device: "cuda"
    load_in_8bit: false

router:
  default_workflow: "search_qa"
  confidence_threshold: 0.7

# New optimization configuration (merged)
optimization:
  enabled: true
  engine_preference: [vllm, pytorch]
  vllm:
    enabled: true
    tensor_parallel_size: 1
    dtype: fp16

monitoring:
  enabled: true
  prometheus:
    enabled: true
    port: 9090
```

Or keep separate files:

```bash
config/
├── system.yaml          # Existing config
└── optimization.yaml    # New config (separate)
```

### Environment Variable Migration

Map existing environment variables to new ones:

```bash
# Before
export CUDA_VISIBLE_DEVICES=0
export MODEL_CACHE_DIR=/path/to/cache

# After (add new variables)
export CUDA_VISIBLE_DEVICES=0
export MODEL_CACHE_DIR=/path/to/cache
export MUAI_OPT_VLLM_ENABLED=true
export MUAI_MON_PROMETHEUS_ENABLED=true
```

## Testing Migration

### Unit Tests

Existing unit tests continue to work:

```python
# tests/unit/test_model_manager.py

def test_model_loading():
    """Existing test - still works"""
    manager = ModelManager()
    model = manager.load_model("qwen_chat")
    assert model is not None

def test_model_loading_with_optimization():
    """New test - with optimization"""
    opt_config = OptimizationConfig(enabled=True)
    opt_manager = OptimizationManager(opt_config)
    manager = ModelManager(optimization_manager=opt_manager)
    model = manager.load_model("qwen_chat")
    assert model is not None
```

### Integration Tests

Add tests for new features:

```python
# tests/integration/test_optimization.py

def test_vllm_fallback():
    """Test fallback when vLLM unavailable"""
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        fallback_on_error=True
    )
    manager = OptimizationManager(config)
    
    # Should fallback to PyTorch if vLLM fails
    result = manager.infer("qwen_chat", {"prompt": "test"})
    assert result.engine_used in ["vllm", "pytorch"]

def test_monitoring_integration():
    """Test monitoring doesn't break execution"""
    prometheus = PrometheusExporter(port=9091)
    prometheus.start_server()
    
    orchestrator = Orchestrator(
        router=Router(),
        model_manager=ModelManager(),
        prometheus_exporter=prometheus
    )
    
    result = orchestrator.execute({"query": "test"})
    assert result is not None
    
    prometheus.stop_server()
```

### Property-Based Tests

Verify backward compatibility:

```python
# tests/property/test_backward_compatibility.py

from hypothesis import given, strategies as st

@given(st.text())
def test_system_works_without_optimization(query):
    """System functions without optimization features"""
    # Standard initialization (no optimization)
    manager = ModelManager()
    workflow = SearchQAWorkflow(manager)
    
    # Should work without errors
    try:
        result = workflow.execute({"query": query})
        assert result is not None
    except Exception as e:
        # Only expected exceptions
        assert isinstance(e, (ValueError, RuntimeError))
```

## Rollback Plan

### Quick Rollback

Disable all features via environment variables:

```bash
# Disable optimization
export MUAI_OPT_ENABLED=false

# Disable monitoring
export MUAI_MON_ENABLED=false

# Restart application
systemctl restart muai-orchestration
```

### Configuration Rollback

Revert to previous configuration:

```bash
# Backup current config
cp config/optimization.yaml config/optimization.yaml.backup

# Restore previous config
git checkout HEAD~1 config/optimization.yaml

# Or use minimal config
cat > config/optimization.yaml << EOF
optimization:
  enabled: false
monitoring:
  enabled: false
EOF

# Restart
systemctl restart muai-orchestration
```

### Code Rollback

Revert to previous version:

```bash
# Git rollback
git revert <commit-hash>
git push

# Redeploy
./deploy.sh
```

### Database/State Rollback

No database changes required - all state is in-memory or configuration.

## Troubleshooting

### Issue: vLLM Import Error

**Symptom:**
```
ImportError: cannot import name 'LLM' from 'vllm'
```

**Solution:**
```bash
# Install vLLM
pip install vllm

# Or disable vLLM
export MUAI_OPT_VLLM_ENABLED=false
```

### Issue: Prometheus Port Conflict

**Symptom:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Change port
export MUAI_MON_PROMETHEUS_PORT=9091

# Or check what's using port 9090
lsof -i :9090
```

### Issue: High Memory Usage

**Symptom:**
```
CUDA out of memory
```

**Solution:**
```yaml
# Reduce GPU memory utilization
optimization:
  vllm:
    gpu_memory_utilization: 0.7  # Reduce from 0.9

# Or reduce batch size
optimization:
  batcher:
    max_batch_size: 8  # Reduce from 32
```

### Issue: Slow Startup

**Symptom:**
Server takes long time to start

**Solution:**
```yaml
# Disable model preloading
monitoring:
  server:
    preload_models: []  # Empty list

# Or reduce preloaded models
monitoring:
  server:
    preload_models:
      - qwen_chat  # Only essential models
```

### Issue: Metrics Not Appearing

**Symptom:**
`curl http://localhost:9090/metrics` returns nothing

**Solution:**
```bash
# Check if Prometheus is enabled
export MUAI_MON_PROMETHEUS_ENABLED=true

# Check if server is running
curl http://localhost:9090/health

# Check logs
tail -f logs/muai.log | grep prometheus
```

### Issue: Traces Not Exported

**Symptom:**
No traces in Jaeger/Zipkin

**Solution:**
```bash
# Check collector endpoint
export MUAI_MON_TRACING_ENDPOINT=http://localhost:4317

# Verify collector is running
curl http://localhost:4317

# Check sample rate
export MUAI_MON_TRACING_SAMPLE_RATE=1.0  # 100% sampling
```

### Issue: Performance Regression

**Symptom:**
System slower after enabling optimization

**Solution:**
```yaml
# Disable batching (might add latency)
optimization:
  batcher:
    enabled: false

# Reduce batch timeout
optimization:
  batcher:
    batch_timeout_ms: 10  # Reduce from 50

# Disable auto-tuning
optimization:
  tuner:
    enabled: false
```

## Migration Checklist

Use this checklist to track migration progress:

- [ ] Install optional dependencies
- [ ] Create optimization configuration file
- [ ] Test with PyTorch only (baseline)
- [ ] Enable Prometheus metrics
- [ ] Verify metrics endpoint
- [ ] Enable vLLM (if available)
- [ ] Test vLLM inference
- [ ] Enable dynamic batching
- [ ] Test batch processing
- [ ] Enable KV caching
- [ ] Test cache hit rates
- [ ] Enable OpenTelemetry tracing
- [ ] Verify traces in collector
- [ ] Enable anomaly detection
- [ ] Test alert delivery
- [ ] Enable server mode
- [ ] Test health/readiness endpoints
- [ ] Run integration tests
- [ ] Run property-based tests
- [ ] Deploy to staging
- [ ] Monitor staging metrics
- [ ] Deploy to production (canary)
- [ ] Monitor production metrics
- [ ] Full production rollout
- [ ] Document configuration
- [ ] Train team on new features

## Next Steps

- [Configuration Guide](./optimization_configuration_guide.md) - Detailed configuration options
- [Configuration Examples](./optimization_configuration_examples.md) - Scenario-based examples
- [Deployment Guide](./optimization_deployment_guide.md) - Production deployment
- [API Reference](./api_reference.md) - Programmatic API

## Support

For migration assistance:

- Check [Troubleshooting](#troubleshooting) section
- Review [GitHub Issues](https://github.com/your-org/muai/issues)
- Contact support team
- Join community Slack channel
