# OpenVINO Configuration Examples

This document provides ready-to-use configuration examples for common OpenVINO backend scenarios.

## Table of Contents

1. [Default PyTorch (Baseline)](#default-pytorch-baseline)
2. [Default OpenVINO Backend](#default-openvino-backend)
3. [Per-Model Backend Overrides](#per-model-backend-overrides)
4. [Device Selection](#device-selection)
5. [Fallback Disabled](#fallback-disabled)
6. [Production Configuration](#production-configuration)
7. [Development Configuration](#development-configuration)
8. [Multi-Device Configuration](#multi-device-configuration)

---

## Default PyTorch (Baseline)

**Use Case**: Maintain existing behavior, no changes

**Configuration**: No configuration file needed, or:

```yaml
# config/backend.yaml
backend:
  default: pytorch
  
  pytorch:
    device: cpu
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

# No backend parameter = PyTorch (default)
manager = ModelManager()
manager.load_model("gpt2", "transformers")
result = manager.generate("gpt2", "Hello world")
```

**Behavior**:
- Uses PyTorch for all models
- Existing behavior preserved
- No OpenVINO required

---

## Default OpenVINO Backend

**Use Case**: Use OpenVINO for all models by default

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: CPU
    enable_fallback: true
    cache_dir: models/openvino
  
  pytorch:
    device: cpu
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

# Reads configuration, uses OpenVINO by default
manager = ModelManager(backend_config="config/backend.yaml")
manager.load_model("gpt2", "transformers")
result = manager.generate("gpt2", "Hello world")
```

**Behavior**:
- All models use OpenVINO by default
- Automatic fallback to PyTorch on errors
- 2-3x performance improvement expected

---

## Per-Model Backend Overrides

**Use Case**: Use OpenVINO for some models, PyTorch for others

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: pytorch  # Default to PyTorch
  
  openvino:
    device: CPU
    enable_fallback: true
  
  pytorch:
    device: cpu

# Specify which models use OpenVINO
model_overrides:
  gpt2: openvino          # Fast inference for GPT-2
  distilgpt2: openvino    # Fast inference for DistilGPT2
  t5-small: openvino      # Fast inference for T5
  llama-7b: pytorch       # Large model stays on PyTorch
  custom-model: pytorch   # Custom model not exported yet
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend_config="config/backend.yaml")

# Uses OpenVINO (from model_overrides)
manager.load_model("gpt2", "transformers")

# Uses PyTorch (from model_overrides)
manager.load_model("llama-7b", "transformers")

# Uses PyTorch (default, no override)
manager.load_model("new-model", "transformers")
```

**Behavior**:
- Fine-grained control per model
- Mix backends in same application
- Easy to add/remove models from OpenVINO

---

## Device Selection

### CPU Only

**Use Case**: Force CPU inference (most compatible)

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: CPU
    enable_fallback: true
```

### GPU Only

**Use Case**: Use GPU if available (Intel iGPU/dGPU)

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: GPU
    enable_fallback: true  # Falls back to CPU if GPU unavailable
```

### AUTO Device Selection

**Use Case**: Let OpenVINO choose optimal device

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: AUTO  # OpenVINO selects best available device
    enable_fallback: true
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

# Uses AUTO device selection
manager = ModelManager(backend_config="config/backend.yaml")
manager.load_model("gpt2", "transformers")

# Check which device was selected
model_info = manager.get_model("gpt2")
print(f"Using device: {model_info['backend_metadata']['device']}")
```

**Behavior**:
- AUTO tries: GPU → CPU
- Optimal performance automatically
- No manual device management

---

## Fallback Disabled

**Use Case**: Strict OpenVINO-only mode, fail fast on errors

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: CPU
    enable_fallback: false  # Raise errors instead of falling back
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend_config="config/backend.yaml")

try:
    manager.load_model("gpt2", "transformers")
except Exception as e:
    print(f"OpenVINO failed: {e}")
    # Handle error explicitly
```

**Behavior**:
- No automatic fallback to PyTorch
- Errors raised immediately
- Useful for debugging OpenVINO issues
- Ensures OpenVINO is actually being used

**When to Use**:
- Testing OpenVINO integration
- Debugging performance issues
- Ensuring models are exported correctly
- Production with strict requirements

---

## Production Configuration

**Use Case**: Robust production deployment with monitoring

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: AUTO
    enable_fallback: true
    cache_dir: /var/lib/muai/models/openvino
    num_streams: 2  # Parallel inference streams
  
  pytorch:
    device: cpu

model_overrides:
  # Production models using OpenVINO
  gpt2: openvino
  distilgpt2: openvino
  t5-small: openvino
  
  # Experimental models using PyTorch
  experimental-model: pytorch

# Logging configuration
logging:
  level: INFO
  backend_events: true
  performance_metrics: true
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize with production config
manager = ModelManager(backend_config="config/backend.yaml")

# Load models
manager.load_model("gpt2", "transformers")
manager.load_model("t5-small", "transformers")

# Monitor performance
stats = manager.get_performance_stats("openvino")
logging.info(f"OpenVINO stats: {stats}")
```

**Features**:
- Automatic device selection
- Fallback enabled for reliability
- Performance monitoring
- Structured logging
- Per-model control

---

## Development Configuration

**Use Case**: Development and testing with verbose logging

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: pytorch  # Safe default for development
  
  openvino:
    device: CPU
    enable_fallback: true
    cache_dir: models/openvino
  
  pytorch:
    device: cpu

model_overrides:
  # Test OpenVINO with specific models
  gpt2: openvino

# Development settings
logging:
  level: DEBUG
  backend_events: true
  fallback_events: true
  performance_metrics: true
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager
import logging

# Verbose logging for development
logging.basicConfig(level=logging.DEBUG)

manager = ModelManager(backend_config="config/backend.yaml")

# Test both backends
manager.load_model("gpt2", "transformers")  # Uses OpenVINO
manager.load_model("t5-small", "transformers")  # Uses PyTorch

# Compare performance
comparison = manager.compare_backends("pytorch", "openvino")
print(f"Performance comparison: {comparison}")
```

**Features**:
- PyTorch default (safe)
- Selective OpenVINO testing
- Verbose logging
- Performance comparison
- Easy debugging

---

## Multi-Device Configuration

**Use Case**: Different devices for different models

**Configuration**:

```yaml
# config/backend.yaml
backend:
  default: openvino
  
  openvino:
    device: AUTO  # Default device
    enable_fallback: true
  
  pytorch:
    device: cpu

model_overrides:
  # Small models on GPU for speed
  gpt2: openvino
  distilgpt2: openvino
  
  # Large models on CPU to avoid GPU memory issues
  llama-7b: pytorch
  llama-13b: pytorch

# Per-model device overrides (advanced)
device_overrides:
  gpt2: GPU
  distilgpt2: GPU
  t5-small: CPU
```

**Code**:
```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend_config="config/backend.yaml")

# Small model on GPU (fast)
manager.load_model("gpt2", "transformers")

# Large model on CPU (memory efficient)
manager.load_model("llama-7b", "transformers")

# Check device allocation
for model_name in ["gpt2", "llama-7b"]:
    model = manager.get_model(model_name)
    print(f"{model_name}: {model['backend_metadata']['device']}")
```

**Output**:
```
gpt2: GPU
llama-7b: cpu
```

---

## Programmatic Configuration

**Use Case**: Configure backends in code without YAML files

### Example 1: Simple OpenVINO

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(
    backend="openvino",
    backend_config={
        "openvino": {
            "device": "CPU",
            "enable_fallback": True
        }
    }
)
```

### Example 2: Per-Model Override

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend="pytorch")

# Override for specific model
manager.load_model("gpt2", "transformers", backend_override="openvino")
manager.load_model("t5-small", "transformers")  # Uses PyTorch
```

### Example 3: Dynamic Configuration

```python
from mm_orch.runtime.model_manager import ModelManager
import os

# Choose backend based on environment
backend = "openvino" if os.getenv("USE_OPENVINO") == "1" else "pytorch"

manager = ModelManager(backend=backend)
manager.load_model("gpt2", "transformers")
```

---

## Configuration Validation

Test your configuration before deployment:

```python
from mm_orch.runtime.backend_config import BackendConfig

# Load and validate configuration
config = BackendConfig("config/backend.yaml")

# Check default backend
print(f"Default backend: {config.get_default_backend()}")

# Check model overrides
for model in ["gpt2", "t5-small", "llama-7b"]:
    backend = config.get_model_backend(model)
    print(f"{model}: {backend or 'default'}")

# Check backend configuration
ov_config = config.get_backend_config("openvino")
print(f"OpenVINO device: {ov_config.get('device', 'CPU')}")
print(f"Fallback enabled: {ov_config.get('enable_fallback', True)}")
```

---

## Quick Reference

| Scenario | Default Backend | Fallback | Device | Use Case |
|----------|----------------|----------|--------|----------|
| Baseline | pytorch | N/A | cpu | Existing behavior |
| All OpenVINO | openvino | true | CPU | Maximum performance |
| Mixed | pytorch | true | CPU | Gradual migration |
| GPU Accelerated | openvino | true | GPU | GPU available |
| Auto Device | openvino | true | AUTO | Optimal hardware |
| Strict Mode | openvino | false | CPU | Testing/debugging |
| Production | openvino | true | AUTO | Robust deployment |
| Development | pytorch | true | CPU | Safe development |

---

## Next Steps

- Review [Migration Guide](openvino_migration_guide.md) for step-by-step instructions
- Read [Performance Guide](openvino_performance_guide.md) for optimization tips
- Check [Example Scripts](../examples/) for working code samples
- See [Troubleshooting](openvino_migration_guide.md#troubleshooting) for common issues
