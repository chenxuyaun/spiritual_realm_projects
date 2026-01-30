# OpenVINO Backend Migration Guide

## Overview

This guide helps you migrate your existing MuAI Multi-Model Orchestration System to use the OpenVINO backend for improved inference performance. The integration is designed for **zero breaking changes** - your existing code will continue to work without modifications.

## Backward Compatibility Guarantees

### What Stays the Same

1. **API Signatures**: All `ModelManager` methods maintain identical signatures
2. **Default Behavior**: Without configuration changes, the system uses PyTorch (existing behavior)
3. **Return Values**: Both backends return results in the same format
4. **Model Cache**: Existing caching behavior is preserved
5. **Error Handling**: Existing error types and messages remain unchanged

### What's New (Opt-In)

1. **Backend Selection**: Choose between PyTorch and OpenVINO
2. **Configuration Options**: YAML-based backend configuration
3. **Performance Monitoring**: Built-in backend performance comparison
4. **Automatic Fallback**: Graceful degradation when OpenVINO fails

## Migration Paths

### Path 1: No Changes Required (Default)

If you don't modify anything, your code continues to use PyTorch:

```python
# This code works exactly as before
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager()
manager.load_model("gpt2", "transformers")
result = manager.generate("gpt2", "Hello world")
```

**Result**: Uses PyTorch backend (100% backward compatible)

### Path 2: Programmatic Backend Selection

Enable OpenVINO for specific use cases:

```python
from mm_orch.runtime.model_manager import ModelManager

# Option A: Set default backend to OpenVINO
manager = ModelManager(backend="openvino")
manager.load_model("gpt2", "transformers")
result = manager.generate("gpt2", "Hello world")

# Option B: Override backend per model
manager = ModelManager(backend="pytorch")  # Default is PyTorch
manager.load_model("gpt2", "transformers", backend_override="openvino")
result = manager.generate("gpt2", "Hello world")
```

**Result**: Uses OpenVINO backend with automatic fallback to PyTorch

### Path 3: Configuration-Based Selection

Configure backends system-wide via YAML:

**Step 1**: Create or update `config/backend.yaml`:

```yaml
backend:
  # Default backend for all models
  default: openvino
  
  # OpenVINO-specific settings
  openvino:
    device: CPU  # CPU, GPU, or AUTO
    enable_fallback: true
    cache_dir: models/openvino
  
  # PyTorch-specific settings
  pytorch:
    device: cpu

# Per-model backend overrides
model_overrides:
  gpt2: openvino
  t5-small: pytorch
```

**Step 2**: Use ModelManager normally:

```python
from mm_orch.runtime.model_manager import ModelManager

# Reads configuration automatically
manager = ModelManager(backend_config="config/backend.yaml")
manager.load_model("gpt2", "transformers")  # Uses OpenVINO (from config)
result = manager.generate("gpt2", "Hello world")
```

**Result**: Backend selection controlled by configuration file

## Step-by-Step Migration

### Step 1: Install OpenVINO

```bash
pip install openvino openvino-dev
```

Verify installation:

```python
import openvino as ov
print(f"OpenVINO version: {ov.__version__}")
```

### Step 2: Export Models to OpenVINO Format

Use the provided export utility:

```bash
# Export with default precision (FP32)
python scripts/export_to_openvino.py gpt2

# Export with FP16 precision (smaller, faster)
python scripts/export_to_openvino.py gpt2 --precision FP16

# Export with custom output path
python scripts/export_to_openvino.py gpt2 --output models/openvino/gpt2
```

This creates:
```
models/openvino/gpt2/
├── gpt2.xml  # Model structure
└── gpt2.bin  # Model weights
```

### Step 3: Choose Migration Path

**For Quick Testing** (Programmatic):

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend="openvino")
manager.load_model("gpt2", "transformers")
result = manager.generate("gpt2", "Hello world")
print(result)
```

**For Production** (Configuration):

1. Create `config/backend.yaml` (see Path 3 above)
2. Update your code to load configuration:

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend_config="config/backend.yaml")
# Rest of your code remains unchanged
```

### Step 4: Verify Performance Improvement

```python
from mm_orch.runtime.model_manager import ModelManager
import time

# Test PyTorch
manager_pt = ModelManager(backend="pytorch")
manager_pt.load_model("gpt2", "transformers")

start = time.time()
result_pt = manager_pt.generate("gpt2", "Hello world", max_length=50)
pytorch_time = time.time() - start

# Test OpenVINO
manager_ov = ModelManager(backend="openvino")
manager_ov.load_model("gpt2", "transformers")

start = time.time()
result_ov = manager_ov.generate("gpt2", "Hello world", max_length=50)
openvino_time = time.time() - start

print(f"PyTorch: {pytorch_time:.3f}s")
print(f"OpenVINO: {openvino_time:.3f}s")
print(f"Speedup: {pytorch_time / openvino_time:.2f}x")
```

Expected output:
```
PyTorch: 2.145s
OpenVINO: 0.823s
Speedup: 2.61x
```

## Advanced Configuration

### Device Selection

```yaml
backend:
  openvino:
    device: AUTO  # Let OpenVINO choose optimal device
    # device: CPU   # Force CPU
    # device: GPU   # Force GPU (if available)
```

Programmatic:

```python
manager = ModelManager(
    backend="openvino",
    backend_config={"openvino": {"device": "AUTO"}}
)
```

### Disable Automatic Fallback

```yaml
backend:
  openvino:
    enable_fallback: false  # Raise errors instead of falling back
```

Programmatic:

```python
manager = ModelManager(
    backend="openvino",
    backend_config={"openvino": {"enable_fallback": False}}
)
```

### Per-Model Backend Selection

```yaml
model_overrides:
  gpt2: openvino        # Use OpenVINO for GPT-2
  t5-small: openvino    # Use OpenVINO for T5
  llama-7b: pytorch     # Use PyTorch for large models
```

## Troubleshooting

### Issue: "OpenVINO backend not available"

**Cause**: OpenVINO not installed

**Solution**:
```bash
pip install openvino openvino-dev
```

### Issue: "Model file not found: models/openvino/gpt2/gpt2.xml"

**Cause**: Model not exported to OpenVINO format

**Solution**:
```bash
python scripts/export_to_openvino.py gpt2
```

### Issue: "Device 'GPU' not available"

**Cause**: GPU device not available on system

**Solution**: System automatically falls back to CPU with warning. To suppress warning:
```yaml
backend:
  openvino:
    device: CPU  # Explicitly use CPU
```

### Issue: OpenVINO inference fails

**Cause**: Model incompatibility or runtime error

**Solution**: System automatically falls back to PyTorch. Check logs:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

To disable fallback and see full error:
```yaml
backend:
  openvino:
    enable_fallback: false
```

### Issue: Performance not improved

**Possible Causes**:
1. Model not exported with optimal precision
2. CPU-bound workload with small models
3. First inference includes warmup time

**Solutions**:
```bash
# Try FP16 precision
python scripts/export_to_openvino.py gpt2 --precision FP16

# Use AUTO device selection
# In config/backend.yaml:
backend:
  openvino:
    device: AUTO
```

### Issue: Different outputs between backends

**Cause**: Floating-point precision differences (expected)

**Solution**: This is normal. Differences are typically < 0.1% due to:
- Different precision levels (FP32 vs FP16)
- Different computation order
- Hardware-specific optimizations

Both outputs are correct; use the faster backend.

## Performance Monitoring

Track and compare backend performance:

```python
from mm_orch.runtime.model_manager import ModelManager

manager = ModelManager(backend="openvino")
manager.load_model("gpt2", "transformers")

# Run some inferences
for i in range(10):
    manager.generate("gpt2", f"Test prompt {i}", max_length=50)

# Get performance statistics
stats = manager.get_performance_stats("openvino")
print(f"Average latency: {stats['avg_latency']:.3f}s")
print(f"P95 latency: {stats['p95_latency']:.3f}s")
print(f"Throughput: {stats['avg_throughput']:.1f} tokens/s")

# Compare backends
comparison = manager.compare_backends("pytorch", "openvino")
print(f"Speedup: {comparison['latency_improvement']:.2f}x")
```

## Migration Checklist

- [ ] Install OpenVINO: `pip install openvino openvino-dev`
- [ ] Export models: `python scripts/export_to_openvino.py <model_name>`
- [ ] Choose migration path (programmatic or configuration)
- [ ] Update code or create configuration file
- [ ] Test with existing workflows
- [ ] Verify performance improvements
- [ ] Monitor for fallback events in logs
- [ ] Update deployment documentation

## Best Practices

1. **Start Small**: Test with one model before migrating all models
2. **Keep Fallback Enabled**: Use automatic fallback in production for robustness
3. **Monitor Performance**: Use built-in monitoring to track improvements
4. **Export Optimization**: Use FP16 precision for better performance
5. **Device Selection**: Use AUTO device for optimal hardware utilization
6. **Configuration Management**: Use YAML configuration for production deployments
7. **Version Control**: Keep both PyTorch and OpenVINO models for flexibility

## Rollback Plan

If you need to revert to PyTorch-only:

**Option 1**: Remove configuration
```bash
rm config/backend.yaml
```

**Option 2**: Update configuration
```yaml
backend:
  default: pytorch
```

**Option 3**: Update code
```python
manager = ModelManager(backend="pytorch")
```

Your system immediately reverts to PyTorch behavior with zero code changes required.

## Next Steps

- Review [Configuration Examples](openvino_config_examples.md) for common scenarios
- Read [Performance Guide](openvino_performance_guide.md) for optimization tips
- Check [Example Scripts](../examples/) for working code samples
- See [API Reference](api_reference.md) for detailed API documentation

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs with `logging.basicConfig(level=logging.DEBUG)`
3. Verify model export: `ls models/openvino/<model_name>/`
4. Test with fallback disabled to see full errors
5. Consult OpenVINO documentation: https://docs.openvino.ai/
