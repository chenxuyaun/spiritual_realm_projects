"""
OpenVINO Backend Switching Example

This script demonstrates dynamic backend switching scenarios, including:
- Switching backends between models
- Runtime backend selection
- Fallback behavior
- Configuration-based switching

Requirements:
- OpenVINO installed: pip install openvino openvino-dev
- Models exported: python scripts/export_to_openvino.py gpt2
"""

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.runtime.backend_factory import BackendFactory
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_switch_between_models():
    """Example 1: Use different backends for different models"""
    print("\n" + "="*60)
    print("Example 1: Different Backends for Different Models")
    print("="*60)
    
    # Create manager with PyTorch default
    manager = ModelManager(backend="pytorch")
    
    # Load first model with PyTorch
    logger.info("Loading gpt2 with PyTorch...")
    manager.load_model("gpt2", "transformers")
    
    # Load second model with OpenVINO override
    logger.info("Loading distilgpt2 with OpenVINO...")
    manager.load_model("distilgpt2", "transformers", backend_override="openvino")
    
    # Generate with both models
    prompt = "The future of technology is"
    
    print(f"\nPrompt: {prompt}")
    
    # PyTorch generation
    start = time.time()
    result_pt = manager.generate("gpt2", prompt, max_length=30)
    pt_time = time.time() - start
    
    # OpenVINO generation
    start = time.time()
    result_ov = manager.generate("distilgpt2", prompt, max_length=30)
    ov_time = time.time() - start
    
    print(f"\nGPT-2 (PyTorch, {pt_time:.3f}s):")
    print(f"  {result_pt}")
    
    print(f"\nDistilGPT-2 (OpenVINO, {ov_time:.3f}s):")
    print(f"  {result_ov}")
    
    # Show backend info
    gpt2_info = manager.get_model("gpt2")
    distilgpt2_info = manager.get_model("distilgpt2")
    
    print(f"\nBackend Information:")
    print(f"  gpt2: {gpt2_info.get('backend_metadata', {}).get('backend', 'unknown')}")
    print(f"  distilgpt2: {distilgpt2_info.get('backend_metadata', {}).get('backend', 'unknown')}")


def example_2_runtime_backend_selection():
    """Example 2: Select backend at runtime based on conditions"""
    print("\n" + "="*60)
    print("Example 2: Runtime Backend Selection")
    print("="*60)
    
    # Simulate runtime conditions
    import random
    
    conditions = {
        "has_openvino": True,  # OpenVINO available
        "prefer_speed": random.choice([True, False]),  # Random preference
        "gpu_available": False,  # No GPU
    }
    
    print(f"Runtime conditions:")
    print(f"  OpenVINO available: {conditions['has_openvino']}")
    print(f"  Prefer speed: {conditions['prefer_speed']}")
    print(f"  GPU available: {conditions['gpu_available']}")
    
    # Select backend based on conditions
    if conditions["has_openvino"] and conditions["prefer_speed"]:
        backend = "openvino"
        print(f"\n→ Selected: OpenVINO (speed optimized)")
    elif conditions["gpu_available"]:
        backend = "pytorch"
        print(f"\n→ Selected: PyTorch (GPU available)")
    else:
        backend = "pytorch"
        print(f"\n→ Selected: PyTorch (default)")
    
    # Initialize with selected backend
    manager = ModelManager(backend=backend)
    manager.load_model("gpt2", "transformers")
    
    result = manager.generate("gpt2", "Runtime selection test", max_length=30)
    print(f"\nGenerated: {result}")


def example_3_fallback_demonstration():
    """Example 3: Demonstrate automatic fallback behavior"""
    print("\n" + "="*60)
    print("Example 3: Automatic Fallback Behavior")
    print("="*60)
    
    # Try OpenVINO with fallback enabled
    print("\nScenario A: Fallback Enabled (default)")
    manager_with_fallback = ModelManager(
        backend="openvino",
        backend_config={
            "openvino": {
                "enable_fallback": True
            }
        }
    )
    
    try:
        logger.info("Loading model with fallback enabled...")
        manager_with_fallback.load_model("gpt2", "transformers")
        
        result = manager_with_fallback.generate("gpt2", "Fallback test", max_length=30)
        print(f"Generated: {result}")
        
        # Check if fallback occurred
        model_info = manager_with_fallback.get_model("gpt2")
        backend_used = model_info.get("backend_metadata", {}).get("backend", "unknown")
        is_fallback = model_info.get("backend_metadata", {}).get("is_fallback", False)
        
        print(f"\nBackend used: {backend_used}")
        if is_fallback:
            print("✓ Fallback to PyTorch occurred")
        else:
            print("✓ OpenVINO used successfully")
            
    except Exception as e:
        logger.error(f"Failed even with fallback: {e}")
    
    # Try OpenVINO with fallback disabled
    print("\n" + "-"*60)
    print("Scenario B: Fallback Disabled")
    manager_no_fallback = ModelManager(
        backend="openvino",
        backend_config={
            "openvino": {
                "enable_fallback": False
            }
        }
    )
    
    try:
        logger.info("Loading model with fallback disabled...")
        manager_no_fallback.load_model("gpt2", "transformers")
        
        result = manager_no_fallback.generate("gpt2", "No fallback test", max_length=30)
        print(f"Generated: {result}")
        print("✓ OpenVINO used successfully")
        
    except Exception as e:
        logger.warning(f"Failed without fallback: {e}")
        print("✗ OpenVINO failed, no fallback available")


def example_4_configuration_switching():
    """Example 4: Switch backends using configuration"""
    print("\n" + "="*60)
    print("Example 4: Configuration-Based Switching")
    print("="*60)
    
    import yaml
    import os
    
    # Create different configurations
    configs = {
        "development": {
            "backend": {
                "default": "pytorch",
                "openvino": {"device": "CPU", "enable_fallback": True},
                "pytorch": {"device": "cpu"}
            }
        },
        "production": {
            "backend": {
                "default": "openvino",
                "openvino": {"device": "AUTO", "enable_fallback": True},
                "pytorch": {"device": "cpu"}
            }
        },
        "testing": {
            "backend": {
                "default": "openvino",
                "openvino": {"device": "CPU", "enable_fallback": False},
                "pytorch": {"device": "cpu"}
            }
        }
    }
    
    os.makedirs("config", exist_ok=True)
    
    # Test each configuration
    for env_name, config in configs.items():
        print(f"\n{'-'*60}")
        print(f"Environment: {env_name.upper()}")
        print(f"{'-'*60}")
        
        # Save configuration
        config_path = f"config/backend_{env_name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Load with configuration
        manager = ModelManager(backend_config=config_path)
        manager.load_model("gpt2", "transformers")
        
        # Generate
        result = manager.generate("gpt2", f"{env_name} test", max_length=20)
        
        # Show backend used
        model_info = manager.get_model("gpt2")
        backend_used = model_info.get("backend_metadata", {}).get("backend", "unknown")
        
        print(f"Default backend: {config['backend']['default']}")
        print(f"Actual backend: {backend_used}")
        print(f"Generated: {result[:50]}...")


def example_5_per_request_switching():
    """Example 5: Switch backends per request (advanced)"""
    print("\n" + "="*60)
    print("Example 5: Per-Request Backend Switching")
    print("="*60)
    
    # Load same model with both backends
    manager_pt = ModelManager(backend="pytorch")
    manager_ov = ModelManager(backend="openvino")
    
    manager_pt.load_model("gpt2", "transformers")
    manager_ov.load_model("gpt2", "transformers")
    
    # Simulate different request types
    requests = [
        {"prompt": "Quick response needed", "priority": "high", "max_length": 20},
        {"prompt": "Detailed analysis required", "priority": "low", "max_length": 100},
        {"prompt": "Real-time processing", "priority": "high", "max_length": 30},
        {"prompt": "Batch processing task", "priority": "low", "max_length": 50},
    ]
    
    print("\nProcessing requests with dynamic backend selection:")
    
    for i, request in enumerate(requests, 1):
        # Select backend based on priority
        if request["priority"] == "high":
            manager = manager_ov  # Use faster OpenVINO
            backend_name = "OpenVINO"
        else:
            manager = manager_pt  # Use PyTorch for low priority
            backend_name = "PyTorch"
        
        # Process request
        start = time.time()
        result = manager.generate("gpt2", request["prompt"], max_length=request["max_length"])
        elapsed = time.time() - start
        
        print(f"\nRequest {i} ({request['priority']} priority):")
        print(f"  Backend: {backend_name}")
        print(f"  Latency: {elapsed:.3f}s")
        print(f"  Prompt: {request['prompt']}")
        print(f"  Result: {result[:50]}...")


def example_6_check_available_backends():
    """Example 6: Check and display available backends"""
    print("\n" + "="*60)
    print("Example 6: Available Backends Detection")
    print("="*60)
    
    factory = BackendFactory()
    available = factory.get_available_backends()
    
    print(f"\nAvailable backends: {', '.join(available)}")
    
    # Test each backend
    for backend_name in available:
        print(f"\n{'-'*60}")
        print(f"Testing {backend_name.upper()} backend")
        print(f"{'-'*60}")
        
        try:
            manager = ModelManager(backend=backend_name)
            manager.load_model("gpt2", "transformers")
            
            start = time.time()
            result = manager.generate("gpt2", "Test", max_length=20)
            elapsed = time.time() - start
            
            print(f"✓ {backend_name} is working")
            print(f"  Latency: {elapsed:.3f}s")
            print(f"  Result: {result[:50]}...")
            
        except Exception as e:
            print(f"✗ {backend_name} failed: {e}")


def example_7_mixed_workload():
    """Example 7: Mixed workload with multiple models and backends"""
    print("\n" + "="*60)
    print("Example 7: Mixed Workload Scenario")
    print("="*60)
    
    # Initialize manager with default backend
    manager = ModelManager(backend="pytorch")
    
    # Define workload
    workload = [
        {"model": "gpt2", "backend": "openvino", "prompt": "Fast inference", "max_length": 30},
        {"model": "gpt2", "backend": "pytorch", "prompt": "Standard inference", "max_length": 30},
        {"model": "distilgpt2", "backend": "openvino", "prompt": "Small model", "max_length": 30},
    ]
    
    print("\nProcessing mixed workload:")
    
    results = []
    for i, task in enumerate(workload, 1):
        print(f"\n{'-'*60}")
        print(f"Task {i}: {task['model']} with {task['backend']}")
        print(f"{'-'*60}")
        
        # Load model with specified backend
        if task["model"] not in [m for m in manager._model_cache.keys()]:
            manager.load_model(task["model"], "transformers", backend_override=task["backend"])
        
        # Generate
        start = time.time()
        result = manager.generate(task["model"], task["prompt"], max_length=task["max_length"])
        elapsed = time.time() - start
        
        # Store results
        results.append({
            "task": i,
            "model": task["model"],
            "backend": task["backend"],
            "latency": elapsed,
            "result": result
        })
        
        print(f"Latency: {elapsed:.3f}s")
        print(f"Result: {result[:50]}...")
    
    # Summary
    print(f"\n{'='*60}")
    print("Workload Summary")
    print(f"{'='*60}")
    
    for r in results:
        print(f"\nTask {r['task']}: {r['model']} ({r['backend']})")
        print(f"  Latency: {r['latency']:.3f}s")


def main():
    """Run all backend switching examples"""
    print("\n" + "="*60)
    print("OpenVINO Backend Switching Examples")
    print("="*60)
    
    examples = [
        ("Different Backends for Different Models", example_1_switch_between_models),
        ("Runtime Backend Selection", example_2_runtime_backend_selection),
        ("Automatic Fallback Behavior", example_3_fallback_demonstration),
        ("Configuration-Based Switching", example_4_configuration_switching),
        ("Per-Request Backend Switching", example_5_per_request_switching),
        ("Available Backends Detection", example_6_check_available_backends),
        ("Mixed Workload Scenario", example_7_mixed_workload),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
