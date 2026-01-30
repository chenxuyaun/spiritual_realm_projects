"""
OpenVINO Backend Basic Usage Example

This script demonstrates basic usage of the OpenVINO backend integration
for the MuAI Multi-Model Orchestration System.

Requirements:
- OpenVINO installed: pip install openvino openvino-dev
- Model exported: python scripts/export_to_openvino.py gpt2
"""

from mm_orch.runtime.model_manager import ModelManager
import logging

# Configure logging to see backend events
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_default_pytorch():
    """Example 1: Default PyTorch backend (backward compatible)"""
    print("\n" + "="*60)
    print("Example 1: Default PyTorch Backend")
    print("="*60)
    
    # No backend parameter = PyTorch (existing behavior)
    manager = ModelManager()
    
    # Load model
    logger.info("Loading model with PyTorch backend...")
    manager.load_model("gpt2", "transformers")
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    logger.info(f"Generating text for prompt: '{prompt}'")
    result = manager.generate("gpt2", prompt, max_length=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")


def example_2_openvino_backend():
    """Example 2: OpenVINO backend with automatic fallback"""
    print("\n" + "="*60)
    print("Example 2: OpenVINO Backend")
    print("="*60)
    
    # Explicitly use OpenVINO backend
    manager = ModelManager(backend="openvino")
    
    # Load model (will use OpenVINO if available, fallback to PyTorch)
    logger.info("Loading model with OpenVINO backend...")
    manager.load_model("gpt2", "transformers")
    
    # Generate text
    prompt = "Machine learning enables computers to"
    logger.info(f"Generating text for prompt: '{prompt}'")
    result = manager.generate("gpt2", prompt, max_length=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {result}")
    
    # Check which backend was actually used
    model_info = manager.get_model("gpt2")
    backend_used = model_info.get("backend_metadata", {}).get("backend", "unknown")
    print(f"\nBackend used: {backend_used}")


def example_3_per_model_override():
    """Example 3: Per-model backend override"""
    print("\n" + "="*60)
    print("Example 3: Per-Model Backend Override")
    print("="*60)
    
    # Default to PyTorch
    manager = ModelManager(backend="pytorch")
    
    # Load one model with PyTorch
    logger.info("Loading gpt2 with PyTorch...")
    manager.load_model("gpt2", "transformers")
    
    # Load another model with OpenVINO override
    logger.info("Loading distilgpt2 with OpenVINO override...")
    manager.load_model("distilgpt2", "transformers", backend_override="openvino")
    
    # Generate with both models
    prompt = "Deep learning is"
    
    result_pt = manager.generate("gpt2", prompt, max_length=30)
    result_ov = manager.generate("distilgpt2", prompt, max_length=30)
    
    print(f"\nPrompt: {prompt}")
    print(f"GPT-2 (PyTorch): {result_pt}")
    print(f"DistilGPT-2 (OpenVINO): {result_ov}")


def example_4_device_selection():
    """Example 4: Device selection for OpenVINO"""
    print("\n" + "="*60)
    print("Example 4: Device Selection")
    print("="*60)
    
    # Try different devices
    devices = ["CPU", "GPU", "AUTO"]
    
    for device in devices:
        try:
            logger.info(f"Trying device: {device}")
            manager = ModelManager(
                backend="openvino",
                backend_config={
                    "openvino": {
                        "device": device,
                        "enable_fallback": True
                    }
                }
            )
            
            manager.load_model("gpt2", "transformers")
            result = manager.generate("gpt2", "Test", max_length=20)
            
            model_info = manager.get_model("gpt2")
            actual_device = model_info.get("backend_metadata", {}).get("device", "unknown")
            print(f"Requested: {device}, Actual: {actual_device}")
            
        except Exception as e:
            logger.error(f"Device {device} failed: {e}")


def example_5_configuration_file():
    """Example 5: Configuration file-based backend selection"""
    print("\n" + "="*60)
    print("Example 5: Configuration File")
    print("="*60)
    
    # Create a temporary configuration
    import yaml
    import os
    
    config = {
        "backend": {
            "default": "openvino",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {
            "gpt2": "openvino"
        }
    }
    
    config_path = "config/backend_example.yaml"
    os.makedirs("config", exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Created configuration file: {config_path}")
    
    # Use configuration file
    manager = ModelManager(backend_config=config_path)
    manager.load_model("gpt2", "transformers")
    
    result = manager.generate("gpt2", "Configuration-based backend", max_length=30)
    print(f"\nGenerated: {result}")
    
    # Check backend
    model_info = manager.get_model("gpt2")
    backend_used = model_info.get("backend_metadata", {}).get("backend", "unknown")
    print(f"Backend from config: {backend_used}")


def example_6_error_handling():
    """Example 6: Error handling and fallback"""
    print("\n" + "="*60)
    print("Example 6: Error Handling and Fallback")
    print("="*60)
    
    # Try to load a model that doesn't exist in OpenVINO format
    manager = ModelManager(
        backend="openvino",
        backend_config={
            "openvino": {
                "enable_fallback": True  # Enable automatic fallback
            }
        }
    )
    
    try:
        logger.info("Attempting to load model (may fallback to PyTorch)...")
        manager.load_model("gpt2", "transformers")
        
        result = manager.generate("gpt2", "Fallback test", max_length=30)
        print(f"\nGenerated: {result}")
        
        # Check if fallback occurred
        model_info = manager.get_model("gpt2")
        backend_used = model_info.get("backend_metadata", {}).get("backend", "unknown")
        is_fallback = model_info.get("backend_metadata", {}).get("is_fallback", False)
        
        print(f"Backend used: {backend_used}")
        print(f"Fallback occurred: {is_fallback}")
        
    except Exception as e:
        logger.error(f"Error: {e}")


def example_7_model_info():
    """Example 7: Getting model information"""
    print("\n" + "="*60)
    print("Example 7: Model Information")
    print("="*60)
    
    manager = ModelManager(backend="openvino")
    manager.load_model("gpt2", "transformers")
    
    # Get model information
    model_info = manager.get_model("gpt2")
    
    print("\nModel Information:")
    print(f"  Model name: gpt2")
    print(f"  Backend: {model_info.get('backend_metadata', {}).get('backend', 'unknown')}")
    print(f"  Device: {model_info.get('backend_metadata', {}).get('device', 'unknown')}")
    print(f"  Is fallback: {model_info.get('backend_metadata', {}).get('is_fallback', False)}")
    
    # Get available backends
    from mm_orch.runtime.backend_factory import BackendFactory
    factory = BackendFactory()
    available = factory.get_available_backends()
    print(f"\nAvailable backends: {', '.join(available)}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("OpenVINO Backend Integration - Basic Usage Examples")
    print("="*60)
    
    examples = [
        ("Default PyTorch Backend", example_1_default_pytorch),
        ("OpenVINO Backend", example_2_openvino_backend),
        ("Per-Model Override", example_3_per_model_override),
        ("Device Selection", example_4_device_selection),
        ("Configuration File", example_5_configuration_file),
        ("Error Handling", example_6_error_handling),
        ("Model Information", example_7_model_info),
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
