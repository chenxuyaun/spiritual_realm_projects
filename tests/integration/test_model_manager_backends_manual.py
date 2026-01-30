"""
Manual integration tests for ModelManager with multiple backends.
Run with: python tests/integration/test_model_manager_backends_manual.py
"""

import sys
import tempfile
import yaml
import os

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


def test_default_pytorch_backend():
    """Test that ModelManager defaults to PyTorch backend."""
    manager = ModelManager(max_cached_models=2)
    assert manager.default_backend == "pytorch", f"Expected 'pytorch', got '{manager.default_backend}'"
    assert "pytorch" in manager._backend_factory.get_available_backends()
    print("✓ Test 1 passed: Default PyTorch backend")


def test_explicit_backend_selection():
    """Test explicit backend selection."""
    manager_pt = ModelManager(backend="pytorch")
    assert manager_pt.default_backend == "pytorch"
    
    available_backends = manager_pt._backend_factory.get_available_backends()
    if "openvino" in available_backends:
        manager_ov = ModelManager(backend="openvino")
        assert manager_ov.default_backend == "openvino"
        print("✓ Test 2 passed: Explicit backend selection (PyTorch and OpenVINO)")
    else:
        print("✓ Test 2 passed: Explicit backend selection (PyTorch only, OpenVINO not available)")


def test_backend_config_from_file():
    """Test loading backend configuration from file."""
    config_data = {
        "backend": {
            "default": "pytorch",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True,
                "cache_dir": "models/openvino"
            },
            "pytorch": {
                "device": "cpu"
            }
        },
        "model_overrides": {
            "test-model-ov": "openvino"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        manager = ModelManager(backend_config=temp_path)
        assert manager.default_backend == "pytorch"
        assert manager._backend_config_loader is not None
        assert manager._backend_config_loader.get_default_backend() == "pytorch"
        print("✓ Test 3 passed: Backend config from file")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_backend_config_from_dict():
    """Test loading backend configuration from dictionary."""
    config_dict = {
        "backend": {
            "default": "pytorch",
            "pytorch": {"device": "cpu"}
        },
        "model_overrides": {}
    }
    
    manager = ModelManager(backend_config=config_dict)
    assert manager.default_backend == "pytorch"
    print("✓ Test 4 passed: Backend config from dict")


def test_per_model_backend_override():
    """Test per-model backend override from configuration."""
    config_data = {
        "backend": {
            "default": "pytorch",
            "openvino": {
                "device": "CPU",
                "enable_fallback": True
            }
        },
        "model_overrides": {
            "test-model-ov": "openvino"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        manager = ModelManager(backend_config=temp_path)
        override = manager._backend_config_loader.get_model_backend("test-model-ov")
        assert override == "openvino", f"Expected 'openvino', got '{override}'"
        print("✓ Test 5 passed: Per-model backend override")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_cache_info_includes_backend():
    """Test that cache info includes backend information."""
    manager = ModelManager(backend="pytorch")
    cache_info = manager.get_cache_info()
    
    assert "default_backend" in cache_info
    assert cache_info["default_backend"] == "pytorch"
    assert "available_backends" in cache_info
    assert isinstance(cache_info["available_backends"], list)
    print("✓ Test 6 passed: Cache info includes backend")


def test_backward_compatibility():
    """Test backward compatibility with existing code."""
    manager = ModelManager(max_cached_models=3, default_device="cpu")
    assert manager.default_backend == "pytorch"
    print("✓ Test 7 passed: Backward compatibility")


def test_api_signatures():
    """Test that existing API signatures are unchanged."""
    manager = ModelManager()
    
    assert hasattr(manager, 'load_model')
    assert hasattr(manager, 'get_model')
    assert hasattr(manager, 'unload_model')
    assert hasattr(manager, 'infer')
    assert hasattr(manager, 'register_model')
    assert hasattr(manager, 'get_cache_info')
    
    import inspect
    sig = inspect.signature(manager.load_model)
    params = list(sig.parameters.keys())
    assert 'model_name' in params
    assert 'device' in params
    assert 'backend_override' in params
    print("✓ Test 8 passed: API signatures unchanged")


def main():
    """Run all tests."""
    print("Running ModelManager backend integration tests...\n")
    
    tests = [
        test_default_pytorch_backend,
        test_explicit_backend_selection,
        test_backend_config_from_file,
        test_backend_config_from_dict,
        test_per_model_backend_override,
        test_cache_info_includes_backend,
        test_backward_compatibility,
        test_api_signatures,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
