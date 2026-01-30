"""
Backward compatibility validation tests for OpenVINO backend integration.

These tests verify that existing code continues to work without modification
after the OpenVINO backend integration.

Requirements tested:
- 3.1: Backward compatibility - existing code works without changes
- 3.2: API signatures unchanged
- 3.5: No breaking changes
"""

import pytest
import torch
from transformers import AutoTokenizer

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


class TestExistingAPICompatibility:
    """Test that existing API remains unchanged."""
    
    def test_model_manager_initialization_without_backend(self):
        """
        Test ModelManager initialization without backend parameter.
        
        This simulates existing code that doesn't know about backends.
        
        Requirement: 3.1 - Backward compatibility
        """
        # Old code pattern - no backend parameter
        manager = ModelManager(max_cached_models=3, default_device="cpu")
        
        # Should default to PyTorch
        assert manager.default_backend == "pytorch"
        assert hasattr(manager, '_backend_factory')
    
    def test_model_manager_initialization_with_legacy_params(self):
        """
        Test ModelManager with only legacy parameters.
        
        Requirement: 3.1 - Backward compatibility
        """
        # Old initialization patterns
        manager1 = ModelManager()
        assert manager1.default_backend == "pytorch"
        
        manager2 = ModelManager(max_cached_models=5)
        assert manager2.default_backend == "pytorch"
        
        manager3 = ModelManager(default_device="cpu")
        assert manager3.default_backend == "pytorch"
        
        manager4 = ModelManager(max_cached_models=2, default_device="cpu")
        assert manager4.default_backend == "pytorch"
    
    def test_load_model_signature_backward_compatible(self):
        """
        Test that load_model accepts old signature.
        
        Requirement: 3.2 - API signatures unchanged
        """
        manager = ModelManager()
        
        # Register a test model
        model_config = ModelConfig(
            name="test-model",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Old signature - without backend_override
        model = manager.load_model("test-model", device="cpu")
        assert model is not None
        
        # Cleanup
        manager.unload_model("test-model")
    
    def test_all_existing_methods_present(self):
        """
        Test that all existing methods are still present.
        
        Requirement: 3.2 - API signatures unchanged
        """
        manager = ModelManager()
        
        # Verify all expected methods exist
        expected_methods = [
            'load_model',
            'get_model',
            'unload_model',
            'infer',
            'register_model',
            'is_registered',
            'is_loaded',
            'get_cache_info',
            'clear_cache',
            'get_performance_stats'
        ]
        
        for method_name in expected_methods:
            assert hasattr(manager, method_name), f"Missing method: {method_name}"
            assert callable(getattr(manager, method_name)), f"Not callable: {method_name}"
    
    def test_method_signatures_unchanged(self):
        """
        Test that method signatures are backward compatible.
        
        Requirement: 3.2 - API signatures unchanged
        """
        import inspect
        
        manager = ModelManager()
        
        # Check load_model signature
        load_model_sig = inspect.signature(manager.load_model)
        load_model_params = list(load_model_sig.parameters.keys())
        assert 'model_name' in load_model_params
        assert 'device' in load_model_params
        # backend_override is optional, so old code doesn't need it
        
        # Check infer signature
        infer_sig = inspect.signature(manager.infer)
        infer_params = list(infer_sig.parameters.keys())
        assert 'model_name' in infer_params
        assert 'inputs' in infer_params
        
        # Check get_model signature
        get_model_sig = inspect.signature(manager.get_model)
        get_model_params = list(get_model_sig.parameters.keys())
        assert 'model_name' in get_model_params


class TestExistingWorkflowsUnchanged:
    """Test that existing workflows work without modification."""
    
    def test_basic_model_loading_workflow(self):
        """
        Test basic model loading workflow (existing pattern).
        
        Requirement: 3.1 - Backward compatibility
        """
        # Existing workflow pattern
        manager = ModelManager(max_cached_models=2)
        
        model_config = ModelConfig(
            name="workflow-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        
        # Register model
        manager.register_model(model_config)
        assert manager.is_registered("workflow-test")
        
        # Load model
        model = manager.load_model("workflow-test", device="cpu")
        assert model is not None
        assert manager.is_loaded("workflow-test")
        
        # Get model
        retrieved_model = manager.get_model("workflow-test")
        assert retrieved_model is not None
        
        # Unload model
        manager.unload_model("workflow-test")
        assert not manager.is_loaded("workflow-test")
    
    def test_inference_workflow_unchanged(self):
        """
        Test inference workflow (existing pattern).
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager()
        
        # Register and load model
        model_config = ModelConfig(
            name="inference-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        model = manager.load_model("inference-test", device="cpu")
        
        # Prepare inputs (existing pattern)
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        
        # Run inference
        outputs = manager.infer("inference-test", inputs)
        assert outputs is not None
        
        # Verify output structure
        if isinstance(outputs, dict):
            assert "logits" in outputs
        else:
            assert hasattr(outputs, "logits")
        
        # Cleanup
        manager.unload_model("inference-test")
    
    def test_cache_management_workflow(self):
        """
        Test cache management workflow (existing pattern).
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager(max_cached_models=2)
        
        # Load multiple models
        for i in range(3):
            model_config = ModelConfig(
                name=f"cache-test-{i}",
                model_path="distilgpt2",
                model_type="transformers",
                device="cpu",
                max_length=50
            )
            manager.register_model(model_config)
            manager.load_model(f"cache-test-{i}", device="cpu")
        
        # Check cache info
        cache_info = manager.get_cache_info()
        assert "current_size" in cache_info
        assert "max_size" in cache_info
        assert cache_info["current_size"] <= cache_info["max_size"]
        
        # Clear cache
        manager.clear_cache()
        cache_info_after = manager.get_cache_info()
        assert cache_info_after["current_size"] == 0


class TestExistingBehaviorPreserved:
    """Test that existing behavior is preserved."""
    
    def test_default_device_behavior(self):
        """
        Test that default device behavior is preserved.
        
        Requirement: 3.1 - Backward compatibility
        """
        # Test with default device
        manager = ModelManager(default_device="cpu")
        
        model_config = ModelConfig(
            name="device-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Load without specifying device (should use default)
        model = manager.load_model("device-test")
        assert model is not None
        
        manager.unload_model("device-test")
    
    def test_cache_eviction_behavior(self):
        """
        Test that cache eviction behavior is preserved.
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager(max_cached_models=2)
        
        # Load models to fill cache
        model_names = []
        for i in range(3):
            model_name = f"eviction-test-{i}"
            model_config = ModelConfig(
                name=model_name,
                model_path="distilgpt2",
                model_type="transformers",
                device="cpu",
                max_length=50
            )
            manager.register_model(model_config)
            manager.load_model(model_name, device="cpu")
            model_names.append(model_name)
        
        # Verify LRU eviction occurred
        cache_info = manager.get_cache_info()
        assert cache_info["current_size"] <= 2
        
        # First model should have been evicted
        assert not manager.is_loaded(model_names[0])
        
        # Cleanup
        for model_name in model_names:
            if manager.is_loaded(model_name):
                manager.unload_model(model_name)
    
    def test_error_handling_preserved(self):
        """
        Test that error handling behavior is preserved.
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager()
        
        # Test loading non-existent model
        with pytest.raises(Exception):
            manager.load_model("non-existent-model", device="cpu")
        
        # Test getting non-loaded model
        with pytest.raises(Exception):
            manager.get_model("non-loaded-model")
        
        # Test inferring on non-loaded model
        with pytest.raises(Exception):
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            inputs = tokenizer("Test", return_tensors="pt")
            manager.infer("non-loaded-model", inputs)


class TestNoBreakingChanges:
    """Test that no breaking changes were introduced."""
    
    def test_model_config_unchanged(self):
        """
        Test that ModelConfig schema is unchanged.
        
        Requirement: 3.5 - No breaking changes
        """
        # Old ModelConfig pattern should still work
        config = ModelConfig(
            name="test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        
        assert config.name == "test"
        assert config.model_path == "distilgpt2"
        assert config.model_type == "transformers"
        assert config.device == "cpu"
        assert config.max_length == 50
    
    def test_cache_info_structure_compatible(self):
        """
        Test that cache info structure is backward compatible.
        
        Requirement: 3.5 - No breaking changes
        """
        manager = ModelManager()
        
        cache_info = manager.get_cache_info()
        
        # Old fields should still exist
        assert "current_size" in cache_info
        assert "max_size" in cache_info
        assert "cached_models" in cache_info
        
        # New fields are additions, not replacements
        assert "default_backend" in cache_info
        assert "available_backends" in cache_info
    
    def test_performance_stats_structure_compatible(self):
        """
        Test that performance stats structure is backward compatible.
        
        Requirement: 3.5 - No breaking changes
        """
        manager = ModelManager()
        
        # Register and load a model
        model_config = ModelConfig(
            name="perf-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        manager.load_model("perf-test", device="cpu")
        
        # Run inference to generate stats
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        inputs = tokenizer("Test", return_tensors="pt")
        manager.infer("perf-test", inputs)
        
        # Get stats
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        manager.unload_model("perf-test")


class TestExistingCodePatterns:
    """Test common existing code patterns."""
    
    def test_simple_usage_pattern(self):
        """
        Test simple usage pattern from existing code.
        
        Requirement: 3.1 - Backward compatibility
        """
        # Simple pattern: initialize, load, infer, cleanup
        manager = ModelManager()
        
        model_config = ModelConfig(
            name="simple-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        
        manager.register_model(model_config)
        manager.load_model("simple-test", device="cpu")
        
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        inputs = tokenizer("Test", return_tensors="pt")
        outputs = manager.infer("simple-test", inputs)
        
        assert outputs is not None
        
        manager.unload_model("simple-test")
    
    def test_multiple_models_pattern(self):
        """
        Test multiple models pattern from existing code.
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager(max_cached_models=3)
        
        # Load multiple models
        model_names = ["model-a", "model-b"]
        for name in model_names:
            config = ModelConfig(
                name=name,
                model_path="distilgpt2",
                model_type="transformers",
                device="cpu",
                max_length=50
            )
            manager.register_model(config)
            manager.load_model(name, device="cpu")
        
        # Verify all loaded
        for name in model_names:
            assert manager.is_loaded(name)
        
        # Cleanup
        for name in model_names:
            manager.unload_model(name)
    
    def test_reloading_pattern(self):
        """
        Test model reloading pattern from existing code.
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager()
        
        model_config = ModelConfig(
            name="reload-test",
            model_path="distilgpt2",
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        
        # Load, unload, reload
        manager.register_model(model_config)
        manager.load_model("reload-test", device="cpu")
        assert manager.is_loaded("reload-test")
        
        manager.unload_model("reload-test")
        assert not manager.is_loaded("reload-test")
        
        manager.load_model("reload-test", device="cpu")
        assert manager.is_loaded("reload-test")
        
        manager.unload_model("reload-test")


# Feature: openvino-backend-integration
# Backward compatibility validation tests
# Tests Requirements: 3.1, 3.2, 3.5
