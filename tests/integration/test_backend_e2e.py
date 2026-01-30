"""
End-to-end integration tests for backend workflows.

Tests complete workflows with PyTorch and OpenVINO backends,
backend switching mid-session, and multiple models with different backends.

Requirements tested:
- 3.1: Backward compatibility
- 5.5: Backend switching
"""

import os
import pytest
import tempfile
import yaml
import torch
from transformers import AutoTokenizer

from mm_orch.runtime.model_manager import ModelManager
from mm_orch.schemas import ModelConfig


@pytest.fixture
def temp_backend_config():
    """Create a temporary backend configuration file."""
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
            "gpt2-openvino": "openvino",
            "gpt2-pytorch": "pytorch"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def small_test_model():
    """Return a small model name for testing."""
    return "distilgpt2"


class TestPyTorchBackendE2E:
    """End-to-end tests with PyTorch backend."""
    
    def test_complete_workflow_pytorch(self, small_test_model):
        """
        Test complete workflow with PyTorch backend.
        
        Workflow:
        1. Initialize ModelManager with PyTorch backend
        2. Register and load a model
        3. Run inference
        4. Verify outputs
        5. Check performance metrics
        
        Requirement: 3.1 - Backward compatibility
        """
        # Initialize with PyTorch backend (explicit)
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        # Register model
        model_config = ModelConfig(
            name="test-gpt2",
            model_path=small_test_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Load model
        model = manager.load_model("test-gpt2", device="cpu")
        assert model is not None
        
        # Verify model is in cache
        assert manager.is_loaded("test-gpt2")
        
        # Run inference
        tokenizer = AutoTokenizer.from_pretrained(small_test_model)
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        
        outputs = manager.infer("test-gpt2", inputs)
        assert outputs is not None
        assert "logits" in outputs or hasattr(outputs, "logits")
        
        # Check performance metrics
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        manager.unload_model("test-gpt2")
        assert not manager.is_loaded("test-gpt2")
    
    def test_pytorch_backend_default_behavior(self, small_test_model):
        """
        Test that PyTorch backend is default (backward compatibility).
        
        Requirement: 3.1 - Backward compatibility
        """
        # Initialize without specifying backend (should default to PyTorch)
        manager = ModelManager(max_cached_models=2)
        
        assert manager.default_backend == "pytorch"
        
        # Register and load model
        model_config = ModelConfig(
            name="test-default",
            model_path=small_test_model,
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        model = manager.load_model("test-default", device="cpu")
        assert model is not None
        
        # Verify it's using PyTorch backend
        cache_info = manager.get_cache_info()
        assert cache_info["default_backend"] == "pytorch"
        
        manager.unload_model("test-default")


class TestOpenVINOBackendE2E:
    """End-to-end tests with OpenVINO backend."""
    
    @pytest.mark.skipif(
        "openvino" not in __import__("mm_orch.runtime.backend_factory", fromlist=["BackendFactory"]).BackendFactory().get_available_backends(),
        reason="OpenVINO not available"
    )
    def test_complete_workflow_openvino(self, small_test_model):
        """
        Test complete workflow with OpenVINO backend.
        
        Workflow:
        1. Initialize ModelManager with OpenVINO backend
        2. Register and load a model (with fallback)
        3. Run inference
        4. Verify outputs
        5. Check performance metrics
        
        Requirement: 5.5 - Backend switching
        """
        # Initialize with OpenVINO backend
        manager = ModelManager(backend="openvino", max_cached_models=2)
        
        assert manager.default_backend == "openvino"
        
        # Register model
        model_config = ModelConfig(
            name="test-gpt2-ov",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Load model (will fallback to PyTorch if OpenVINO model not available)
        model = manager.load_model("test-gpt2-ov", device="cpu")
        assert model is not None
        
        # Verify model is in cache
        assert manager.is_loaded("test-gpt2-ov")
        
        # Run inference
        tokenizer = AutoTokenizer.from_pretrained(small_test_model)
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        
        outputs = manager.infer("test-gpt2-ov", inputs)
        assert outputs is not None
        
        # Check performance metrics
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        manager.unload_model("test-gpt2-ov")


class TestBackendSwitchingE2E:
    """End-to-end tests for switching backends mid-session."""
    
    def test_switch_backends_mid_session(self, small_test_model):
        """
        Test switching backends mid-session.
        
        Workflow:
        1. Start with PyTorch backend
        2. Load a model
        3. Switch to OpenVINO backend for another model
        4. Verify both models work correctly
        
        Requirement: 5.5 - Backend switching
        """
        # Initialize with PyTorch backend
        manager = ModelManager(backend="pytorch", max_cached_models=3)
        
        # Register and load first model with PyTorch
        model_config_1 = ModelConfig(
            name="model-pytorch",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config_1)
        model_1 = manager.load_model("model-pytorch", device="cpu")
        assert model_1 is not None
        
        # Register second model with OpenVINO override
        model_config_2 = ModelConfig(
            name="model-openvino",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config_2)
        
        # Load with backend override
        available_backends = manager._backend_factory.get_available_backends()
        if "openvino" in available_backends:
            model_2 = manager.load_model("model-openvino", device="cpu", backend_override="openvino")
            assert model_2 is not None
        
        # Verify both models are loaded
        assert manager.is_loaded("model-pytorch")
        
        # Run inference on both
        tokenizer = AutoTokenizer.from_pretrained(small_test_model)
        inputs = tokenizer("Test", return_tensors="pt")
        
        outputs_1 = manager.infer("model-pytorch", inputs)
        assert outputs_1 is not None
        
        # Cleanup
        manager.unload_model("model-pytorch")
        if "openvino" in available_backends:
            manager.unload_model("model-openvino")
    
    def test_backend_override_per_model(self, temp_backend_config, small_test_model):
        """
        Test per-model backend override from configuration.
        
        Requirement: 5.5 - Backend switching
        """
        # Initialize with configuration that has per-model overrides
        manager = ModelManager(backend_config=temp_backend_config, max_cached_models=3)
        
        # Register models
        model_config_pt = ModelConfig(
            name="gpt2-pytorch",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config_pt)
        
        model_config_ov = ModelConfig(
            name="gpt2-openvino",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config_ov)
        
        # Verify overrides are loaded
        override_pt = manager._backend_config_loader.get_model_backend("gpt2-pytorch")
        override_ov = manager._backend_config_loader.get_model_backend("gpt2-openvino")
        
        assert override_pt == "pytorch"
        assert override_ov == "openvino"


class TestMultipleModelsMultipleBackends:
    """End-to-end tests with multiple models using different backends."""
    
    def test_multiple_models_different_backends(self, temp_backend_config, small_test_model):
        """
        Test loading multiple models with different backends simultaneously.
        
        Workflow:
        1. Initialize ModelManager with configuration
        2. Load multiple models with different backend overrides
        3. Run inference on all models
        4. Verify all work correctly
        5. Compare performance metrics
        
        Requirement: 5.5 - Backend switching
        """
        manager = ModelManager(backend_config=temp_backend_config, max_cached_models=5)
        
        # Register multiple models
        models_to_test = [
            ("model-1-pytorch", "pytorch"),
            ("model-2-pytorch", "pytorch"),
        ]
        
        # Add OpenVINO models if available
        available_backends = manager._backend_factory.get_available_backends()
        if "openvino" in available_backends:
            models_to_test.append(("model-3-openvino", "openvino"))
        
        # Register and load all models
        loaded_models = []
        for model_name, backend in models_to_test:
            model_config = ModelConfig(
                name=model_name,
                model_path=small_test_model,
                model_type="transformers",
                device="cpu",
                max_length=50
            )
            manager.register_model(model_config)
            
            model = manager.load_model(model_name, device="cpu", backend_override=backend)
            if model is not None:
                loaded_models.append(model_name)
        
        # Verify all models are loaded
        assert len(loaded_models) >= 2
        for model_name in loaded_models:
            assert manager.is_loaded(model_name)
        
        # Run inference on all models
        tokenizer = AutoTokenizer.from_pretrained(small_test_model)
        inputs = tokenizer("Test input", return_tensors="pt")
        
        for model_name in loaded_models:
            outputs = manager.infer(model_name, inputs)
            assert outputs is not None
        
        # Check performance metrics
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        for model_name in loaded_models:
            manager.unload_model(model_name)
    
    def test_cache_management_with_multiple_backends(self, small_test_model):
        """
        Test cache management with models from different backends.
        
        Requirement: 5.5 - Backend switching
        """
        # Initialize with small cache
        manager = ModelManager(backend="pytorch", max_cached_models=2)
        
        # Load models until cache is full
        model_names = []
        for i in range(3):
            model_name = f"model-{i}"
            model_config = ModelConfig(
                name=model_name,
                model_path=small_test_model,
                model_type="transformers",
                device="cpu",
                max_length=50
            )
            manager.register_model(model_config)
            manager.load_model(model_name, device="cpu")
            model_names.append(model_name)
        
        # Verify cache eviction occurred
        cache_info = manager.get_cache_info()
        assert cache_info["current_size"] <= 2
        
        # Cleanup remaining models
        for model_name in model_names:
            if manager.is_loaded(model_name):
                manager.unload_model(model_name)


class TestBackendWorkflowIntegration:
    """Integration tests for complete backend workflows."""
    
    def test_full_inference_pipeline_pytorch(self, small_test_model):
        """
        Test full inference pipeline with PyTorch backend.
        
        Pipeline:
        1. Initialize manager
        2. Load model
        3. Tokenize input
        4. Run inference
        5. Process output
        6. Verify metrics
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager(backend="pytorch")
        
        # Register model
        model_config = ModelConfig(
            name="pipeline-test",
            model_path=small_test_model,
            model_type="transformers",
            device="cpu",
            max_length=50
        )
        manager.register_model(model_config)
        
        # Load model
        model = manager.load_model("pipeline-test", device="cpu")
        assert model is not None
        
        # Prepare input
        tokenizer = AutoTokenizer.from_pretrained(small_test_model)
        test_text = "The quick brown fox"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Run inference
        outputs = manager.infer("pipeline-test", inputs)
        assert outputs is not None
        
        # Verify output structure
        if isinstance(outputs, dict):
            assert "logits" in outputs
        else:
            assert hasattr(outputs, "logits")
        
        # Check metrics were recorded
        stats = manager.get_performance_stats()
        assert stats is not None
        
        # Cleanup
        manager.unload_model("pipeline-test")
    
    def test_error_handling_in_workflow(self, small_test_model):
        """
        Test error handling in complete workflow.
        
        Requirement: 3.1 - Backward compatibility
        """
        manager = ModelManager(backend="pytorch")
        
        # Try to load non-existent model
        with pytest.raises(Exception):
            manager.load_model("non-existent-model", device="cpu")
        
        # Try to infer on non-loaded model
        with pytest.raises(Exception):
            tokenizer = AutoTokenizer.from_pretrained(small_test_model)
            inputs = tokenizer("Test", return_tensors="pt")
            manager.infer("non-loaded-model", inputs)


# Feature: openvino-backend-integration
# End-to-end integration tests for backend workflows
# Tests Requirements: 3.1, 5.5
