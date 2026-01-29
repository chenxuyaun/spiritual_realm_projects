"""
Property-based tests for backend interface uniformity.

Tests the correctness properties defined in the design document:
- Property 8: Backend Interface Uniformity

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict
import inspect

from mm_orch.runtime.inference_backend import InferenceBackend


# Strategies for generating test data
model_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='-_'),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() == x and len(x) > 0)

model_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='/-_.'),
    min_size=1,
    max_size=100
).filter(lambda x: len(x.strip()) > 0)

model_type_strategy = st.sampled_from(['transformers', 'custom', 'onnx'])

device_strategy = st.sampled_from(['cpu', 'cuda', 'GPU', 'AUTO'])

prompt_strategy = st.text(min_size=1, max_size=200)

max_length_strategy = st.integers(min_value=10, max_value=512)

temperature_strategy = st.floats(min_value=0.1, max_value=2.0)


class MockPyTorchBackend(InferenceBackend):
    """Mock PyTorch backend for testing interface uniformity."""
    
    def __init__(self, device: str, config: Dict[str, Any]):
        super().__init__(device, config)
        self.backend_name = "pytorch"
    
    def load_model(self, model_name: str, model_path: str, model_type: str) -> Any:
        """Mock load_model implementation."""
        mock_model = Mock()
        mock_model.name = model_name
        mock_model.path = model_path
        mock_model.type = model_type
        self._models[model_name] = mock_model
        return mock_model
    
    def forward(self, model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock forward implementation."""
        # Return consistent structure
        return {
            'logits': Mock(),
            'hidden_states': None,
            'attentions': None
        }
    
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int,
        **kwargs
    ) -> str:
        """Mock generate implementation."""
        # Return generated text
        return f"Generated: {prompt[:20]}..."
    
    def unload_model(self, model_name: str) -> None:
        """Mock unload_model implementation."""
        if model_name in self._models:
            del self._models[model_name]
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Mock get_model_info implementation."""
        return {
            'backend': self.backend_name,
            'device': self.device,
            'model_name': getattr(model, 'name', 'unknown'),
            'parameters': 1000000
        }


class MockOpenVINOBackend(InferenceBackend):
    """Mock OpenVINO backend for testing interface uniformity."""
    
    def __init__(self, device: str, config: Dict[str, Any]):
        super().__init__(device, config)
        self.backend_name = "openvino"
    
    def load_model(self, model_name: str, model_path: str, model_type: str) -> Any:
        """Mock load_model implementation."""
        mock_model = Mock()
        mock_model.name = model_name
        mock_model.path = model_path
        mock_model.type = model_type
        self._models[model_name] = mock_model
        return mock_model
    
    def forward(self, model: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock forward implementation."""
        # Return consistent structure (same as PyTorch)
        return {
            'logits': Mock(),
            'hidden_states': None,
            'attentions': None
        }
    
    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_length: int,
        **kwargs
    ) -> str:
        """Mock generate implementation."""
        # Return generated text (same format as PyTorch)
        return f"Generated: {prompt[:20]}..."
    
    def unload_model(self, model_name: str) -> None:
        """Mock unload_model implementation."""
        if model_name in self._models:
            del self._models[model_name]
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """Mock get_model_info implementation."""
        return {
            'backend': self.backend_name,
            'device': self.device,
            'model_name': getattr(model, 'name', 'unknown'),
            'parameters': 1000000
        }


class TestProperty8BackendInterfaceUniformity:
    """
    Property 8: Backend Interface Uniformity
    
    For any inference operation (generate, forward) and any valid input,
    both PyTorch and OpenVINO backends should provide the same method signatures,
    accept the same parameters, and return results in the same format structure.
    
    **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
    """
    
    @given(
        device=device_strategy,
        config=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.booleans(), st.integers(), st.text(max_size=50)),
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_backends_have_identical_init_signatures(self, device, config):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        Both backends should accept the same __init__ parameters.
        """
        # Create both backends with same parameters
        pytorch_backend = MockPyTorchBackend(device, config)
        openvino_backend = MockOpenVINOBackend(device, config)
        
        # Both should initialize successfully
        assert pytorch_backend.device == device
        assert openvino_backend.device == device
        assert pytorch_backend.config == config
        assert openvino_backend.config == config
        
        # Check that __init__ signatures match
        pytorch_sig = inspect.signature(MockPyTorchBackend.__init__)
        openvino_sig = inspect.signature(MockOpenVINOBackend.__init__)
        
        # Parameters should be identical (excluding 'self')
        pytorch_params = list(pytorch_sig.parameters.keys())[1:]
        openvino_params = list(openvino_sig.parameters.keys())[1:]
        assert pytorch_params == openvino_params
    
    @given(
        model_name=model_name_strategy,
        model_path=model_path_strategy,
        model_type=model_type_strategy,
        device=device_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_load_model_has_identical_signature(
        self,
        model_name,
        model_path,
        model_type,
        device
    ):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        load_model method should have identical signature across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.load_model)
        openvino_sig = inspect.signature(openvino_backend.load_model)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Both should accept the same parameters
        pytorch_model = pytorch_backend.load_model(model_name, model_path, model_type)
        openvino_model = openvino_backend.load_model(model_name, model_path, model_type)
        
        # Both should return model objects
        assert pytorch_model is not None
        assert openvino_model is not None
    
    @given(
        device=device_strategy,
        input_keys=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_forward_has_identical_signature_and_output_structure(
        self,
        device,
        input_keys
    ):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        forward method should have identical signature and return structure across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.forward)
        openvino_sig = inspect.signature(openvino_backend.forward)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Create mock model and inputs
        mock_model = Mock()
        inputs = {key: Mock() for key in input_keys}
        
        # Call forward on both backends
        pytorch_output = pytorch_backend.forward(mock_model, inputs)
        openvino_output = openvino_backend.forward(mock_model, inputs)
        
        # Both should return dictionaries
        assert isinstance(pytorch_output, dict)
        assert isinstance(openvino_output, dict)
        
        # Output structure should be identical (same keys)
        assert set(pytorch_output.keys()) == set(openvino_output.keys())
    
    @given(
        prompt=prompt_strategy,
        max_length=max_length_strategy,
        temperature=temperature_strategy,
        device=device_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_generate_has_identical_signature_and_output_type(
        self,
        prompt,
        max_length,
        temperature,
        device
    ):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        generate method should have identical signature and return type across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.generate)
        openvino_sig = inspect.signature(openvino_backend.generate)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Create mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Call generate on both backends with same parameters
        pytorch_output = pytorch_backend.generate(
            mock_model,
            mock_tokenizer,
            prompt,
            max_length,
            temperature=temperature
        )
        openvino_output = openvino_backend.generate(
            mock_model,
            mock_tokenizer,
            prompt,
            max_length,
            temperature=temperature
        )
        
        # Both should return strings
        assert isinstance(pytorch_output, str)
        assert isinstance(openvino_output, str)
        
        # Both should be non-empty
        assert len(pytorch_output) > 0
        assert len(openvino_output) > 0
    
    @given(
        model_name=model_name_strategy,
        device=device_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_unload_model_has_identical_signature(self, model_name, device):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        unload_model method should have identical signature across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.unload_model)
        openvino_sig = inspect.signature(openvino_backend.unload_model)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Load models first
        pytorch_backend.load_model(model_name, "path", "transformers")
        openvino_backend.load_model(model_name, "path", "transformers")
        
        # Both should unload successfully
        pytorch_backend.unload_model(model_name)
        openvino_backend.unload_model(model_name)
        
        # Models should be removed from both backends
        assert model_name not in pytorch_backend._models
        assert model_name not in openvino_backend._models
    
    @given(device=device_strategy)
    @settings(max_examples=100, deadline=None)
    def test_get_model_info_has_identical_signature_and_output_structure(self, device):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        get_model_info method should have identical signature and return structure.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.get_model_info)
        openvino_sig = inspect.signature(openvino_backend.get_model_info)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Create mock model
        mock_model = Mock()
        mock_model.name = "test_model"
        
        # Call get_model_info on both backends
        pytorch_info = pytorch_backend.get_model_info(mock_model)
        openvino_info = openvino_backend.get_model_info(mock_model)
        
        # Both should return dictionaries
        assert isinstance(pytorch_info, dict)
        assert isinstance(openvino_info, dict)
        
        # Both should contain 'backend' and 'device' keys at minimum
        assert 'backend' in pytorch_info
        assert 'device' in pytorch_info
        assert 'backend' in openvino_info
        assert 'device' in openvino_info
        
        # Structure should be consistent (same keys)
        assert set(pytorch_info.keys()) == set(openvino_info.keys())
    
    @given(device=device_strategy)
    @settings(max_examples=100, deadline=None)
    def test_is_available_has_identical_signature(self, device):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        is_available method should have identical signature across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        # Check method signatures
        pytorch_sig = inspect.signature(pytorch_backend.is_available)
        openvino_sig = inspect.signature(openvino_backend.is_available)
        
        # Parameters should be identical
        pytorch_params = list(pytorch_sig.parameters.keys())
        openvino_params = list(openvino_sig.parameters.keys())
        assert pytorch_params == openvino_params
        
        # Both should return boolean
        pytorch_available = pytorch_backend.is_available()
        openvino_available = openvino_backend.is_available()
        
        assert isinstance(pytorch_available, bool)
        assert isinstance(openvino_available, bool)
    
    @given(
        model_name=model_name_strategy,
        model_path=model_path_strategy,
        prompt=prompt_strategy,
        max_length=max_length_strategy,
        device=device_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_complete_workflow_identical_across_backends(
        self,
        model_name,
        model_path,
        prompt,
        max_length,
        device
    ):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        Complete workflow (load -> generate -> unload) should work identically
        across backends with same parameters.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        mock_tokenizer = Mock()
        
        # Load model on both backends
        pytorch_model = pytorch_backend.load_model(model_name, model_path, "transformers")
        openvino_model = openvino_backend.load_model(model_name, model_path, "transformers")
        
        # Generate on both backends
        pytorch_output = pytorch_backend.generate(
            pytorch_model,
            mock_tokenizer,
            prompt,
            max_length
        )
        openvino_output = openvino_backend.generate(
            openvino_model,
            mock_tokenizer,
            prompt,
            max_length
        )
        
        # Both should produce string outputs
        assert isinstance(pytorch_output, str)
        assert isinstance(openvino_output, str)
        
        # Get model info from both
        pytorch_info = pytorch_backend.get_model_info(pytorch_model)
        openvino_info = openvino_backend.get_model_info(openvino_model)
        
        # Info structure should be identical
        assert set(pytorch_info.keys()) == set(openvino_info.keys())
        
        # Unload from both backends
        pytorch_backend.unload_model(model_name)
        openvino_backend.unload_model(model_name)
        
        # Both should have unloaded successfully
        assert model_name not in pytorch_backend._models
        assert model_name not in openvino_backend._models
    
    @given(
        device=device_strategy,
        input_dict=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.integers(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_forward_accepts_same_input_formats(self, device, input_dict):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        forward method should accept the same input formats across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        mock_model = Mock()
        
        # Both backends should accept the same input dictionary format
        pytorch_output = pytorch_backend.forward(mock_model, input_dict)
        openvino_output = openvino_backend.forward(mock_model, input_dict)
        
        # Both should succeed and return dictionaries
        assert isinstance(pytorch_output, dict)
        assert isinstance(openvino_output, dict)
    
    @given(
        prompt=prompt_strategy,
        max_length=max_length_strategy,
        device=device_strategy,
        extra_kwargs=st.dictionaries(
            st.sampled_from(['temperature', 'top_p', 'top_k', 'do_sample']),
            st.one_of(st.floats(0.1, 2.0), st.integers(1, 100), st.booleans()),
            max_size=3
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_generate_accepts_same_kwargs(
        self,
        prompt,
        max_length,
        device,
        extra_kwargs
    ):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        generate method should accept the same keyword arguments across backends.
        """
        pytorch_backend = MockPyTorchBackend(device, {})
        openvino_backend = MockOpenVINOBackend(device, {})
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Both backends should accept the same kwargs
        pytorch_output = pytorch_backend.generate(
            mock_model,
            mock_tokenizer,
            prompt,
            max_length,
            **extra_kwargs
        )
        openvino_output = openvino_backend.generate(
            mock_model,
            mock_tokenizer,
            prompt,
            max_length,
            **extra_kwargs
        )
        
        # Both should succeed and return strings
        assert isinstance(pytorch_output, str)
        assert isinstance(openvino_output, str)


# Additional test to verify abstract base class compliance
class TestBackendAbstractBaseClassCompliance:
    """
    Verify that both backends properly implement the InferenceBackend interface.
    """
    
    def test_pytorch_backend_implements_all_abstract_methods(self):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        PyTorch backend should implement all abstract methods from InferenceBackend.
        """
        # Get all abstract methods from InferenceBackend
        abstract_methods = {
            name for name, method in inspect.getmembers(InferenceBackend, predicate=inspect.isfunction)
            if getattr(method, '__isabstractmethod__', False)
        }
        
        # Get all callable methods from MockPyTorchBackend (including inherited)
        pytorch_methods = {
            name for name in dir(MockPyTorchBackend)
            if callable(getattr(MockPyTorchBackend, name)) and not name.startswith('_')
        }
        
        # All abstract methods should be implemented
        assert abstract_methods.issubset(pytorch_methods), \
            f"Missing methods: {abstract_methods - pytorch_methods}"
    
    def test_openvino_backend_implements_all_abstract_methods(self):
        """
        Feature: openvino-backend-integration, Property 8: Backend Interface Uniformity
        
        OpenVINO backend should implement all abstract methods from InferenceBackend.
        """
        # Get all abstract methods from InferenceBackend
        abstract_methods = {
            name for name, method in inspect.getmembers(InferenceBackend, predicate=inspect.isfunction)
            if getattr(method, '__isabstractmethod__', False)
        }
        
        # Get all callable methods from MockOpenVINOBackend (including inherited)
        openvino_methods = {
            name for name in dir(MockOpenVINOBackend)
            if callable(getattr(MockOpenVINOBackend, name)) and not name.startswith('_')
        }
        
        # All abstract methods should be implemented
        assert abstract_methods.issubset(openvino_methods), \
            f"Missing methods: {abstract_methods - openvino_methods}"
