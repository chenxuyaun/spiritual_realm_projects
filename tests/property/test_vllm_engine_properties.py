"""
Property-based tests for vLLM engine.

Feature: advanced-optimization-monitoring
Tests Properties 1 and 3 specific to vLLM engine from the design document.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization import VLLMEngine, VLLMConfig


# Strategy for generating valid model names
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_/'
))

# Strategy for generating tensor parallel sizes
tensor_parallel_sizes = st.integers(min_value=1, max_value=8)

# Strategy for generating dtypes
dtypes = st.sampled_from(["auto", "fp16", "fp32", "bf16"])

# Strategy for generating GPU memory utilization
gpu_memory_utils = st.floats(min_value=0.1, max_value=1.0)

# Strategy for generating prompts
prompts = st.lists(
    st.text(min_size=1, max_size=100),
    min_size=1,
    max_size=10
)


# Feature: advanced-optimization-monitoring, Property 1: Engine selection respects availability (vLLM)
@given(
    vllm_enabled=st.booleans()
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_1_vllm_availability_detection(vllm_enabled):
    """
    Property 1: Engine selection respects availability (vLLM).
    
    For any vLLM configuration, the engine should correctly detect whether
    vLLM is available based on library installation and configuration.
    
    Validates: Requirements 1.1, 1.5
    """
    # Setup: Create vLLM config
    config = VLLMConfig(enabled=vllm_enabled)
    engine = VLLMEngine(config)
    
    if not vllm_enabled:
        # Verify: Engine should report unavailable when disabled
        assert engine.is_available() is False, (
            "Engine should be unavailable when disabled in configuration"
        )
    else:
        # Verify: Availability check should not raise exceptions
        try:
            is_available = engine.is_available()
            assert isinstance(is_available, bool), (
                "is_available() should return a boolean"
            )
        except Exception as e:
            pytest.fail(f"is_available() should not raise exceptions: {e}")


# Feature: advanced-optimization-monitoring, Property 3: Engine configuration is applied correctly (vLLM)
@given(
    tensor_parallel_size=tensor_parallel_sizes,
    dtype=dtypes,
    gpu_memory_utilization=gpu_memory_utils
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_3_vllm_configuration_applied(
    tensor_parallel_size, dtype, gpu_memory_utilization
):
    """
    Property 3: Engine configuration is applied correctly (vLLM).
    
    For any valid vLLM configuration, the engine should be initialized with
    the specified tensor parallelism, dtype, and memory settings.
    
    Validates: Requirements 1.5
    """
    # Setup: Create vLLM config with specific parameters
    config = VLLMConfig(
        enabled=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization
    )
    engine = VLLMEngine(config)
    
    # Verify: Configuration is stored correctly
    assert engine.config.tensor_parallel_size == tensor_parallel_size, (
        f"Expected tensor_parallel_size={tensor_parallel_size}, "
        f"got {engine.config.tensor_parallel_size}"
    )
    assert engine.config.dtype == dtype, (
        f"Expected dtype={dtype}, got {engine.config.dtype}"
    )
    assert engine.config.gpu_memory_utilization == gpu_memory_utilization, (
        f"Expected gpu_memory_utilization={gpu_memory_utilization}, "
        f"got {engine.config.gpu_memory_utilization}"
    )


# Additional property: Model loading respects configuration
@given(
    model_name=model_names,
    tensor_parallel_size=tensor_parallel_sizes,
    dtype=dtypes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_model_loading_respects_configuration(
    model_name, tensor_parallel_size, dtype
):
    """
    Additional property: Model loading uses configured parameters.
    
    For any model and configuration, load_model should attempt to use the
    configured tensor parallelism and dtype settings.
    """
    config = VLLMConfig(
        enabled=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype
    )
    engine = VLLMEngine(config)
    
    # Skip test if vLLM is not installed
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")
    
    # Mock vLLM availability and LLM class
    with patch.object(engine, 'is_available', return_value=True):
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            
            # Execute: Load model
            result = engine.load_model(model_name)
            
            # Verify: LLM was initialized with correct parameters
            if result:
                mock_llm_class.assert_called_once()
                call_kwargs = mock_llm_class.call_args[1]
                
                assert call_kwargs['model'] == model_name
                assert call_kwargs['tensor_parallel_size'] == tensor_parallel_size
                assert call_kwargs['dtype'] == dtype
                assert call_kwargs['gpu_memory_utilization'] == config.gpu_memory_utilization


# Additional property: Model loading with overrides
@given(
    model_name=model_names,
    config_tp=tensor_parallel_sizes,
    override_tp=tensor_parallel_sizes,
    config_dtype=dtypes,
    override_dtype=dtypes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_model_loading_overrides_work(
    model_name, config_tp, override_tp, config_dtype, override_dtype
):
    """
    Additional property: Model loading overrides take precedence.
    
    For any model loading with override parameters, the overrides should
    take precedence over configuration defaults.
    """
    # Skip test if vLLM is not installed
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")
    
    config = VLLMConfig(
        enabled=True,
        tensor_parallel_size=config_tp,
        dtype=config_dtype
    )
    engine = VLLMEngine(config)
    
    # Mock vLLM availability and LLM class
    with patch.object(engine, 'is_available', return_value=True):
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            
            # Execute: Load model with overrides
            result = engine.load_model(
                model_name,
                tensor_parallel_size=override_tp,
                dtype=override_dtype
            )
            
            # Verify: Overrides were used
            if result:
                mock_llm_class.assert_called_once()
                call_kwargs = mock_llm_class.call_args[1]
                
                assert call_kwargs['tensor_parallel_size'] == override_tp, (
                    f"Expected override tensor_parallel_size={override_tp}, "
                    f"got {call_kwargs['tensor_parallel_size']}"
                )
                assert call_kwargs['dtype'] == override_dtype, (
                    f"Expected override dtype={override_dtype}, "
                    f"got {call_kwargs['dtype']}"
                )


# Additional property: Generate requires loaded model
@given(
    prompts_list=prompts
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_generate_requires_loaded_model(prompts_list):
    """
    Additional property: Generate fails gracefully without loaded model.
    
    For any generate call without a loaded model, the engine should raise
    a clear error message.
    """
    # Setup: Create engine without loading model
    config = VLLMConfig(enabled=True)
    engine = VLLMEngine(config)
    
    # Verify: No model is loaded
    assert engine.get_loaded_model() is None
    
    # Execute and verify: Generate should raise RuntimeError
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.generate(prompts_list)


# Additional property: Unload clears model state
@given(
    model_name=model_names
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_unload_clears_model_state(model_name):
    """
    Additional property: Unload clears model state.
    
    For any loaded model, calling unload_model should clear the model state
    and allow loading a different model.
    """
    # Skip test if vLLM is not installed
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")
    
    config = VLLMConfig(enabled=True)
    engine = VLLMEngine(config)
    
    # Mock vLLM availability and LLM class
    with patch.object(engine, 'is_available', return_value=True):
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            
            # Execute: Load and unload model
            engine.load_model(model_name)
            assert engine.get_loaded_model() == model_name
            
            engine.unload_model()
            
            # Verify: Model state is cleared
            assert engine.get_loaded_model() is None, (
                "get_loaded_model() should return None after unload"
            )
            assert engine._llm is None, (
                "_llm should be None after unload"
            )


# Additional property: Configuration validation
@given(
    tensor_parallel_size=st.integers(min_value=-10, max_value=0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_tensor_parallel_rejected(tensor_parallel_size):
    """
    Additional property: Invalid tensor parallelism is rejected.
    
    For any invalid tensor_parallel_size (< 1), configuration creation
    should raise a ValueError.
    """
    # Execute and verify: Invalid config should raise ValueError
    with pytest.raises(ValueError, match="tensor_parallel_size must be >= 1"):
        VLLMConfig(tensor_parallel_size=tensor_parallel_size)


@given(
    gpu_memory_utilization=st.floats(min_value=-1.0, max_value=-0.01) | st.floats(min_value=1.01, max_value=2.0)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_gpu_memory_rejected(gpu_memory_utilization):
    """
    Additional property: Invalid GPU memory utilization is rejected.
    
    For any invalid gpu_memory_utilization (< 0 or > 1), configuration
    creation should raise a ValueError.
    """
    # Execute and verify: Invalid config should raise ValueError
    with pytest.raises(ValueError, match="gpu_memory_utilization must be between 0.0 and 1.0"):
        VLLMConfig(gpu_memory_utilization=gpu_memory_utilization)


@given(
    dtype=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in {"auto", "fp16", "fp32", "bf16"}
    )
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_dtype_rejected(dtype):
    """
    Additional property: Invalid dtype is rejected.
    
    For any invalid dtype (not in allowed set), configuration creation
    should raise a ValueError.
    """
    # Execute and verify: Invalid config should raise ValueError
    with pytest.raises(ValueError, match="dtype must be one of"):
        VLLMConfig(dtype=dtype)


# Additional property: Generate output format consistency
@given(
    num_prompts=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_generate_output_format_consistency(num_prompts):
    """
    Additional property: Generate returns consistent output format.
    
    For any number of prompts, generate should return a list of dictionaries
    with consistent keys (text, prompt, finish_reason, tokens_generated).
    """
    # Skip test if vLLM is not installed
    try:
        import vllm  # noqa: F401
    except ImportError:
        pytest.skip("vLLM not installed")
    
    config = VLLMConfig(enabled=True)
    engine = VLLMEngine(config)
    
    # Mock vLLM availability and LLM class
    with patch.object(engine, 'is_available', return_value=True):
        with patch('vllm.LLM') as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance
            
            # Mock vLLM output format
            mock_vllm_outputs = []
            for i in range(num_prompts):
                mock_output = MagicMock()
                mock_output.prompt = f"prompt_{i}"
                mock_output.outputs = [MagicMock()]
                mock_output.outputs[0].text = f"generated_text_{i}"
                mock_output.outputs[0].finish_reason = "stop"
                mock_output.outputs[0].token_ids = list(range(10))
                mock_vllm_outputs.append(mock_output)
            
            mock_llm_instance.generate.return_value = mock_vllm_outputs
            
            # Execute: Load model and generate
            engine.load_model("test-model")
            prompts_list = [f"prompt_{i}" for i in range(num_prompts)]
            outputs = engine.generate(prompts_list)
            
            # Verify: Output format is consistent
            assert len(outputs) == num_prompts, (
                f"Expected {num_prompts} outputs, got {len(outputs)}"
            )
            
            required_keys = {"text", "prompt", "finish_reason", "tokens_generated"}
            for i, output in enumerate(outputs):
                assert isinstance(output, dict), (
                    f"Output {i} should be a dictionary"
                )
                assert required_keys.issubset(output.keys()), (
                    f"Output {i} missing required keys. "
                    f"Expected {required_keys}, got {output.keys()}"
                )
                assert isinstance(output["text"], str)
                assert isinstance(output["prompt"], str)
                assert isinstance(output["tokens_generated"], int)
