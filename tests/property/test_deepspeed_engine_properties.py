"""
Property-based tests for DeepSpeed engine.

Feature: advanced-optimization-monitoring
Tests Properties 1 and 3 specific to DeepSpeed engine from the design document.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, patch, MagicMock
import torch

from mm_orch.optimization import DeepSpeedEngine, DeepSpeedConfig


# Strategy for generating valid model names
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_/'
))

# Strategy for generating tensor parallel sizes
tensor_parallel_sizes = st.integers(min_value=1, max_value=8)

# Strategy for generating pipeline parallel sizes
pipeline_parallel_sizes = st.integers(min_value=1, max_value=4)

# Strategy for generating dtypes
dtypes = st.sampled_from(["fp16", "fp32", "bf16"])

# Strategy for generating prompts
prompts = st.lists(
    st.text(min_size=1, max_size=100),
    min_size=1,
    max_size=10
)


# Feature: advanced-optimization-monitoring, Property 1: Engine selection respects availability (DeepSpeed)
@given(
    deepspeed_enabled=st.booleans()
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_1_deepspeed_availability_detection(deepspeed_enabled):
    """
    Property 1: Engine selection respects availability (DeepSpeed).
    
    For any DeepSpeed configuration, the engine should correctly detect whether
    DeepSpeed is available based on library installation and configuration.
    
    Validates: Requirements 2.1
    """
    # Setup: Create DeepSpeed config
    config = DeepSpeedConfig(enabled=deepspeed_enabled)
    engine = DeepSpeedEngine(config)
    
    if not deepspeed_enabled:
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


# Feature: advanced-optimization-monitoring, Property 3: Engine configuration is applied correctly (DeepSpeed)
@given(
    tensor_parallel=tensor_parallel_sizes,
    pipeline_parallel=pipeline_parallel_sizes,
    dtype=dtypes,
    replace_with_kernel_inject=st.booleans()
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_3_deepspeed_configuration_applied(
    tensor_parallel, pipeline_parallel, dtype, replace_with_kernel_inject
):
    """
    Property 3: Engine configuration is applied correctly (DeepSpeed).
    
    For any valid DeepSpeed configuration, the engine should be initialized with
    the specified tensor parallelism, pipeline parallelism, dtype, and kernel settings.
    
    Validates: Requirements 2.2, 2.3
    """
    # Setup: Create DeepSpeed config with specific parameters
    config = DeepSpeedConfig(
        enabled=True,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        dtype=dtype,
        replace_with_kernel_inject=replace_with_kernel_inject
    )
    engine = DeepSpeedEngine(config)
    
    # Verify: Configuration is stored correctly
    assert engine.config.tensor_parallel == tensor_parallel, (
        f"Expected tensor_parallel={tensor_parallel}, "
        f"got {engine.config.tensor_parallel}"
    )
    assert engine.config.pipeline_parallel == pipeline_parallel, (
        f"Expected pipeline_parallel={pipeline_parallel}, "
        f"got {engine.config.pipeline_parallel}"
    )
    assert engine.config.dtype == dtype, (
        f"Expected dtype={dtype}, got {engine.config.dtype}"
    )
    assert engine.config.replace_with_kernel_inject == replace_with_kernel_inject, (
        f"Expected replace_with_kernel_inject={replace_with_kernel_inject}, "
        f"got {engine.config.replace_with_kernel_inject}"
    )


# Additional property: Model loading respects configuration
@given(
    model_name=model_names,
    tensor_parallel=tensor_parallel_sizes,
    pipeline_parallel=pipeline_parallel_sizes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_model_loading_respects_configuration(
    model_name, tensor_parallel, pipeline_parallel
):
    """
    Additional property: Model loading uses configured parameters.
    
    For any model and configuration, load_model should attempt to use the
    configured tensor and pipeline parallelism settings.
    """
    config = DeepSpeedConfig(
        enabled=True,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel
    )
    engine = DeepSpeedEngine(config)
    
    # Skip test if DeepSpeed is not installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("DeepSpeed not installed")
    
    # Mock DeepSpeed availability and init_inference
    with patch.object(engine, 'is_available', return_value=True):
        with patch('deepspeed.init_inference') as mock_init_inference:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    mock_ds_model = MagicMock()
                    mock_init_inference.return_value = mock_ds_model
                    
                    # Execute: Load model
                    result = engine.load_model(model_name)
                    
                    # Verify: init_inference was called with correct parameters
                    if result:
                        mock_init_inference.assert_called_once()
                        call_kwargs = mock_init_inference.call_args[1]
                        
                        assert call_kwargs['mp_size'] == tensor_parallel, (
                            f"Expected mp_size={tensor_parallel}, "
                            f"got {call_kwargs['mp_size']}"
                        )
                        assert call_kwargs['replace_with_kernel_inject'] == config.replace_with_kernel_inject


# Additional property: Model loading with overrides
@given(
    model_name=model_names,
    config_tp=tensor_parallel_sizes,
    override_tp=tensor_parallel_sizes,
    config_pp=pipeline_parallel_sizes,
    override_pp=pipeline_parallel_sizes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_model_loading_overrides_work(
    model_name, config_tp, override_tp, config_pp, override_pp
):
    """
    Additional property: Model loading overrides take precedence.
    
    For any model loading with override parameters, the overrides should
    take precedence over configuration defaults.
    """
    # Skip test if DeepSpeed is not installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("DeepSpeed not installed")
    
    config = DeepSpeedConfig(
        enabled=True,
        tensor_parallel=config_tp,
        pipeline_parallel=config_pp
    )
    engine = DeepSpeedEngine(config)
    
    # Mock DeepSpeed availability and init_inference
    with patch.object(engine, 'is_available', return_value=True):
        with patch('deepspeed.init_inference') as mock_init_inference:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    mock_ds_model = MagicMock()
                    mock_init_inference.return_value = mock_ds_model
                    
                    # Execute: Load model with overrides
                    result = engine.load_model(
                        model_name,
                        tensor_parallel=override_tp,
                        pipeline_parallel=override_pp
                    )
                    
                    # Verify: Overrides were used
                    if result:
                        mock_init_inference.assert_called_once()
                        call_kwargs = mock_init_inference.call_args[1]
                        
                        assert call_kwargs['mp_size'] == override_tp, (
                            f"Expected override mp_size={override_tp}, "
                            f"got {call_kwargs['mp_size']}"
                        )


# Additional property: Infer requires loaded model
@given(
    batch_size=st.integers(min_value=1, max_value=10),
    seq_length=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_infer_requires_loaded_model(batch_size, seq_length):
    """
    Additional property: Infer fails gracefully without loaded model.
    
    For any infer call without a loaded model, the engine should raise
    a clear error message.
    """
    # Setup: Create engine without loading model
    config = DeepSpeedConfig(enabled=True)
    engine = DeepSpeedEngine(config)
    
    # Verify: No model is loaded
    assert engine.get_loaded_model() is None
    
    # Create dummy inputs
    inputs = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length))
    }
    
    # Execute and verify: Infer should raise RuntimeError
    with pytest.raises(RuntimeError, match="No model loaded"):
        engine.infer(inputs)


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
    config = DeepSpeedConfig(enabled=True)
    engine = DeepSpeedEngine(config)
    
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
    # Skip test if DeepSpeed is not installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("DeepSpeed not installed")
    
    config = DeepSpeedConfig(enabled=True)
    engine = DeepSpeedEngine(config)
    
    # Mock DeepSpeed availability and init_inference
    with patch.object(engine, 'is_available', return_value=True):
        with patch('deepspeed.init_inference') as mock_init_inference:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    mock_ds_model = MagicMock()
                    mock_init_inference.return_value = mock_ds_model
                    
                    # Execute: Load and unload model
                    engine.load_model(model_name)
                    assert engine.get_loaded_model() == model_name
                    
                    engine.unload_model()
                    
                    # Verify: Model state is cleared
                    assert engine.get_loaded_model() is None, (
                        "get_loaded_model() should return None after unload"
                    )
                    assert engine._model is None, (
                        "_model should be None after unload"
                    )
                    assert engine._tokenizer is None, (
                        "_tokenizer should be None after unload"
                    )


# Additional property: Configuration validation
@given(
    tensor_parallel=st.integers(min_value=-10, max_value=0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_tensor_parallel_rejected(tensor_parallel):
    """
    Additional property: Invalid tensor parallelism is rejected.
    
    For any invalid tensor_parallel (< 1), configuration creation
    should raise a ValueError.
    """
    # Execute and verify: Invalid config should raise ValueError
    with pytest.raises(ValueError, match="tensor_parallel must be >= 1"):
        DeepSpeedConfig(tensor_parallel=tensor_parallel)


@given(
    pipeline_parallel=st.integers(min_value=-10, max_value=0),
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_invalid_pipeline_parallel_rejected(pipeline_parallel):
    """
    Additional property: Invalid pipeline parallelism is rejected.
    
    For any invalid pipeline_parallel (< 1), configuration creation
    should raise a ValueError.
    """
    # Execute and verify: Invalid config should raise ValueError
    with pytest.raises(ValueError, match="pipeline_parallel must be >= 1"):
        DeepSpeedConfig(pipeline_parallel=pipeline_parallel)


@given(
    dtype=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in {"fp16", "fp32", "bf16"}
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
        DeepSpeedConfig(dtype=dtype)


# Additional property: Dtype conversion consistency
@given(
    dtype=dtypes
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_dtype_conversion_consistency(dtype):
    """
    Additional property: Dtype conversion is consistent.
    
    For any valid dtype string, _get_torch_dtype should return the
    corresponding torch dtype.
    """
    config = DeepSpeedConfig(enabled=True, dtype=dtype)
    engine = DeepSpeedEngine(config)
    
    torch_dtype = engine._get_torch_dtype()
    
    # Verify: Correct torch dtype is returned
    expected_dtypes = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    
    assert torch_dtype == expected_dtypes[dtype], (
        f"Expected torch dtype {expected_dtypes[dtype]} for {dtype}, "
        f"got {torch_dtype}"
    )


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
    with consistent keys (text, prompt, tokens_generated).
    """
    # Skip test if DeepSpeed is not installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("DeepSpeed not installed")
    
    config = DeepSpeedConfig(enabled=True)
    engine = DeepSpeedEngine(config)
    
    # Mock DeepSpeed availability and model
    with patch.object(engine, 'is_available', return_value=True):
        with patch('deepspeed.init_inference') as mock_init_inference:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    # Mock tokenizer behavior
                    mock_tokenizer_instance.return_value = {
                        "input_ids": torch.randint(0, 1000, (num_prompts, 10)),
                        "attention_mask": torch.ones(num_prompts, 10)
                    }
                    
                    # Mock model generate behavior
                    generated_ids = torch.randint(0, 1000, (num_prompts, 20))
                    mock_model_instance.generate.return_value = generated_ids
                    
                    # Mock tokenizer decode behavior
                    def mock_decode(ids, skip_special_tokens=False):
                        return f"generated_text_{ids[0].item()}"
                    mock_tokenizer_instance.decode.side_effect = mock_decode
                    
                    # Mock parameters method for device detection
                    mock_param = MagicMock()
                    mock_param.device = torch.device("cpu")
                    mock_model_instance.parameters.return_value = [mock_param]
                    
                    mock_ds_model = mock_model_instance
                    mock_init_inference.return_value = mock_ds_model
                    
                    # Execute: Load model and generate
                    engine.load_model("test-model")
                    prompts_list = [f"prompt_{i}" for i in range(num_prompts)]
                    outputs = engine.generate(prompts_list)
                    
                    # Verify: Output format is consistent
                    assert len(outputs) == num_prompts, (
                        f"Expected {num_prompts} outputs, got {len(outputs)}"
                    )
                    
                    required_keys = {"text", "prompt", "tokens_generated"}
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


# Additional property: Infer output format consistency
@given(
    batch_size=st.integers(min_value=1, max_value=10),
    seq_length=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_infer_output_format_consistency(batch_size, seq_length):
    """
    Additional property: Infer returns consistent output format.
    
    For any input tensors, infer should return a dictionary with
    at least a 'logits' key containing output tensors.
    """
    # Skip test if DeepSpeed is not installed
    try:
        import deepspeed  # noqa: F401
    except ImportError:
        pytest.skip("DeepSpeed not installed")
    
    config = DeepSpeedConfig(enabled=True)
    engine = DeepSpeedEngine(config)
    
    # Mock DeepSpeed availability and model
    with patch.object(engine, 'is_available', return_value=True):
        with patch('deepspeed.init_inference') as mock_init_inference:
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    # Setup mocks
                    mock_model_instance = MagicMock()
                    mock_model.return_value = mock_model_instance
                    
                    mock_tokenizer_instance = MagicMock()
                    mock_tokenizer.return_value = mock_tokenizer_instance
                    
                    # Mock model forward behavior
                    mock_output = MagicMock()
                    mock_output.logits = torch.randn(batch_size, seq_length, 1000)
                    mock_model_instance.return_value = mock_output
                    
                    # Mock parameters method for device detection
                    mock_param = MagicMock()
                    mock_param.device = torch.device("cpu")
                    mock_model_instance.parameters.return_value = [mock_param]
                    
                    mock_ds_model = mock_model_instance
                    mock_init_inference.return_value = mock_ds_model
                    
                    # Execute: Load model and infer
                    engine.load_model("test-model")
                    inputs = {
                        "input_ids": torch.randint(0, 1000, (batch_size, seq_length))
                    }
                    outputs = engine.infer(inputs)
                    
                    # Verify: Output format is consistent
                    assert isinstance(outputs, dict), (
                        "Infer should return a dictionary"
                    )
                    assert "logits" in outputs, (
                        "Output should contain 'logits' key"
                    )
                    assert isinstance(outputs["logits"], torch.Tensor), (
                        "logits should be a torch.Tensor"
                    )
