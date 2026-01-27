"""
Property-based tests for engine selection and fallback logic.

Feature: advanced-optimization-monitoring
Tests Properties 1, 2, and 51 from the design document.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, patch

from mm_orch.optimization import (
    OptimizationManager,
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig,
    EngineType,
    InferenceResult,
)


# Strategy for generating valid model names
model_names = st.text(min_size=1, max_size=50, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    whitelist_characters='-_'
))

# Strategy for generating engine names
engine_names = st.sampled_from([
    EngineType.VLLM.value,
    EngineType.DEEPSPEED.value,
    EngineType.ONNX.value,
    EngineType.PYTORCH.value
])

# Strategy for generating lists of engines to disable
engine_lists = st.lists(
    st.sampled_from([
        EngineType.VLLM.value,
        EngineType.DEEPSPEED.value,
        EngineType.ONNX.value
    ]),
    max_size=3,
    unique=True
)

# Strategy for generating input dictionaries
input_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.booleans()
    ),
    min_size=0,
    max_size=10
)


# Feature: advanced-optimization-monitoring, Property 1: Engine selection respects availability and preference
@given(
    model_name=model_names,
    inputs=input_dicts,
    preferred_engine=engine_names
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_1_engine_selection_respects_availability_and_preference(
    model_name, inputs, preferred_engine
):
    """
    Property 1: Engine selection respects availability and preference.
    
    For any model and inference request, when an engine is available and preferred,
    the system should use that engine for inference.
    
    Validates: Requirements 1.1, 2.1, 3.1
    """
    # Setup: Create config with all engines enabled
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        deepspeed=DeepSpeedConfig(enabled=True),
        onnx=ONNXConfig(enabled=True),
        engine_preference=[
            EngineType.VLLM.value,
            EngineType.DEEPSPEED.value,
            EngineType.ONNX.value,
            EngineType.PYTORCH.value
        ]
    )
    manager = OptimizationManager(config)
    
    # Mock the preferred engine to be available and succeed
    mock_output = {"output": "test_output"}
    
    if preferred_engine == EngineType.VLLM.value:
        manager._engine_registry[EngineType.VLLM.value].available = True
        manager._infer_vllm = Mock(return_value=mock_output)
    elif preferred_engine == EngineType.DEEPSPEED.value:
        manager._engine_registry[EngineType.DEEPSPEED.value].available = True
        manager._infer_deepspeed = Mock(return_value=mock_output)
    elif preferred_engine == EngineType.ONNX.value:
        manager._engine_registry[EngineType.ONNX.value].available = True
        manager._infer_onnx = Mock(return_value=mock_output)
    elif preferred_engine == EngineType.PYTORCH.value:
        manager._infer_pytorch = Mock(return_value=mock_output)
    
    # Execute: Perform inference with engine preference
    result = manager.infer(model_name, inputs, engine_preference=preferred_engine)
    
    # Verify: The preferred engine should be used
    assert result.engine_used == preferred_engine, (
        f"Expected engine {preferred_engine} to be used, "
        f"but {result.engine_used} was used instead"
    )
    assert result.outputs == mock_output
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0


# Feature: advanced-optimization-monitoring, Property 2: Fallback chain is followed on engine failure
@given(
    model_name=model_names,
    inputs=input_dicts,
    failing_engines=engine_lists
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_2_fallback_chain_followed_on_failure(
    model_name, inputs, failing_engines
):
    """
    Property 2: Fallback chain is followed on engine failure.
    
    For any inference request, when the preferred engine fails or is unavailable,
    the system should attempt the next engine in the fallback chain
    (vLLM → DeepSpeed → ONNX → PyTorch).
    
    Validates: Requirements 1.4, 1.6, 2.4, 2.5, 3.4, 3.5, 15.1, 15.2, 15.3
    """
    # Setup: Create config with fallback enabled
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        deepspeed=DeepSpeedConfig(enabled=True),
        onnx=ONNXConfig(enabled=True),
        engine_preference=[
            EngineType.VLLM.value,
            EngineType.DEEPSPEED.value,
            EngineType.ONNX.value,
            EngineType.PYTORCH.value
        ],
        fallback_on_error=True
    )
    manager = OptimizationManager(config)
    
    # Mock engines: failing engines raise errors, others succeed
    mock_output = {"output": "test_output"}
    
    # Mark all engines as available initially
    manager._engine_registry[EngineType.VLLM.value].available = True
    manager._engine_registry[EngineType.DEEPSPEED.value].available = True
    manager._engine_registry[EngineType.ONNX.value].available = True
    
    # Configure failing engines
    if EngineType.VLLM.value in failing_engines:
        manager._infer_vllm = Mock(side_effect=RuntimeError("vLLM failed"))
    else:
        manager._infer_vllm = Mock(return_value=mock_output)
    
    if EngineType.DEEPSPEED.value in failing_engines:
        manager._infer_deepspeed = Mock(side_effect=RuntimeError("DeepSpeed failed"))
    else:
        manager._infer_deepspeed = Mock(return_value=mock_output)
    
    if EngineType.ONNX.value in failing_engines:
        manager._infer_onnx = Mock(side_effect=RuntimeError("ONNX failed"))
    else:
        manager._infer_onnx = Mock(return_value=mock_output)
    
    # PyTorch always succeeds (fallback)
    manager._infer_pytorch = Mock(return_value=mock_output)
    
    # Execute: Perform inference
    result = manager.infer(model_name, inputs)
    
    # Verify: The first available (non-failing) engine in the chain should be used
    expected_engines = [
        EngineType.VLLM.value,
        EngineType.DEEPSPEED.value,
        EngineType.ONNX.value,
        EngineType.PYTORCH.value
    ]
    available_engines = [e for e in expected_engines if e not in failing_engines]
    
    assert result.engine_used == available_engines[0], (
        f"Expected first available engine {available_engines[0]} to be used, "
        f"but {result.engine_used} was used instead. "
        f"Failing engines: {failing_engines}"
    )
    assert result.outputs == mock_output


# Feature: advanced-optimization-monitoring, Property 51: Missing libraries trigger fallback
@given(
    model_name=model_names,
    inputs=input_dicts,
    disabled_engines=engine_lists
)
@settings(max_examples=100, deadline=None)
@pytest.mark.property
def test_property_51_missing_libraries_trigger_fallback(
    model_name, inputs, disabled_engines
):
    """
    Property 51: Missing libraries trigger fallback.
    
    For any optimization library that is not installed, the system should detect
    its absence and fall back to standard inference.
    
    Validates: Requirements 13.2
    """
    # Setup: Create config with some engines disabled (simulating missing libraries)
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=EngineType.VLLM.value not in disabled_engines),
        deepspeed=DeepSpeedConfig(enabled=EngineType.DEEPSPEED.value not in disabled_engines),
        onnx=ONNXConfig(enabled=EngineType.ONNX.value not in disabled_engines),
        engine_preference=[
            EngineType.VLLM.value,
            EngineType.DEEPSPEED.value,
            EngineType.ONNX.value,
            EngineType.PYTORCH.value
        ],
        fallback_on_error=True
    )
    manager = OptimizationManager(config)
    
    # Mock available engines to succeed
    mock_output = {"output": "test_output"}
    
    if EngineType.VLLM.value not in disabled_engines:
        manager._engine_registry[EngineType.VLLM.value].available = True
        manager._infer_vllm = Mock(return_value=mock_output)
    
    if EngineType.DEEPSPEED.value not in disabled_engines:
        manager._engine_registry[EngineType.DEEPSPEED.value].available = True
        manager._infer_deepspeed = Mock(return_value=mock_output)
    
    if EngineType.ONNX.value not in disabled_engines:
        manager._engine_registry[EngineType.ONNX.value].available = True
        manager._infer_onnx = Mock(return_value=mock_output)
    
    # PyTorch always available
    manager._infer_pytorch = Mock(return_value=mock_output)
    
    # Execute: Perform inference
    result = manager.infer(model_name, inputs)
    
    # Verify: Should use first available engine (not disabled)
    expected_engines = [
        EngineType.VLLM.value,
        EngineType.DEEPSPEED.value,
        EngineType.ONNX.value,
        EngineType.PYTORCH.value
    ]
    available_engines = [e for e in expected_engines if e not in disabled_engines]
    
    assert result.engine_used in available_engines, (
        f"Expected one of {available_engines} to be used, "
        f"but {result.engine_used} was used instead"
    )
    
    # Verify: Disabled engines should be marked as unavailable
    for engine in disabled_engines:
        status = manager._engine_registry.get(engine)
        assert status is not None
        assert status.available is False, (
            f"Engine {engine} should be unavailable (disabled), "
            f"but is marked as available"
        )
        assert "disabled in configuration" in status.error_message


# Additional property test: Engine preference order is respected
@given(
    model_name=model_names,
    inputs=input_dicts,
    preference_order=st.permutations([
        EngineType.VLLM.value,
        EngineType.DEEPSPEED.value,
        EngineType.ONNX.value,
        EngineType.PYTORCH.value
    ])
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_engine_preference_order_respected(model_name, inputs, preference_order):
    """
    Additional property: Engine preference order is respected.
    
    For any valid preference order, the system should attempt engines in that order.
    """
    # Setup: Create config with custom preference order
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        deepspeed=DeepSpeedConfig(enabled=True),
        onnx=ONNXConfig(enabled=True),
        engine_preference=list(preference_order),
        fallback_on_error=True
    )
    manager = OptimizationManager(config)
    
    # Mock all engines to succeed
    mock_output = {"output": "test_output"}
    manager._engine_registry[EngineType.VLLM.value].available = True
    manager._engine_registry[EngineType.DEEPSPEED.value].available = True
    manager._engine_registry[EngineType.ONNX.value].available = True
    
    manager._infer_vllm = Mock(return_value=mock_output)
    manager._infer_deepspeed = Mock(return_value=mock_output)
    manager._infer_onnx = Mock(return_value=mock_output)
    manager._infer_pytorch = Mock(return_value=mock_output)
    
    # Execute: Perform inference
    result = manager.infer(model_name, inputs)
    
    # Verify: First engine in preference order should be used
    assert result.engine_used == preference_order[0], (
        f"Expected first engine in preference order {preference_order[0]} to be used, "
        f"but {result.engine_used} was used instead"
    )


# Additional property test: Fallback disabled prevents fallback
@given(
    model_name=model_names,
    inputs=input_dicts,
    failing_engine=st.sampled_from([
        EngineType.VLLM.value,
        EngineType.DEEPSPEED.value,
        EngineType.ONNX.value
    ])
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_fallback_disabled_prevents_fallback(model_name, inputs, failing_engine):
    """
    Additional property: When fallback is disabled, errors are not caught.
    
    For any engine failure when fallback_on_error is False, the error should
    propagate without attempting other engines.
    """
    # Setup: Create config with fallback disabled
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        deepspeed=DeepSpeedConfig(enabled=True),
        onnx=ONNXConfig(enabled=True),
        engine_preference=[failing_engine, EngineType.PYTORCH.value],
        fallback_on_error=False
    )
    manager = OptimizationManager(config)
    
    # Mock the failing engine to raise an error
    error_message = f"{failing_engine} failed"
    
    if failing_engine == EngineType.VLLM.value:
        manager._engine_registry[EngineType.VLLM.value].available = True
        manager._infer_vllm = Mock(side_effect=RuntimeError(error_message))
    elif failing_engine == EngineType.DEEPSPEED.value:
        manager._engine_registry[EngineType.DEEPSPEED.value].available = True
        manager._infer_deepspeed = Mock(side_effect=RuntimeError(error_message))
    elif failing_engine == EngineType.ONNX.value:
        manager._engine_registry[EngineType.ONNX.value].available = True
        manager._infer_onnx = Mock(side_effect=RuntimeError(error_message))
    
    # Mock PyTorch to succeed (should not be called)
    manager._infer_pytorch = Mock(return_value={"output": "test"})
    
    # Execute and verify: Should raise error without fallback
    with pytest.raises(RuntimeError, match=error_message):
        manager.infer(model_name, inputs)
    
    # Verify: PyTorch should not have been called
    manager._infer_pytorch.assert_not_called()


# Additional property test: Critical errors mark engine unavailable
@given(
    model_name=model_names,
    inputs=input_dicts,
    critical_error_type=st.sampled_from([
        ImportError,
        RuntimeError,
        MemoryError
    ])
)
@settings(max_examples=50, deadline=None)
@pytest.mark.property
def test_critical_errors_mark_engine_unavailable(
    model_name, inputs, critical_error_type
):
    """
    Additional property: Critical errors mark engine as unavailable.
    
    For any critical error during inference, the engine should be marked as
    unavailable for future requests.
    """
    # Setup: Create config
    config = OptimizationConfig(
        vllm=VLLMConfig(enabled=True),
        engine_preference=[EngineType.VLLM.value, EngineType.PYTORCH.value],
        fallback_on_error=True
    )
    manager = OptimizationManager(config)
    
    # Mock vLLM to fail with critical error
    manager._engine_registry[EngineType.VLLM.value].available = True
    manager._infer_vllm = Mock(side_effect=critical_error_type("Critical failure"))
    
    # Mock PyTorch to succeed
    manager._infer_pytorch = Mock(return_value={"output": "test"})
    
    # Execute: Perform inference (should fallback to PyTorch)
    result = manager.infer(model_name, inputs)
    
    # Verify: vLLM should be marked unavailable
    vllm_status = manager._engine_registry[EngineType.VLLM.value]
    assert vllm_status.available is False, (
        "vLLM should be marked unavailable after critical error"
    )
    assert "Critical failure" in vllm_status.error_message
    
    # Verify: PyTorch was used as fallback
    assert result.engine_used == EngineType.PYTORCH.value
