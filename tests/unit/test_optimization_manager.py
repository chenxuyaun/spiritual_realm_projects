"""
Unit tests for OptimizationManager.

Tests engine detection, availability checking, fallback chain logic,
and engine status tracking.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization import (
    OptimizationManager,
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig,
    EngineStatus,
    EngineType,
    InferenceResult,
)


class TestOptimizationManagerInitialization:
    """Test OptimizationManager initialization and engine detection."""
    
    def test_initialization_with_default_config(self):
        """Test manager initializes with default configuration."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        assert manager.config == config
        assert len(manager._engine_registry) == 4  # vllm, deepspeed, onnx, pytorch
        assert EngineType.PYTORCH.value in manager._engine_registry
    
    def test_pytorch_always_available(self):
        """Test PyTorch engine is always marked as available."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        pytorch_status = manager._engine_registry[EngineType.PYTORCH.value]
        assert pytorch_status.available is True
        assert pytorch_status.error_message is None
    
    @patch('mm_orch.optimization.manager.logger')
    def test_initialization_logs_available_engines(self, mock_logger):
        """Test initialization logs available engines."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Should log initialization and available engines
        assert mock_logger.info.call_count >= 2
    
    def test_engine_preference_copied_from_config(self):
        """Test engine preference is copied from configuration."""
        config = OptimizationConfig(
            engine_preference=["onnx", "pytorch"]
        )
        manager = OptimizationManager(config)
        
        assert manager._engine_preference == ["onnx", "pytorch"]


class TestEngineAvailabilityChecking:
    """Test engine availability detection."""
    
    def test_vllm_disabled_in_config(self):
        """Test vLLM marked unavailable when disabled in config."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False)
        )
        manager = OptimizationManager(config)
        
        vllm_status = manager._engine_registry[EngineType.VLLM.value]
        assert vllm_status.available is False
        assert "disabled in configuration" in vllm_status.error_message
    
    def test_deepspeed_disabled_in_config(self):
        """Test DeepSpeed marked unavailable when disabled in config."""
        config = OptimizationConfig(
            deepspeed=DeepSpeedConfig(enabled=False)
        )
        manager = OptimizationManager(config)
        
        deepspeed_status = manager._engine_registry[EngineType.DEEPSPEED.value]
        assert deepspeed_status.available is False
        assert "disabled in configuration" in deepspeed_status.error_message
    
    def test_onnx_disabled_in_config(self):
        """Test ONNX marked unavailable when disabled in config."""
        config = OptimizationConfig(
            onnx=ONNXConfig(enabled=False)
        )
        manager = OptimizationManager(config)
        
        onnx_status = manager._engine_registry[EngineType.ONNX.value]
        assert onnx_status.available is False
        assert "disabled in configuration" in onnx_status.error_message
    
    @patch('mm_orch.optimization.manager.logger')
    def test_missing_library_logs_warning(self, mock_logger):
        """Test missing libraries are logged as warnings."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # At least one engine should be unavailable (vllm, deepspeed, or onnx)
        # and should log a warning
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert len(warning_calls) > 0
    
    def test_engine_status_has_timestamp(self):
        """Test engine status includes last_check timestamp."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        for status in manager._engine_registry.values():
            assert status.last_check is not None
            assert isinstance(status.last_check, datetime)


class TestGetAvailableEngines:
    """Test get_available_engines method."""
    
    def test_returns_available_engines_in_preference_order(self):
        """Test returns only available engines in preference order."""
        config = OptimizationConfig(
            engine_preference=["vllm", "deepspeed", "onnx", "pytorch"]
        )
        manager = OptimizationManager(config)
        
        available = manager.get_available_engines()
        
        # PyTorch should always be available
        assert EngineType.PYTORCH.value in available
        
        # Should be in preference order
        for i in range(len(available) - 1):
            pref_idx_i = config.engine_preference.index(available[i])
            pref_idx_next = config.engine_preference.index(available[i + 1])
            assert pref_idx_i < pref_idx_next
    
    def test_returns_empty_list_when_all_disabled(self):
        """Test returns only pytorch when all other engines disabled."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False),
            deepspeed=DeepSpeedConfig(enabled=False),
            onnx=ONNXConfig(enabled=False),
            engine_preference=["vllm", "deepspeed", "onnx"]  # No pytorch initially
        )
        manager = OptimizationManager(config)
        
        available = manager.get_available_engines()
        
        # Should only have pytorch (added as fallback by config)
        assert available == ["pytorch"]
    
    def test_model_name_parameter_accepted(self):
        """Test model_name parameter is accepted (for future use)."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Should not raise error
        available = manager.get_available_engines(model_name="test-model")
        assert isinstance(available, list)


class TestGetEngineStatus:
    """Test get_engine_status method."""
    
    def test_returns_all_statuses_when_no_name_specified(self):
        """Test returns all engine statuses when no name specified."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        statuses = manager.get_engine_status()
        
        assert len(statuses) == 4  # vllm, deepspeed, onnx, pytorch
        assert all(isinstance(s, EngineStatus) for s in statuses.values())
    
    def test_returns_specific_status_when_name_specified(self):
        """Test returns specific engine status when name specified."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        status = manager.get_engine_status(EngineType.PYTORCH.value)
        
        assert len(status) == 1
        assert EngineType.PYTORCH.value in status
        assert status[EngineType.PYTORCH.value].available is True
    
    def test_returns_empty_dict_for_unknown_engine(self):
        """Test returns empty dict for unknown engine name."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        status = manager.get_engine_status("unknown-engine")
        
        assert status == {}
    
    def test_returned_status_is_copy(self):
        """Test returned status is a copy, not reference."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        statuses1 = manager.get_engine_status()
        statuses2 = manager.get_engine_status()
        
        # Should be equal but not the same object
        assert statuses1 == statuses2
        assert statuses1 is not statuses2


class TestInferenceExecution:
    """Test inference execution and fallback logic."""
    
    def test_infer_raises_when_no_engines_available(self):
        """Test infer raises RuntimeError when no engines available."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False),
            deepspeed=DeepSpeedConfig(enabled=False),
            onnx=ONNXConfig(enabled=False),
            engine_preference=["vllm", "deepspeed", "onnx"]  # No pytorch initially
        )
        manager = OptimizationManager(config)
        
        # PyTorch is added as fallback but not implemented yet
        # So it will fail with "All engines failed" message
        with pytest.raises(RuntimeError, match="All engines failed"):
            manager.infer("test-model", {"input": "test"})
    
    def test_infer_uses_preferred_engine_when_specified(self):
        """Test infer attempts preferred engine first."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock PyTorch inference to succeed
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        result = manager.infer(
            "test-model",
            {"input": "test"},
            engine_preference="pytorch"
        )
        
        assert result.engine_used == EngineType.PYTORCH.value
        manager._infer_pytorch.assert_called_once()
    
    def test_infer_falls_back_on_engine_failure(self):
        """Test infer falls back to next engine on failure."""
        config = OptimizationConfig(
            engine_preference=["vllm", "pytorch"]
        )
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail, PyTorch to succeed
        manager._infer_vllm = Mock(side_effect=RuntimeError("vLLM failed"))
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        # Mark vLLM as available for testing
        manager._engine_registry[EngineType.VLLM.value].available = True
        
        result = manager.infer("test-model", {"input": "test"})
        
        assert result.engine_used == EngineType.PYTORCH.value
        manager._infer_vllm.assert_called_once()
        manager._infer_pytorch.assert_called_once()
    
    def test_infer_raises_when_all_engines_fail(self):
        """Test infer raises RuntimeError when all engines fail."""
        config = OptimizationConfig(
            engine_preference=["pytorch"]
        )
        manager = OptimizationManager(config)
        
        # Mock PyTorch to fail
        manager._infer_pytorch = Mock(side_effect=RuntimeError("PyTorch failed"))
        
        with pytest.raises(RuntimeError, match="All engines failed"):
            manager.infer("test-model", {"input": "test"})
    
    def test_infer_respects_fallback_disabled(self):
        """Test infer doesn't fallback when fallback_on_error is False."""
        config = OptimizationConfig(
            engine_preference=["vllm", "pytorch"],
            fallback_on_error=False
        )
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail
        manager._infer_vllm = Mock(side_effect=RuntimeError("vLLM failed"))
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        # Mark vLLM as available
        manager._engine_registry[EngineType.VLLM.value].available = True
        
        with pytest.raises(RuntimeError, match="vLLM failed"):
            manager.infer("test-model", {"input": "test"})
        
        # PyTorch should not be called
        manager._infer_pytorch.assert_not_called()
    
    @patch('mm_orch.optimization.manager.logger')
    def test_infer_logs_engine_attempts(self, mock_logger):
        """Test infer logs engine attempts and results."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock PyTorch to succeed
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        manager.infer("test-model", {"input": "test"})
        
        # Should log debug and info messages
        assert mock_logger.debug.called
        assert mock_logger.info.called


class TestInferenceResult:
    """Test InferenceResult creation and properties."""
    
    def test_inference_result_creation(self):
        """Test InferenceResult can be created with required fields."""
        result = InferenceResult(
            outputs={"output": "test"},
            engine_used="pytorch",
            latency_ms=10.5,
            batch_size=1,
            cache_hit=False
        )
        
        assert result.outputs == {"output": "test"}
        assert result.engine_used == "pytorch"
        assert result.latency_ms == 10.5
        assert result.batch_size == 1
        assert result.cache_hit is False
    
    def test_inference_result_default_metadata(self):
        """Test InferenceResult initializes metadata if not provided."""
        result = InferenceResult(
            outputs={"output": "test"},
            engine_used="pytorch",
            latency_ms=10.5
        )
        
        assert result.metadata is not None
        assert isinstance(result.metadata, dict)
    
    def test_inference_result_with_metadata(self):
        """Test InferenceResult accepts custom metadata."""
        metadata = {"model_name": "test-model", "custom": "value"}
        result = InferenceResult(
            outputs={"output": "test"},
            engine_used="pytorch",
            latency_ms=10.5,
            metadata=metadata
        )
        
        assert result.metadata == metadata
    
    def test_inference_result_to_dict(self):
        """Test InferenceResult serialization to dictionary."""
        result = InferenceResult(
            outputs={"output": "test"},
            engine_used="pytorch",
            latency_ms=10.5,
            batch_size=2,
            cache_hit=True,
            metadata={"model": "test-model"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["outputs"] == {"output": "test"}
        assert result_dict["engine_used"] == "pytorch"
        assert result_dict["latency_ms"] == 10.5
        assert result_dict["batch_size"] == 2
        assert result_dict["cache_hit"] is True
        assert result_dict["metadata"] == {"model": "test-model"}
    
    def test_inference_result_from_dict(self):
        """Test InferenceResult deserialization from dictionary."""
        data = {
            "outputs": {"output": "test"},
            "engine_used": "pytorch",
            "latency_ms": 10.5,
            "batch_size": 2,
            "cache_hit": True,
            "metadata": {"model": "test-model"}
        }
        
        result = InferenceResult.from_dict(data)
        
        assert result.outputs == {"output": "test"}
        assert result.engine_used == "pytorch"
        assert result.latency_ms == 10.5
        assert result.batch_size == 2
        assert result.cache_hit is True
        assert result.metadata == {"model": "test-model"}
    
    def test_inference_result_from_dict_with_defaults(self):
        """Test InferenceResult deserialization uses defaults for optional fields."""
        data = {
            "outputs": {"output": "test"},
            "engine_used": "pytorch",
            "latency_ms": 10.5
        }
        
        result = InferenceResult.from_dict(data)
        
        assert result.outputs == {"output": "test"}
        assert result.engine_used == "pytorch"
        assert result.latency_ms == 10.5
        assert result.batch_size == 1  # Default
        assert result.cache_hit is False  # Default
        assert result.metadata == {}  # Default
    
    def test_inference_result_from_dict_missing_required_field(self):
        """Test InferenceResult deserialization raises error for missing required fields."""
        data = {
            "outputs": {"output": "test"},
            "engine_used": "pytorch"
            # Missing latency_ms
        }
        
        with pytest.raises(KeyError, match="Missing required field: latency_ms"):
            InferenceResult.from_dict(data)
    
    def test_inference_result_roundtrip_serialization(self):
        """Test InferenceResult can be serialized and deserialized without loss."""
        original = InferenceResult(
            outputs={"output": "test", "score": 0.95},
            engine_used="vllm",
            latency_ms=15.3,
            batch_size=4,
            cache_hit=True,
            metadata={"model": "qwen-chat", "timestamp": "2024-01-01T00:00:00"}
        )
        
        # Serialize and deserialize
        data = original.to_dict()
        restored = InferenceResult.from_dict(data)
        
        # Verify all fields match
        assert restored.outputs == original.outputs
        assert restored.engine_used == original.engine_used
        assert restored.latency_ms == original.latency_ms
        assert restored.batch_size == original.batch_size
        assert restored.cache_hit == original.cache_hit
        assert restored.metadata == original.metadata


class TestEngineStatusTracking:
    """Test engine status tracking and updates."""
    
    def test_critical_error_marks_engine_unavailable(self):
        """Test critical errors mark engine as unavailable."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail with critical error
        manager._infer_vllm = Mock(side_effect=MemoryError("Out of memory"))
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        # Mark vLLM as available initially
        manager._engine_registry[EngineType.VLLM.value].available = True
        
        # Execute inference (should fallback to pytorch)
        result = manager.infer("test-model", {"input": "test"})
        
        # vLLM should now be marked unavailable
        vllm_status = manager._engine_registry[EngineType.VLLM.value]
        assert vllm_status.available is False
        assert "Out of memory" in vllm_status.error_message
    
    def test_non_critical_error_keeps_engine_available(self):
        """Test non-critical errors don't mark engine as unavailable."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail with non-critical error
        manager._infer_vllm = Mock(side_effect=ValueError("Invalid input"))
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        # Mark vLLM as available initially
        manager._engine_registry[EngineType.VLLM.value].available = True
        
        # Execute inference (should fallback to pytorch)
        result = manager.infer("test-model", {"input": "test"})
        
        # vLLM should still be marked available
        vllm_status = manager._engine_registry[EngineType.VLLM.value]
        assert vllm_status.available is True
    
    def test_engine_status_timestamp_updated_on_failure(self):
        """Test engine status timestamp is updated when marked unavailable."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Get initial timestamp
        initial_time = manager._engine_registry[EngineType.VLLM.value].last_check
        
        # Mark engine unavailable
        manager._mark_engine_unavailable(EngineType.VLLM.value, "Test error")
        
        # Timestamp should be updated
        updated_time = manager._engine_registry[EngineType.VLLM.value].last_check
        assert updated_time >= initial_time


class TestEngineSpecificInference:
    """Test engine-specific inference methods."""
    
    def test_vllm_inference_requires_prompts(self):
        """Test vLLM inference requires 'prompts' key in inputs."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Should raise ValueError when 'prompts' key is missing
        with pytest.raises(ValueError, match="vLLM inference requires 'prompts' key in inputs"):
            manager._infer_vllm("test-model", {"input": "test"})
    
    def test_deepspeed_inference_not_implemented(self):
        """Test DeepSpeed inference raises NotImplementedError."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        with pytest.raises(NotImplementedError, match="DeepSpeed inference not yet implemented"):
            manager._infer_deepspeed("test-model", {"input": "test"})
    
    def test_onnx_inference_not_implemented(self):
        """Test ONNX inference raises NotImplementedError."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        with pytest.raises(NotImplementedError, match="ONNX inference not yet implemented"):
            manager._infer_onnx("test-model", {"input": "test"})
    
    def test_pytorch_inference_not_implemented(self):
        """Test PyTorch inference raises NotImplementedError."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        with pytest.raises(NotImplementedError, match="PyTorch inference integration not yet implemented"):
            manager._infer_pytorch("test-model", {"input": "test"})


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unknown_engine_in_preference_raises_error(self):
        """Test unknown engine in infer raises ValueError during execution."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Manually add unknown engine to preference
        manager._engine_preference = ["unknown-engine"]
        manager._engine_registry["unknown-engine"] = EngineStatus(
            name="unknown-engine",
            available=True,
            models_loaded=[]
        )
        
        # Should raise RuntimeError with "All engines failed" since unknown engine fails
        with pytest.raises(RuntimeError, match="All engines failed"):
            manager.infer("test-model", {"input": "test"})
    
    def test_empty_inputs_accepted(self):
        """Test empty inputs dictionary is accepted."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock PyTorch to succeed
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        # Should not raise error
        result = manager.infer("test-model", {})
        assert result.engine_used == EngineType.PYTORCH.value
    
    def test_inference_result_includes_timestamp(self):
        """Test inference result includes timestamp in metadata."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Mock PyTorch to succeed
        manager._infer_pytorch = Mock(return_value={"output": "test"})
        
        result = manager.infer("test-model", {"input": "test"})
        
        assert "timestamp" in result.metadata
        assert "model_name" in result.metadata
        assert result.metadata["model_name"] == "test-model"
