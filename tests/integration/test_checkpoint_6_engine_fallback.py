"""
Integration tests for Checkpoint 6: Optimization engines with fallback.

This test suite verifies that:
1. All engines can be initialized (or gracefully fail)
2. Fallback chain works with simulated failures
3. Existing ModelManager integration works
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization.config import (
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig
)
from mm_orch.optimization.manager import OptimizationManager, EngineType
from mm_orch.optimization.vllm_engine import VLLMEngine
from mm_orch.optimization.deepspeed_engine import DeepSpeedEngine
from mm_orch.optimization.onnx_engine import ONNXEngine


class TestEngineInitialization:
    """Test that all engines can be initialized or gracefully fail."""
    
    def test_vllm_engine_initialization(self):
        """Test vLLM engine initialization."""
        config = VLLMConfig(enabled=True)
        engine = VLLMEngine(config)
        
        # Should not raise exception
        assert engine is not None
        assert engine.config == config
        
        # Check availability (may be False if vLLM not installed)
        is_available = engine.is_available()
        assert isinstance(is_available, bool)
        
        if not is_available:
            print("vLLM not available - this is expected if not installed")
    
    def test_deepspeed_engine_initialization(self):
        """Test DeepSpeed engine initialization."""
        config = DeepSpeedConfig(enabled=True)
        engine = DeepSpeedEngine(config)
        
        # Should not raise exception
        assert engine is not None
        assert engine.config == config
        
        # Check availability (may be False if DeepSpeed not installed)
        is_available = engine.is_available()
        assert isinstance(is_available, bool)
        
        if not is_available:
            print("DeepSpeed not available - this is expected if not installed")
    
    def test_onnx_engine_initialization(self):
        """Test ONNX Runtime engine initialization."""
        config = ONNXConfig(enabled=True)
        engine = ONNXEngine(config)
        
        # Should not raise exception
        assert engine is not None
        assert engine.config == config
        
        # Check availability (may be False if ONNX Runtime not installed)
        is_available = engine.is_available()
        assert isinstance(is_available, bool)
        
        if not is_available:
            print("ONNX Runtime not available - this is expected if not installed")
    
    def test_optimization_manager_initialization(self):
        """Test OptimizationManager initialization with all engines."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Should not raise exception
        assert manager is not None
        assert manager.config == config
        
        # Check engine registry is populated
        assert len(manager._engine_registry) > 0
        
        # PyTorch should always be available
        pytorch_status = manager.get_engine_status("pytorch")
        assert "pytorch" in pytorch_status
        assert pytorch_status["pytorch"].available is True
    
    def test_disabled_engines_are_unavailable(self):
        """Test that disabled engines are marked as unavailable."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False),
            deepspeed=DeepSpeedConfig(enabled=False),
            onnx=ONNXConfig(enabled=False)
        )
        manager = OptimizationManager(config)
        
        # All optimization engines should be unavailable
        vllm_status = manager.get_engine_status("vllm")
        if "vllm" in vllm_status:
            assert vllm_status["vllm"].available is False
            assert "disabled" in vllm_status["vllm"].error_message.lower()
        
        deepspeed_status = manager.get_engine_status("deepspeed")
        if "deepspeed" in deepspeed_status:
            assert deepspeed_status["deepspeed"].available is False
            assert "disabled" in deepspeed_status["deepspeed"].error_message.lower()
        
        onnx_status = manager.get_engine_status("onnx")
        if "onnx" in onnx_status:
            assert onnx_status["onnx"].available is False
            assert "disabled" in onnx_status["onnx"].error_message.lower()
        
        # PyTorch should still be available
        pytorch_status = manager.get_engine_status("pytorch")
        assert pytorch_status["pytorch"].available is True


class TestFallbackChain:
    """Test fallback chain with simulated failures."""
    
    def test_fallback_to_pytorch_when_all_engines_unavailable(self):
        """Test that system falls back to PyTorch when all engines fail."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False),
            deepspeed=DeepSpeedConfig(enabled=False),
            onnx=ONNXConfig(enabled=False),
            fallback_on_error=True
        )
        manager = OptimizationManager(config)
        
        # Only PyTorch should be available
        available_engines = manager.get_available_engines()
        assert "pytorch" in available_engines
        assert "vllm" not in available_engines
        assert "deepspeed" not in available_engines
        assert "onnx" not in available_engines
    
    def test_fallback_on_engine_failure(self):
        """Test fallback when preferred engine fails."""
        config = OptimizationConfig(fallback_on_error=True)
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("vLLM failed")):
            # Mock DeepSpeed to fail
            with patch.object(manager, '_infer_deepspeed', side_effect=RuntimeError("DeepSpeed failed")):
                # Mock ONNX to fail
                with patch.object(manager, '_infer_onnx', side_effect=RuntimeError("ONNX failed")):
                    # Mock PyTorch to succeed
                    with patch.object(manager, '_infer_pytorch', return_value={"output": "success"}):
                        # Should fall back to PyTorch
                        result = manager.infer(
                            "test-model",
                            {"input": "test"},
                            engine_preference="vllm"
                        )
                        
                        assert result.engine_used == "pytorch"
                        assert result.outputs == {"output": "success"}
    
    def test_no_fallback_when_disabled(self):
        """Test that fallback is disabled when configured."""
        config = OptimizationConfig(fallback_on_error=False)
        manager = OptimizationManager(config)
        
        # Mock all engines to fail (including PyTorch since it's not implemented yet)
        with patch.object(manager, '_infer_pytorch', side_effect=RuntimeError("PyTorch failed")):
            # Should raise exception instead of falling back
            with pytest.raises(RuntimeError, match="PyTorch failed"):
                manager.infer(
                    "test-model",
                    {"input": "test"},
                    engine_preference="pytorch"
                )
    
    def test_engine_preference_order(self):
        """Test that engine preference order is respected."""
        config = OptimizationConfig(
            engine_preference=["onnx", "deepspeed", "vllm", "pytorch"]
        )
        manager = OptimizationManager(config)
        
        # Mock all engines to succeed
        with patch.object(manager, '_infer_vllm', return_value={"output": "vllm"}):
            with patch.object(manager, '_infer_deepspeed', return_value={"output": "deepspeed"}):
                with patch.object(manager, '_infer_onnx', return_value={"output": "onnx"}):
                    with patch.object(manager, '_infer_pytorch', return_value={"output": "pytorch"}):
                        # Should try ONNX first (based on preference order)
                        # But only if ONNX is available
                        available = manager.get_available_engines()
                        if "onnx" in available:
                            result = manager.infer("test-model", {"input": "test"})
                            assert result.engine_used == "onnx"
    
    def test_critical_error_marks_engine_unavailable(self):
        """Test that critical errors mark engine as unavailable."""
        config = OptimizationConfig(fallback_on_error=True)
        manager = OptimizationManager(config)
        
        # Initially, check if vLLM is available
        initial_status = manager.get_engine_status("vllm")
        if "vllm" not in initial_status or not initial_status["vllm"].available:
            pytest.skip("vLLM not available for this test")
        
        # Mock vLLM to fail with critical error
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("Critical failure")):
            with patch.object(manager, '_infer_pytorch', return_value={"output": "success"}):
                # Should fall back to PyTorch
                result = manager.infer("test-model", {"input": "test"}, engine_preference="vllm")
                assert result.engine_used == "pytorch"
                
                # vLLM should now be marked as unavailable
                updated_status = manager.get_engine_status("vllm")
                assert updated_status["vllm"].available is False
                assert "Critical failure" in updated_status["vllm"].error_message


class TestEngineStatusReporting:
    """Test engine status reporting functionality."""
    
    def test_get_all_engine_status(self):
        """Test getting status for all engines."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        all_status = manager.get_engine_status()
        
        # Should have status for all engines
        assert "pytorch" in all_status
        assert all_status["pytorch"].available is True
        
        # Other engines may or may not be available
        for engine_name in ["vllm", "deepspeed", "onnx"]:
            if engine_name in all_status:
                status = all_status[engine_name]
                assert isinstance(status.available, bool)
                assert status.last_check is not None
    
    def test_get_specific_engine_status(self):
        """Test getting status for a specific engine."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        pytorch_status = manager.get_engine_status("pytorch")
        assert "pytorch" in pytorch_status
        assert pytorch_status["pytorch"].available is True
        assert pytorch_status["pytorch"].name == "pytorch"
    
    def test_get_available_engines_list(self):
        """Test getting list of available engines."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        available = manager.get_available_engines()
        
        # Should be a list
        assert isinstance(available, list)
        
        # PyTorch should always be available
        assert "pytorch" in available
        
        # All returned engines should be in preference order
        for i in range(len(available) - 1):
            curr_idx = config.engine_preference.index(available[i])
            next_idx = config.engine_preference.index(available[i + 1])
            assert curr_idx < next_idx


class TestModelManagerIntegration:
    """Test integration with existing ModelManager."""
    
    def test_optimization_manager_can_coexist_with_model_manager(self):
        """Test that OptimizationManager can coexist with ModelManager."""
        # This is a placeholder test - actual integration will be done in task 19.1
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Should be able to create both without conflicts
        assert manager is not None
        
        # TODO: Add actual ModelManager integration tests in task 19.1
        print("ModelManager integration will be tested in task 19.1")
    
    def test_inference_result_serialization(self):
        """Test that InferenceResult can be serialized/deserialized."""
        from mm_orch.optimization.manager import InferenceResult
        
        # Create a result
        result = InferenceResult(
            outputs={"text": "Hello, world!"},
            engine_used="pytorch",
            latency_ms=10.5,
            batch_size=1,
            cache_hit=False,
            metadata={"model": "test"}
        )
        
        # Serialize
        result_dict = result.to_dict()
        assert result_dict["engine_used"] == "pytorch"
        assert result_dict["latency_ms"] == 10.5
        
        # Deserialize
        restored = InferenceResult.from_dict(result_dict)
        assert restored.engine_used == result.engine_used
        assert restored.latency_ms == result.latency_ms
        assert restored.outputs == result.outputs


class TestGracefulDegradation:
    """Test graceful degradation scenarios."""
    
    def test_missing_library_graceful_failure(self):
        """Test graceful failure when optimization library is missing."""
        config = OptimizationConfig()
        manager = OptimizationManager(config)
        
        # Check each engine's status
        all_status = manager.get_engine_status()
        
        for engine_name, status in all_status.items():
            if not status.available and engine_name != "pytorch":
                # Should have a clear error message
                assert status.error_message is not None
                assert len(status.error_message) > 0
                print(f"{engine_name}: {status.error_message}")
    
    def test_all_engines_fail_raises_error(self):
        """Test that error is raised when all engines fail."""
        config = OptimizationConfig(fallback_on_error=True)
        manager = OptimizationManager(config)
        
        # Mock all engines to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("vLLM failed")):
            with patch.object(manager, '_infer_deepspeed', side_effect=RuntimeError("DeepSpeed failed")):
                with patch.object(manager, '_infer_onnx', side_effect=RuntimeError("ONNX failed")):
                    with patch.object(manager, '_infer_pytorch', side_effect=RuntimeError("PyTorch failed")):
                        # Should raise error with clear message
                        with pytest.raises(RuntimeError, match="All engines failed"):
                            manager.infer("test-model", {"input": "test"})
    
    def test_no_available_engines_raises_error(self):
        """Test error when no engines are available."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=False),
            deepspeed=DeepSpeedConfig(enabled=False),
            onnx=ONNXConfig(enabled=False)
        )
        manager = OptimizationManager(config)
        
        # Mark PyTorch as unavailable (simulate catastrophic failure)
        manager._engine_registry["pytorch"].available = False
        
        # Should raise error
        with pytest.raises(RuntimeError, match="No available engines"):
            manager.infer("test-model", {"input": "test"})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
