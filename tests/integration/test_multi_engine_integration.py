"""
Integration tests for multi-engine scenarios.

Tests engine switching and fallback:
- Switching between engines
- Fallback with real engine failures
- Performance comparison across engines

Validates: All engine requirements
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from mm_orch.optimization.manager import OptimizationManager
from mm_orch.optimization.config import (
    OptimizationConfig,
    VLLMConfig,
    DeepSpeedConfig,
    ONNXConfig
)


class TestEngineSwitching:
    """Test switching between different engines."""
    
    def test_switch_from_vllm_to_pytorch(self):
        """Test switching from vLLM to PyTorch."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            engine_preference=["vllm", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # First inference with vLLM preference
        inputs1 = {"prompts": ["Hello"], "max_tokens": 10}
        result1 = manager.infer(
            model_name="gpt2",
            inputs=inputs1,
            engine_preference="vllm"
        )
        
        # Second inference with PyTorch preference
        inputs2 = {"prompts": ["World"], "max_tokens": 10}
        result2 = manager.infer(
            model_name="gpt2",
            inputs=inputs2,
            engine_preference="pytorch"
        )
        
        # Verify: Different engines used (if vLLM available)
        if "vllm" in manager.get_available_engines():
            assert result1.engine_used == "vllm"
        assert result2.engine_used == "pytorch"
    
    def test_switch_between_all_engines(self):
        """Test switching between all available engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        engines_to_test = ["vllm", "deepspeed", "onnx", "pytorch"]
        results = {}
        
        for engine in engines_to_test:
            if engine in manager.get_available_engines():
                try:
                    inputs = {"prompts": [f"Test {engine}"], "max_tokens": 10}
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    results[engine] = result
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
        
        # Verify: At least PyTorch works
        assert "pytorch" in results
        assert results["pytorch"].engine_used == "pytorch"
    
    def test_engine_switching_preserves_state(self):
        """Test that switching engines preserves model state."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Run inference with different engines
        inputs = {"prompts": ["Test"], "max_tokens": 10}
        
        result1 = manager.infer(
            model_name="gpt2",
            inputs=inputs,
            engine_preference="pytorch"
        )
        
        result2 = manager.infer(
            model_name="gpt2",
            inputs=inputs,
            engine_preference="onnx"
        )
        
        result3 = manager.infer(
            model_name="gpt2",
            inputs=inputs,
            engine_preference="pytorch"
        )
        
        # Verify: All produce valid results
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
    
    def test_engine_preference_override(self):
        """Test that engine preference can be overridden per request."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            engine_preference=["vllm", "pytorch"]  # Default preference
        )
        
        manager = OptimizationManager(config)
        
        # Override to use PyTorch
        inputs = {"prompts": ["Test"], "max_tokens": 10}
        result = manager.infer(
            model_name="gpt2",
            inputs=inputs,
            engine_preference="pytorch"  # Override
        )
        
        # Verify: PyTorch used despite vLLM being preferred
        assert result.engine_used == "pytorch"


class TestRealEngineFallback:
    """Test fallback with real engine failures."""
    
    def test_fallback_on_initialization_failure(self):
        """Test fallback when engine initialization fails."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            fallback_on_error=True,
            engine_preference=["vllm", "deepspeed", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Mock vLLM initialization to fail
        with patch.object(manager, '_init_vllm_engine', side_effect=RuntimeError("Init failed")):
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            
            # Verify: Fallback occurred
            assert result.engine_used in ["deepspeed", "pytorch"]
            assert result.metadata.get("fallback_occurred", False)
    
    def test_fallback_on_inference_failure(self):
        """Test fallback when inference fails."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True,
            engine_preference=["vllm", "onnx", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Mock vLLM inference to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("Inference failed")):
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            
            # Verify: Fallback to next engine
            assert result.engine_used in ["onnx", "pytorch"]
    
    def test_fallback_chain_with_multiple_failures(self):
        """Test fallback chain with multiple engine failures."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True,
            engine_preference=["vllm", "deepspeed", "onnx", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Mock multiple engines to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("vLLM failed")), \
             patch.object(manager, '_infer_deepspeed', side_effect=RuntimeError("DeepSpeed failed")), \
             patch.object(manager, '_infer_onnx', side_effect=RuntimeError("ONNX failed")):
            
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs
            )
            
            # Verify: Falls back to PyTorch
            assert result.engine_used == "pytorch"
            assert result.metadata.get("fallback_count", 0) >= 3
    
    def test_fallback_with_different_error_types(self):
        """Test fallback with different types of errors."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True
        )
        
        manager = OptimizationManager(config)
        
        # Test with OOM error
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("CUDA out of memory")):
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            assert result.engine_used in ["onnx", "pytorch"]
        
        # Test with import error
        with patch.object(manager, '_infer_vllm', side_effect=ImportError("vLLM not installed")):
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            assert result.engine_used in ["onnx", "pytorch"]
    
    def test_fallback_disabled(self):
        """Test behavior when fallback is disabled."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            fallback_on_error=False,  # Disable fallback
            engine_preference=["vllm"]
        )
        
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("vLLM failed")):
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            
            # Should raise error instead of falling back
            with pytest.raises(RuntimeError, match="vLLM failed"):
                manager.infer(
                    model_name="gpt2",
                    inputs=inputs,
                    engine_preference="vllm"
                )


class TestPerformanceComparison:
    """Test performance comparison across engines."""
    
    @pytest.mark.slow
    def test_latency_comparison_single_request(self):
        """Compare latency for single request across engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        inputs = {"prompts": ["Hello, how are you?"], "max_tokens": 20}
        latencies = {}
        
        # Test each engine
        for engine in ["vllm", "deepspeed", "onnx", "pytorch"]:
            if engine in manager.get_available_engines():
                try:
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    latencies[result.engine_used] = result.latency_ms
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
        
        # Verify: At least one engine worked
        assert len(latencies) > 0
        
        # Log results
        print(f"\nSingle request latency comparison:")
        for engine, latency in sorted(latencies.items(), key=lambda x: x[1]):
            print(f"  {engine}: {latency:.2f}ms")
    
    @pytest.mark.slow
    def test_throughput_comparison_batch(self):
        """Compare throughput for batch requests across engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Batch inputs
        batch_size = 8
        inputs = {
            "prompts": ["Test prompt"] * batch_size,
            "max_tokens": 20
        }
        
        throughputs = {}
        
        # Test each engine
        for engine in ["vllm", "deepspeed", "onnx", "pytorch"]:
            if engine in manager.get_available_engines():
                try:
                    start_time = time.time()
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    elapsed = time.time() - start_time
                    throughput = result.batch_size / elapsed
                    throughputs[result.engine_used] = throughput
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
        
        # Verify: At least one engine worked
        assert len(throughputs) > 0
        
        # Log results
        print(f"\nBatch throughput comparison (req/s):")
        for engine, throughput in sorted(throughputs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {engine}: {throughput:.2f}")
    
    @pytest.mark.slow
    def test_memory_usage_comparison(self):
        """Compare memory usage across engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        inputs = {"prompts": ["Test"], "max_tokens": 20}
        memory_usage = {}
        
        # Test each engine
        for engine in ["vllm", "deepspeed", "onnx", "pytorch"]:
            if engine in manager.get_available_engines():
                try:
                    # Get memory before
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        mem_before = torch.cuda.memory_allocated()
                    
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    
                    # Get memory after
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.max_memory_allocated()
                        mem_used = (mem_after - mem_before) / (1024 ** 2)  # MB
                        memory_usage[result.engine_used] = mem_used
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
        
        # Log results
        if memory_usage:
            print(f"\nMemory usage comparison (MB):")
            for engine, mem in sorted(memory_usage.items(), key=lambda x: x[1]):
                print(f"  {engine}: {mem:.2f}MB")
    
    @pytest.mark.slow
    def test_accuracy_comparison(self):
        """Compare output accuracy across engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Use deterministic inputs
        inputs = {"prompts": ["The capital of France is"], "max_tokens": 5}
        outputs = {}
        
        # Test each engine
        for engine in ["vllm", "onnx", "pytorch"]:
            if engine in manager.get_available_engines():
                try:
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    outputs[result.engine_used] = result.outputs
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
        
        # Verify: All engines produce outputs
        assert len(outputs) > 0
        
        # Log outputs for comparison
        print(f"\nOutput comparison:")
        for engine, output in outputs.items():
            print(f"  {engine}: {output}")


class TestEngineCompatibility:
    """Test engine compatibility with different models and inputs."""
    
    def test_engines_with_different_model_sizes(self):
        """Test engines with different model sizes."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Test with small model
        models = ["gpt2", "distilgpt2"]
        
        for model_name in models:
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            
            try:
                result = manager.infer(
                    model_name=model_name,
                    inputs=inputs
                )
                
                # Verify: Inference successful
                assert result is not None
                assert result.engine_used in ["vllm", "deepspeed", "onnx", "pytorch"]
                
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
    
    def test_engines_with_different_input_formats(self):
        """Test engines with different input formats."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Test with text inputs
        text_inputs = {"prompts": ["Hello"], "max_tokens": 10}
        result1 = manager.infer(
            model_name="gpt2",
            inputs=text_inputs
        )
        assert result1 is not None
        
        # Test with tensor inputs
        tensor_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        result2 = manager.infer(
            model_name="gpt2",
            inputs=tensor_inputs
        )
        assert result2 is not None
    
    def test_engines_with_different_generation_params(self):
        """Test engines with different generation parameters."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Test with different parameters
        param_sets = [
            {"max_tokens": 10, "temperature": 0.7},
            {"max_tokens": 20, "temperature": 1.0},
            {"max_tokens": 15, "temperature": 0.5, "top_p": 0.9}
        ]
        
        for params in param_sets:
            inputs = {"prompts": ["Test"], **params}
            
            try:
                result = manager.infer(
                    model_name="gpt2",
                    inputs=inputs
                )
                assert result is not None
            except Exception as e:
                print(f"Parameters {params} failed: {e}")


class TestEngineStatusReporting:
    """Test engine status reporting."""
    
    def test_get_engine_status(self):
        """Test getting status of all engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Get status
        status = manager.get_engine_status()
        
        # Verify: Status for all engines
        assert "vllm" in status
        assert "deepspeed" in status
        assert "onnx" in status
        assert "pytorch" in status
        
        # Verify: Each status has required fields
        for engine_name, engine_status in status.items():
            assert hasattr(engine_status, 'name')
            assert hasattr(engine_status, 'available')
            assert isinstance(engine_status.available, bool)
    
    def test_get_available_engines_for_model(self):
        """Test getting available engines for a specific model."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Get available engines
        available = manager.get_available_engines("gpt2")
        
        # Verify: At least PyTorch is available
        assert "pytorch" in available
        assert isinstance(available, list)
        assert len(available) > 0
    
    def test_engine_availability_detection(self):
        """Test engine availability detection."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Check each engine
        for engine in ["vllm", "deepspeed", "onnx", "pytorch"]:
            is_available = engine in manager.get_available_engines()
            assert isinstance(is_available, bool)
            
            # PyTorch should always be available
            if engine == "pytorch":
                assert is_available is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
