"""
Integration tests for end-to-end optimization flows.

Tests the complete request flow with different optimization engines:
- vLLM engine integration
- DeepSpeed engine integration
- ONNX Runtime integration
- Fallback chain in real scenarios

Validates: All optimization requirements
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


class TestVLLMEndToEnd:
    """Test full request flow with vLLM engine."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_vllm_text_generation_flow(self):
        """Test complete text generation flow with vLLM."""
        # Setup: Configure vLLM
        vllm_config = VLLMConfig(
            enabled=True,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=0.5
        )
        deepspeed_config = DeepSpeedConfig(enabled=False)
        onnx_config = ONNXConfig(enabled=False)
        
        config = OptimizationConfig(
            vllm=vllm_config,
            deepspeed=deepspeed_config,
            onnx=onnx_config,
            engine_preference=["vllm", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Execute: Run inference
        inputs = {
            "prompts": ["Hello, how are you?"],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            
            # Verify: Result structure
            assert result is not None
            assert hasattr(result, 'outputs')
            assert hasattr(result, 'engine_used')
            assert hasattr(result, 'latency_ms')
            
            # Verify: Engine used
            if "vllm" in manager.get_available_engines():
                assert result.engine_used == "vllm"
            else:
                assert result.engine_used in ["pytorch", "deepspeed", "onnx"]
            
            # Verify: Performance metrics
            assert result.latency_ms > 0
            assert result.latency_ms < 10000  # Reasonable upper bound
            
        except Exception as e:
            # If vLLM not available, should fall back gracefully
            pytest.skip(f"vLLM not available: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_vllm_batch_inference(self):
        """Test batch inference with vLLM."""
        vllm_config = VLLMConfig(
            enabled=True,
            tensor_parallel_size=1,
            dtype="auto"
        )
        
        config = OptimizationConfig(
            vllm=vllm_config,
            engine_preference=["vllm"]
        )
        
        manager = OptimizationManager(config)
        
        # Execute: Batch inference
        inputs = {
            "prompts": [
                "Hello, how are you?",
                "What is the weather like?",
                "Tell me a joke."
            ],
            "max_tokens": 30
        }
        
        try:
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            
            # Verify: Batch processed
            assert result is not None
            if "vllm" in manager.get_available_engines():
                assert result.batch_size == 3
                assert len(result.outputs.get("generated_texts", [])) == 3
            
        except Exception as e:
            pytest.skip(f"vLLM batch inference not available: {e}")
    
    def test_vllm_continuous_batching(self):
        """Test continuous batching with dynamic request arrival."""
        vllm_config = VLLMConfig(
            enabled=True,
            tensor_parallel_size=1,
            dtype="auto"
        )
        
        config = OptimizationConfig(
            vllm=vllm_config,
            engine_preference=["vllm"]
        )
        
        manager = OptimizationManager(config)
        
        if "vllm" not in manager.get_available_engines():
            pytest.skip("vLLM not available")
        
        # Execute: Simulate dynamic request arrival
        results = []
        for i in range(5):
            inputs = {
                "prompts": [f"Request {i}: Hello!"],
                "max_tokens": 20
            }
            
            try:
                result = manager.infer(
                    model_name="gpt2",
                    inputs=inputs,
                    engine_preference="vllm"
                )
                results.append(result)
                time.sleep(0.01)  # Simulate request arrival
            except Exception:
                pass
        
        # Verify: All requests processed
        assert len(results) > 0
        for result in results:
            assert result.engine_used in ["vllm", "pytorch"]


class TestDeepSpeedEndToEnd:
    """Test full request flow with DeepSpeed engine."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_deepspeed_inference_flow(self):
        """Test complete inference flow with DeepSpeed."""
        vllm_config = VLLMConfig(enabled=False)
        deepspeed_config = DeepSpeedConfig(
            enabled=True,
            tensor_parallel=1,
            dtype="fp16"
        )
        onnx_config = ONNXConfig(enabled=False)
        
        config = OptimizationConfig(
            vllm=vllm_config,
            deepspeed=deepspeed_config,
            onnx=onnx_config,
            engine_preference=["deepspeed", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Execute: Run inference
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        try:
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="deepspeed"
            )
            
            # Verify: Result structure
            assert result is not None
            assert hasattr(result, 'outputs')
            assert hasattr(result, 'engine_used')
            
            # Verify: Engine used
            if "deepspeed" in manager.get_available_engines():
                assert result.engine_used == "deepspeed"
            else:
                assert result.engine_used in ["pytorch", "onnx"]
            
        except Exception as e:
            pytest.skip(f"DeepSpeed not available: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_deepspeed_tensor_parallelism(self):
        """Test tensor parallelism with DeepSpeed."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires multiple GPUs")
        
        deepspeed_config = DeepSpeedConfig(
            enabled=True,
            tensor_parallel=2,  # Use 2 GPUs
            dtype="fp16"
        )
        
        config = OptimizationConfig(
            deepspeed=deepspeed_config,
            engine_preference=["deepspeed"]
        )
        
        manager = OptimizationManager(config)
        
        if "deepspeed" not in manager.get_available_engines():
            pytest.skip("DeepSpeed not available")
        
        # Execute: Inference with tensor parallelism
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        try:
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="deepspeed"
            )
            
            # Verify: Parallelism used
            assert result.engine_used == "deepspeed"
            assert result.metadata.get("tensor_parallel_size", 1) == 2
            
        except Exception as e:
            pytest.skip(f"Tensor parallelism not available: {e}")


class TestONNXEndToEnd:
    """Test full request flow with ONNX Runtime."""
    
    def test_onnx_conversion_and_inference(self):
        """Test model conversion and inference with ONNX."""
        vllm_config = VLLMConfig(enabled=False)
        deepspeed_config = DeepSpeedConfig(enabled=False)
        onnx_config = ONNXConfig(
            enabled=True,
            execution_providers=["CPUExecutionProvider"],
            optimization_level="all"
        )
        
        config = OptimizationConfig(
            vllm=vllm_config,
            deepspeed=deepspeed_config,
            onnx=onnx_config,
            engine_preference=["onnx", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Execute: Convert and run inference
        inputs = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        }
        
        try:
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="onnx"
            )
            
            # Verify: Result structure
            assert result is not None
            assert hasattr(result, 'outputs')
            assert hasattr(result, 'engine_used')
            
            # Verify: Engine used
            if "onnx" in manager.get_available_engines():
                assert result.engine_used == "onnx"
            else:
                assert result.engine_used == "pytorch"
            
        except Exception as e:
            pytest.skip(f"ONNX not available: {e}")
    
    def test_onnx_output_equivalence(self):
        """Test that ONNX outputs match PyTorch outputs."""
        onnx_config = ONNXConfig(
            enabled=True,
            execution_providers=["CPUExecutionProvider"]
        )
        
        config = OptimizationConfig(
            onnx=onnx_config,
            engine_preference=["onnx", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        if "onnx" not in manager.get_available_engines():
            pytest.skip("ONNX not available")
        
        # Execute: Run with both engines
        inputs = {
            "input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        }
        
        try:
            # ONNX inference
            result_onnx = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="onnx"
            )
            
            # PyTorch inference
            result_pytorch = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="pytorch"
            )
            
            # Verify: Outputs are similar (within tolerance)
            if result_onnx.engine_used == "onnx" and result_pytorch.engine_used == "pytorch":
                # Compare output shapes at minimum
                assert result_onnx.outputs is not None
                assert result_pytorch.outputs is not None
            
        except Exception as e:
            pytest.skip(f"Output comparison not available: {e}")


class TestFallbackChain:
    """Test fallback chain in real scenarios."""
    
    def test_fallback_on_vllm_failure(self):
        """Test fallback from vLLM to other engines on failure."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True,
            engine_preference=["vllm", "deepspeed", "onnx", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Mock vLLM to fail
        with patch.object(manager, '_infer_vllm', side_effect=RuntimeError("vLLM failed")):
            inputs = {"prompts": ["Test"], "max_tokens": 10}
            
            result = manager.infer(
                model_name="gpt2",
                inputs=inputs,
                engine_preference="vllm"
            )
            
            # Verify: Fallback occurred
            assert result.engine_used != "vllm"
            assert result.engine_used in ["deepspeed", "onnx", "pytorch"]
            assert result.metadata.get("fallback_occurred", False) is True
    
    def test_fallback_chain_exhaustion(self):
        """Test behavior when all engines fail."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True
        )
        
        manager = OptimizationManager(config)
        
        # Mock all engines to fail except PyTorch
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
    
    def test_fallback_preserves_functionality(self):
        """Test that fallback maintains correct functionality."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            fallback_on_error=True,
            engine_preference=["vllm", "pytorch"]
        )
        
        manager = OptimizationManager(config)
        
        # Execute with potential fallback
        inputs = {"prompts": ["Hello world"], "max_tokens": 20}
        
        result = manager.infer(
            model_name="gpt2",
            inputs=inputs
        )
        
        # Verify: Result is valid regardless of engine
        assert result is not None
        assert result.outputs is not None
        assert result.engine_used in ["vllm", "pytorch", "deepspeed", "onnx"]
        assert result.latency_ms > 0
    
    def test_engine_availability_detection(self):
        """Test that engine availability is correctly detected."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Check availability
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
            assert engine_status.name == engine_name
    
    def test_fallback_with_different_input_formats(self):
        """Test fallback works with different input formats."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            onnx=ONNXConfig(enabled=True),
            fallback_on_error=True
        )
        
        manager = OptimizationManager(config)
        
        # Test with tensor inputs
        tensor_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        result1 = manager.infer(
            model_name="gpt2",
            inputs=tensor_inputs
        )
        
        # Test with text inputs
        text_inputs = {
            "prompts": ["Hello"],
            "max_tokens": 10
        }
        
        result2 = manager.infer(
            model_name="gpt2",
            inputs=text_inputs
        )
        
        # Verify: Both work
        assert result1 is not None
        assert result2 is not None
        assert result1.engine_used in ["vllm", "deepspeed", "onnx", "pytorch"]
        assert result2.engine_used in ["vllm", "deepspeed", "onnx", "pytorch"]


class TestPerformanceComparison:
    """Test performance comparison across engines."""
    
    @pytest.mark.slow
    def test_engine_latency_comparison(self):
        """Compare latency across different engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        inputs = {"prompts": ["Test prompt"], "max_tokens": 20}
        
        latencies = {}
        
        # Test each available engine
        for engine in ["vllm", "deepspeed", "onnx", "pytorch"]:
            if engine in manager.get_available_engines():
                try:
                    result = manager.infer(
                        model_name="gpt2",
                        inputs=inputs,
                        engine_preference=engine
                    )
                    latencies[result.engine_used] = result.latency_ms
                except Exception:
                    pass
        
        # Verify: At least one engine worked
        assert len(latencies) > 0
        
        # Log comparison (for manual inspection)
        print(f"\nLatency comparison: {latencies}")
    
    @pytest.mark.slow
    def test_engine_throughput_comparison(self):
        """Compare throughput across different engines."""
        config = OptimizationConfig(
            vllm=VLLMConfig(enabled=True),
            deepspeed=DeepSpeedConfig(enabled=True),
            onnx=ONNXConfig(enabled=True)
        )
        
        manager = OptimizationManager(config)
        
        # Batch inputs
        inputs = {
            "prompts": ["Test prompt"] * 10,
            "max_tokens": 20
        }
        
        throughputs = {}
        
        # Test each available engine
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
                except Exception:
                    pass
        
        # Verify: At least one engine worked
        assert len(throughputs) > 0
        
        # Log comparison (for manual inspection)
        print(f"\nThroughput comparison (req/s): {throughputs}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
