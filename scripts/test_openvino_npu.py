#!/usr/bin/env python3
"""
OpenVINO NPU Test Script
Tests Intel NPU performance with GPT-2 model
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_openvino_devices():
    """Test available OpenVINO devices."""
    print("=" * 80)
    print("OpenVINO Device Detection")
    print("=" * 80)
    
    try:
        from openvino.runtime import Core
        
        core = Core()
        devices = core.available_devices
        
        print(f"Available devices: {devices}")
        print()
        
        for device in devices:
            print(f"Device: {device}")
            try:
                properties = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  Full name: {properties}")
            except:
                print(f"  Full name: Not available")
            print()
            
        return devices
        
    except Exception as e:
        print(f"Error: {e}")
        return []


def test_model_export():
    """Export GPT-2 model to OpenVINO format."""
    print("=" * 80)
    print("Model Export to OpenVINO")
    print("=" * 80)
    
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer
        
        model_id = "gpt2"
        export_dir = Path("models/openvino/gpt2")
        
        print(f"Exporting {model_id} to OpenVINO format...")
        print(f"Export directory: {export_dir}")
        print()
        
        # Check if already exported
        if export_dir.exists() and (export_dir / "openvino_model.xml").exists():
            print("✅ Model already exported")
            return export_dir
        
        # Export model
        print("Exporting model (this may take a few minutes)...")
        start_time = time.time()
        
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            compile=False  # Don't compile yet
        )
        
        # Save exported model
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(export_dir)
        
        export_time = time.time() - start_time
        print(f"✅ Model exported in {export_time:.2f} seconds")
        print(f"   Saved to: {export_dir}")
        print()
        
        return export_dir
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def benchmark_device(device_name, model_dir, num_runs=5):
    """Benchmark model on specific device."""
    print("=" * 80)
    print(f"Benchmarking on {device_name}")
    print("=" * 80)
    
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer
        import torch
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model for specific device
        print(f"Loading model on {device_name}...")
        start_load = time.time()
        
        model = OVModelForCausalLM.from_pretrained(
            model_dir,
            device=device_name,
            compile=True
        )
        
        load_time = time.time() - start_load
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        print()
        
        # Prepare test input
        test_prompt = "The quick brown fox jumps over the lazy dog"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Warmup run
        print("Warming up...")
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        print("✅ Warmup complete")
        print()
        
        # Benchmark runs
        print(f"Running {num_runs} benchmark iterations...")
        latencies = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False
                )
            
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            print(f"  Run {i+1}/{num_runs}: {latency:.2f} ms")
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Calculate tokens/second
        output_length = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_per_sec = (output_length * 1000) / avg_latency
        
        print()
        print("Results:")
        print(f"  Average latency: {avg_latency:.2f} ms")
        print(f"  Min latency: {min_latency:.2f} ms")
        print(f"  Max latency: {max_latency:.2f} ms")
        print(f"  Tokens generated: {output_length}")
        print(f"  Tokens/second: {tokens_per_sec:.2f}")
        print()
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated text:")
        print(f"  {generated_text}")
        print()
        
        return {
            "device": device_name,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "tokens_per_sec": tokens_per_sec,
            "load_time_sec": load_time
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(results):
    """Compare benchmark results across devices."""
    print("=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print()
    
    if not results:
        print("No results to compare")
        return
    
    # Find baseline (CPU)
    cpu_result = next((r for r in results if r and r['device'] == 'CPU'), None)
    
    print(f"{'Device':<10} {'Latency (ms)':<15} {'Tokens/s':<12} {'Speedup':<10} {'Load Time (s)':<15}")
    print("-" * 80)
    
    for result in results:
        if not result:
            continue
            
        device = result['device']
        latency = result['avg_latency_ms']
        tokens_per_sec = result['tokens_per_sec']
        load_time = result['load_time_sec']
        
        # Calculate speedup vs CPU
        if cpu_result and device != 'CPU':
            speedup = cpu_result['avg_latency_ms'] / latency
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "baseline"
        
        print(f"{device:<10} {latency:<15.2f} {tokens_per_sec:<12.2f} {speedup_str:<10} {load_time:<15.2f}")
    
    print()
    
    # Recommendations
    print("Recommendations:")
    print()
    
    if any(r and r['device'] == 'NPU' for r in results):
        npu_result = next(r for r in results if r and r['device'] == 'NPU')
        if cpu_result:
            speedup = cpu_result['avg_latency_ms'] / npu_result['avg_latency_ms']
            print(f"✅ NPU provides {speedup:.2f}x speedup over CPU")
            print(f"   Use NPU for production deployment")
        print()
    
    if any(r and r['device'] == 'GPU' for r in results):
        gpu_result = next(r for r in results if r and r['device'] == 'GPU')
        if cpu_result:
            speedup = cpu_result['avg_latency_ms'] / gpu_result['avg_latency_ms']
            print(f"✅ iGPU provides {speedup:.2f}x speedup over CPU")
            print(f"   Use iGPU as fallback if NPU is busy")
        print()


def main():
    """Main execution."""
    print("=" * 80)
    print("OpenVINO NPU Performance Test")
    print("=" * 80)
    print()
    
    # Step 1: Detect devices
    devices = test_openvino_devices()
    
    if not devices:
        print("❌ No OpenVINO devices found")
        return
    
    # Step 2: Export model
    model_dir = test_model_export()
    
    if not model_dir:
        print("❌ Model export failed")
        return
    
    # Step 3: Benchmark each device
    results = []
    
    # Always test CPU first (baseline)
    if 'CPU' in devices:
        result = benchmark_device('CPU', model_dir, num_runs=3)
        results.append(result)
    
    # Test NPU if available
    if 'NPU' in devices:
        result = benchmark_device('NPU', model_dir, num_runs=3)
        results.append(result)
    
    # Test GPU if available
    if 'GPU' in devices:
        result = benchmark_device('GPU', model_dir, num_runs=3)
        results.append(result)
    
    # Step 4: Compare results
    compare_results(results)
    
    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
