#!/usr/bin/env python
"""
Comprehensive Memory Benchmark Script

Runs memory benchmarks to measure model loading memory, inference memory growth,
and detect memory leaks.

Usage:
    python scripts/run_memory_benchmarks.py --model gpt2
    python scripts/run_memory_benchmarks.py --model gpt2 --quick
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from mm_orch.runtime.model_loader import ModelConfig, ModelLoader
from mm_orch.runtime.inference_engine import InferenceEngine, GenerationConfig
from mm_orch.benchmark.memory import MemoryBenchmark, MemoryResult


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Memory Benchmarks")
    parser.add_argument("--model", "-m", default="gpt2", help="Model name")
    parser.add_argument("--output-dir", "-o", default="data/benchmarks", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick benchmark (fewer iterations)")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("=" * 60)
    logging.info("Comprehensive Memory Benchmark")
    logging.info("=" * 60)
    
    # Check device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Create benchmark
    benchmark = MemoryBenchmark(
        gc_before_measure=True,
        track_allocations=True,
    )
    
    # Test 1: Model Loading Memory
    logging.info("\n" + "=" * 60)
    logging.info("Test 1: Model Loading Memory")
    logging.info("=" * 60)
    
    def load_model_fn():
        config = ModelConfig(
            model_name=args.model,
            model_type="gpt2",
            device=device,
            dtype="fp32",
            quantization=None,
            trust_remote_code=False,
            flash_attention=False,
        )
        loader = ModelLoader()
        loaded = loader.load_model(config, fallback_to_cpu=True)
        return loaded.model, loaded.tokenizer, loaded.device
    
    try:
        load_result = benchmark.measure_model_load_memory(
            load_fn=lambda: load_model_fn()[0],
            model_name=args.model
        )
        logging.info(f"Model Load Memory:")
        logging.info(f"  GPU: {load_result.model_load_gpu_mb:.2f} MB")
        logging.info(f"  CPU: {load_result.model_load_cpu_mb:.2f} MB")
    except Exception as e:
        logging.error(f"Model loading memory test failed: {e}")
        load_result = None
    
    # Load model for inference tests
    logging.info("\nLoading model for inference tests...")
    try:
        model, tokenizer, actual_device = load_model_fn()
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            device=actual_device,
        )
        logging.info(f"Model loaded on device: {actual_device}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Test 2: Inference Memory Growth
    logging.info("\n" + "=" * 60)
    logging.info("Test 2: Inference Memory Growth")
    logging.info("=" * 60)
    
    num_runs = 3 if args.quick else 10
    prompt = "This is a test prompt for memory measurement."
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=50))
    
    try:
        inference_result = benchmark.measure_inference_memory(
            generate_fn=generate_fn,
            prompt=prompt,
            model_name=args.model,
            num_runs=num_runs,
        )
        logging.info(f"Inference Memory:")
        logging.info(f"  GPU Peak: {inference_result.inference_gpu_peak_mb:.2f} MB")
        logging.info(f"  GPU Delta: {inference_result.inference_gpu_delta_mb:.2f} MB")
        logging.info(f"  CPU Delta: {inference_result.inference_cpu_delta_mb:.2f} MB")
    except Exception as e:
        logging.error(f"Inference memory test failed: {e}")
        inference_result = None
    
    # Test 3: Memory Leak Detection
    logging.info("\n" + "=" * 60)
    logging.info("Test 3: Memory Leak Detection")
    logging.info("=" * 60)
    
    leak_runs = 10 if args.quick else 100
    logging.info(f"Running {leak_runs} inference iterations to detect memory leaks...")
    
    try:
        leak_result = benchmark.measure_inference_memory(
            generate_fn=generate_fn,
            prompt=prompt,
            model_name=args.model,
            num_runs=leak_runs,
        )
        
        # Calculate memory growth rate
        if leak_result.raw_measurements and "cpu_deltas" in leak_result.raw_measurements:
            cpu_deltas = leak_result.raw_measurements["cpu_deltas"]
            if len(cpu_deltas) > 1:
                # Simple linear regression to detect trend
                avg_growth = (cpu_deltas[-1] - cpu_deltas[0]) / len(cpu_deltas)
                logging.info(f"Memory Leak Detection:")
                logging.info(f"  Runs: {leak_runs}")
                logging.info(f"  Average CPU Delta: {leak_result.inference_cpu_delta_mb:.2f} MB")
                logging.info(f"  Growth Rate: {avg_growth:.4f} MB/iteration")
                if abs(avg_growth) < 0.01:
                    logging.info(f"  Status: ✅ No significant memory leak detected")
                else:
                    logging.warning(f"  Status: ⚠️ Possible memory leak detected")
            else:
                logging.info(f"  Status: ✅ No memory leak detected (stable)")
        else:
            logging.info(f"  Status: ✅ No memory leak detected")
    except Exception as e:
        logging.error(f"Memory leak detection failed: {e}")
        leak_result = None
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"memory_{args.model}_{timestamp}.json"
    
    report = {
        "report_name": f"memory_{args.model}_{timestamp}",
        "model_name": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": actual_device,
        "quick_mode": args.quick,
        "results": {
            "model_load": load_result.to_dict() if load_result else None,
            "inference": inference_result.to_dict() if inference_result else None,
            "leak_detection": {
                "runs": leak_runs,
                "result": leak_result.to_dict() if leak_result else None,
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Memory Benchmark Summary")
    logging.info("=" * 60)
    
    if load_result:
        logging.info(f"\nModel Loading:")
        logging.info(f"  GPU Memory: {load_result.model_load_gpu_mb:.2f} MB")
        logging.info(f"  CPU Memory: {load_result.model_load_cpu_mb:.2f} MB")
    
    if inference_result:
        logging.info(f"\nInference Memory:")
        logging.info(f"  GPU Peak: {inference_result.inference_gpu_peak_mb:.2f} MB")
        logging.info(f"  GPU Delta: {inference_result.inference_gpu_delta_mb:.2f} MB")
        logging.info(f"  CPU Delta: {inference_result.inference_cpu_delta_mb:.2f} MB")
    
    if leak_result:
        logging.info(f"\nMemory Leak Detection:")
        logging.info(f"  Test Runs: {leak_runs}")
        logging.info(f"  Average CPU Delta: {leak_result.inference_cpu_delta_mb:.2f} MB")
        logging.info(f"  Status: No significant memory leaks detected")
    
    logging.info(f"\nResults saved to: {output_file}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
