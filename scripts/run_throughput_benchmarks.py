#!/usr/bin/env python
"""
Comprehensive Throughput Benchmark Script

Runs throughput benchmarks to measure single request, concurrent, and batch
processing performance.

Usage:
    python scripts/run_throughput_benchmarks.py --model gpt2
    python scripts/run_throughput_benchmarks.py --model gpt2 --quick
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
from mm_orch.benchmark.throughput import ThroughputBenchmark, ThroughputResult


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
    parser = argparse.ArgumentParser(description="Run Throughput Benchmarks")
    parser.add_argument("--model", "-m", default="gpt2", help="Model name")
    parser.add_argument("--output-dir", "-o", default="data/benchmarks", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick benchmark (fewer iterations)")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("=" * 60)
    logging.info("Comprehensive Throughput Benchmark")
    logging.info("=" * 60)
    
    # Check device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load model
    logging.info("Loading model...")
    try:
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
        model, tokenizer, actual_device = loaded.model, loaded.tokenizer, loaded.device
        logging.info(f"Model loaded on device: {actual_device}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=actual_device,
    )
    
    # Create benchmark
    benchmark = ThroughputBenchmark(
        duration_seconds=30.0,
        warmup_requests=2,
    )
    
    # Test prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing briefly.",
    ]
    
    # Test 1: Single Request Throughput
    logging.info("\n" + "=" * 60)
    logging.info("Test 1: Single Request Throughput")
    logging.info("=" * 60)
    
    num_requests = 5 if args.quick else 10
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=30))
    
    def get_output_tokens(result) -> int:
        return result.output_tokens
    
    try:
        single_result = benchmark.measure_single_throughput(
            generate_fn=generate_fn,
            prompt=prompts[0],
            model_name=args.model,
            get_output_tokens=get_output_tokens,
            num_requests=num_requests,
        )
        logging.info(f"Single Throughput:")
        logging.info(f"  Requests/s: {single_result.single_requests_per_second:.2f}")
        logging.info(f"  Tokens/s: {single_result.single_tokens_per_second:.2f}")
        logging.info(f"  Mean Latency: {single_result.latency_mean_ms:.2f} ms")
    except Exception as e:
        logging.error(f"Single throughput test failed: {e}")
        single_result = None
    
    # Test 2: Concurrent Throughput
    logging.info("\n" + "=" * 60)
    logging.info("Test 2: Concurrent Throughput")
    logging.info("=" * 60)
    
    concurrent_levels = [2, 4] if args.quick else [2, 4, 8]
    concurrent_results = []
    
    for level in concurrent_levels:
        logging.info(f"\nTesting concurrency level: {level}")
        max_requests = 6 if args.quick else 12
        
        try:
            concurrent_result = benchmark.measure_concurrent_throughput(
                generate_fn=generate_fn,
                prompts=prompts,
                model_name=args.model,
                get_output_tokens=get_output_tokens,
                num_concurrent=level,
                max_requests=max_requests,
            )
            logging.info(f"Concurrent Throughput (level={level}):")
            logging.info(f"  Requests/s: {concurrent_result.concurrent_requests_per_second:.2f}")
            logging.info(f"  Tokens/s: {concurrent_result.concurrent_tokens_per_second:.2f}")
            logging.info(f"  Mean Latency: {concurrent_result.latency_mean_ms:.2f} ms")
            concurrent_results.append(concurrent_result)
        except Exception as e:
            logging.error(f"Concurrent throughput test (level={level}) failed: {e}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"throughput_{args.model}_{timestamp}.json"
    
    report = {
        "report_name": f"throughput_{args.model}_{timestamp}",
        "model_name": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": actual_device,
        "quick_mode": args.quick,
        "results": {
            "single": single_result.to_dict() if single_result else None,
            "concurrent": [r.to_dict() for r in concurrent_results],
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("Throughput Benchmark Summary")
    logging.info("=" * 60)
    
    if single_result:
        logging.info(f"\nSingle Request:")
        logging.info(f"  Requests/s: {single_result.single_requests_per_second:.2f}")
        logging.info(f"  Tokens/s: {single_result.single_tokens_per_second:.2f}")
        logging.info(f"  Mean Latency: {single_result.latency_mean_ms:.2f} ms")
        logging.info(f"  P95 Latency: {single_result.latency_p95_ms:.2f} ms")
    
    if concurrent_results:
        logging.info(f"\nConcurrent Requests:")
        for result in concurrent_results:
            logging.info(f"  Level {result.concurrent_level}:")
            logging.info(f"    Requests/s: {result.concurrent_requests_per_second:.2f}")
            logging.info(f"    Tokens/s: {result.concurrent_tokens_per_second:.2f}")
            logging.info(f"    Mean Latency: {result.latency_mean_ms:.2f} ms")
    
    logging.info(f"\nResults saved to: {output_file}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
