#!/usr/bin/env python
"""
Performance Benchmark Report Generator

This script generates a comprehensive benchmark report for the GPT-2 model,
including latency, memory, and throughput measurements.

Usage:
    python scripts/generate_benchmark_report.py
    python scripts/generate_benchmark_report.py --model gpt2
    python scripts/generate_benchmark_report.py --format csv
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from mm_orch.runtime.model_loader import ModelConfig, ModelLoader
from mm_orch.runtime.inference_engine import InferenceEngine, GenerationConfig
from mm_orch.benchmark.latency import LatencyBenchmark, LatencyResult
from mm_orch.benchmark.memory import MemoryBenchmark, MemoryResult
from mm_orch.benchmark.throughput import ThroughputBenchmark, ThroughputResult
from mm_orch.benchmark.reporter import BenchmarkReporter, BenchmarkReport


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


def load_model(model_name: str = "gpt2", device: str = "auto") -> tuple:
    """Load model and tokenizer."""
    logging.info(f"Loading model: {model_name}")
    
    config = ModelConfig(
        model_name=model_name,
        model_type="gpt2",
        device=device,
        dtype="fp32",
        quantization=None,
        trust_remote_code=False,
        flash_attention=False,
    )
    
    loader = ModelLoader()
    loaded = loader.load_model(config, fallback_to_cpu=True)
    
    logging.info(f"Model loaded on device: {loaded.device}")
    return loaded.model, loaded.tokenizer, loaded.device


def run_latency_benchmark(
    engine: InferenceEngine,
    model_name: str,
    warmup_runs: int = 2,
    test_runs: int = 5
) -> LatencyResult:
    """Run latency benchmark."""
    logging.info("Running latency benchmark...")
    
    benchmark = LatencyBenchmark(
        warmup_runs=warmup_runs,
        test_runs=test_runs,
    )
    
    prompt = "What is the capital of France?"
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=50))
    
    def get_output_tokens(result) -> int:
        return result.output_tokens
    
    result = benchmark.run_latency_suite(
        generate_fn=generate_fn,
        prompt=prompt,
        model_name=model_name,
        get_output_tokens=get_output_tokens,
        test_name="latency_benchmark",
    )
    
    logging.info(f"Latency Results:")
    logging.info(f"  TTFT Mean: {result.ttft_mean * 1000:.2f} ms")
    logging.info(f"  Tokens/s: {result.tokens_per_second_mean:.2f}")
    logging.info(f"  E2E Latency: {result.e2e_latency_mean * 1000:.2f} ms")
    
    return result


def run_memory_benchmark(
    engine: InferenceEngine,
    model_name: str
) -> MemoryResult:
    """Run memory benchmark."""
    logging.info("Running memory benchmark...")
    
    benchmark = MemoryBenchmark(
        gc_before_measure=True,
        track_allocations=True,
    )
    
    prompt = "This is a test prompt for memory measurement."
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=50))
    
    result = benchmark.measure_inference_memory(
        generate_fn=generate_fn,
        prompt=prompt,
        model_name=model_name,
        num_runs=3,
    )
    
    logging.info(f"Memory Results:")
    logging.info(f"  Inference GPU Peak: {result.inference_gpu_peak_mb:.2f} MB")
    logging.info(f"  Inference GPU Delta: {result.inference_gpu_delta_mb:.2f} MB")
    logging.info(f"  Inference CPU Delta: {result.inference_cpu_delta_mb:.2f} MB")
    
    return result


def run_throughput_benchmark(
    engine: InferenceEngine,
    model_name: str,
    num_requests: int = 5
) -> ThroughputResult:
    """Run throughput benchmark."""
    logging.info("Running throughput benchmark...")
    
    benchmark = ThroughputBenchmark(
        duration_seconds=30.0,
        warmup_requests=2,
    )
    
    prompt = "Hello, how are you?"
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=30))
    
    def get_output_tokens(result) -> int:
        return result.output_tokens
    
    result = benchmark.measure_single_throughput(
        generate_fn=generate_fn,
        prompt=prompt,
        model_name=model_name,
        get_output_tokens=get_output_tokens,
        num_requests=num_requests,
    )
    
    logging.info(f"Throughput Results:")
    logging.info(f"  Requests/s: {result.single_requests_per_second:.2f}")
    logging.info(f"  Tokens/s: {result.single_tokens_per_second:.2f}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument("--model", "-m", default="gpt2", help="Model name")
    parser.add_argument("--format", "-f", default="json", choices=["json", "csv"], help="Output format")
    parser.add_argument("--output-dir", "-o", default="data/benchmarks", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick benchmark (fewer iterations)")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("=" * 60)
    logging.info("Performance Benchmark Report Generator")
    logging.info("=" * 60)
    
    # Check device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load model
    try:
        model, tokenizer, actual_device = load_model(args.model, device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=actual_device,
    )
    
    # Benchmark parameters
    warmup_runs = 1 if args.quick else 2
    test_runs = 3 if args.quick else 5
    num_requests = 3 if args.quick else 5
    
    # Run benchmarks
    latency_results = []
    memory_results = []
    throughput_results = []
    
    try:
        # Latency benchmark
        latency_result = run_latency_benchmark(
            engine, args.model,
            warmup_runs=warmup_runs, test_runs=test_runs
        )
        latency_results.append(latency_result)
    except Exception as e:
        logging.warning(f"Latency benchmark failed: {e}")
    
    try:
        # Memory benchmark
        memory_result = run_memory_benchmark(engine, args.model)
        memory_results.append(memory_result)
    except Exception as e:
        logging.warning(f"Memory benchmark failed: {e}")
    
    try:
        # Throughput benchmark
        throughput_result = run_throughput_benchmark(
            engine, args.model, num_requests=num_requests
        )
        throughput_results.append(throughput_result)
    except Exception as e:
        logging.warning(f"Throughput benchmark failed: {e}")
    
    # Generate report
    logging.info("Generating benchmark report...")
    
    reporter = BenchmarkReporter(
        output_dir=args.output_dir,
        include_system_info=True,
    )
    
    report = reporter.create_report(
        model_name=args.model,
        latency_results=latency_results,
        memory_results=memory_results,
        throughput_results=throughput_results,
        metadata={
            "device": actual_device,
            "quick_mode": args.quick,
            "warmup_runs": warmup_runs,
            "test_runs": test_runs,
        }
    )
    
    # Save report
    if args.format == "json":
        filepath = reporter.save_json(report)
    else:
        filepath = reporter.save_csv(report)
    
    # Print summary
    logging.info("=" * 60)
    logging.info("Benchmark Report Summary")
    logging.info("=" * 60)
    
    summary = report.get_summary()
    logging.info(f"Model: {summary['model_name']}")
    logging.info(f"Timestamp: {summary['timestamp']}")
    
    if "latency" in summary:
        logging.info(f"Latency:")
        logging.info(f"  TTFT: {summary['latency']['ttft_mean_ms']:.2f} ms")
        logging.info(f"  Tokens/s: {summary['latency']['tokens_per_second']:.2f}")
    
    if "memory" in summary:
        logging.info(f"Memory:")
        logging.info(f"  GPU Peak: {summary['memory']['inference_gpu_peak_mb']:.2f} MB")
    
    if "throughput" in summary:
        logging.info(f"Throughput:")
        logging.info(f"  Requests/s: {summary['throughput']['requests_per_second']:.2f}")
        logging.info(f"  Tokens/s: {summary['throughput']['tokens_per_second']:.2f}")
    
    logging.info(f"\nReport saved to: {filepath}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
