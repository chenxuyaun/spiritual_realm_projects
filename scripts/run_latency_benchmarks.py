#!/usr/bin/env python
"""
Comprehensive Latency Benchmark Script

Runs latency benchmarks with different input and output lengths to establish
baseline performance metrics.

Usage:
    python scripts/run_latency_benchmarks.py --model gpt2
    python scripts/run_latency_benchmarks.py --model gpt2 --quick
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
from mm_orch.benchmark.latency import LatencyBenchmark, LatencyResult


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


def create_prompt(target_length: int) -> str:
    """Create a prompt of approximately target_length tokens."""
    base_prompts = {
        128: "What is the capital of France? Explain briefly.",
        512: "Explain the difference between machine learning and deep learning in detail. "
             "Include examples of algorithms, use cases, and key differences in approach.",
        1024: "Write a comprehensive guide on how to build a REST API using Python and FastAPI. "
              "Include sections on: 1) Setting up the development environment, 2) Creating basic endpoints, "
              "3) Implementing authentication and authorization, 4) Database integration with SQLAlchemy, "
              "5) Error handling and validation, 6) Testing strategies, 7) Deployment considerations.",
        2048: "Provide a detailed analysis of modern software architecture patterns. "
              "Cover the following topics in depth: 1) Microservices architecture - benefits, challenges, "
              "and best practices, 2) Event-driven architecture - event sourcing, CQRS, and message queues, "
              "3) Serverless architecture - FaaS, BaaS, and deployment strategies, 4) Monolithic vs distributed "
              "systems - trade-offs and migration strategies, 5) API design patterns - REST, GraphQL, gRPC, "
              "6) Data consistency patterns - eventual consistency, distributed transactions, saga pattern, "
              "7) Scalability patterns - horizontal vs vertical scaling, load balancing, caching strategies, "
              "8) Security patterns - authentication, authorization, encryption, API security best practices."
    }
    
    # Find closest match
    closest = min(base_prompts.keys(), key=lambda x: abs(x - target_length))
    prompt = base_prompts[closest]
    
    # Adjust length if needed
    if len(prompt) < target_length:
        # Repeat to reach target length
        repeat_count = (target_length // len(prompt)) + 1
        prompt = (prompt + " ") * repeat_count
        prompt = prompt[:target_length]
    
    return prompt


def run_latency_test(
    engine: InferenceEngine,
    model_name: str,
    input_length: int,
    output_length: int,
    warmup_runs: int,
    test_runs: int
) -> LatencyResult:
    """Run latency test for specific input/output lengths."""
    logging.info(f"Testing input={input_length}, output={output_length}")
    
    benchmark = LatencyBenchmark(
        warmup_runs=warmup_runs,
        test_runs=test_runs,
    )
    
    prompt = create_prompt(input_length)
    
    def generate_fn(p: str):
        return engine.generate(p, config=GenerationConfig(max_new_tokens=output_length))
    
    def get_output_tokens(result) -> int:
        return result.output_tokens
    
    test_name = f"latency_in{input_length}_out{output_length}"
    
    result = benchmark.run_latency_suite(
        generate_fn=generate_fn,
        prompt=prompt,
        model_name=model_name,
        get_output_tokens=get_output_tokens,
        test_name=test_name,
    )
    
    # Update lengths
    result.input_length = input_length
    result.output_length = output_length
    
    logging.info(f"Results: TTFT={result.ttft_mean*1000:.2f}ms, TPS={result.tokens_per_second_mean:.2f}")
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Latency Benchmarks")
    parser.add_argument("--model", "-m", default="gpt2", help="Model name")
    parser.add_argument("--output-dir", "-o", default="data/benchmarks", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick benchmark (fewer iterations)")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("=" * 60)
    logging.info("Comprehensive Latency Benchmark")
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
    test_runs = 5 if args.quick else 10
    
    # Test configurations
    test_configs = [
        (128, 64),    # Short prompt, short output
        (512, 128),   # Medium prompt, medium output
        (1024, 256),  # Long prompt, long output
        (2048, 512),  # Very long prompt, long output
    ]
    
    if args.quick:
        # Skip the longest test in quick mode
        test_configs = test_configs[:3]
    
    # Run tests
    results = []
    for input_len, output_len in test_configs:
        try:
            result = run_latency_test(
                engine, args.model,
                input_len, output_len,
                warmup_runs, test_runs
            )
            results.append(result)
        except Exception as e:
            logging.error(f"Test failed for input={input_len}, output={output_len}: {e}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"latency_{args.model}_{timestamp}.json"
    
    report = {
        "report_name": f"latency_{args.model}_{timestamp}",
        "model_name": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": actual_device,
        "quick_mode": args.quick,
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
        "results": [r.to_dict() for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info("=" * 60)
    logging.info("Latency Benchmark Summary")
    logging.info("=" * 60)
    
    for result in results:
        logging.info(f"\nInput: {result.input_length} tokens, Output: {result.output_length} tokens")
        logging.info(f"  TTFT: {result.ttft_mean*1000:.2f} ms (±{result.ttft_std*1000:.2f})")
        logging.info(f"  Tokens/s: {result.tokens_per_second_mean:.2f} (±{result.tokens_per_second_std:.2f})")
        logging.info(f"  E2E Latency: {result.e2e_latency_mean*1000:.2f} ms (±{result.e2e_latency_std*1000:.2f})")
    
    logging.info(f"\nResults saved to: {output_file}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
