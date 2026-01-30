"""
OpenVINO Backend Performance Comparison Example

This script demonstrates how to compare performance between PyTorch and OpenVINO
backends using the built-in performance monitoring capabilities.

Requirements:
- OpenVINO installed: pip install openvino openvino-dev
- Model exported: python scripts/export_to_openvino.py gpt2
"""

from mm_orch.runtime.model_manager import ModelManager
import time
import statistics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_latency(backend_name, model_name="gpt2", num_runs=10, max_length=50):
    """
    Benchmark inference latency for a specific backend.
    
    Args:
        backend_name: 'pytorch' or 'openvino'
        model_name: Name of the model to benchmark
        num_runs: Number of inference runs
        max_length: Maximum generation length
        
    Returns:
        Dictionary with latency statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name.upper()} - Latency")
    print(f"{'='*60}")
    
    # Initialize manager
    manager = ModelManager(backend=backend_name)
    
    # Load model
    logger.info(f"Loading {model_name} with {backend_name} backend...")
    load_start = time.time()
    manager.load_model(model_name, "transformers")
    load_time = time.time() - load_start
    
    # Warmup
    logger.info("Warming up...")
    manager.generate(model_name, "warmup", max_length=10)
    
    # Benchmark
    logger.info(f"Running {num_runs} inference iterations...")
    latencies = []
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning enables",
        "Deep learning models",
        "Natural language processing",
        "Computer vision applications"
    ]
    
    for i in range(num_runs):
        prompt = test_prompts[i % len(test_prompts)]
        start = time.time()
        result = manager.generate(model_name, prompt, max_length=max_length)
        latency = time.time() - start
        latencies.append(latency)
        
        if i < 3:  # Show first few results
            print(f"  Run {i+1}: {latency:.3f}s - '{prompt[:30]}...'")
    
    # Calculate statistics
    stats = {
        "backend": backend_name,
        "model": model_name,
        "num_runs": num_runs,
        "load_time": load_time,
        "avg_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "stdev_latency": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
        "p99_latency": sorted(latencies)[int(0.99 * len(latencies))],
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Load time: {stats['load_time']:.3f}s")
    print(f"  Average latency: {stats['avg_latency']:.3f}s")
    print(f"  Median latency: {stats['median_latency']:.3f}s")
    print(f"  Min latency: {stats['min_latency']:.3f}s")
    print(f"  Max latency: {stats['max_latency']:.3f}s")
    print(f"  Std dev: {stats['stdev_latency']:.3f}s")
    print(f"  P95 latency: {stats['p95_latency']:.3f}s")
    print(f"  P99 latency: {stats['p99_latency']:.3f}s")
    
    return stats


def benchmark_throughput(backend_name, model_name="gpt2", duration_seconds=30, max_length=50):
    """
    Benchmark inference throughput for a specific backend.
    
    Args:
        backend_name: 'pytorch' or 'openvino'
        model_name: Name of the model to benchmark
        duration_seconds: How long to run the benchmark
        max_length: Maximum generation length
        
    Returns:
        Dictionary with throughput statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name.upper()} - Throughput")
    print(f"{'='*60}")
    
    # Initialize manager
    manager = ModelManager(backend=backend_name)
    
    # Load model
    logger.info(f"Loading {model_name} with {backend_name} backend...")
    manager.load_model(model_name, "transformers")
    
    # Warmup
    logger.info("Warming up...")
    manager.generate(model_name, "warmup", max_length=10)
    
    # Benchmark
    logger.info(f"Running throughput test for {duration_seconds} seconds...")
    start_time = time.time()
    count = 0
    total_tokens = 0
    
    test_prompts = [
        "The future of AI",
        "Machine learning",
        "Deep learning",
        "Natural language",
        "Computer vision"
    ]
    
    while time.time() - start_time < duration_seconds:
        prompt = test_prompts[count % len(test_prompts)]
        result = manager.generate(model_name, prompt, max_length=max_length)
        count += 1
        total_tokens += max_length  # Approximate
        
        if count % 5 == 0:
            elapsed = time.time() - start_time
            current_throughput = count / elapsed
            print(f"  {count} requests in {elapsed:.1f}s ({current_throughput:.2f} req/s)")
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    stats = {
        "backend": backend_name,
        "model": model_name,
        "duration": elapsed,
        "total_requests": count,
        "total_tokens": total_tokens,
        "requests_per_sec": count / elapsed,
        "tokens_per_sec": total_tokens / elapsed,
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Total time: {stats['duration']:.1f}s")
    print(f"  Requests/sec: {stats['requests_per_sec']:.2f}")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.1f}")
    
    return stats


def compare_backends(model_name="gpt2", num_runs=10, max_length=50):
    """
    Compare PyTorch and OpenVINO backends side-by-side.
    
    Args:
        model_name: Name of the model to benchmark
        num_runs: Number of inference runs
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Backend Comparison: PyTorch vs OpenVINO")
    print(f"{'='*60}")
    
    # Benchmark PyTorch
    pytorch_stats = benchmark_latency("pytorch", model_name, num_runs, max_length)
    
    # Benchmark OpenVINO
    openvino_stats = benchmark_latency("openvino", model_name, num_runs, max_length)
    
    # Calculate improvements
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    
    latency_speedup = pytorch_stats["avg_latency"] / openvino_stats["avg_latency"]
    load_speedup = pytorch_stats["load_time"] / openvino_stats["load_time"]
    
    print(f"\nModel: {model_name}")
    print(f"Runs: {num_runs}")
    print(f"Max length: {max_length}")
    
    print(f"\nLoad Time:")
    print(f"  PyTorch: {pytorch_stats['load_time']:.3f}s")
    print(f"  OpenVINO: {openvino_stats['load_time']:.3f}s")
    print(f"  Speedup: {load_speedup:.2f}x")
    
    print(f"\nAverage Latency:")
    print(f"  PyTorch: {pytorch_stats['avg_latency']:.3f}s")
    print(f"  OpenVINO: {openvino_stats['avg_latency']:.3f}s")
    print(f"  Speedup: {latency_speedup:.2f}x")
    
    print(f"\nMedian Latency:")
    print(f"  PyTorch: {pytorch_stats['median_latency']:.3f}s")
    print(f"  OpenVINO: {openvino_stats['median_latency']:.3f}s")
    
    print(f"\nP95 Latency:")
    print(f"  PyTorch: {pytorch_stats['p95_latency']:.3f}s")
    print(f"  OpenVINO: {openvino_stats['p95_latency']:.3f}s")
    
    # Determine winner
    if latency_speedup > 1.5:
        print(f"\n✓ OpenVINO is {latency_speedup:.2f}x faster than PyTorch")
    elif latency_speedup > 1.1:
        print(f"\n✓ OpenVINO is slightly faster ({latency_speedup:.2f}x)")
    else:
        print(f"\n⚠ Performance similar or PyTorch faster")


def compare_throughput(model_name="gpt2", duration_seconds=30, max_length=50):
    """
    Compare throughput between PyTorch and OpenVINO backends.
    
    Args:
        model_name: Name of the model to benchmark
        duration_seconds: How long to run each benchmark
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Throughput Comparison: PyTorch vs OpenVINO")
    print(f"{'='*60}")
    
    # Benchmark PyTorch
    pytorch_stats = benchmark_throughput("pytorch", model_name, duration_seconds, max_length)
    
    # Benchmark OpenVINO
    openvino_stats = benchmark_throughput("openvino", model_name, duration_seconds, max_length)
    
    # Calculate improvements
    print(f"\n{'='*60}")
    print("Throughput Comparison Summary")
    print(f"{'='*60}")
    
    throughput_improvement = openvino_stats["requests_per_sec"] / pytorch_stats["requests_per_sec"]
    tokens_improvement = openvino_stats["tokens_per_sec"] / pytorch_stats["tokens_per_sec"]
    
    print(f"\nModel: {model_name}")
    print(f"Duration: {duration_seconds}s")
    print(f"Max length: {max_length}")
    
    print(f"\nRequests per Second:")
    print(f"  PyTorch: {pytorch_stats['requests_per_sec']:.2f}")
    print(f"  OpenVINO: {openvino_stats['requests_per_sec']:.2f}")
    print(f"  Improvement: {throughput_improvement:.2f}x")
    
    print(f"\nTokens per Second:")
    print(f"  PyTorch: {pytorch_stats['tokens_per_sec']:.1f}")
    print(f"  OpenVINO: {openvino_stats['tokens_per_sec']:.1f}")
    print(f"  Improvement: {tokens_improvement:.2f}x")
    
    # Determine winner
    if throughput_improvement > 1.5:
        print(f"\n✓ OpenVINO has {throughput_improvement:.2f}x higher throughput")
    elif throughput_improvement > 1.1:
        print(f"\n✓ OpenVINO has slightly higher throughput ({throughput_improvement:.2f}x)")
    else:
        print(f"\n⚠ Throughput similar or PyTorch higher")


def benchmark_memory_usage(backend_name, model_name="gpt2"):
    """
    Benchmark memory usage for a specific backend.
    
    Args:
        backend_name: 'pytorch' or 'openvino'
        model_name: Name of the model to benchmark
        
    Returns:
        Dictionary with memory statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking {backend_name.upper()} - Memory Usage")
    print(f"{'='*60}")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory: {baseline:.1f} MB")
        
        # Load model
        logger.info(f"Loading {model_name} with {backend_name} backend...")
        manager = ModelManager(backend=backend_name)
        manager.load_model(model_name, "transformers")
        
        loaded = process.memory_info().rss / 1024 / 1024  # MB
        model_size = loaded - baseline
        print(f"Memory after loading: {loaded:.1f} MB")
        print(f"Model size: {model_size:.1f} MB")
        
        # Run inference
        logger.info("Running inference...")
        manager.generate(model_name, "Test prompt", max_length=50)
        
        inference = process.memory_info().rss / 1024 / 1024  # MB
        inference_overhead = inference - loaded
        print(f"Memory after inference: {inference:.1f} MB")
        print(f"Inference overhead: {inference_overhead:.1f} MB")
        
        return {
            "backend": backend_name,
            "baseline_mb": baseline,
            "model_size_mb": model_size,
            "inference_overhead_mb": inference_overhead,
            "total_mb": inference
        }
        
    except ImportError:
        print("psutil not installed. Install with: pip install psutil")
        return None


def main():
    """Run performance comparison examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenVINO Performance Comparison")
    parser.add_argument("--model", default="gpt2", help="Model name to benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of latency benchmark runs")
    parser.add_argument("--duration", type=int, default=30, help="Throughput benchmark duration (seconds)")
    parser.add_argument("--max-length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--test", choices=["latency", "throughput", "memory", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("OpenVINO Backend Performance Comparison")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")
    print(f"Duration: {args.duration}s")
    print(f"Max length: {args.max_length}")
    
    try:
        if args.test in ["latency", "all"]:
            compare_backends(args.model, args.runs, args.max_length)
        
        if args.test in ["throughput", "all"]:
            compare_throughput(args.model, args.duration, args.max_length)
        
        if args.test in ["memory", "all"]:
            pytorch_mem = benchmark_memory_usage("pytorch", args.model)
            openvino_mem = benchmark_memory_usage("openvino", args.model)
            
            if pytorch_mem and openvino_mem:
                print(f"\n{'='*60}")
                print("Memory Comparison Summary")
                print(f"{'='*60}")
                print(f"\nModel Size:")
                print(f"  PyTorch: {pytorch_mem['model_size_mb']:.1f} MB")
                print(f"  OpenVINO: {openvino_mem['model_size_mb']:.1f} MB")
                reduction = (1 - openvino_mem['model_size_mb'] / pytorch_mem['model_size_mb']) * 100
                print(f"  Reduction: {reduction:.1f}%")
        
        print(f"\n{'='*60}")
        print("Performance comparison completed!")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
