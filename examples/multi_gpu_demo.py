"""
Multi-GPU Support Demo

This example demonstrates the multi-GPU support features including:
- GPU detection and allocation
- Tensor parallelism with vLLM
- Pipeline parallelism with DeepSpeed
- GPU health monitoring
- Per-GPU metrics collection

Requirements: CUDA-capable GPUs, vLLM or DeepSpeed installed
"""

import time
from mm_orch.optimization import (
    VLLMEngine,
    VLLMConfig,
    DeepSpeedEngine,
    DeepSpeedConfig,
    get_gpu_manager,
    create_gpu_monitor,
)
from mm_orch.monitoring import PrometheusExporter
from mm_orch.logger import get_logger

logger = get_logger(__name__)


def demo_gpu_detection():
    """Demonstrate GPU detection and information gathering."""
    print("\n" + "=" * 60)
    print("GPU Detection Demo")
    print("=" * 60)
    
    manager = get_gpu_manager()
    gpus = manager.detect_gpus()
    
    print(f"\nDetected {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"\nGPU {gpu.device_id}:")
        print(f"  Name: {gpu.name}")
        print(f"  Total Memory: {gpu.total_memory_mb:.2f} MB")
        print(f"  Available Memory: {gpu.available_memory_mb:.2f} MB")
        print(f"  Utilization: {gpu.utilization_percent:.1f}%")
        if gpu.temperature_celsius:
            print(f"  Temperature: {gpu.temperature_celsius}°C")
        print(f"  Health: {'Healthy' if gpu.is_healthy else 'Unhealthy'}")
        if gpu.error_message:
            print(f"  Error: {gpu.error_message}")


def demo_tensor_parallelism():
    """Demonstrate tensor parallelism with vLLM."""
    print("\n" + "=" * 60)
    print("Tensor Parallelism Demo (vLLM)")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    # Check if we have enough GPUs
    gpus = manager.detect_gpus()
    if len(gpus) < 2:
        print("\nSkipping: Need at least 2 GPUs for tensor parallelism demo")
        return
    
    # Configure vLLM with tensor parallelism
    config = VLLMConfig(
        enabled=True,
        tensor_parallel_size=2,
        dtype="auto"
    )
    
    engine = VLLMEngine(config)
    
    if not engine.is_available():
        print("\nSkipping: vLLM not available")
        return
    
    print("\nConfigured vLLM with tensor parallelism (TP=2)")
    
    # Note: Actual model loading would require a real model
    # This is just a demonstration of the configuration
    print("Configuration:")
    print(f"  Tensor Parallel Size: {config.tensor_parallel_size}")
    print(f"  Data Type: {config.dtype}")
    print(f"  GPU Memory Utilization: {config.gpu_memory_utilization}")
    
    # Show GPU allocation (would happen during load_model)
    try:
        gpu_ids, strategy = manager.allocate_gpus(tensor_parallel=2)
        print(f"\nGPU Allocation:")
        print(f"  Strategy: {strategy}")
        print(f"  Allocated GPUs: {gpu_ids}")
    except RuntimeError as e:
        print(f"\nGPU allocation failed: {e}")


def demo_pipeline_parallelism():
    """Demonstrate pipeline parallelism with DeepSpeed."""
    print("\n" + "=" * 60)
    print("Pipeline Parallelism Demo (DeepSpeed)")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    # Check if we have enough GPUs
    gpus = manager.detect_gpus()
    if len(gpus) < 2:
        print("\nSkipping: Need at least 2 GPUs for pipeline parallelism demo")
        return
    
    # Configure DeepSpeed with pipeline parallelism
    config = DeepSpeedConfig(
        enabled=True,
        tensor_parallel=1,
        pipeline_parallel=2,
        dtype="fp16"
    )
    
    engine = DeepSpeedEngine(config)
    
    if not engine.is_available():
        print("\nSkipping: DeepSpeed not available")
        return
    
    print("\nConfigured DeepSpeed with pipeline parallelism (PP=2)")
    
    print("Configuration:")
    print(f"  Tensor Parallel: {config.tensor_parallel}")
    print(f"  Pipeline Parallel: {config.pipeline_parallel}")
    print(f"  Data Type: {config.dtype}")
    
    # Show GPU allocation and load balancing
    try:
        gpu_ids, strategy = manager.allocate_gpus(
            tensor_parallel=1,
            pipeline_parallel=2
        )
        print(f"\nGPU Allocation:")
        print(f"  Strategy: {strategy}")
        print(f"  Allocated GPUs: {gpu_ids}")
        
        # Show pipeline allocation
        allocation = manager.balance_load(
            tensor_parallel=1,
            pipeline_parallel=2
        )
        print(f"\nPipeline Allocation:")
        for stage_idx, stage_gpus in enumerate(allocation):
            print(f"  Stage {stage_idx}: GPUs {stage_gpus}")
    except RuntimeError as e:
        print(f"\nGPU allocation failed: {e}")


def demo_hybrid_parallelism():
    """Demonstrate hybrid parallelism (tensor + pipeline)."""
    print("\n" + "=" * 60)
    print("Hybrid Parallelism Demo (DeepSpeed)")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    # Check if we have enough GPUs
    gpus = manager.detect_gpus()
    if len(gpus) < 4:
        print("\nSkipping: Need at least 4 GPUs for hybrid parallelism demo")
        return
    
    # Configure DeepSpeed with hybrid parallelism
    config = DeepSpeedConfig(
        enabled=True,
        tensor_parallel=2,
        pipeline_parallel=2,
        dtype="fp16"
    )
    
    print("\nConfigured DeepSpeed with hybrid parallelism (TP=2, PP=2)")
    
    print("Configuration:")
    print(f"  Tensor Parallel: {config.tensor_parallel}")
    print(f"  Pipeline Parallel: {config.pipeline_parallel}")
    print(f"  Total GPUs Needed: {config.tensor_parallel * config.pipeline_parallel}")
    
    # Show GPU allocation and load balancing
    try:
        gpu_ids, strategy = manager.allocate_gpus(
            tensor_parallel=2,
            pipeline_parallel=2
        )
        print(f"\nGPU Allocation:")
        print(f"  Strategy: {strategy}")
        print(f"  Allocated GPUs: {gpu_ids}")
        
        # Show pipeline allocation with tensor parallelism
        allocation = manager.balance_load(
            tensor_parallel=2,
            pipeline_parallel=2
        )
        print(f"\nHybrid Allocation:")
        for stage_idx, stage_gpus in enumerate(allocation):
            print(f"  Pipeline Stage {stage_idx}: GPUs {stage_gpus} (Tensor Parallel)")
    except RuntimeError as e:
        print(f"\nGPU allocation failed: {e}")


def demo_gpu_health_monitoring():
    """Demonstrate GPU health monitoring."""
    print("\n" + "=" * 60)
    print("GPU Health Monitoring Demo")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    print("\nMonitoring GPU health...")
    health_status = manager.monitor_gpu_health()
    
    print(f"\nHealth Status for {len(health_status)} GPU(s):")
    for gpu_id, (is_healthy, error) in health_status.items():
        status = "Healthy" if is_healthy else "Unhealthy"
        print(f"  GPU {gpu_id}: {status}")
        if error:
            print(f"    Error: {error}")


def demo_gpu_failure_recovery():
    """Demonstrate GPU failure detection and recovery."""
    print("\n" + "=" * 60)
    print("GPU Failure Recovery Demo")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    # Simulate a scenario where we need recovery
    print("\nSimulating GPU failure scenario...")
    print("Original allocation: 2 GPUs for tensor parallelism")
    
    # Simulate GPU 1 failure
    simulated_failed_gpu = 1
    print(f"\nSimulating failure of GPU {simulated_failed_gpu}")
    
    # Attempt recovery
    try:
        new_allocation, strategy = manager.recover_from_failure(
            failed_gpu_id=simulated_failed_gpu,
            tensor_parallel=2,
            pipeline_parallel=1
        )
        
        if new_allocation:
            print(f"\nRecovery successful!")
            print(f"  Strategy: {strategy}")
            print(f"  New GPU allocation: {new_allocation}")
        else:
            print(f"\nRecovery strategy: {strategy}")
            print("  Falling back to CPU mode")
    except Exception as e:
        print(f"\nRecovery failed: {e}")
    
    # Reset failed GPU tracking
    manager.reset_failed_gpus()
    print("\nReset failed GPU tracking")


def demo_per_gpu_metrics():
    """Demonstrate per-GPU metrics collection and Prometheus export."""
    print("\n" + "=" * 60)
    print("Per-GPU Metrics Demo")
    print("=" * 60)
    
    manager = get_gpu_manager()
    
    # Get metrics for all GPUs
    all_metrics = manager.get_all_gpu_metrics()
    
    print(f"\nCollected metrics for {len(all_metrics)} GPU(s):")
    for gpu_id, metrics in all_metrics.items():
        print(f"\nGPU {gpu_id}:")
        print(f"  Memory Used: {metrics['memory_used_mb']:.2f} MB")
        print(f"  Memory Available: {metrics['memory_available_mb']:.2f} MB")
        print(f"  Memory Total: {metrics['memory_total_mb']:.2f} MB")
        print(f"  Utilization: {metrics['utilization_percent']:.1f}%")
        if 'temperature_celsius' in metrics:
            print(f"  Temperature: {metrics['temperature_celsius']}°C")
    
    # Demonstrate Prometheus integration
    print("\n" + "-" * 60)
    print("Prometheus Integration")
    print("-" * 60)
    
    try:
        exporter = PrometheusExporter(port=9091, enabled=True)
        
        if exporter.is_enabled():
            print("\nRecording metrics to Prometheus...")
            
            # Record metrics for all GPUs
            for gpu_id, metrics in all_metrics.items():
                exporter.record_per_gpu_metrics(
                    gpu_id=gpu_id,
                    memory_used_mb=metrics['memory_used_mb'],
                    memory_available_mb=metrics['memory_available_mb'],
                    utilization_percent=metrics['utilization_percent'],
                    temperature_celsius=metrics.get('temperature_celsius'),
                    is_healthy=True
                )
            
            print("Metrics recorded successfully!")
            print("\nPrometheus metrics available at: http://localhost:9091/metrics")
            print("\nExample metrics:")
            print("  gpu_memory_used_bytes{gpu_id=\"0\"}")
            print("  gpu_memory_available_bytes{gpu_id=\"0\"}")
            print("  gpu_utilization_percent{gpu_id=\"0\"}")
            print("  gpu_temperature_celsius{gpu_id=\"0\"}")
            print("  gpu_health_status{gpu_id=\"0\"}")
        else:
            print("\nPrometheus export disabled")
    except Exception as e:
        print(f"\nPrometheus integration error: {e}")


def demo_continuous_monitoring():
    """Demonstrate continuous GPU monitoring with Prometheus."""
    print("\n" + "=" * 60)
    print("Continuous GPU Monitoring Demo")
    print("=" * 60)
    
    try:
        # Create Prometheus exporter
        exporter = PrometheusExporter(port=9092, enabled=True)
        
        if not exporter.is_enabled():
            print("\nSkipping: Prometheus export disabled")
            return
        
        # Start Prometheus server
        exporter.start_server()
        print("\nPrometheus server started on port 9092")
        
        # Create GPU monitor
        monitor = create_gpu_monitor(
            prometheus_exporter=exporter,
            monitoring_interval=2.0,  # Check every 2 seconds
            auto_start=True
        )
        
        print("GPU monitoring started (interval: 2 seconds)")
        print("\nMonitoring for 10 seconds...")
        print("Metrics available at: http://localhost:9092/metrics")
        
        # Monitor for 10 seconds
        time.sleep(10)
        
        # Stop monitoring
        monitor.stop()
        print("\nGPU monitoring stopped")
        
    except Exception as e:
        print(f"\nContinuous monitoring error: {e}")


def main():
    """Run all multi-GPU demos."""
    print("\n" + "=" * 60)
    print("Multi-GPU Support Demo")
    print("=" * 60)
    print("\nThis demo showcases multi-GPU support features:")
    print("- GPU detection and information")
    print("- Tensor parallelism (vLLM)")
    print("- Pipeline parallelism (DeepSpeed)")
    print("- Hybrid parallelism (DeepSpeed)")
    print("- GPU health monitoring")
    print("- GPU failure recovery")
    print("- Per-GPU metrics collection")
    print("- Continuous monitoring with Prometheus")
    
    try:
        # Run demos
        demo_gpu_detection()
        demo_tensor_parallelism()
        demo_pipeline_parallelism()
        demo_hybrid_parallelism()
        demo_gpu_health_monitoring()
        demo_gpu_failure_recovery()
        demo_per_gpu_metrics()
        demo_continuous_monitoring()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
