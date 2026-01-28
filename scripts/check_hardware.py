#!/usr/bin/env python3
"""
Hardware Detection Script
Checks for available hardware acceleration options including GPU, NPU, etc.
"""

import platform
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_system_info():
    """Check basic system information."""
    print("=" * 80)
    print("System Information")
    print("=" * 80)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {sys.version}")
    print()


def check_cpu():
    """Check CPU information."""
    try:
        import psutil
        print("=" * 80)
        print("CPU Information")
        print("=" * 80)
        print(f"Physical cores: {psutil.cpu_count(logical=False)}")
        print(f"Logical cores: {psutil.cpu_count(logical=True)}")
        print(f"Total RAM: {round(psutil.virtual_memory().total / (1024**3), 2)} GB")
        print(f"Available RAM: {round(psutil.virtual_memory().available / (1024**3), 2)} GB")
        print()
    except ImportError:
        print("psutil not installed, skipping CPU details")
        print()


def check_pytorch():
    """Check PyTorch and available backends."""
    try:
        import torch
        print("=" * 80)
        print("PyTorch Information")
        print("=" * 80)
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch built with CUDA: {torch.version.cuda is not None}")
        print()
        
        # CUDA
        print("CUDA Support:")
        print(f"  Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("  No CUDA devices found")
        print()
        
        # MPS (Apple Silicon)
        print("MPS (Apple Silicon) Support:")
        if hasattr(torch.backends, 'mps'):
            print(f"  Available: {torch.backends.mps.is_available()}")
            if torch.backends.mps.is_available():
                print(f"  Built: {torch.backends.mps.is_built()}")
        else:
            print("  Not available (not Apple Silicon)")
        print()
        
        # DirectML (Windows)
        print("DirectML (Windows) Support:")
        if hasattr(torch, 'dml'):
            print(f"  Available: True")
            print(f"  Note: DirectML support detected")
        else:
            print("  Not available in current PyTorch build")
            print("  Note: Install torch-directml for Windows GPU acceleration")
        print()
        
        # ROCm (AMD)
        print("ROCm (AMD) Support:")
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"  Available: True")
            print(f"  Version: {torch.version.hip}")
        else:
            print("  Not available")
        print()
        
    except ImportError:
        print("PyTorch not installed")
        print()


def check_intel_npu():
    """Check for Intel NPU (Neural Processing Unit)."""
    print("=" * 80)
    print("Intel NPU Detection")
    print("=" * 80)
    
    # Check for Intel OpenVINO
    try:
        import openvino
        print(f"OpenVINO installed: {openvino.__version__}")
        print("  Note: OpenVINO can utilize Intel NPU if available")
        
        # Try to detect NPU device
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            print(f"  Available devices: {devices}")
            
            if 'NPU' in devices:
                print("  ✅ Intel NPU detected!")
            else:
                print("  ⚠️ No NPU device found in OpenVINO")
        except Exception as e:
            print(f"  Could not query devices: {e}")
    except ImportError:
        print("OpenVINO not installed")
        print("  To use Intel NPU, install: pip install openvino")
    print()


def check_onnxruntime():
    """Check ONNX Runtime and available execution providers."""
    print("=" * 80)
    print("ONNX Runtime")
    print("=" * 80)
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("  ✅ CUDA support available")
        if 'DmlExecutionProvider' in providers:
            print("  ✅ DirectML support available (Windows GPU)")
        if 'OpenVINOExecutionProvider' in providers:
            print("  ✅ OpenVINO support available (Intel NPU/GPU)")
        if 'TensorrtExecutionProvider' in providers:
            print("  ✅ TensorRT support available")
            
    except ImportError:
        print("ONNX Runtime not installed")
        print("  Install with: pip install onnxruntime")
        print("  For GPU: pip install onnxruntime-gpu")
        print("  For DirectML: pip install onnxruntime-directml")
    print()


def check_transformers():
    """Check transformers library."""
    print("=" * 80)
    print("Transformers Library")
    print("=" * 80)
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Check for optimum
        try:
            import optimum
            try:
                print(f"Optimum version: {optimum.__version__}")
            except AttributeError:
                print("Optimum installed (version info not available)")
            print("  Note: Optimum provides hardware-specific optimizations")
        except ImportError:
            print("Optimum not installed")
            print("  Install with: pip install optimum")
            
    except ImportError:
        print("Transformers not installed")
    print()


def provide_recommendations():
    """Provide recommendations based on detected hardware."""
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    
    import torch
    
    if torch.cuda.is_available():
        print("✅ CUDA GPU detected - You have the best option!")
        print("   Current PyTorch: CPU-only")
        print("   Recommendation: Install CUDA-enabled PyTorch")
        print("   Command: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
    else:
        print("⚠️ No CUDA GPU detected")
        print()
        
        # Check for Intel processor
        if 'Intel' in platform.processor():
            print("💡 Intel Processor Detected - Consider these options:")
            print()
            print("1. Intel OpenVINO (Best for Intel NPU/iGPU)")
            print("   - Optimized for Intel hardware")
            print("   - Supports NPU if available (Core Ultra processors)")
            print("   - Install: pip install openvino openvino-dev")
            print("   - 2-3x speedup possible")
            print()
            print("2. ONNX Runtime with OpenVINO")
            print("   - Install: pip install onnxruntime")
            print("   - Use OpenVINOExecutionProvider")
            print()
            print("3. Intel Extension for PyTorch")
            print("   - Install: pip install intel-extension-for-pytorch")
            print("   - Optimizes PyTorch for Intel CPUs")
            print()
        
        # Check for Windows
        if platform.system() == 'Windows':
            print("💡 Windows Detected - Consider DirectML:")
            print("   - DirectML uses Windows GPU acceleration")
            print("   - Works with Intel/AMD/NVIDIA GPUs")
            print("   - Install: pip install torch-directml")
            print("   - Or: pip install onnxruntime-directml")
            print()
        
        print("4. CPU Optimizations (Current setup)")
        print("   - Use model quantization (INT8/INT4)")
        print("   - Enable multi-threading")
        print("   - Use smaller models")
        print("   - Current performance: ~1.4 req/s")
        print()
        
        print("5. Cloud GPU (Recommended for production)")
        print("   - Google Colab (Free T4 GPU)")
        print("   - AWS EC2 (g4dn instances)")
        print("   - Azure (NC series)")
        print("   - Expected: 7-14x speedup")
        print()


def main():
    """Main execution."""
    check_system_info()
    check_cpu()
    check_pytorch()
    check_intel_npu()
    check_onnxruntime()
    check_transformers()
    provide_recommendations()
    
    print("=" * 80)
    print("Hardware Check Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
