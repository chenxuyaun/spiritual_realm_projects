#!/usr/bin/env python3
"""
Export PyTorch models to OpenVINO IR format.

This script exports HuggingFace transformers models to OpenVINO Intermediate
Representation (IR) format for optimized inference on Intel hardware.

Usage:
    python scripts/export_to_openvino.py gpt2 --precision FP32
    python scripts/export_to_openvino.py distilgpt2 --precision FP16 --output models/openvino
    python scripts/export_to_openvino.py --all
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mm_orch.logger import get_logger

# Check OpenVINO availability
try:
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer, AutoConfig
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError as e:
    OPENVINO_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ModelExporter:
    """Export PyTorch models to OpenVINO format."""
    
    SUPPORTED_PRECISIONS = ["FP32", "FP16", "INT8"]
    DEFAULT_MODELS = ["gpt2", "distilgpt2"]
    
    def __init__(
        self,
        output_base_dir: str = "models/openvino",
        verbose: bool = False
    ):
        """
        Initialize model exporter.
        
        Args:
            output_base_dir: Base directory for exported models
            verbose: Enable verbose logging
        """
        self.output_base_dir = Path(output_base_dir)
        self.logger = get_logger()
        
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        if not OPENVINO_AVAILABLE:
            raise RuntimeError(
                f"OpenVINO not available: {IMPORT_ERROR}\n"
                "Install with: pip install optimum[openvino] openvino openvino-dev"
            )
    
    def export_model(
        self,
        model_name: str,
        precision: str = "FP32",
        model_path: Optional[str] = None,
        output_path: Optional[str] = None,
        validate: bool = True
    ) -> Path:
        """
        Export a model to OpenVINO format.
        
        Args:
            model_name: Name of the model (e.g., 'gpt2', 'distilgpt2')
            precision: Precision level ('FP32', 'FP16', 'INT8')
            model_path: Path to PyTorch model (defaults to HuggingFace hub)
            output_path: Custom output path (defaults to models/openvino/{model_name})
            validate: Validate export by loading the model
            
        Returns:
            Path to exported model directory
            
        Raises:
            ValueError: If precision is invalid
            RuntimeError: If export fails
        """
        # Validate precision
        if precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Invalid precision: {precision}. "
                f"Supported: {', '.join(self.SUPPORTED_PRECISIONS)}"
            )
        
        # Determine paths
        source_path = model_path or model_name
        if output_path:
            target_path = Path(output_path)
        else:
            target_path = self.output_base_dir / model_name
        
        # Create output directory
        target_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Exporting model to OpenVINO",
            context={
                "model": model_name,
                "source": source_path,
                "target": str(target_path),
                "precision": precision
            }
        )
        
        try:
            # Export model
            self._export_with_optimum(
                source_path,
                target_path,
                precision
            )
            
            # Validate export
            if validate:
                self._validate_export(target_path, model_name)
            
            self.logger.info(
                f"Model exported successfully",
                context={
                    "model": model_name,
                    "path": str(target_path),
                    "precision": precision
                }
            )
            
            return target_path
            
        except Exception as e:
            self.logger.error(
                f"Model export failed",
                context={
                    "model": model_name,
                    "error": str(e)
                }
            )
            raise RuntimeError(f"Export failed for {model_name}: {e}")
    
    def _export_with_optimum(
        self,
        source_path: str,
        target_path: Path,
        precision: str
    ) -> None:
        """
        Export model using Optimum Intel.
        
        Args:
            source_path: Source model path or HuggingFace model ID
            target_path: Target directory for exported model
            precision: Precision level
        """
        import time
        start_time = time.time()
        
        # Load and export model
        self.logger.info(f"Loading model from {source_path}...")
        
        # Configure export based on precision
        export_kwargs = {
            "export": True,
            "compile": False  # Don't compile yet, just export
        }
        
        # For INT8, we would need calibration data
        # For now, we'll export as FP32/FP16 and note INT8 requires additional steps
        if precision == "INT8":
            self.logger.warning(
                "INT8 quantization requires calibration data. "
                "Exporting as FP32 first. Use OpenVINO POT for INT8 quantization."
            )
        
        # Export model
        model = OVModelForCausalLM.from_pretrained(
            source_path,
            **export_kwargs
        )
        
        # Save exported model
        self.logger.info(f"Saving exported model to {target_path}...")
        model.save_pretrained(target_path)
        
        # Also save tokenizer
        self.logger.info("Saving tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(source_path)
        tokenizer.save_pretrained(target_path)
        
        # Save config
        config = AutoConfig.from_pretrained(source_path)
        config.save_pretrained(target_path)
        
        export_time = time.time() - start_time
        self.logger.info(
            f"Export completed in {export_time:.2f}s",
            context={"precision": precision}
        )
    
    def _validate_export(self, model_path: Path, model_name: str) -> None:
        """
        Validate exported model by loading it.
        
        Args:
            model_path: Path to exported model
            model_name: Model name for logging
            
        Raises:
            RuntimeError: If validation fails
        """
        self.logger.info(f"Validating exported model: {model_name}")
        
        try:
            # Check required files exist
            xml_file = model_path / "openvino_model.xml"
            bin_file = model_path / "openvino_model.bin"
            
            if not xml_file.exists():
                raise FileNotFoundError(f"Missing XML file: {xml_file}")
            if not bin_file.exists():
                raise FileNotFoundError(f"Missing BIN file: {bin_file}")
            
            # Try loading the model
            model = OVModelForCausalLM.from_pretrained(
                model_path,
                device="CPU",
                compile=True
            )
            
            # Try loading tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Quick inference test
            test_input = "Hello"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            self.logger.info(
                f"Validation successful",
                context={
                    "model": model_name,
                    "test_input": test_input,
                    "test_output": result
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Validation failed: {e}")
    
    def export_all(
        self,
        precision: str = "FP32",
        models: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Export multiple models.
        
        Args:
            precision: Precision level for all models
            models: List of model names (defaults to DEFAULT_MODELS)
            
        Returns:
            List of paths to exported models
        """
        models_to_export = models or self.DEFAULT_MODELS
        exported_paths = []
        failed_models = []
        
        self.logger.info(
            f"Exporting {len(models_to_export)} models",
            context={"models": models_to_export, "precision": precision}
        )
        
        for model_name in models_to_export:
            try:
                path = self.export_model(model_name, precision=precision)
                exported_paths.append(path)
            except Exception as e:
                self.logger.error(
                    f"Failed to export {model_name}: {e}"
                )
                failed_models.append(model_name)
        
        # Summary
        self.logger.info(
            f"Export complete",
            context={
                "successful": len(exported_paths),
                "failed": len(failed_models),
                "failed_models": failed_models
            }
        )
        
        return exported_paths
    
    def list_exported_models(self) -> List[str]:
        """
        List all exported models in the output directory.
        
        Returns:
            List of exported model names
        """
        if not self.output_base_dir.exists():
            return []
        
        exported = []
        for item in self.output_base_dir.iterdir():
            if item.is_dir():
                xml_file = item / "openvino_model.xml"
                if xml_file.exists():
                    exported.append(item.name)
        
        return exported


def main():
    """Main entry point for the export script."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to OpenVINO IR format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model with default precision (FP32)
  python scripts/export_to_openvino.py gpt2
  
  # Export with FP16 precision
  python scripts/export_to_openvino.py distilgpt2 --precision FP16
  
  # Export to custom directory
  python scripts/export_to_openvino.py gpt2 --output /path/to/models
  
  # Export all default models
  python scripts/export_to_openvino.py --all
  
  # Export specific models
  python scripts/export_to_openvino.py --all --models gpt2 distilgpt2 t5-small
  
  # List exported models
  python scripts/export_to_openvino.py --list
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Name of the model to export (e.g., 'gpt2', 'distilgpt2')"
    )
    
    parser.add_argument(
        "--precision",
        choices=ModelExporter.SUPPORTED_PRECISIONS,
        default="FP32",
        help="Precision level for export (default: FP32)"
    )
    
    parser.add_argument(
        "--output",
        help="Output directory (default: models/openvino)"
    )
    
    parser.add_argument(
        "--model-path",
        help="Path to source PyTorch model (default: HuggingFace hub)"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after export"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all default models"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of models to export (use with --all)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all exported models"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check OpenVINO availability
    if not OPENVINO_AVAILABLE:
        print(f"Error: OpenVINO not available: {IMPORT_ERROR}", file=sys.stderr)
        print("\nInstall with:", file=sys.stderr)
        print("  pip install optimum[openvino] openvino openvino-dev", file=sys.stderr)
        sys.exit(1)
    
    # Create exporter
    try:
        exporter = ModelExporter(
            output_base_dir=args.output or "models/openvino",
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error initializing exporter: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle list command
    if args.list:
        exported = exporter.list_exported_models()
        if exported:
            print("Exported models:")
            for model in exported:
                print(f"  - {model}")
        else:
            print("No exported models found")
        sys.exit(0)
    
    # Handle export commands
    try:
        if args.all:
            # Export multiple models
            exporter.export_all(
                precision=args.precision,
                models=args.models
            )
        elif args.model_name:
            # Export single model
            exporter.export_model(
                model_name=args.model_name,
                precision=args.precision,
                model_path=args.model_path,
                output_path=args.output,
                validate=not args.no_validate
            )
        else:
            parser.print_help()
            sys.exit(1)
        
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
