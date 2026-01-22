#!/usr/bin/env python
"""
GPT-2 End-to-End Test Script

This script runs a complete E2E test with the GPT-2 model to verify:
1. Model loading works correctly
2. Inference produces valid output
3. E2E validation passes

Usage:
    python scripts/run_gpt2_e2e_test.py
    python scripts/run_gpt2_e2e_test.py --verbose
    python scripts/run_gpt2_e2e_test.py --save-results
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
from mm_orch.runtime.real_model_manager import RealModelManager
from mm_orch.validation.e2e_validator import (
    E2EValidator,
    SearchQATestCase,
    ConversationTestCase,
    ValidationResult,
)


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


def load_gpt2_model(device: str = "auto") -> tuple:
    """
    Load GPT-2 model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer, actual_device)
    """
    logging.info("Loading GPT-2 model...")
    
    config = ModelConfig(
        model_name="gpt2",
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
    logging.info(f"Model dtype: {loaded.dtype}")
    logging.info(f"Memory footprint: {loaded.memory_footprint / 1024**2:.2f} MB")
    
    return loaded.model, loaded.tokenizer, loaded.device


def create_inference_engine(model, tokenizer, device: str) -> InferenceEngine:
    """Create inference engine with GPT-2 model."""
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    return InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        default_config=config,
    )


def run_basic_generation_test(engine: InferenceEngine) -> Dict[str, Any]:
    """Run basic generation test."""
    logging.info("Running basic generation test...")
    
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
    ]
    
    results = []
    for prompt in prompts:
        start = time.time()
        result = engine.generate(prompt)
        elapsed = time.time() - start
        
        results.append({
            "prompt": prompt,
            "response": result.text[:200],  # Truncate for display
            "tokens_per_second": result.tokens_per_second,
            "output_tokens": result.output_tokens,
            "elapsed_time": elapsed,
        })
        
        logging.info(f"Prompt: {prompt[:50]}...")
        logging.info(f"Response: {result.text[:100]}...")
        logging.info(f"Tokens/s: {result.tokens_per_second:.2f}")
    
    return {
        "test_name": "basic_generation",
        "passed": all(r["output_tokens"] > 0 for r in results),
        "results": results,
    }


def run_e2e_validation(engine: InferenceEngine) -> Dict[str, Any]:
    """Run E2E validation with test cases."""
    logging.info("Running E2E validation...")
    
    # Create generate function for validator
    def generate_fn(prompt: str) -> str:
        result = engine.generate(prompt)
        return result.text
    
    # Create validator
    validator = E2EValidator(generate_fn=generate_fn)
    
    # Define test cases
    search_qa_cases = [
        SearchQATestCase(
            id="e2e_sq_001",
            name="Basic factual question",
            query="What is the capital of France?",
            expected_keywords=["Paris"],
            max_response_time=30.0,
        ),
        SearchQATestCase(
            id="e2e_sq_002",
            name="Programming question",
            query="What is Python used for?",
            expected_keywords=["programming", "language"],
            max_response_time=30.0,
        ),
    ]
    
    conversation_cases = [
        ConversationTestCase(
            id="e2e_conv_001",
            name="Simple context test",
            turns=[
                {"role": "user", "content": "Hello, I am learning Python."},
                {"role": "user", "content": "What should I learn first?"},
            ],
            context_keywords=["Python"],
            max_turn_time=15.0,
        ),
    ]
    
    # Run validation
    results = validator.run_validation_suite(
        search_qa_cases=search_qa_cases,
        conversation_cases=conversation_cases,
    )
    
    summary = validator.get_summary()
    
    logging.info(f"E2E Validation Summary:")
    logging.info(f"  Total: {summary['total']}")
    logging.info(f"  Passed: {summary['passed']}")
    logging.info(f"  Failed: {summary['failed']}")
    logging.info(f"  Pass Rate: {summary['pass_rate']:.1f}%")
    
    return {
        "test_name": "e2e_validation",
        "passed": summary["pass_rate"] >= 50.0,  # At least 50% pass rate
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }


def run_streaming_test(engine: InferenceEngine) -> Dict[str, Any]:
    """Run streaming generation test."""
    logging.info("Running streaming generation test...")
    
    prompt = "Write a short poem about coding:"
    
    try:
        tokens = []
        start = time.time()
        
        for token in engine.generate_stream(prompt):
            tokens.append(token)
            if len(tokens) > 50:  # Limit for test
                break
        
        elapsed = time.time() - start
        full_response = "".join(tokens)
        
        logging.info(f"Streaming response: {full_response[:100]}...")
        logging.info(f"Tokens received: {len(tokens)}")
        logging.info(f"Time: {elapsed:.2f}s")
        
        return {
            "test_name": "streaming_generation",
            "passed": len(tokens) > 0,
            "tokens_received": len(tokens),
            "elapsed_time": elapsed,
            "response_preview": full_response[:200],
        }
    except Exception as e:
        logging.warning(f"Streaming test failed: {e}")
        return {
            "test_name": "streaming_generation",
            "passed": False,
            "error": str(e),
        }


def run_batch_test(engine: InferenceEngine) -> Dict[str, Any]:
    """Run batch generation test."""
    logging.info("Running batch generation test...")
    
    prompts = [
        "The weather today is",
        "My favorite food is",
        "The best programming language is",
    ]
    
    try:
        start = time.time()
        results = engine.batch_generate(prompts)
        elapsed = time.time() - start
        
        batch_results = []
        for i, result in enumerate(results):
            batch_results.append({
                "prompt": prompts[i],
                "response": result.text[:100],
                "tokens_per_second": result.tokens_per_second,
            })
            logging.info(f"Batch {i+1}: {result.text[:50]}...")
        
        return {
            "test_name": "batch_generation",
            "passed": len(results) == len(prompts),
            "batch_size": len(prompts),
            "elapsed_time": elapsed,
            "results": batch_results,
        }
    except Exception as e:
        logging.warning(f"Batch test failed: {e}")
        return {
            "test_name": "batch_generation",
            "passed": False,
            "error": str(e),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPT-2 E2E Test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--save-results", "-s", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", "-o", default="data/e2e_results", help="Output directory")
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("=" * 60)
    logging.info("GPT-2 End-to-End Test")
    logging.info("=" * 60)
    
    # Check device
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Load model
    try:
        model, tokenizer, actual_device = load_gpt2_model(device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create inference engine
    engine = create_inference_engine(model, tokenizer, actual_device)
    
    # Run tests
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": actual_device,
        "model": "gpt2",
        "tests": [],
    }
    
    # Test 1: Basic generation
    result = run_basic_generation_test(engine)
    all_results["tests"].append(result)
    
    # Test 2: E2E validation
    result = run_e2e_validation(engine)
    all_results["tests"].append(result)
    
    # Test 3: Streaming (optional, may not work on all setups)
    result = run_streaming_test(engine)
    all_results["tests"].append(result)
    
    # Test 4: Batch generation
    result = run_batch_test(engine)
    all_results["tests"].append(result)
    
    # Summary
    logging.info("=" * 60)
    logging.info("Test Summary")
    logging.info("=" * 60)
    
    passed = sum(1 for t in all_results["tests"] if t.get("passed", False))
    total = len(all_results["tests"])
    
    for test in all_results["tests"]:
        status = "✓ PASSED" if test.get("passed", False) else "✗ FAILED"
        logging.info(f"  {test['test_name']}: {status}")
    
    logging.info(f"\nOverall: {passed}/{total} tests passed")
    
    all_results["summary"] = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total * 100 if total > 0 else 0,
    }
    
    # Save results if requested
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"gpt2_e2e_results_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"\nResults saved to: {output_file}")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
