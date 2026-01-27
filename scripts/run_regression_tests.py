#!/usr/bin/env python3
"""Regression test harness for workflow validation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.orchestration.graph_executor import GraphExecutor
from mm_orch.orchestration.state import State
from mm_orch.registries.workflow_registry import WorkflowRegistry
from mm_orch.registries.workflow_definitions import register_all_workflows
from mm_orch.runtime.model_manager import ModelManager
from mm_orch.observability.tracer import Tracer
from mm_orch.observability.quality_signals import QualitySignals
from mm_orch.observability.cost_stats import CostStatsManager


class RegressionTestHarness:
    """Harness for running regression tests on workflows."""
    
    def __init__(self, dataset_path: str, output_dir: str = "data/regression_results"):
        """
        Initialize regression test harness.
        
        Args:
            dataset_path: Path to JSONL test dataset
            output_dir: Directory for test results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.workflow_registry = WorkflowRegistry()
        register_all_workflows(self.workflow_registry)
        
        self.model_manager = ModelManager()
        self.tracer = Tracer(str(self.output_dir / "test_traces.jsonl"))
        self.cost_stats = CostStatsManager(str(self.output_dir / "test_costs.json"))
        
        self.results: List[Dict[str, Any]] = []
    
    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from JSONL file."""
        test_cases = []
        
        if not self.dataset_path.exists():
            print(f"Error: Dataset file not found: {self.dataset_path}")
            return test_cases
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    test_cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse test case: {e}")
        
        return test_cases
    
    def run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_case: Test case dictionary with question, expected_workflow, etc.
            
        Returns:
            Test result dictionary
        """
        question = test_case.get("question", "")
        expected_workflow = test_case.get("expected_workflow")
        expected_fields = test_case.get("expected_fields", [])
        quality_thresholds = test_case.get("quality_thresholds", {})
        
        result = {
            "question": question,
            "expected_workflow": expected_workflow,
            "passed": False,
            "errors": []
        }
        
        try:
            # Get workflow definition
            workflow_name = expected_workflow or "search_qa"
            workflow_def = self.workflow_registry.get(workflow_name)
            
            # Create initial state
            initial_state: State = {
                "question": question,
                "meta": {"mode": "default"}
            }
            
            # Execute workflow
            executor = GraphExecutor(self.workflow_registry, self.tracer)
            runtime = {"model_manager": self.model_manager}
            
            final_state = executor.execute(
                workflow_def.graph,
                initial_state,
                runtime
            )
            
            # Validate expected fields
            for field in expected_fields:
                if field not in final_state or not final_state[field]:
                    result["errors"].append(f"Missing or empty field: {field}")
            
            # Validate quality thresholds
            # Note: In real implementation, would calculate quality signals
            # For now, just check if we got an answer
            if "final_answer" in expected_fields:
                answer_length = len(final_state.get("final_answer", ""))
                min_length = quality_thresholds.get("min_answer_length", 0)
                if answer_length < min_length:
                    result["errors"].append(
                        f"Answer too short: {answer_length} < {min_length}"
                    )
            
            result["passed"] = len(result["errors"]) == 0
            result["final_state"] = {
                k: v for k, v in final_state.items()
                if k in ["question", "final_answer", "citations"]
            }
            
        except Exception as e:
            result["errors"].append(f"Execution failed: {str(e)}")
            result["exception"] = str(e)
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all test cases and generate summary.
        
        Returns:
            Summary dictionary with results
        """
        test_cases = self.load_test_cases()
        
        if not test_cases:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "results": []
            }
        
        print(f"Running {len(test_cases)} regression tests...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"  Test {i}/{len(test_cases)}: {test_case.get('question', '')[:50]}...")
            result = self.run_test_case(test_case)
            self.results.append(result)
        
        # Calculate summary
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        
        summary = {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0.0,
            "timestamp": datetime.now().isoformat(),
            "results": self.results
        }
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any], output_file: str = None):
        """
        Generate and save test report.
        
        Args:
            summary: Test summary dictionary
            output_file: Optional output file path
        """
        if output_file is None:
            output_file = str(self.output_dir / "regression_report.json")
        
        # Save detailed report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary to console
        print("\n" + "="*60)
        print("REGRESSION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Timestamp: {summary['timestamp']}")
        
        # Show failed tests
        if summary['failed'] > 0:
            print("\nFailed Tests:")
            for result in summary['results']:
                if not result['passed']:
                    print(f"  - {result['question'][:60]}...")
                    for error in result['errors']:
                        print(f"    Error: {error}")
        
        print(f"\nDetailed report saved to: {output_file}")
        print("="*60)


def main():
    """Main entry point for regression test harness."""
    parser = argparse.ArgumentParser(
        description="Run regression tests on workflows"
    )
    parser.add_argument(
        "--dataset",
        default="tests/fixtures/regression_cases.jsonl",
        help="Path to test dataset JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        default="data/regression_results",
        help="Directory for test results"
    )
    parser.add_argument(
        "--report",
        help="Path to output report file (default: output_dir/regression_report.json)"
    )
    
    args = parser.parse_args()
    
    # Run tests
    harness = RegressionTestHarness(args.dataset, args.output_dir)
    summary = harness.run_all_tests()
    harness.generate_report(summary, args.report)
    
    # Exit with error code if tests failed
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == "__main__":
    main()
