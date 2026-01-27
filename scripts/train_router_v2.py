"""
Training script for Router v2 classifier.

This script reads execution traces from JSONL files, extracts questions and
chosen workflows, trains a TF-IDF + LogisticRegression classifier, and saves
the trained models to disk.

Requirements: 15.2, 17.1, 17.2, 17.4

Usage:
    python scripts/train_router_v2.py --traces data/traces/workflow_traces.jsonl \
                                       --output-dir models/router_v2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.routing.router_v2 import RouterV2
from mm_orch.logger import get_logger


logger = get_logger(__name__)


def load_traces(trace_file: Path) -> List[dict]:
    """
    Load execution traces from JSONL file.
    
    Args:
        trace_file: Path to JSONL trace file
        
    Returns:
        List of trace dictionaries
        
    Raises:
        FileNotFoundError: If trace file doesn't exist
    """
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")
    
    traces = []
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                trace = json.loads(line)
                traces.append(trace)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse trace line",
                    line_num=line_num,
                    error=str(e)
                )
    
    logger.info(f"Loaded {len(traces)} traces from {trace_file}")
    return traces


def extract_training_data(traces: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract questions and workflow labels from traces.
    
    Args:
        traces: List of trace dictionaries
        
    Returns:
        Tuple of (questions, workflows) lists
    """
    questions = []
    workflows = []
    
    for trace in traces:
        # Extract question
        question = trace.get('question', '').strip()
        if not question:
            continue
        
        # Extract chosen workflow
        workflow = trace.get('chosen_workflow', '').strip()
        if not workflow:
            continue
        
        questions.append(question)
        workflows.append(workflow)
    
    logger.info(
        f"Extracted {len(questions)} training examples",
        unique_workflows=len(set(workflows))
    )
    
    return questions, workflows


def train_classifier(
    questions: List[str],
    workflows: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[TfidfVectorizer, LogisticRegression, dict]:
    """
    Train TF-IDF vectorizer and LogisticRegression classifier.
    
    Args:
        questions: List of question texts
        workflows: List of workflow labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (vectorizer, classifier, metrics)
    """
    logger.info("Training RouterV2 classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        questions, workflows,
        test_size=test_size,
        random_state=random_state,
        stratify=workflows
    )
    
    logger.info(
        f"Split data: {len(X_train)} train, {len(X_test)} test"
    )
    
    # Train TF-IDF vectorizer
    logger.info("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info(
        f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}"
    )
    
    # Train classifier
    logger.info("Training LogisticRegression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        multi_class='multinomial',
        solver='lbfgs'
    )
    classifier.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test accuracy: {accuracy:.3f}")
    
    # Generate classification report
    report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0
    )
    
    # Log per-class metrics
    for workflow, metrics in report.items():
        if isinstance(metrics, dict):
            logger.info(
                f"Workflow: {workflow}",
                precision=round(metrics.get('precision', 0), 3),
                recall=round(metrics.get('recall', 0), 3),
                f1=round(metrics.get('f1-score', 0), 3),
                support=metrics.get('support', 0)
            )
    
    metrics = {
        'accuracy': accuracy,
        'num_train': len(X_train),
        'num_test': len(X_test),
        'num_features': len(vectorizer.vocabulary_),
        'num_workflows': len(classifier.classes_),
        'workflows': list(classifier.classes_)
    }
    
    return vectorizer, classifier, metrics


def save_models(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    output_dir: Path,
    metrics: dict
) -> None:
    """
    Save trained models and metrics to disk.
    
    Args:
        vectorizer: Trained vectorizer
        classifier: Trained classifier
        output_dir: Directory to save models
        metrics: Training metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    vectorizer_path = output_dir / "vectorizer.pkl"
    classifier_path = output_dir / "classifier.pkl"
    
    RouterV2.save_models(
        vectorizer, classifier,
        str(vectorizer_path), str(classifier_path)
    )
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved models and metrics to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Router v2 classifier from execution traces"
    )
    parser.add_argument(
        '--traces',
        type=str,
        required=True,
        help='Path to JSONL trace file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/router_v2',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    trace_file = Path(args.traces)
    output_dir = Path(args.output_dir)
    
    try:
        # Load traces
        traces = load_traces(trace_file)
        
        if len(traces) == 0:
            logger.error("No traces loaded. Cannot train classifier.")
            sys.exit(1)
        
        # Extract training data
        questions, workflows = extract_training_data(traces)
        
        if len(questions) < 10:
            logger.error(
                f"Insufficient training data: {len(questions)} examples. "
                "Need at least 10 examples."
            )
            sys.exit(1)
        
        # Train classifier
        vectorizer, classifier, metrics = train_classifier(
            questions, workflows,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save models
        save_models(vectorizer, classifier, output_dir, metrics)
        
        logger.info("Training complete!")
        logger.info(f"Models saved to: {output_dir}")
        logger.info(f"Test accuracy: {metrics['accuracy']:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
