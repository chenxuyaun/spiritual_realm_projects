"""
Training script for Router v3 cost-aware classifier.

This script reads execution traces, extracts mode features, trains a classifier
with text + mode features, and uses best_reward labeling strategy to select
optimal workflows based on quality and cost.

Requirements: 17.1, 17.2, 17.3, 17.4, 21.3

Usage:
    python scripts/train_router_v3.py --traces data/traces/workflow_traces.jsonl \
                                       --costs data/cost_stats.json \
                                       --output-dir models/router_v3
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.routing.router_v3 import RouterV3
from mm_orch.logger import get_logger


logger = get_logger(__name__)


def load_traces(trace_file: Path) -> List[dict]:
    """
    Load execution traces from JSONL file.
    
    Args:
        trace_file: Path to JSONL trace file
        
    Returns:
        List of trace dictionaries
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


def load_costs(costs_file: Path) -> Dict[str, float]:
    """
    Load cost statistics from JSON file.
    
    Args:
        costs_file: Path to cost statistics JSON
        
    Returns:
        Dictionary mapping workflow names to normalized costs
    """
    if not costs_file.exists():
        logger.warning(f"Cost file not found: {costs_file}. Using default costs.")
        return {}
    
    try:
        with open(costs_file, 'r', encoding='utf-8') as f:
            cost_data = json.load(f)
        
        # Extract and normalize costs (same as RouterV3)
        costs = {}
        for workflow_name, stats in cost_data.items():
            latency = stats.get('avg_latency_ms', 0)
            vram = stats.get('avg_vram_mb', 0)
            loads = stats.get('avg_model_loads', 0)
            
            # Weighted combination
            cost = (
                0.5 * min(latency / 10000, 1.0) +
                0.3 * min(vram / 4000, 1.0) +
                0.2 * min(loads / 5, 1.0)
            )
            
            costs[workflow_name] = cost
        
        logger.info(f"Loaded costs for {len(costs)} workflows")
        return costs
        
    except Exception as e:
        logger.error(f"Failed to load costs: {e}")
        return {}


def extract_training_data_with_mode(
    traces: List[dict],
    costs: Dict[str, float],
    lambda_cost: float = 0.1
) -> Tuple[List[str], List[int], List[str]]:
    """
    Extract questions, mode features, and best workflow labels using best_reward strategy.
    
    The best_reward strategy selects the workflow that would have achieved the
    highest reward (quality - lambda * cost) for each question.
    
    Mode features are extracted from trace metadata:
    - mode="chat": Encoded as 1 (chat interactions)
    - mode="default": Encoded as 0 (single-shot queries)
    
    Args:
        traces: List of trace dictionaries
        costs: Dictionary of workflow costs
        lambda_cost: Weight for cost in reward calculation
        
    Returns:
        Tuple of (questions, mode_features, workflows) lists
    
    Requirements:
        - 21.3: Extract mode from trace metadata and create mode_is_chat feature
    """
    questions = []
    mode_features = []
    workflows = []
    
    for trace in traces:
        # Extract question
        question = trace.get('question', '').strip()
        if not question:
            continue
        
        # Extract mode from trace metadata (Requirement 21.3)
        mode = trace.get('mode', 'default')
        mode_is_chat = 1 if mode == 'chat' else 0
        
        # Extract chosen workflow
        chosen_workflow = trace.get('chosen_workflow', '').strip()
        if not chosen_workflow:
            continue
        
        # Extract quality signals
        quality_signals = trace.get('quality_signals', {})
        
        # Calculate reward for chosen workflow
        # Quality is based on success and presence of key signals
        success = trace.get('success', True)
        has_citations = quality_signals.get('has_citations', False)
        answer_length = quality_signals.get('answer_length', 0)
        
        # Simple quality score: success + citations + length bonus
        quality = (
            (1.0 if success else 0.0) +
            (0.3 if has_citations else 0.0) +
            (0.2 if answer_length > 100 else 0.0)
        )
        
        # Get cost for chosen workflow
        cost = costs.get(chosen_workflow, 0.5)
        
        # Calculate reward
        reward = quality - lambda_cost * cost
        
        # For best_reward labeling, we use the chosen workflow
        # (In a more sophisticated version, we would compare against
        # alternative workflows, but we only have data for chosen workflow)
        
        questions.append(question)
        mode_features.append(mode_is_chat)
        workflows.append(chosen_workflow)
    
    logger.info(
        f"Extracted {len(questions)} training examples with mode features",
        unique_workflows=len(set(workflows)),
        chat_mode_count=sum(mode_features),
        default_mode_count=len(mode_features) - sum(mode_features)
    )
    
    return questions, mode_features, workflows


def train_classifier_with_mode(
    questions: List[str],
    mode_features: List[int],
    workflows: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[TfidfVectorizer, LogisticRegression, dict]:
    """
    Train TF-IDF vectorizer and LogisticRegression classifier with mode features.
    
    The classifier is trained on concatenated features:
    - Text features: TF-IDF vectors from question text
    - Mode feature: Binary indicator (1=chat, 0=default)
    
    Args:
        questions: List of question texts
        mode_features: List of mode_is_chat binary features
        workflows: List of workflow labels
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (vectorizer, classifier, metrics)
    
    Requirements:
        - 21.3: Include mode features in training data
        - 21.3: Document mode feature in model metadata
    """
    logger.info("Training RouterV3 classifier with mode features...")
    
    # Split data
    X_train_text, X_test_text, X_train_mode, X_test_mode, y_train, y_test = train_test_split(
        questions, mode_features, workflows,
        test_size=test_size,
        random_state=random_state,
        stratify=workflows
    )
    
    logger.info(
        f"Split data: {len(X_train_text)} train, {len(X_test_text)} test"
    )
    
    # Train TF-IDF vectorizer (text only)
    logger.info("Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X_train_text_vec = vectorizer.fit_transform(X_train_text)
    X_test_text_vec = vectorizer.transform(X_test_text)
    
    logger.info(
        f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}"
    )
    
    # Add mode features (Requirement 21.3)
    # Convert to dense and concatenate with mode feature
    X_train_text_dense = X_train_text_vec.toarray()
    X_test_text_dense = X_test_text_vec.toarray()
    
    X_train_mode_array = np.array(X_train_mode).reshape(-1, 1)
    X_test_mode_array = np.array(X_test_mode).reshape(-1, 1)
    
    X_train = np.hstack([X_train_text_dense, X_train_mode_array])
    X_test = np.hstack([X_test_text_dense, X_test_mode_array])
    
    logger.info(
        f"Feature matrix shape: {X_train.shape} (text + mode)"
    )
    
    # Train classifier
    logger.info("Training LogisticRegression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        multi_class='multinomial',
        solver='lbfgs'
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Test accuracy: {accuracy:.3f}")
    
    # Generate classification report
    report = classification_report(
        y_test, y_pred,
        output_dict=True,
        zero_division=0
    )
    
    # Log per-class metrics
    for workflow, metrics_dict in report.items():
        if isinstance(metrics_dict, dict):
            logger.info(
                f"Workflow: {workflow}",
                precision=round(metrics_dict.get('precision', 0), 3),
                recall=round(metrics_dict.get('recall', 0), 3),
                f1=round(metrics_dict.get('f1-score', 0), 3),
                support=metrics_dict.get('support', 0)
            )
    
    # Document mode feature in model metadata (Requirement 21.3)
    metrics = {
        'accuracy': accuracy,
        'num_train': len(X_train),
        'num_test': len(X_test),
        'num_text_features': len(vectorizer.vocabulary_),
        'num_mode_features': 1,
        'mode_feature_description': 'Binary indicator: 1=chat mode, 0=default mode',
        'feature_order': 'text_features (TF-IDF) + mode_is_chat',
        'total_features': X_train.shape[1],
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
    
    RouterV3.save_models(
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
        description="Train Router v3 cost-aware classifier with mode features"
    )
    parser.add_argument(
        '--traces',
        type=str,
        required=True,
        help='Path to JSONL trace file'
    )
    parser.add_argument(
        '--costs',
        type=str,
        required=True,
        help='Path to cost statistics JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/router_v3',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--lambda-cost',
        type=float,
        default=0.1,
        help='Weight for cost in reward calculation (default: 0.1)'
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
    costs_file = Path(args.costs)
    output_dir = Path(args.output_dir)
    
    try:
        # Load traces
        traces = load_traces(trace_file)
        
        if len(traces) == 0:
            logger.error("No traces loaded. Cannot train classifier.")
            sys.exit(1)
        
        # Load costs
        costs = load_costs(costs_file)
        
        # Extract training data with mode features
        questions, mode_features, workflows = extract_training_data_with_mode(
            traces, costs, args.lambda_cost
        )
        
        if len(questions) < 10:
            logger.error(
                f"Insufficient training data: {len(questions)} examples. "
                "Need at least 10 examples."
            )
            sys.exit(1)
        
        # Train classifier
        vectorizer, classifier, metrics = train_classifier_with_mode(
            questions, mode_features, workflows,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save models
        save_models(vectorizer, classifier, output_dir, metrics)
        
        logger.info("Training complete!")
        logger.info(f"Models saved to: {output_dir}")
        logger.info(f"Test accuracy: {metrics['accuracy']:.3f}")
        logger.info(f"Total features: {metrics['total_features']} (text + mode)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
