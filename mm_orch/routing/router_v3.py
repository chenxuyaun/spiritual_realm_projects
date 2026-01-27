"""
Router v3: Cost-aware workflow selection with mode features.

This router extends Router v2 by incorporating cost statistics and mode features
to make cost-aware routing decisions. It balances quality (predicted probability)
against cost (latency, VRAM, model loads).

Requirements: 16.1, 16.2, 16.3, 16.4
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from mm_orch.orchestration.state import State
from mm_orch.logger import get_logger


logger = get_logger(__name__)


class RouterV3:
    """
    Cost-aware router with mode features.
    
    This router extends the classifier-based approach with:
    1. Cost statistics: Incorporates average latency, VRAM, and model loads
    2. Mode features: One-hot encoding of execution mode (chat vs default)
    3. Cost-aware scoring: quality - lambda * cost
    
    The router learns to balance quality and cost, preferring faster/cheaper
    workflows when appropriate while maintaining quality.
    
    Example:
        router = RouterV3(
            "models/vectorizer.pkl",
            "models/classifier.pkl",
            "data/cost_stats.json"
        )
        workflow, confidence, candidates = router.route("搜索Python", state)
        # Returns cost-aware selection
    """
    
    def __init__(
        self,
        vectorizer_path: str,
        classifier_path: str,
        costs_path: str,
        lambda_cost: float = 0.1
    ):
        """
        Initialize router with trained models and cost statistics.
        
        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer
            classifier_path: Path to saved classifier model
            costs_path: Path to JSON file with cost statistics
            lambda_cost: Weight for cost in scoring (default: 0.1)
            
        Raises:
            FileNotFoundError: If model or cost files don't exist
        """
        self.vectorizer_path = Path(vectorizer_path)
        self.classifier_path = Path(classifier_path)
        self.costs_path = Path(costs_path)
        self.lambda_cost = lambda_cost
        
        # Load models
        self.vectorizer = self._load_vectorizer()
        self.classifier = self._load_classifier()
        
        # Load cost statistics
        self.workflow_costs = self._load_costs()
        
        # Get workflow names from classifier
        self.workflow_names = list(self.classifier.classes_)
        
        logger.info(
            "RouterV3 initialized",
            vectorizer_path=str(self.vectorizer_path),
            classifier_path=str(self.classifier_path),
            costs_path=str(self.costs_path),
            lambda_cost=lambda_cost,
            num_workflows=len(self.workflow_names)
        )
    
    def _load_vectorizer(self):
        """Load TF-IDF vectorizer from disk."""
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer not found: {self.vectorizer_path}"
            )
        
        try:
            vectorizer = joblib.load(self.vectorizer_path)
            logger.debug("Loaded vectorizer", path=str(self.vectorizer_path))
            return vectorizer
        except Exception as e:
            logger.error(
                "Failed to load vectorizer",
                path=str(self.vectorizer_path),
                error=str(e)
            )
            raise
    
    def _load_classifier(self):
        """Load classifier model from disk."""
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found: {self.classifier_path}"
            )
        
        try:
            classifier = joblib.load(self.classifier_path)
            logger.debug("Loaded classifier", path=str(self.classifier_path))
            return classifier
        except Exception as e:
            logger.error(
                "Failed to load classifier",
                path=str(self.classifier_path),
                error=str(e)
            )
            raise
    
    def _load_costs(self) -> Dict[str, float]:
        """
        Load cost statistics from JSON file.
        
        Returns:
            Dictionary mapping workflow names to normalized costs
        """
        if not self.costs_path.exists():
            logger.warning(
                f"Cost statistics not found: {self.costs_path}. "
                "Using default costs."
            )
            return {}
        
        try:
            with open(self.costs_path, 'r', encoding='utf-8') as f:
                cost_data = json.load(f)
            
            # Extract and normalize costs
            costs = {}
            for workflow_name, stats in cost_data.items():
                # Combine latency, VRAM, and model loads into single cost metric
                # Normalize each component to [0, 1] range
                latency = stats.get('avg_latency_ms', 0)
                vram = stats.get('avg_vram_mb', 0)
                loads = stats.get('avg_model_loads', 0)
                
                # Weighted combination (latency is most important)
                cost = (
                    0.5 * min(latency / 10000, 1.0) +  # Normalize to 10s max
                    0.3 * min(vram / 4000, 1.0) +      # Normalize to 4GB max
                    0.2 * min(loads / 5, 1.0)          # Normalize to 5 loads max
                )
                
                costs[workflow_name] = cost
            
            logger.debug(
                f"Loaded cost statistics for {len(costs)} workflows"
            )
            return costs
            
        except Exception as e:
            logger.error(
                "Failed to load cost statistics",
                path=str(self.costs_path),
                error=str(e)
            )
            return {}
    
    def route(
        self,
        question: str,
        state: State
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Route question to appropriate workflow using cost-aware scoring.
        
        Args:
            question: User's question text
            state: Current workflow state (contains mode in meta)
            
        Returns:
            Tuple of (workflow_name, score, candidates) where:
            - workflow_name: Selected workflow
            - score: Cost-aware score (quality - lambda * cost)
            - candidates: List of (workflow_name, score) tuples, sorted by score
            
        Requirements:
            - 21.1: Extract mode from State.meta
            - 21.2: Encode mode as one-hot feature
            - 21.4: Use mode features in prediction
            
        Example:
            workflow, score, candidates = router.route("搜索Python", state)
            # Returns cost-aware selection
        """
        if not question or not question.strip():
            # Default to chat for empty questions
            return "chat_generate", 0.5, [("search_qa", 0.3)]
        
        question = question.strip()
        
        # Extract mode feature from state (Requirement 21.1)
        mode = self._extract_mode(state)
        mode_is_chat = 1 if mode == "chat" else 0
        
        # Vectorize question
        X_text = self.vectorizer.transform([question])
        
        # Add mode feature (one-hot encoding) (Requirement 21.2)
        # Convert sparse matrix to dense and add mode feature
        X_text_dense = X_text.toarray()
        X_with_mode = np.hstack([X_text_dense, np.array([[mode_is_chat]])])
        
        # Get quality predictions (probabilities) using mode features (Requirement 21.4)
        # Note: The classifier was trained with mode features, so we must include them
        quality_probs = self.classifier.predict_proba(X_with_mode)[0]
        
        # Calculate cost-aware scores
        scores = []
        for workflow, quality in zip(self.workflow_names, quality_probs):
            cost = self.workflow_costs.get(workflow, 0.5)  # Default cost
            score = quality - self.lambda_cost * cost
            scores.append((workflow, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Best workflow is highest score
        best_workflow, best_score = scores[0]
        
        # Log decision
        self._log_decision(question, mode, best_workflow, best_score, scores)
        
        return best_workflow, best_score, scores
    
    def _extract_mode(self, state: State) -> str:
        """
        Extract execution mode from state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Mode string ("chat" or "default")
        """
        meta = state.get("meta", {})
        return meta.get("mode", "default")
    
    def _log_decision(
        self,
        question: str,
        mode: str,
        workflow: str,
        score: float,
        candidates: List[Tuple[str, float]]
    ) -> None:
        """
        Log routing decision.
        
        Args:
            question: User's question
            mode: Execution mode
            workflow: Selected workflow
            score: Cost-aware score
            candidates: All candidates with scores
        """
        # Format top 3 candidates for logging
        top_candidates = {wf: round(s, 3) for wf, s in candidates[:3]}
        
        logger.info(
            "RouterV3 decision",
            question_preview=question[:100],
            mode=mode,
            selected_workflow=workflow,
            score=round(score, 3),
            top_candidates=top_candidates
        )
    
    @staticmethod
    def save_models(
        vectorizer,
        classifier,
        vectorizer_path: str,
        classifier_path: str
    ) -> None:
        """
        Save trained models to disk.
        
        This is a utility method for the training script to save models
        after training.
        
        Args:
            vectorizer: Trained TF-IDF vectorizer with mode features
            classifier: Trained classifier model
            vectorizer_path: Path to save vectorizer
            classifier_path: Path to save classifier
        """
        vectorizer_path = Path(vectorizer_path)
        classifier_path = Path(classifier_path)
        
        # Ensure directories exist
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)
        
        logger.info(
            "Saved RouterV3 models",
            vectorizer_path=str(vectorizer_path),
            classifier_path=str(classifier_path)
        )
