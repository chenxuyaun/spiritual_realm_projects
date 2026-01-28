"""
Router v2: Classifier-based workflow selection.

This router uses a trained machine learning classifier (TF-IDF + LogisticRegression)
to predict the most appropriate workflow based on the question text.

Requirements: 15.1, 15.3, 15.4
"""

import joblib
from pathlib import Path
from typing import List, Tuple

from mm_orch.orchestration.state import State
from mm_orch.logger import get_logger


logger = get_logger(__name__)


class RouterV2:
    """
    Classifier-based router using TF-IDF + LogisticRegression.

    This router uses a trained machine learning model to predict workflow
    selection based on question text. It provides probability distributions
    over all workflows, enabling confidence-based routing decisions.

    The classifier is trained on execution traces where questions are labeled
    with the workflows that were actually used.

    Example:
        router = RouterV2("models/vectorizer.pkl", "models/classifier.pkl")
        workflow, confidence, candidates = router.route("搜索Python教程", state)
        # Returns: ("search_qa", 0.85, [("search_qa_fast", 0.10), ...])
    """

    def __init__(self, vectorizer_path: str, classifier_path: str):
        """
        Initialize router with trained models.

        Args:
            vectorizer_path: Path to saved TF-IDF vectorizer
            classifier_path: Path to saved classifier model

        Raises:
            FileNotFoundError: If model files don't exist
            Exception: If models fail to load
        """
        self.vectorizer_path = Path(vectorizer_path)
        self.classifier_path = Path(classifier_path)

        # Load models
        self.vectorizer = self._load_vectorizer()
        self.classifier = self._load_classifier()

        # Get workflow names from classifier
        self.workflow_names = list(self.classifier.classes_)

        logger.info(
            "RouterV2 initialized",
            vectorizer_path=str(self.vectorizer_path),
            classifier_path=str(self.classifier_path),
            num_workflows=len(self.workflow_names),
        )

    def _load_vectorizer(self):
        """
        Load TF-IDF vectorizer from disk.

        Returns:
            Loaded vectorizer

        Raises:
            FileNotFoundError: If vectorizer file doesn't exist
        """
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {self.vectorizer_path}")

        try:
            vectorizer = joblib.load(self.vectorizer_path)
            logger.debug("Loaded vectorizer", path=str(self.vectorizer_path))
            return vectorizer
        except Exception as e:
            logger.error("Failed to load vectorizer", path=str(self.vectorizer_path), error=str(e))
            raise

    def _load_classifier(self):
        """
        Load classifier model from disk.

        Returns:
            Loaded classifier

        Raises:
            FileNotFoundError: If classifier file doesn't exist
        """
        if not self.classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {self.classifier_path}")

        try:
            classifier = joblib.load(self.classifier_path)
            logger.debug("Loaded classifier", path=str(self.classifier_path))
            return classifier
        except Exception as e:
            logger.error("Failed to load classifier", path=str(self.classifier_path), error=str(e))
            raise

    def route(self, question: str, state: State) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Route question to appropriate workflow using classifier.

        Args:
            question: User's question text
            state: Current workflow state

        Returns:
            Tuple of (workflow_name, confidence, candidates) where:
            - workflow_name: Selected workflow
            - confidence: Probability for selected workflow (0.0 to 1.0)
            - candidates: List of (workflow_name, probability) tuples, sorted by probability

        Example:
            workflow, conf, candidates = router.route("搜索Python", state)
            # Returns: ("search_qa", 0.85, [("search_qa_fast", 0.10), ...])
        """
        if not question or not question.strip():
            # Default to chat for empty questions
            return "chat_generate", 0.5, [("search_qa", 0.3)]

        question = question.strip()

        # Vectorize question
        X = self.vectorizer.transform([question])

        # Get probability distribution
        probs = self.classifier.predict_proba(X)[0]

        # Create candidates list (workflow, probability) sorted by probability
        candidates = list(zip(self.workflow_names, probs))
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Best workflow is highest probability
        best_workflow, best_prob = candidates[0]

        # Log decision
        self._log_decision(question, best_workflow, best_prob, candidates)

        return best_workflow, best_prob, candidates

    def _log_decision(
        self, question: str, workflow: str, confidence: float, candidates: List[Tuple[str, float]]
    ) -> None:
        """
        Log routing decision.

        Args:
            question: User's question
            workflow: Selected workflow
            confidence: Confidence score
            candidates: All candidates with probabilities
        """
        # Format top 3 candidates for logging
        top_candidates = {wf: round(prob, 3) for wf, prob in candidates[:3]}

        logger.info(
            "RouterV2 decision",
            question_preview=question[:100],
            selected_workflow=workflow,
            confidence=round(confidence, 3),
            top_candidates=top_candidates,
        )

    @staticmethod
    def save_models(vectorizer, classifier, vectorizer_path: str, classifier_path: str) -> None:
        """
        Save trained models to disk.

        This is a utility method for the training script to save models
        after training.

        Args:
            vectorizer: Trained TF-IDF vectorizer
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
            "Saved RouterV2 models",
            vectorizer_path=str(vectorizer_path),
            classifier_path=str(classifier_path),
        )
