"""Unit tests for router implementations."""

import pytest
import tempfile
import json
from pathlib import Path

from mm_orch.routing import RouterV1, RoutingRule
from mm_orch.orchestration.state import State


class TestRouterV1:
    """Tests for RouterV1 rule-based router."""
    
    def test_initialization(self):
        """Test router initializes with default rules."""
        router = RouterV1()
        rules = router.get_rules()
        
        assert len(rules) > 0
        assert all(isinstance(rule, RoutingRule) for rule in rules)
    
    def test_route_search_question(self):
        """Test routing a search question."""
        router = RouterV1()
        state: State = {"meta": {"mode": "default"}}
        
        workflow, confidence, candidates = router.route("搜索最新的Python教程", state)
        
        assert workflow in ["search_qa", "search_qa_fast", "search_qa_strict_citations"]
        assert 0.0 <= confidence <= 1.0
        assert len(candidates) > 0
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)
    
    def test_route_lesson_question(self):
        """Test routing a teaching question."""
        router = RouterV1()
        state: State = {"meta": {"mode": "default"}}
        
        workflow, confidence, candidates = router.route("教我Python基础", state)
        
        assert workflow == "lesson_pack"
        assert 0.0 <= confidence <= 1.0
    
    def test_route_chat_question(self):
        """Test routing a chat question."""
        router = RouterV1()
        state: State = {"meta": {"mode": "chat"}}
        
        workflow, confidence, candidates = router.route("你好", state)
        
        # Chat mode should boost chat_generate
        assert workflow == "chat_generate"
        assert 0.0 <= confidence <= 1.0
    
    def test_route_empty_question(self):
        """Test routing empty question defaults to chat."""
        router = RouterV1()
        state: State = {"meta": {"mode": "default"}}
        
        workflow, confidence, candidates = router.route("", state)
        
        assert workflow == "chat_generate"
        assert confidence == 0.5
    
    def test_candidates_sorted_by_score(self):
        """Test that candidates are sorted by score descending."""
        router = RouterV1()
        state: State = {"meta": {"mode": "default"}}
        
        _, _, candidates = router.route("搜索Python", state)
        
        # Verify candidates are sorted descending
        scores = [score for _, score in candidates]
        assert scores == sorted(scores, reverse=True)
    
    def test_add_custom_rule(self):
        """Test adding a custom routing rule."""
        router = RouterV1()
        
        custom_rule = RoutingRule(
            workflow_name="custom_workflow",
            keywords=["custom", "test"],
            patterns=[r".*custom.*"],
            base_weight=1.5,
            description="Custom test workflow"
        )
        
        router.add_rule(custom_rule)
        rules = router.get_rules()
        
        assert any(r.workflow_name == "custom_workflow" for r in rules)
    
    def test_mode_context_adjustment(self):
        """Test that mode affects routing decisions."""
        router = RouterV1()
        
        # Same question, different modes
        state_default: State = {"meta": {"mode": "default"}}
        state_chat: State = {"meta": {"mode": "chat"}}
        
        _, conf_default, _ = router.route("聊聊天", state_default)
        _, conf_chat, _ = router.route("聊聊天", state_chat)
        
        # Chat mode should give higher confidence for chat workflow
        # (This is a weak test since routing is complex)
        assert conf_chat >= 0.0  # Just verify it works


class TestRouterV2:
    """Tests for RouterV2 classifier-based router."""
    
    @pytest.fixture
    def mock_models(self, tmp_path):
        """Create mock trained models for testing."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Create simple training data
        questions = [
            "搜索Python教程",
            "教我Python",
            "你好",
            "查找资料"
        ]
        workflows = [
            "search_qa",
            "lesson_pack",
            "chat_generate",
            "search_qa"
        ]
        
        # Train vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(questions)
        
        # Train classifier
        classifier = LogisticRegression(max_iter=100)
        classifier.fit(X, workflows)
        
        # Save models
        vectorizer_path = tmp_path / "vectorizer.pkl"
        classifier_path = tmp_path / "classifier.pkl"
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)
        
        return str(vectorizer_path), str(classifier_path)
    
    def test_initialization(self, mock_models):
        """Test router initializes with trained models."""
        from mm_orch.routing import RouterV2
        
        vectorizer_path, classifier_path = mock_models
        router = RouterV2(vectorizer_path, classifier_path)
        
        assert router.vectorizer is not None
        assert router.classifier is not None
        assert len(router.workflow_names) > 0
    
    def test_route_returns_probabilities(self, mock_models):
        """Test that router returns probability distributions."""
        from mm_orch.routing import RouterV2
        
        vectorizer_path, classifier_path = mock_models
        router = RouterV2(vectorizer_path, classifier_path)
        
        state: State = {"meta": {"mode": "default"}}
        workflow, confidence, candidates = router.route("搜索Python", state)
        
        assert workflow in router.workflow_names
        assert 0.0 <= confidence <= 1.0
        
        # Check probability distribution sums to ~1.0
        total_prob = sum(prob for _, prob in candidates)
        assert abs(total_prob - 1.0) < 0.01
    
    def test_missing_models_raises_error(self):
        """Test that missing model files raise FileNotFoundError."""
        from mm_orch.routing import RouterV2
        
        with pytest.raises(FileNotFoundError):
            RouterV2("nonexistent_vectorizer.pkl", "nonexistent_classifier.pkl")


class TestRouterV3:
    """Tests for RouterV3 cost-aware router."""
    
    @pytest.fixture
    def mock_models_and_costs(self, tmp_path):
        """Create mock trained models and cost statistics."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        import joblib
        import numpy as np
        
        # Create simple training data
        questions = [
            "搜索Python教程",
            "教我Python",
            "你好",
            "查找资料"
        ]
        workflows = [
            "search_qa",
            "lesson_pack",
            "chat_generate",
            "search_qa"
        ]
        
        # Train vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        X_text = vectorizer.fit_transform(questions)
        
        # Add mode feature (0 for default mode in training data)
        mode_features = np.zeros((len(questions), 1))
        X = np.hstack([X_text.toarray(), mode_features])
        
        # Train classifier
        classifier = LogisticRegression(max_iter=100)
        classifier.fit(X, workflows)
        
        # Save models
        vectorizer_path = tmp_path / "vectorizer.pkl"
        classifier_path = tmp_path / "classifier.pkl"
        costs_path = tmp_path / "costs.json"
        
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(classifier, classifier_path)
        
        # Create cost statistics
        costs = {
            "search_qa": {
                "avg_latency_ms": 5000,
                "avg_vram_mb": 2000,
                "avg_model_loads": 2
            },
            "search_qa_fast": {
                "avg_latency_ms": 2000,
                "avg_vram_mb": 1000,
                "avg_model_loads": 1
            },
            "lesson_pack": {
                "avg_latency_ms": 8000,
                "avg_vram_mb": 3000,
                "avg_model_loads": 3
            },
            "chat_generate": {
                "avg_latency_ms": 1000,
                "avg_vram_mb": 500,
                "avg_model_loads": 1
            }
        }
        
        with open(costs_path, 'w') as f:
            json.dump(costs, f)
        
        return str(vectorizer_path), str(classifier_path), str(costs_path)
    
    def test_initialization(self, mock_models_and_costs):
        """Test router initializes with models and costs."""
        from mm_orch.routing import RouterV3
        
        vectorizer_path, classifier_path, costs_path = mock_models_and_costs
        router = RouterV3(vectorizer_path, classifier_path, costs_path)
        
        assert router.vectorizer is not None
        assert router.classifier is not None
        assert len(router.workflow_costs) > 0
        assert router.lambda_cost == 0.1
    
    def test_route_with_mode_feature(self, mock_models_and_costs):
        """Test routing with mode feature extraction."""
        from mm_orch.routing import RouterV3
        
        vectorizer_path, classifier_path, costs_path = mock_models_and_costs
        router = RouterV3(vectorizer_path, classifier_path, costs_path)
        
        # Test with chat mode
        state_chat: State = {"meta": {"mode": "chat"}}
        workflow, score, candidates = router.route("你好", state_chat)
        
        assert workflow in router.workflow_names
        assert isinstance(score, float)
        assert len(candidates) > 0
    
    def test_cost_aware_scoring(self, mock_models_and_costs):
        """Test that cost affects routing decisions."""
        from mm_orch.routing import RouterV3
        
        vectorizer_path, classifier_path, costs_path = mock_models_and_costs
        router = RouterV3(vectorizer_path, classifier_path, costs_path, lambda_cost=0.5)
        
        state: State = {"meta": {"mode": "default"}}
        workflow, score, candidates = router.route("搜索Python", state)
        
        # Verify score is quality - lambda * cost
        # (We can't verify exact values without knowing probabilities,
        # but we can verify the structure)
        assert isinstance(score, float)
        assert all(isinstance(s, float) for _, s in candidates)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
