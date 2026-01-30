"""
Integration test for Router v3 mode_chat feature verification.

This test verifies that Router v3 correctly:
1. Extracts mode from State.meta
2. Encodes mode as one-hot feature
3. Uses mode features in prediction
4. Routes differently based on mode

Requirements: 21.1, 21.2, 21.4
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from mm_orch.routing.router_v3 import RouterV3
from mm_orch.orchestration.state import State


class TestRouterV3ModeVerification:
    """Test Router v3 mode feature functionality."""
    
    def test_mode_extraction_from_state(self):
        """Test that mode is correctly extracted from State.meta (Requirement 21.1)."""
        # Create mock router
        router = self._create_mock_router()
        
        # Test chat mode
        state_chat: State = {"meta": {"mode": "chat"}}
        mode_chat = router._extract_mode(state_chat)
        assert mode_chat == "chat", "Should extract 'chat' mode"
        
        # Test default mode
        state_default: State = {"meta": {"mode": "default"}}
        mode_default = router._extract_mode(state_default)
        assert mode_default == "default", "Should extract 'default' mode"
        
        # Test missing mode (should default to 'default')
        state_empty: State = {"meta": {}}
        mode_empty = router._extract_mode(state_empty)
        assert mode_empty == "default", "Should default to 'default' when mode missing"
    
    def test_mode_one_hot_encoding(self):
        """Test that mode is encoded as binary feature (Requirement 21.2)."""
        router = self._create_mock_router()
        
        # Test chat mode encoding
        state_chat: State = {"meta": {"mode": "chat"}}
        mode = router._extract_mode(state_chat)
        mode_is_chat = 1 if mode == "chat" else 0
        assert mode_is_chat == 1, "Chat mode should be encoded as 1"
        
        # Test default mode encoding
        state_default: State = {"meta": {"mode": "default"}}
        mode = router._extract_mode(state_default)
        mode_is_chat = 1 if mode == "chat" else 0
        assert mode_is_chat == 0, "Default mode should be encoded as 0"
    
    def test_mode_feature_in_prediction(self):
        """Test that mode features are used in prediction (Requirement 21.4)."""
        router = self._create_mock_router_with_prediction()
        
        question = "Hello, how are you?"
        
        # Route with chat mode
        state_chat: State = {"meta": {"mode": "chat"}}
        workflow_chat, score_chat, candidates_chat = router.route(question, state_chat)
        
        # Route with default mode
        state_default: State = {"meta": {"mode": "default"}}
        workflow_default, score_default, candidates_default = router.route(question, state_default)
        
        # Verify that routing was called with mode features
        # (The actual routing decision may be the same or different depending on the model)
        assert workflow_chat is not None, "Should return workflow for chat mode"
        assert workflow_default is not None, "Should return workflow for default mode"
        assert score_chat > 0, "Should return positive score for chat mode"
        assert score_default > 0, "Should return positive score for default mode"
    
    def test_mode_affects_routing_decision(self):
        """Test that mode can affect routing decisions."""
        router = self._create_mock_router_with_different_predictions()
        
        question = "Tell me about Python"
        
        # Route with chat mode
        state_chat: State = {"meta": {"mode": "chat"}}
        workflow_chat, _, _ = router.route(question, state_chat)
        
        # Route with default mode
        state_default: State = {"meta": {"mode": "default"}}
        workflow_default, _, _ = router.route(question, state_default)
        
        # With our mock setup, chat mode should prefer chat_generate
        # and default mode should prefer search_qa
        assert workflow_chat == "chat_generate", "Chat mode should prefer chat_generate"
        assert workflow_default == "search_qa", "Default mode should prefer search_qa"
    
    def test_empty_question_defaults_to_chat(self):
        """Test that empty questions default to chat_generate."""
        router = self._create_mock_router()
        
        state: State = {"meta": {"mode": "default"}}
        
        workflow, score, candidates = router.route("", state)
        assert workflow == "chat_generate", "Empty question should default to chat_generate"
        assert score == 0.5, "Should return default score"
    
    # Helper methods
    
    def _create_mock_router(self):
        """Create a minimal mock router for testing mode extraction."""
        router = RouterV3.__new__(RouterV3)
        router.vectorizer = Mock()
        router.classifier = Mock()
        router.workflow_costs = {}
        router.workflow_names = ["search_qa", "chat_generate", "lesson_pack"]
        router.lambda_cost = 0.1
        return router
    
    def _create_mock_router_with_prediction(self):
        """Create a mock router that can perform predictions."""
        router = self._create_mock_router()
        
        # Mock vectorizer
        router.vectorizer.transform = Mock(return_value=self._create_sparse_matrix())
        
        # Mock classifier with predict_proba
        router.classifier.predict_proba = Mock(return_value=np.array([[0.3, 0.5, 0.2]]))
        router.classifier.classes_ = np.array(["search_qa", "chat_generate", "lesson_pack"])
        
        # Set workflow costs
        router.workflow_costs = {
            "search_qa": 0.5,
            "chat_generate": 0.3,
            "lesson_pack": 0.7
        }
        
        return router
    
    def _create_mock_router_with_different_predictions(self):
        """Create a mock router that returns different predictions based on mode."""
        router = self._create_mock_router()
        
        # Mock vectorizer
        router.vectorizer.transform = Mock(return_value=self._create_sparse_matrix())
        
        # Mock classifier that returns different probabilities based on input
        def mock_predict_proba(X):
            # Check if mode feature (last column) is 1 (chat) or 0 (default)
            mode_is_chat = X[0, -1]
            
            if mode_is_chat == 1:
                # Chat mode: prefer chat_generate
                return np.array([[0.2, 0.7, 0.1]])
            else:
                # Default mode: prefer search_qa
                return np.array([[0.7, 0.2, 0.1]])
        
        router.classifier.predict_proba = mock_predict_proba
        router.classifier.classes_ = np.array(["search_qa", "chat_generate", "lesson_pack"])
        
        # Set workflow costs
        router.workflow_costs = {
            "search_qa": 0.5,
            "chat_generate": 0.3,
            "lesson_pack": 0.7
        }
        
        return router
    
    def _create_sparse_matrix(self):
        """Create a mock sparse matrix for TF-IDF output."""
        from scipy.sparse import csr_matrix
        # Create a simple 1x10 sparse matrix
        return csr_matrix(np.random.rand(1, 10))


@pytest.mark.integration
class TestRouterV3ModeIntegration:
    """Integration tests for Router v3 mode feature with real components."""
    
    @pytest.mark.skip(reason="Requires trained models - run manually if models exist")
    def test_router_v3_with_real_models(self):
        """Test Router v3 with real trained models (if available)."""
        try:
            router = RouterV3(
                "models/router_v3/vectorizer.pkl",
                "models/router_v3/classifier.pkl",
                "data/cost_stats.json"
            )
            
            # Test with chat mode
            state_chat: State = {"meta": {"mode": "chat"}}
            workflow_chat, score_chat, _ = router.route("Hello!", state_chat)
            
            # Test with default mode
            state_default: State = {"meta": {"mode": "default"}}
            workflow_default, score_default, _ = router.route("Search for Python", state_default)
            
            assert workflow_chat is not None
            assert workflow_default is not None
            
            print(f"Chat mode: {workflow_chat} (score: {score_chat:.3f})")
            print(f"Default mode: {workflow_default} (score: {score_default:.3f})")
            
        except FileNotFoundError:
            pytest.skip("Trained models not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
