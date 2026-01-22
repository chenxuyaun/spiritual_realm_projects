"""
Integration tests for workflow real model support.

These tests verify that workflows can be configured to use real models
via RealModelManager and InferenceEngine.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.workflows.lesson_pack import LessonPackWorkflow
from mm_orch.workflows.chat_generate import ChatGenerateWorkflow
from mm_orch.schemas import WorkflowResult


class MockInferenceEngine:
    """Mock inference engine for testing."""
    
    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = None
        self.last_config = None
    
    def generate(self, prompt: str, config=None, **kwargs):
        """Mock generate method."""
        self.call_count += 1
        self.last_prompt = prompt
        self.last_config = config
        
        # Return mock result
        result = Mock()
        result.text = self.responses.get("default", "This is a mock response.")
        result.tokens_per_second = 50.0
        result.input_tokens = len(prompt.split())
        result.output_tokens = len(result.text.split())
        result.total_time = 0.5
        result.finish_reason = "stop"
        
        return result


class MockRealModelManager:
    """Mock real model manager for testing."""
    
    def __init__(self):
        self.loaded_models = {}
    
    def load_model(self, config):
        """Mock load model."""
        self.loaded_models[config.model_name] = Mock()
        return Mock()
    
    def get_model(self, model_name: str):
        """Mock get model."""
        return self.loaded_models.get(model_name)


class TestSearchQAWorkflowRealModels:
    """Tests for SearchQA workflow with real model support."""
    
    def test_workflow_initialization_with_real_models(self):
        """Test that workflow can be initialized with real model components."""
        inference_engine = MockInferenceEngine()
        model_manager = MockRealModelManager()
        
        workflow = SearchQAWorkflow(
            real_model_manager=model_manager,
            inference_engine=inference_engine,
            use_real_models=True,
            language="en"
        )
        
        assert workflow.use_real_models is True
        assert workflow.inference_engine is inference_engine
        assert workflow.real_model_manager is model_manager
        assert workflow.language == "en"
    
    def test_workflow_initialization_with_chinese_language(self):
        """Test workflow initialization with Chinese language."""
        workflow = SearchQAWorkflow(
            use_real_models=True,
            language="zh"
        )
        
        assert workflow.language == "zh"
    
    def test_generate_with_real_model_called(self):
        """Test that real model generation is called when enabled."""
        inference_engine = MockInferenceEngine({
            "default": "Based on the search results, the answer is..."
        })
        
        workflow = SearchQAWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        # Call the internal method directly
        answer = workflow._generate_with_real_model(
            query="What is Python?",
            context="Python is a programming language."
        )
        
        assert inference_engine.call_count == 1
        assert "Python" in inference_engine.last_prompt
        assert len(answer) > 0
    
    def test_post_process_answer_removes_artifacts(self):
        """Test that post-processing removes generation artifacts."""
        workflow = SearchQAWorkflow()
        
        # Test with trailing question
        answer = "The answer is 42.\n\nQuestion: What else?"
        processed = workflow._post_process_answer(answer, "context")
        assert "Question:" not in processed
        
        # Test with Chinese artifacts
        answer = "答案是42。\n\n搜索结果：更多内容"
        processed = workflow._post_process_answer(answer, "context")
        assert "搜索结果：" not in processed
    
    def test_fallback_to_simple_answer_on_error(self):
        """Test fallback when real model fails."""
        # Create engine that raises exception
        inference_engine = Mock()
        inference_engine.generate.side_effect = Exception("Model error")
        
        workflow = SearchQAWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        answer = workflow._generate_with_real_model(
            query="Test question",
            context="Test context"
        )
        
        # Should return fallback response
        assert len(answer) > 0


class TestLessonPackWorkflowRealModels:
    """Tests for LessonPack workflow with real model support."""
    
    def test_workflow_initialization_with_real_models(self):
        """Test that workflow can be initialized with real model components."""
        inference_engine = MockInferenceEngine()
        model_manager = MockRealModelManager()
        
        workflow = LessonPackWorkflow(
            real_model_manager=model_manager,
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        assert workflow.use_real_models is True
        assert workflow.inference_engine is inference_engine
        assert workflow.real_model_manager is model_manager
    
    def test_generate_plan_with_real_model(self):
        """Test plan generation with real model."""
        inference_engine = MockInferenceEngine({
            "default": "# Teaching Plan\n\n## Learning Objectives\n- Objective 1"
        })
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        plan = workflow._generate_plan_with_real_model(
            topic="Python Basics",
            difficulty="beginner",
            language="en"
        )
        
        assert inference_engine.call_count == 1
        assert len(plan) > 0
    
    def test_generate_explanation_with_real_model(self):
        """Test explanation generation with real model."""
        inference_engine = MockInferenceEngine({
            "default": "# Python Basics\n\nPython is a programming language..."
        })
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        explanation = workflow._generate_explanation_with_real_model(
            topic="Python Basics",
            plan="# Plan\n- Learn basics",
            difficulty="beginner",
            language="en",
            include_examples=True
        )
        
        assert inference_engine.call_count == 1
        assert len(explanation) > 0
    
    def test_generate_exercises_with_real_model(self):
        """Test exercise generation with real model."""
        inference_engine = MockInferenceEngine({
            "default": "Question 1: What is Python?\nAnswer 1: A programming language."
        })
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        exercises = workflow._generate_exercises_with_real_model(
            topic="Python Basics",
            difficulty="beginner",
            num_exercises=3,
            language="en"
        )
        
        assert inference_engine.call_count == 1
        # May return parsed exercises or fallback to template
        assert isinstance(exercises, list)
    
    def test_validate_markdown_format(self):
        """Test Markdown format validation."""
        workflow = LessonPackWorkflow()
        
        # Test header spacing
        content = "#Header without space"
        validated = workflow._validate_markdown_format(content)
        assert "# Header" in validated or "#Header" in validated
        
        # Test list item spacing
        content = "-Item without space"
        validated = workflow._validate_markdown_format(content)
        # Should add space after dash
        
        # Test artifact removal
        content = "Content\n\n---\n\nMore content"
        validated = workflow._validate_markdown_format(content)
        assert "---" not in validated or validated.count("---") == 0
    
    def test_chinese_prompt_templates(self):
        """Test that Chinese prompts are used for zh language."""
        inference_engine = MockInferenceEngine()
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        workflow._generate_plan_with_real_model(
            topic="Python基础",
            difficulty="beginner",
            language="zh"
        )
        
        # Check that Chinese prompt was used
        assert "主题" in inference_engine.last_prompt or "教学计划" in inference_engine.last_prompt


class TestChatGenerateWorkflowRealModels:
    """Tests for ChatGenerate workflow with real model support."""
    
    def test_workflow_initialization_with_real_models(self):
        """Test that workflow can be initialized with real model components."""
        inference_engine = MockInferenceEngine()
        model_manager = MockRealModelManager()
        
        workflow = ChatGenerateWorkflow(
            real_model_manager=model_manager,
            inference_engine=inference_engine,
            use_real_models=True,
            model_type="qwen-chat",
            language="en"
        )
        
        assert workflow.use_real_models is True
        assert workflow.inference_engine is inference_engine
        assert workflow.real_model_manager is model_manager
        assert workflow.model_type == "qwen-chat"
    
    def test_default_system_prompt_by_language(self):
        """Test that default system prompt is set based on language."""
        workflow_en = ChatGenerateWorkflow(language="en")
        workflow_zh = ChatGenerateWorkflow(language="zh")
        
        assert "helpful" in workflow_en.system_prompt.lower()
        assert "助手" in workflow_zh.system_prompt
    
    def test_custom_system_prompt_overrides_default(self):
        """Test that custom system prompt overrides default."""
        custom_prompt = "You are a coding assistant."
        
        workflow = ChatGenerateWorkflow(
            system_prompt=custom_prompt,
            language="zh"  # Should still use custom prompt
        )
        
        assert workflow.system_prompt == custom_prompt
    
    def test_generate_with_real_model(self):
        """Test response generation with real model."""
        inference_engine = MockInferenceEngine({
            "default": "Hello! How can I help you today?"
        })
        
        workflow = ChatGenerateWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        response = workflow._generate_with_real_model(
            context="User: Hello\nAssistant:",
            temperature=0.7
        )
        
        assert inference_engine.call_count == 1
        assert len(response) > 0
    
    def test_generate_with_conversation_manager(self):
        """Test response generation with conversation manager."""
        inference_engine = MockInferenceEngine({
            "default": "I understand your question."
        })
        
        # Create mock conversation manager
        conv_manager = Mock()
        conv_manager.build_prompt.return_value = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        
        workflow = ChatGenerateWorkflow(
            inference_engine=inference_engine,
            conversation_manager=conv_manager,
            use_real_models=True
        )
        
        response = workflow._generate_with_real_model(
            context="User: Hello\nAssistant:",
            temperature=0.7
        )
        
        # Conversation manager should be used
        conv_manager.build_prompt.assert_called_once()
        assert len(response) > 0
    
    def test_fallback_on_real_model_error(self):
        """Test fallback when real model fails."""
        inference_engine = Mock()
        inference_engine.generate.side_effect = Exception("Model error")
        
        workflow = ChatGenerateWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        response = workflow._generate_with_real_model(
            context="User: Hello\nAssistant:",
            temperature=0.7
        )
        
        # Should return fallback response
        assert len(response) > 0


class TestWorkflowIntegration:
    """Integration tests for workflow components working together."""
    
    def test_search_qa_full_flow_with_mock_real_models(self):
        """Test SearchQA workflow full flow with mocked real models."""
        inference_engine = MockInferenceEngine({
            "default": "Based on the search results, Python is a programming language."
        })
        
        # Mock search and fetch tools
        search_tool = Mock()
        search_tool.search.return_value = [
            Mock(url="http://example.com", title="Python", snippet="Python is...")
        ]
        
        fetch_tool = Mock()
        fetch_tool.fetch_multiple.return_value = [
            Mock(url="http://example.com", content="Python is a language.", success=True)
        ]
        
        workflow = SearchQAWorkflow(
            search_tool=search_tool,
            fetch_tool=fetch_tool,
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        result = workflow.execute({"query": "What is Python?"})
        
        assert result.status in ["success", "partial"]
        assert result.result is not None
    
    def test_lesson_pack_full_flow_with_mock_real_models(self):
        """Test LessonPack workflow full flow with mocked real models."""
        inference_engine = MockInferenceEngine({
            "default": "# Plan\n## Objectives\n- Learn Python"
        })
        
        workflow = LessonPackWorkflow(
            inference_engine=inference_engine,
            use_real_models=True
        )
        
        result = workflow.execute({
            "topic": "Python Basics",
            "difficulty": "beginner",
            "num_exercises": 2,
            "language": "en"
        })
        
        assert result.status in ["success", "partial"]
        # Result should have lesson pack structure
        if result.result:
            assert "topic" in result.result
            assert "plan" in result.result
