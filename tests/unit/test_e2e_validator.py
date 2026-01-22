"""
端到端验证器模块单元测试
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from mm_orch.validation.e2e_validator import (
    E2EValidator,
    ValidationResult,
    TestCase,
    SearchQATestCase,
    LessonPackTestCase,
    ConversationTestCase,
)


class TestTestCase:
    """TestCase测试"""
    
    def test_create_test_case(self):
        """测试创建测试用例"""
        case = TestCase(
            id="test_001",
            name="Test case",
            description="A test case",
            tags=["tag1", "tag2"],
        )
        
        assert case.id == "test_001"
        assert case.name == "Test case"
        assert case.description == "A test case"
        assert case.tags == ["tag1", "tag2"]
    
    def test_to_dict(self):
        """测试转换为字典"""
        case = TestCase(
            id="test_001",
            name="Test case",
            tags=["tag1"],
        )
        
        d = case.to_dict()
        
        assert d["id"] == "test_001"
        assert d["name"] == "Test case"
        assert d["tags"] == ["tag1"]


class TestSearchQATestCase:
    """SearchQATestCase测试"""
    
    def test_create_search_qa_case(self):
        """测试创建SearchQA测试用例"""
        case = SearchQATestCase(
            id="sq_001",
            name="Search QA test",
            query="What is Python?",
            expected_keywords=["Python", "programming"],
            expected_sources=2,
            max_response_time=30.0,
        )
        
        assert case.query == "What is Python?"
        assert case.expected_keywords == ["Python", "programming"]
        assert case.expected_sources == 2
        assert case.max_response_time == 30.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        case = SearchQATestCase(
            id="sq_001",
            name="Test",
            query="Test query",
            expected_keywords=["keyword"],
        )
        
        d = case.to_dict()
        
        assert d["query"] == "Test query"
        assert d["expected_keywords"] == ["keyword"]


class TestLessonPackTestCase:
    """LessonPackTestCase测试"""
    
    def test_create_lesson_pack_case(self):
        """测试创建LessonPack测试用例"""
        case = LessonPackTestCase(
            id="lp_001",
            name="Lesson test",
            topic="Python basics",
            level="beginner",
            expected_sections=["intro", "examples"],
            min_content_length=500,
        )
        
        assert case.topic == "Python basics"
        assert case.level == "beginner"
        assert case.expected_sections == ["intro", "examples"]
        assert case.min_content_length == 500
    
    def test_to_dict(self):
        """测试转换为字典"""
        case = LessonPackTestCase(
            id="lp_001",
            name="Test",
            topic="Test topic",
            level="intermediate",
        )
        
        d = case.to_dict()
        
        assert d["topic"] == "Test topic"
        assert d["level"] == "intermediate"


class TestConversationTestCase:
    """ConversationTestCase测试"""
    
    def test_create_conversation_case(self):
        """测试创建对话测试用例"""
        case = ConversationTestCase(
            id="conv_001",
            name="Conversation test",
            turns=[
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "How are you?"},
            ],
            context_keywords=["hello"],
            max_turn_time=10.0,
        )
        
        assert len(case.turns) == 2
        assert case.context_keywords == ["hello"]
        assert case.max_turn_time == 10.0
    
    def test_to_dict(self):
        """测试转换为字典"""
        case = ConversationTestCase(
            id="conv_001",
            name="Test",
            turns=[{"role": "user", "content": "Hi"}],
        )
        
        d = case.to_dict()
        
        assert len(d["turns"]) == 1


class TestValidationResult:
    """ValidationResult测试"""
    
    def test_create_result(self):
        """测试创建验证结果"""
        result = ValidationResult(
            test_case_id="test_001",
            test_case_name="Test",
            passed=True,
            timestamp=datetime.now(),
            duration=1.5,
            response="Test response",
        )
        
        assert result.test_case_id == "test_001"
        assert result.passed is True
        assert result.duration == 1.5
    
    def test_to_dict(self):
        """测试转换为字典"""
        result = ValidationResult(
            test_case_id="test_001",
            test_case_name="Test",
            passed=True,
            timestamp=datetime.now(),
            duration=1.5,
            checks={"check1": True},
            metrics={"metric1": 0.5},
        )
        
        d = result.to_dict()
        
        assert d["test_case_id"] == "test_001"
        assert d["passed"] is True
        assert d["checks"]["check1"] is True
        assert d["metrics"]["metric1"] == 0.5
    
    def test_to_dict_truncates_long_response(self):
        """测试长响应截断"""
        long_response = "x" * 1000
        result = ValidationResult(
            test_case_id="test_001",
            test_case_name="Test",
            passed=True,
            timestamp=datetime.now(),
            duration=1.0,
            response=long_response,
        )
        
        d = result.to_dict()
        
        assert len(d["response"]) == 500


class TestE2EValidator:
    """E2EValidator测试"""
    
    def test_init(self):
        """测试初始化"""
        validator = E2EValidator()
        
        assert validator.generate_fn is None
        assert validator.search_qa_fn is None
        assert validator._results == []
    
    def test_init_with_functions(self):
        """测试带函数初始化"""
        gen_fn = MagicMock()
        search_fn = MagicMock()
        
        validator = E2EValidator(
            generate_fn=gen_fn,
            search_qa_fn=search_fn,
        )
        
        assert validator.generate_fn == gen_fn
        assert validator.search_qa_fn == search_fn
    
    def test_validate_search_qa_with_search_fn(self):
        """测试使用search_qa_fn验证SearchQA"""
        search_fn = MagicMock(return_value={
            "answer": "Paris is the capital of France."
        })
        
        validator = E2EValidator(search_qa_fn=search_fn)
        
        case = SearchQATestCase(
            id="sq_001",
            name="Capital test",
            query="What is the capital of France?",
            expected_keywords=["Paris", "France"],
            max_response_time=30.0,
        )
        
        result = validator.validate_search_qa(case)
        
        assert result.passed is True
        assert result.checks["keywords_found"] is True
        assert result.checks["response_not_empty"] is True
        search_fn.assert_called_once_with("What is the capital of France?")
    
    def test_validate_search_qa_with_generate_fn(self):
        """测试使用generate_fn验证SearchQA"""
        gen_fn = MagicMock(return_value="Paris is the capital of France.")
        
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = SearchQATestCase(
            id="sq_001",
            name="Capital test",
            query="What is the capital of France?",
            expected_keywords=["Paris"],
        )
        
        result = validator.validate_search_qa(case)
        
        assert result.passed is True
        gen_fn.assert_called_once()
    
    def test_validate_search_qa_no_function(self):
        """测试无函数时验证SearchQA"""
        validator = E2EValidator()
        
        case = SearchQATestCase(
            id="sq_001",
            name="Test",
            query="Test query",
        )
        
        result = validator.validate_search_qa(case)
        
        assert result.passed is False
        assert "No search_qa_fn or generate_fn" in result.error_message
    
    def test_validate_search_qa_timeout(self):
        """测试SearchQA超时"""
        import time
        
        def slow_fn(query):
            time.sleep(0.1)
            return "Response"
        
        validator = E2EValidator(generate_fn=slow_fn)
        
        case = SearchQATestCase(
            id="sq_001",
            name="Test",
            query="Test",
            max_response_time=0.05,  # 50ms
        )
        
        result = validator.validate_search_qa(case)
        
        assert result.checks["response_time_ok"] is False
    
    def test_validate_lesson_pack_with_lesson_fn(self):
        """测试使用lesson_pack_fn验证LessonPack"""
        lesson_fn = MagicMock(return_value={
            "content": "# Introduction\n\nThis is a lesson about Python.\n\n## Variables\n\nVariables store data.\n\n## Examples\n\nHere are some examples." * 10
        })
        
        validator = E2EValidator(lesson_pack_fn=lesson_fn)
        
        case = LessonPackTestCase(
            id="lp_001",
            name="Python lesson",
            topic="Python basics",
            level="beginner",
            expected_sections=["introduction", "variables"],
            min_content_length=100,
        )
        
        result = validator.validate_lesson_pack(case)
        
        assert result.passed is True
        assert result.checks["has_markdown_headers"] is True
        lesson_fn.assert_called_once_with("Python basics", "beginner")
    
    def test_validate_lesson_pack_with_generate_fn(self):
        """测试使用generate_fn验证LessonPack"""
        gen_fn = MagicMock(return_value="# Lesson\n\nContent here." * 50)
        
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = LessonPackTestCase(
            id="lp_001",
            name="Test",
            topic="Test topic",
            level="beginner",
            min_content_length=100,
        )
        
        result = validator.validate_lesson_pack(case)
        
        assert result.passed is True
        gen_fn.assert_called_once()
    
    def test_validate_lesson_pack_content_too_short(self):
        """测试LessonPack内容过短"""
        gen_fn = MagicMock(return_value="Short content")
        
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = LessonPackTestCase(
            id="lp_001",
            name="Test",
            topic="Test",
            min_content_length=1000,
        )
        
        result = validator.validate_lesson_pack(case)
        
        assert result.checks["content_length_ok"] is False
    
    def test_validate_multi_turn_with_chat_fn(self):
        """测试使用chat_fn验证多轮对话"""
        responses = ["Hello!", "Your name is Alice."]
        call_count = [0]
        
        def chat_fn(history):
            response = responses[call_count[0]]
            call_count[0] += 1
            return response
        
        validator = E2EValidator(chat_fn=chat_fn)
        
        case = ConversationTestCase(
            id="conv_001",
            name="Name test",
            turns=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "user", "content": "What is my name?"},
            ],
            context_keywords=["Alice"],
        )
        
        result = validator.validate_multi_turn(case)
        
        assert result.passed is True
        assert result.checks["context_maintained"] is True
    
    def test_validate_multi_turn_with_generate_fn(self):
        """测试使用generate_fn验证多轮对话"""
        gen_fn = MagicMock(return_value="Response with Alice mentioned")
        
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = ConversationTestCase(
            id="conv_001",
            name="Test",
            turns=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "user", "content": "What is my name?"},
            ],
            context_keywords=["Alice"],
        )
        
        result = validator.validate_multi_turn(case)
        
        assert result.passed is True
    
    def test_run_validation_suite(self):
        """测试运行验证套件"""
        gen_fn = MagicMock(return_value="# Response\n\nContent here." * 50)
        
        validator = E2EValidator(generate_fn=gen_fn)
        
        search_cases = [
            SearchQATestCase(id="sq_001", name="Test 1", query="Query 1"),
        ]
        lesson_cases = [
            LessonPackTestCase(id="lp_001", name="Test 2", topic="Topic", min_content_length=100),
        ]
        conv_cases = [
            ConversationTestCase(
                id="conv_001",
                name="Test 3",
                turns=[{"role": "user", "content": "Hi"}],
            ),
        ]
        
        results = validator.run_validation_suite(
            search_qa_cases=search_cases,
            lesson_pack_cases=lesson_cases,
            conversation_cases=conv_cases,
        )
        
        assert len(results) == 3
    
    def test_get_results(self):
        """测试获取结果"""
        gen_fn = MagicMock(return_value="Response")
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = SearchQATestCase(id="sq_001", name="Test", query="Query")
        validator.validate_search_qa(case)
        
        results = validator.get_results()
        
        assert len(results) == 1
    
    def test_get_summary(self):
        """测试获取摘要"""
        # Response needs to be > 10 chars to pass response_length_ok check
        gen_fn = MagicMock(return_value="This is a valid response with enough content.")
        validator = E2EValidator(generate_fn=gen_fn)
        
        case1 = SearchQATestCase(id="sq_001", name="Test 1", query="Query 1")
        case2 = SearchQATestCase(id="sq_002", name="Test 2", query="Query 2")
        
        validator.validate_search_qa(case1)
        validator.validate_search_qa(case2)
        
        summary = validator.get_summary()
        
        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["pass_rate"] == 100.0
    
    def test_get_summary_empty(self):
        """测试空结果摘要"""
        validator = E2EValidator()
        
        summary = validator.get_summary()
        
        assert summary["total"] == 0
        assert summary["pass_rate"] == 0.0
    
    def test_clear_results(self):
        """测试清除结果"""
        gen_fn = MagicMock(return_value="Response")
        validator = E2EValidator(generate_fn=gen_fn)
        
        case = SearchQATestCase(id="sq_001", name="Test", query="Query")
        validator.validate_search_qa(case)
        
        assert len(validator.get_results()) == 1
        
        validator.clear_results()
        
        assert len(validator.get_results()) == 0
    
    def test_load_test_cases(self):
        """测试加载测试用例"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({
                "search_qa": [
                    {"id": "sq_001", "name": "Test", "query": "Query"}
                ],
                "lesson_pack": [
                    {"id": "lp_001", "name": "Test", "topic": "Topic"}
                ],
            }, f)
            filepath = f.name
        
        try:
            cases = E2EValidator.load_test_cases(filepath)
            
            assert "search_qa" in cases
            assert "lesson_pack" in cases
            assert len(cases["search_qa"]) == 1
            assert cases["search_qa"][0].query == "Query"
        finally:
            os.unlink(filepath)
    
    def test_save_results(self):
        """测试保存结果"""
        results = [
            ValidationResult(
                test_case_id="test_001",
                test_case_name="Test",
                passed=True,
                timestamp=datetime.now(),
                duration=1.0,
            )
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            
            E2EValidator.save_results(results, filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath, "r") as f:
                data = json.load(f)
            
            assert len(data) == 1
            assert data[0]["test_case_id"] == "test_001"


class TestE2EValidatorIntegration:
    """E2EValidator集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        # 模拟生成函数
        def mock_generate(prompt):
            if "capital" in prompt.lower():
                return "Paris is the capital of France."
            elif "lesson" in prompt.lower():
                return "# Introduction\n\nThis is a lesson.\n\n## Content\n\nHere is the content." * 20
            else:
                return "This is a response."
        
        validator = E2EValidator(generate_fn=mock_generate)
        
        # 运行测试
        search_cases = [
            SearchQATestCase(
                id="sq_001",
                name="Capital test",
                query="What is the capital of France?",
                expected_keywords=["Paris"],
            )
        ]
        
        lesson_cases = [
            LessonPackTestCase(
                id="lp_001",
                name="Lesson test",
                topic="Test topic",
                min_content_length=100,
            )
        ]
        
        results = validator.run_validation_suite(
            search_qa_cases=search_cases,
            lesson_pack_cases=lesson_cases,
        )
        
        assert len(results) == 2
        
        # 获取摘要
        summary = validator.get_summary()
        assert summary["total"] == 2
        
        # 保存结果
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "results.json")
            E2EValidator.save_results(results, filepath)
            assert os.path.exists(filepath)
