"""
端到端验证器模块

提供SearchQA、LessonPack和多轮对话的E2E验证功能。
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """测试用例基类"""
    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class SearchQATestCase(TestCase):
    """SearchQA测试用例"""
    query: str = ""
    expected_keywords: List[str] = field(default_factory=list)
    expected_sources: int = 0  # 期望的引用源数量
    max_response_time: float = 30.0  # 最大响应时间（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = super().to_dict()
        d.update({
            "query": self.query,
            "expected_keywords": self.expected_keywords,
            "expected_sources": self.expected_sources,
            "max_response_time": self.max_response_time,
        })
        return d


@dataclass
class LessonPackTestCase(TestCase):
    """LessonPack测试用例"""
    topic: str = ""
    level: str = "beginner"  # beginner, intermediate, advanced
    expected_sections: List[str] = field(default_factory=list)
    min_content_length: int = 500
    max_response_time: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = super().to_dict()
        d.update({
            "topic": self.topic,
            "level": self.level,
            "expected_sections": self.expected_sections,
            "min_content_length": self.min_content_length,
            "max_response_time": self.max_response_time,
        })
        return d


@dataclass
class ConversationTestCase(TestCase):
    """多轮对话测试用例"""
    turns: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user", "content": "..."}]
    context_keywords: List[str] = field(default_factory=list)  # 后续回答应包含的上下文关键词
    max_turn_time: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = super().to_dict()
        d.update({
            "turns": self.turns,
            "context_keywords": self.context_keywords,
            "max_turn_time": self.max_turn_time,
        })
        return d


@dataclass
class ValidationResult:
    """验证结果"""
    test_case_id: str
    test_case_name: str
    passed: bool
    timestamp: datetime
    duration: float  # seconds
    
    # 详细结果
    response: str = ""
    error_message: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "test_case_id": self.test_case_id,
            "test_case_name": self.test_case_name,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "response": self.response[:500] if self.response else "",  # 截断长响应
            "error_message": self.error_message,
            "checks": self.checks,
            "metrics": self.metrics,
        }


class E2EValidator:
    """
    端到端验证器
    
    验证真实模型在各种场景下的功能正确性：
    - SearchQA场景
    - LessonPack场景
    - 多轮对话场景
    """
    
    def __init__(
        self,
        generate_fn: Optional[Callable[[str], str]] = None,
        search_qa_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        lesson_pack_fn: Optional[Callable[[str, str], Dict[str, Any]]] = None,
        chat_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ):
        """
        初始化验证器
        
        Args:
            generate_fn: 通用生成函数
            search_qa_fn: SearchQA工作流函数
            lesson_pack_fn: LessonPack工作流函数
            chat_fn: 多轮对话函数
        """
        self.generate_fn = generate_fn
        self.search_qa_fn = search_qa_fn
        self.lesson_pack_fn = lesson_pack_fn
        self.chat_fn = chat_fn
        
        self._results: List[ValidationResult] = []
    
    def validate_search_qa(
        self,
        test_case: SearchQATestCase
    ) -> ValidationResult:
        """
        验证SearchQA场景
        
        Args:
            test_case: SearchQA测试用例
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"Validating SearchQA: {test_case.name}")
        
        start_time = time.perf_counter()
        checks = {}
        metrics = {}
        response = ""
        error_message = ""
        
        try:
            if self.search_qa_fn:
                result = self.search_qa_fn(test_case.query)
                response = result.get("answer", "")
            elif self.generate_fn:
                response = self.generate_fn(test_case.query)
            else:
                raise ValueError("No search_qa_fn or generate_fn provided")
            
            duration = time.perf_counter() - start_time
            metrics["response_time"] = duration
            
            # 检查响应时间
            checks["response_time_ok"] = duration <= test_case.max_response_time
            
            # 检查关键词
            if test_case.expected_keywords:
                response_lower = response.lower()
                found_keywords = sum(
                    1 for kw in test_case.expected_keywords
                    if kw.lower() in response_lower
                )
                checks["keywords_found"] = found_keywords >= len(test_case.expected_keywords) // 2
                metrics["keywords_found_ratio"] = found_keywords / len(test_case.expected_keywords)
            else:
                checks["keywords_found"] = True
            
            # 检查响应非空
            checks["response_not_empty"] = len(response.strip()) > 0
            
            # 检查响应长度合理
            checks["response_length_ok"] = 10 < len(response) < 10000
            
            passed = all(checks.values())
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            error_message = str(e)
            passed = False
            logger.error(f"SearchQA validation failed: {e}")
        
        result = ValidationResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=passed,
            timestamp=datetime.now(),
            duration=duration,
            response=response,
            error_message=error_message,
            checks=checks,
            metrics=metrics,
        )
        
        self._results.append(result)
        return result
    
    def validate_lesson_pack(
        self,
        test_case: LessonPackTestCase
    ) -> ValidationResult:
        """
        验证LessonPack场景
        
        Args:
            test_case: LessonPack测试用例
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"Validating LessonPack: {test_case.name}")
        
        start_time = time.perf_counter()
        checks = {}
        metrics = {}
        response = ""
        error_message = ""
        
        try:
            if self.lesson_pack_fn:
                result = self.lesson_pack_fn(test_case.topic, test_case.level)
                response = result.get("content", "")
            elif self.generate_fn:
                prompt = f"Create a {test_case.level} level lesson about: {test_case.topic}"
                response = self.generate_fn(prompt)
            else:
                raise ValueError("No lesson_pack_fn or generate_fn provided")
            
            duration = time.perf_counter() - start_time
            metrics["response_time"] = duration
            
            # 检查响应时间
            checks["response_time_ok"] = duration <= test_case.max_response_time
            
            # 检查内容长度
            checks["content_length_ok"] = len(response) >= test_case.min_content_length
            metrics["content_length"] = len(response)
            
            # 检查Markdown格式
            checks["has_markdown_headers"] = bool(re.search(r'^#+\s', response, re.MULTILINE))
            
            # 检查期望的章节
            if test_case.expected_sections:
                response_lower = response.lower()
                found_sections = sum(
                    1 for section in test_case.expected_sections
                    if section.lower() in response_lower
                )
                checks["sections_found"] = found_sections >= len(test_case.expected_sections) // 2
                metrics["sections_found_ratio"] = found_sections / len(test_case.expected_sections)
            else:
                checks["sections_found"] = True
            
            # 检查响应非空
            checks["response_not_empty"] = len(response.strip()) > 0
            
            passed = all(checks.values())
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            error_message = str(e)
            passed = False
            logger.error(f"LessonPack validation failed: {e}")
        
        result = ValidationResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=passed,
            timestamp=datetime.now(),
            duration=duration,
            response=response,
            error_message=error_message,
            checks=checks,
            metrics=metrics,
        )
        
        self._results.append(result)
        return result
    
    def validate_multi_turn(
        self,
        test_case: ConversationTestCase
    ) -> ValidationResult:
        """
        验证多轮对话场景
        
        Args:
            test_case: 对话测试用例
            
        Returns:
            ValidationResult: 验证结果
        """
        logger.info(f"Validating multi-turn conversation: {test_case.name}")
        
        start_time = time.perf_counter()
        checks = {}
        metrics = {}
        responses = []
        error_message = ""
        
        try:
            history = []
            turn_times = []
            
            for i, turn in enumerate(test_case.turns):
                if turn["role"] == "user":
                    history.append(turn)
                    
                    turn_start = time.perf_counter()
                    
                    if self.chat_fn:
                        response = self.chat_fn(history)
                    elif self.generate_fn:
                        # 构建简单的对话提示
                        prompt = "\n".join(
                            f"{t['role']}: {t['content']}" for t in history
                        )
                        prompt += "\nassistant:"
                        response = self.generate_fn(prompt)
                    else:
                        raise ValueError("No chat_fn or generate_fn provided")
                    
                    turn_time = time.perf_counter() - turn_start
                    turn_times.append(turn_time)
                    
                    history.append({"role": "assistant", "content": response})
                    responses.append(response)
                    
                    # 检查单轮响应时间
                    checks[f"turn_{i}_time_ok"] = turn_time <= test_case.max_turn_time
            
            duration = time.perf_counter() - start_time
            metrics["total_time"] = duration
            metrics["avg_turn_time"] = sum(turn_times) / len(turn_times) if turn_times else 0
            
            # 检查上下文保持
            if test_case.context_keywords and responses:
                last_response_lower = responses[-1].lower()
                found_context = sum(
                    1 for kw in test_case.context_keywords
                    if kw.lower() in last_response_lower
                )
                checks["context_maintained"] = found_context > 0
                metrics["context_keywords_found"] = found_context
            else:
                checks["context_maintained"] = True
            
            # 检查所有响应非空
            checks["all_responses_not_empty"] = all(
                len(r.strip()) > 0 for r in responses
            )
            
            passed = all(checks.values())
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            error_message = str(e)
            passed = False
            logger.error(f"Multi-turn validation failed: {e}")
        
        result = ValidationResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=passed,
            timestamp=datetime.now(),
            duration=duration,
            response="\n---\n".join(responses) if responses else "",
            error_message=error_message,
            checks=checks,
            metrics=metrics,
        )
        
        self._results.append(result)
        return result
    
    def run_validation_suite(
        self,
        search_qa_cases: Optional[List[SearchQATestCase]] = None,
        lesson_pack_cases: Optional[List[LessonPackTestCase]] = None,
        conversation_cases: Optional[List[ConversationTestCase]] = None,
    ) -> List[ValidationResult]:
        """
        运行完整验证套件
        
        Args:
            search_qa_cases: SearchQA测试用例列表
            lesson_pack_cases: LessonPack测试用例列表
            conversation_cases: 对话测试用例列表
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        results = []
        
        if search_qa_cases:
            logger.info(f"Running {len(search_qa_cases)} SearchQA tests")
            for case in search_qa_cases:
                results.append(self.validate_search_qa(case))
        
        if lesson_pack_cases:
            logger.info(f"Running {len(lesson_pack_cases)} LessonPack tests")
            for case in lesson_pack_cases:
                results.append(self.validate_lesson_pack(case))
        
        if conversation_cases:
            logger.info(f"Running {len(conversation_cases)} conversation tests")
            for case in conversation_cases:
                results.append(self.validate_multi_turn(case))
        
        return results
    
    def get_results(self) -> List[ValidationResult]:
        """获取所有验证结果"""
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        passed = sum(1 for r in self._results if r.passed)
        failed = len(self._results) - passed
        
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self._results) * 100,
            "avg_duration": sum(r.duration for r in self._results) / len(self._results),
        }
    
    def clear_results(self) -> None:
        """清除验证结果"""
        self._results.clear()
    
    @staticmethod
    def load_test_cases(filepath: str) -> Dict[str, List[TestCase]]:
        """
        从JSON文件加载测试用例
        
        Args:
            filepath: JSON文件路径
            
        Returns:
            Dict[str, List[TestCase]]: 测试用例字典
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        result = {}
        
        if "search_qa" in data:
            result["search_qa"] = [
                SearchQATestCase(**case) for case in data["search_qa"]
            ]
        
        if "lesson_pack" in data:
            result["lesson_pack"] = [
                LessonPackTestCase(**case) for case in data["lesson_pack"]
            ]
        
        if "conversation" in data:
            result["conversation"] = [
                ConversationTestCase(**case) for case in data["conversation"]
            ]
        
        return result
    
    @staticmethod
    def save_results(
        results: List[ValidationResult],
        filepath: str
    ) -> None:
        """
        保存验证结果到JSON文件
        
        Args:
            results: 验证结果列表
            filepath: 输出文件路径
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                ensure_ascii=False
            )
