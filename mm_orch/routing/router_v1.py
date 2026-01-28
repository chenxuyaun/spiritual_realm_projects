"""
Router v1: Rule-based workflow selection.

This router uses keyword matching and regex patterns to select workflows.
It provides confidence scores and ranked candidates for routing decisions.

Requirements: 14.1, 14.2, 14.3, 14.4
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

from mm_orch.orchestration.state import State
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RoutingRule:
    """Rule definition for workflow routing."""

    workflow_name: str
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    base_weight: float = 1.0
    description: str = ""


class RouterV1:
    """
    Rule-based router using keyword matching and regex patterns.

    This router evaluates user questions against a set of rules to determine
    the most appropriate workflow. Each rule contains keywords and patterns
    that match specific types of questions.

    The router returns:
    - workflow_name: The selected workflow
    - confidence: Score between 0 and 1
    - candidates: Ranked list of alternative workflows

    Example:
        router = RouterV1()
        workflow, confidence, candidates = router.route("搜索最新的Python教程", state)
        # Returns: ("search_qa", 0.85, [("search_qa_fast", 0.65), ...])
    """

    def __init__(self):
        """Initialize router with default rules."""
        self._rules = self._initialize_rules()
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

        logger.info("RouterV1 initialized", num_rules=len(self._rules))

    def _initialize_rules(self) -> List[RoutingRule]:
        """
        Initialize default routing rules.

        Returns:
            List of routing rules for different workflows
        """
        return [
            # search_qa - Web search based Q&A
            RoutingRule(
                workflow_name="search_qa",
                keywords=[
                    "搜索",
                    "查找",
                    "查询",
                    "search",
                    "find",
                    "look up",
                    "最新",
                    "latest",
                    "recent",
                    "news",
                    "新闻",
                    "什么是",
                    "what is",
                    "who is",
                    "谁是",
                    "怎么样",
                    "how about",
                    "how is",
                    "在哪里",
                    "where",
                    "哪里",
                ],
                patterns=[
                    r".*搜索.*",
                    r".*查一下.*",
                    r".*帮我找.*",
                    r".*search\s+for.*",
                    r".*look\s+up.*",
                    r".*最新的?.*是什么.*",
                    r".*现在.*怎么样.*",
                ],
                base_weight=1.0,
                description="Web search based Q&A",
            ),
            # search_qa_fast - Fast search without summarization
            RoutingRule(
                workflow_name="search_qa_fast",
                keywords=["快速", "简单", "fast", "quick", "简要", "brief"],
                patterns=[
                    r".*快速.*回答.*",
                    r".*简单.*说.*",
                    r".*quick\s+answer.*",
                    r".*briefly.*",
                ],
                base_weight=0.9,
                description="Fast search without summarization",
            ),
            # search_qa_strict_citations - Search with citation validation
            RoutingRule(
                workflow_name="search_qa_strict_citations",
                keywords=["引用", "来源", "citation", "source", "参考", "reference"],
                patterns=[
                    r".*需要.*引用.*",
                    r".*标注.*来源.*",
                    r".*with\s+citations.*",
                    r".*cite\s+sources.*",
                ],
                base_weight=1.1,
                description="Search with strict citation validation",
            ),
            # summarize_url - Summarize single URL
            RoutingRule(
                workflow_name="summarize_url",
                keywords=["总结", "摘要", "summarize", "summary", "概括"],
                patterns=[
                    r".*总结.*网页.*",
                    r".*摘要.*文章.*",
                    r".*summarize\s+this.*",
                    r".*summary\s+of.*",
                ],
                base_weight=1.0,
                description="Summarize single URL",
            ),
            # lesson_pack - Teaching content generation
            RoutingRule(
                workflow_name="lesson_pack",
                keywords=[
                    "教学",
                    "课程",
                    "教案",
                    "讲解",
                    "teach",
                    "lesson",
                    "学习",
                    "learn",
                    "教我",
                    "teach me",
                    "练习",
                    "exercise",
                    "习题",
                    "作业",
                    "课件",
                    "教材",
                    "知识点",
                    "概念",
                ],
                patterns=[
                    r".*教我.*",
                    r".*讲解.*",
                    r".*生成.*教案.*",
                    r".*创建.*课程.*",
                    r".*teach\s+me.*",
                    r".*explain.*to\s+me.*",
                    r".*generate.*lesson.*",
                    r".*学习.*怎么.*",
                ],
                base_weight=1.2,
                description="Teaching content generation",
            ),
            # chat_generate - Multi-turn conversation
            RoutingRule(
                workflow_name="chat_generate",
                keywords=[
                    "聊天",
                    "对话",
                    "chat",
                    "talk",
                    "conversation",
                    "你好",
                    "hello",
                    "hi",
                    "嗨",
                    "谢谢",
                    "thanks",
                    "thank you",
                    "再见",
                    "bye",
                    "goodbye",
                ],
                patterns=[
                    r"^你好.*",
                    r"^hi\b.*",
                    r"^hello\b.*",
                    r".*聊聊.*",
                    r".*说说.*",
                    r".*let's\s+talk.*",
                    r".*chat\s+with.*",
                ],
                base_weight=0.8,
                description="Multi-turn conversation",
            ),
            # rag_qa - Knowledge base Q&A
            RoutingRule(
                workflow_name="rag_qa",
                keywords=[
                    "知识库",
                    "文档",
                    "document",
                    "knowledge base",
                    "根据",
                    "based on",
                    "according to",
                    "资料",
                    "材料",
                    "reference",
                    "本地",
                    "local",
                    "内部",
                    "internal",
                ],
                patterns=[
                    r".*根据.*文档.*",
                    r".*在.*知识库.*中.*",
                    r".*from\s+the\s+document.*",
                    r".*based\s+on.*knowledge.*",
                    r".*查阅.*资料.*",
                ],
                base_weight=1.1,
                description="RAG-based knowledge Q&A",
            ),
            # self_ask_search_qa - Complex question decomposition
            RoutingRule(
                workflow_name="self_ask_search_qa",
                keywords=[
                    "比较",
                    "对比",
                    "compare",
                    "comparison",
                    "分析",
                    "analyze",
                    "analysis",
                    "综合",
                    "comprehensive",
                    "详细",
                    "多个",
                    "multiple",
                    "several",
                    "关系",
                    "relationship",
                    "联系",
                ],
                patterns=[
                    r".*比较.*和.*",
                    r".*对比.*与.*",
                    r".*compare.*and.*",
                    r".*分析.*原因.*",
                    r".*综合.*考虑.*",
                    r".*详细.*解释.*为什么.*",
                ],
                base_weight=1.3,
                description="Complex question decomposition",
            ),
        ]

    def _compile_patterns(self) -> None:
        """Precompile regex patterns for efficiency."""
        for rule in self._rules:
            patterns = []
            for pattern in rule.patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    patterns.append(compiled)
                except re.error as e:
                    logger.warning("Invalid regex pattern", pattern=pattern, error=str(e))
            self._compiled_patterns[rule.workflow_name] = patterns

    def route(self, question: str, state: State) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Route question to appropriate workflow.

        Args:
            question: User's question text
            state: Current workflow state (may contain context)

        Returns:
            Tuple of (workflow_name, confidence, candidates) where:
            - workflow_name: Selected workflow
            - confidence: Confidence score (0.0 to 1.0)
            - candidates: List of (workflow_name, score) tuples, sorted by score

        Example:
            workflow, conf, candidates = router.route("搜索Python", state)
            # Returns: ("search_qa", 0.85, [("search_qa_fast", 0.65), ...])
        """
        if not question or not question.strip():
            # Default to chat for empty questions
            return "chat_generate", 0.5, [("search_qa", 0.3)]

        question = question.strip()

        # Calculate scores for all workflows
        scores = self._calculate_scores(question, state)

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get best workflow
        best_workflow, best_score = sorted_scores[0]

        # Normalize confidence
        confidence = self._normalize_confidence(best_score, scores)

        # Build candidates list (all workflows sorted by score)
        candidates = [
            (wf, self._normalize_confidence(score, scores)) for wf, score in sorted_scores
        ]

        # Log decision
        self._log_decision(question, best_workflow, confidence, scores)

        return best_workflow, confidence, candidates

    def _calculate_scores(self, question: str, state: State) -> Dict[str, float]:
        """
        Calculate scores for all workflows.

        Args:
            question: User's question
            state: Current state (may contain context)

        Returns:
            Dictionary mapping workflow names to scores
        """
        scores: Dict[str, float] = {}
        question_lower = question.lower()

        for rule in self._rules:
            score = self._calculate_rule_score(question_lower, rule)

            # Apply context adjustments
            score = self._apply_context_adjustment(score, rule, state)

            scores[rule.workflow_name] = score

        # Ensure minimum score for all workflows
        for rule in self._rules:
            if rule.workflow_name not in scores:
                scores[rule.workflow_name] = 0.1

        return scores

    def _calculate_rule_score(self, question: str, rule: RoutingRule) -> float:
        """
        Calculate score for a single rule.

        Args:
            question: Lowercase question text
            rule: Routing rule to evaluate

        Returns:
            Score for this rule
        """
        score = 0.0

        # Keyword matching
        keyword_matches = 0
        for keyword in rule.keywords:
            if keyword.lower() in question:
                keyword_matches += 1

        if keyword_matches > 0:
            # Each keyword match contributes 0.3, capped at 1.0
            keyword_score = min(keyword_matches * 0.3, 1.0)
            score += keyword_score

        # Pattern matching
        patterns = self._compiled_patterns.get(rule.workflow_name, [])
        pattern_matches = 0
        for pattern in patterns:
            if pattern.search(question):
                pattern_matches += 1

        if pattern_matches > 0:
            # Each pattern match contributes 0.4, capped at 1.2
            pattern_score = min(pattern_matches * 0.4, 1.2)
            score += pattern_score

        # Apply base weight
        score *= rule.base_weight

        return score

    def _apply_context_adjustment(self, score: float, rule: RoutingRule, state: State) -> float:
        """
        Adjust score based on state context.

        Args:
            score: Base score
            rule: Routing rule
            state: Current state

        Returns:
            Adjusted score
        """
        adjusted_score = score

        # Get metadata
        meta = state.get("meta", {})

        # If mode is chat, boost chat_generate
        mode = meta.get("mode", "default")
        if mode == "chat" and rule.workflow_name == "chat_generate":
            adjusted_score *= 1.3

        # If conversation_id exists, boost chat workflows
        if state.get("conversation_id"):
            if rule.workflow_name in ["chat_generate", "lesson_pack"]:
                adjusted_score *= 1.2

        # If kb_sources exist, boost rag_qa
        if state.get("kb_sources"):
            if rule.workflow_name == "rag_qa":
                adjusted_score *= 1.4

        return adjusted_score

    def _normalize_confidence(self, score: float, all_scores: Dict[str, float]) -> float:
        """
        Normalize score to confidence in [0, 1].

        Args:
            score: Raw score
            all_scores: All workflow scores

        Returns:
            Normalized confidence between 0 and 1
        """
        if score <= 0:
            return 0.1  # Minimum confidence

        # Calculate total score
        total_score = sum(all_scores.values())
        if total_score <= 0:
            return 0.1

        # Relative confidence (score vs total)
        relative_confidence = score / total_score

        # Absolute confidence (score itself, normalized)
        # Score >= 1.5 gives high confidence
        absolute_confidence = min(score / 1.5, 1.0)

        # Combine: 60% relative, 40% absolute
        confidence = relative_confidence * 0.6 + absolute_confidence * 0.4

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _log_decision(
        self, question: str, workflow: str, confidence: float, scores: Dict[str, float]
    ) -> None:
        """
        Log routing decision.

        Args:
            question: User's question
            workflow: Selected workflow
            confidence: Confidence score
            scores: All workflow scores
        """
        # Format scores for logging
        score_summary = {wf: round(score, 3) for wf, score in scores.items()}

        # Get matched rules
        matched_rules = []
        for rule in self._rules:
            if rule.workflow_name == workflow:
                matched_rules.append(rule.description)

        logger.info(
            "RouterV1 decision",
            question_preview=question[:100],
            selected_workflow=workflow,
            confidence=round(confidence, 3),
            scores=score_summary,
            matched_rules=matched_rules,
        )

    def add_rule(self, rule: RoutingRule) -> None:
        """
        Add a new routing rule.

        Args:
            rule: Routing rule to add
        """
        self._rules.append(rule)

        # Compile patterns for new rule
        patterns = []
        for pattern in rule.patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append(compiled)
            except re.error as e:
                logger.warning("Invalid regex pattern in new rule", pattern=pattern, error=str(e))
        self._compiled_patterns[rule.workflow_name] = patterns

        logger.info(
            "Added routing rule",
            workflow=rule.workflow_name,
            num_keywords=len(rule.keywords),
            num_patterns=len(rule.patterns),
        )

    def get_rules(self) -> List[RoutingRule]:
        """
        Get all routing rules.

        Returns:
            List of routing rules
        """
        return self._rules.copy()
