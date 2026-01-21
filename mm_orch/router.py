"""
Router路由器 - 智能工作流选择

Router负责分析用户请求并选择最合适的工作流。
实现基于规则的分类器（关键词匹配、正则表达式），
支持置信度评分和低置信度时返回多个候选。

需求: 1.1, 9.1, 9.2, 9.3, 9.4, 9.5
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from mm_orch.schemas import UserRequest, WorkflowSelection, WorkflowType, IntentType
from mm_orch.logger import get_logger
from mm_orch.exceptions import ValidationError


logger = get_logger(__name__)


# 低置信度阈值 - 低于此值时返回候选列表
LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class RoutingRule:
    """路由规则定义"""

    workflow_type: WorkflowType
    intent_type: IntentType
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    base_weight: float = 1.0
    description: str = ""


class Router:
    """
    Router路由器 - 分析用户请求并选择最合适的工作流

    实现策略:
    - 基于规则的分类器（关键词匹配、正则表达式）
    - 置信度评分机制
    - 低置信度时返回多个候选
    - 路由决策日志记录

    属性1: 工作流路由一致性
    对于任何用户请求，Router应该返回一个有效的WorkflowSelection对象，
    其中workflow_type必须是WorkflowType枚举中的一个值，
    且confidence值在0到1之间。

    属性23: Router低置信度处理
    当confidence值低于阈值（0.6）时，返回的WorkflowSelection
    应该包含非空的alternatives列表。
    """

    def __init__(self, confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD):
        """
        初始化Router

        Args:
            confidence_threshold: 低置信度阈值，低于此值时返回候选列表
        """
        self.confidence_threshold = confidence_threshold
        self._rules = self._initialize_rules()
        self._compiled_patterns: Dict[WorkflowType, List[re.Pattern]] = {}
        self._compile_patterns()

        logger.info(
            "Router initialized",
            confidence_threshold=confidence_threshold,
            num_rules=len(self._rules),
        )

    def _initialize_rules(self) -> List[RoutingRule]:
        """初始化路由规则"""
        return [
            # SearchQA - 搜索问答
            RoutingRule(
                workflow_type=WorkflowType.SEARCH_QA,
                intent_type=IntentType.QUESTION_ANSWERING,
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
            # LessonPack - 教学内容生成
            RoutingRule(
                workflow_type=WorkflowType.LESSON_PACK,
                intent_type=IntentType.TEACHING,
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
                base_weight=1.2,  # 教学请求通常更明确
                description="Teaching content generation",
            ),
            # ChatGenerate - 多轮对话
            RoutingRule(
                workflow_type=WorkflowType.CHAT_GENERATE,
                intent_type=IntentType.CONVERSATION,
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
                base_weight=0.8,  # 对话是默认回退
                description="Multi-turn conversation",
            ),
            # RAGQA - 知识库问答
            RoutingRule(
                workflow_type=WorkflowType.RAG_QA,
                intent_type=IntentType.KNOWLEDGE_QUERY,
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
            # SelfAskSearchQA - 复杂问题分解
            RoutingRule(
                workflow_type=WorkflowType.SELF_ASK_SEARCH_QA,
                intent_type=IntentType.COMPLEX_REASONING,
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
                base_weight=1.3,  # 复杂问题权重更高
                description="Complex question decomposition and search",
            ),
        ]

    def _compile_patterns(self) -> None:
        """预编译正则表达式模式"""
        for rule in self._rules:
            patterns = []
            for pattern in rule.patterns:
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                    patterns.append(compiled)
                except re.error as e:
                    logger.warning("Invalid regex pattern", pattern=pattern, error=str(e))
            self._compiled_patterns[rule.workflow_type] = patterns

    def route(self, request: UserRequest) -> WorkflowSelection:
        """
        分析请求并返回工作流选择

        Args:
            request: 用户请求对象，包含query、context等

        Returns:
            WorkflowSelection: 包含workflow_type、confidence、parameters

        Raises:
            ValidationError: 如果请求无效
        """
        # 验证请求
        if not request or not request.query:
            raise ValidationError("Request query cannot be empty")

        query = request.query.strip()

        # 计算每个工作流的得分
        scores = self._calculate_scores(query, request.context)

        # 按得分排序
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 获取最高得分的工作流
        best_workflow, best_score = sorted_scores[0]

        # 归一化置信度到 [0, 1]
        confidence = self._normalize_confidence(best_score, scores)

        # 构建参数
        parameters = self._build_parameters(request, best_workflow)

        # 构建候选列表（如果置信度低）
        alternatives = None
        if confidence < self.confidence_threshold:
            alternatives = self._build_alternatives(
                sorted_scores[1:], scores, request  # 排除最佳选择
            )

        # 创建选择结果
        selection = WorkflowSelection(
            workflow_type=best_workflow,
            confidence=confidence,
            parameters=parameters,
            alternatives=alternatives,
        )

        # 记录路由决策
        self._log_routing_decision(request, selection, scores)

        return selection

    def classify_intent(self, query: str) -> IntentType:
        """
        分类用户意图

        Args:
            query: 用户查询文本

        Returns:
            IntentType: 意图类型枚举
        """
        if not query:
            return IntentType.CONVERSATION

        query = query.strip().lower()

        # 计算每个意图的得分
        intent_scores: Dict[IntentType, float] = {}

        for rule in self._rules:
            score = self._calculate_rule_score(query, rule)
            intent_type = rule.intent_type

            if intent_type not in intent_scores:
                intent_scores[intent_type] = 0.0
            intent_scores[intent_type] = max(intent_scores[intent_type], score)

        # 返回得分最高的意图
        if not intent_scores:
            return IntentType.CONVERSATION

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0]

    def _calculate_scores(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[WorkflowType, float]:
        """
        计算每个工作流的得分

        Args:
            query: 用户查询
            context: 可选的上下文信息

        Returns:
            Dict[WorkflowType, float]: 工作流类型到得分的映射
        """
        scores: Dict[WorkflowType, float] = {}
        query_lower = query.lower()

        for rule in self._rules:
            score = self._calculate_rule_score(query_lower, rule)

            # 应用上下文调整
            if context:
                score = self._apply_context_adjustment(score, rule, context)

            scores[rule.workflow_type] = score

        # 确保所有工作流类型都有得分
        for wf_type in WorkflowType:
            if wf_type not in scores:
                scores[wf_type] = 0.1  # 最小基础分

        return scores

    def _calculate_rule_score(self, query: str, rule: RoutingRule) -> float:
        """
        计算单个规则的得分

        Args:
            query: 小写的查询文本
            rule: 路由规则

        Returns:
            float: 规则得分
        """
        score = 0.0

        # 关键词匹配得分
        keyword_matches = 0
        for keyword in rule.keywords:
            if keyword.lower() in query:
                keyword_matches += 1

        if keyword_matches > 0:
            # 关键词匹配贡献
            keyword_score = min(keyword_matches * 0.3, 1.0)
            score += keyword_score

        # 正则表达式匹配得分
        patterns = self._compiled_patterns.get(rule.workflow_type, [])
        pattern_matches = 0
        for pattern in patterns:
            if pattern.search(query):
                pattern_matches += 1

        if pattern_matches > 0:
            # 模式匹配贡献（权重更高）
            pattern_score = min(pattern_matches * 0.4, 1.2)
            score += pattern_score

        # 应用基础权重
        score *= rule.base_weight

        return score

    def _apply_context_adjustment(
        self, score: float, rule: RoutingRule, context: Dict[str, Any]
    ) -> float:
        """
        根据上下文调整得分

        Args:
            score: 原始得分
            rule: 路由规则
            context: 上下文信息

        Returns:
            float: 调整后的得分
        """
        adjusted_score = score

        # 如果上下文指定了首选工作流
        preferred_workflow = context.get("preferred_workflow")
        if preferred_workflow:
            if rule.workflow_type.value == preferred_workflow:
                adjusted_score *= 1.5

        # 如果有会话ID，倾向于对话工作流
        if context.get("session_id"):
            if rule.workflow_type == WorkflowType.CHAT_GENERATE:
                adjusted_score *= 1.2

        # 如果指定了知识库，倾向于RAG
        if context.get("knowledge_base") or context.get("documents"):
            if rule.workflow_type == WorkflowType.RAG_QA:
                adjusted_score *= 1.3

        return adjusted_score

    def _normalize_confidence(
        self, best_score: float, all_scores: Dict[WorkflowType, float]
    ) -> float:
        """
        归一化置信度到 [0, 1] 范围

        Args:
            best_score: 最高得分
            all_scores: 所有得分

        Returns:
            float: 归一化的置信度
        """
        if best_score <= 0:
            return 0.1  # 最小置信度

        # 计算总分
        total_score = sum(all_scores.values())
        if total_score <= 0:
            return 0.1

        # 相对置信度
        relative_confidence = best_score / total_score

        # 绝对置信度（基于得分本身）
        # 得分 >= 1.0 时置信度较高
        absolute_confidence = min(best_score / 1.5, 1.0)

        # 综合置信度
        confidence = relative_confidence * 0.6 + absolute_confidence * 0.4

        # 确保在 [0, 1] 范围内
        return max(0.0, min(1.0, confidence))

    def _build_parameters(
        self, request: UserRequest, workflow_type: WorkflowType
    ) -> Dict[str, Any]:
        """
        构建工作流参数

        Args:
            request: 用户请求
            workflow_type: 选择的工作流类型

        Returns:
            Dict[str, Any]: 工作流参数
        """
        parameters: Dict[str, Any] = {"query": request.query}

        # 添加会话ID（如果有）
        if request.session_id:
            parameters["session_id"] = request.session_id

        # 添加上下文（如果有）
        if request.context:
            parameters["context"] = request.context

        # 添加用户偏好（如果有）
        if request.preferences:
            parameters["preferences"] = request.preferences

        # 根据工作流类型添加特定参数
        if workflow_type == WorkflowType.LESSON_PACK:
            parameters["topic"] = request.query
        elif workflow_type == WorkflowType.CHAT_GENERATE:
            parameters["message"] = request.query
        elif workflow_type == WorkflowType.RAG_QA:
            # RAG可能需要top_k参数
            parameters["top_k"] = request.preferences.get("top_k", 5) if request.preferences else 5

        return parameters

    def _build_alternatives(
        self,
        sorted_scores: List[Tuple[WorkflowType, float]],
        all_scores: Dict[WorkflowType, float],
        request: UserRequest,
        max_alternatives: int = 3,
    ) -> Optional[List[WorkflowSelection]]:
        """
        构建候选工作流列表

        Args:
            sorted_scores: 排序后的得分列表（不含最佳选择）
            all_scores: 所有得分
            request: 用户请求
            max_alternatives: 最大候选数量

        Returns:
            List[WorkflowSelection]: 候选工作流列表，如果没有有效候选则返回None
        """
        alternatives = []

        for workflow_type, score in sorted_scores[:max_alternatives]:
            # 即使得分为0，也可以作为候选（当所有得分都很低时）
            confidence = self._normalize_confidence(score, all_scores)
            parameters = self._build_parameters(request, workflow_type)

            alternative = WorkflowSelection(
                workflow_type=workflow_type,
                confidence=confidence,
                parameters=parameters,
                alternatives=None,  # 候选不再嵌套候选
            )
            alternatives.append(alternative)

        return alternatives if alternatives else None

    def _log_routing_decision(
        self, request: UserRequest, selection: WorkflowSelection, scores: Dict[WorkflowType, float]
    ) -> None:
        """
        记录路由决策日志

        Args:
            request: 用户请求
            selection: 路由选择结果
            scores: 所有工作流得分
        """
        # 格式化得分用于日志
        score_summary = {wf_type.value: round(score, 3) for wf_type, score in scores.items()}

        log_data = {
            "query_preview": request.query[:100] if len(request.query) > 100 else request.query,
            "selected_workflow": selection.workflow_type.value,
            "confidence": round(selection.confidence, 3),
            "scores": score_summary,
            "has_alternatives": selection.alternatives is not None,
            "num_alternatives": len(selection.alternatives) if selection.alternatives else 0,
        }

        if selection.confidence < self.confidence_threshold:
            logger.info("Low confidence routing decision", **log_data)
        else:
            logger.debug("Routing decision", **log_data)

    def get_workflow_description(self, workflow_type: WorkflowType) -> str:
        """
        获取工作流描述

        Args:
            workflow_type: 工作流类型

        Returns:
            str: 工作流描述
        """
        for rule in self._rules:
            if rule.workflow_type == workflow_type:
                return rule.description
        return "Unknown workflow"

    def add_rule(self, rule: RoutingRule) -> None:
        """
        添加新的路由规则

        Args:
            rule: 路由规则
        """
        self._rules.append(rule)

        # 编译新规则的模式
        patterns = []
        for pattern in rule.patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append(compiled)
            except re.error as e:
                logger.warning("Invalid regex pattern in new rule", pattern=pattern, error=str(e))
        self._compiled_patterns[rule.workflow_type] = patterns

        logger.info(
            "Added new routing rule",
            workflow_type=rule.workflow_type.value,
            num_keywords=len(rule.keywords),
            num_patterns=len(rule.patterns),
        )

    def get_rules(self) -> List[RoutingRule]:
        """
        获取所有路由规则

        Returns:
            List[RoutingRule]: 路由规则列表
        """
        return self._rules.copy()

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        设置置信度阈值

        Args:
            threshold: 新的阈值（0.0 到 1.0）

        Raises:
            ValueError: 如果阈值不在有效范围内
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.confidence_threshold = threshold
        logger.info("Updated confidence threshold", new_threshold=threshold)


# 便捷函数
def create_router(confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD) -> Router:
    """
    创建Router实例的工厂函数

    Args:
        confidence_threshold: 置信度阈值

    Returns:
        Router: Router实例
    """
    return Router(confidence_threshold=confidence_threshold)


# 默认Router实例（单例模式）
_default_router: Optional[Router] = None


def get_router() -> Router:
    """
    获取默认Router实例（单例）

    Returns:
        Router: 默认Router实例
    """
    global _default_router
    if _default_router is None:
        _default_router = Router()
    return _default_router


def reset_router() -> None:
    """重置默认Router实例"""
    global _default_router
    _default_router = None
