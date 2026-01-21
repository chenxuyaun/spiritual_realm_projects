"""
Router路由器属性测试

使用Hypothesis进行基于属性的测试，验证Router的正确性属性。

属性1: 工作流路由一致性
属性23: Router低置信度处理

验证需求: 1.1, 9.1-9.5
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from mm_orch.router import Router, RoutingRule, LOW_CONFIDENCE_THRESHOLD
from mm_orch.schemas import (
    UserRequest,
    WorkflowSelection,
    WorkflowType,
    IntentType
)
from mm_orch.exceptions import ValidationError


# 自定义策略：生成有效的查询文本
valid_query_strategy = st.text(
    min_size=1,
    max_size=500,
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'S', 'Z'),
        blacklist_characters='\x00'
    )
).filter(lambda x: x.strip())  # 确保不是纯空白


# 自定义策略：生成UserRequest
user_request_strategy = st.builds(
    UserRequest,
    query=valid_query_strategy,
    context=st.none() | st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.text(min_size=0, max_size=50),
        max_size=5
    ),
    session_id=st.none() | st.uuids().map(str),
    preferences=st.none() | st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.integers(), st.floats(allow_nan=False), st.text(max_size=20)),
        max_size=5
    )
)


# 自定义策略：生成置信度阈值
confidence_threshold_strategy = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False
)


class TestProperty1WorkflowRoutingConsistency:
    """
    属性1: 工作流路由一致性
    
    对于任何用户请求，Router路由器应该返回一个有效的WorkflowSelection对象，
    其中workflow_type必须是WorkflowType枚举中的一个值，
    且confidence值在0到1之间。
    
    **Validates: Requirements 1.1, 9.1, 9.2, 9.3**
    """
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_route_returns_valid_workflow_selection(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        对于任何用户请求，Router应该返回有效的WorkflowSelection对象
        """
        router = Router()
        
        result = router.route(request)
        
        # 验证返回类型
        assert isinstance(result, WorkflowSelection), \
            f"Expected WorkflowSelection, got {type(result)}"
        
        # 验证workflow_type是有效的枚举值
        assert isinstance(result.workflow_type, WorkflowType), \
            f"workflow_type should be WorkflowType enum, got {type(result.workflow_type)}"
        assert result.workflow_type in WorkflowType, \
            f"workflow_type {result.workflow_type} not in WorkflowType enum"
        
        # 验证confidence在[0, 1]范围内
        assert 0.0 <= result.confidence <= 1.0, \
            f"confidence {result.confidence} not in [0, 1]"
        
        # 验证parameters是字典
        assert isinstance(result.parameters, dict), \
            f"parameters should be dict, got {type(result.parameters)}"
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_route_parameters_contain_query(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        返回的参数应该包含原始查询
        """
        router = Router()
        
        result = router.route(request)
        
        # 参数应该包含query
        assert "query" in result.parameters, \
            "parameters should contain 'query'"
        assert result.parameters["query"] == request.query, \
            "parameters['query'] should match request.query"
    
    @given(
        request=user_request_strategy,
        threshold=confidence_threshold_strategy
    )
    @settings(max_examples=100)
    def test_route_with_different_thresholds(
        self,
        request: UserRequest,
        threshold: float
    ):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        无论置信度阈值如何设置，路由结果都应该有效
        """
        router = Router(confidence_threshold=threshold)
        
        result = router.route(request)
        
        # 结果应该始终有效
        assert isinstance(result, WorkflowSelection)
        assert isinstance(result.workflow_type, WorkflowType)
        assert 0.0 <= result.confidence <= 1.0
    
    @given(query=valid_query_strategy)
    @settings(max_examples=100)
    def test_classify_intent_returns_valid_intent(self, query: str):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        意图分类应该返回有效的IntentType
        """
        router = Router()
        
        intent = router.classify_intent(query)
        
        assert isinstance(intent, IntentType), \
            f"Expected IntentType, got {type(intent)}"
        assert intent in IntentType, \
            f"intent {intent} not in IntentType enum"


class TestProperty23LowConfidenceHandling:
    """
    属性23: Router低置信度处理
    
    对于任何路由决策，当confidence值低于阈值（如0.6）时，
    返回的WorkflowSelection应该包含非空的alternatives列表，
    列出其他可能的工作流选项。
    
    **Validates: Requirements 9.4**
    """
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_low_confidence_has_alternatives(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 23: Router低置信度处理
        
        当置信度低于阈值时，应该返回候选列表
        """
        router = Router()
        
        result = router.route(request)
        
        # 如果置信度低于阈值，应该有候选
        if result.confidence < router.confidence_threshold:
            assert result.alternatives is not None, \
                f"Low confidence ({result.confidence}) should have alternatives"
            assert len(result.alternatives) > 0, \
                "alternatives should not be empty when confidence is low"
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_alternatives_are_valid_selections(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 23: Router低置信度处理
        
        候选列表中的每个选项都应该是有效的WorkflowSelection
        """
        router = Router()
        
        result = router.route(request)
        
        if result.alternatives:
            for alt in result.alternatives:
                # 每个候选都应该是有效的WorkflowSelection
                assert isinstance(alt, WorkflowSelection), \
                    f"Alternative should be WorkflowSelection, got {type(alt)}"
                assert isinstance(alt.workflow_type, WorkflowType), \
                    f"Alternative workflow_type should be WorkflowType"
                assert 0.0 <= alt.confidence <= 1.0, \
                    f"Alternative confidence {alt.confidence} not in [0, 1]"
                assert isinstance(alt.parameters, dict), \
                    "Alternative parameters should be dict"
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_alternatives_exclude_best_choice(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 23: Router低置信度处理
        
        候选列表不应该包含最佳选择
        """
        router = Router()
        
        result = router.route(request)
        
        if result.alternatives:
            alt_types = [alt.workflow_type for alt in result.alternatives]
            assert result.workflow_type not in alt_types, \
                "alternatives should not include the best choice"
    
    @given(request=user_request_strategy)
    @settings(max_examples=100)
    def test_alternatives_have_lower_confidence(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 23: Router低置信度处理
        
        候选的置信度应该不高于最佳选择
        """
        router = Router()
        
        result = router.route(request)
        
        if result.alternatives:
            for alt in result.alternatives:
                assert alt.confidence <= result.confidence, \
                    f"Alternative confidence {alt.confidence} should not exceed best {result.confidence}"
    
    @given(
        request=user_request_strategy,
        threshold=st.floats(min_value=0.9, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=50)
    def test_high_threshold_more_alternatives(
        self,
        request: UserRequest,
        threshold: float
    ):
        """
        Feature: muai-orchestration-system, Property 23: Router低置信度处理
        
        高阈值应该导致更多的低置信度情况
        """
        router = Router(confidence_threshold=threshold)
        
        result = router.route(request)
        
        # 高阈值时，如果置信度低于阈值，应该有候选
        if result.confidence < threshold:
            assert result.alternatives is not None, \
                f"With high threshold {threshold}, low confidence {result.confidence} should have alternatives"


class TestRouterDeterminism:
    """
    测试Router的确定性行为
    
    相同的输入应该产生相同的输出
    """
    
    @given(request=user_request_strategy)
    @settings(max_examples=50)
    def test_route_is_deterministic(self, request: UserRequest):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        相同的请求应该产生相同的路由结果
        """
        router = Router()
        
        result1 = router.route(request)
        result2 = router.route(request)
        
        # 相同请求应该产生相同的工作流类型
        assert result1.workflow_type == result2.workflow_type, \
            "Same request should route to same workflow"
        
        # 置信度应该相同
        assert result1.confidence == result2.confidence, \
            "Same request should have same confidence"
    
    @given(query=valid_query_strategy)
    @settings(max_examples=50)
    def test_classify_intent_is_deterministic(self, query: str):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        相同的查询应该产生相同的意图分类
        """
        router = Router()
        
        intent1 = router.classify_intent(query)
        intent2 = router.classify_intent(query)
        
        assert intent1 == intent2, \
            "Same query should classify to same intent"


class TestRouterRobustness:
    """
    测试Router的健壮性
    
    Router应该能够处理各种边缘情况
    """
    
    @given(
        query=st.text(
            min_size=1,
            max_size=10000,
            alphabet=st.characters(blacklist_characters='\x00')
        ).filter(lambda x: x.strip())
    )
    @settings(max_examples=50)
    def test_handles_long_queries(self, query: str):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        Router应该能够处理长查询
        """
        router = Router()
        request = UserRequest(query=query)
        
        result = router.route(request)
        
        assert isinstance(result, WorkflowSelection)
        assert 0.0 <= result.confidence <= 1.0
    
    @given(
        query=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(
                whitelist_categories=('L', 'N', 'P', 'S', 'Z', 'M', 'C'),
                blacklist_characters='\x00'
            )
        ).filter(lambda x: x.strip())
    )
    @settings(max_examples=50)
    def test_handles_special_characters(self, query: str):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        Router应该能够处理特殊字符
        """
        router = Router()
        request = UserRequest(query=query)
        
        result = router.route(request)
        
        assert isinstance(result, WorkflowSelection)
        assert 0.0 <= result.confidence <= 1.0
    
    @given(
        context=st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(
                st.none(),
                st.booleans(),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.text(max_size=100),
                st.lists(st.text(max_size=20), max_size=5)
            ),
            max_size=10
        )
    )
    @settings(max_examples=50)
    def test_handles_various_contexts(self, context: dict):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        Router应该能够处理各种上下文
        """
        router = Router()
        request = UserRequest(query="测试查询", context=context)
        
        result = router.route(request)
        
        assert isinstance(result, WorkflowSelection)
        assert 0.0 <= result.confidence <= 1.0


class TestRouterWorkflowCoverage:
    """
    测试Router的工作流覆盖
    
    确保所有工作流类型都可以被路由到
    """
    
    def test_all_workflow_types_reachable(self):
        """
        Feature: muai-orchestration-system, Property 1: 工作流路由一致性
        
        所有工作流类型都应该可以通过某些查询到达
        """
        router = Router()
        
        # 为每种工作流类型准备测试查询
        test_queries = {
            WorkflowType.SEARCH_QA: "搜索最新的Python教程",
            WorkflowType.LESSON_PACK: "教我机器学习基础",
            WorkflowType.CHAT_GENERATE: "你好",
            WorkflowType.RAG_QA: "根据文档回答这个问题",
            WorkflowType.SELF_ASK_SEARCH_QA: "比较Python和Java的优缺点"
        }
        
        reached_types = set()
        
        for expected_type, query in test_queries.items():
            request = UserRequest(query=query)
            result = router.route(request)
            reached_types.add(result.workflow_type)
        
        # 验证所有工作流类型都被覆盖
        all_types = set(WorkflowType)
        assert reached_types == all_types, \
            f"Not all workflow types reachable. Missing: {all_types - reached_types}"
