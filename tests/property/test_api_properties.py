"""
API属性测试

使用Hypothesis进行基于属性的测试，验证API的正确性属性。

属性34: API JSON格式
属性35: API错误响应格式
属性36: API认证保护

验证需求: 13.1-13.4
"""

import json
import time
from typing import Any, Dict, Optional

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from fastapi.testclient import TestClient

from mm_orch.api.app import create_app, reset_app
from mm_orch.api.auth import (
    APIKeyAuth,
    get_auth_manager,
    set_auth_manager,
    reset_auth_manager
)
from mm_orch.api.schemas import (
    QueryRequest,
    ChatRequest,
    RAGUploadRequest,
    RAGQueryRequest,
    LessonPackRequest
)


# ============ 测试夹具 ============

@pytest.fixture
def app():
    """创建测试应用"""
    reset_app()
    reset_auth_manager()
    
    # 创建带有测试API密钥的应用
    test_api_key = "test_api_key_12345678901234567890"
    app = create_app(
        auth_enabled=True,
        api_keys={test_api_key}
    )
    
    yield app
    
    reset_app()
    reset_auth_manager()


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def api_key():
    """测试API密钥"""
    return "test_api_key_12345678901234567890"


@pytest.fixture
def auth_headers(api_key):
    """认证请求头"""
    return {"X-API-Key": api_key}


@pytest.fixture
def app_no_auth():
    """创建无认证的测试应用"""
    reset_app()
    reset_auth_manager()
    
    app = create_app(auth_enabled=False)
    
    yield app
    
    reset_app()
    reset_auth_manager()


@pytest.fixture
def client_no_auth(app_no_auth):
    """无认证的测试客户端"""
    return TestClient(app_no_auth)


# ============ 策略定义 ============

# 有效查询文本策略
valid_query_strategy = st.text(
    min_size=1,
    max_size=500,
    alphabet=st.characters(
        whitelist_categories=('L', 'N', 'P', 'Z'),
        whitelist_characters=' '
    )
).filter(lambda x: x.strip())

# 会话ID策略
session_id_strategy = st.one_of(
    st.none(),
    st.uuids().map(str)
)

# 上下文策略
context_strategy = st.one_of(
    st.none(),
    st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        values=st.text(max_size=100),
        max_size=5
    )
)

# API密钥策略
api_key_strategy = st.text(min_size=16, max_size=64, alphabet=st.characters(
    whitelist_categories=('L', 'N')
))


# ============ 属性34: API JSON格式 ============

class TestAPIJSONFormat:
    """
    属性34: API JSON格式
    
    对于任何API请求和响应，数据应该使用JSON格式编码，
    且Content-Type头应该为'application/json'。
    
    验证需求: 13.2
    """
    
    def test_health_endpoint_returns_json(self, client):
        """
        Feature: muai-orchestration-system, Property 34: API JSON格式
        
        健康检查端点应返回有效的JSON响应
        """
        response = client.get("/api/health")
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # 验证响应是有效的JSON
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert "timestamp" in data
    
    @given(query=valid_query_strategy)
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_query_endpoint_returns_json(self, client_no_auth, query):
        """
        Feature: muai-orchestration-system, Property 34: API JSON格式
        
        对于任何有效查询，查询端点应返回有效的JSON响应
        """
        assume(query.strip())
        
        response = client_no_auth.post(
            "/api/query",
            json={"query": query}
        )
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        # 响应应该可以解析为JSON
        data = response.json()
        assert isinstance(data, dict)
        
        # 响应应该包含必需字段
        assert "success" in data
        assert "timestamp" in data
    
    @given(message=valid_query_strategy)
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chat_endpoint_returns_json(self, client_no_auth, message):
        """
        Feature: muai-orchestration-system, Property 34: API JSON格式
        
        对于任何有效消息，对话端点应返回有效的JSON响应
        """
        assume(message.strip())
        
        response = client_no_auth.post(
            "/api/chat",
            json={"message": message}
        )
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        # 响应应该可以解析为JSON
        data = response.json()
        assert isinstance(data, dict)
        
        # 响应应该包含必需字段
        assert "success" in data
        assert "session_id" in data
        assert "timestamp" in data
    
    def test_status_endpoint_returns_json(self, client, auth_headers):
        """
        Feature: muai-orchestration-system, Property 34: API JSON格式
        
        系统状态端点应返回有效的JSON响应
        """
        response = client.get("/api/status", headers=auth_headers)
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert "version" in data
        assert "uptime" in data


# ============ 属性35: API错误响应格式 ============

class TestAPIErrorResponseFormat:
    """
    属性35: API错误响应格式
    
    对于任何格式错误的API请求，响应应该包含清晰的错误信息（error字段）、
    适当的HTTP状态码（4xx或5xx），且响应体应该是有效的JSON。
    
    验证需求: 13.3, 15.3
    """
    
    def test_empty_query_returns_validation_error(self, client_no_auth):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        空查询应返回验证错误
        """
        response = client_no_auth.post(
            "/api/query",
            json={"query": ""}
        )
        
        # 应该返回4xx错误
        assert 400 <= response.status_code < 500
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        # 响应应该包含错误信息
        data = response.json()
        assert isinstance(data, dict)
        assert "success" in data or "error" in data
    
    def test_missing_required_field_returns_validation_error(self, client_no_auth):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        缺少必需字段应返回验证错误
        """
        response = client_no_auth.post(
            "/api/query",
            json={}
        )
        
        # 应该返回422验证错误
        assert response.status_code == 422
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        # 响应应该包含错误信息
        data = response.json()
        assert isinstance(data, dict)
        assert "error" in data or "detail" in data
    
    def test_invalid_json_returns_error(self, client_no_auth):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        无效的JSON应返回错误
        """
        response = client_no_auth.post(
            "/api/query",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        # 应该返回4xx错误
        assert 400 <= response.status_code < 500
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
    
    @given(
        top_k=st.integers(min_value=-100, max_value=0)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_top_k_returns_validation_error(self, client_no_auth, top_k):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        无效的top_k参数应返回验证错误
        """
        response = client_no_auth.post(
            "/api/rag/query",
            json={"query": "test query", "top_k": top_k}
        )
        
        # 应该返回验证错误
        assert response.status_code == 422
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        data = response.json()
        assert isinstance(data, dict)
    
    def test_invalid_workflow_type_returns_error(self, client_no_auth):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        无效的工作流类型应返回错误
        """
        response = client_no_auth.post(
            "/api/query",
            json={"query": "test", "workflow_type": "invalid_workflow"}
        )
        
        # 应该返回验证错误
        assert response.status_code == 422
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_error_response_has_required_fields(self, client_no_auth):
        """
        Feature: muai-orchestration-system, Property 35: API错误响应格式
        
        错误响应应包含必需的字段
        """
        response = client_no_auth.post(
            "/api/query",
            json={}
        )
        
        data = response.json()
        
        # 错误响应应该包含success=False或error字段
        has_error_indicator = (
            data.get("success") == False or
            "error" in data or
            "detail" in data
        )
        assert has_error_indicator
        
        # 应该有时间戳
        assert "timestamp" in data or "detail" in data


# ============ 属性36: API认证保护 ============

class TestAPIAuthProtection:
    """
    属性36: API认证保护
    
    对于任何需要认证的API端点，未提供有效认证凭据的请求应该被拒绝，
    返回401或403状态码。
    
    验证需求: 13.4
    """
    
    def test_status_endpoint_requires_auth(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        状态端点应要求认证
        """
        response = client.get("/api/status")
        
        # 应该返回401未授权
        assert response.status_code == 401
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
        
        data = response.json()
        assert "detail" in data
    
    def test_query_endpoint_requires_auth(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        查询端点应要求认证
        """
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        # 应该返回401未授权
        assert response.status_code == 401
    
    def test_chat_endpoint_requires_auth(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        对话端点应要求认证
        """
        response = client.post(
            "/api/chat",
            json={"message": "hello"}
        )
        
        # 应该返回401未授权
        assert response.status_code == 401
    
    def test_rag_upload_endpoint_requires_auth(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        RAG上传端点应要求认证
        """
        response = client.post(
            "/api/rag/upload",
            json={"content": "test document"}
        )
        
        # 应该返回401未授权
        assert response.status_code == 401
    
    def test_rag_query_endpoint_requires_auth(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        RAG查询端点应要求认证
        """
        response = client.post(
            "/api/rag/query",
            json={"query": "test query"}
        )
        
        # 应该返回401未授权
        assert response.status_code == 401
    
    @given(invalid_key=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'), max_codepoint=127)))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_api_key_rejected(self, client, invalid_key):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        无效的API密钥应被拒绝
        """
        # 确保不是有效的测试密钥
        assume(invalid_key != "test_api_key_12345678901234567890")
        
        response = client.get(
            "/api/status",
            headers={"X-API-Key": invalid_key}
        )
        
        # 应该返回403禁止访问
        assert response.status_code == 403
        
        # 响应应该是JSON格式
        assert "application/json" in response.headers.get("content-type", "")
    
    def test_valid_api_key_accepted(self, client, auth_headers):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        有效的API密钥应被接受
        """
        response = client.get("/api/status", headers=auth_headers)
        
        # 应该返回200成功
        assert response.status_code == 200
    
    def test_health_endpoint_no_auth_required(self, client):
        """
        Feature: muai-orchestration-system, Property 36: API认证保护
        
        健康检查端点不应要求认证
        """
        response = client.get("/api/health")
        
        # 应该返回200成功
        assert response.status_code == 200


# ============ API密钥认证管理器测试 ============

class TestAPIKeyAuthManager:
    """
    API密钥认证管理器的属性测试
    """
    
    @given(api_key=api_key_strategy)
    @settings(max_examples=50)
    def test_added_key_is_valid(self, api_key):
        """
        添加的API密钥应该被验证为有效
        """
        auth = APIKeyAuth(enabled=True)
        auth.add_api_key(api_key)
        
        assert auth.validate_api_key(api_key)
    
    @given(api_key=api_key_strategy)
    @settings(max_examples=50)
    def test_removed_key_is_invalid(self, api_key):
        """
        移除的API密钥应该被验证为无效
        """
        auth = APIKeyAuth(enabled=True)
        auth.add_api_key(api_key)
        auth.remove_api_key(api_key)
        
        assert not auth.validate_api_key(api_key)
    
    @given(
        key1=api_key_strategy,
        key2=api_key_strategy
    )
    @settings(max_examples=30)
    def test_different_keys_are_independent(self, key1, key2):
        """
        不同的API密钥应该独立验证
        """
        assume(key1 != key2)
        
        auth = APIKeyAuth(enabled=True)
        auth.add_api_key(key1)
        
        assert auth.validate_api_key(key1)
        assert not auth.validate_api_key(key2)
    
    def test_disabled_auth_accepts_any_key(self):
        """
        禁用认证时应接受任何密钥
        """
        auth = APIKeyAuth(enabled=False)
        
        assert auth.validate_api_key("any_key")
        assert auth.validate_api_key(None)
        assert auth.validate_api_key("")
    
    def test_generated_key_is_valid(self):
        """
        生成的API密钥应该自动添加并有效
        """
        auth = APIKeyAuth(enabled=True)
        generated_key = auth.generate_api_key()
        
        assert len(generated_key) >= 16
        assert auth.validate_api_key(generated_key)
    
    @given(num_keys=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_multiple_keys_count(self, num_keys):
        """
        添加多个密钥后数量应正确
        """
        auth = APIKeyAuth(enabled=True)
        
        for i in range(num_keys):
            auth.generate_api_key()
        
        assert auth.get_num_keys() == num_keys


# ============ 请求ID测试 ============

class TestRequestID:
    """
    请求ID的属性测试
    """
    
    def test_response_includes_request_id(self, client_no_auth):
        """
        响应应包含请求ID
        """
        response = client_no_auth.get("/api/health")
        
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"]
    
    def test_custom_request_id_preserved(self, client_no_auth):
        """
        自定义请求ID应被保留
        """
        custom_id = "custom-request-id-12345"
        response = client_no_auth.get(
            "/api/health",
            headers={"X-Request-ID": custom_id}
        )
        
        assert response.headers.get("X-Request-ID") == custom_id
    
    def test_process_time_header_present(self, client_no_auth):
        """
        响应应包含处理时间头
        """
        response = client_no_auth.get("/api/health")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0


# ============ 端点响应结构测试 ============

class TestEndpointResponseStructure:
    """
    端点响应结构的属性测试
    """
    
    def test_health_response_structure(self, client_no_auth):
        """
        健康检查响应应有正确的结构
        """
        response = client_no_auth.get("/api/health")
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert isinstance(data["timestamp"], (int, float))
    
    def test_status_response_structure(self, client, auth_headers):
        """
        状态响应应有正确的结构
        """
        response = client.get("/api/status", headers=auth_headers)
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "uptime" in data
        assert "components" in data
        assert "metrics" in data
        assert "timestamp" in data
        
        assert isinstance(data["uptime"], (int, float))
        assert isinstance(data["components"], dict)
        assert isinstance(data["metrics"], dict)
    
    @given(query=valid_query_strategy)
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_query_response_structure(self, client_no_auth, query):
        """
        查询响应应有正确的结构
        """
        assume(query.strip())
        
        response = client_no_auth.post(
            "/api/query",
            json={"query": query}
        )
        data = response.json()
        
        assert "success" in data
        assert "status" in data
        assert "metadata" in data
        assert "timestamp" in data
        assert "request_id" in data
        
        assert isinstance(data["success"], bool)
        assert isinstance(data["metadata"], dict)
    
    @given(message=valid_query_strategy)
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_chat_response_structure(self, client_no_auth, message):
        """
        对话响应应有正确的结构
        """
        assume(message.strip())
        
        response = client_no_auth.post(
            "/api/chat",
            json={"message": message}
        )
        data = response.json()
        
        assert "success" in data
        assert "session_id" in data
        assert "timestamp" in data
        assert "request_id" in data
        
        assert isinstance(data["success"], bool)
        assert isinstance(data["session_id"], str)
