"""
API单元测试

测试REST API的核心功能。

需求: 13.1, 13.2, 13.3, 13.4
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from fastapi.testclient import TestClient

from mm_orch.api.app import create_app, reset_app, get_uptime
from mm_orch.api.auth import (
    APIKeyAuth,
    get_auth_manager,
    set_auth_manager,
    reset_auth_manager,
    verify_api_key
)
from mm_orch.api.schemas import (
    QueryRequest,
    ChatRequest,
    RAGUploadRequest,
    RAGQueryRequest,
    LessonPackRequest,
    WorkflowTypeEnum
)


# ============ 测试夹具 ============

@pytest.fixture(autouse=True)
def reset_state():
    """每个测试前后重置状态"""
    reset_app()
    reset_auth_manager()
    yield
    reset_app()
    reset_auth_manager()


@pytest.fixture
def test_api_key():
    """测试API密钥"""
    return "test_api_key_for_unit_tests_1234567890"


@pytest.fixture
def app(test_api_key):
    """创建测试应用"""
    return create_app(
        auth_enabled=True,
        api_keys={test_api_key}
    )


@pytest.fixture
def client(app):
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def auth_headers(test_api_key):
    """认证请求头"""
    return {"X-API-Key": test_api_key}


@pytest.fixture
def app_no_auth():
    """创建无认证的测试应用"""
    return create_app(auth_enabled=False)


@pytest.fixture
def client_no_auth(app_no_auth):
    """无认证的测试客户端"""
    return TestClient(app_no_auth)


# ============ 应用创建测试 ============

class TestAppCreation:
    """应用创建测试"""
    
    def test_create_app_default(self):
        """测试默认应用创建"""
        app = create_app()
        
        assert app is not None
        assert app.title == "MuAI多模型编排系统API"
    
    def test_create_app_custom_title(self):
        """测试自定义标题"""
        app = create_app(title="Custom API")
        
        assert app.title == "Custom API"
    
    def test_create_app_with_api_keys(self, test_api_key):
        """测试带API密钥创建"""
        app = create_app(
            auth_enabled=True,
            api_keys={test_api_key}
        )
        
        auth_manager = get_auth_manager()
        assert auth_manager.enabled
        assert auth_manager.validate_api_key(test_api_key)
    
    def test_create_app_auth_disabled(self):
        """测试禁用认证"""
        app = create_app(auth_enabled=False)
        
        auth_manager = get_auth_manager()
        assert not auth_manager.enabled


# ============ 健康检查测试 ============

class TestHealthEndpoint:
    """健康检查端点测试"""
    
    def test_health_check_success(self, client):
        """测试健康检查成功"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_health_check_no_auth_required(self, client):
        """测试健康检查不需要认证"""
        # 不提供API密钥
        response = client.get("/api/health")
        
        assert response.status_code == 200


# ============ 系统状态测试 ============

class TestStatusEndpoint:
    """系统状态端点测试"""
    
    def test_status_requires_auth(self, client):
        """测试状态端点需要认证"""
        response = client.get("/api/status")
        
        assert response.status_code == 401
    
    def test_status_with_auth(self, client, auth_headers):
        """测试带认证的状态请求"""
        response = client.get("/api/status", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime" in data
        assert "components" in data
        assert "metrics" in data
    
    def test_status_response_structure(self, client, auth_headers):
        """测试状态响应结构"""
        response = client.get("/api/status", headers=auth_headers)
        data = response.json()
        
        # 验证组件状态
        assert "orchestrator" in data["components"]
        assert "consciousness" in data["components"]
        
        # 验证指标
        assert "total_requests" in data["metrics"]
        assert "success_rate" in data["metrics"]


# ============ 查询端点测试 ============

class TestQueryEndpoint:
    """查询端点测试"""
    
    def test_query_requires_auth(self, client):
        """测试查询端点需要认证"""
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 401
    
    def test_query_with_auth(self, client, auth_headers):
        """测试带认证的查询请求"""
        response = client.post(
            "/api/query",
            json={"query": "什么是机器学习？"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "status" in data
        assert "metadata" in data
    
    def test_query_empty_query_rejected(self, client_no_auth):
        """测试空查询被拒绝"""
        response = client_no_auth.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 422
    
    def test_query_missing_query_rejected(self, client_no_auth):
        """测试缺少查询字段被拒绝"""
        response = client_no_auth.post(
            "/api/query",
            json={}
        )
        
        assert response.status_code == 422
    
    def test_query_with_workflow_type(self, client, auth_headers):
        """测试指定工作流类型"""
        response = client.post(
            "/api/query",
            json={
                "query": "test query",
                "workflow_type": "search_qa"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_query_invalid_workflow_type(self, client_no_auth):
        """测试无效的工作流类型"""
        response = client_no_auth.post(
            "/api/query",
            json={
                "query": "test query",
                "workflow_type": "invalid_type"
            }
        )
        
        assert response.status_code == 422


# ============ 对话端点测试 ============

class TestChatEndpoint:
    """对话端点测试"""
    
    def test_chat_requires_auth(self, client):
        """测试对话端点需要认证"""
        response = client.post(
            "/api/chat",
            json={"message": "hello"}
        )
        
        assert response.status_code == 401
    
    def test_chat_with_auth(self, client, auth_headers):
        """测试带认证的对话请求"""
        response = client.post(
            "/api/chat",
            json={"message": "你好"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "session_id" in data
    
    def test_chat_creates_session(self, client, auth_headers):
        """测试对话创建会话"""
        response = client.post(
            "/api/chat",
            json={"message": "hello"},
            headers=auth_headers
        )
        
        data = response.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0
    
    def test_chat_with_session_id(self, client, auth_headers):
        """测试带会话ID的对话"""
        session_id = "test-session-123"
        response = client.post(
            "/api/chat",
            json={
                "message": "hello",
                "session_id": session_id
            },
            headers=auth_headers
        )
        
        data = response.json()
        # 会话ID应该被保留或更新
        assert "session_id" in data
    
    def test_chat_empty_message_rejected(self, client_no_auth):
        """测试空消息被拒绝"""
        response = client_no_auth.post(
            "/api/chat",
            json={"message": ""}
        )
        
        assert response.status_code == 422


# ============ RAG端点测试 ============

class TestRAGEndpoints:
    """RAG端点测试"""
    
    def test_rag_upload_requires_auth(self, client):
        """测试RAG上传需要认证"""
        response = client.post(
            "/api/rag/upload",
            json={"content": "test document"}
        )
        
        assert response.status_code == 401
    
    def test_rag_upload_with_auth(self, client, auth_headers):
        """测试带认证的RAG上传"""
        response = client.post(
            "/api/rag/upload",
            json={
                "content": "这是一篇测试文档，包含一些内容。",
                "metadata": {"source": "test"}
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "message" in data
    
    def test_rag_upload_empty_content_rejected(self, client_no_auth):
        """测试空内容被拒绝"""
        response = client_no_auth.post(
            "/api/rag/upload",
            json={"content": ""}
        )
        
        assert response.status_code == 422
    
    def test_rag_query_requires_auth(self, client):
        """测试RAG查询需要认证"""
        response = client.post(
            "/api/rag/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 401
    
    def test_rag_query_with_auth(self, client, auth_headers):
        """测试带认证的RAG查询"""
        response = client.post(
            "/api/rag/query",
            json={"query": "什么是深度学习？"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "sources" in data
    
    def test_rag_query_with_top_k(self, client, auth_headers):
        """测试带top_k参数的RAG查询"""
        response = client.post(
            "/api/rag/query",
            json={
                "query": "test query",
                "top_k": 3
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_rag_query_invalid_top_k(self, client_no_auth):
        """测试无效的top_k参数"""
        response = client_no_auth.post(
            "/api/rag/query",
            json={
                "query": "test query",
                "top_k": -1
            }
        )
        
        assert response.status_code == 422


# ============ 教学包端点测试 ============

class TestLessonEndpoint:
    """教学包端点测试"""
    
    def test_lesson_requires_auth(self, client):
        """测试教学包端点需要认证"""
        response = client.post(
            "/api/lesson",
            json={"topic": "Python基础"}
        )
        
        assert response.status_code == 401
    
    def test_lesson_with_auth(self, client, auth_headers):
        """测试带认证的教学包请求"""
        response = client.post(
            "/api/lesson",
            json={"topic": "Python基础语法"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "topic" in data
    
    def test_lesson_with_level(self, client, auth_headers):
        """测试带难度级别的教学包请求"""
        response = client.post(
            "/api/lesson",
            json={
                "topic": "机器学习入门",
                "level": "beginner"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_lesson_invalid_level(self, client_no_auth):
        """测试无效的难度级别"""
        response = client_no_auth.post(
            "/api/lesson",
            json={
                "topic": "test",
                "level": "invalid_level"
            }
        )
        
        assert response.status_code == 422


# ============ 认证管理器测试 ============

class TestAPIKeyAuth:
    """API密钥认证管理器测试"""
    
    def test_add_and_validate_key(self):
        """测试添加和验证密钥"""
        auth = APIKeyAuth(enabled=True)
        key = "test_key_1234567890123456"
        
        auth.add_api_key(key)
        
        assert auth.validate_api_key(key)
    
    def test_remove_key(self):
        """测试移除密钥"""
        auth = APIKeyAuth(enabled=True)
        key = "test_key_1234567890123456"
        
        auth.add_api_key(key)
        assert auth.validate_api_key(key)
        
        auth.remove_api_key(key)
        assert not auth.validate_api_key(key)
    
    def test_generate_key(self):
        """测试生成密钥"""
        auth = APIKeyAuth(enabled=True)
        
        key = auth.generate_api_key()
        
        assert len(key) >= 16
        assert auth.validate_api_key(key)
    
    def test_disabled_auth(self):
        """测试禁用认证"""
        auth = APIKeyAuth(enabled=False)
        
        assert auth.validate_api_key("any_key")
        assert auth.validate_api_key(None)
    
    def test_short_key_rejected(self):
        """测试短密钥被拒绝"""
        auth = APIKeyAuth(enabled=True)
        
        auth.add_api_key("short")  # 太短，不会被添加
        
        assert not auth.validate_api_key("short")
    
    def test_key_count(self):
        """测试密钥计数"""
        auth = APIKeyAuth(enabled=True)
        
        assert auth.get_num_keys() == 0
        
        auth.generate_api_key()
        assert auth.get_num_keys() == 1
        
        auth.generate_api_key()
        assert auth.get_num_keys() == 2


# ============ 请求头测试 ============

class TestRequestHeaders:
    """请求头测试"""
    
    def test_request_id_in_response(self, client_no_auth):
        """测试响应包含请求ID"""
        response = client_no_auth.get("/api/health")
        
        assert "X-Request-ID" in response.headers
    
    def test_custom_request_id(self, client_no_auth):
        """测试自定义请求ID"""
        custom_id = "my-custom-request-id"
        response = client_no_auth.get(
            "/api/health",
            headers={"X-Request-ID": custom_id}
        )
        
        assert response.headers["X-Request-ID"] == custom_id
    
    def test_process_time_header(self, client_no_auth):
        """测试处理时间头"""
        response = client_no_auth.get("/api/health")
        
        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
    
    def test_content_type_json(self, client_no_auth):
        """测试Content-Type为JSON"""
        response = client_no_auth.get("/api/health")
        
        assert "application/json" in response.headers["content-type"]
