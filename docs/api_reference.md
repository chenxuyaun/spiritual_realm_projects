# MuAI API 参考文档

本文档详细描述MuAI系统的RESTful API接口。

## 基础信息

- 基础URL: `http://localhost:8000/api`
- 内容类型: `application/json`
- 认证方式: Bearer Token (可选)

## 认证

当启用API认证时，需要在请求头中包含API密钥：

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "你好"}'
```

环境变量配置：
```bash
MUAI_ENABLE_AUTH=true
MUAI_API_KEY=your-secret-key
```

---

## 端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/status` | GET | 系统状态 |
| `/query` | POST | 通用查询 |
| `/chat` | POST | 对话接口 |
| `/search` | POST | 搜索问答 |
| `/lesson` | POST | 教学包生成 |
| `/rag/upload` | POST | RAG文档上传 |
| `/rag/query` | POST | RAG问答 |
| `/consciousness` | GET | 意识状态 |
| `/metrics` | GET | 性能指标 |
| `/models` | GET | 模型列表 |
| `/models/{name}` | GET | 模型详情 |
| `/benchmark` | POST | 基准测试 |

---

## 健康检查

### GET /health

检查API服务是否正常运行。不需要认证。

**响应**:
```json
{
  "status": "ok",
  "timestamp": 1705312200.123
}
```

---

## 系统状态

### GET /status

获取系统运行状态和性能指标。

**响应**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime": 3600.5,
  "components": {
    "orchestrator": {
      "status": "healthy",
      "workflows": 5
    },
    "consciousness": {
      "status": "healthy",
      "stage": "adult"
    },
    "vector_db": {
      "status": "healthy",
      "documents": 100
    }
  },
  "metrics": {
    "total_requests": 1234,
    "success_rate": 0.95,
    "avg_response_time": 2.5
  },
  "timestamp": 1705312200.123
}
```

---

## 通用查询

### POST /query

提交查询请求，系统自动路由到合适的工作流。

**请求**:
```json
{
  "query": "什么是机器学习？",
  "workflow_type": null,
  "context": {},
  "session_id": null,
  "preferences": {}
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 查询内容 |
| workflow_type | string | 否 | 指定工作流类型 |
| context | object | 否 | 上下文信息 |
| session_id | string | 否 | 会话ID |
| preferences | object | 否 | 偏好设置 |

**工作流类型**:
- `search_qa`: 搜索问答
- `lesson_pack`: 教学生成
- `chat_generate`: 对话生成
- `rag_qa`: RAG问答
- `self_ask_search_qa`: 自问自答搜索

**响应**:
```json
{
  "success": true,
  "result": "机器学习是...",
  "status": "success",
  "metadata": {
    "workflow_type": "chat_generate",
    "confidence": 0.85
  },
  "error": null,
  "execution_time": 1.23,
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 对话接口

### POST /chat

发送消息进行对话，支持多轮对话。

**请求**:
```json
{
  "message": "你好！",
  "session_id": "session-123",
  "context": {}
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| message | string | 是 | 消息内容 |
| session_id | string | 否 | 会话ID（不提供则自动生成） |
| context | object | 否 | 上下文信息 |

**响应**:
```json
{
  "success": true,
  "response": "你好！有什么我可以帮助你的吗？",
  "session_id": "session-123",
  "message_id": "msg-456",
  "timestamp": 1705312200.123,
  "request_id": "abc-123",
  "error": null
}
```

---

## 搜索问答

### POST /search

通过网络搜索回答问题。

**请求**:
```json
{
  "query": "2024年诺贝尔物理学奖得主是谁？",
  "context": {},
  "preferences": {
    "max_results": 5
  }
}
```

**响应**:
```json
{
  "success": true,
  "result": "2024年诺贝尔物理学奖...",
  "status": "success",
  "metadata": {
    "sources": [
      {"url": "https://...", "title": "..."}
    ],
    "search_count": 5
  },
  "error": null,
  "execution_time": 3.5,
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 教学包生成

### POST /lesson

根据课题生成完整的教学内容。

**请求**:
```json
{
  "topic": "Python基础语法",
  "level": "beginner",
  "num_exercises": 3
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| topic | string | 是 | 教学主题 |
| level | string | 否 | 难度级别 (beginner/intermediate/advanced) |
| num_exercises | int | 否 | 练习题数量 |

**响应**:
```json
{
  "success": true,
  "topic": "Python基础语法",
  "plan": "# 教学计划\n\n## 学习目标\n...",
  "explanation": "# Python基础语法详解\n\n...",
  "exercises": [
    {
      "question": "编写一个打印Hello World的程序",
      "answer": "print('Hello World')"
    }
  ],
  "timestamp": 1705312200.123,
  "request_id": "abc-123",
  "error": null
}
```

---

## RAG文档上传

### POST /rag/upload

上传文档到RAG知识库。

**请求**:
```json
{
  "content": "这是文档内容...",
  "metadata": {
    "source": "manual",
    "title": "文档标题"
  },
  "chunk_size": 500,
  "chunk_overlap": 50
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| content | string | 是 | 文档内容 |
| metadata | object | 否 | 文档元数据 |
| chunk_size | int | 否 | 分块大小（默认500） |
| chunk_overlap | int | 否 | 分块重叠（默认50） |

**响应**:
```json
{
  "success": true,
  "document_id": "doc-456",
  "chunks_count": 15,
  "message": "Document uploaded successfully",
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## RAG问答

### POST /rag/query

基于知识库进行问答。

**请求**:
```json
{
  "query": "项目的主要功能是什么？",
  "top_k": 5,
  "threshold": 0.7
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | string | 是 | 查询问题 |
| top_k | int | 否 | 返回结果数量（默认5） |
| threshold | float | 否 | 相似度阈值（默认0.7） |

**响应**:
```json
{
  "success": true,
  "answer": "项目的主要功能包括...",
  "sources": [
    {
      "content": "相关文档片段...",
      "metadata": {"source": "readme.md", "chunk_id": 3},
      "score": 0.85
    }
  ],
  "timestamp": 1705312200.123,
  "request_id": "abc-123",
  "error": null
}
```

---

## 意识状态

### GET /consciousness

获取意识模块状态。

**响应**:
```json
{
  "success": true,
  "consciousness": {
    "development_stage": "adult",
    "emotion": {
      "valence": 0.5,
      "arousal": 0.3,
      "dominant_emotion": "neutral"
    },
    "motivation": {
      "active_goals": 2,
      "top_goal": "assist_user"
    },
    "metacognition": {
      "confidence": 0.8,
      "uncertainty": 0.2
    }
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 性能指标

### GET /metrics

获取工作流执行指标。

**响应**:
```json
{
  "success": true,
  "orchestrator": {
    "execution_count": 1234,
    "success_rate": 0.95,
    "average_execution_time": 2.5,
    "registered_workflows": 5
  },
  "workflows": {
    "search_qa": {
      "count": 500,
      "avg_time": 3.2,
      "success_rate": 0.92
    },
    "chat_generate": {
      "count": 600,
      "avg_time": 1.5,
      "success_rate": 0.98
    }
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 模型管理

### GET /models

获取可用模型列表和已加载模型状态。

**响应**:
```json
{
  "success": true,
  "available_models": ["gpt2", "gpt2-medium", "qwen-7b-chat"],
  "model_configs": {
    "gpt2": {
      "model_type": "gpt2",
      "quantization": null,
      "device": "auto"
    },
    "qwen-7b-chat": {
      "model_type": "qwen-chat",
      "quantization": null,
      "device": "auto"
    }
  },
  "loaded_models": ["gpt2"],
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

### GET /models/{model_name}

获取特定模型的详细信息。

**路径参数**:
- `model_name`: 模型名称

**响应**:
```json
{
  "success": true,
  "model_name": "gpt2",
  "config": {
    "model_name": "gpt2",
    "model_type": "gpt2",
    "device": "auto",
    "dtype": "fp32",
    "quantization": null,
    "trust_remote_code": false,
    "flash_attention": false,
    "max_context_length": 1024
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

**错误响应** (404):
```json
{
  "success": false,
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'unknown-model' not found"
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 基准测试

### POST /benchmark

运行模型性能基准测试。

**查询参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_name | string | 否 | 模型名称（默认gpt2） |
| test_types | array | 否 | 测试类型列表 |

**测试类型**:
- `latency`: 延迟测试
- `memory`: 内存测试
- `throughput`: 吞吐量测试

**请求示例**:
```bash
curl -X POST "http://localhost:8000/api/benchmark?model_name=gpt2&test_types=latency&test_types=memory" \
  -H "Authorization: Bearer your-api-key"
```

**响应**:
```json
{
  "success": true,
  "model_name": "gpt2",
  "results": {
    "latency": {
      "ttft_ms": 45.2,
      "tokens_per_second": 35.8,
      "e2e_latency_ms": 2850.5
    },
    "memory": {
      "model_load_gb": 1.2,
      "inference_delta_gb": 0.15,
      "peak_gb": 1.45
    },
    "throughput": {
      "single_request_tps": 35.8,
      "concurrent_rps": 12.5
    }
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

---

## 错误响应

所有错误响应遵循统一格式：

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述"
  },
  "timestamp": 1705312200.123,
  "request_id": "abc-123"
}
```

### 错误代码

| HTTP状态码 | 错误代码 | 说明 |
|------------|----------|------|
| 400 | VALIDATION_ERROR | 请求格式错误 |
| 401 | UNAUTHORIZED | 未提供认证信息 |
| 403 | FORBIDDEN | 认证失败 |
| 404 | NOT_FOUND | 资源不存在 |
| 404 | MODEL_NOT_FOUND | 模型不存在 |
| 404 | CONFIG_NOT_FOUND | 配置文件不存在 |
| 422 | UNPROCESSABLE_ENTITY | 请求验证失败 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |
| 501 | BENCHMARK_NOT_AVAILABLE | 基准测试模块不可用 |
| 503 | SERVICE_UNAVAILABLE | 服务暂时不可用 |

---

## SDK示例

### Python

```python
import requests

BASE_URL = "http://localhost:8000/api"
API_KEY = "your-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 通用查询
response = requests.post(
    f"{BASE_URL}/query",
    headers=headers,
    json={"query": "什么是机器学习？"}
)
print(response.json())

# 对话
response = requests.post(
    f"{BASE_URL}/chat",
    headers=headers,
    json={
        "message": "你好！",
        "session_id": "my-session"
    }
)
print(response.json())

# 获取模型列表
response = requests.get(
    f"{BASE_URL}/models",
    headers=headers
)
print(response.json())

# 运行基准测试
response = requests.post(
    f"{BASE_URL}/benchmark",
    headers=headers,
    params={"model_name": "gpt2", "test_types": ["latency"]}
)
print(response.json())
```

### cURL

```bash
# 健康检查
curl http://localhost:8000/api/health

# 通用查询
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是机器学习？"}'

# 对话
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "你好！", "session_id": "my-session"}'

# 获取模型列表
curl http://localhost:8000/api/models \
  -H "Authorization: Bearer your-api-key"

# 运行基准测试
curl -X POST "http://localhost:8000/api/benchmark?model_name=gpt2" \
  -H "Authorization: Bearer your-api-key"
```

---

## 速率限制

默认情况下，API没有速率限制。在生产环境中，建议配置反向代理（如Nginx）进行速率限制。

## WebSocket支持

当前版本不支持WebSocket。流式响应将在未来版本中添加。

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 0.1.0 | 2024-01 | 初始版本 |
| 0.2.0 | 2024-01 | 添加模型管理和基准测试端点 |
