# MuAI多模型编排系统

<p align="center">
  <strong>MuAI Multi-Model Orchestration System</strong><br>
  具备意识模块的通用AI系统，支持多工作流编排和教学能力
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#使用指南">使用指南</a> •
  <a href="#api文档">API文档</a> •
  <a href="#配置说明">配置说明</a> •
  <a href="#开发指南">开发指南</a>
</p>

---

## 功能特性

### 🔄 多工作流编排
- **智能路由**: 自动分析用户意图，选择最合适的处理工作流
- **5种核心工作流**: search_qa、lesson_pack、chat_generate、rag_qa、self_ask_search_qa
- **降级策略**: 当主要方案失败时自动切换到备用方案

### 🧠 意识模块系统
- **自我模型 (SelfModel)**: 维护系统能力清单和状态信息
- **世界模型 (WorldModel)**: 维护外部环境知识表示
- **元认知 (Metacognition)**: 监控任务执行，提供策略建议
- **动机系统 (MotivationSystem)**: 管理目标层次和优先级
- **情感系统 (EmotionSystem)**: 处理情感状态，影响响应风格
- **发展系统 (DevelopmentSystem)**: 渐进式能力解锁

### 📚 教学助手
- 自动生成结构化教学计划
- 生成详细讲解内容
- 创建配套练习题和答案

### 🔍 搜索增强问答
- 网络搜索 → 内容抓取 → 摘要生成 → 答案生成
- 支持DuckDuckGo搜索
- 使用trafilatura进行网页内容提取

### 📖 RAG知识库问答
- 基于FAISS的向量检索
- 文档分块和向量化
- 答案来源标注

### 💬 多轮对话
- 上下文感知的对话生成
- 持久化对话历史
- 滑动窗口策略管理长对话

### 🤖 真实模型集成
- 支持HuggingFace模型（Qwen-Chat、GPT-2等）
- 8-bit/4-bit量化支持（bitsandbytes、GPTQ）
- FlashAttention加速
- LRU模型缓存策略
- GPU/CPU自动切换

### 📊 性能基准测试
- 延迟测试（TTFT、tokens/s）
- 内存占用测试
- 吞吐量测试
- JSON/CSV报告生成

### ⚡ 高级优化和监控
- **推理引擎**: vLLM、DeepSpeed、ONNX Runtime集成
- **动态批处理**: 自动请求批处理以提高吞吐量
- **KV缓存**: Transformer键值对缓存优化
- **Prometheus监控**: 性能指标导出
- **OpenTelemetry追踪**: 分布式请求追踪
- **异常检测**: 自动性能告警
- **服务器模式**: 长期运行的推理服务器
- **自动调优**: 基于工作负载的参数优化

详细文档请参阅 [优化和监控文档](docs/optimization_README.md)

---

## 快速开始

### 系统要求

- **Python**: 3.8+
- **GPU**: NVIDIA T4 (15GB) 或 A100（推荐）
- **CPU**: 支持CPU运行（速度较慢）
- **内存**: 至少16GB RAM

### 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd muai-orchestration
```

2. **创建虚拟环境**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置系统**
```bash
# 复制示例配置
cp config/config.example.yaml config/system.yaml

# 编辑配置文件（可选）
# 根据您的环境修改 config/system.yaml
```

5. **验证安装**
```bash
# 运行测试
pytest tests/ -v

# 检查系统状态
python -m mm_orch.main --status
```

### 快速使用

```bash
# 单次查询模式
python -m mm_orch.main "什么是机器学习？"

# 指定工作流
python -m mm_orch.main --workflow search_qa "最新的AI发展趋势"

# 交互式对话模式
python -m mm_orch.main --mode chat

# 使用真实模型
python -m mm_orch.main --real-models --model gpt2 "你好"

# 查看可用模型
python -m mm_orch.main --model-info

# 运行基准测试
python -m mm_orch.main --benchmark --model gpt2

# 启动API服务
python -m mm_orch.main --serve
```

---

## 使用指南

### 命令行接口 (CLI)

#### 基本用法

```bash
python -m mm_orch.main [OPTIONS] [QUERY]
```

#### 选项说明

| 选项 | 说明 | 示例 |
|------|------|------|
| `--workflow` | 指定工作流类型 | `--workflow search_qa` |
| `--mode` | 运行模式 (query/chat) | `--mode chat` |
| `--serve` | 启动API服务器 | `--serve` |
| `--config` | 指定配置文件 | `--config custom.yaml` |
| `--status` | 显示系统状态 | `--status` |
| `--verbose` | 详细输出 | `--verbose` |
| `--model` | 指定模型 | `--model gpt2` |
| `--real-models` | 使用真实模型 | `--real-models` |
| `--benchmark` | 运行基准测试 | `--benchmark` |
| `--model-info` | 查看模型信息 | `--model-info` |

#### 工作流类型

| 工作流 | 说明 | 适用场景 |
|--------|------|----------|
| `search_qa` | 搜索问答 | 需要最新信息的问题 |
| `lesson_pack` | 教学生成 | 生成教学内容 |
| `chat_generate` | 对话生成 | 日常对话交互 |
| `rag_qa` | 知识库问答 | 基于本地文档的问答 |
| `self_ask_search_qa` | 自问自答搜索 | 复杂问题分解 |

#### 使用示例

```bash
# 搜索问答
python -m mm_orch.main --workflow search_qa "2024年诺贝尔物理学奖得主是谁？"

# 生成教学内容
python -m mm_orch.main --workflow lesson_pack "Python基础语法"

# 交互式对话
python -m mm_orch.main --mode chat
> 你好，请介绍一下你自己
> 你能做什么？
> exit  # 退出对话

# 知识库问答（需要先上传文档）
python -m mm_orch.main --workflow rag_qa "项目的主要功能是什么？"
```

### Python API

```python
from mm_orch.orchestrator import WorkflowOrchestrator
from mm_orch.router import Router
from mm_orch.schemas import UserRequest, WorkflowType

# 初始化
orchestrator = WorkflowOrchestrator()
router = Router()

# 自动路由查询
request = UserRequest(query="什么是深度学习？")
selection = router.route(request)
result = orchestrator.execute_workflow(
    selection.workflow_type,
    selection.parameters
)
print(result.result)

# 指定工作流
result = orchestrator.execute_workflow(
    WorkflowType.SEARCH_QA,
    {"query": "最新的AI新闻"}
)

# 多轮对话
result = orchestrator.execute_workflow(
    WorkflowType.CHAT_GENERATE,
    {
        "session_id": "my-session",
        "message": "你好！"
    }
)
```

### 意识模块使用

```python
from mm_orch.consciousness import get_consciousness, save_consciousness

# 获取意识核心实例
consciousness = get_consciousness()

# 查看系统状态
status = consciousness.get_status_summary()
print(status)

# 获取策略建议
from mm_orch.schemas import Task
task = Task(type="qa", query="复杂问题")
suggestion = consciousness.get_strategy_suggestion(task)

# 更新情感状态
consciousness.emotion.update_from_result(success=True, user_feedback="positive")

# 保存意识状态
save_consciousness(force=True)
```

---

## API文档

### 启动API服务

```bash
# 默认配置启动
python -m mm_orch.main --serve

# 指定端口
python -m mm_orch.main --serve --port 8080

# 启用认证
MUAI_ENABLE_AUTH=true MUAI_API_KEY=your-secret-key python -m mm_orch.main --serve
```

### API端点

#### POST /api/query
通用查询接口，自动路由到合适的工作流。

**请求**:
```json
{
  "query": "什么是机器学习？",
  "context": {},
  "session_id": null
}
```

**响应**:
```json
{
  "result": "机器学习是...",
  "workflow_type": "chat_generate",
  "metadata": {
    "confidence": 0.85,
    "processing_time": 1.23
  },
  "status": "success"
}
```

#### POST /api/chat
对话接口，支持多轮对话。

**请求**:
```json
{
  "session_id": "session-123",
  "message": "你好！"
}
```

**响应**:
```json
{
  "response": "你好！有什么我可以帮助你的吗？",
  "session_id": "session-123",
  "turn_count": 1
}
```

#### POST /api/rag/upload
上传文档到知识库。

**请求** (multipart/form-data):
- `file`: 文档文件 (支持 .txt, .pdf, .md)
- `metadata`: JSON格式的元数据（可选）

**响应**:
```json
{
  "document_id": "doc-456",
  "chunks_count": 15,
  "status": "indexed"
}
```

#### POST /api/rag/query
知识库问答。

**请求**:
```json
{
  "query": "项目的主要功能是什么？",
  "top_k": 5
}
```

**响应**:
```json
{
  "answer": "项目的主要功能包括...",
  "sources": [
    {
      "content": "...",
      "metadata": {"source": "readme.md", "chunk_id": 3}
    }
  ]
}
```

#### GET /api/status
获取系统状态。

**响应**:
```json
{
  "status": "healthy",
  "development_stage": "adult",
  "loaded_models": ["qwen_chat", "minilm_embedding"],
  "consciousness": {
    "emotion": {"valence": 0.5, "arousal": 0.3},
    "motivation": {"active_goals": 2}
  },
  "metrics": {
    "total_requests": 1234,
    "avg_response_time": 2.5
  }
}
```

### 认证

当启用API认证时，需要在请求头中包含API密钥：

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "你好"}'
```

### 错误响应

所有错误响应遵循统一格式：

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request format",
    "details": {"field": "query", "issue": "required"}
  },
  "status": "failed"
}
```

| HTTP状态码 | 错误类型 | 说明 |
|------------|----------|------|
| 400 | VALIDATION_ERROR | 请求格式错误 |
| 401 | UNAUTHORIZED | 未提供认证信息 |
| 403 | FORBIDDEN | 认证失败 |
| 404 | NOT_FOUND | 资源不存在 |
| 500 | INTERNAL_ERROR | 服务器内部错误 |
| 503 | SERVICE_UNAVAILABLE | 服务暂时不可用 |

---

## 配置说明

### 配置文件位置

- 主配置文件: `config/system.yaml`
- 示例配置: `config/config.example.yaml`

### 主要配置项

#### 系统配置
```yaml
system:
  log_level: INFO          # 日志级别
  max_cached_models: 3     # 最大缓存模型数
  development_stage: adult # 发展阶段
  device: auto             # 推理设备
```

#### 模型配置
```yaml
models:
  qwen_chat:
    model_path: "Qwen/Qwen-7B-Chat"
    device: auto
    quantization: null  # 可选: 8bit, 4bit
    max_length: 2048
    temperature: 0.7
```

#### API配置
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  enable_auth: false
  api_key: null
```

### 环境变量

支持通过环境变量覆盖配置：

| 环境变量 | 对应配置 | 说明 |
|----------|----------|------|
| `MUAI_LOG_LEVEL` | system.log_level | 日志级别 |
| `MUAI_DEVICE` | system.device | 推理设备 |
| `MUAI_API_HOST` | api.host | API监听地址 |
| `MUAI_API_PORT` | api.port | API监听端口 |
| `MUAI_API_KEY` | api.api_key | API密钥 |
| `MUAI_ENABLE_AUTH` | api.enable_auth | 启用认证 |

---

## 文档

### 核心文档
- [API参考](docs/api_reference.md) - API接口文档
- [意识系统指南](docs/consciousness_guide.md) - 意识模块使用指南
- [基准测试指南](docs/benchmark_guide.md) - 性能测试指南

### 优化和监控文档
- [优化和监控概述](docs/optimization_README.md) - 功能概述和快速开始
- [配置指南](docs/optimization_configuration_guide.md) - 详细配置参数说明
- [配置示例](docs/optimization_configuration_examples.md) - 场景化配置示例
- [迁移指南](docs/optimization_migration_guide.md) - 从现有系统迁移
- [部署指南](docs/optimization_deployment_guide.md) - 生产环境部署
- [快速参考](docs/optimization_quick_reference.md) - 常用配置和命令

---

## 项目结构

```
.
├── mm_orch/                    # 主源代码
│   ├── __init__.py
│   ├── main.py                 # CLI入口
│   ├── orchestrator.py         # 工作流编排器
│   ├── router.py               # 智能路由器
│   ├── schemas.py              # 数据模型
│   ├── config.py               # 配置管理
│   ├── logger.py               # 日志系统
│   ├── error_handler.py        # 错误处理
│   ├── retry.py                # 重试机制
│   ├── metrics.py              # 性能监控
│   ├── consciousness/          # 意识模块
│   │   ├── core.py             # 意识核心
│   │   ├── self_model.py       # 自我模型
│   │   ├── world_model.py      # 世界模型
│   │   ├── metacognition.py    # 元认知
│   │   ├── motivation.py       # 动机系统
│   │   ├── emotion.py          # 情感系统
│   │   └── development.py      # 发展系统
│   ├── workflows/              # 工作流实现
│   │   ├── base.py             # 基类
│   │   ├── search_qa.py        # 搜索问答
│   │   ├── lesson_pack.py      # 教学生成
│   │   ├── chat_generate.py    # 对话生成
│   │   ├── rag_qa.py           # RAG问答
│   │   └── self_ask_search_qa.py
│   ├── runtime/                # 运行时管理
│   │   ├── model_manager.py    # 模型管理
│   │   └── vector_db.py        # 向量库管理
│   ├── storage/                # 存储管理
│   │   ├── chat_storage.py     # 对话存储
│   │   └── persistence.py      # 持久化
│   ├── tools/                  # 外部工具
│   │   ├── web_search.py       # 网络搜索
│   │   └── fetch_url.py        # 网页抓取
│   └── api/                    # API接口
│       ├── app.py              # FastAPI应用
│       ├── routes.py           # 路由定义
│       ├── schemas.py          # API模型
│       └── auth.py             # 认证
├── tests/                      # 测试套件
│   ├── unit/                   # 单元测试
│   ├── property/               # 属性测试
│   ├── integration/            # 集成测试
│   └── fixtures/               # 测试夹具
├── config/                     # 配置文件
│   ├── system.yaml             # 主配置
│   └── config.example.yaml     # 示例配置
├── data/                       # 运行时数据
│   ├── vector_db/              # 向量索引
│   ├── chat_history/           # 对话历史
│   └── traces/                 # 执行追踪
├── docs/                       # 文档
├── scripts/                    # 工具脚本
├── requirements.txt            # Python依赖
├── pyproject.toml              # 项目配置
├── setup.py                    # 安装脚本
├── Makefile                    # 常用命令
└── README.md                   # 本文件
```

---

## 开发指南

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install -e .

# 安装pre-commit hooks（可选）
pip install pre-commit
pre-commit install
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行单元测试
pytest tests/unit/ -v

# 运行属性测试
pytest tests/property/ -v

# 运行特定测试文件
pytest tests/unit/test_router.py -v

# 带覆盖率报告
pytest --cov=mm_orch --cov-report=html tests/

# 运行特定测试
pytest tests/unit/test_router.py::test_route_search_query -v
```

### 代码质量

```bash
# 使用Makefile
make lint      # 运行所有检查
make format    # 格式化代码
make test      # 运行测试
make coverage  # 生成覆盖率报告

# 或手动运行
black mm_orch/ tests/           # 代码格式化
flake8 mm_orch/ tests/          # 代码检查
mypy mm_orch/                   # 类型检查
```

### 添加新工作流

1. 在 `mm_orch/workflows/` 创建新文件
2. 继承 `BaseWorkflow` 类
3. 实现 `execute()` 和 `validate_parameters()` 方法
4. 在 `WorkflowOrchestrator` 中注册
5. 更新 `Router` 的路由规则
6. 添加对应的测试

```python
# mm_orch/workflows/my_workflow.py
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.schemas import WorkflowResult

class MyWorkflow(BaseWorkflow):
    def execute(self, parameters: dict) -> WorkflowResult:
        # 实现工作流逻辑
        result = self._process(parameters)
        return WorkflowResult(
            result=result,
            metadata={},
            status="success"
        )
    
    def validate_parameters(self, parameters: dict) -> bool:
        return "query" in parameters
    
    def get_required_models(self) -> list:
        return ["qwen_chat"]
```

### 测试规范

- 单元测试: 测试特定示例和边缘情况
- 属性测试: 使用Hypothesis验证通用属性
- 每个属性测试至少运行100次迭代

```python
# tests/property/test_my_workflow_properties.py
from hypothesis import given, strategies as st
import pytest

@given(query=st.text(min_size=1, max_size=500))
def test_workflow_returns_valid_result(query):
    """
    Feature: muai-orchestration-system, Property X: 结果有效性
    
    对于任何有效查询，工作流应返回有效的结果对象
    """
    workflow = MyWorkflow()
    result = workflow.execute({"query": query})
    
    assert result is not None
    assert result.status in ["success", "partial", "failed"]
```

---

## 常见问题

### Q: GPU内存不足怎么办？

A: 可以尝试以下方案：
1. 减少 `max_cached_models` 配置
2. 启用模型量化 (`quantization: 8bit` 或 `4bit`)
3. 使用更小的模型（如 `distilgpt2` 替代 `gpt2`）
4. 设置 `device: cpu` 使用CPU运行

### Q: 如何添加自定义知识库？

A: 使用RAG上传接口：
```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@your_document.txt" \
  -F 'metadata={"source": "custom"}'
```

### Q: 如何查看系统日志？

A: 日志默认输出到控制台。可以配置 `storage.log_file_path` 保存到文件：
```yaml
storage:
  log_file_path: "logs/muai.log"
```

### Q: 如何在生产环境部署？

A: 建议：
1. 启用API认证
2. 使用反向代理（如Nginx）
3. 配置适当的资源限制
4. 设置日志级别为INFO或WARNING
5. 定期备份数据目录

---

## 许可证

[待定]

## 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 更新日志

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本更新历史。
