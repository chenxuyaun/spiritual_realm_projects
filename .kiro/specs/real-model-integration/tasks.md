# 真实模型集成与验证 - 任务列表

## 任务概览

本任务列表基于需求文档和设计文档，按照依赖关系和优先级排序。

---

## 第一阶段：基础设施 (Foundation)

### 1. 项目配置更新
- [x] 1.1 更新pyproject.toml添加新依赖（bitsandbytes, auto-gptq, flash-attn）
- [x] 1.2 创建config/models.yaml模型配置文件
- [x] 1.3 创建config/benchmark.yaml基准测试配置文件
- [x] 1.4 更新requirements.txt

### 2. 异常类定义
- [x] 2.1 在mm_orch/exceptions.py中添加ModelLoadError
- [x] 2.2 添加InferenceError异常类
- [x] 2.3 添加OutOfMemoryError异常类
- [x] 2.4 添加QuantizationError异常类

---

## 第二阶段：模型管理 (Model Management)

### 3. 量化管理器实现
- [x] 3.1 创建mm_orch/runtime/quantization.py
- [x] 3.2 实现get_quantization_config方法（8bit/4bit配置）
- [x] 3.3 实现load_gptq_model方法
- [x] 3.4 编写量化管理器单元测试

### 4. 模型加载器实现
- [x] 4.1 创建mm_orch/runtime/model_loader.py
- [x] 4.2 实现HuggingFace模型加载（支持trust_remote_code）
- [x] 4.3 实现设备自动检测和分配
- [x] 4.4 实现数据类型自动选择（fp32/fp16/bf16）
- [x] 4.5 集成量化加载支持
- [x] 4.6 编写模型加载器单元测试

### 5. 内存监控器实现
- [x] 5.1 创建mm_orch/runtime/memory_monitor.py
- [x] 5.2 实现GPU内存监控（torch.cuda.memory_allocated）
- [x] 5.3 实现CPU内存监控（psutil）
- [x] 5.4 实现内存阈值告警
- [x] 5.5 编写内存监控器单元测试

### 6. RealModelManager实现
- [x] 6.1 创建mm_orch/runtime/real_model_manager.py
- [x] 6.2 实现load_model方法
- [x] 6.3 实现unload_model方法
- [x] 6.4 实现LRU缓存策略
- [x] 6.5 实现GPU/CPU自动切换
- [x] 6.6 编写RealModelManager单元测试
- [x] 6.7 编写RealModelManager属性测试（P1.1-P1.4）

---

## 第三阶段：推理引擎 (Inference Engine)

### 7. FlashAttention集成
- [x] 7.1 创建mm_orch/runtime/flash_attention.py
- [x] 7.2 实现FlashAttention可用性检测
- [x] 7.3 实现自动启用/回退逻辑
- [x] 7.4 编写FlashAttention单元测试

### 8. 推理引擎实现
- [x] 8.1 创建mm_orch/runtime/inference_engine.py
- [x] 8.2 实现generate方法（单次生成）
- [x] 8.3 实现generate_stream方法（流式生成）
- [x] 8.4 实现batch_generate方法（批量生成）
- [x] 8.5 实现生成参数验证
- [x] 8.6 编写推理引擎单元测试
- [x] 8.7 编写推理引擎属性测试（P2.1-P2.4）

### 9. 对话管理器实现
- [x] 9.1 创建mm_orch/runtime/conversation.py
- [x] 9.2 实现Qwen-Chat对话格式构建
- [x] 9.3 实现GPT-2对话格式构建
- [x] 9.4 实现历史截断策略
- [x] 9.5 编写对话管理器单元测试
- [x] 9.6 编写对话管理器属性测试（P3.1-P3.3）

---

## 第四阶段：工作流集成 (Workflow Integration)

### 10. SearchQA工作流适配
- [ ] 10.1 更新mm_orch/workflows/search_qa.py支持真实模型
- [ ] 10.2 实现检索资料提示词模板
- [ ] 10.3 实现答案后处理（引用验证）
- [ ] 10.4 编写SearchQA集成测试

### 11. LessonPack工作流适配
- [ ] 11.1 更新mm_orch/workflows/lesson_pack.py支持真实模型
- [ ] 11.2 实现课程生成提示词模板
- [ ] 11.3 实现Markdown格式验证
- [ ] 11.4 编写LessonPack集成测试

### 12. ChatGenerate工作流适配
- [ ] 12.1 更新mm_orch/workflows/chat_generate.py支持真实模型
- [ ] 12.2 集成对话管理器
- [ ] 12.3 实现多轮对话支持
- [ ] 12.4 编写ChatGenerate集成测试

---

## 第五阶段：性能基准测试 (Benchmarking)

### 13. 延迟基准测试实现
- [x] 13.1 创建mm_orch/benchmark/latency.py
- [x] 13.2 实现TTFT测量
- [x] 13.3 实现tokens/s测量
- [x] 13.4 实现端到端延迟测量
- [x] 13.5 编写延迟基准测试单元测试

### 14. 内存基准测试实现
- [x] 14.1 创建mm_orch/benchmark/memory.py
- [x] 14.2 实现模型加载内存测量
- [x] 14.3 实现推理内存增长测量
- [x] 14.4 实现量化对比测量
- [x] 14.5 编写内存基准测试单元测试

### 15. 吞吐量基准测试实现
- [x] 15.1 创建mm_orch/benchmark/throughput.py
- [x] 15.2 实现单请求吞吐量测量
- [x] 15.3 实现并发吞吐量测量
- [x] 15.4 实现批处理吞吐量测量
- [x] 15.5 编写吞吐量基准测试单元测试

### 16. 基准测试报告生成
- [x] 16.1 创建mm_orch/benchmark/reporter.py
- [x] 16.2 实现JSON报告生成
- [x] 16.3 实现CSV报告生成
- [x] 16.4 实现系统信息收集
- [x] 16.5 编写报告生成单元测试

---

## 第六阶段：端到端验证 (E2E Validation)

### 17. E2E验证器实现
- [x] 17.1 创建mm_orch/validation/e2e_validator.py
- [x] 17.2 实现SearchQA场景验证
- [x] 17.3 实现LessonPack场景验证
- [x] 17.4 实现多轮对话验证
- [x] 17.5 编写E2E验证器单元测试

### 18. 测试用例集
- [x] 18.1 创建tests/fixtures/search_qa_cases.json
- [x] 18.2 创建tests/fixtures/lesson_pack_cases.json
- [x] 18.3 创建tests/fixtures/conversation_cases.json
- [x] 18.4 编写测试用例加载器

---

## 第七阶段：CLI与API集成 (Integration)

### 19. CLI更新
- [ ] 19.1 更新mm_orch/main.py添加模型选择参数
- [ ] 19.2 添加基准测试命令
- [ ] 19.3 添加模型信息查看命令
- [ ] 19.4 编写CLI集成测试

### 20. API更新
- [ ] 20.1 更新mm_orch/api/routes.py添加模型管理端点
- [ ] 20.2 添加基准测试端点
- [ ] 20.3 添加模型状态查询端点
- [ ] 20.4 编写API集成测试

---

## 第八阶段：文档与最终验证 (Documentation & Final)

### 21. 文档更新
- [ ] 21.1 更新README.md添加真实模型使用说明
- [ ] 21.2 创建docs/real_model_guide.md使用指南
- [ ] 21.3 创建docs/benchmark_guide.md基准测试指南
- [ ] 21.4 更新API文档

### 22. 最终检查点
- [ ] 22.1 运行完整测试套件
- [ ] 22.2 验证测试覆盖率≥80%
- [ ] 22.3 运行GPT-2模型端到端测试
- [ ] 22.4 生成性能基准报告
- [ ] 22.5 代码审查和清理

---

## 可选任务 (Optional)

### 23. 高级优化
- [ ]* 23.1 实现vLLM集成（高吞吐推理）
- [ ]* 23.2 实现DeepSpeed推理优化
- [ ]* 23.3 实现ONNX Runtime支持
- [ ]* 23.4 实现模型并行（多GPU）

### 24. 监控与可观测性
- [ ]* 24.1 实现Prometheus指标导出
- [ ]* 24.2 实现推理追踪（OpenTelemetry）
- [ ]* 24.3 实现性能仪表板

---

## 任务依赖关系

```
1 (配置) ──┬──> 2 (异常) ──> 3 (量化) ──> 4 (加载器) ──┐
           │                                           │
           └──> 5 (内存监控) ─────────────────────────>├──> 6 (ModelManager)
                                                       │
7 (FlashAttn) ─────────────────────────────────────────┘
                                                       │
                                                       v
                                              8 (推理引擎) ──> 9 (对话管理)
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    v                                  v                                  v
           10 (SearchQA)                      11 (LessonPack)                    12 (ChatGenerate)
                    │                                  │                                  │
                    └──────────────────────────────────┼──────────────────────────────────┘
                                                       │
                    ┌──────────────────────────────────┼──────────────────────────────────┐
                    │                                  │                                  │
                    v                                  v                                  v
           13 (延迟基准)                       14 (内存基准)                      15 (吞吐基准)
                    │                                  │                                  │
                    └──────────────────────────────────┼──────────────────────────────────┘
                                                       │
                                                       v
                                              16 (报告生成)
                                                       │
                                                       v
                                              17 (E2E验证) ──> 18 (测试用例)
                                                       │
                    ┌──────────────────────────────────┴──────────────────────────────────┐
                    │                                                                      │
                    v                                                                      v
              19 (CLI更新)                                                          20 (API更新)
                    │                                                                      │
                    └──────────────────────────────────┬──────────────────────────────────┘
                                                       │
                                                       v
                                              21 (文档) ──> 22 (最终检查)
```

---

## 估算工时

| 阶段 | 任务数 | 估算工时 |
|------|--------|----------|
| 第一阶段：基础设施 | 8 | 4小时 |
| 第二阶段：模型管理 | 19 | 16小时 |
| 第三阶段：推理引擎 | 18 | 14小时 |
| 第四阶段：工作流集成 | 12 | 10小时 |
| 第五阶段：性能基准 | 17 | 12小时 |
| 第六阶段：E2E验证 | 9 | 8小时 |
| 第七阶段：CLI/API集成 | 8 | 6小时 |
| 第八阶段：文档与最终 | 9 | 6小时 |
| **总计** | **100** | **76小时** |
