"""
MuAI多模型编排系统 - 命令行接口

提供交互式对话模式和单次查询模式的命令行工具。

需求: 1.1
支持真实模型集成和基准测试。
"""

import argparse
import sys
import uuid
import json
from typing import Optional, Dict, Any

from mm_orch.schemas import UserRequest, WorkflowType
from mm_orch.orchestrator import get_orchestrator, create_orchestrator
from mm_orch.router import get_router
from mm_orch.consciousness.core import get_consciousness
from mm_orch.logger import get_logger, configure_logger

# Phase B integration with fallback
try:
    from mm_orch.orchestration.phase_b_orchestrator import get_phase_b_orchestrator

    PHASE_B_AVAILABLE = True
except ImportError:
    PHASE_B_AVAILABLE = False


logger = get_logger(__name__)


class CLI:
    """
    命令行接口类

    支持:
    - 单次查询模式: 执行单个查询并返回结果
    - 交互式对话模式: 持续对话直到用户退出
    - 指定工作流模式: 直接指定要使用的工作流
    - 模型选择: 指定使用的模型
    - 基准测试: 运行性能基准测试
    """

    def __init__(
        self,
        orchestrator=None,
        verbose: bool = False,
        model: Optional[str] = None,
        use_real_models: bool = False,
        use_phase_b: bool = False,
    ):
        """
        初始化CLI

        Args:
            orchestrator: 工作流编排器实例
            verbose: 是否显示详细输出
            model: 指定使用的模型名称
            use_real_models: 是否使用真实模型
            use_phase_b: 是否使用Phase B orchestrator (with fallback to Phase A)
        """
        # Use Phase B orchestrator if requested and available
        if use_phase_b and PHASE_B_AVAILABLE:
            logger.info("Using Phase B orchestrator with Phase A fallback")
            self.orchestrator = orchestrator or get_phase_b_orchestrator()
            self.using_phase_b = True
        else:
            if use_phase_b and not PHASE_B_AVAILABLE:
                logger.warning("Phase B requested but not available, using Phase A")
            self.orchestrator = orchestrator or get_orchestrator()
            self.using_phase_b = False

        self.verbose = verbose
        self.model = model
        self.use_real_models = use_real_models
        self.session_id: Optional[str] = None

    def run_single_query(self, query: str, workflow: Optional[str] = None) -> str:
        """
        执行单次查询

        Args:
            query: 用户查询
            workflow: 可选的工作流类型

        Returns:
            str: 查询结果
        """
        try:
            if workflow:
                # 指定工作流模式
                workflow_type = self._parse_workflow_type(workflow)
                if workflow_type is None:
                    return f"错误: 未知的工作流类型 '{workflow}'"

                result = self.orchestrator.execute_workflow(
                    workflow_type=workflow_type,
                    parameters={"query": query, "topic": query, "message": query},
                )
            else:
                # 自动路由模式
                request = UserRequest(query=query)
                result = self.orchestrator.process_request(request)

            if result.status == "success":
                return self._format_result(result.result)
            elif result.status == "partial":
                output = self._format_result(result.result)
                if result.error:
                    output += f"\n\n[警告: {result.error}]"
                return output
            else:
                return f"错误: {result.error or '未知错误'}"

        except Exception as e:
            logger.error("Query execution failed", error=str(e))
            return f"执行错误: {str(e)}"

    def run_interactive(self) -> None:
        """
        运行交互式对话模式
        """
        self.session_id = str(uuid.uuid4())

        print("\n" + "=" * 60)
        print("MuAI多模型编排系统 - 交互式对话模式")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助")
        print("输入 'status' 查看系统状态")
        print("输入 'workflow <type>' 切换工作流模式")
        print("=" * 60 + "\n")

        current_workflow: Optional[str] = None

        while True:
            try:
                # 显示提示符
                prompt = f"[{current_workflow or 'auto'}] > " if current_workflow else "> "
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # 处理特殊命令
                lower_input = user_input.lower()

                if lower_input in ("quit", "exit", "q"):
                    print("\n再见！")
                    break

                if lower_input == "help":
                    self._print_help()
                    continue

                if lower_input == "status":
                    self._print_status()
                    continue

                if lower_input.startswith("workflow "):
                    workflow_name = user_input[9:].strip()
                    if workflow_name == "auto":
                        current_workflow = None
                        print("已切换到自动路由模式")
                    elif self._parse_workflow_type(workflow_name):
                        current_workflow = workflow_name
                        print(f"已切换到 {workflow_name} 工作流")
                    else:
                        print(f"未知的工作流类型: {workflow_name}")
                        self._print_available_workflows()
                    continue

                if lower_input == "workflows":
                    self._print_available_workflows()
                    continue

                if lower_input == "clear":
                    self.session_id = str(uuid.uuid4())
                    print("已清除对话历史，开始新会话")
                    continue

                # 执行查询
                print()  # 空行
                result = self._execute_interactive_query(user_input, current_workflow)
                print(result)
                print()  # 空行

            except KeyboardInterrupt:
                print("\n\n已中断，再见！")
                break
            except EOFError:
                print("\n再见！")
                break

    def _execute_interactive_query(self, query: str, workflow: Optional[str] = None) -> str:
        """
        执行交互式查询

        Args:
            query: 用户查询
            workflow: 可选的工作流类型

        Returns:
            str: 查询结果
        """
        try:
            if workflow:
                workflow_type = self._parse_workflow_type(workflow)
                if workflow_type is None:
                    return f"错误: 未知的工作流类型 '{workflow}'"

                params = {
                    "query": query,
                    "topic": query,
                    "message": query,
                    "session_id": self.session_id,
                }
                result = self.orchestrator.execute_workflow(
                    workflow_type=workflow_type, parameters=params
                )
            else:
                request = UserRequest(query=query, session_id=self.session_id)
                result = self.orchestrator.process_request(request)

            # 格式化输出
            output_parts = []

            if self.verbose and result.metadata:
                routing_info = result.metadata.get("routing", {})
                if routing_info:
                    output_parts.append(
                        f"[路由: {routing_info.get('workflow_type', 'unknown')}, "
                        f"置信度: {routing_info.get('confidence', 0):.2f}]"
                    )

                exec_time = result.metadata.get("execution_time")
                if exec_time:
                    output_parts.append(f"[耗时: {exec_time:.2f}s]")

            if result.status == "success":
                output_parts.append(self._format_result(result.result))
            elif result.status == "partial":
                output_parts.append(self._format_result(result.result))
                if result.error:
                    output_parts.append(f"\n[警告: {result.error}]")
            else:
                output_parts.append(f"错误: {result.error or '未知错误'}")

            return "\n".join(output_parts)

        except Exception as e:
            logger.error("Interactive query failed", error=str(e))
            return f"执行错误: {str(e)}"

    def _format_result(self, result) -> str:
        """
        格式化结果输出

        Args:
            result: 工作流结果

        Returns:
            str: 格式化的字符串
        """
        if result is None:
            return "[无结果]"

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            # 处理结构化教学包结果
            if "lesson_explain_structured" in result:
                return self._format_structured_lesson(result["lesson_explain_structured"])

            # 处理传统教学包结果
            if "plan" in result and "explanation" in result:
                parts = []
                parts.append("=== 教学计划 ===")
                parts.append(result.get("plan", ""))
                parts.append("\n=== 讲解内容 ===")
                parts.append(result.get("explanation", ""))

                exercises = result.get("exercises", [])
                if exercises:
                    parts.append("\n=== 练习题 ===")
                    for i, ex in enumerate(exercises, 1):
                        parts.append(f"\n题目 {i}: {ex.get('question', '')}")
                        parts.append(f"答案: {ex.get('answer', '')}")

                return "\n".join(parts)

            # 处理RAG结果
            if "answer" in result and "sources" in result:
                parts = [result.get("answer", "")]
                sources = result.get("sources", [])
                if sources:
                    parts.append("\n--- 来源 ---")
                    for src in sources[:3]:
                        parts.append(f"- {src}")
                return "\n".join(parts)

            # 通用字典格式化
            return str(result)

        return str(result)

    def _format_structured_lesson(self, lesson_data: Dict[str, Any]) -> str:
        """
        Format structured lesson for CLI display.

        Args:
            lesson_data: StructuredLesson JSON dictionary

        Returns:
            Formatted string with clear sections, numbered examples, and bullet points

        Requirements:
            - 20.3: List examples with numbering
            - 20.4: Display key points as bullet points
        """
        from mm_orch.workflows.lesson_structure import StructuredLesson

        try:
            lesson = StructuredLesson.from_json(lesson_data)

            parts = []

            # Display topic and grade prominently
            parts.append("=" * 60)
            parts.append(f"主题: {lesson.topic}")
            parts.append(f"年级/难度: {lesson.grade}")
            parts.append("=" * 60)
            parts.append("")

            # Display each section with clear headers
            for i, section in enumerate(lesson.sections, 1):
                parts.append(f"【第{i}部分：{section.name}】")
                parts.append("-" * 60)
                parts.append("")

                # Teacher content
                parts.append("教师讲解:")
                parts.append(section.teacher_say)
                parts.append("")

                # Student responses (if present)
                if section.student_may_say:
                    parts.append("学生可能的回答:")
                    parts.append(section.student_may_say)
                    parts.append("")

                # Examples with numbering
                if section.examples:
                    parts.append("示例:")
                    for j, example in enumerate(section.examples, 1):
                        parts.append(f"  {j}. {example}")
                    parts.append("")

                # Questions with numbering
                if section.questions:
                    parts.append("问题:")
                    for j, question in enumerate(section.questions, 1):
                        parts.append(f"  {j}. {question}")
                    parts.append("")

                # Key points as bullet points
                if section.key_points:
                    parts.append("要点:")
                    for point in section.key_points:
                        parts.append(f"  • {point}")
                    parts.append("")

                # Teaching tips (if present)
                if section.tips:
                    parts.append("教学提示:")
                    parts.append(f"  💡 {section.tips}")
                    parts.append("")

            parts.append("=" * 60)

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"Failed to format structured lesson: {e}")
            # Fallback to simple dict display
            return str(lesson_data)

    def _parse_workflow_type(self, workflow_name: str) -> Optional[WorkflowType]:
        """
        解析工作流类型名称

        Args:
            workflow_name: 工作流名称

        Returns:
            WorkflowType或None
        """
        name_lower = workflow_name.lower().strip()

        # 支持多种命名方式
        mapping = {
            "search_qa": WorkflowType.SEARCH_QA,
            "searchqa": WorkflowType.SEARCH_QA,
            "search": WorkflowType.SEARCH_QA,
            "lesson_pack": WorkflowType.LESSON_PACK,
            "lessonpack": WorkflowType.LESSON_PACK,
            "lesson": WorkflowType.LESSON_PACK,
            "teach": WorkflowType.LESSON_PACK,
            "chat_generate": WorkflowType.CHAT_GENERATE,
            "chatgenerate": WorkflowType.CHAT_GENERATE,
            "chat": WorkflowType.CHAT_GENERATE,
            "rag_qa": WorkflowType.RAG_QA,
            "ragqa": WorkflowType.RAG_QA,
            "rag": WorkflowType.RAG_QA,
            "self_ask_search_qa": WorkflowType.SELF_ASK_SEARCH_QA,
            "selfasksearchqa": WorkflowType.SELF_ASK_SEARCH_QA,
            "self_ask": WorkflowType.SELF_ASK_SEARCH_QA,
            "selfask": WorkflowType.SELF_ASK_SEARCH_QA,
        }

        return mapping.get(name_lower)

    def _print_help(self) -> None:
        """打印帮助信息"""
        help_text = """
命令帮助:
  quit, exit, q    - 退出程序
  help             - 显示此帮助信息
  status           - 显示系统状态
  workflows        - 显示可用的工作流类型
  workflow <type>  - 切换到指定工作流模式
  workflow auto    - 切换回自动路由模式
  clear            - 清除对话历史，开始新会话

工作流类型:
  search_qa        - 搜索问答（网络搜索）
  lesson_pack      - 教学内容生成
  chat_generate    - 多轮对话
  rag_qa           - 知识库问答
  self_ask_search_qa - 复杂问题分解搜索

示例:
  > 什么是人工智能？
  > workflow lesson
  > 教我Python基础
  > workflow auto
"""
        print(help_text)

    def _print_status(self) -> None:
        """打印系统状态"""
        stats = self.orchestrator.get_statistics()

        print("\n=== 系统状态 ===")
        print(f"会话ID: {self.session_id or '无'}")
        print(f"已注册工作流: {stats.get('registered_workflows', 0)}")
        print(f"执行次数: {stats.get('execution_count', 0)}")
        print(f"成功率: {stats.get('success_rate', 0):.1%}")
        print(f"平均耗时: {stats.get('average_execution_time', 0):.2f}s")

        # 意识状态
        try:
            consciousness = get_consciousness()
            status = consciousness.get_status_summary()
            print(f"\n=== 意识状态 ===")
            print(f"发展阶段: {status.get('development_stage', 'unknown')}")
            emotion = status.get("emotion_state", {})
            print(
                f"情感状态: valence={emotion.get('valence', 0):.2f}, arousal={emotion.get('arousal', 0):.2f}"
            )
        except Exception:
            pass

        print()

    def _print_available_workflows(self) -> None:
        """打印可用的工作流"""
        print("\n可用的工作流类型:")
        print("  search_qa (search)     - 搜索问答")
        print("  lesson_pack (lesson)   - 教学内容生成")
        print("  chat_generate (chat)   - 多轮对话")
        print("  rag_qa (rag)           - 知识库问答")
        print("  self_ask_search_qa     - 复杂问题分解")
        print("  auto                   - 自动路由（默认）")
        print()


def run_benchmark(
    model_name: str = "gpt2", output_dir: str = "data/benchmarks", output_format: str = "json"
) -> int:
    """
    运行基准测试

    Args:
        model_name: 要测试的模型名称
        output_dir: 输出目录
        output_format: 输出格式 (json/csv)

    Returns:
        int: 退出码
    """
    try:
        from mm_orch.benchmark.latency import LatencyBenchmark
        from mm_orch.benchmark.memory import MemoryBenchmark
        from mm_orch.benchmark.throughput import ThroughputBenchmark
        from mm_orch.benchmark.reporter import BenchmarkReporter

        print(f"\n=== 基准测试: {model_name} ===\n")

        # 创建基准测试实例
        latency_bench = LatencyBenchmark()
        memory_bench = MemoryBenchmark()
        throughput_bench = ThroughputBenchmark()
        reporter = BenchmarkReporter(output_dir=output_dir)

        results = []

        # 运行延迟测试
        print("运行延迟测试...")
        try:
            latency_result = latency_bench.run_benchmark(
                model_name=model_name,
                test_prompts=["Hello, how are you?", "What is Python?"],
                num_runs=3,
            )
            results.append(latency_result)
            print(f"  TTFT: {latency_result.metrics.get('avg_ttft', 0):.3f}s")
            print(f"  Tokens/s: {latency_result.metrics.get('avg_tokens_per_second', 0):.1f}")
        except Exception as e:
            print(f"  延迟测试失败: {e}")

        # 运行内存测试
        print("\n运行内存测试...")
        try:
            memory_result = memory_bench.run_benchmark(model_name=model_name)
            results.append(memory_result)
            print(f"  模型内存: {memory_result.metrics.get('model_memory_mb', 0):.1f} MB")
            print(f"  峰值内存: {memory_result.metrics.get('peak_memory_mb', 0):.1f} MB")
        except Exception as e:
            print(f"  内存测试失败: {e}")

        # 运行吞吐量测试
        print("\n运行吞吐量测试...")
        try:
            throughput_result = throughput_bench.run_benchmark(
                model_name=model_name, num_requests=5, concurrent_levels=[1, 2]
            )
            results.append(throughput_result)
            print(
                f"  单请求吞吐: {throughput_result.metrics.get('single_throughput', 0):.1f} tokens/s"
            )
        except Exception as e:
            print(f"  吞吐量测试失败: {e}")

        # 生成报告
        if results:
            print("\n生成报告...")
            if output_format == "json":
                report_path = reporter.generate_json_report(results)
            else:
                report_path = reporter.generate_csv_report(results)
            print(f"报告已保存到: {report_path}")

        print("\n=== 基准测试完成 ===\n")
        return 0

    except ImportError as e:
        print(f"错误: 缺少基准测试模块 - {e}")
        return 1
    except Exception as e:
        print(f"基准测试错误: {e}")
        return 1


def show_model_info(model_name: Optional[str] = None) -> int:
    """
    显示模型信息

    Args:
        model_name: 模型名称（可选，不提供则显示所有可用模型）

    Returns:
        int: 退出码
    """
    try:
        import yaml
        from pathlib import Path

        # 读取模型配置
        config_path = Path("config/models.yaml")
        if not config_path.exists():
            print("错误: 未找到模型配置文件 config/models.yaml")
            return 1

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        models = config.get("models", {})

        if model_name:
            # 显示特定模型信息
            if model_name not in models:
                print(f"错误: 未找到模型 '{model_name}'")
                print(f"可用模型: {', '.join(models.keys())}")
                return 1

            model_config = models[model_name]
            print(f"\n=== 模型信息: {model_name} ===\n")
            print(f"  HuggingFace名称: {model_config.get('model_name', 'N/A')}")
            print(f"  模型类型: {model_config.get('model_type', 'N/A')}")
            print(f"  设备: {model_config.get('device', 'auto')}")
            print(f"  数据类型: {model_config.get('dtype', 'auto')}")
            print(f"  量化: {model_config.get('quantization', '无')}")
            print(f"  FlashAttention: {model_config.get('flash_attention', False)}")
            print(f"  最大上下文长度: {model_config.get('max_context_length', 'N/A')}")
            print()
        else:
            # 显示所有可用模型
            print("\n=== 可用模型 ===\n")
            for name, cfg in models.items():
                quant = cfg.get("quantization", "")
                quant_str = f" ({quant})" if quant else ""
                print(f"  {name}: {cfg.get('model_type', 'unknown')}{quant_str}")
            print()
            print("使用 --model-info <model_name> 查看详细信息")
            print()

        return 0

    except Exception as e:
        print(f"错误: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    Returns:
        argparse.ArgumentParser: 参数解析器
    """
    parser = argparse.ArgumentParser(
        prog="mm_orch",
        description="MuAI多模型编排系统 - 命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次查询（自动路由）
  python -m mm_orch "什么是人工智能？"
  
  # 指定工作流
  python -m mm_orch --workflow search_qa "最新的AI新闻"
  python -m mm_orch -w lesson "Python基础教程"
  
  # 使用真实模型
  python -m mm_orch --real-models --model qwen-7b-chat "你好"
  
  # 交互式对话模式
  python -m mm_orch --mode chat
  python -m mm_orch -m chat
  
  # 运行基准测试
  python -m mm_orch --benchmark --model gpt2
  
  # 查看模型信息
  python -m mm_orch --model-info
  python -m mm_orch --model-info gpt2
  
  # 详细输出
  python -m mm_orch -v "你好"
""",
    )

    parser.add_argument("query", nargs="?", help="要执行的查询（如果不提供则进入交互模式）")

    parser.add_argument(
        "-m",
        "--mode",
        choices=["query", "chat"],
        default="query",
        help="运行模式: query=单次查询, chat=交互式对话 (默认: query)",
    )

    parser.add_argument(
        "-w",
        "--workflow",
        choices=[
            "search_qa",
            "lesson_pack",
            "chat_generate",
            "rag_qa",
            "self_ask_search_qa",
            "auto",
        ],
        default=None,
        help="指定工作流类型（默认: 自动路由）",
    )

    # 模型相关参数
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定使用的模型名称（如 gpt2, qwen-7b-chat）",
    )

    parser.add_argument(
        "--real-models",
        action="store_true",
        help="使用真实模型进行推理（需要GPU或足够的内存）",
    )

    # Phase B integration
    parser.add_argument(
        "--phase-b",
        action="store_true",
        help="使用Phase B orchestrator (graph-based execution with fallback to Phase A)",
    )

    # 基准测试参数
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="运行性能基准测试",
    )

    parser.add_argument(
        "--benchmark-output",
        type=str,
        default="data/benchmarks",
        help="基准测试输出目录（默认: data/benchmarks）",
    )

    parser.add_argument(
        "--benchmark-format",
        choices=["json", "csv"],
        default="json",
        help="基准测试报告格式（默认: json）",
    )

    # 模型信息
    parser.add_argument(
        "--model-info",
        nargs="?",
        const="",
        default=None,
        help="显示模型信息（不带参数显示所有模型，带参数显示特定模型）",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="显示详细输出（包括路由信息和执行时间）"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="日志级别 (默认: WARNING)",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    return parser


def main(args=None) -> int:
    """
    主入口函数

    Args:
        args: 命令行参数（用于测试）

    Returns:
        int: 退出码
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    # 配置日志
    configure_logger(level=parsed_args.log_level)

    try:
        # 处理模型信息命令
        if parsed_args.model_info is not None:
            model_name = parsed_args.model_info if parsed_args.model_info else None
            return show_model_info(model_name)

        # 处理基准测试命令
        if parsed_args.benchmark:
            model_name = parsed_args.model or "gpt2"
            return run_benchmark(
                model_name=model_name,
                output_dir=parsed_args.benchmark_output,
                output_format=parsed_args.benchmark_format,
            )

        # 创建CLI实例
        cli = CLI(
            verbose=parsed_args.verbose,
            model=parsed_args.model,
            use_real_models=parsed_args.real_models,
            use_phase_b=parsed_args.phase_b,
        )

        # 确定运行模式
        if parsed_args.mode == "chat" or (not parsed_args.query and parsed_args.mode != "query"):
            # 交互式模式
            cli.run_interactive()
            return 0

        if parsed_args.query:
            # 单次查询模式
            workflow = parsed_args.workflow if parsed_args.workflow != "auto" else None
            result = cli.run_single_query(parsed_args.query, workflow)
            print(result)
            return 0

        # 没有查询也没有指定chat模式，进入交互模式
        cli.run_interactive()
        return 0

    except KeyboardInterrupt:
        print("\n已中断")
        return 130
    except Exception as e:
        logger.error("CLI error", error=str(e))
        print(f"错误: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
