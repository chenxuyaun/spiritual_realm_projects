"""
MuAI多模型编排系统 - 命令行接口

提供交互式对话模式和单次查询模式的命令行工具。

需求: 1.1
"""

import argparse
import sys
import uuid
from typing import Optional

from mm_orch.schemas import UserRequest, WorkflowType
from mm_orch.orchestrator import get_orchestrator, create_orchestrator
from mm_orch.router import get_router
from mm_orch.consciousness.core import get_consciousness
from mm_orch.logger import get_logger, configure_logger
from mm_orch.config import get_config


logger = get_logger(__name__)


class CLI:
    """
    命令行接口类
    
    支持:
    - 单次查询模式: 执行单个查询并返回结果
    - 交互式对话模式: 持续对话直到用户退出
    - 指定工作流模式: 直接指定要使用的工作流
    """
    
    def __init__(
        self,
        orchestrator=None,
        verbose: bool = False
    ):
        """
        初始化CLI
        
        Args:
            orchestrator: 工作流编排器实例
            verbose: 是否显示详细输出
        """
        self.orchestrator = orchestrator or get_orchestrator()
        self.verbose = verbose
        self.session_id: Optional[str] = None
        
    def run_single_query(
        self,
        query: str,
        workflow: Optional[str] = None
    ) -> str:
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
                    parameters={"query": query, "topic": query, "message": query}
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
                
                if lower_input in ('quit', 'exit', 'q'):
                    print("\n再见！")
                    break
                
                if lower_input == 'help':
                    self._print_help()
                    continue
                
                if lower_input == 'status':
                    self._print_status()
                    continue
                
                if lower_input.startswith('workflow '):
                    workflow_name = user_input[9:].strip()
                    if workflow_name == 'auto':
                        current_workflow = None
                        print("已切换到自动路由模式")
                    elif self._parse_workflow_type(workflow_name):
                        current_workflow = workflow_name
                        print(f"已切换到 {workflow_name} 工作流")
                    else:
                        print(f"未知的工作流类型: {workflow_name}")
                        self._print_available_workflows()
                    continue
                
                if lower_input == 'workflows':
                    self._print_available_workflows()
                    continue
                
                if lower_input == 'clear':
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
    
    def _execute_interactive_query(
        self,
        query: str,
        workflow: Optional[str] = None
    ) -> str:
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
                    "session_id": self.session_id
                }
                result = self.orchestrator.execute_workflow(
                    workflow_type=workflow_type,
                    parameters=params
                )
            else:
                request = UserRequest(
                    query=query,
                    session_id=self.session_id
                )
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
            # 处理教学包结果
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
            emotion = status.get('emotion_state', {})
            print(f"情感状态: valence={emotion.get('valence', 0):.2f}, arousal={emotion.get('arousal', 0):.2f}")
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
  
  # 交互式对话模式
  python -m mm_orch --mode chat
  python -m mm_orch -m chat
  
  # 详细输出
  python -m mm_orch -v "你好"
"""
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="要执行的查询（如果不提供则进入交互模式）"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["query", "chat"],
        default="query",
        help="运行模式: query=单次查询, chat=交互式对话 (默认: query)"
    )
    
    parser.add_argument(
        "-w", "--workflow",
        choices=[
            "search_qa", "lesson_pack", "chat_generate",
            "rag_qa", "self_ask_search_qa", "auto"
        ],
        default=None,
        help="指定工作流类型（默认: 自动路由）"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出（包括路由信息和执行时间）"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="日志级别 (默认: WARNING)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
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
        # 创建CLI实例
        cli = CLI(verbose=parsed_args.verbose)
        
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
