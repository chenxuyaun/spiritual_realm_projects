"""
CLI单元测试

测试命令行接口的各种功能：
- 命令解析
- 单次查询模式
- 交互模式命令处理
- 工作流类型解析
- 结果格式化

需求: 1.1
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

from mm_orch.main import CLI, create_parser, main
from mm_orch.schemas import WorkflowResult, WorkflowType, UserRequest


class TestCLIParser:
    """测试命令行参数解析"""
    
    def test_parser_creation(self):
        """测试解析器创建"""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "mm_orch"
    
    def test_parse_query_argument(self):
        """测试解析查询参数"""
        parser = create_parser()
        args = parser.parse_args(["什么是AI？"])
        assert args.query == "什么是AI？"
        assert args.mode == "query"
    
    def test_parse_mode_chat(self):
        """测试解析chat模式"""
        parser = create_parser()
        args = parser.parse_args(["--mode", "chat"])
        assert args.mode == "chat"
        assert args.query is None
    
    def test_parse_mode_short(self):
        """测试解析短格式模式参数"""
        parser = create_parser()
        args = parser.parse_args(["-m", "chat"])
        assert args.mode == "chat"
    
    def test_parse_workflow(self):
        """测试解析工作流参数"""
        parser = create_parser()
        args = parser.parse_args(["--workflow", "search_qa", "test query"])
        assert args.workflow == "search_qa"
        assert args.query == "test query"
    
    def test_parse_workflow_short(self):
        """测试解析短格式工作流参数"""
        parser = create_parser()
        args = parser.parse_args(["-w", "lesson_pack", "Python教程"])
        assert args.workflow == "lesson_pack"
    
    def test_parse_verbose(self):
        """测试解析详细输出参数"""
        parser = create_parser()
        args = parser.parse_args(["-v", "test"])
        assert args.verbose is True
    
    def test_parse_log_level(self):
        """测试解析日志级别参数"""
        parser = create_parser()
        args = parser.parse_args(["--log-level", "DEBUG", "test"])
        assert args.log_level == "DEBUG"
    
    def test_parse_default_values(self):
        """测试默认值"""
        parser = create_parser()
        args = parser.parse_args(["test"])
        assert args.mode == "query"
        assert args.workflow is None
        assert args.verbose is False
        assert args.log_level == "WARNING"
    
    def test_parse_all_workflow_types(self):
        """测试所有工作流类型都可以解析"""
        parser = create_parser()
        workflow_types = [
            "search_qa", "lesson_pack", "chat_generate",
            "rag_qa", "self_ask_search_qa", "auto"
        ]
        for wf in workflow_types:
            args = parser.parse_args(["-w", wf, "test"])
            assert args.workflow == wf


class TestCLI:
    """测试CLI类"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """创建模拟的编排器"""
        orchestrator = Mock()
        orchestrator.get_statistics.return_value = {
            "registered_workflows": 5,
            "execution_count": 10,
            "success_rate": 0.9,
            "average_execution_time": 1.5
        }
        return orchestrator
    
    @pytest.fixture
    def cli(self, mock_orchestrator):
        """创建CLI实例"""
        return CLI(orchestrator=mock_orchestrator, verbose=False)
    
    def test_cli_initialization(self, cli, mock_orchestrator):
        """测试CLI初始化"""
        assert cli.orchestrator == mock_orchestrator
        assert cli.verbose is False
        assert cli.session_id is None
    
    def test_cli_verbose_mode(self, mock_orchestrator):
        """测试详细模式"""
        cli = CLI(orchestrator=mock_orchestrator, verbose=True)
        assert cli.verbose is True
    
    def test_parse_workflow_type_search_qa(self, cli):
        """测试解析search_qa工作流"""
        assert cli._parse_workflow_type("search_qa") == WorkflowType.SEARCH_QA
        assert cli._parse_workflow_type("searchqa") == WorkflowType.SEARCH_QA
        assert cli._parse_workflow_type("search") == WorkflowType.SEARCH_QA
        assert cli._parse_workflow_type("SEARCH_QA") == WorkflowType.SEARCH_QA
    
    def test_parse_workflow_type_lesson_pack(self, cli):
        """测试解析lesson_pack工作流"""
        assert cli._parse_workflow_type("lesson_pack") == WorkflowType.LESSON_PACK
        assert cli._parse_workflow_type("lessonpack") == WorkflowType.LESSON_PACK
        assert cli._parse_workflow_type("lesson") == WorkflowType.LESSON_PACK
        assert cli._parse_workflow_type("teach") == WorkflowType.LESSON_PACK
    
    def test_parse_workflow_type_chat_generate(self, cli):
        """测试解析chat_generate工作流"""
        assert cli._parse_workflow_type("chat_generate") == WorkflowType.CHAT_GENERATE
        assert cli._parse_workflow_type("chatgenerate") == WorkflowType.CHAT_GENERATE
        assert cli._parse_workflow_type("chat") == WorkflowType.CHAT_GENERATE
    
    def test_parse_workflow_type_rag_qa(self, cli):
        """测试解析rag_qa工作流"""
        assert cli._parse_workflow_type("rag_qa") == WorkflowType.RAG_QA
        assert cli._parse_workflow_type("ragqa") == WorkflowType.RAG_QA
        assert cli._parse_workflow_type("rag") == WorkflowType.RAG_QA
    
    def test_parse_workflow_type_self_ask(self, cli):
        """测试解析self_ask_search_qa工作流"""
        assert cli._parse_workflow_type("self_ask_search_qa") == WorkflowType.SELF_ASK_SEARCH_QA
        assert cli._parse_workflow_type("selfasksearchqa") == WorkflowType.SELF_ASK_SEARCH_QA
        assert cli._parse_workflow_type("self_ask") == WorkflowType.SELF_ASK_SEARCH_QA
        assert cli._parse_workflow_type("selfask") == WorkflowType.SELF_ASK_SEARCH_QA
    
    def test_parse_workflow_type_unknown(self, cli):
        """测试解析未知工作流类型"""
        assert cli._parse_workflow_type("unknown") is None
        assert cli._parse_workflow_type("invalid") is None
        assert cli._parse_workflow_type("") is None
    
    def test_format_result_string(self, cli):
        """测试格式化字符串结果"""
        result = cli._format_result("Hello, World!")
        assert result == "Hello, World!"
    
    def test_format_result_none(self, cli):
        """测试格式化None结果"""
        result = cli._format_result(None)
        assert result == "[无结果]"
    
    def test_format_result_lesson_pack(self, cli):
        """测试格式化教学包结果"""
        lesson_result = {
            "plan": "教学计划内容",
            "explanation": "讲解内容",
            "exercises": [
                {"question": "问题1", "answer": "答案1"},
                {"question": "问题2", "answer": "答案2"}
            ]
        }
        result = cli._format_result(lesson_result)
        assert "教学计划" in result
        assert "讲解内容" in result
        assert "练习题" in result
        assert "问题1" in result
        assert "答案1" in result
    
    def test_format_result_rag(self, cli):
        """测试格式化RAG结果"""
        rag_result = {
            "answer": "这是答案",
            "sources": ["来源1", "来源2"]
        }
        result = cli._format_result(rag_result)
        assert "这是答案" in result
        assert "来源" in result
        assert "来源1" in result
    
    def test_format_result_dict(self, cli):
        """测试格式化普通字典结果"""
        dict_result = {"key": "value"}
        result = cli._format_result(dict_result)
        assert "key" in result
        assert "value" in result


class TestCLISingleQuery:
    """测试单次查询模式"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """创建模拟的编排器"""
        return Mock()
    
    @pytest.fixture
    def cli(self, mock_orchestrator):
        """创建CLI实例"""
        return CLI(orchestrator=mock_orchestrator)
    
    def test_single_query_success(self, cli, mock_orchestrator):
        """测试成功的单次查询"""
        mock_orchestrator.process_request.return_value = WorkflowResult(
            result="这是答案",
            metadata={},
            status="success"
        )
        
        result = cli.run_single_query("什么是AI？")
        assert result == "这是答案"
        mock_orchestrator.process_request.assert_called_once()
    
    def test_single_query_partial(self, cli, mock_orchestrator):
        """测试部分成功的查询"""
        mock_orchestrator.process_request.return_value = WorkflowResult(
            result="部分结果",
            metadata={},
            status="partial",
            error="部分失败"
        )
        
        result = cli.run_single_query("测试查询")
        assert "部分结果" in result
        assert "警告" in result
        assert "部分失败" in result
    
    def test_single_query_failed(self, cli, mock_orchestrator):
        """测试失败的查询"""
        mock_orchestrator.process_request.return_value = WorkflowResult(
            result=None,
            metadata={},
            status="failed",
            error="查询失败"
        )
        
        result = cli.run_single_query("测试查询")
        assert "错误" in result
        assert "查询失败" in result
    
    def test_single_query_with_workflow(self, cli, mock_orchestrator):
        """测试指定工作流的查询"""
        mock_orchestrator.execute_workflow.return_value = WorkflowResult(
            result="搜索结果",
            metadata={},
            status="success"
        )
        
        result = cli.run_single_query("测试", workflow="search_qa")
        assert result == "搜索结果"
        mock_orchestrator.execute_workflow.assert_called_once()
        call_args = mock_orchestrator.execute_workflow.call_args
        assert call_args[1]["workflow_type"] == WorkflowType.SEARCH_QA
    
    def test_single_query_unknown_workflow(self, cli, mock_orchestrator):
        """测试未知工作流类型"""
        result = cli.run_single_query("测试", workflow="unknown_workflow")
        assert "错误" in result
        assert "未知的工作流类型" in result
    
    def test_single_query_exception(self, cli, mock_orchestrator):
        """测试查询异常处理"""
        mock_orchestrator.process_request.side_effect = Exception("测试异常")
        
        result = cli.run_single_query("测试")
        assert "执行错误" in result
        assert "测试异常" in result


class TestCLIInteractive:
    """测试交互式模式"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """创建模拟的编排器"""
        orchestrator = Mock()
        orchestrator.get_statistics.return_value = {
            "registered_workflows": 5,
            "execution_count": 0,
            "success_rate": 0.0,
            "average_execution_time": 0.0
        }
        return orchestrator
    
    @pytest.fixture
    def cli(self, mock_orchestrator):
        """创建CLI实例"""
        return CLI(orchestrator=mock_orchestrator)
    
    def test_interactive_query_success(self, cli, mock_orchestrator):
        """测试交互式查询成功"""
        cli.session_id = "test-session"
        mock_orchestrator.process_request.return_value = WorkflowResult(
            result="交互结果",
            metadata={"routing": {"workflow_type": "chat_generate", "confidence": 0.9}},
            status="success"
        )
        
        result = cli._execute_interactive_query("你好")
        assert "交互结果" in result
    
    def test_interactive_query_with_workflow(self, cli, mock_orchestrator):
        """测试交互式指定工作流查询"""
        cli.session_id = "test-session"
        mock_orchestrator.execute_workflow.return_value = WorkflowResult(
            result="工作流结果",
            metadata={},
            status="success"
        )
        
        result = cli._execute_interactive_query("测试", workflow="search")
        assert "工作流结果" in result
    
    def test_interactive_verbose_output(self, mock_orchestrator):
        """测试详细输出模式"""
        cli = CLI(orchestrator=mock_orchestrator, verbose=True)
        cli.session_id = "test-session"
        mock_orchestrator.process_request.return_value = WorkflowResult(
            result="结果",
            metadata={
                "routing": {"workflow_type": "chat_generate", "confidence": 0.85},
                "execution_time": 1.23
            },
            status="success"
        )
        
        result = cli._execute_interactive_query("测试")
        assert "路由" in result
        assert "置信度" in result
        assert "耗时" in result


class TestMainFunction:
    """测试main函数"""
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_single_query(self, mock_logging, mock_cli_class):
        """测试main函数单次查询"""
        mock_cli = Mock()
        mock_cli.run_single_query.return_value = "结果"
        mock_cli_class.return_value = mock_cli
        
        with patch('builtins.print') as mock_print:
            result = main(["测试查询"])
        
        assert result == 0
        mock_cli.run_single_query.assert_called_once_with("测试查询", None)
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_with_workflow(self, mock_logging, mock_cli_class):
        """测试main函数指定工作流"""
        mock_cli = Mock()
        mock_cli.run_single_query.return_value = "结果"
        mock_cli_class.return_value = mock_cli
        
        with patch('builtins.print'):
            result = main(["-w", "search_qa", "测试"])
        
        assert result == 0
        mock_cli.run_single_query.assert_called_once_with("测试", "search_qa")
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_chat_mode(self, mock_logging, mock_cli_class):
        """测试main函数chat模式"""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        result = main(["-m", "chat"])
        
        assert result == 0
        mock_cli.run_interactive.assert_called_once()
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_no_query_enters_interactive(self, mock_logging, mock_cli_class):
        """测试无查询时进入交互模式"""
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        
        result = main([])
        
        assert result == 0
        mock_cli.run_interactive.assert_called_once()
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_keyboard_interrupt(self, mock_logging, mock_cli_class):
        """测试键盘中断处理"""
        mock_cli = Mock()
        mock_cli.run_interactive.side_effect = KeyboardInterrupt()
        mock_cli_class.return_value = mock_cli
        
        with patch('builtins.print'):
            result = main(["-m", "chat"])
        
        assert result == 130
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_exception(self, mock_logging, mock_cli_class):
        """测试异常处理"""
        mock_cli_class.side_effect = Exception("初始化失败")
        
        with patch('builtins.print'):
            result = main(["测试"])
        
        assert result == 1
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_verbose_flag(self, mock_logging, mock_cli_class):
        """测试详细输出标志"""
        mock_cli = Mock()
        mock_cli.run_single_query.return_value = "结果"
        mock_cli_class.return_value = mock_cli
        
        with patch('builtins.print'):
            main(["-v", "测试"])
        
        mock_cli_class.assert_called_once_with(verbose=True, model=None, use_real_models=False, use_phase_b=False)
    
    @patch('mm_orch.main.CLI')
    @patch('mm_orch.main.configure_logger')
    def test_main_log_level(self, mock_logging, mock_cli_class):
        """测试日志级别设置"""
        mock_cli = Mock()
        mock_cli.run_single_query.return_value = "结果"
        mock_cli_class.return_value = mock_cli
        
        with patch('builtins.print'):
            main(["--log-level", "DEBUG", "测试"])
        
        mock_logging.assert_called_once_with(level="DEBUG")


class TestCLIHelpers:
    """测试CLI辅助方法"""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """创建模拟的编排器"""
        orchestrator = Mock()
        orchestrator.get_statistics.return_value = {
            "registered_workflows": 5,
            "execution_count": 10,
            "success_rate": 0.9,
            "average_execution_time": 1.5
        }
        return orchestrator
    
    @pytest.fixture
    def cli(self, mock_orchestrator):
        """创建CLI实例"""
        return CLI(orchestrator=mock_orchestrator)
    
    def test_print_help(self, cli, capsys):
        """测试打印帮助信息"""
        cli._print_help()
        captured = capsys.readouterr()
        assert "quit" in captured.out
        assert "help" in captured.out
        assert "status" in captured.out
        assert "workflow" in captured.out
    
    def test_print_status(self, cli, mock_orchestrator, capsys):
        """测试打印状态信息"""
        cli.session_id = "test-session-id"
        
        with patch('mm_orch.main.get_consciousness') as mock_consciousness:
            mock_consciousness.return_value.get_status_summary.return_value = {
                "development_stage": "adult",
                "emotion_state": {"valence": 0.5, "arousal": 0.3}
            }
            cli._print_status()
        
        captured = capsys.readouterr()
        assert "系统状态" in captured.out
        assert "test-session-id" in captured.out
        assert "5" in captured.out  # registered_workflows
    
    def test_print_available_workflows(self, cli, capsys):
        """测试打印可用工作流"""
        cli._print_available_workflows()
        captured = capsys.readouterr()
        assert "search_qa" in captured.out
        assert "lesson_pack" in captured.out
        assert "chat_generate" in captured.out
        assert "rag_qa" in captured.out
        assert "self_ask_search_qa" in captured.out
