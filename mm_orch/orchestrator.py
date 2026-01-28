"""
Workflow Orchestrator - 工作流编排器

WorkflowOrchestrator负责协调工作流执行、意识模块交互和资源管理。
它是系统的核心编排组件，整合Router路由、工作流执行和意识核心系统。

需求: 1.2, 1.3, 1.4, 6.2
属性4: 结果结构完整性
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import traceback

from mm_orch.schemas import (
    UserRequest,
    WorkflowResult,
    WorkflowSelection,
    WorkflowType,
    SystemEvent,
    Task,
)
from mm_orch.workflows.base import BaseWorkflow
from mm_orch.workflows.search_qa import SearchQAWorkflow
from mm_orch.workflows.lesson_pack import LessonPackWorkflow
from mm_orch.workflows.chat_generate import ChatGenerateWorkflow
from mm_orch.workflows.rag_qa import RAGQAWorkflow
from mm_orch.workflows.self_ask_search_qa import SelfAskSearchQAWorkflow
from mm_orch.consciousness.core import ConsciousnessCore, get_consciousness
from mm_orch.router import Router, get_router
from mm_orch.logger import get_logger
from mm_orch.exceptions import ValidationError, WorkflowError, ResourceError, OrchestrationError

# Optional monitoring support
try:
    from mm_orch.monitoring.prometheus_exporter import PrometheusExporter
    from mm_orch.monitoring.otel_tracer import OTelTracer
    from mm_orch.monitoring.performance_monitor import PerformanceMonitor
    from mm_orch.monitoring.anomaly_detector import AnomalyDetector
    from mm_orch.monitoring.config import AnomalyConfig
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    PrometheusExporter = None
    OTelTracer = None
    PerformanceMonitor = None
    AnomalyDetector = None
    AnomalyConfig = None


logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """工作流执行上下文"""

    request: UserRequest
    selection: Optional[WorkflowSelection] = None
    workflow: Optional[BaseWorkflow] = None
    result: Optional[WorkflowResult] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None

    @property
    def execution_time(self) -> float:
        """获取执行时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class WorkflowOrchestrator:
    """
    工作流编排器 - 协调工作流执行、意识模块交互和资源管理

    职责:
    - 注册和管理工作流
    - 执行工作流并处理结果
    - 与意识核心系统集成
    - 结果验证和错误处理

    属性4: 结果结构完整性
    对于任何成功执行的工作流，返回的WorkflowResult对象应该包含
    非空的result字段、metadata字典和status='success'。

    需求:
    - 1.2: 按照预定义的步骤序列执行工作流
    - 1.3: 记录错误信息并返回可理解的错误响应
    - 1.4: 返回结构化的执行结果
    - 6.2: 协调意识模块的交互和信息流动
    """

    def __init__(
        self,
        router: Optional[Router] = None,
        consciousness: Optional[ConsciousnessCore] = None,
        model_manager: Optional[Any] = None,
        auto_register_workflows: bool = True,
        enable_monitoring: bool = False,
        prometheus_port: int = 9090,
        otel_endpoint: Optional[str] = None,
    ):
        """
        初始化工作流编排器

        Args:
            router: Router路由器实例
            consciousness: 意识核心实例
            model_manager: 模型管理器实例
            auto_register_workflows: 是否自动注册默认工作流
            enable_monitoring: Whether to enable monitoring features (Requirement 13.1, 13.4)
            prometheus_port: Port for Prometheus metrics endpoint
            otel_endpoint: OpenTelemetry endpoint URL
        """
        self.router = router or get_router()
        self.consciousness = consciousness or get_consciousness()
        self.model_manager = model_manager

        # Initialize monitoring components (Requirement 13.1, 13.4)
        self.enable_monitoring = enable_monitoring and MONITORING_AVAILABLE
        self.prometheus_exporter = None
        self.otel_tracer = None
        self.performance_monitor = None
        self.anomaly_detector = None
        
        if self.enable_monitoring:
            self._initialize_monitoring(prometheus_port, otel_endpoint)
        elif enable_monitoring and not MONITORING_AVAILABLE:
            logger.warning(
                "Monitoring requested but monitoring modules not available. "
                "System will function without monitoring."
            )

        # 工作流注册表
        self._workflows: Dict[WorkflowType, BaseWorkflow] = {}

        # 执行统计
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_execution_time = 0.0

        # 自动注册默认工作流
        if auto_register_workflows:
            self._register_default_workflows()

        logger.info(
            "WorkflowOrchestrator initialized",
            num_workflows=len(self._workflows),
            has_consciousness=self.consciousness is not None,
            monitoring_enabled=self.enable_monitoring,
        )
    
    def _initialize_monitoring(self, prometheus_port: int, otel_endpoint: Optional[str]):
        """
        Initialize monitoring components.
        
        Args:
            prometheus_port: Port for Prometheus metrics endpoint
            otel_endpoint: OpenTelemetry endpoint URL
            
        Requirement 13.1, 13.4: Initialize monitoring components
        """
        try:
            # Initialize Prometheus exporter
            self.prometheus_exporter = PrometheusExporter(
                port=prometheus_port,
                enabled=True
            )
            self.prometheus_exporter.start_server()
            logger.info(f"Prometheus exporter initialized on port {prometheus_port}")
            
            # Initialize OpenTelemetry tracer
            self.otel_tracer = OTelTracer(
                service_name="muai-orchestration",
                endpoint=otel_endpoint,
                enabled=True
            )
            logger.info("OpenTelemetry tracer initialized")
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor(
                max_history_seconds=3600,
                resource_sample_interval=10
            )
            logger.info("Performance monitor initialized")
            
            # Initialize anomaly detector
            anomaly_config = AnomalyConfig(
                enabled=True,
                latency_threshold_ms=1000.0,
                error_rate_threshold=0.05,
                memory_threshold_percent=90.0,
                throughput_threshold_rps=1.0,
                alert_destinations=["log"],
                alert_rate_limit_seconds=300
            )
            self.anomaly_detector = AnomalyDetector(
                config=anomaly_config,
                performance_monitor=self.performance_monitor
            )
            logger.info("Anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            # Disable monitoring on initialization failure (graceful degradation)
            self.enable_monitoring = False
            self.prometheus_exporter = None
            self.otel_tracer = None
            self.performance_monitor = None
            self.anomaly_detector = None

    def _register_default_workflows(self) -> None:
        """注册默认工作流"""
        # SearchQA工作流
        self.register_workflow(
            SearchQAWorkflow(
                model_manager=self.model_manager,
                tracer=self.otel_tracer if self.enable_monitoring else None
            )
        )

        # LessonPack工作流
        self.register_workflow(LessonPackWorkflow(model_manager=self.model_manager))

        # ChatGenerate工作流
        self.register_workflow(
            ChatGenerateWorkflow(
                model_manager=self.model_manager,
                tracer=self.otel_tracer if self.enable_monitoring else None
            )
        )

        # RAGQA工作流
        self.register_workflow(
            RAGQAWorkflow(
                model_manager=self.model_manager,
                tracer=self.otel_tracer if self.enable_monitoring else None
            )
        )

        # SelfAskSearchQA工作流
        self.register_workflow(SelfAskSearchQAWorkflow(model_manager=self.model_manager))

        logger.debug(
            "Registered default workflows", workflows=[wf.value for wf in self._workflows.keys()]
        )

    def register_workflow(self, workflow: BaseWorkflow) -> None:
        """
        注册新的工作流

        Args:
            workflow: 工作流实例

        Raises:
            ValidationError: 如果工作流无效
        """
        if not isinstance(workflow, BaseWorkflow):
            raise ValidationError("Workflow must be an instance of BaseWorkflow")

        if not hasattr(workflow, "workflow_type") or workflow.workflow_type is None:
            raise ValidationError("Workflow must have a valid workflow_type")

        self._workflows[workflow.workflow_type] = workflow

        logger.info(
            "Registered workflow",
            workflow_type=workflow.workflow_type.value,
            workflow_name=workflow.name,
        )

    def unregister_workflow(self, workflow_type: WorkflowType) -> bool:
        """
        注销工作流

        Args:
            workflow_type: 工作流类型

        Returns:
            是否成功注销
        """
        if workflow_type in self._workflows:
            del self._workflows[workflow_type]
            logger.info("Unregistered workflow", workflow_type=workflow_type.value)
            return True
        return False

    def get_workflow(self, workflow_type: WorkflowType) -> Optional[BaseWorkflow]:
        """
        获取工作流实例

        Args:
            workflow_type: 工作流类型

        Returns:
            工作流实例或None
        """
        return self._workflows.get(workflow_type)

    def get_registered_workflows(self) -> List[WorkflowType]:
        """获取所有已注册的工作流类型"""
        return list(self._workflows.keys())

    def execute_workflow(
        self, workflow_type: WorkflowType, parameters: Dict[str, Any]
    ) -> WorkflowResult:
        """
        执行指定的工作流

        Args:
            workflow_type: 工作流类型
            parameters: 工作流参数

        Returns:
            WorkflowResult: 包含result、metadata、status

        Raises:
            OrchestrationError: 如果工作流未注册或执行失败
        """
        ctx = ExecutionContext(
            request=UserRequest(query=parameters.get("query", "")), start_time=time.time()
        )

        # Create tracing span for workflow execution (Requirement 13.4)
        span = None
        if self.enable_monitoring and self.otel_tracer:
            try:
                span = self.otel_tracer.trace_operation(
                    operation_name=f"execute_workflow.{workflow_type.value}",
                    workflow_type=workflow_type.value,
                    parameters=list(parameters.keys())
                ).__enter__()
            except Exception as e:
                logger.warning(f"Failed to create tracing span: {e}")
                span = None

        try:
            # 获取工作流
            workflow = self._workflows.get(workflow_type)
            if not workflow:
                raise OrchestrationError(f"Workflow '{workflow_type.value}' is not registered")

            ctx.workflow = workflow

            # 通知意识核心任务开始
            self._notify_task_start(workflow_type, parameters)

            # 获取策略建议
            strategy = self._get_strategy_suggestion(workflow_type, parameters)

            # 应用策略建议到参数
            enhanced_params = self._apply_strategy(parameters, strategy)

            logger.info(
                "Executing workflow",
                workflow_type=workflow_type.value,
                parameters=list(enhanced_params.keys()),
            )

            # 执行工作流
            result = workflow.run(enhanced_params)
            ctx.result = result
            ctx.end_time = time.time()

            # 验证结果结构 (属性4)
            result = self._validate_and_enhance_result(result, workflow_type, ctx)

            # Record performance metrics (Requirement 13.4)
            if self.enable_monitoring and self.performance_monitor:
                try:
                    self.performance_monitor.record_latency(
                        operation=f"workflow.{workflow_type.value}",
                        latency_ms=ctx.execution_time * 1000,
                        metadata={
                            "status": result.status,
                            "workflow_type": workflow_type.value
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to record performance metrics: {e}")

            # 通知意识核心任务完成
            self._notify_task_complete(workflow_type, result)

            # 更新统计
            self._update_statistics(result)

            return result

        except ValidationError as e:
            ctx.end_time = time.time()
            ctx.error = str(e)
            return self._create_error_result(
                error=f"Validation error: {str(e)}", workflow_type=workflow_type, ctx=ctx
            )

        except WorkflowError as e:
            ctx.end_time = time.time()
            ctx.error = str(e)
            self._notify_task_error(workflow_type, e)
            return self._create_error_result(
                error=f"Workflow error: {str(e)}", workflow_type=workflow_type, ctx=ctx
            )

        except ResourceError as e:
            ctx.end_time = time.time()
            ctx.error = str(e)
            self._notify_task_error(workflow_type, e)
            return self._create_error_result(
                error=f"Resource error: {str(e)}", workflow_type=workflow_type, ctx=ctx
            )

        except Exception as e:
            ctx.end_time = time.time()
            ctx.error = str(e)
            logger.error(
                "Unexpected error in workflow execution",
                workflow_type=workflow_type.value,
                error_type=type(e).__name__,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
            self._notify_task_error(workflow_type, e)
            return self._create_error_result(
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                workflow_type=workflow_type,
                ctx=ctx,
            )
        finally:
            # Close span if it was created
            if span:
                try:
                    span.__exit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Failed to close tracing span: {e}")

    def process_request(self, request: UserRequest) -> WorkflowResult:
        """
        处理用户请求 - 自动路由并执行工作流

        Args:
            request: 用户请求

        Returns:
            WorkflowResult: 执行结果
        """
        ctx = ExecutionContext(request=request, start_time=time.time())

        try:
            # 使用Router进行路由
            selection = self.router.route(request)
            ctx.selection = selection

            logger.info(
                "Request routed",
                workflow_type=selection.workflow_type.value,
                confidence=selection.confidence,
            )

            # 执行选定的工作流
            result = self.execute_workflow(
                workflow_type=selection.workflow_type, parameters=selection.parameters
            )

            # 添加路由信息到metadata
            result.metadata["routing"] = {
                "workflow_type": selection.workflow_type.value,
                "confidence": selection.confidence,
                "has_alternatives": selection.alternatives is not None,
            }

            return result

        except Exception as e:
            ctx.end_time = time.time()
            ctx.error = str(e)
            logger.error(
                "Request processing failed",
                error=str(e),
                query=request.query[:50] if request.query else "",
            )
            return self._create_error_result(
                error=str(e),
                workflow_type=ctx.selection.workflow_type if ctx.selection else None,
                ctx=ctx,
            )

    def _notify_task_start(self, workflow_type: WorkflowType, parameters: Dict[str, Any]) -> None:
        """通知意识核心任务开始"""
        if self.consciousness:
            task = Task.create(task_type=workflow_type.value, parameters=parameters)
            event = SystemEvent(
                event_type="task_start",
                data={"task": task, "workflow_type": workflow_type.value},
                source="orchestrator",
            )
            self.consciousness.update_state(event)

    def _notify_task_complete(self, workflow_type: WorkflowType, result: WorkflowResult) -> None:
        """通知意识核心任务完成"""
        if self.consciousness:
            success = result.status == "success"
            score = 1.0 if success else (0.5 if result.status == "partial" else 0.0)

            event = SystemEvent(
                event_type="task_complete",
                data={
                    "task_type": workflow_type.value,
                    "success": success,
                    "score": score,
                    "execution_time": result.execution_time,
                },
                source="orchestrator",
            )
            self.consciousness.update_state(event)

    def _notify_task_error(self, workflow_type: WorkflowType, error: Exception) -> None:
        """通知意识核心任务错误"""
        if self.consciousness:
            event = SystemEvent(
                event_type="task_error",
                data={
                    "task_type": workflow_type.value,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                source="orchestrator",
            )
            self.consciousness.update_state(event)

    def _get_strategy_suggestion(
        self, workflow_type: WorkflowType, parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        从意识核心获取策略建议

        Args:
            workflow_type: 工作流类型
            parameters: 工作流参数

        Returns:
            策略建议字典或None
        """
        if not self.consciousness:
            return None

        try:
            task = Task.create(task_type=workflow_type.value, parameters=parameters)
            suggestion = self.consciousness.get_strategy_suggestion(task)

            return {
                "strategy": suggestion.strategy,
                "confidence": suggestion.confidence,
                "parameters": suggestion.parameters,
            }
        except Exception as e:
            logger.warning("Failed to get strategy suggestion", error=str(e))
            return None

    def _apply_strategy(
        self, parameters: Dict[str, Any], strategy: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        应用策略建议到参数

        Args:
            parameters: 原始参数
            strategy: 策略建议

        Returns:
            增强后的参数
        """
        if not strategy:
            return parameters

        enhanced = parameters.copy()

        # 应用策略参数
        strategy_params = strategy.get("parameters", {})

        # 应用情感修饰符
        emotion_modifiers = strategy_params.get("emotion_modifiers", {})
        if emotion_modifiers:
            enhanced["_emotion_modifiers"] = emotion_modifiers

        # 应用响应风格
        response_style = strategy_params.get("response_style", {})
        if response_style:
            enhanced["_response_style"] = response_style

        # 应用复杂度限制
        max_complexity = strategy_params.get("max_complexity")
        if max_complexity is not None:
            enhanced["_max_complexity"] = max_complexity

        return enhanced

    def _validate_and_enhance_result(
        self, result: WorkflowResult, workflow_type: WorkflowType, ctx: ExecutionContext
    ) -> WorkflowResult:
        """
        验证并增强结果结构

        属性4: 结果结构完整性
        对于任何成功执行的工作流，返回的WorkflowResult对象应该包含
        非空的result字段、metadata字典和status='success'。

        Args:
            result: 原始结果
            workflow_type: 工作流类型
            ctx: 执行上下文

        Returns:
            验证并增强后的结果
        """
        # 确保metadata存在
        if result.metadata is None:
            result.metadata = {}

        # 添加标准metadata字段
        result.metadata["workflow_type"] = workflow_type.value
        result.metadata["execution_time"] = ctx.execution_time
        result.metadata["orchestrator_version"] = "1.0.0"

        # 验证成功状态的结果
        if result.status == "success":
            if result.result is None:
                # 如果成功但result为None，降级为partial
                result.status = "partial"
                result.error = result.error or "Result is None despite success status"
                logger.warning(
                    "Downgraded success to partial due to None result",
                    workflow_type=workflow_type.value,
                )

        # 确保execution_time被设置
        if result.execution_time is None:
            result.execution_time = ctx.execution_time

        return result

    def _create_error_result(
        self, error: str, workflow_type: Optional[WorkflowType], ctx: ExecutionContext
    ) -> WorkflowResult:
        """
        创建错误结果

        Args:
            error: 错误信息
            workflow_type: 工作流类型
            ctx: 执行上下文

        Returns:
            错误WorkflowResult
        """
        metadata = {
            "workflow_type": workflow_type.value if workflow_type else "unknown",
            "execution_time": ctx.execution_time,
            "orchestrator_version": "1.0.0",
            "error_context": {
                "query": ctx.request.query[:100] if ctx.request and ctx.request.query else None,
                "has_selection": ctx.selection is not None,
            },
        }

        # 记录错误
        logger.error(
            "Workflow execution failed",
            error=error,
            workflow_type=workflow_type.value if workflow_type else "unknown",
            execution_time=ctx.execution_time,
        )

        self._failure_count += 1

        return WorkflowResult(
            result=None,
            metadata=metadata,
            status="failed",
            error=error,
            execution_time=ctx.execution_time,
        )

    def _update_statistics(self, result: WorkflowResult) -> None:
        """更新执行统计"""
        self._execution_count += 1

        if result.status == "success":
            self._success_count += 1
        elif result.status == "failed":
            self._failure_count += 1

        if result.execution_time:
            self._total_execution_time += result.execution_time

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取执行统计

        Returns:
            统计信息字典
        """
        avg_time = (
            self._total_execution_time / self._execution_count if self._execution_count > 0 else 0.0
        )

        success_rate = (
            self._success_count / self._execution_count if self._execution_count > 0 else 0.0
        )

        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "partial_count": self._execution_count - self._success_count - self._failure_count,
            "success_rate": success_rate,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
            "registered_workflows": len(self._workflows),
        }

    def get_workflow_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取各工作流的执行指标

        Returns:
            工作流指标字典
        """
        metrics = {}
        for wf_type, workflow in self._workflows.items():
            metrics[wf_type.value] = workflow.get_metrics()
        return metrics

    def reset_statistics(self) -> None:
        """重置执行统计"""
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_execution_time = 0.0
        logger.info("Orchestrator statistics reset")


# 单例实例
_orchestrator_instance: Optional[WorkflowOrchestrator] = None


def get_orchestrator(
    router: Optional[Router] = None,
    consciousness: Optional[ConsciousnessCore] = None,
    model_manager: Optional[Any] = None,
) -> WorkflowOrchestrator:
    """
    获取WorkflowOrchestrator单例实例

    Args:
        router: Router实例
        consciousness: 意识核心实例
        model_manager: 模型管理器实例

    Returns:
        WorkflowOrchestrator实例
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = WorkflowOrchestrator(
            router=router, 
            consciousness=consciousness, 
            model_manager=model_manager,
            enable_monitoring=False  # Default to disabled for backward compatibility
        )
    return _orchestrator_instance


def reset_orchestrator() -> None:
    """重置WorkflowOrchestrator单例"""
    global _orchestrator_instance
    _orchestrator_instance = None


def create_orchestrator(
    router: Optional[Router] = None,
    consciousness: Optional[ConsciousnessCore] = None,
    model_manager: Optional[Any] = None,
    auto_register_workflows: bool = True,
    enable_monitoring: bool = False,
    prometheus_port: int = 9090,
    otel_endpoint: Optional[str] = None,
) -> WorkflowOrchestrator:
    """
    创建新的WorkflowOrchestrator实例

    Args:
        router: Router实例
        consciousness: 意识核心实例
        model_manager: 模型管理器实例
        auto_register_workflows: 是否自动注册默认工作流
        enable_monitoring: Whether to enable monitoring features
        prometheus_port: Port for Prometheus metrics endpoint
        otel_endpoint: OpenTelemetry endpoint URL

    Returns:
        新的WorkflowOrchestrator实例
    """
    return WorkflowOrchestrator(
        router=router,
        consciousness=consciousness,
        model_manager=model_manager,
        auto_register_workflows=auto_register_workflows,
        enable_monitoring=enable_monitoring,
        prometheus_port=prometheus_port,
        otel_endpoint=otel_endpoint,
    )
