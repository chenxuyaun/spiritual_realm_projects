"""
Base workflow abstract class for MuAI Multi-Model Orchestration System.

This module defines the BaseWorkflow abstract class that all workflow
implementations must inherit from. It provides a common interface for
workflow execution, parameter validation, and model requirements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time
import traceback

from mm_orch.schemas import WorkflowResult, WorkflowType
from mm_orch.logger import get_logger
from mm_orch.exceptions import ValidationError, WorkflowError


logger = get_logger(__name__)


class BaseWorkflow(ABC):
    """
    Abstract base class for all workflow implementations.

    All workflows must inherit from this class and implement the required
    abstract methods: execute() and validate_parameters().

    Attributes:
        workflow_type: The type of workflow this class implements
        name: Human-readable name of the workflow
        description: Description of what the workflow does
    """

    workflow_type: WorkflowType
    name: str = "BaseWorkflow"
    description: str = "Base workflow class"

    def __init__(self):
        """Initialize the base workflow."""
        self._execution_count = 0
        self._total_execution_time = 0.0

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the workflow with the given parameters.

        This is the core method that implements the workflow logic.
        Subclasses must implement this method.

        Args:
            parameters: Dictionary of parameters for the workflow execution.
                       The required parameters depend on the specific workflow.

        Returns:
            WorkflowResult: The result of the workflow execution, containing
                           the result data, metadata, and status.

        Raises:
            ValidationError: If parameters are invalid
            WorkflowError: If workflow execution fails
        """
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate the parameters for this workflow.

        Subclasses must implement this method to check that all required
        parameters are present and have valid values.

        Args:
            parameters: Dictionary of parameters to validate

        Returns:
            bool: True if parameters are valid, False otherwise

        Raises:
            ValidationError: If parameters are invalid (with details)
        """
        pass

    def get_required_models(self) -> List[str]:
        """
        Return the list of model names required by this workflow.

        Subclasses should override this method to specify which models
        they need. The ModelManager will use this to preload models.

        Returns:
            List[str]: List of model identifiers required by this workflow
        """
        return []

    def get_required_parameters(self) -> List[str]:
        """
        Return the list of required parameter names for this workflow.

        Subclasses should override this method to specify which parameters
        are required for execution.

        Returns:
            List[str]: List of required parameter names
        """
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        """
        Return optional parameters with their default values.

        Subclasses should override this method to specify optional parameters
        and their default values.

        Returns:
            Dict[str, Any]: Dictionary mapping parameter names to default values
        """
        return {}

    def run(self, parameters: Dict[str, Any]) -> WorkflowResult:
        """
        Run the workflow with validation and error handling.

        This method wraps the execute() method with:
        - Parameter validation
        - Execution timing
        - Error handling and logging
        - Metrics collection

        Args:
            parameters: Dictionary of parameters for the workflow

        Returns:
            WorkflowResult: The result of the workflow execution
        """
        start_time = time.time()

        try:
            # Validate parameters
            logger.debug(
                "Validating workflow parameters",
                workflow=self.name,
                parameters=list(parameters.keys()),
            )

            if not self.validate_parameters(parameters):
                raise ValidationError(f"Invalid parameters for workflow {self.name}")

            # Merge with default values for optional parameters
            merged_params = self._merge_with_defaults(parameters)

            # Execute the workflow
            logger.info(
                "Executing workflow",
                workflow=self.name,
                workflow_type=self.workflow_type.value if self.workflow_type else None,
            )

            result = self.execute(merged_params)

            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Update metrics
            self._execution_count += 1
            self._total_execution_time += execution_time

            logger.info(
                "Workflow completed",
                workflow=self.name,
                status=result.status,
                execution_time=execution_time,
            )

            return result

        except ValidationError as e:
            execution_time = time.time() - start_time
            logger.error("Workflow validation failed", workflow=self.name, error=str(e))
            return WorkflowResult(
                result=None,
                metadata={"workflow": self.name},
                status="failed",
                error=f"Validation error: {str(e)}",
                execution_time=execution_time,
            )

        except WorkflowError as e:
            execution_time = time.time() - start_time
            logger.error(
                "Workflow execution failed",
                workflow=self.name,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
            return WorkflowResult(
                result=None,
                metadata={"workflow": self.name},
                status="failed",
                error=f"Workflow error: {str(e)}",
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Unexpected error in workflow",
                workflow=self.name,
                error_type=type(e).__name__,
                error=str(e),
                stack_trace=traceback.format_exc(),
            )
            return WorkflowResult(
                result=None,
                metadata={"workflow": self.name},
                status="failed",
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                execution_time=execution_time,
            )

    def _merge_with_defaults(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge provided parameters with default values for optional parameters.

        Args:
            parameters: User-provided parameters

        Returns:
            Dict[str, Any]: Merged parameters with defaults applied
        """
        defaults = self.get_optional_parameters()
        merged = defaults.copy()
        merged.update(parameters)
        return merged

    def _validate_required_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Check that all required parameters are present.

        Args:
            parameters: Parameters to check

        Returns:
            bool: True if all required parameters are present

        Raises:
            ValidationError: If required parameters are missing
        """
        required = self.get_required_parameters()
        missing = [p for p in required if p not in parameters]

        if missing:
            raise ValidationError(f"Missing required parameters: {', '.join(missing)}")

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics for this workflow.

        Returns:
            Dict[str, Any]: Dictionary containing execution metrics
        """
        avg_time = (
            self._total_execution_time / self._execution_count if self._execution_count > 0 else 0.0
        )

        return {
            "workflow": self.name,
            "workflow_type": self.workflow_type.value if self.workflow_type else None,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, type={self.workflow_type})"
