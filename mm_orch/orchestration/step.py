"""
Step Protocol - Unified interface for all workflow steps.

This module defines the Step protocol that all workflow steps must implement,
providing a consistent interface for composable operations in graph-based workflows.
"""

from typing import Protocol, List, Any, runtime_checkable
from mm_orch.orchestration.state import State


@runtime_checkable
class Step(Protocol):
    """
    Protocol defining the interface for workflow steps.
    
    All steps must implement this protocol to be compatible with the Graph Executor.
    Steps can be implemented as classes or functions with the required attributes.
    
    Attributes:
        name: Unique identifier for the step
        input_keys: List of State keys required as input
        output_keys: List of State keys produced as output
    
    Methods:
        run: Execute the step logic and return updated State
    
    Example:
        class MyStep:
            name = "my_step"
            input_keys = ["question"]
            output_keys = ["result"]
            
            def run(self, state: State, runtime: Any) -> State:
                result = process(state["question"])
                return {**state, "result": result}
    """
    
    name: str
    input_keys: List[str]
    output_keys: List[str]
    
    def run(self, state: State, runtime: Any) -> State:
        """
        Execute step logic and return updated state.
        
        Args:
            state: Current workflow state containing input data
            runtime: Runtime context providing access to models, tools, etc.
        
        Returns:
            Updated State object with output fields populated
        
        Raises:
            KeyError: If required input_keys are missing from state
            Exception: Any step-specific errors during execution
        """
        ...
