"""
BaseStep - Abstract base class for workflow steps.

This module provides a base implementation of the Step protocol with validation,
helper methods, and support for both function-based and class-based steps.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from mm_orch.orchestration.state import State
from mm_orch.orchestration.cost_tracker import CostTracker


class BaseStep(ABC):
    """
    Abstract base class for workflow steps with validation and helper methods.

    This class provides common functionality for steps including:
    - Input validation to ensure required keys exist in State
    - Helper methods for State updates
    - Consistent error handling

    Subclasses must implement:
    - name: Class attribute or property
    - input_keys: Class attribute or property
    - output_keys: Class attribute or property
    - execute: Method containing step logic

    Example:
        class MyStep(BaseStep):
            name = "my_step"
            input_keys = ["question"]
            output_keys = ["result"]

            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                result = self.process(state["question"])
                return {"result": result}
    """

    # Subclasses must define these
    name: str
    input_keys: List[str]
    output_keys: List[str]

    # Cost tracking
    _cost_tracker: Optional[CostTracker] = None
    _current_cost_id: Optional[str] = None

    def set_cost_tracker(self, cost_tracker: CostTracker) -> None:
        """
        Set the cost tracker for this step.

        Args:
            cost_tracker: CostTracker instance to use
        """
        self._cost_tracker = cost_tracker

    def run(self, state: State, runtime: Any) -> State:
        """
        Execute step with validation and error handling.

        This method:
        1. Validates that all input_keys exist in state
        2. Starts cost tracking if tracker is available
        3. Calls the execute method implemented by subclass
        4. Ends cost tracking and stores metrics in state
        5. Merges the returned updates into state
        6. Returns the updated state

        Args:
            state: Current workflow state
            runtime: Runtime context

        Returns:
            Updated State with output fields populated

        Raises:
            KeyError: If required input_keys are missing from state
        """
        # Validate input keys
        self.validate_inputs(state)

        # Start cost tracking
        model_loads_before = 0
        if self._cost_tracker:
            self._current_cost_id = self._cost_tracker.start_step(self.name)
            # Track model loads from runtime if available
            if hasattr(runtime, "model_manager") and hasattr(
                runtime.model_manager, "get_load_count"
            ):
                try:
                    count = runtime.model_manager.get_load_count()
                    if isinstance(count, int):
                        model_loads_before = count
                except (TypeError, AttributeError):
                    pass

        try:
            # Execute step logic
            updates = self.execute(state, runtime)

            # End cost tracking
            if self._cost_tracker and self._current_cost_id:
                # Calculate model loads during this step
                model_loads = 0
                if hasattr(runtime, "model_manager") and hasattr(
                    runtime.model_manager, "get_load_count"
                ):
                    try:
                        count = runtime.model_manager.get_load_count()
                        if isinstance(count, int) and isinstance(model_loads_before, int):
                            model_loads = count - model_loads_before
                    except (TypeError, AttributeError):
                        pass

                cost = self._cost_tracker.end_step(self._current_cost_id, model_loads=model_loads)

                # Store cost in state metadata
                if "meta" not in updates:
                    updates["meta"] = state.get("meta", {}).copy()
                if "step_costs" not in updates["meta"]:
                    updates["meta"]["step_costs"] = []
                updates["meta"]["step_costs"].append(cost.to_dict())

            # Merge updates into state
            return self.update_state(state, updates)

        except Exception:
            # End cost tracking on error
            if self._cost_tracker and self._current_cost_id:
                model_loads = 0
                if hasattr(runtime, "model_manager") and hasattr(
                    runtime.model_manager, "get_load_count"
                ):
                    try:
                        count = runtime.model_manager.get_load_count()
                        if isinstance(count, int) and isinstance(model_loads_before, int):
                            model_loads = count - model_loads_before
                    except (TypeError, AttributeError):
                        pass
                self._cost_tracker.end_step(self._current_cost_id, model_loads=model_loads)
            raise

    def validate_inputs(self, state: State) -> None:
        """
        Validate that all required input_keys exist in state.

        Args:
            state: State to validate

        Raises:
            KeyError: If any required input key is missing
        """
        missing_keys = [key for key in self.input_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Step '{self.name}' requires input keys {missing_keys} "
                f"but they are missing from state. Available keys: {list(state.keys())}"
            )

    @abstractmethod
    def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
        """
        Execute step logic and return updates to apply to state.

        Subclasses must implement this method with their specific logic.

        Args:
            state: Current workflow state (read-only, do not modify)
            runtime: Runtime context providing access to models, tools, etc.

        Returns:
            Dictionary of updates to merge into state. Keys should match output_keys.

        Example:
            def execute(self, state: State, runtime: Any) -> Dict[str, Any]:
                question = state["question"]
                result = self.process(question)
                return {"result": result}
        """

    def update_state(self, state: State, updates: Dict[str, Any]) -> State:
        """
        Merge updates into state, preserving existing fields.

        This helper creates a new State dict with all existing fields
        plus the updates, following the immutability pattern.

        Args:
            state: Original state
            updates: Dictionary of fields to update

        Returns:
            New State with updates applied
        """
        return {**state, **updates}

    def get_input(self, state: State, key: str, default: Any = None) -> Any:
        """
        Safely get an input value from state with optional default.

        Args:
            state: State to read from
            key: Key to retrieve
            default: Default value if key is missing

        Returns:
            Value from state or default
        """
        return state.get(key, default)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"input_keys={self.input_keys}, "
            f"output_keys={self.output_keys})"
        )


class FunctionStep:
    """
    Wrapper to convert a function into a Step.

    This allows function-based steps to be used alongside class-based steps
    in the graph executor.

    Example:
        def my_function(state: State, runtime: Any) -> State:
            return {**state, "result": process(state["question"])}

        step = FunctionStep(
            name="my_step",
            input_keys=["question"],
            output_keys=["result"],
            func=my_function
        )
    """

    def __init__(self, name: str, input_keys: List[str], output_keys: List[str], func: Any):
        """
        Initialize function-based step.

        Args:
            name: Step name
            input_keys: Required input keys
            output_keys: Produced output keys
            func: Function to execute (signature: (State, runtime) -> State)
        """
        self.name = name
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.func = func

    def run(self, state: State, runtime: Any) -> State:
        """Execute the wrapped function."""
        # Validate inputs
        missing_keys = [key for key in self.input_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Step '{self.name}' requires input keys {missing_keys} "
                f"but they are missing from state"
            )

        # Execute function
        return self.func(state, runtime)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"FunctionStep("
            f"name='{self.name}', "
            f"input_keys={self.input_keys}, "
            f"output_keys={self.output_keys})"
        )
