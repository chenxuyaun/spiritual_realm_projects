"""
Graph Executor - Executes workflow graphs with tracing support.

This module provides the GraphExecutor that orchestrates step execution
in workflow graphs, supporting linear chains, conditional branching,
and comprehensive tracing with cost tracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
import time
from mm_orch.orchestration.state import State
from mm_orch.orchestration.step import Step
from mm_orch.orchestration.cost_tracker import CostTracker


@dataclass
class GraphNode:
    """
    Node in a workflow graph.

    Attributes:
        step_name: Name of the step to execute at this node
        next_nodes: List of possible next node names
        condition: Optional function to determine which next node to take
                  Signature: (State) -> bool
                  If None, takes the first next_node
                  If multiple next_nodes, condition determines which one

    Example:
        # Linear node (single next)
        node1 = GraphNode(step_name="search", next_nodes=["fetch"])

        # Conditional branch
        node2 = GraphNode(
            step_name="check_cache",
            next_nodes=["use_cache", "fetch_new"],
            condition=lambda state: "cached_result" in state
        )
    """

    step_name: str
    next_nodes: List[str] = field(default_factory=list)
    condition: Optional[Callable[[State], bool]] = None


class GraphExecutor:
    """
    Executes workflow graphs with support for linear chains and branching.

    The executor:
    1. Validates the graph structure (no cycles, all steps exist)
    2. Executes steps in order determined by the graph
    3. Traces each step execution (start/end times, success/failure)
    4. Handles conditional branching based on state
    5. Propagates exceptions with full context

    Example:
        # Define graph
        graph = {
            "start": GraphNode(step_name="search", next_nodes=["fetch"]),
            "fetch": GraphNode(step_name="fetch_urls", next_nodes=["summarize"]),
            "summarize": GraphNode(step_name="summarize", next_nodes=["end"]),
            "end": GraphNode(step_name="end", next_nodes=[])
        }

        # Execute
        executor = GraphExecutor(step_registry, tracer)
        final_state = executor.execute(graph, initial_state, runtime)
    """

    def __init__(self, step_registry: Dict[str, Step], tracer: Optional[Any] = None):
        """
        Initialize graph executor.

        Args:
            step_registry: Dictionary mapping step names to Step instances
            tracer: Optional tracer for recording execution details
        """
        self.step_registry = step_registry
        self.tracer = tracer
        self.cost_tracker = CostTracker()

        # Set cost tracker on all steps
        for step in step_registry.values():
            if hasattr(step, "set_cost_tracker"):
                step.set_cost_tracker(self.cost_tracker)

    def execute(
        self,
        graph: Dict[str, GraphNode],
        initial_state: State,
        runtime: Any,
        start_node: str = "start",
    ) -> State:
        """
        Execute workflow graph from start node to completion.

        Args:
            graph: Dictionary mapping node names to GraphNode instances
            initial_state: Initial state to begin execution
            runtime: Runtime context for steps
            start_node: Name of the starting node (default: "start")

        Returns:
            Final state after all steps complete

        Raises:
            ValueError: If graph is invalid (cycles, missing steps, etc.)
            KeyError: If a node references a non-existent step
            Exception: Any exception raised by step execution
        """
        # Validate graph
        self._validate_graph(graph)

        # Execute graph
        current_node_name = start_node
        state = initial_state
        visited_nodes: List[str] = []

        while current_node_name != "end":
            # Check for infinite loops
            if current_node_name in visited_nodes:
                # Allow revisiting nodes in some cases, but detect true cycles
                if visited_nodes.count(current_node_name) > len(graph):
                    raise ValueError(
                        f"Detected cycle in graph execution at node '{current_node_name}'. "
                        f"Visited path: {visited_nodes}"
                    )

            visited_nodes.append(current_node_name)

            # Get current node
            if current_node_name not in graph:
                raise ValueError(
                    f"Node '{current_node_name}' not found in graph. "
                    f"Available nodes: {list(graph.keys())}"
                )

            node = graph[current_node_name]

            # Get step
            if node.step_name not in self.step_registry:
                raise KeyError(
                    f"Step '{node.step_name}' not found in step registry. "
                    f"Available steps: {list(self.step_registry.keys())}"
                )

            step = self.step_registry[node.step_name]

            # Trace step start
            trace_id = None
            if self.tracer:
                trace_id = self.tracer.start_step(node.step_name, state)

            # Execute step (cost tracking happens inside step.run)
            try:
                state = step.run(state, runtime)

                # Trace step success
                if self.tracer and trace_id:
                    # Get model loads from state if available
                    step_costs = state.get("meta", {}).get("step_costs", [])
                    latest_cost = step_costs[-1] if step_costs else {}
                    model_loads = latest_cost.get("model_loads", 0)

                    # Pass only model_loads to tracer (it calculates latency/vram internally)
                    self.tracer.end_step(trace_id, state, success=True, model_loads=model_loads)

            except Exception as e:
                # Trace step failure
                if self.tracer and trace_id:
                    self.tracer.end_step(trace_id, state, success=False, error=e)

                # Re-raise with context
                raise RuntimeError(
                    f"Step '{node.step_name}' failed at node '{current_node_name}'"
                ) from e

            # Select next node
            current_node_name = self._select_next_node(node, state, graph)

        return state

    def _select_next_node(self, node: GraphNode, state: State, graph: Dict[str, GraphNode]) -> str:
        """
        Select the next node to execute based on conditions.

        Args:
            node: Current node
            state: Current state
            graph: Full graph for validation

        Returns:
            Name of next node to execute
        """
        # If no next nodes, we're done
        if not node.next_nodes:
            return "end"

        # If single next node, take it
        if len(node.next_nodes) == 1:
            return node.next_nodes[0]

        # Multiple next nodes - use condition
        if node.condition is None:
            # No condition specified, take first
            return node.next_nodes[0]

        # Evaluate condition for each next node
        for next_node_name in node.next_nodes:
            if next_node_name not in graph:
                continue

            next_node = graph[next_node_name]

            # If this node has a condition, evaluate it
            if next_node.condition and next_node.condition(state):
                return next_node_name

        # If no condition matched, take first
        return node.next_nodes[0]

    def _validate_graph(self, graph: Dict[str, GraphNode]) -> None:
        """
        Validate graph structure.

        Checks:
        - Graph has a "start" node
        - All referenced next_nodes exist in graph or are "end"
        - All referenced steps exist in step registry
        - No obvious cycles (basic check)

        Args:
            graph: Graph to validate

        Raises:
            ValueError: If graph structure is invalid
            KeyError: If referenced steps don't exist
        """
        # Check for start node
        if "start" not in graph:
            raise ValueError("Graph must have a 'start' node")

        # Check all next_nodes exist
        for node_name, node in graph.items():
            for next_node_name in node.next_nodes:
                if next_node_name != "end" and next_node_name not in graph:
                    raise ValueError(
                        f"Node '{node_name}' references non-existent next node '{next_node_name}'"
                    )

            # Check step exists
            if node.step_name not in self.step_registry:
                raise KeyError(
                    f"Node '{node_name}' references non-existent step '{node.step_name}'. "
                    f"Available steps: {list(self.step_registry.keys())}"
                )

        # Basic cycle detection using DFS
        self._detect_cycles(graph)

    def _detect_cycles(self, graph: Dict[str, GraphNode]) -> None:
        """
        Detect cycles in graph using depth-first search.

        Args:
            graph: Graph to check

        Raises:
            ValueError: If a cycle is detected
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node_name: str, path: List[str]) -> bool:
            """DFS helper to detect cycles."""
            if node_name == "end":
                return False

            if node_name in rec_stack:
                # Found a cycle
                cycle_start = path.index(node_name)
                cycle = path[cycle_start:] + [node_name]
                raise ValueError(f"Cycle detected in graph: {' -> '.join(cycle)}")

            if node_name in visited:
                return False

            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            # Visit all next nodes
            node = graph.get(node_name)
            if node:
                for next_node_name in node.next_nodes:
                    if dfs(next_node_name, path.copy()):
                        return True

            rec_stack.remove(node_name)
            return False

        # Start DFS from start node
        dfs("start", [])


class SimpleTracer:
    """
    Simple tracer implementation for development and testing.

    This is a minimal tracer that can be used before the full
    observability system is implemented in Phase B4.
    """

    def __init__(self):
        """Initialize tracer with empty trace list."""
        self.traces: List[Dict[str, Any]] = []
        self._active_traces: Dict[str, Dict[str, Any]] = {}

    def start_step(self, step_name: str, state: State) -> str:
        """
        Begin tracing a step execution.

        Args:
            step_name: Name of the step
            state: Current state

        Returns:
            Trace ID for this step execution
        """
        trace_id = f"{step_name}_{time.time()}"
        self._active_traces[trace_id] = {
            "step_name": step_name,
            "start_time": time.time(),
            "state_keys": list(state.keys()),
        }
        return trace_id

    def end_step(
        self,
        trace_id: str,
        state: State,
        success: bool,
        error: Optional[Exception] = None,
        latency_ms: float = 0.0,
        vram_peak_mb: int = 0,
        model_loads: int = 0,
    ) -> None:
        """
        Complete step trace.

        Args:
            trace_id: Trace ID from start_step
            state: Final state
            success: Whether step succeeded
            error: Exception if step failed
            latency_ms: Step execution latency in milliseconds
            vram_peak_mb: Peak VRAM usage in megabytes
            model_loads: Number of model loads performed
        """
        if trace_id not in self._active_traces:
            return

        trace = self._active_traces[trace_id]
        trace["end_time"] = time.time()

        # Use provided latency if available, otherwise calculate
        if latency_ms > 0:
            trace["latency_ms"] = latency_ms
        else:
            trace["latency_ms"] = (trace["end_time"] - trace["start_time"]) * 1000

        trace["success"] = success
        trace["vram_peak_mb"] = vram_peak_mb
        trace["model_loads"] = model_loads

        if error:
            trace["error"] = str(error)
            trace["error_type"] = type(error).__name__

        self.traces.append(trace)
        del self._active_traces[trace_id]

    def get_traces(self) -> List[Dict[str, Any]]:
        """Get all completed traces."""
        return self.traces

    def clear(self) -> None:
        """Clear all traces."""
        self.traces.clear()
        self._active_traces.clear()
