"""Tracer component for comprehensive execution tracing."""

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class StepTrace:
    """Trace record for a single step execution."""
    step_name: str
    start_time: float
    end_time: float = 0.0
    latency_ms: float = 0.0
    vram_peak_mb: int = 0
    model_loads: int = 0
    success: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class WorkflowTrace:
    """Complete trace for workflow execution."""
    request_id: str
    conversation_id: Optional[str]
    question: str
    chosen_workflow: str
    router_version: str
    mode: str
    turn_index: Optional[int]
    steps: List[StepTrace]
    urls_used: List[str]
    final_answer: str
    quality_signals: Dict[str, Any]
    cost_stats: Dict[str, float]
    timestamp: float
    success: bool = True


class Tracer:
    """Manages execution tracing with JSONL format."""
    
    def __init__(self, output_path: str):
        """
        Initialize tracer with output file path.
        
        Args:
            output_path: Path to JSONL file for trace output
        """
        self.output_path = Path(output_path)
        self._active_traces: Dict[str, StepTrace] = {}
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_step(self, step_name: str, state: Dict[str, Any]) -> str:
        """
        Begin tracing a step execution.
        
        Args:
            step_name: Name of the step being executed
            state: Current workflow state
            
        Returns:
            Trace ID for this step execution
        """
        trace_id = f"{step_name}_{time.time()}_{id(state)}"
        
        self._active_traces[trace_id] = StepTrace(
            step_name=step_name,
            start_time=time.time(),
            vram_peak_mb=self._get_vram_usage(),
            model_loads=0,
            success=False
        )
        
        return trace_id
    
    def end_step(
        self, 
        trace_id: str, 
        state: Dict[str, Any], 
        success: bool, 
        error: Optional[Exception] = None,
        model_loads: int = 0
    ):
        """
        Complete step trace.
        
        Args:
            trace_id: ID returned from start_step
            state: Final workflow state
            success: Whether step completed successfully
            error: Exception if step failed
            model_loads: Number of models loaded during step
        """
        if trace_id not in self._active_traces:
            return
        
        trace = self._active_traces[trace_id]
        trace.end_time = time.time()
        trace.latency_ms = (trace.end_time - trace.start_time) * 1000
        trace.success = success
        trace.model_loads = model_loads
        trace.vram_peak_mb = max(trace.vram_peak_mb, self._get_vram_usage())
        
        if error:
            trace.error = str(error)
            trace.error_type = type(error).__name__
            trace.stack_trace = ''.join(traceback.format_exception(
                type(error), error, error.__traceback__
            ))
    
    def get_step_trace(self, trace_id: str) -> Optional[StepTrace]:
        """
        Retrieve a completed step trace.
        
        Args:
            trace_id: ID of the step trace
            
        Returns:
            StepTrace if found, None otherwise
        """
        return self._active_traces.get(trace_id)
    
    def clear_step_trace(self, trace_id: str):
        """Remove a step trace from active traces."""
        self._active_traces.pop(trace_id, None)
    
    def write_workflow_trace(self, trace: WorkflowTrace):
        """
        Write complete workflow trace to JSONL file.
        
        Traces are appended to the file without overwriting existing records.
        
        Args:
            trace: Complete workflow trace to write
        """
        try:
            with open(self.output_path, 'a', encoding='utf-8') as f:
                trace_dict = asdict(trace)
                f.write(json.dumps(trace_dict, ensure_ascii=False) + '\n')
        except Exception as e:
            # Log to stderr but don't crash
            import sys
            print(f"Failed to write trace: {e}", file=sys.stderr)
    
    def _get_vram_usage(self) -> int:
        """
        Get current VRAM usage in MB.
        
        Returns:
            VRAM usage in megabytes, or 0 if not available
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() // (1024 * 1024)
        except ImportError:
            pass
        return 0
