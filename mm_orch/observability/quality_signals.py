"""Quality signals calculation for workflow executions."""

from dataclasses import dataclass
from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from mm_orch.observability.tracer import WorkflowTrace


@dataclass
class QualitySignals:
    """Automatic quality metrics for execution."""
    citation_count: int
    answer_length: int
    has_search: bool
    has_citations: bool
    has_structure: bool  # For lessons: has sections
    failure_occurred: bool
    
    @classmethod
    def from_trace(cls, trace: "WorkflowTrace", state: Dict[str, Any]) -> "QualitySignals":
        """
        Calculate quality signals from trace and state.
        
        Args:
            trace: Complete workflow trace
            state: Final workflow state
            
        Returns:
            QualitySignals with calculated metrics
        """
        # Import here to avoid circular dependency
        from mm_orch.observability.tracer import WorkflowTrace
        
        citations = state.get("citations", [])
        final_answer = state.get("final_answer", "")
        
        # Check if any step is web_search
        has_search = any(
            step.step_name == "web_search" or "search" in step.step_name.lower()
            for step in trace.steps
        )
        
        # Check if lesson has structured output
        has_structure = "lesson_explain_structured" in state
        
        # Check if any step failed
        failure_occurred = any(not step.success for step in trace.steps)
        
        return cls(
            citation_count=len(citations),
            answer_length=len(final_answer),
            has_search=has_search,
            has_citations=len(citations) > 0,
            has_structure=has_structure,
            failure_occurred=failure_occurred
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "citation_count": self.citation_count,
            "answer_length": self.answer_length,
            "has_search": self.has_search,
            "has_citations": self.has_citations,
            "has_structure": self.has_structure,
            "failure_occurred": self.failure_occurred
        }
