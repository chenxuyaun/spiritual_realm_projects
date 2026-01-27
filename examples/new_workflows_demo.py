"""
Demo of Phase B3 new workflow variants.

This script demonstrates the three new workflows:
1. summarize_url: Fetch and summarize a single URL
2. search_qa_fast: Fast search QA with reduced summarization
3. search_qa_strict_citations: Search QA with citation validation
"""

from mm_orch.orchestration import (
    State,
    GraphExecutor,
    SimpleTracer,
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep
)
from mm_orch.orchestration.workflow_steps import (
    FetchSingleUrlStep,
    FetchTopNStep,
    CitationValidationStep,
    SummarizeToAnswerStep,
    AnswerGenerateFromDocsStep
)
from mm_orch.registries.workflow_definitions import (
    create_summarize_url_workflow,
    create_search_qa_fast_workflow,
    create_search_qa_strict_citations_workflow
)


def demo_summarize_url():
    """Demonstrate the summarize_url workflow."""
    print("\n" + "="*80)
    print("DEMO 1: Summarize URL Workflow")
    print("="*80)
    
    # Create steps
    fetch_step = FetchSingleUrlStep()
    summarize_step = SummarizeToAnswerStep()
    
    # Create step registry
    step_registry = {
        "fetch_single_url": fetch_step,
        "summarize_to_answer": summarize_step
    }
    
    # Get workflow definition
    workflow = create_summarize_url_workflow()
    
    print(f"\nWorkflow: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"Steps: {[node.step_name for node in workflow.graph.values()]}")
    
    # Create executor
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    # Initial state
    initial_state: State = {
        "question": "https://www.python.org/about/"
    }
    
    print(f"\nInput URL: {initial_state['question']}")
    print("\nExecuting workflow...")
    
    try:
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, runtime=None)
        
        print("\n✓ Workflow completed successfully!")
        print(f"\nSummary length: {len(final_state.get('final_answer', ''))} characters")
        print(f"Citations: {final_state.get('citations', [])}")
        
        # Show traces
        traces = tracer.get_traces()
        print(f"\nExecution traces:")
        for trace in traces:
            print(f"  - {trace['step_name']}: {trace['latency_ms']:.2f}ms")
    
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")


def demo_search_qa_fast():
    """Demonstrate the search_qa_fast workflow."""
    print("\n" + "="*80)
    print("DEMO 2: Search QA Fast Workflow")
    print("="*80)
    
    # Create steps
    search_step = WebSearchStep()
    fetch_step = FetchTopNStep(n=2)
    answer_step = AnswerGenerateFromDocsStep()
    
    # Create step registry
    step_registry = {
        "web_search": search_step,
        "fetch_top_n": fetch_step,
        "answer_generate_from_docs": answer_step
    }
    
    # Get workflow definition
    workflow = create_search_qa_fast_workflow()
    
    print(f"\nWorkflow: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"Steps: {[node.step_name for node in workflow.graph.values()]}")
    print(f"Note: Skips summarization for faster execution")
    
    # Create executor
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    # Initial state
    initial_state: State = {
        "question": "What is Python programming language?"
    }
    
    print(f"\nQuestion: {initial_state['question']}")
    print("\nExecuting workflow...")
    
    try:
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, runtime=None)
        
        print("\n✓ Workflow completed successfully!")
        print(f"\nAnswer length: {len(final_state.get('final_answer', ''))} characters")
        print(f"Citations: {len(final_state.get('citations', []))} sources")
        print(f"Search results: {len(final_state.get('search_results', []))} results")
        print(f"Docs fetched: {len(final_state.get('docs', {}))} documents")
        
        # Show traces
        traces = tracer.get_traces()
        print(f"\nExecution traces:")
        total_time = sum(t['latency_ms'] for t in traces)
        for trace in traces:
            print(f"  - {trace['step_name']}: {trace['latency_ms']:.2f}ms")
        print(f"\nTotal execution time: {total_time:.2f}ms")
    
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")


def demo_search_qa_strict_citations():
    """Demonstrate the search_qa_strict_citations workflow."""
    print("\n" + "="*80)
    print("DEMO 3: Search QA Strict Citations Workflow")
    print("="*80)
    
    # Create steps
    search_step = WebSearchStep()
    fetch_step = FetchUrlStep()
    summarize_step = SummarizeStep()
    answer_step = AnswerGenerateStep()
    validate_step = CitationValidationStep()
    
    # Create step registry
    step_registry = {
        "web_search": search_step,
        "fetch_url": fetch_step,
        "summarize": summarize_step,
        "answer_generate": answer_step,
        "citation_validation": validate_step
    }
    
    # Get workflow definition
    workflow = create_search_qa_strict_citations_workflow()
    
    print(f"\nWorkflow: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"Steps: {[node.step_name for node in workflow.graph.values()]}")
    print(f"Note: Includes citation validation step")
    
    # Create executor
    tracer = SimpleTracer()
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    # Initial state
    initial_state: State = {
        "question": "What are the key features of Python?"
    }
    
    print(f"\nQuestion: {initial_state['question']}")
    print("\nExecuting workflow...")
    
    try:
        # Execute
        final_state = executor.execute(workflow.graph, initial_state, runtime=None)
        
        print("\n✓ Workflow completed successfully!")
        print(f"\nAnswer length: {len(final_state.get('final_answer', ''))} characters")
        print(f"Citations: {len(final_state.get('citations', []))} sources")
        
        # Show validation results
        validation_passed = final_state.get('validation_passed', False)
        validation_errors = final_state.get('validation_errors', [])
        
        print(f"\nCitation Validation:")
        if validation_passed:
            print("  ✓ Validation PASSED")
        else:
            print("  ✗ Validation FAILED")
            for error in validation_errors:
                print(f"    - {error}")
        
        # Show traces
        traces = tracer.get_traces()
        print(f"\nExecution traces:")
        for trace in traces:
            print(f"  - {trace['step_name']}: {trace['latency_ms']:.2f}ms")
    
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("Phase B3 New Workflow Variants Demo")
    print("="*80)
    print("\nThis demo showcases three new workflow variants:")
    print("1. summarize_url: Quick URL summarization")
    print("2. search_qa_fast: Fast search QA (skips summarization)")
    print("3. search_qa_strict_citations: Search QA with citation validation")
    
    # Run demos
    demo_summarize_url()
    demo_search_qa_fast()
    demo_search_qa_strict_citations()
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


if __name__ == "__main__":
    main()
