"""
Phase B Orchestration Demo

This script demonstrates the new Phase B orchestration system with:
- Step API for composable operations
- State-driven execution
- Graph-based workflow definition
- Tracing and observability

Run with: python examples/phase_b_orchestration_demo.py
"""

from mm_orch.orchestration import (
    State,
    GraphExecutor,
    GraphNode,
    SimpleTracer,
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep
)


def main():
    """Run a simple search QA workflow using the new orchestration system."""
    
    print("=" * 70)
    print("Phase B Orchestration System Demo")
    print("=" * 70)
    print()
    
    # Step 1: Create workflow steps
    print("1. Creating workflow steps...")
    search_step = WebSearchStep(max_results=3)
    fetch_step = FetchUrlStep(max_content_length=1000)
    summarize_step = SummarizeStep(max_summary_length=300)
    answer_step = AnswerGenerateStep(max_context_length=2000)
    
    # Step 2: Build step registry
    print("2. Building step registry...")
    step_registry = {
        "search": search_step,
        "fetch": fetch_step,
        "summarize": summarize_step,
        "answer": answer_step
    }
    
    # Step 3: Define workflow graph
    print("3. Defining workflow graph (linear chain)...")
    graph = {
        "start": GraphNode(step_name="search", next_nodes=["fetch_node"]),
        "fetch_node": GraphNode(step_name="fetch", next_nodes=["summarize_node"]),
        "summarize_node": GraphNode(step_name="summarize", next_nodes=["answer_node"]),
        "answer_node": GraphNode(step_name="answer", next_nodes=["end"]),
        "end": GraphNode(step_name="answer", next_nodes=[])
    }
    
    print("   Graph structure:")
    print("   start (search) → fetch → summarize → answer → end")
    print()
    
    # Step 4: Create tracer for observability
    print("4. Creating tracer for observability...")
    tracer = SimpleTracer()
    
    # Step 5: Create graph executor
    print("5. Creating graph executor...")
    executor = GraphExecutor(step_registry, tracer=tracer)
    
    # Step 6: Prepare initial state
    question = "What is Python programming language?"
    print(f"6. Preparing initial state with question: '{question}'")
    
    initial_state: State = {
        "question": question,
        "meta": {
            "mode": "default",
            "workflow": "search_qa"
        }
    }
    
    # Step 7: Execute workflow
    print()
    print("7. Executing workflow...")
    print("-" * 70)
    
    try:
        final_state = executor.execute(graph, initial_state, runtime=None)
        
        print("-" * 70)
        print()
        print("8. Workflow completed successfully!")
        print()
        
        # Display results
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print()
        
        print(f"Question: {final_state['question']}")
        print()
        
        print(f"Search Results: {len(final_state.get('search_results', []))} found")
        for i, result in enumerate(final_state.get('search_results', [])[:3], 1):
            print(f"  {i}. {result.get('title', 'N/A')}")
            print(f"     {result.get('url', 'N/A')}")
        print()
        
        print(f"Documents Fetched: {len(final_state.get('docs', {}))}")
        print()
        
        print(f"Summaries Generated: {len(final_state.get('summaries', {}))}")
        print()
        
        print("Answer:")
        print("-" * 70)
        answer = final_state.get('final_answer', 'No answer generated')
        # Truncate long answers for display
        if len(answer) > 500:
            answer = answer[:500] + "..."
        print(answer)
        print("-" * 70)
        print()
        
        print("Citations:")
        for i, citation in enumerate(final_state.get('citations', []), 1):
            print(f"  [{i}] {citation}")
        print()
        
        # Display trace information
        print("=" * 70)
        print("EXECUTION TRACE")
        print("=" * 70)
        print()
        
        traces = tracer.get_traces()
        print(f"Total steps executed: {len(traces)}")
        print()
        
        for i, trace in enumerate(traces, 1):
            status = "✓" if trace['success'] else "✗"
            print(f"{i}. {status} {trace['step_name']}")
            print(f"   Latency: {trace['latency_ms']:.2f}ms")
            if trace.get('error'):
                print(f"   Error: {trace['error']}")
        print()
        
        # Calculate total execution time
        total_time = sum(t['latency_ms'] for t in traces)
        print(f"Total execution time: {total_time:.2f}ms")
        print()
        
    except Exception as e:
        print("-" * 70)
        print()
        print(f"ERROR: Workflow failed with error: {e}")
        print()
        
        # Display partial trace
        traces = tracer.get_traces()
        if traces:
            print("Partial execution trace:")
            for trace in traces:
                status = "✓" if trace['success'] else "✗"
                print(f"  {status} {trace['step_name']} ({trace['latency_ms']:.2f}ms)")
        print()
    
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
