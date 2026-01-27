"""
Workflow Registry Demo

This example demonstrates how to use the Workflow Registry to:
1. Register workflow definitions
2. Retrieve and execute workflows
3. Query workflows by capability and tags
"""

from mm_orch.orchestration.graph_executor import GraphExecutor, GraphNode, SimpleTracer
from mm_orch.orchestration.workflow_steps import (
    WebSearchStep,
    FetchUrlStep,
    SummarizeStep,
    AnswerGenerateStep
)
from mm_orch.orchestration.state import State
from mm_orch.registries.workflow_registry import (
    WorkflowRegistry,
    WorkflowDefinition,
    get_workflow_registry,
    reset_workflow_registry
)
from mm_orch.registries.workflow_definitions import (
    create_search_qa_workflow,
    register_default_workflows
)


class MockRuntime:
    """Mock runtime for demo purposes."""
    pass


def main():
    """Run workflow registry demo."""
    print("=" * 70)
    print("Workflow Registry Demo")
    print("=" * 70)
    print()
    
    # Reset global registry for clean demo
    reset_workflow_registry()
    
    # Step 1: Create step registry
    print("Step 1: Creating step registry...")
    step_registry = {
        "web_search": WebSearchStep(),
        "fetch_url": FetchUrlStep(),
        "summarize": SummarizeStep(),
        "answer_generate": AnswerGenerateStep(),
    }
    print(f"  ✓ Registered {len(step_registry)} steps")
    print()
    
    # Step 2: Create workflow registry
    print("Step 2: Creating workflow registry...")
    workflow_registry = WorkflowRegistry(step_registry)
    print("  ✓ Workflow registry created")
    print()
    
    # Step 3: Register search_qa workflow
    print("Step 3: Registering search_qa workflow...")
    search_qa_workflow = create_search_qa_workflow()
    workflow_registry.register(search_qa_workflow)
    print(f"  ✓ Registered '{search_qa_workflow.name}'")
    print(f"    Description: {search_qa_workflow.description}")
    print(f"    Capabilities: {', '.join(search_qa_workflow.required_capabilities)}")
    print(f"    Tags: {', '.join(search_qa_workflow.tags)}")
    print(f"    Graph nodes: {len(search_qa_workflow.graph)}")
    print()
    
    # Step 4: Register a custom workflow
    print("Step 4: Registering custom fast workflow...")
    fast_workflow = WorkflowDefinition(
        name="search_qa_fast",
        description="Fast search QA without summarization",
        graph={
            "start": GraphNode(step_name="web_search", next_nodes=["fetch"]),
            "fetch": GraphNode(step_name="fetch_url", next_nodes=["answer"]),
            "answer": GraphNode(step_name="answer_generate", next_nodes=["end"]),
        },
        required_capabilities=["search", "fetch", "generate"],
        tags=["qa", "fast", "search"],
        metadata={
            "typical_latency_ms": 3000,
            "complexity": "low"
        }
    )
    workflow_registry.register(fast_workflow)
    print(f"  ✓ Registered '{fast_workflow.name}'")
    print(f"    Description: {fast_workflow.description}")
    print(f"    Tags: {', '.join(fast_workflow.tags)}")
    print()
    
    # Step 5: List all workflows
    print("Step 5: Listing all registered workflows...")
    all_workflows = workflow_registry.list_all()
    print(f"  Total workflows: {len(all_workflows)}")
    for wf_name in all_workflows:
        wf = workflow_registry.get(wf_name)
        print(f"    - {wf_name}: {wf.description}")
    print()
    
    # Step 6: Query by capability
    print("Step 6: Querying workflows by capability...")
    search_workflows = workflow_registry.get_by_capability("search")
    print(f"  Workflows with 'search' capability: {len(search_workflows)}")
    for wf in search_workflows:
        print(f"    - {wf.name}")
    print()
    
    # Step 7: Query by tag
    print("Step 7: Querying workflows by tag...")
    fast_workflows = workflow_registry.get_by_tag("fast")
    print(f"  Workflows with 'fast' tag: {len(fast_workflows)}")
    for wf in fast_workflows:
        print(f"    - {wf.name}")
    print()
    
    # Step 8: Execute a workflow (dry run without actual models)
    print("Step 8: Demonstrating workflow execution structure...")
    workflow = workflow_registry.get("search_qa_fast")
    print(f"  Workflow: {workflow.name}")
    print(f"  Execution flow:")
    
    # Show the execution flow
    current_node = "start"
    step_num = 1
    while current_node != "end":
        if current_node not in workflow.graph:
            break
        node = workflow.graph[current_node]
        print(f"    {step_num}. {node.step_name}")
        current_node = node.next_nodes[0] if node.next_nodes else "end"
        step_num += 1
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. Workflow Registry centralizes workflow definitions")
    print("  2. Workflows are validated during registration")
    print("  3. Workflows can be queried by capability and tags")
    print("  4. Graph structure defines execution flow")
    print("  5. Metadata enables cost-aware routing decisions")


if __name__ == "__main__":
    main()
