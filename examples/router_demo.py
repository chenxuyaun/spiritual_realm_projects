"""
Demo script for router implementations.

This script demonstrates how to use RouterV1, RouterV2, and RouterV3
for intelligent workflow selection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mm_orch.routing import RouterV1, RouterV2, RouterV3, RoutingRule
from mm_orch.orchestration.state import State


def demo_router_v1():
    """Demonstrate RouterV1 rule-based routing."""
    print("=" * 60)
    print("RouterV1: Rule-Based Routing")
    print("=" * 60)
    
    router = RouterV1()
    
    # Test questions
    questions = [
        "搜索最新的Python教程",
        "教我如何学习机器学习",
        "你好，今天天气怎么样？",
        "总结这篇文章的内容",
        "比较Python和Java的区别"
    ]
    
    for question in questions:
        state: State = {"meta": {"mode": "default"}}
        workflow, confidence, candidates = router.route(question, state)
        
        print(f"\nQuestion: {question}")
        print(f"Selected: {workflow} (confidence: {confidence:.3f})")
        print(f"Top 3 candidates:")
        for wf, score in candidates[:3]:
            print(f"  - {wf}: {score:.3f}")
    
    # Test with chat mode
    print("\n" + "-" * 60)
    print("Testing with chat mode:")
    state_chat: State = {"meta": {"mode": "chat"}}
    workflow, confidence, _ = router.route("你好", state_chat)
    print(f"Question: 你好 (mode=chat)")
    print(f"Selected: {workflow} (confidence: {confidence:.3f})")


def demo_router_v2():
    """Demonstrate RouterV2 classifier-based routing."""
    print("\n" + "=" * 60)
    print("RouterV2: Classifier-Based Routing")
    print("=" * 60)
    
    # Note: This requires trained models
    print("\nRouterV2 requires trained models:")
    print("  1. Train models using: python scripts/train_router_v2.py")
    print("  2. Models will be saved to: models/router_v2/")
    print("  3. Then initialize router with model paths")
    
    print("\nExample usage:")
    print("""
    from mm_orch.routing import RouterV2
    
    router = RouterV2(
        "models/router_v2/vectorizer.pkl",
        "models/router_v2/classifier.pkl"
    )
    
    state = {"meta": {"mode": "default"}}
    workflow, confidence, candidates = router.route("搜索Python", state)
    
    print(f"Selected: {workflow} (probability: {confidence:.3f})")
    print(f"Probability distribution:")
    for wf, prob in candidates:
        print(f"  {wf}: {prob:.3f}")
    """)


def demo_router_v3():
    """Demonstrate RouterV3 cost-aware routing."""
    print("\n" + "=" * 60)
    print("RouterV3: Cost-Aware Routing")
    print("=" * 60)
    
    # Note: This requires trained models and cost statistics
    print("\nRouterV3 requires trained models and cost statistics:")
    print("  1. Collect execution traces with cost data")
    print("  2. Train models using: python scripts/train_router_v3.py")
    print("  3. Models will be saved to: models/router_v3/")
    print("  4. Cost statistics should be in: data/cost_stats.json")
    
    print("\nExample usage:")
    print("""
    from mm_orch.routing import RouterV3
    
    router = RouterV3(
        "models/router_v3/vectorizer.pkl",
        "models/router_v3/classifier.pkl",
        "data/cost_stats.json",
        lambda_cost=0.1  # Weight for cost in scoring
    )
    
    # With chat mode
    state = {"meta": {"mode": "chat"}}
    workflow, score, candidates = router.route("你好", state)
    
    print(f"Selected: {workflow} (score: {score:.3f})")
    print(f"Cost-aware scores (quality - lambda * cost):")
    for wf, s in candidates[:3]:
        print(f"  {wf}: {s:.3f}")
    """)


def demo_custom_rules():
    """Demonstrate adding custom routing rules."""
    print("\n" + "=" * 60)
    print("Custom Routing Rules")
    print("=" * 60)
    
    router = RouterV1()
    
    # Add a custom rule
    custom_rule = RoutingRule(
        workflow_name="custom_analysis",
        keywords=["分析", "研究", "调查", "analyze", "research"],
        patterns=[
            r".*深入.*分析.*",
            r".*详细.*研究.*",
            r".*analyze.*in.*detail.*"
        ],
        base_weight=1.2,
        description="Deep analysis workflow"
    )
    
    router.add_rule(custom_rule)
    
    # Test with custom rule
    question = "深入分析Python的性能优化"
    state: State = {"meta": {"mode": "default"}}
    workflow, confidence, candidates = router.route(question, state)
    
    print(f"\nQuestion: {question}")
    print(f"Selected: {workflow} (confidence: {confidence:.3f})")
    print(f"\nNote: Custom rule 'custom_analysis' was added to the router")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Router Implementation Demo")
    print("=" * 60)
    
    demo_router_v1()
    demo_router_v2()
    demo_router_v3()
    demo_custom_rules()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
