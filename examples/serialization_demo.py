"""
Demonstration of State serialization utilities.

This example shows how to use the serialization module for:
- Basic serialization and deserialization
- Pretty-printing for debugging
- Safe serialization with fallback
- Validation before serialization
- Integration with workflow execution
"""

from mm_orch.orchestration.state import State
from mm_orch.orchestration.serialization import (
    state_to_json,
    json_to_state,
    state_to_dict,
    validate_state_serializable,
    serialize_state_safely,
)


def demo_basic_serialization():
    """Demonstrate basic serialization and deserialization."""
    print("=" * 60)
    print("Basic Serialization Demo")
    print("=" * 60)
    
    # Create a State
    state: State = {
        "question": "What is Python?",
        "final_answer": "Python is a high-level programming language.",
        "citations": ["https://python.org"],
        "meta": {
            "mode": "default",
            "router_version": "v3",
            "workflow": "search_qa"
        }
    }
    
    print("\nOriginal State:")
    print(state)
    
    # Serialize to JSON
    json_str = state_to_json(state)
    print("\nSerialized JSON (compact):")
    print(json_str)
    
    # Deserialize back
    restored = json_to_state(json_str)
    print("\nDeserialized State:")
    print(restored)
    
    # Verify round trip
    assert restored["question"] == state["question"]
    assert restored["final_answer"] == state["final_answer"]
    print("\n✅ Round trip successful!")


def demo_pretty_printing():
    """Demonstrate pretty-printed JSON output."""
    print("\n" + "=" * 60)
    print("Pretty Printing Demo")
    print("=" * 60)
    
    state: State = {
        "question": "Explain machine learning",
        "search_results": [
            {
                "title": "Machine Learning Basics",
                "url": "https://example.com/ml",
                "snippet": "ML is a subset of AI..."
            }
        ],
        "meta": {
            "mode": "default",
            "workflow": "search_qa"
        }
    }
    
    # Pretty-printed with 2-space indentation
    json_str = state_to_json(state, indent=2)
    print("\nPretty-printed JSON:")
    print(json_str)


def demo_nested_structures():
    """Demonstrate handling of nested structures."""
    print("\n" + "=" * 60)
    print("Nested Structures Demo")
    print("=" * 60)
    
    state: State = {
        "lesson_explain_structured": {
            "topic": "Python Basics",
            "grade": "High School",
            "sections": [
                {
                    "name": "Introduction",
                    "teacher_say": "Today we'll learn Python...",
                    "examples": ["print('Hello')", "x = 5"],
                    "key_points": ["Python is easy", "Python is powerful"]
                },
                {
                    "name": "Variables",
                    "teacher_say": "Variables store data...",
                    "examples": ["x = 10", "name = 'Alice'"],
                    "key_points": ["Variables have names", "Variables have types"]
                }
            ]
        },
        "meta": {
            "workflow": "lesson_pack"
        }
    }
    
    print("\nState with nested structures:")
    json_str = state_to_json(state, indent=2)
    print(json_str)
    
    # Deserialize and verify nested structure preserved
    restored = json_to_state(json_str)
    sections = restored["lesson_explain_structured"]["sections"]
    print(f"\n✅ Preserved {len(sections)} sections")
    print(f"✅ First section has {len(sections[0]['examples'])} examples")


def demo_unicode_support():
    """Demonstrate Unicode and special character support."""
    print("\n" + "=" * 60)
    print("Unicode Support Demo")
    print("=" * 60)
    
    state: State = {
        "question": "什么是Python？",
        "final_answer": "Python是一种高级编程语言。",
        "meta": {
            "language": "中文",
            "emoji": "🐍"
        }
    }
    
    print("\nState with Unicode characters:")
    json_str = state_to_json(state, indent=2)
    print(json_str)
    
    # Deserialize and verify
    restored = json_to_state(json_str)
    assert "Python" in restored["question"]
    assert "🐍" in restored["meta"]["emoji"]
    print("\n✅ Unicode characters preserved!")


def demo_validation():
    """Demonstrate validation before serialization."""
    print("\n" + "=" * 60)
    print("Validation Demo")
    print("=" * 60)
    
    # Valid state
    valid_state: State = {
        "question": "Test question",
        "meta": {}
    }
    
    is_valid, error = validate_state_serializable(valid_state)
    print(f"\nValid state check: {is_valid}")
    print(f"Error: {error}")
    
    # The validation will pass for most States since they're just dicts
    assert is_valid
    print("✅ Validation successful!")


def demo_safe_serialization():
    """Demonstrate safe serialization with fallback."""
    print("\n" + "=" * 60)
    print("Safe Serialization Demo")
    print("=" * 60)
    
    state: State = {
        "question": "Test question",
        "meta": {"key": "value"}
    }
    
    # Safe serialization (won't crash on error)
    json_str = serialize_state_safely(state, fallback='{"error": true}')
    print("\nSafely serialized:")
    print(json_str)
    
    # This should succeed and return actual serialization
    assert "question" in json_str
    print("✅ Safe serialization successful!")


def demo_state_to_dict():
    """Demonstrate state_to_dict for intermediate processing."""
    print("\n" + "=" * 60)
    print("State to Dict Demo")
    print("=" * 60)
    
    state: State = {
        "question": "Test",
        "citations": ["url1", "url2"],
        "meta": {"nested": {"value": 42}}
    }
    
    # Convert to plain dict
    result = state_to_dict(state)
    print("\nConverted to dict:")
    print(result)
    print(f"\nType: {type(result)}")
    print(f"Citations count: {len(result['citations'])}")
    print(f"Nested value: {result['meta']['nested']['value']}")
    print("✅ Conversion successful!")


def demo_real_world_workflow():
    """Demonstrate serialization in a real workflow scenario."""
    print("\n" + "=" * 60)
    print("Real-World Workflow Demo")
    print("=" * 60)
    
    # Simulate a search_qa workflow state
    state: State = {
        "question": "What is machine learning?",
        "search_results": [
            {
                "title": "Machine Learning - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Machine_learning",
                "snippet": "Machine learning is a field of study..."
            },
            {
                "title": "Introduction to ML",
                "url": "https://example.com/ml-intro",
                "snippet": "ML enables computers to learn..."
            }
        ],
        "docs": {
            "https://en.wikipedia.org/wiki/Machine_learning": "Full Wikipedia article content...",
            "https://example.com/ml-intro": "Full tutorial content..."
        },
        "summaries": {
            "https://en.wikipedia.org/wiki/Machine_learning": "ML is a subset of AI that enables systems to learn...",
            "https://example.com/ml-intro": "ML tutorial covering basics..."
        },
        "final_answer": "Machine learning is a field of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "citations": [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://example.com/ml-intro"
        ],
        "meta": {
            "mode": "default",
            "router_version": "v3",
            "workflow": "search_qa",
            "timestamp": 1234567890.0
        }
    }
    
    print("\nWorkflow State Summary:")
    print(f"Question: {state['question']}")
    print(f"Search results: {len(state['search_results'])}")
    print(f"Documents fetched: {len(state['docs'])}")
    print(f"Citations: {len(state['citations'])}")
    
    # Serialize for trace logging
    json_str = state_to_json(state)
    print(f"\nSerialized size: {len(json_str)} bytes")
    
    # Deserialize and verify
    restored = json_to_state(json_str)
    assert len(restored["search_results"]) == 2
    assert len(restored["docs"]) == 2
    assert len(restored["citations"]) == 2
    print("✅ Workflow state serialization successful!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("State Serialization Demonstration")
    print("=" * 60)
    
    demo_basic_serialization()
    demo_pretty_printing()
    demo_nested_structures()
    demo_unicode_support()
    demo_validation()
    demo_safe_serialization()
    demo_state_to_dict()
    demo_real_world_workflow()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed successfully! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
