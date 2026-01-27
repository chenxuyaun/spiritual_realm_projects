"""
Mode-Aware Routing Demo

This script demonstrates how mode features affect routing decisions in Router v3.
It shows the difference between routing in "chat" mode vs "default" mode.

Requirements: 21.1, 21.2, 21.4

Run with: python examples/mode_aware_routing_demo.py
"""

from mm_orch.orchestration.state_utils import create_state, get_mode_from_state
from mm_orch.runtime.conversation import ConversationManager


def demo_state_creation():
    """Demonstrate State creation with mode."""
    print("=" * 70)
    print("Demo 1: State Creation with Mode")
    print("=" * 70)
    print()
    
    # Create state for CLI single-shot query
    print("1. Creating State for CLI single-shot query:")
    state_cli = create_state("What is Python?", mode="default")
    print(f"   Question: {state_cli['question']}")
    print(f"   Mode: {get_mode_from_state(state_cli)}")
    print()
    
    # Create state for chat interaction
    print("2. Creating State for chat interaction:")
    state_chat = create_state(
        "Tell me more about that",
        mode="chat",
        conversation_id="session123",
        turn_index=3
    )
    print(f"   Question: {state_chat['question']}")
    print(f"   Mode: {get_mode_from_state(state_chat)}")
    print(f"   Conversation ID: {state_chat.get('conversation_id')}")
    print(f"   Turn Index: {state_chat.get('turn_index')}")
    print()


def demo_conversation_manager_mode():
    """Demonstrate ConversationManager mode detection."""
    print("=" * 70)
    print("Demo 2: Conversation Manager Mode Detection")
    print("=" * 70)
    print()
    
    manager = ConversationManager()
    
    print("1. Empty conversation:")
    print(f"   Mode: {manager.get_mode()}")
    print(f"   Message count: {manager.get_message_count()}")
    print()
    
    print("2. After adding user message:")
    manager.add_user_input("Hello, how are you?")
    print(f"   Mode: {manager.get_mode()}")
    print(f"   Message count: {manager.get_message_count()}")
    print()
    
    print("3. After adding assistant response:")
    manager.add_assistant_response("I'm doing well, thank you!")
    print(f"   Mode: {manager.get_mode()}")
    print(f"   Message count: {manager.get_message_count()}")
    print()
    
    print("4. Conversation manager to_dict includes mode:")
    data = manager.to_dict()
    print(f"   Mode in dict: {data.get('mode')}")
    print()


def demo_mode_feature_encoding():
    """Demonstrate mode feature encoding."""
    print("=" * 70)
    print("Demo 3: Mode Feature Encoding")
    print("=" * 70)
    print()
    
    print("Mode features are encoded as binary (one-hot):")
    print()
    
    # Chat mode
    mode_chat = "chat"
    mode_is_chat_1 = 1 if mode_chat == "chat" else 0
    print(f"1. mode='chat' → mode_is_chat = {mode_is_chat_1}")
    print()
    
    # Default mode
    mode_default = "default"
    mode_is_chat_2 = 1 if mode_default == "chat" else 0
    print(f"2. mode='default' → mode_is_chat = {mode_is_chat_2}")
    print()
    
    print("This binary feature is concatenated with text features:")
    print("   [text_feature_1, text_feature_2, ..., mode_is_chat]")
    print()


def demo_router_v3_mode_usage():
    """Demonstrate how Router v3 uses mode features."""
    print("=" * 70)
    print("Demo 4: Router v3 Mode Feature Usage")
    print("=" * 70)
    print()
    
    print("Router v3 routing process with mode features:")
    print()
    
    print("1. Extract mode from State.meta")
    print("   mode = state.get('meta', {}).get('mode', 'default')")
    print()
    
    print("2. Encode mode as one-hot feature")
    print("   mode_is_chat = 1 if mode == 'chat' else 0")
    print()
    
    print("3. Vectorize question text")
    print("   X_text = vectorizer.transform([question])")
    print()
    
    print("4. Concatenate text and mode features")
    print("   X_with_mode = np.hstack([X_text.toarray(), [[mode_is_chat]]])")
    print()
    
    print("5. Predict using combined features")
    print("   quality_probs = classifier.predict_proba(X_with_mode)[0]")
    print()
    
    print("6. Calculate cost-aware scores")
    print("   score = quality - lambda_cost * cost")
    print()
    
    print("Note: Router v3 requires trained models to run.")
    print("      Train models using: python scripts/train_router_v3.py")
    print()


def demo_mode_impact_on_routing():
    """Demonstrate how mode affects routing decisions."""
    print("=" * 70)
    print("Demo 5: Mode Impact on Routing Decisions")
    print("=" * 70)
    print()
    
    print("Example: Question = 'Tell me about Python'")
    print()
    
    print("With mode='default' (single-shot query):")
    print("   Top workflow: search_qa")
    print("   Reasoning: Factual question → web search")
    print("   Score: 0.75")
    print()
    
    print("With mode='chat' (conversation):")
    print("   Top workflow: chat_generate")
    print("   Reasoning: Conversational context → chat response")
    print("   Score: 0.82")
    print()
    
    print("The mode feature helps the router understand context:")
    print("   - Chat mode → prefer conversational workflows")
    print("   - Default mode → prefer search/factual workflows")
    print()


def demo_training_with_mode():
    """Demonstrate training with mode features."""
    print("=" * 70)
    print("Demo 6: Training Router v3 with Mode Features")
    print("=" * 70)
    print()
    
    print("Training process:")
    print()
    
    print("1. Collect execution traces with mode metadata:")
    print("   {")
    print("     'question': 'Hello',")
    print("     'mode': 'chat',")
    print("     'chosen_workflow': 'chat_generate',")
    print("     'quality_signals': {...},")
    print("     'success': true")
    print("   }")
    print()
    
    print("2. Extract mode features from traces:")
    print("   mode = trace.get('mode', 'default')")
    print("   mode_is_chat = 1 if mode == 'chat' else 0")
    print()
    
    print("3. Create training data:")
    print("   questions = ['Hello', 'What is Python?', ...]")
    print("   mode_features = [1, 0, ...]  # 1=chat, 0=default")
    print("   workflows = ['chat_generate', 'search_qa', ...]")
    print()
    
    print("4. Train classifier with combined features:")
    print("   X = [text_features + mode_features]")
    print("   classifier.fit(X, workflows)")
    print()
    
    print("5. Save models with metadata:")
    print("   metadata = {")
    print("     'mode_feature_description': 'Binary: 1=chat, 0=default',")
    print("     'feature_order': 'text_features + mode_is_chat',")
    print("     ...")
    print("   }")
    print()
    
    print("Run training:")
    print("   python scripts/train_router_v3.py \\")
    print("       --traces data/traces/workflow_traces.jsonl \\")
    print("       --costs data/cost_stats.json \\")
    print("       --output-dir models/router_v3")
    print()


def main():
    """Run all demos."""
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "MODE-AWARE ROUTING DEMO" + " " * 25 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    
    demo_state_creation()
    print()
    
    demo_conversation_manager_mode()
    print()
    
    demo_mode_feature_encoding()
    print()
    
    demo_router_v3_mode_usage()
    print()
    
    demo_mode_impact_on_routing()
    print()
    
    demo_training_with_mode()
    
    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Mode is set in State.meta for routing")
    print("  2. ConversationManager detects mode automatically")
    print("  3. Router v3 encodes mode as binary feature")
    print("  4. Mode affects workflow selection preferences")
    print("  5. Training includes mode features from traces")
    print()
    print("Next Steps:")
    print("  - Collect traces with mode metadata")
    print("  - Train Router v3 with mode features")
    print("  - Deploy and monitor routing decisions")
    print()


if __name__ == "__main__":
    main()
