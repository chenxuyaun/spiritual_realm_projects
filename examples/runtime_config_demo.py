"""
Runtime Configuration Update Demo

This example demonstrates how to use the ConfigurationManager to update
non-critical configuration parameters at runtime without restarting the system.
"""

import time
from mm_orch.optimization import (
    create_config_manager,
    ConfigurationChange,
)


def on_config_change(change: ConfigurationChange):
    """Callback function invoked when configuration changes."""
    print(f"  [CALLBACK] Configuration changed: {change.parameter}")
    print(f"             Old value: {change.old_value}")
    print(f"             New value: {change.new_value}")
    print(f"             Timestamp: {change.timestamp}")


def main():
    print("=" * 70)
    print("Runtime Configuration Update Demo")
    print("=" * 70)
    
    # Create configuration manager with default configuration
    print("\n1. Creating configuration manager with defaults...")
    manager = create_config_manager()
    config = manager.get_config()
    
    print(f"   Initial batch size: {config.batcher.max_batch_size}")
    print(f"   Initial cache size: {config.cache.max_memory_mb} MB")
    print(f"   Initial queue capacity: {config.server.queue_capacity}")
    
    # Register a callback to be notified of changes
    print("\n2. Registering change callback...")
    manager.register_change_callback(on_config_change)
    
    # Update individual parameters
    print("\n3. Updating individual parameters...")
    print("   Updating batch size from 32 to 64...")
    manager.update_parameter("batcher.max_batch_size", 64)
    
    print("\n   Updating cache size from 4096 MB to 8192 MB...")
    manager.update_parameter("cache.max_memory_mb", 8192)
    
    print("\n   Updating queue capacity from 100 to 200...")
    manager.update_parameter("server.queue_capacity", 200)
    
    # Verify updates were applied
    config = manager.get_config()
    print("\n4. Verifying updates...")
    print(f"   Current batch size: {config.batcher.max_batch_size}")
    print(f"   Current cache size: {config.cache.max_memory_mb} MB")
    print(f"   Current queue capacity: {config.server.queue_capacity}")
    
    # View change history
    print("\n5. Viewing change history...")
    history = manager.get_change_history(limit=5)
    for i, change in enumerate(history, 1):
        print(f"   Change {i}: {change.parameter}")
        print(f"            {change.old_value} -> {change.new_value}")
    
    # Demonstrate reload from dictionary
    print("\n6. Reloading configuration from dictionary...")
    new_config = {
        "batcher": {
            "max_batch_size": 128,
            "batch_timeout_ms": 100,
        },
        "cache": {
            "max_memory_mb": 16384,
        },
    }
    
    changes = manager.reload_config(config_dict=new_config)
    print(f"   Applied {len(changes)} changes:")
    for change in changes:
        print(f"     - {change.parameter}: {change.old_value} -> {change.new_value}")
    
    # Show non-critical parameters
    print("\n7. Non-critical parameters that can be updated at runtime:")
    params = manager.get_non_critical_parameters()
    for param in sorted(params):
        print(f"   - {param}")
    
    # Demonstrate error handling for critical parameters
    print("\n8. Attempting to update critical parameter (should fail)...")
    try:
        manager.update_parameter("vllm.enabled", False)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   Expected error: {e}")
    
    # Demonstrate validation
    print("\n9. Attempting invalid update (should fail)...")
    try:
        manager.update_parameter("batcher.max_batch_size", -1)
        print("   ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"   Expected error: {e}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
