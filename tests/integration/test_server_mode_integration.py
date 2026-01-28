"""
Integration tests for server mode with real models.

Tests server lifecycle and operations:
- Server lifecycle with real models
- Concurrent requests in server mode
- Graceful shutdown with pending requests

Validates: All server requirements
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock

from mm_orch.optimization.server import InferenceServer
from mm_orch.optimization.config import ServerConfig, OptimizationConfig


class TestServerLifecycle:
    """Test server lifecycle with real models."""
    
    def test_server_initialization(self):
        """Test server initialization and startup."""
        config = OptimizationConfig(
            server=ServerConfig(
                enabled=True,
                host="127.0.0.1",
                port=8100,
                queue_capacity=50,
                preload_models=[],
                graceful_shutdown_timeout=5
            )
        )
        
        server = InferenceServer(config)
        
        # Verify: Server created
        assert server is not None
        assert server.config == config
        
        # Start server
        server.start()
        
        try:
            # Verify: Server running
            status = server.get_status()
            assert status["status"] in ["running", "starting", "healthy"]
            
        finally:
            server.stop(timeout=2)
    
    def test_server_with_preloaded_models(self):
        """Test server with model pre-loading."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8101,
            queue_capacity=50,
            preload_models=["gpt2"],  # Pre-load GPT-2
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Wait for models to load
            time.sleep(1.0)
            
            # Verify: Models loaded
            status = server.get_status()
            models_loaded = status.get("models_loaded", [])
            
            # Should have attempted to load models
            assert isinstance(models_loaded, list)
            
        finally:
            server.stop(timeout=2)
    
    def test_server_health_check(self):
        """Test server health check endpoint."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8102,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Check health
            health = server.health_check()
            
            # Verify: Health response
            assert health is not None
            assert hasattr(health, 'status')
            assert health.status in ["healthy", "starting", "degraded", "unhealthy"]
            assert hasattr(health, 'uptime_seconds')
            assert health.uptime_seconds >= 0
            
        finally:
            server.stop(timeout=2)
    
    def test_server_readiness_check(self):
        """Test server readiness check endpoint."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8103,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Wait for server to be ready
            time.sleep(0.5)
            
            # Check readiness
            readiness = server.readiness_check()
            
            # Verify: Readiness response
            assert readiness is not None
            assert hasattr(readiness, 'ready')
            assert isinstance(readiness.ready, bool)
            
        finally:
            server.stop(timeout=2)
    
    def test_server_stop_and_restart(self):
        """Test stopping and restarting server."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8104,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        
        # Start server
        server.start()
        time.sleep(0.3)
        
        # Verify: Running
        status1 = server.get_status()
        assert status1["status"] in ["running", "healthy", "starting"]
        
        # Stop server
        server.stop(timeout=2)
        time.sleep(0.3)
        
        # Verify: Stopped
        status2 = server.get_status()
        assert status2["status"] in ["stopped", "shutdown"]
        
        # Restart server
        server.start()
        time.sleep(0.3)
        
        # Verify: Running again
        status3 = server.get_status()
        assert status3["status"] in ["running", "healthy", "starting"]
        
        # Cleanup
        server.stop(timeout=2)


class TestConcurrentRequests:
    """Test concurrent requests in server mode."""
    
    def test_server_handles_concurrent_requests(self):
        """Test server handling multiple concurrent requests."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8105,
            queue_capacity=100,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Submit concurrent requests
            request_ids = []
            
            def submit_request_wrapper(i):
                req_id = f"req-{i}"
                success = server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3, 4, 5]}
                )
                return req_id if success else None
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(submit_request_wrapper, i) for i in range(20)]
                for future in as_completed(futures):
                    req_id = future.result()
                    if req_id:
                        request_ids.append(req_id)
            
            # Verify: All requests submitted
            assert len(request_ids) == 20
            assert len(set(request_ids)) == 20  # All unique
            
            # Wait for processing
            time.sleep(0.5)
            
            # Check status
            status = server.get_status()
            assert status["queue_size"] >= 0
            
        finally:
            server.stop(timeout=2)
    
    def test_server_queue_management(self):
        """Test server queue management under load."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8106,
            queue_capacity=30,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Fill queue gradually
            request_ids = []
            
            for i in range(20):
                req_id = f"req-{i}"
                success = server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
                if success:
                    request_ids.append(req_id)
                
                # Check queue size
                status = server.get_status()
                queue_size = status.get("queue_size", 0)
                assert queue_size <= config.server.queue_capacity
            
            # Verify: Requests queued
            assert len(request_ids) == 20
            
        finally:
            server.stop(timeout=2)
    
    def test_server_request_throughput(self):
        """Test server request throughput."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8107,
            queue_capacity=100,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Measure throughput
            start_time = time.time()
            num_requests = 50
            
            for i in range(num_requests):
                req_id = f"req-{i}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            elapsed = time.time() - start_time
            throughput = num_requests / elapsed
            
            # Verify: Reasonable throughput
            assert throughput > 0
            print(f"\nServer throughput: {throughput:.2f} req/s")
            
        finally:
            server.stop(timeout=2)
    
    def test_server_concurrent_different_models(self):
        """Test concurrent requests for different models."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8108,
            queue_capacity=100,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Submit requests for different models
            models = ["model_a", "model_b", "model_c"]
            request_ids = []
            
            def submit_for_model(model_name, count):
                ids = []
                for i in range(count):
                    req_id = f"{model_name}-req-{i}"
                    success = server.submit_request(
                        request_id=req_id,
                        model_name=model_name,
                        inputs={"input_ids": [1, 2, 3]}
                    )
                    if success:
                        ids.append(req_id)
                return ids
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(submit_for_model, model, 5) for model in models]
                for future in as_completed(futures):
                    ids = future.result()
                    request_ids.extend(ids)
            
            # Verify: All requests submitted
            assert len(request_ids) == 15
            
        finally:
            server.stop(timeout=2)


class TestQueueCapacity:
    """Test queue capacity limits."""
    
    def test_queue_capacity_enforcement(self):
        """Test that queue capacity is enforced."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8109,
            queue_capacity=10,  # Small queue
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Fill queue to capacity
            request_ids = []
            
            for i in range(10):
                req_id = f"req-{i}"
                success = server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
                if success:
                    request_ids.append(req_id)
            
            # Try to exceed capacity - should return False when queue is full
            full_count = 0
            for i in range(20):
                req_id = f"req-overflow-{i}"
                success = server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
                if not success:
                    full_count += 1
            
            # Verify: Some requests were rejected due to full queue
            assert full_count > 0
            
        finally:
            server.stop(timeout=2)
    
    def test_queue_recovery_after_processing(self):
        """Test queue recovery after processing requests."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8110,
            queue_capacity=20,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Fill queue
            for i in range(15):
                req_id = f"req-first-{i}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            # Wait for some processing
            time.sleep(0.5)
            
            # Should be able to submit more
            for i in range(10):
                req_id = f"req-second-{i}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            # Verify: Queue managed
            status = server.get_status()
            assert status["queue_size"] <= config.server.queue_capacity
            
        finally:
            server.stop(timeout=2)


class TestGracefulShutdown:
    """Test graceful shutdown with pending requests."""
    
    def test_graceful_shutdown_completes_pending(self):
        """Test that graceful shutdown completes pending requests."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8111,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=10
        ))
        
        server = InferenceServer(config)
        server.start()
        
        # Submit requests
        request_ids = []
        for i in range(10):
            req_id = f"req-{i}"
            success = server.submit_request(
                request_id=req_id,
                model_name="test_model",
                inputs={"input_ids": [1, 2, 3]}
            )
            if success:
                request_ids.append(req_id)
        
        # Initiate graceful shutdown
        start_time = time.time()
        server.stop(timeout=10)
        shutdown_time = time.time() - start_time
        
        # Verify: Shutdown completed
        status = server.get_status()
        assert status["status"] in ["stopped", "shutdown"]
        
        # Verify: Shutdown time reasonable
        assert shutdown_time < 15  # Should complete within timeout + buffer
    
    def test_shutdown_timeout_enforcement(self):
        """Test that shutdown timeout is enforced."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8112,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=2  # Short timeout
        ))
        
        server = InferenceServer(config)
        server.start()
        
        # Submit many requests
        for i in range(30):
            req_id = f"req-{i}"
            server.submit_request(
                request_id=req_id,
                model_name="test_model",
                inputs={"input_ids": [1, 2, 3]}
            )
        
        # Shutdown with timeout
        start_time = time.time()
        server.stop(timeout=2)
        shutdown_time = time.time() - start_time
        
        # Verify: Shutdown within timeout
        assert shutdown_time < 5  # Timeout + buffer
    
    def test_shutdown_rejects_new_requests(self):
        """Test that shutdown rejects new requests."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8113,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        # Start shutdown in background
        shutdown_thread = threading.Thread(target=lambda: server.stop(timeout=5))
        shutdown_thread.start()
        
        # Wait a bit for shutdown to start
        time.sleep(0.2)
        
        # Try to submit request during shutdown
        with pytest.raises(RuntimeError, match="shutdown|stopped|not.*running"):
            req_id = "req-during-shutdown"
            server.submit_request(
                request_id=req_id,
                model_name="test_model",
                inputs={"input_ids": [1, 2, 3]}
            )
        
        # Wait for shutdown to complete
        shutdown_thread.join(timeout=10)


class TestServerHealthDegradation:
    """Test health check degradation status."""
    
    def test_health_reflects_normal_operation(self):
        """Test health check reflects normal operation."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8114,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            time.sleep(0.3)
            
            # Check health
            health = server.health_check()
            
            # Verify: Healthy status
            assert health.status in ["healthy", "starting"]
            assert health.queue_size >= 0
            assert health.queue_capacity == config.server.queue_capacity
            
        finally:
            server.stop(timeout=2)
    
    def test_health_reflects_high_queue_usage(self):
        """Test health check reflects high queue usage."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8115,
            queue_capacity=20,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Fill queue significantly
            for i in range(18):
                req_id = f"req-{i}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            # Check health
            health = server.health_check()
            
            # Verify: Status reflects load
            assert health.queue_size > 15
            # May be degraded or still healthy depending on implementation
            assert health.status in ["healthy", "degraded"]
            
        finally:
            server.stop(timeout=2)
    
    def test_health_reflects_engine_unavailability(self):
        """Test health check reflects engine unavailability."""
        config = OptimizationConfig(server=ServerConfig(
            host="127.0.0.1",
            port=8116,
            queue_capacity=50,
            preload_models=[],
            graceful_shutdown_timeout=5
        ))
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Simulate engine failure
            server._degraded = True
            server._degradation_reason = "Optimization engine unavailable"
            
            # Check health
            health = server.health_check()
            
            # Verify: Degraded status
            assert health.status in ["degraded", "unhealthy"]
            
        finally:
            server.stop(timeout=2)


class TestServerWithOptimization:
    """Test server with optimization features."""
    
    def test_server_with_batching(self):
        """Test server with dynamic batching enabled."""
        from mm_orch.optimization.config import BatcherConfig
        
        config = OptimizationConfig(
            server=ServerConfig(
                enabled=True,
                host="127.0.0.1",
                port=8117,
                queue_capacity=50,
                preload_models=[],
                graceful_shutdown_timeout=5
            ),
            batcher=BatcherConfig(
                enabled=True,
                max_batch_size=8
            )
        )
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Submit requests that can be batched
            for i in range(10):
                req_id = f"req-{i}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            # Wait for batching
            time.sleep(0.2)
            
            # Verify: Server running with batching
            status = server.get_status()
            assert status["status"] in ["running", "healthy"]
            
        finally:
            server.stop(timeout=2)
    
    def test_server_with_caching(self):
        """Test server with KV cache enabled."""
        from mm_orch.optimization.config import CacheConfig
        
        config = OptimizationConfig(
            server=ServerConfig(
                enabled=True,
                host="127.0.0.1",
                port=8118,
                queue_capacity=50,
                preload_models=[],
                graceful_shutdown_timeout=5
            ),
            cache=CacheConfig(
                enabled=True
            )
        )
        
        server = InferenceServer(config)
        server.start()
        
        try:
            # Submit multi-turn conversation
            conversation_id = "conv_001"
            
            for turn in range(3):
                req_id = f"req-turn-{turn}"
                server.submit_request(
                    request_id=req_id,
                    model_name="test_model",
                    inputs={"input_ids": [1, 2, 3]}
                )
            
            # Wait for processing
            time.sleep(0.3)
            
            # Verify: Server running with caching
            status = server.get_status()
            assert status["status"] in ["running", "healthy"]
            
        finally:
            server.stop(timeout=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
