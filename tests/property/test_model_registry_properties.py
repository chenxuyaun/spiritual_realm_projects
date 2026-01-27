"""
Property-based tests for Model Registry.

Tests universal properties that must hold for all model registrations
and capability queries using Hypothesis for randomized testing.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from mm_orch.registries.model_registry import ModelRegistry, ModelMetadata


# Strategy for generating valid model names
model_names = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',)),  # Exclude surrogates
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() != "")

# Strategy for generating capability lists
capabilities = st.lists(
    st.text(
        alphabet=st.characters(blacklist_categories=('Cs',)),
        min_size=1,
        max_size=20
    ).filter(lambda x: x.strip() != ""),
    min_size=1,
    max_size=5,
    unique=True
)

# Strategy for generating VRAM values (non-negative integers)
vram_values = st.integers(min_value=0, max_value=100000)

# Strategy for generating boolean values
bool_values = st.booleans()

# Strategy for generating valid device policies
device_policies = st.sampled_from(["gpu_on_demand", "cpu_only", "gpu_resident"])

# Strategy for generating model paths
model_paths = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',)),
    min_size=1,
    max_size=100
).filter(lambda x: x.strip() != "")

# Strategy for generating quantization configs (optional)
quantization_configs = st.one_of(
    st.none(),
    st.dictionaries(
        keys=st.text(
            alphabet=st.characters(blacklist_categories=('Cs',)),
            min_size=1,
            max_size=30
        ).filter(lambda x: x.strip() != ""),
        values=st.one_of(
            st.booleans(),
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.integers(min_value=0, max_value=1000),
            st.text(max_size=50)
        ),
        max_size=5
    )
)


class TestModelMetadataValidation:
    """
    Property 7: Model Metadata Validation
    
    For any model registration attempt, if any required metadata field
    (name, capabilities, expected_vram_mb, supports_quant, preferred_device_policy)
    is missing, the Model_Registry must reject the registration with a descriptive error.
    
    Validates: Requirements 5.2, 5.4
    """

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        caps=capabilities,
        vram=vram_values,
        supports_quant=bool_values,
        device_policy=device_policies,
        model_path=model_paths,
        quant_config=quantization_configs
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=None
    )
    def test_valid_metadata_accepted(self, name, caps, vram, supports_quant, device_policy, model_path, quant_config):
        """
        Property 7: Valid Metadata Acceptance
        
        Verifies that model metadata with all required fields is accepted
        and can be registered successfully.
        """
        # Create valid metadata
        metadata = ModelMetadata(
            name=name,
            capabilities=caps,
            expected_vram_mb=vram,
            supports_quant=supports_quant,
            preferred_device_policy=device_policy,
            model_path=model_path,
            quantization_config=quant_config
        )
        
        # Should not raise any exception
        assert metadata.name == name
        assert metadata.capabilities == caps
        assert metadata.expected_vram_mb == vram
        assert metadata.supports_quant == supports_quant
        assert metadata.preferred_device_policy == device_policy
        assert metadata.model_path == model_path
        assert metadata.quantization_config == quant_config

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        caps=capabilities,
        vram=vram_values,
        supports_quant=bool_values,
        device_policy=device_policies,
        model_path=model_paths
    )
    @settings(max_examples=100)
    def test_missing_name_rejected(self, caps, vram, supports_quant, device_policy, model_path):
        """
        Property 7: Missing Name Rejection
        
        Verifies that metadata with missing or empty name is rejected
        with a descriptive error.
        """
        with pytest.raises(ValueError, match="Missing required field: name"):
            ModelMetadata(
                name="",
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        vram=vram_values,
        supports_quant=bool_values,
        device_policy=device_policies,
        model_path=model_paths
    )
    @settings(max_examples=100)
    def test_empty_capabilities_rejected(self, name, vram, supports_quant, device_policy, model_path):
        """
        Property 7: Empty Capabilities Rejection
        
        Verifies that metadata with empty capabilities list is rejected
        with a descriptive error.
        """
        with pytest.raises(ValueError, match="must have at least one capability"):
            ModelMetadata(
                name=name,
                capabilities=[],
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        caps=capabilities,
        vram=vram_values,
        supports_quant=bool_values,
        device_policy=device_policies
    )
    @settings(max_examples=100)
    def test_missing_model_path_rejected(self, name, caps, vram, supports_quant, device_policy):
        """
        Property 7: Missing Model Path Rejection
        
        Verifies that metadata with missing or empty model_path is rejected
        with a descriptive error.
        """
        with pytest.raises(ValueError, match="Missing required field: model_path"):
            ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=""
            )

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        caps=capabilities,
        negative_vram=st.integers(max_value=-1),
        supports_quant=bool_values,
        device_policy=device_policies,
        model_path=model_paths
    )
    @settings(max_examples=100)
    def test_negative_vram_rejected(self, name, caps, negative_vram, supports_quant, device_policy, model_path):
        """
        Property 7: Negative VRAM Rejection
        
        Verifies that metadata with negative VRAM value is rejected
        with a descriptive error.
        """
        with pytest.raises(ValueError, match="expected_vram_mb must be non-negative"):
            ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=negative_vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        caps=capabilities,
        vram=vram_values,
        supports_quant=bool_values,
        invalid_policy=st.text(min_size=1, max_size=30).filter(
            lambda x: x not in ["gpu_on_demand", "cpu_only", "gpu_resident"]
        ),
        model_path=model_paths
    )
    @settings(max_examples=100)
    def test_invalid_device_policy_rejected(self, name, caps, vram, supports_quant, invalid_policy, model_path):
        """
        Property 7: Invalid Device Policy Rejection
        
        Verifies that metadata with invalid device policy is rejected
        with a descriptive error.
        """
        with pytest.raises(ValueError, match="preferred_device_policy must be one of"):
            ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=invalid_policy,
                model_path=model_path
            )

    # Feature: extensible-orchestration-phase-b, Property 7: Model Metadata Validation
    @given(
        name=model_names,
        caps=capabilities,
        vram=vram_values,
        supports_quant=bool_values,
        device_policy=device_policies,
        model_path=model_paths,
        quant_config=quantization_configs
    )
    @settings(max_examples=100)
    def test_valid_metadata_registration(self, name, caps, vram, supports_quant, device_policy, model_path, quant_config):
        """
        Property 7: Valid Metadata Registration
        
        Verifies that valid metadata can be successfully registered
        in the Model Registry.
        """
        registry = ModelRegistry()
        
        metadata = ModelMetadata(
            name=name,
            capabilities=caps,
            expected_vram_mb=vram,
            supports_quant=supports_quant,
            preferred_device_policy=device_policy,
            model_path=model_path,
            quantization_config=quant_config
        )
        
        # Should not raise any exception
        registry.register(metadata)
        
        # Should be retrievable
        retrieved = registry.get(name)
        assert retrieved.name == name
        assert retrieved.capabilities == caps
        assert retrieved.expected_vram_mb == vram


class TestCapabilityQueryCorrectness:
    """
    Property 8: Capability Query Correctness (Model Registry part)
    
    For any capability string and Model_Registry, querying by that capability
    must return only items whose capabilities list contains that capability,
    and must return all such items.
    
    Validates: Requirements 5.3, 5.4
    """

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]  # Unique by name
        ),
        query_capability=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() != "")
    )
    @settings(max_examples=100)
    def test_capability_query_returns_only_matching_models(self, models, query_capability):
        """
        Property 8: Capability Query Correctness
        
        Verifies that capability queries return exactly the models that
        have the queried capability - no more, no less.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Query by capability
        matching_models = registry.find_by_capability(query_capability)
        
        # Determine expected matches
        expected_matches = {
            name for name, caps, _, _, _, _ in models
            if query_capability in caps
        }
        
        # Verify correctness
        actual_matches = {m.name for m in matching_models}
        assert actual_matches == expected_matches, \
            f"Query for '{query_capability}' must return exactly the models with that capability"
        
        # Verify no false positives
        for model_metadata in matching_models:
            assert query_capability in model_metadata.capabilities, \
                f"Model '{model_metadata.name}' in results must have capability '{query_capability}'"
        
        # Verify no false negatives
        for name, caps, _, _, _, _ in models:
            if query_capability in caps:
                assert name in actual_matches, \
                    f"Model '{name}' with capability '{query_capability}' must be in results"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=2,
            max_size=15,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_completeness(self, models):
        """
        Property 8: Capability Query Completeness
        
        Verifies that for each capability present in any model,
        querying for that capability returns all models with it.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Collect all unique capabilities
        all_capabilities = set()
        for _, caps, _, _, _, _ in models:
            all_capabilities.update(caps)
        
        # For each capability, verify query returns all matching models
        for capability in all_capabilities:
            matching_models = registry.find_by_capability(capability)
            
            # Find expected models with this capability
            expected_models = {
                name for name, caps, _, _, _, _ in models
                if capability in caps
            }
            
            actual_models = {m.name for m in matching_models}
            assert actual_models == expected_models, \
                f"Query for '{capability}' must return all models with that capability"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        ),
        nonexistent_capability=st.text(min_size=1, max_size=30).filter(lambda x: x.strip() != "")
    )
    @settings(max_examples=100)
    def test_capability_query_empty_for_nonexistent(self, models, nonexistent_capability):
        """
        Property 8: Capability Query for Nonexistent Capability
        
        Verifies that querying for a capability that no model has
        returns an empty list.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _, _, _, _ in models:
            all_capabilities.update(caps)
        
        # Assume the query capability doesn't exist in any model
        assume(nonexistent_capability not in all_capabilities)
        
        # Query should return empty list
        matching_models = registry.find_by_capability(nonexistent_capability)
        assert matching_models == [], \
            f"Query for nonexistent capability '{nonexistent_capability}' must return empty list"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_consistency(self, models):
        """
        Property 8: Capability Query Consistency
        
        Verifies that querying the same capability multiple times
        returns consistent results.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _, _, _, _ in models:
            all_capabilities.update(caps)
        
        # Query each capability multiple times
        for capability in all_capabilities:
            result1 = registry.find_by_capability(capability)
            result2 = registry.find_by_capability(capability)
            result3 = registry.find_by_capability(capability)
            
            # Results must be consistent
            names1 = {m.name for m in result1}
            names2 = {m.name for m in result2}
            names3 = {m.name for m in result3}
            
            assert names1 == names2 == names3, \
                f"Multiple queries for '{capability}' must return consistent results"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_intersection(self, models):
        """
        Property 8: Capability Query Intersection
        
        Verifies that models with multiple capabilities appear in
        query results for each of their capabilities.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # For each model with multiple capabilities
        for name, caps, _, _, _, _ in models:
            if len(caps) > 1:
                # Model must appear in results for each of its capabilities
                for capability in caps:
                    matching_models = registry.find_by_capability(capability)
                    matching_names = {m.name for m in matching_models}
                    assert name in matching_names, \
                        f"Model '{name}' with capability '{capability}' must appear in query results"

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_preserves_metadata(self, models):
        """
        Property 8: Capability Query Preserves Metadata
        
        Verifies that models returned by capability queries have
        all their original metadata intact.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _, _, _, _ in models:
            all_capabilities.update(caps)
        
        # For each capability query, verify metadata is intact
        for capability in all_capabilities:
            matching_models = registry.find_by_capability(capability)
            
            for model_metadata in matching_models:
                # Find original model data
                original = next(
                    (name, caps, vram, sq, dp, mp) 
                    for name, caps, vram, sq, dp, mp in models 
                    if name == model_metadata.name
                )
                name, caps, vram, supports_quant, device_policy, model_path = original
                
                # Verify all metadata fields are preserved
                assert model_metadata.name == name
                assert model_metadata.capabilities == caps
                assert model_metadata.expected_vram_mb == vram
                assert model_metadata.supports_quant == supports_quant
                assert model_metadata.preferred_device_policy == device_policy
                assert model_metadata.model_path == model_path

    # Feature: extensible-orchestration-phase-b, Property 8: Capability Query Correctness
    @given(
        models=st.lists(
            st.tuples(
                model_names,
                capabilities,
                vram_values,
                bool_values,
                device_policies,
                model_paths
            ),
            min_size=1,
            max_size=20,
            unique_by=lambda x: x[0]
        )
    )
    @settings(max_examples=100)
    def test_capability_query_after_operations(self, models):
        """
        Property 8: Capability Query After Registry Operations
        
        Verifies that capability queries remain correct even after
        other registry operations like listing and checking existence.
        """
        registry = ModelRegistry()
        
        # Register all models
        for name, caps, vram, supports_quant, device_policy, model_path in models:
            metadata = ModelMetadata(
                name=name,
                capabilities=caps,
                expected_vram_mb=vram,
                supports_quant=supports_quant,
                preferred_device_policy=device_policy,
                model_path=model_path
            )
            registry.register(metadata)
        
        # Perform various operations
        _ = registry.list_all()
        for name, _, _, _, _, _ in models:
            _ = registry.has(name)
        
        # Collect all capabilities
        all_capabilities = set()
        for _, caps, _, _, _, _ in models:
            all_capabilities.update(caps)
        
        # Verify capability queries still work correctly
        for capability in all_capabilities:
            matching_models = registry.find_by_capability(capability)
            
            expected_models = {
                name for name, caps, _, _, _, _ in models
                if capability in caps
            }
            
            actual_models = {m.name for m in matching_models}
            assert actual_models == expected_models, \
                f"Query for '{capability}' must still return correct results after operations"
