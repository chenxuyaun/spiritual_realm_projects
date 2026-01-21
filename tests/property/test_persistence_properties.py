"""
Property-based tests for data persistence.

Tests the following properties:
- Property 38: 意识状态持久化往返 (Consciousness state persistence round-trip)
- Property 39: 数据导出导入往返 (Data export/import round-trip)

Requirements verified:
- 14.3: Periodically save consciousness module state to disk
- 14.4: Automatically restore state on system restart
- 14.5: Provide data export and import functionality
"""

import os
import json
import tempfile
import shutil
from typing import Any, Dict

import pytest
from hypothesis import given, strategies as st, settings, assume

from mm_orch.storage.persistence import (
    ConsciousnessPersistence,
    DataExporter,
    PersistenceConfig,
    reset_persistence,
)
from mm_orch.consciousness.core import ConsciousnessCore
from mm_orch.schemas import ConsciousnessState


# Strategies for generating test data
@st.composite
def consciousness_state_dict(draw):
    """Generate a valid consciousness state dictionary."""
    return {
        "self_model": {
            "state": {
                "status": draw(st.sampled_from(["idle", "processing", "error_recovery"])),
                "current_task": draw(st.none() | st.text(min_size=1, max_size=50)),
                "load": draw(st.floats(min_value=0.0, max_value=1.0)),
                "health": draw(st.floats(min_value=0.0, max_value=1.0)),
            },
            "capabilities": {},
            "performance_history": [],
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "world_model": {
            "entities": {},
            "users": {},
            "knowledge": {
                "environment": {"type": "development", "resources": {}},
                "facts": {},
                "rules": {},
            },
            "entity_types": ["concept", "topic", "tool", "model", "workflow"],
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "metacognition": {
            "active_tasks": {},
            "completed_tasks": [],
            "strategy_records": {},
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "motivation": {
            "goals": {},
            "drive_levels": {
                "curiosity": draw(st.floats(min_value=0.0, max_value=1.0)),
                "helpfulness": draw(st.floats(min_value=0.0, max_value=1.0)),
                "accuracy": draw(st.floats(min_value=0.0, max_value=1.0)),
                "efficiency": draw(st.floats(min_value=0.0, max_value=1.0)),
                "creativity": draw(st.floats(min_value=0.0, max_value=1.0)),
            },
            "goal_counter": draw(st.integers(min_value=0, max_value=1000)),
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "emotion": {
            "valence": draw(st.floats(min_value=-1.0, max_value=1.0)),
            "arousal": draw(st.floats(min_value=0.0, max_value=1.0)),
            "event_history": [],
            "last_update": draw(st.floats(min_value=0, max_value=2e9)),
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "development": {
            "current_stage": draw(st.sampled_from(["infant", "child", "adolescent", "adult"])),
            "metrics": {
                "task_count": draw(st.integers(min_value=0, max_value=10000)),
                "success_count": draw(st.integers(min_value=0, max_value=10000)),
                "total_score": draw(st.floats(min_value=0.0, max_value=10000.0)),
                "time_active_hours": draw(st.floats(min_value=0.0, max_value=10000.0)),
                "stage_start_time": draw(st.floats(min_value=0, max_value=2e9)),
            },
            "learning_records": [],
            "record_counter": draw(st.integers(min_value=0, max_value=10000)),
            "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        },
        "initialized_at": draw(st.floats(min_value=0, max_value=2e9)),
        "saved_at": draw(st.floats(min_value=0, max_value=2e9)),
    }


@st.composite
def simple_json_dict(draw):
    """Generate a simple JSON-serializable dictionary."""
    return draw(st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('L', 'N'),
            whitelist_characters='_'
        )),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
        ),
        max_size=10
    ))


class TestConsciousnessPersistenceRoundTrip:
    """
    Property 38: 意识状态持久化往返
    
    For any consciousness module state, saving to disk and loading back
    should result in equivalent state (all key field values match).
    
    **Validates: Requirements 14.3, 14.4**
    """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_persistence()
    
    @given(state=consciousness_state_dict())
    @settings(max_examples=50, deadline=None)
    def test_consciousness_state_round_trip(self, state: Dict[str, Any]):
        """
        Feature: muai-orchestration-system, Property 38: 意识状态持久化往返
        
        For any consciousness state, save → load should preserve all key values.
        """
        state_path = os.path.join(self.temp_dir, "state.json")
        
        # Create persistence with state handlers
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False
        )
        
        # Track saved and loaded state
        saved_state = None
        loaded_state = None
        
        def getter():
            return state.copy()
        
        def setter(data):
            nonlocal loaded_state
            loaded_state = data
        
        persistence.register_state_handlers(getter, setter)
        
        # Save state
        result = persistence.save_state(force=True)
        assert result is True, "Save should succeed"
        
        # Verify file exists
        assert os.path.exists(state_path), "State file should exist"
        
        # Load state
        result = persistence.load_state()
        assert result is True, "Load should succeed"
        
        # Verify round-trip equivalence for key fields
        assert loaded_state is not None, "State should be loaded"
        
        # Check self_model state
        assert loaded_state["self_model"]["state"]["status"] == state["self_model"]["state"]["status"]
        assert loaded_state["self_model"]["state"]["load"] == state["self_model"]["state"]["load"]
        assert loaded_state["self_model"]["state"]["health"] == state["self_model"]["state"]["health"]
        
        # Check emotion state
        assert loaded_state["emotion"]["valence"] == state["emotion"]["valence"]
        assert loaded_state["emotion"]["arousal"] == state["emotion"]["arousal"]
        
        # Check motivation drive levels
        for drive, level in state["motivation"]["drive_levels"].items():
            assert loaded_state["motivation"]["drive_levels"][drive] == level
        
        # Check development stage
        assert loaded_state["development"]["current_stage"] == state["development"]["current_stage"]
    
    @given(state=consciousness_state_dict())
    @settings(max_examples=30, deadline=None)
    def test_consciousness_core_round_trip(self, state: Dict[str, Any]):
        """
        Feature: muai-orchestration-system, Property 38: 意识状态持久化往返
        
        ConsciousnessCore save/load should preserve state.
        """
        state_path = os.path.join(self.temp_dir, "core_state.json")
        
        # Create and configure consciousness core
        core = ConsciousnessCore(config={"state_path": state_path})
        
        # Set emotion values from generated state
        core.emotion.set_emotion(
            state["emotion"]["valence"],
            state["emotion"]["arousal"]
        )
        
        # Set development stage
        core.development.set_stage(state["development"]["current_stage"])
        
        # Set drive levels
        for drive, level in state["motivation"]["drive_levels"].items():
            core.motivation.set_drive_level(drive, level)
        
        # Save state
        result = core.save_state(state_path, force=True)
        assert result is True, "Save should succeed"
        
        # Create new core and load state
        new_core = ConsciousnessCore()
        result = new_core.load_state(state_path)
        assert result is True, "Load should succeed"
        
        # Verify emotion state
        orig_valence, orig_arousal = core.emotion.get_emotion_values()
        new_valence, new_arousal = new_core.emotion.get_emotion_values()
        
        # Allow small floating point differences due to decay
        assert abs(new_valence - orig_valence) < 0.1
        assert abs(new_arousal - orig_arousal) < 0.1
        
        # Verify development stage
        assert new_core.development.get_current_stage() == core.development.get_current_stage()
    
    def test_backup_creation_on_save(self):
        """Test that backups are created when saving over existing state."""
        state_path = os.path.join(self.temp_dir, "state.json")
        
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False,
            max_backups=3
        )
        
        state_v1 = {"version": 1, "data": "first"}
        state_v2 = {"version": 2, "data": "second"}
        
        persistence.register_state_handlers(
            lambda: state_v1,
            lambda x: None
        )
        
        # First save
        persistence.save_state(force=True)
        
        # Update state and save again
        persistence.register_state_handlers(
            lambda: state_v2,
            lambda x: None
        )
        persistence.save_state(force=True)
        
        # Check that backup was created
        backup_files = [f for f in os.listdir(self.temp_dir) if f.startswith("state_backup_")]
        assert len(backup_files) >= 1, "Backup should be created"
    
    def test_load_from_backup_on_corruption(self):
        """Test that loading falls back to backup on corruption."""
        state_path = os.path.join(self.temp_dir, "state.json")
        
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False
        )
        
        valid_state = {"valid": True, "data": "test"}
        loaded_state = None
        
        def setter(data):
            nonlocal loaded_state
            loaded_state = data
        
        persistence.register_state_handlers(
            lambda: valid_state,
            setter
        )
        
        # Save valid state
        persistence.save_state(force=True)
        
        # Corrupt the main state file
        with open(state_path, 'w') as f:
            f.write("invalid json {{{")
        
        # Create a valid backup
        backup_path = os.path.join(self.temp_dir, "state_backup_12345.json")
        with open(backup_path, 'w') as f:
            json.dump(valid_state, f)
        
        # Load should fall back to backup
        result = persistence.load_state()
        assert result is True, "Should load from backup"
        assert loaded_state is not None


class TestDataExportImportRoundTrip:
    """
    Property 39: 数据导出导入往返
    
    For any system data (chat history, vector db, config), export → import
    should result in equivalent data.
    
    **Validates: Requirement 14.5**
    """
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PersistenceConfig(
            consciousness_state_path=os.path.join(self.temp_dir, ".consciousness", "state.json"),
            chat_history_path=os.path.join(self.temp_dir, "chat_history"),
            vector_db_path=os.path.join(self.temp_dir, "vector_db"),
            export_path=os.path.join(self.temp_dir, "exports"),
        )
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_persistence()
    
    @given(state=consciousness_state_dict())
    @settings(max_examples=30, deadline=None)
    def test_consciousness_state_export_import_round_trip(self, state: Dict[str, Any]):
        """
        Feature: muai-orchestration-system, Property 39: 数据导出导入往返
        
        Consciousness state export → import should preserve data.
        """
        exporter = DataExporter(self.config)
        
        # Create consciousness state file
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        os.makedirs(state_dir, exist_ok=True)
        
        with open(self.config.consciousness_state_path, 'w') as f:
            json.dump(state, f)
        
        # Export
        export_path = exporter.export_consciousness_state()
        assert os.path.exists(export_path), "Export file should exist"
        
        # Remove original
        os.remove(self.config.consciousness_state_path)
        
        # Import
        result = exporter.import_consciousness_state(export_path, overwrite=True)
        assert result is True, "Import should succeed"
        
        # Verify round-trip
        with open(self.config.consciousness_state_path, 'r') as f:
            imported_state = json.load(f)
        
        # Check key fields match
        assert imported_state["emotion"]["valence"] == state["emotion"]["valence"]
        assert imported_state["emotion"]["arousal"] == state["emotion"]["arousal"]
        assert imported_state["development"]["current_stage"] == state["development"]["current_stage"]
    
    @given(chat_data=st.lists(simple_json_dict(), min_size=1, max_size=5))
    @settings(max_examples=20, deadline=None)
    def test_full_export_import_round_trip(self, chat_data):
        """
        Feature: muai-orchestration-system, Property 39: 数据导出导入往返
        
        Full system export → import should preserve all data.
        """
        exporter = DataExporter(self.config)
        
        # Create test data
        # Consciousness state
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        os.makedirs(state_dir, exist_ok=True)
        test_state = {"test": "consciousness", "value": 42}
        with open(self.config.consciousness_state_path, 'w') as f:
            json.dump(test_state, f)
        
        # Chat history
        os.makedirs(self.config.chat_history_path, exist_ok=True)
        for i, data in enumerate(chat_data):
            chat_file = os.path.join(self.config.chat_history_path, f"session_{i}.json")
            with open(chat_file, 'w') as f:
                json.dump(data, f)
        
        # Export all
        export_path = exporter.export_all()
        assert os.path.exists(export_path), "Export file should exist"
        
        # Clear original data
        shutil.rmtree(state_dir, ignore_errors=True)
        shutil.rmtree(self.config.chat_history_path, ignore_errors=True)
        
        # Import all
        result = exporter.import_all(export_path, overwrite=True)
        assert result["success"] is True, f"Import should succeed: {result}"
        assert "consciousness" in result["imported_components"]
        assert "chat_history" in result["imported_components"]
        
        # Verify consciousness state
        with open(self.config.consciousness_state_path, 'r') as f:
            imported_state = json.load(f)
        assert imported_state["test"] == "consciousness"
        assert imported_state["value"] == 42
        
        # Verify chat history
        for i, original_data in enumerate(chat_data):
            chat_file = os.path.join(self.config.chat_history_path, f"session_{i}.json")
            assert os.path.exists(chat_file), f"Chat file {i} should exist"
            with open(chat_file, 'r') as f:
                imported_data = json.load(f)
            assert imported_data == original_data
    
    def test_export_creates_valid_zip(self):
        """Test that export creates a valid zip file with manifest."""
        exporter = DataExporter(self.config)
        
        # Create minimal test data
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        os.makedirs(state_dir, exist_ok=True)
        with open(self.config.consciousness_state_path, 'w') as f:
            json.dump({"test": True}, f)
        
        # Export
        export_path = exporter.export_all()
        
        # Verify zip structure
        import zipfile
        with zipfile.ZipFile(export_path, 'r') as zf:
            names = zf.namelist()
            assert "manifest.json" in names, "Manifest should be in zip"
            assert "consciousness/state.json" in names, "Consciousness state should be in zip"
            
            # Verify manifest
            manifest = json.loads(zf.read("manifest.json"))
            assert "version" in manifest
            assert "exported_at" in manifest
            assert "components" in manifest
    
    def test_import_without_overwrite_fails_on_existing(self):
        """Test that import fails when overwrite=False and data exists."""
        exporter = DataExporter(self.config)
        
        # Create test data
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        os.makedirs(state_dir, exist_ok=True)
        with open(self.config.consciousness_state_path, 'w') as f:
            json.dump({"original": True}, f)
        
        # Export
        export_path = exporter.export_consciousness_state()
        
        # Try to import without overwrite
        from mm_orch.exceptions import ValidationError
        with pytest.raises(ValidationError):
            exporter.import_consciousness_state(export_path, overwrite=False)
    
    def test_list_exports(self):
        """Test listing available exports."""
        exporter = DataExporter(self.config)
        
        # Create test data and export multiple times
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        os.makedirs(state_dir, exist_ok=True)
        with open(self.config.consciousness_state_path, 'w') as f:
            json.dump({"test": True}, f)
        
        # Create multiple exports
        export1 = exporter.export_all()
        export2 = exporter.export_consciousness_state()
        
        # List exports
        exports = exporter.list_exports()
        assert len(exports) >= 2, "Should list at least 2 exports"
        
        # Verify export info structure
        for export in exports:
            assert "filename" in export
            assert "path" in export
            assert "size" in export
            assert "created_at" in export


class TestPersistenceEdgeCases:
    """Test edge cases and error handling for persistence."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_persistence()
    
    def test_save_without_registered_handlers(self):
        """Test that save fails gracefully without handlers."""
        state_path = os.path.join(self.temp_dir, "state.json")
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False
        )
        
        result = persistence.save_state(force=True)
        assert result is False, "Save should fail without handlers"
    
    def test_load_nonexistent_file(self):
        """Test that load handles nonexistent file gracefully."""
        state_path = os.path.join(self.temp_dir, "nonexistent.json")
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False
        )
        
        persistence.register_state_handlers(
            lambda: {},
            lambda x: None
        )
        
        result = persistence.load_state()
        assert result is False, "Load should fail for nonexistent file"
    
    def test_get_state_info(self):
        """Test getting state file information."""
        state_path = os.path.join(self.temp_dir, "state.json")
        persistence = ConsciousnessPersistence(
            state_path=state_path,
            enable_auto_save=False
        )
        
        # Before save
        info = persistence.get_state_info()
        assert info["exists"] is False
        
        # After save
        persistence.register_state_handlers(
            lambda: {"test": True},
            lambda x: None
        )
        persistence.save_state(force=True)
        
        info = persistence.get_state_info()
        assert info["exists"] is True
        assert "file_size" in info
        assert "modified_at" in info
