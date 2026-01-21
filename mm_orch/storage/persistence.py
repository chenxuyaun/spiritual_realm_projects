"""
Data Persistence Module.

This module implements data persistence functionality for the MuAI system:
- Consciousness state persistence with periodic saving
- Startup state recovery
- Data export and import for backup/migration

Requirements:
- 14.3: Periodically save consciousness module state to disk
- 14.4: Automatically restore state on system restart
- 14.5: Provide data export and import functionality

Properties verified:
- Property 38: 意识状态持久化往返
- Property 39: 数据导出导入往返
"""

import json
import os
import time
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import zipfile
import tempfile

from mm_orch.logger import get_logger
from mm_orch.exceptions import StorageError, ValidationError


logger = get_logger(__name__)


@dataclass
class PersistenceConfig:
    """Configuration for persistence operations."""

    consciousness_state_path: str = ".consciousness/state.json"
    chat_history_path: str = "data/chat_history"
    vector_db_path: str = "data/vector_db"
    export_path: str = "data/exports"
    auto_save_interval: float = 300.0  # 5 minutes
    max_backups: int = 5
    enable_auto_save: bool = True


class ConsciousnessPersistence:
    """
    Manages consciousness state persistence.

    Provides:
    - Periodic auto-save functionality
    - State recovery on startup
    - Backup management

    Implements requirements 14.3, 14.4.
    """

    def __init__(
        self,
        state_path: Optional[str] = None,
        auto_save_interval: float = 300.0,
        max_backups: int = 5,
        enable_auto_save: bool = True,
    ):
        """
        Initialize consciousness persistence.

        Args:
            state_path: Path to save state file
            auto_save_interval: Interval between auto-saves in seconds
            max_backups: Maximum number of backup files to keep
            enable_auto_save: Whether to enable automatic saving
        """
        self.state_path = state_path or ".consciousness/state.json"
        self.auto_save_interval = auto_save_interval
        self.max_backups = max_backups
        self.enable_auto_save = enable_auto_save

        self._last_save_time: float = 0.0
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()
        self._lock = threading.Lock()
        self._state_getter: Optional[Callable[[], Dict[str, Any]]] = None
        self._state_setter: Optional[Callable[[Dict[str, Any]], None]] = None

        # Ensure directory exists
        self._ensure_directory()

        logger.info(
            "ConsciousnessPersistence initialized",
            state_path=self.state_path,
            auto_save_interval=auto_save_interval,
        )

    def _ensure_directory(self) -> None:
        """Ensure the state directory exists."""
        directory = os.path.dirname(self.state_path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def register_state_handlers(
        self, getter: Callable[[], Dict[str, Any]], setter: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register state getter and setter functions.

        Args:
            getter: Function that returns current state as dict
            setter: Function that restores state from dict
        """
        self._state_getter = getter
        self._state_setter = setter

    def save_state(self, force: bool = False) -> bool:
        """
        Save consciousness state to disk.

        Property 38: State should be serializable and recoverable.

        Args:
            force: If True, save even if recently saved

        Returns:
            True if save was successful
        """
        if not self._state_getter:
            logger.warning("No state getter registered")
            return False

        with self._lock:
            # Check if we should save
            if not force and (time.time() - self._last_save_time) < 60:
                return False

            try:
                state = self._state_getter()

                # Add metadata
                state["_persistence_metadata"] = {
                    "saved_at": time.time(),
                    "saved_at_iso": datetime.now().isoformat(),
                    "version": "1.0",
                }

                # Create backup before overwriting
                if os.path.exists(self.state_path):
                    self._create_backup()

                # Write to temp file first, then rename (atomic operation)
                temp_path = f"{self.state_path}.tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)

                # Atomic rename
                os.replace(temp_path, self.state_path)

                self._last_save_time = time.time()
                logger.debug(f"Saved consciousness state to {self.state_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save consciousness state: {e}")
                return False

    def load_state(self) -> bool:
        """
        Load consciousness state from disk.

        Property 38: Loaded state should match saved state.

        Returns:
            True if load was successful
        """
        if not self._state_setter:
            logger.warning("No state setter registered")
            return False

        if not os.path.exists(self.state_path):
            logger.info(f"No state file found at {self.state_path}")
            return False

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Remove metadata before restoring
            metadata = state.pop("_persistence_metadata", {})

            self._state_setter(state)

            saved_at = metadata.get("saved_at_iso", "unknown")
            logger.info(f"Loaded consciousness state from {self.state_path} (saved at {saved_at})")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in state file: {e}")
            return self._try_load_backup()
        except Exception as e:
            logger.error(f"Failed to load consciousness state: {e}")
            return self._try_load_backup()

    def _create_backup(self) -> None:
        """Create a backup of the current state file."""
        if not os.path.exists(self.state_path):
            return

        backup_dir = os.path.dirname(self.state_path)
        backup_name = f"state_backup_{int(time.time())}.json"
        backup_path = os.path.join(backup_dir, backup_name)

        try:
            shutil.copy2(self.state_path, backup_path)
            self._cleanup_old_backups()
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond max_backups limit."""
        backup_dir = os.path.dirname(self.state_path)
        if not backup_dir:
            backup_dir = "."

        try:
            backups = []
            for f in os.listdir(backup_dir):
                if f.startswith("state_backup_") and f.endswith(".json"):
                    path = os.path.join(backup_dir, f)
                    backups.append((path, os.path.getmtime(path)))

            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)

            # Remove old backups
            for path, _ in backups[self.max_backups :]:
                os.remove(path)

        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    def _try_load_backup(self) -> bool:
        """Try to load from the most recent backup."""
        backup_dir = os.path.dirname(self.state_path)
        if not backup_dir:
            backup_dir = "."

        try:
            backups = []
            for f in os.listdir(backup_dir):
                if f.startswith("state_backup_") and f.endswith(".json"):
                    path = os.path.join(backup_dir, f)
                    backups.append((path, os.path.getmtime(path)))

            if not backups:
                return False

            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)

            # Try each backup
            for backup_path, _ in backups:
                try:
                    with open(backup_path, "r", encoding="utf-8") as f:
                        state = json.load(f)

                    state.pop("_persistence_metadata", None)
                    self._state_setter(state)

                    logger.info(f"Loaded consciousness state from backup: {backup_path}")
                    return True
                except Exception:
                    continue

            return False

        except Exception as e:
            logger.error(f"Failed to load from backup: {e}")
            return False

    def start_auto_save(self) -> None:
        """Start the auto-save background thread."""
        if not self.enable_auto_save:
            return

        if self._auto_save_thread and self._auto_save_thread.is_alive():
            return

        self._stop_auto_save.clear()
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop, daemon=True, name="consciousness-auto-save"
        )
        self._auto_save_thread.start()
        logger.info("Started consciousness auto-save thread")

    def stop_auto_save(self) -> None:
        """Stop the auto-save background thread."""
        self._stop_auto_save.set()
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5.0)
        logger.info("Stopped consciousness auto-save thread")

    def _auto_save_loop(self) -> None:
        """Background loop for periodic auto-saving."""
        while not self._stop_auto_save.is_set():
            self._stop_auto_save.wait(timeout=self.auto_save_interval)
            if not self._stop_auto_save.is_set():
                self.save_state(force=True)

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the persisted state.

        Returns:
            Dictionary with state file information
        """
        info = {
            "state_path": self.state_path,
            "exists": os.path.exists(self.state_path),
            "auto_save_enabled": self.enable_auto_save,
            "auto_save_interval": self.auto_save_interval,
            "last_save_time": self._last_save_time,
        }

        if info["exists"]:
            stat = os.stat(self.state_path)
            info["file_size"] = stat.st_size
            info["modified_at"] = stat.st_mtime
            info["modified_at_iso"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return info


class DataExporter:
    """
    Handles data export and import for backup and migration.

    Exports:
    - Consciousness state
    - Chat history
    - Vector database indices
    - Configuration

    Implements requirement 14.5.
    """

    def __init__(self, config: Optional[PersistenceConfig] = None):
        """
        Initialize the data exporter.

        Args:
            config: Persistence configuration
        """
        self.config = config or PersistenceConfig()
        self._ensure_export_directory()

    def _ensure_export_directory(self) -> None:
        """Ensure the export directory exists."""
        Path(self.config.export_path).mkdir(parents=True, exist_ok=True)

    def export_all(self, output_path: Optional[str] = None) -> str:
        """
        Export all system data to a zip file.

        Property 39: Exported data should be importable and equivalent.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to the created export file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or os.path.join(
            self.config.export_path, f"muai_export_{timestamp}.zip"
        )

        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Export manifest
                manifest = {
                    "version": "1.0",
                    "exported_at": datetime.now().isoformat(),
                    "components": [],
                }

                # Export consciousness state
                if os.path.exists(self.config.consciousness_state_path):
                    zf.write(self.config.consciousness_state_path, "consciousness/state.json")
                    manifest["components"].append("consciousness")

                # Export chat history
                if os.path.exists(self.config.chat_history_path):
                    self._add_directory_to_zip(zf, self.config.chat_history_path, "chat_history")
                    manifest["components"].append("chat_history")

                # Export vector database
                if os.path.exists(self.config.vector_db_path):
                    self._add_directory_to_zip(zf, self.config.vector_db_path, "vector_db")
                    manifest["components"].append("vector_db")

                # Write manifest
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            logger.info(f"Exported system data to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise StorageError(f"Export failed: {e}")

    def _add_directory_to_zip(self, zf: zipfile.ZipFile, source_dir: str, archive_dir: str) -> None:
        """Add a directory to a zip file."""
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, source_dir)
                archive_path = os.path.join(archive_dir, rel_path)
                zf.write(file_path, archive_path)

    def import_all(self, import_path: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Import system data from a zip file.

        Property 39: Imported data should match exported data.

        Args:
            import_path: Path to the import zip file
            overwrite: Whether to overwrite existing data

        Returns:
            Dictionary with import results
        """
        if not os.path.exists(import_path):
            raise ValidationError(f"Import file not found: {import_path}")

        results = {
            "success": True,
            "imported_components": [],
            "errors": [],
        }

        try:
            with zipfile.ZipFile(import_path, "r") as zf:
                # Read manifest
                try:
                    manifest_data = zf.read("manifest.json")
                    manifest = json.loads(manifest_data)
                except Exception:
                    manifest = {"components": []}

                # Import consciousness state
                if "consciousness/state.json" in zf.namelist():
                    try:
                        self._import_file(
                            zf,
                            "consciousness/state.json",
                            self.config.consciousness_state_path,
                            overwrite,
                        )
                        results["imported_components"].append("consciousness")
                    except Exception as e:
                        results["errors"].append(f"consciousness: {e}")

                # Import chat history
                chat_files = [n for n in zf.namelist() if n.startswith("chat_history/")]
                if chat_files:
                    try:
                        self._import_directory(
                            zf, "chat_history/", self.config.chat_history_path, overwrite
                        )
                        results["imported_components"].append("chat_history")
                    except Exception as e:
                        results["errors"].append(f"chat_history: {e}")

                # Import vector database
                vector_files = [n for n in zf.namelist() if n.startswith("vector_db/")]
                if vector_files:
                    try:
                        self._import_directory(
                            zf, "vector_db/", self.config.vector_db_path, overwrite
                        )
                        results["imported_components"].append("vector_db")
                    except Exception as e:
                        results["errors"].append(f"vector_db: {e}")

            if results["errors"]:
                results["success"] = False

            logger.info(f"Imported data from {import_path}: {results}")
            return results

        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            raise StorageError(f"Import failed: {e}")

    def _import_file(
        self, zf: zipfile.ZipFile, archive_path: str, target_path: str, overwrite: bool
    ) -> None:
        """Import a single file from the archive."""
        if os.path.exists(target_path) and not overwrite:
            raise ValidationError(f"File exists and overwrite=False: {target_path}")

        # Ensure directory exists
        target_dir = os.path.dirname(target_path)
        if target_dir:
            Path(target_dir).mkdir(parents=True, exist_ok=True)

        # Extract file
        data = zf.read(archive_path)
        with open(target_path, "wb") as f:
            f.write(data)

    def _import_directory(
        self, zf: zipfile.ZipFile, archive_prefix: str, target_dir: str, overwrite: bool
    ) -> None:
        """Import a directory from the archive."""
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        for name in zf.namelist():
            if name.startswith(archive_prefix) and not name.endswith("/"):
                rel_path = name[len(archive_prefix) :]
                target_path = os.path.join(target_dir, rel_path)

                if os.path.exists(target_path) and not overwrite:
                    continue

                # Ensure subdirectory exists
                target_subdir = os.path.dirname(target_path)
                if target_subdir:
                    Path(target_subdir).mkdir(parents=True, exist_ok=True)

                data = zf.read(name)
                with open(target_path, "wb") as f:
                    f.write(data)

    def export_consciousness_state(self, output_path: Optional[str] = None) -> str:
        """
        Export only consciousness state.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to the exported file
        """
        if not os.path.exists(self.config.consciousness_state_path):
            raise ValidationError("No consciousness state to export")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or os.path.join(
            self.config.export_path, f"consciousness_state_{timestamp}.json"
        )

        shutil.copy2(self.config.consciousness_state_path, output_path)
        logger.info(f"Exported consciousness state to {output_path}")
        return output_path

    def import_consciousness_state(self, import_path: str, overwrite: bool = False) -> bool:
        """
        Import consciousness state from a file.

        Args:
            import_path: Path to the state file
            overwrite: Whether to overwrite existing state

        Returns:
            True if import was successful
        """
        if not os.path.exists(import_path):
            raise ValidationError(f"Import file not found: {import_path}")

        if os.path.exists(self.config.consciousness_state_path) and not overwrite:
            raise ValidationError("State file exists and overwrite=False")

        # Validate JSON
        try:
            with open(import_path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in import file: {e}")

        # Ensure directory exists
        state_dir = os.path.dirname(self.config.consciousness_state_path)
        if state_dir:
            Path(state_dir).mkdir(parents=True, exist_ok=True)

        shutil.copy2(import_path, self.config.consciousness_state_path)
        logger.info(f"Imported consciousness state from {import_path}")
        return True

    def list_exports(self) -> List[Dict[str, Any]]:
        """
        List available export files.

        Returns:
            List of export file information
        """
        exports = []

        if not os.path.exists(self.config.export_path):
            return exports

        for filename in os.listdir(self.config.export_path):
            if filename.endswith(".zip") or filename.endswith(".json"):
                path = os.path.join(self.config.export_path, filename)
                stat = os.stat(path)
                exports.append(
                    {
                        "filename": filename,
                        "path": path,
                        "size": stat.st_size,
                        "created_at": stat.st_mtime,
                        "created_at_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                )

        # Sort by creation time (newest first)
        exports.sort(key=lambda x: x["created_at"], reverse=True)
        return exports


# Singleton instances
_consciousness_persistence: Optional[ConsciousnessPersistence] = None
_data_exporter: Optional[DataExporter] = None


def get_consciousness_persistence(
    state_path: Optional[str] = None, **kwargs
) -> ConsciousnessPersistence:
    """
    Get the singleton ConsciousnessPersistence instance.

    Args:
        state_path: Optional state file path
        **kwargs: Additional configuration options

    Returns:
        ConsciousnessPersistence singleton instance
    """
    global _consciousness_persistence

    if _consciousness_persistence is None:
        _consciousness_persistence = ConsciousnessPersistence(state_path=state_path, **kwargs)

    return _consciousness_persistence


def get_data_exporter(config: Optional[PersistenceConfig] = None) -> DataExporter:
    """
    Get the singleton DataExporter instance.

    Args:
        config: Optional persistence configuration

    Returns:
        DataExporter singleton instance
    """
    global _data_exporter

    if _data_exporter is None:
        _data_exporter = DataExporter(config)

    return _data_exporter


def reset_persistence() -> None:
    """Reset singleton instances (mainly for testing)."""
    global _consciousness_persistence, _data_exporter

    if _consciousness_persistence:
        _consciousness_persistence.stop_auto_save()

    _consciousness_persistence = None
    _data_exporter = None
