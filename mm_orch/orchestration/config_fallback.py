"""
Configuration Fallback Module - Phase B configuration with Phase A fallback.

This module provides configuration management that gracefully falls back to
Phase A behavior when Phase B configuration is missing or invalid.

The fallback strategy ensures:
1. Phase B components check for their configuration files
2. If missing/invalid, fall back to Phase A defaults
3. All fallback decisions are logged for debugging
4. No crashes due to missing configuration

Requirement 22.4: Configuration fallback logic
"""

import os
import json
import yaml
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from mm_orch.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ConfigFallbackResult:
    """
    Result of configuration loading with fallback information.

    Attributes:
        config: The loaded configuration (Phase B or Phase A fallback)
        used_fallback: Whether fallback was used
        fallback_reason: Reason for fallback (if used)
        config_source: Source of configuration ("phase_b", "phase_a", "default")
    """

    config: Dict[str, Any]
    used_fallback: bool
    fallback_reason: Optional[str] = None
    config_source: str = "phase_b"


class ConfigurationManager:
    """
    Configuration manager with Phase A fallback support.

    This class handles loading configuration files with automatic fallback
    to Phase A defaults when Phase B configuration is unavailable.

    Example:
        manager = ConfigurationManager()

        # Try to load Phase B router config, fall back to Phase A
        result = manager.load_router_config()
        if result.used_fallback:
            logger.warning(f"Using fallback: {result.fallback_reason}")

        router_config = result.config
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Optional custom config directory (defaults to ./config)
        """
        self.config_dir = Path(config_dir or "config")
        self.phase_b_dir = self.config_dir / "phase_b"
        self.phase_a_dir = self.config_dir

        logger.debug(
            "ConfigurationManager initialized",
            config_dir=str(self.config_dir),
            phase_b_dir=str(self.phase_b_dir),
        )

    def load_router_config(self) -> ConfigFallbackResult:
        """
        Load router configuration with fallback.

        Tries to load:
        1. config/phase_b/router.yaml (Phase B)
        2. config/router.yaml (Phase A)
        3. Default configuration

        Returns:
            ConfigFallbackResult with router configuration
        """
        # Try Phase B configuration
        phase_b_path = self.phase_b_dir / "router.yaml"
        if phase_b_path.exists():
            try:
                config = self._load_yaml(phase_b_path)
                if self._validate_router_config(config):
                    logger.info("Loaded Phase B router configuration", path=str(phase_b_path))
                    return ConfigFallbackResult(
                        config=config, used_fallback=False, config_source="phase_b"
                    )
                else:
                    logger.warning(
                        "Phase B router config invalid, falling back", path=str(phase_b_path)
                    )
            except Exception as e:
                logger.warning(f"Failed to load Phase B router config: {e}")

        # Try Phase A configuration
        phase_a_path = self.phase_a_dir / "router.yaml"
        if phase_a_path.exists():
            try:
                config = self._load_yaml(phase_a_path)
                logger.info("Using Phase A router configuration (fallback)", path=str(phase_a_path))
                return ConfigFallbackResult(
                    config=config,
                    used_fallback=True,
                    fallback_reason="Phase B config not found or invalid",
                    config_source="phase_a",
                )
            except Exception as e:
                logger.warning(f"Failed to load Phase A router config: {e}")

        # Use default configuration
        logger.info("Using default router configuration (fallback)")
        return ConfigFallbackResult(
            config=self._get_default_router_config(),
            used_fallback=True,
            fallback_reason="No configuration files found",
            config_source="default",
        )

    def load_workflow_registry_config(self) -> ConfigFallbackResult:
        """
        Load workflow registry configuration with fallback.

        Tries to load:
        1. config/phase_b/workflows.yaml (Phase B)
        2. Default workflow list (Phase A behavior)

        Returns:
            ConfigFallbackResult with workflow registry configuration
        """
        # Try Phase B configuration
        phase_b_path = self.phase_b_dir / "workflows.yaml"
        if phase_b_path.exists():
            try:
                config = self._load_yaml(phase_b_path)
                if self._validate_workflow_registry_config(config):
                    logger.info(
                        "Loaded Phase B workflow registry configuration", path=str(phase_b_path)
                    )
                    return ConfigFallbackResult(
                        config=config, used_fallback=False, config_source="phase_b"
                    )
                else:
                    logger.warning("Phase B workflow registry config invalid, falling back")
            except Exception as e:
                logger.warning(f"Failed to load Phase B workflow registry config: {e}")

        # Use default (Phase A behavior)
        logger.info("Using default workflow registry (Phase A fallback)")
        return ConfigFallbackResult(
            config=self._get_default_workflow_registry_config(),
            used_fallback=True,
            fallback_reason="Phase B config not found or invalid",
            config_source="default",
        )

    def load_tracer_config(self) -> ConfigFallbackResult:
        """
        Load tracer configuration with fallback.

        Tries to load:
        1. config/phase_b/tracer.yaml (Phase B)
        2. Default tracer configuration

        Returns:
            ConfigFallbackResult with tracer configuration
        """
        # Try Phase B configuration
        phase_b_path = self.phase_b_dir / "tracer.yaml"
        if phase_b_path.exists():
            try:
                config = self._load_yaml(phase_b_path)
                if self._validate_tracer_config(config):
                    logger.info("Loaded Phase B tracer configuration", path=str(phase_b_path))
                    return ConfigFallbackResult(
                        config=config, used_fallback=False, config_source="phase_b"
                    )
                else:
                    logger.warning("Phase B tracer config invalid, falling back")
            except Exception as e:
                logger.warning(f"Failed to load Phase B tracer config: {e}")

        # Use default
        logger.info("Using default tracer configuration (fallback)")
        return ConfigFallbackResult(
            config=self._get_default_tracer_config(),
            used_fallback=True,
            fallback_reason="Phase B config not found or invalid",
            config_source="default",
        )

    def load_model_registry_config(self) -> ConfigFallbackResult:
        """
        Load model registry configuration with fallback.

        Tries to load:
        1. config/phase_b/models.yaml (Phase B)
        2. config/models.yaml (Phase A)
        3. Default model configuration

        Returns:
            ConfigFallbackResult with model registry configuration
        """
        # Try Phase B configuration
        phase_b_path = self.phase_b_dir / "models.yaml"
        if phase_b_path.exists():
            try:
                config = self._load_yaml(phase_b_path)
                if self._validate_model_registry_config(config):
                    logger.info(
                        "Loaded Phase B model registry configuration", path=str(phase_b_path)
                    )
                    return ConfigFallbackResult(
                        config=config, used_fallback=False, config_source="phase_b"
                    )
                else:
                    logger.warning("Phase B model registry config invalid, falling back")
            except Exception as e:
                logger.warning(f"Failed to load Phase B model registry config: {e}")

        # Try Phase A configuration
        phase_a_path = self.phase_a_dir / "models.yaml"
        if phase_a_path.exists():
            try:
                config = self._load_yaml(phase_a_path)
                logger.info("Using Phase A model configuration (fallback)", path=str(phase_a_path))
                return ConfigFallbackResult(
                    config=config,
                    used_fallback=True,
                    fallback_reason="Phase B config not found or invalid",
                    config_source="phase_a",
                )
            except Exception as e:
                logger.warning(f"Failed to load Phase A model config: {e}")

        # Use default
        logger.info("Using default model configuration (fallback)")
        return ConfigFallbackResult(
            config=self._get_default_model_registry_config(),
            used_fallback=True,
            fallback_reason="No configuration files found",
            config_source="default",
        )

    def load_tool_registry_config(self) -> ConfigFallbackResult:
        """
        Load tool registry configuration with fallback.

        Tries to load:
        1. config/phase_b/tools.yaml (Phase B)
        2. Default tool configuration

        Returns:
            ConfigFallbackResult with tool registry configuration
        """
        # Try Phase B configuration
        phase_b_path = self.phase_b_dir / "tools.yaml"
        if phase_b_path.exists():
            try:
                config = self._load_yaml(phase_b_path)
                if self._validate_tool_registry_config(config):
                    logger.info(
                        "Loaded Phase B tool registry configuration", path=str(phase_b_path)
                    )
                    return ConfigFallbackResult(
                        config=config, used_fallback=False, config_source="phase_b"
                    )
                else:
                    logger.warning("Phase B tool registry config invalid, falling back")
            except Exception as e:
                logger.warning(f"Failed to load Phase B tool registry config: {e}")

        # Use default
        logger.info("Using default tool registry configuration (fallback)")
        return ConfigFallbackResult(
            config=self._get_default_tool_registry_config(),
            used_fallback=True,
            fallback_reason="Phase B config not found or invalid",
            config_source="default",
        )

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_router_config(self, config: Dict[str, Any]) -> bool:
        """Validate router configuration structure."""
        # Check for required fields
        if "router_version" not in config:
            return False

        # Version-specific validation
        version = config["router_version"]
        if version == "v3":
            return all(k in config for k in ["vectorizer_path", "classifier_path", "costs_path"])
        elif version == "v2":
            return all(k in config for k in ["vectorizer_path", "classifier_path"])
        elif version == "v1":
            return "rules" in config

        return False

    def _validate_workflow_registry_config(self, config: Dict[str, Any]) -> bool:
        """Validate workflow registry configuration structure."""
        if "workflows" not in config:
            return False

        workflows = config["workflows"]
        if not isinstance(workflows, list):
            return False

        # Each workflow should have name and steps
        for workflow in workflows:
            if not isinstance(workflow, dict):
                return False
            if "name" not in workflow or "steps" not in workflow:
                return False

        return True

    def _validate_tracer_config(self, config: Dict[str, Any]) -> bool:
        """Validate tracer configuration structure."""
        required_fields = ["output_path", "enabled"]
        return all(req_field in config for req_field in required_fields)

    def _validate_model_registry_config(self, config: Dict[str, Any]) -> bool:
        """Validate model registry configuration structure."""
        if "models" not in config:
            return False

        models = config["models"]
        if not isinstance(models, list):
            return False

        # Each model should have required metadata
        for model in models:
            if not isinstance(model, dict):
                return False
            required_fields = ["name", "capabilities", "expected_vram_mb"]
            if not all(req_field in model for req_field in required_fields):
                return False

        return True

    def _validate_tool_registry_config(self, config: Dict[str, Any]) -> bool:
        """Validate tool registry configuration structure."""
        if "tools" not in config:
            return False

        tools = config["tools"]
        if not isinstance(tools, list):
            return False

        # Each tool should have name and capabilities
        for tool in tools:
            if not isinstance(tool, dict):
                return False
            if "name" not in tool or "capabilities" not in tool:
                return False

        return True

    def _get_default_router_config(self) -> Dict[str, Any]:
        """Get default router configuration (Phase A behavior)."""
        return {
            "router_version": "v1",
            "rules": [
                {"pattern": "最新|今天|新闻", "workflow": "search_qa", "confidence": 0.9},
                {"pattern": "总结|摘要", "workflow": "summarize_url", "confidence": 0.8},
                {"pattern": "讲解|教学|课程", "workflow": "lesson_pack", "confidence": 0.85},
                {"pattern": "知识库|文档|规范", "workflow": "rag_qa", "confidence": 0.8},
            ],
            "default_workflow": "search_qa_fast",
            "fallback_confidence": 0.5,
        }

    def _get_default_workflow_registry_config(self) -> Dict[str, Any]:
        """Get default workflow registry configuration (Phase A workflows)."""
        return {
            "workflows": [
                {"name": "search_qa", "enabled": True},
                {"name": "lesson_pack", "enabled": True},
                {"name": "chat_generate", "enabled": True},
                {"name": "rag_qa", "enabled": True},
                {"name": "self_ask_search_qa", "enabled": True},
            ]
        }

    def _get_default_tracer_config(self) -> Dict[str, Any]:
        """Get default tracer configuration."""
        return {
            "enabled": True,
            "output_path": "data/traces/workflow_traces.jsonl",
            "include_step_traces": True,
            "include_quality_signals": True,
            "include_cost_stats": True,
        }

    def _get_default_model_registry_config(self) -> Dict[str, Any]:
        """Get default model registry configuration."""
        return {
            "models": [
                {
                    "name": "qwen-chat",
                    "capabilities": ["generate", "chat"],
                    "expected_vram_mb": 2048,
                    "supports_quant": True,
                    "preferred_device_policy": "gpu_on_demand",
                },
                {
                    "name": "t5-small",
                    "capabilities": ["summarize"],
                    "expected_vram_mb": 512,
                    "supports_quant": False,
                    "preferred_device_policy": "gpu_resident",
                },
            ]
        }

    def _get_default_tool_registry_config(self) -> Dict[str, Any]:
        """Get default tool registry configuration."""
        return {
            "tools": [
                {"name": "web_search", "capabilities": ["search", "web"], "enabled": True},
                {"name": "fetch_url", "capabilities": ["fetch", "web", "scrape"], "enabled": True},
            ]
        }


# Singleton instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(config_dir: Optional[str] = None) -> ConfigurationManager:
    """
    Get singleton ConfigurationManager instance.

    Args:
        config_dir: Optional custom config directory

    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager


def reset_config_manager() -> None:
    """Reset singleton ConfigurationManager instance."""
    global _config_manager
    _config_manager = None
