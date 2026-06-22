"""Legacy feature-flag routing for incremental migration.

This module provides the LegacyRouter class that decides whether each pipeline
stage uses the new (refactored) behavior or the legacy behavior, based on the
current feature flag configuration. When a flag is disabled, the corresponding
stage uses legacy behavior, maintaining backward compatibility during migration.

Requirements: 13.1, 13.2, 13.5
"""

from __future__ import annotations

import logging
from enum import Enum

from finflow_agent.pipeline.feature_flags import FeatureFlags

logger = logging.getLogger(__name__)


class PipelineRoute(str, Enum):
    """Route decision for a pipeline stage.

    NEW_PIPELINE indicates the refactored behavior should be used.
    LEGACY_PIPELINE indicates the pre-refactor behavior should be used.
    """

    NEW_PIPELINE = "new_pipeline"
    LEGACY_PIPELINE = "legacy_pipeline"


class LegacyRouter:
    """Routes pipeline stages to new or legacy behavior based on feature flags.

    When a feature flag is disabled, the corresponding pipeline stage uses the
    legacy behavior, ensuring backward compatibility during incremental migration
    (Requirement 13.2).

    Flag transitions are logged with previous state, new state, and timestamp
    (Requirement 13.5).
    """

    def __init__(self, flags: FeatureFlags) -> None:
        """Initialize the router with the current feature flag configuration.

        Args:
            flags: The active feature flag configuration.
        """
        self._flags = flags

    @property
    def flags(self) -> FeatureFlags:
        """Return the current feature flags."""
        return self._flags

    def route_extraction(self) -> PipelineRoute:
        """Route the extraction stage.

        Returns NEW_PIPELINE if ENABLE_SEMANTIC_DRAFT_PIPELINE is enabled,
        otherwise LEGACY_PIPELINE for backward-compatible extraction.
        """
        if self._flags.ENABLE_SEMANTIC_DRAFT_PIPELINE:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def route_grounding(self) -> PipelineRoute:
        """Route the grounding stage.

        Returns NEW_PIPELINE if ENABLE_PREFLIGHT_GROUNDING is enabled,
        otherwise LEGACY_PIPELINE for execution-time grounding.
        """
        if self._flags.ENABLE_PREFLIGHT_GROUNDING:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def route_coverage(self) -> PipelineRoute:
        """Route the coverage validation stage.

        Returns NEW_PIPELINE if ENABLE_DETERMINISTIC_COVERAGE is enabled,
        otherwise LEGACY_PIPELINE for legacy coverage checking.
        """
        if self._flags.ENABLE_DETERMINISTIC_COVERAGE:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def route_repair(self) -> PipelineRoute:
        """Route the semantic repair stage.

        Returns NEW_PIPELINE if ENABLE_BOUNDED_REPAIR is enabled,
        otherwise LEGACY_PIPELINE for legacy repair behavior.
        """
        if self._flags.ENABLE_BOUNDED_REPAIR:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def route_clarification(self) -> PipelineRoute:
        """Route the clarification stage.

        Returns NEW_PIPELINE if ENABLE_CLARIFICATION_AS_DRAFT_PATCHING is enabled,
        otherwise LEGACY_PIPELINE for legacy clarification behavior.
        """
        if self._flags.ENABLE_CLARIFICATION_AS_DRAFT_PATCHING:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def route_schema_caching(self) -> PipelineRoute:
        """Route the schema caching stage.

        Returns NEW_PIPELINE if ENABLE_SCHEMA_CACHING is enabled,
        otherwise LEGACY_PIPELINE for uncached schema inference.
        """
        if self._flags.ENABLE_SCHEMA_CACHING:
            return PipelineRoute.NEW_PIPELINE
        return PipelineRoute.LEGACY_PIPELINE

    def update_flags(self, new_flags: FeatureFlags) -> None:
        """Update the feature flags, logging transitions for any changed flags.

        For each flag that changed value, logs the transition with previous state,
        new state, and effective timestamp (Requirement 13.5).

        Args:
            new_flags: The new feature flag configuration to apply.
        """
        old_flags = self._flags

        # Check each flag field for transitions
        flag_fields = [
            "ENABLE_SEMANTIC_DRAFT_PIPELINE",
            "ENABLE_PREFLIGHT_GROUNDING",
            "DISABLE_EXECUTION_TIME_GROUNDING",
            "ENABLE_LLM_COVERAGE_SHADOW",
            "ENABLE_DETERMINISTIC_COVERAGE",
            "ENABLE_BOUNDED_REPAIR",
            "ENABLE_SCHEMA_CACHING",
            "ENABLE_CLARIFICATION_AS_DRAFT_PATCHING",
        ]

        for flag_name in flag_fields:
            old_value = getattr(old_flags, flag_name)
            new_value = getattr(new_flags, flag_name)
            if old_value != new_value:
                # Delegate logging to the FeatureFlags.log_transition method
                # which records previous state, new state, and timestamp
                new_flags.log_transition(flag_name, old_value, new_value)

        self._flags = new_flags
