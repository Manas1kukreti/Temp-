"""Unit tests for the legacy feature-flag routing module.

Tests validate that:
- Routing correctly maps feature flags to NEW_PIPELINE or LEGACY_PIPELINE
- Backward compatibility: disabled flags route to legacy behavior (Req 13.2)
- Flag transitions are logged with previous state, new state, timestamp (Req 13.5)
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.pipeline.feature_flags import FeatureFlags
from finflow_agent.pipeline.legacy_router import LegacyRouter, PipelineRoute


class TestPipelineRouteEnum:
    """Test PipelineRoute enum values."""

    def test_new_pipeline_value(self):
        assert PipelineRoute.NEW_PIPELINE == "new_pipeline"

    def test_legacy_pipeline_value(self):
        assert PipelineRoute.LEGACY_PIPELINE == "legacy_pipeline"


class TestLegacyRouterDefaults:
    """Test routing with default (all-off except deterministic coverage) flags."""

    def setup_method(self):
        self.flags = FeatureFlags()
        self.router = LegacyRouter(self.flags)

    def test_extraction_defaults_to_legacy(self):
        """ENABLE_SEMANTIC_DRAFT_PIPELINE defaults False → legacy."""
        assert self.router.route_extraction() == PipelineRoute.LEGACY_PIPELINE

    def test_grounding_defaults_to_legacy(self):
        """ENABLE_PREFLIGHT_GROUNDING defaults False → legacy."""
        assert self.router.route_grounding() == PipelineRoute.LEGACY_PIPELINE

    def test_coverage_defaults_to_new(self):
        """ENABLE_DETERMINISTIC_COVERAGE defaults True → new."""
        assert self.router.route_coverage() == PipelineRoute.NEW_PIPELINE

    def test_repair_defaults_to_legacy(self):
        """ENABLE_BOUNDED_REPAIR defaults False → legacy."""
        assert self.router.route_repair() == PipelineRoute.LEGACY_PIPELINE

    def test_clarification_defaults_to_legacy(self):
        """ENABLE_CLARIFICATION_AS_DRAFT_PATCHING defaults False → legacy."""
        assert self.router.route_clarification() == PipelineRoute.LEGACY_PIPELINE

    def test_schema_caching_defaults_to_legacy(self):
        """ENABLE_SCHEMA_CACHING defaults False → legacy."""
        assert self.router.route_schema_caching() == PipelineRoute.LEGACY_PIPELINE


class TestLegacyRouterEnabledFlags:
    """Test routing when flags are enabled."""

    def test_extraction_routes_to_new_when_enabled(self):
        flags = FeatureFlags(ENABLE_SEMANTIC_DRAFT_PIPELINE=True)
        router = LegacyRouter(flags)
        assert router.route_extraction() == PipelineRoute.NEW_PIPELINE

    def test_grounding_routes_to_new_when_enabled(self):
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=True,
        )
        router = LegacyRouter(flags)
        assert router.route_grounding() == PipelineRoute.NEW_PIPELINE

    def test_coverage_routes_to_legacy_when_disabled(self):
        flags = FeatureFlags(ENABLE_DETERMINISTIC_COVERAGE=False)
        router = LegacyRouter(flags)
        assert router.route_coverage() == PipelineRoute.LEGACY_PIPELINE

    def test_repair_routes_to_new_when_enabled(self):
        flags = FeatureFlags(ENABLE_BOUNDED_REPAIR=True)
        router = LegacyRouter(flags)
        assert router.route_repair() == PipelineRoute.NEW_PIPELINE

    def test_clarification_routes_to_new_when_enabled(self):
        flags = FeatureFlags(
            ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=True,
            ENABLE_SEMANTIC_DRAFT_PIPELINE=True,
        )
        router = LegacyRouter(flags)
        assert router.route_clarification() == PipelineRoute.NEW_PIPELINE

    def test_schema_caching_routes_to_new_when_enabled(self):
        flags = FeatureFlags(ENABLE_SCHEMA_CACHING=True)
        router = LegacyRouter(flags)
        assert router.route_schema_caching() == PipelineRoute.NEW_PIPELINE


class TestLegacyRouterBackwardCompatibility:
    """Test backward compatibility: disabled flags always route to legacy (Req 13.2)."""

    def test_all_disabled_routes_all_legacy(self):
        """When all flags disabled, every stage uses legacy behavior."""
        flags = FeatureFlags(ENABLE_DETERMINISTIC_COVERAGE=False)
        router = LegacyRouter(flags)
        assert router.route_extraction() == PipelineRoute.LEGACY_PIPELINE
        assert router.route_grounding() == PipelineRoute.LEGACY_PIPELINE
        assert router.route_coverage() == PipelineRoute.LEGACY_PIPELINE
        assert router.route_repair() == PipelineRoute.LEGACY_PIPELINE
        assert router.route_clarification() == PipelineRoute.LEGACY_PIPELINE
        assert router.route_schema_caching() == PipelineRoute.LEGACY_PIPELINE


class TestUpdateFlags:
    """Test flag update with transition logging (Req 13.5)."""

    def test_update_flags_changes_routing(self):
        """Updating flags should change subsequent routing decisions."""
        flags = FeatureFlags()
        router = LegacyRouter(flags)
        assert router.route_extraction() == PipelineRoute.LEGACY_PIPELINE

        new_flags = FeatureFlags(ENABLE_SEMANTIC_DRAFT_PIPELINE=True)
        router.update_flags(new_flags)
        assert router.route_extraction() == PipelineRoute.NEW_PIPELINE

    def test_update_flags_logs_transitions(self, caplog):
        """Flag changes should be logged with flag name, old/new value."""
        flags = FeatureFlags()
        router = LegacyRouter(flags)

        new_flags = FeatureFlags(
            ENABLE_SEMANTIC_DRAFT_PIPELINE=True,
            ENABLE_SCHEMA_CACHING=True,
        )
        with caplog.at_level(logging.INFO):
            router.update_flags(new_flags)

        # Should log exactly two transitions
        transition_records = [
            r for r in caplog.records if "Feature flag transition" in r.message
        ]
        assert len(transition_records) == 2

        # Check content of logged messages
        messages = [r.message for r in transition_records]
        assert any("ENABLE_SEMANTIC_DRAFT_PIPELINE" in m for m in messages)
        assert any("ENABLE_SCHEMA_CACHING" in m for m in messages)
        assert all("False" in m and "True" in m for m in messages)

    def test_update_flags_logs_timestamp(self, caplog):
        """Transition log should include a timestamp."""
        flags = FeatureFlags()
        router = LegacyRouter(flags)

        new_flags = FeatureFlags(ENABLE_BOUNDED_REPAIR=True)
        with caplog.at_level(logging.INFO):
            router.update_flags(new_flags)

        record = caplog.records[0]
        # ISO format timestamp includes 'T' separator
        assert "T" in record.message

    def test_update_flags_no_log_when_unchanged(self, caplog):
        """No transitions logged when flags don't change."""
        flags = FeatureFlags()
        router = LegacyRouter(flags)

        same_flags = FeatureFlags()
        with caplog.at_level(logging.INFO):
            router.update_flags(same_flags)

        transition_records = [
            r for r in caplog.records if "Feature flag transition" in r.message
        ]
        assert len(transition_records) == 0

    def test_update_flags_logs_disable_transition(self, caplog):
        """Disabling a flag should also log the transition."""
        flags = FeatureFlags(ENABLE_SCHEMA_CACHING=True)
        router = LegacyRouter(flags)

        new_flags = FeatureFlags(ENABLE_SCHEMA_CACHING=False)
        with caplog.at_level(logging.INFO):
            router.update_flags(new_flags)

        transition_records = [
            r for r in caplog.records if "Feature flag transition" in r.message
        ]
        assert len(transition_records) == 1
        assert "ENABLE_SCHEMA_CACHING" in transition_records[0].message
        assert "True" in transition_records[0].message
        assert "False" in transition_records[0].message


class TestFlagsProperty:
    """Test the flags property accessor."""

    def test_flags_property_returns_current_flags(self):
        flags = FeatureFlags(ENABLE_BOUNDED_REPAIR=True)
        router = LegacyRouter(flags)
        assert router.flags is flags

    def test_flags_property_reflects_update(self):
        flags = FeatureFlags()
        router = LegacyRouter(flags)
        new_flags = FeatureFlags(ENABLE_BOUNDED_REPAIR=True)
        router.update_flags(new_flags)
        assert router.flags is new_flags
