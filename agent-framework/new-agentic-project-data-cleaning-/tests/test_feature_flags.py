"""Unit tests for the feature flag system.

Tests validate compatibility rules, startup validation, flag transition logging,
and mixed-mode prevention.
"""

import os
import sys
import logging
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.pipeline.feature_flags import (
    FeatureFlags,
    validate_feature_flags_at_startup,
)


class TestFeatureFlagsDefaults:
    """Test default flag values."""

    def test_default_flags_are_valid(self):
        """Default flag combination should pass compatibility validation."""
        flags = FeatureFlags()
        errors = flags.validate_compatibility()
        assert errors == []

    def test_default_values(self):
        """Verify each default matches the design specification."""
        flags = FeatureFlags()
        assert flags.ENABLE_SEMANTIC_DRAFT_PIPELINE is True
        assert flags.ENABLE_PREFLIGHT_GROUNDING is True
        assert flags.DISABLE_EXECUTION_TIME_GROUNDING is True
        assert flags.ENABLE_LLM_COVERAGE_SHADOW is True
        assert flags.ENABLE_DETERMINISTIC_COVERAGE is True
        assert flags.ENABLE_BOUNDED_REPAIR is True
        assert flags.ENABLE_SCHEMA_CACHING is True
        assert flags.ENABLE_CLARIFICATION_AS_DRAFT_PATCHING is True


class TestCompatibilityRules:
    """Test compatibility rule validation."""

    def test_preflight_without_disabling_execution_time_grounding(self):
        """Req 19.2: ENABLE_PREFLIGHT_GROUNDING requires DISABLE_EXECUTION_TIME_GROUNDING."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=False,
        )
        errors = flags.validate_compatibility()
        assert len(errors) >= 1
        assert any("Preflight grounding requires" in e for e in errors)

    def test_preflight_with_execution_time_grounding_disabled_is_valid(self):
        """Valid: preflight on + exec-time grounding off."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []

    def test_clarification_patching_without_draft_pipeline(self):
        """Req 19.3: ENABLE_CLARIFICATION_AS_DRAFT_PATCHING requires ENABLE_SEMANTIC_DRAFT_PIPELINE."""
        flags = FeatureFlags(
            ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=True,
            ENABLE_SEMANTIC_DRAFT_PIPELINE=False,
        )
        errors = flags.validate_compatibility()
        assert len(errors) >= 1
        assert any("Draft patching requires" in e for e in errors)

    def test_clarification_patching_with_draft_pipeline_is_valid(self):
        """Valid: draft patching on + draft pipeline on."""
        flags = FeatureFlags(
            ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=True,
            ENABLE_SEMANTIC_DRAFT_PIPELINE=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []

    def test_llm_shadow_without_deterministic_coverage(self):
        """Req 19.4: ENABLE_LLM_COVERAGE_SHADOW requires ENABLE_DETERMINISTIC_COVERAGE."""
        flags = FeatureFlags(
            ENABLE_LLM_COVERAGE_SHADOW=True,
            ENABLE_DETERMINISTIC_COVERAGE=False,
        )
        errors = flags.validate_compatibility()
        assert len(errors) >= 1
        assert any("LLM shadow requires" in e for e in errors)

    def test_llm_shadow_with_deterministic_coverage_is_valid(self):
        """Valid: LLM shadow on + deterministic coverage on."""
        flags = FeatureFlags(
            ENABLE_LLM_COVERAGE_SHADOW=True,
            ENABLE_DETERMINISTIC_COVERAGE=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []

    def test_multiple_violations_reported(self):
        """All violations should be reported, not just the first."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=False,
            ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=True,
            ENABLE_SEMANTIC_DRAFT_PIPELINE=False,
            ENABLE_LLM_COVERAGE_SHADOW=True,
            ENABLE_DETERMINISTIC_COVERAGE=False,
        )
        errors = flags.validate_compatibility()
        assert len(errors) >= 3


class TestMixedModePrevention:
    """Test mixed-mode prevention (Req 13.6)."""

    def test_mixed_mode_detected(self):
        """Both preflight grounding active and legacy exec-time grounding active is invalid."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=False,
        )
        errors = flags.validate_compatibility()
        # Should have at least one error related to this conflict
        assert len(errors) >= 1

    def test_no_mixed_mode_when_exec_time_disabled(self):
        """No mixed mode when execution-time grounding is properly disabled."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []

    def test_no_mixed_mode_when_preflight_disabled(self):
        """No mixed mode when preflight grounding is off (legacy path only)."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=False,
            DISABLE_EXECUTION_TIME_GROUNDING=False,
        )
        errors = flags.validate_compatibility()
        assert errors == []


class TestStartupValidation:
    """Test startup validation behavior."""

    def test_valid_flags_do_not_exit(self):
        """Valid flags should not cause sys.exit."""
        flags = FeatureFlags()
        # Should not raise or exit
        validate_feature_flags_at_startup(flags)

    def test_invalid_flags_cause_exit(self):
        """Invalid flag combinations should cause sys.exit(1)."""
        flags = FeatureFlags(
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            validate_feature_flags_at_startup(flags)
        assert exc_info.value.code == 1

    def test_startup_logs_errors(self, caplog):
        """Startup should log each error and the invalid state."""
        flags = FeatureFlags(
            ENABLE_LLM_COVERAGE_SHADOW=True,
            ENABLE_DETERMINISTIC_COVERAGE=False,
        )
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                validate_feature_flags_at_startup(flags)

        assert any("compatibility error" in r.message for r in caplog.records)
        assert any("Invalid feature flag state" in r.message for r in caplog.records)


class TestFlagTransitionLogging:
    """Test flag transition logging (Req 13.5)."""

    def test_log_transition_logs_change(self, caplog):
        """Transition logging should record flag name, old value, new value."""
        flags = FeatureFlags()
        with caplog.at_level(logging.INFO):
            flags.log_transition("ENABLE_SCHEMA_CACHING", False, True)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "ENABLE_SCHEMA_CACHING" in record.message
        assert "False" in record.message
        assert "True" in record.message

    def test_log_transition_includes_timestamp(self, caplog):
        """Transition log should include a timestamp."""
        flags = FeatureFlags()
        with caplog.at_level(logging.INFO):
            flags.log_transition("ENABLE_BOUNDED_REPAIR", False, True)

        record = caplog.records[0]
        # ISO format timestamp includes 'T' separator and timezone info
        assert "T" in record.message


class TestStrictMode:
    """Test that strict mode prevents silent coercion."""

    def test_rejects_string_for_bool(self):
        """Strict mode should reject non-bool values for flag fields."""
        with pytest.raises(Exception):
            FeatureFlags(ENABLE_PREFLIGHT_GROUNDING="yes")  # type: ignore

    def test_rejects_int_for_bool(self):
        """Strict mode should reject int values for flag fields."""
        with pytest.raises(Exception):
            FeatureFlags(ENABLE_PREFLIGHT_GROUNDING=1)  # type: ignore


class TestAllValidCombinations:
    """Smoke test some valid flag combinations."""

    def test_all_flags_enabled_with_proper_dependencies(self):
        """All flags enabled with correct dependencies should be valid."""
        flags = FeatureFlags(
            ENABLE_SEMANTIC_DRAFT_PIPELINE=True,
            ENABLE_PREFLIGHT_GROUNDING=True,
            DISABLE_EXECUTION_TIME_GROUNDING=True,
            ENABLE_LLM_COVERAGE_SHADOW=True,
            ENABLE_DETERMINISTIC_COVERAGE=True,
            ENABLE_BOUNDED_REPAIR=True,
            ENABLE_SCHEMA_CACHING=True,
            ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []

    def test_only_deterministic_coverage_enabled(self):
        """Only deterministic coverage (the default True flag) should be valid."""
        flags = FeatureFlags(
            ENABLE_DETERMINISTIC_COVERAGE=True,
        )
        errors = flags.validate_compatibility()
        assert errors == []
