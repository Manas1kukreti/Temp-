"""Feature flag system with compatibility matrix validation.

This module defines the pipeline feature flags and their compatibility constraints.
Invalid flag combinations are caught at startup, preventing mixed modes where both
legacy and new components could own the same decision.

Requirements: 13.1, 13.3, 13.5, 13.6, 19.1, 19.2, 19.3, 19.4, 19.5
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar

import logging
import sys

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class FeatureFlags(BaseModel):
    """Pipeline feature flags with compatibility validation.

    Each flag controls incremental rollout of the semantic grounding refactor.
    Compatibility rules ensure the system cannot enter a mixed mode where both
    legacy execution-time grounding and preflight grounding are active for the
    same reference type.
    """

    model_config = ConfigDict(strict=True)

    # --- Feature flags with defaults ---
    ENABLE_SEMANTIC_DRAFT_PIPELINE: bool = True
    ENABLE_PREFLIGHT_GROUNDING: bool = True
    DISABLE_EXECUTION_TIME_GROUNDING: bool = True
    ENABLE_LLM_COVERAGE_SHADOW: bool = True
    ENABLE_DETERMINISTIC_COVERAGE: bool = True
    ENABLE_BOUNDED_REPAIR: bool = True
    ENABLE_SCHEMA_CACHING: bool = True
    ENABLE_CLARIFICATION_AS_DRAFT_PATCHING: bool = True

    # --- Compatibility rules ---
    # Each rule is (if_flag_is_true, requires_flag_is_true, error_message)
    COMPATIBILITY_RULES: ClassVar[list[tuple[str, str, str]]] = [
        (
            "ENABLE_PREFLIGHT_GROUNDING",
            "DISABLE_EXECUTION_TIME_GROUNDING",
            "Preflight grounding requires execution-time grounding disabled"
            " (ENABLE_PREFLIGHT_GROUNDING=true requires"
            " DISABLE_EXECUTION_TIME_GROUNDING=true)",
        ),
        (
            "ENABLE_CLARIFICATION_AS_DRAFT_PATCHING",
            "ENABLE_SEMANTIC_DRAFT_PIPELINE",
            "Draft patching requires semantic draft pipeline"
            " (ENABLE_CLARIFICATION_AS_DRAFT_PATCHING=true requires"
            " ENABLE_SEMANTIC_DRAFT_PIPELINE=true)",
        ),
        (
            "ENABLE_LLM_COVERAGE_SHADOW",
            "ENABLE_DETERMINISTIC_COVERAGE",
            "LLM shadow requires deterministic coverage enabled"
            " (ENABLE_LLM_COVERAGE_SHADOW=true requires"
            " ENABLE_DETERMINISTIC_COVERAGE=true)",
        ),
    ]

    # --- Mixed-mode prevention (Req 13.6) ---
    # Preflight grounding and legacy execution-time grounding cannot both be active.
    # This is enforced via the first compatibility rule: enabling preflight grounding
    # requires disabling execution-time grounding. An additional explicit check is
    # included in validate_compatibility for clarity.

    def validate_compatibility(self) -> list[str]:
        """Validate feature flag combinations and return a list of errors.

        An empty list means the flag combination is valid.
        """
        errors: list[str] = []

        # Check compatibility rules
        for if_flag, requires_flag, msg in self.COMPATIBILITY_RULES:
            if getattr(self, if_flag) and not getattr(self, requires_flag):
                errors.append(msg)

        # Mixed-mode prevention (Req 13.6): both legacy execution-time grounding
        # AND preflight grounding cannot be active for the same reference type.
        # Legacy execution-time grounding is active when DISABLE_EXECUTION_TIME_GROUNDING
        # is False. Preflight grounding is active when ENABLE_PREFLIGHT_GROUNDING is True.
        if (
            self.ENABLE_PREFLIGHT_GROUNDING
            and not self.DISABLE_EXECUTION_TIME_GROUNDING
        ):
            # This overlaps with the first compatibility rule but provides
            # a distinct error message focused on the mixed-mode concern.
            mixed_mode_msg = (
                "Mixed mode not permitted: both legacy execution-time grounding"
                " and preflight grounding cannot be active simultaneously"
            )
            if mixed_mode_msg not in errors and not any(
                "Preflight grounding requires" in e for e in errors
            ):
                errors.append(mixed_mode_msg)

        return errors

    def log_transition(
        self, flag_name: str, old_value: bool, new_value: bool
    ) -> None:
        """Log a feature flag transition with previous state, new state, and timestamp.

        Requirement 13.5: When a feature flag is toggled, the system logs the
        transition with previous state, new state, and effective timestamp.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Feature flag transition: %s changed from %s to %s at %s",
            flag_name,
            old_value,
            new_value,
            timestamp,
        )


def validate_feature_flags_at_startup(flags: FeatureFlags) -> None:
    """Validate feature flags at application startup.

    If invalid flag combinations are detected, logs the conflicting flags
    and expected valid state, then terminates with a non-zero exit code.

    Requirements: 13.3, 19.5
    """
    errors = flags.validate_compatibility()
    if errors:
        for err in errors:
            logger.error("Feature flag compatibility error: %s", err)
        logger.error(
            "Invalid feature flag state: %s. "
            "Fix the incompatible flag combination and restart.",
            flags.model_dump(),
        )
        sys.exit(1)
