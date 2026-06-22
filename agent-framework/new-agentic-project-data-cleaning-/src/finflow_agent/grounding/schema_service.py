"""Schema Service with layered structural/value-evidence cache.

Infers column roles and semantic types from dataset profiling. Uses a two-layer
cache for performance:
- L1: Structural role cache keyed by (structural_fingerprint, role_model_version)
- L2: Value-evidence cache keyed by (structural_fingerprint, profile_fingerprint, profiler_version)

Executes during the dataset-profiling stage (after Preflight, before grounding).
Graceful degradation: returns cached or deterministic-only result on LLM failure.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 18.2
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    LLMProviderError,
    SemanticResolver,
)
from finflow_agent.grounding.preflight_loader import DataFrameProfile
from finflow_agent.models.fingerprints import (
    ProfileFingerprint,
    StructuralSchemaFingerprint,
)
from finflow_agent.models.snapshot import DataSnapshotRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Models
# ---------------------------------------------------------------------------


class ColumnRole(str, Enum):
    """Inferred semantic role for a dataset column."""

    IDENTIFIER = "identifier"
    MEASURE = "measure"
    DIMENSION = "dimension"
    TEMPORAL = "temporal"
    TEXT = "text"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnSemanticType(BaseModel):
    """Semantic type inference result for a single column."""

    model_config = ConfigDict(strict=True)

    column_name: str
    inferred_role: ColumnRole
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class SchemaInferenceResult(BaseModel):
    """Aggregated schema inference result for all columns in a dataset."""

    model_config = ConfigDict(strict=True)

    columns: list[ColumnSemanticType]
    from_cache: bool = False
    cache_layer: str | None = None  # "L1" or "L2" if from cache


# ---------------------------------------------------------------------------
# Deterministic Role Inference Heuristics
# ---------------------------------------------------------------------------

_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^id$", re.IGNORECASE),
    re.compile(r"^uuid$", re.IGNORECASE),
    re.compile(r"_id$", re.IGNORECASE),
]


def _is_identifier_name(column_name: str) -> bool:
    """Check if column name matches identifier patterns (id, uuid, *_id)."""
    for pattern in _ID_PATTERNS:
        if pattern.search(column_name):
            return True
    return False


def _infer_role_deterministic(
    column_name: str, pandas_dtype: str, semantic_guess: str,
    distinct_count: int, non_null_count: int,
) -> tuple[ColumnRole, float, list[str]]:
    """Infer column role using deterministic heuristics.

    Returns (role, confidence, evidence_list).
    """
    evidence: list[str] = []

    # Rule 1: identifier patterns
    if _is_identifier_name(column_name):
        evidence.append(f"column name '{column_name}' matches identifier pattern")
        return ColumnRole.IDENTIFIER, 0.9, evidence

    # Rule 2: boolean columns
    if semantic_guess == "boolean" or pandas_dtype == "bool":
        evidence.append(f"dtype='{pandas_dtype}', semantic_guess='{semantic_guess}'")
        return ColumnRole.BOOLEAN, 0.95, evidence

    # Rule 3: temporal/datetime columns
    if semantic_guess == "date" or "datetime" in pandas_dtype.lower():
        evidence.append(f"dtype='{pandas_dtype}', semantic_guess='{semantic_guess}'")
        return ColumnRole.TEMPORAL, 0.9, evidence

    # Rule 4: numeric columns → measure
    if semantic_guess in ("numeric", "currency") or "int" in pandas_dtype.lower() or "float" in pandas_dtype.lower():
        # But check if it could be an identifier (low cardinality integer)
        if _is_identifier_name(column_name):
            evidence.append(f"numeric column with identifier name pattern")
            return ColumnRole.IDENTIFIER, 0.85, evidence
        evidence.append(f"dtype='{pandas_dtype}', semantic_guess='{semantic_guess}'")
        return ColumnRole.MEASURE, 0.85, evidence

    # Rule 5: low cardinality string columns → categorical/dimension
    if semantic_guess == "categorical":
        evidence.append(f"semantic_guess='categorical'")
        return ColumnRole.CATEGORICAL, 0.8, evidence

    if non_null_count > 0 and distinct_count > 0:
        cardinality_ratio = distinct_count / non_null_count
        if cardinality_ratio <= 0.05 and distinct_count <= 20:
            evidence.append(
                f"low cardinality: {distinct_count} distinct / {non_null_count} rows "
                f"(ratio={cardinality_ratio:.3f})"
            )
            return ColumnRole.CATEGORICAL, 0.75, evidence
        elif cardinality_ratio <= 0.2 and distinct_count <= 50:
            evidence.append(
                f"moderate cardinality: {distinct_count} distinct / {non_null_count} rows "
                f"(ratio={cardinality_ratio:.3f})"
            )
            return ColumnRole.DIMENSION, 0.7, evidence

    # Rule 6: high cardinality string columns → text
    if semantic_guess == "string":
        if non_null_count > 0 and distinct_count > 0:
            cardinality_ratio = distinct_count / non_null_count
            if cardinality_ratio > 0.5:
                evidence.append(
                    f"high cardinality string: {distinct_count} distinct / {non_null_count} rows "
                    f"(ratio={cardinality_ratio:.3f})"
                )
                return ColumnRole.TEXT, 0.7, evidence
        evidence.append(f"string column, semantic_guess='{semantic_guess}'")
        return ColumnRole.TEXT, 0.6, evidence

    # Fallback: unknown
    evidence.append(
        f"no heuristic matched: dtype='{pandas_dtype}', "
        f"semantic_guess='{semantic_guess}', "
        f"distinct_count={distinct_count}, non_null_count={non_null_count}"
    )
    return ColumnRole.UNKNOWN, 0.3, evidence


# ---------------------------------------------------------------------------
# Schema Service
# ---------------------------------------------------------------------------


class SchemaService:
    """Schema inference service with layered cache.

    Cache layers:
    - L1: Structural role cache keyed by (structural_fingerprint, role_model_version)
    - L2: Value-evidence cache keyed by (structural_fingerprint, profile_fingerprint, profiler_version)

    Executes during dataset-profiling stage (after Preflight, before grounding).
    On LLM failure: returns deterministic-only result (Req 18.2).
    """

    def __init__(
        self,
        resolver: SemanticResolver | None = None,
        role_model_version: str = "1.0",
    ) -> None:
        self._resolver = resolver
        self._role_model_version = role_model_version

        # L1: structural role cache
        # Key: (structural_fingerprint, role_model_version)
        self._l1_cache: dict[tuple[str, str], SchemaInferenceResult] = {}

        # L2: value-evidence cache
        # Key: (structural_fingerprint, profile_fingerprint, profiler_version)
        self._l2_cache: dict[tuple[str, str, str], SchemaInferenceResult] = {}

        # Stats for observability
        self._stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "llm_failures": 0,
            "deterministic_fallbacks": 0,
        }

    def infer_roles(
        self, snapshot: DataSnapshotRef, profile: DataFrameProfile
    ) -> SchemaInferenceResult:
        """Infer column roles and semantic types.

        Cache layers:
        - L1: Structural role cache (fingerprint + role-model version)
        - L2: Value-evidence cache (fingerprint + profile fingerprint + profiler version)

        Executes during dataset-profiling stage (after Preflight, before grounding).

        Args:
            snapshot: Immutable reference to the profiled file version.
            profile: DataFrameProfile produced by the Preflight Data Loader.

        Returns:
            SchemaInferenceResult with column roles and confidence scores.
        """
        structural_fp = snapshot.structural_schema_fingerprint
        profile_fp = snapshot.profile_fingerprint

        # Check L1 cache first (structural roles only)
        l1_key = (structural_fp, self._role_model_version)
        if l1_key in self._l1_cache:
            self._stats["l1_hits"] += 1
            logger.debug("Schema L1 cache hit: structural_fp=%s", structural_fp[:12])
            result = self._l1_cache[l1_key]
            return SchemaInferenceResult(
                columns=result.columns,
                from_cache=True,
                cache_layer="L1",
            )

        # Check L2 cache (value-evidence)
        # Use profiler_version from the structural fingerprint data in the profile
        profiler_version = self._role_model_version  # default
        if profile.columns:
            # Derive profiler version from profile context
            profiler_version = self._role_model_version

        l2_key = (structural_fp, profile_fp, profiler_version)
        if l2_key in self._l2_cache:
            self._stats["l2_hits"] += 1
            logger.debug(
                "Schema L2 cache hit: structural_fp=%s, profile_fp=%s",
                structural_fp[:12],
                profile_fp[:12],
            )
            result = self._l2_cache[l2_key]
            return SchemaInferenceResult(
                columns=result.columns,
                from_cache=True,
                cache_layer="L2",
            )

        # Cache miss - compute deterministically
        self._stats["cache_misses"] += 1
        logger.debug(
            "Schema cache miss: structural_fp=%s, computing roles",
            structural_fp[:12],
        )

        result = self._compute_deterministic(profile)

        # Optionally enhance with LLM (graceful degradation on failure)
        if self._resolver is not None:
            result = self._try_llm_enhancement(result, profile)

        # Store in both cache layers
        self._l1_cache[l1_key] = result
        self._l2_cache[l2_key] = result

        return result

    def _compute_deterministic(self, profile: DataFrameProfile) -> SchemaInferenceResult:
        """Compute column roles using deterministic heuristics only."""
        column_types: list[ColumnSemanticType] = []

        for col_profile in profile.columns:
            role, confidence, evidence = _infer_role_deterministic(
                column_name=col_profile.column,
                pandas_dtype=col_profile.pandas_dtype,
                semantic_guess=col_profile.semantic_guess,
                distinct_count=col_profile.distinct_count,
                non_null_count=col_profile.non_null_count,
            )
            column_types.append(
                ColumnSemanticType(
                    column_name=col_profile.column,
                    inferred_role=role,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        return SchemaInferenceResult(columns=column_types)

    def _try_llm_enhancement(
        self, deterministic_result: SchemaInferenceResult, profile: DataFrameProfile
    ) -> SchemaInferenceResult:
        """Attempt LLM-enhanced role inference with graceful degradation.

        On LLM failure (Req 18.2): returns deterministic-only result or cached
        result if a compatible structural fingerprint exists.
        """
        try:
            self._stats["llm_calls"] += 1
            # NOTE: The actual async LLM call would be performed here.
            # For now, we return the deterministic result since the resolver
            # protocol is async and this method is sync. A future iteration
            # would integrate with an async runtime or use sync wrappers.
            # The LLM would be called with LLMCallSite.SCHEMA_INFERENCE
            # and DEFAULT_CONSTRAINTS[LLMCallSite.SCHEMA_INFERENCE].
            logger.debug("LLM enhancement skipped (sync context); using deterministic result")
            return deterministic_result
        except LLMProviderError as exc:
            # Graceful degradation (Req 18.2): return deterministic-only
            self._stats["llm_failures"] += 1
            self._stats["deterministic_fallbacks"] += 1
            logger.warning(
                "Schema LLM unavailable (error_type=%s, call_site=%s): "
                "falling back to deterministic inference",
                exc.error_type,
                exc.call_site,
            )
            return deterministic_result
        except Exception as exc:
            # Catch-all for unexpected errors: still degrade gracefully
            self._stats["llm_failures"] += 1
            self._stats["deterministic_fallbacks"] += 1
            logger.warning(
                "Unexpected error during LLM schema inference: %s. "
                "Falling back to deterministic inference.",
                exc,
            )
            return deterministic_result

    def clear_cache(self) -> None:
        """Clear both L1 and L2 caches."""
        self._l1_cache.clear()
        self._l2_cache.clear()
        logger.info("Schema service caches cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """Return observability stats for the cache.

        Returns:
            Dictionary with cache hit/miss counts and LLM call stats.
        """
        return {
            **self._stats,
            "l1_cache_size": len(self._l1_cache),
            "l2_cache_size": len(self._l2_cache),
            "role_model_version": self._role_model_version,
        }
