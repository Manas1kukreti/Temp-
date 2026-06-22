"""Evidence models for the shared grounding layer.

Defines scoring, configuration, and result types used by both
Column Grounder and Predicate Grounder.

Key types:
- ScoredCandidate: A column candidate with deterministic scoring breakdown.
- GroundingConfig: Configuration knobs for grounding operations.
- GroundingMethod: Enum indicating how a reference was resolved.
- ColumnGroundingResult: Result of resolving a standalone column reference.
- PredicateGroundingResult: Result of resolving a complete filter predicate.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ScoredCandidate(BaseModel):
    """A column candidate with deterministic scoring breakdown.

    Each candidate exposes individual scoring dimensions and evidence
    so that observability, tie-breaking, and clarification can inspect
    *why* a candidate was scored as it was.
    """

    model_config = ConfigDict(strict=True)

    column_name: str
    total_score: float = Field(ge=0.0, le=1.0)
    token_overlap_score: float = Field(ge=0.0, le=1.0)
    value_concept_score: float = Field(ge=0.0, le=1.0)
    semantic_type_score: float = Field(ge=0.0, le=1.0)
    name_similarity_score: float = Field(ge=0.0, le=1.0)
    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)


class GroundingConfig(BaseModel):
    """Configuration for grounding operations.

    Controls confidence thresholds, ambiguity margins, and behavioral
    flags that govern when the pipeline resolves autonomously vs.
    routes to clarification or LLM fallback.
    """

    model_config = ConfigDict(strict=True)

    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    """Minimum score below which a grounding result is not trusted."""

    ambiguity_margin: float = Field(default=0.1, ge=0.0, le=1.0)
    """Threshold below which two candidate scores are considered too close to resolve."""

    llm_fallback_enabled: bool = True
    """Whether LLM fallback is allowed when deterministic scoring is insufficient."""

    destructive_action_extra_caution: bool = True
    """Whether destructive operations require clarification on close candidates."""


class GroundingMethod(str, Enum):
    """How a column/predicate reference was resolved."""

    DETERMINISTIC = "deterministic"
    """Resolved purely through deterministic scoring (above confidence threshold)."""

    LLM_FALLBACK = "llm_fallback"
    """Resolved via LLM fallback after deterministic scoring was insufficient."""

    CLARIFICATION = "clarification"
    """Resolved via user clarification."""


class ColumnGroundingResult(BaseModel):
    """Result of resolving a standalone column reference.

    Produced by Column Grounder for projections, sorts, drops, renames.
    """

    model_config = ConfigDict(strict=True)

    resolved_column: str | None = None
    """Physical column name, or None if unresolved."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the resolution (0.0 = no confidence, 1.0 = certain)."""

    method: GroundingMethod
    """How the resolution was achieved."""

    evidence: list[ScoredCandidate] = Field(default_factory=list)
    """Scored candidates considered during resolution."""


class PredicateGroundingResult(BaseModel):
    """Result of resolving a complete filter predicate (field + operator + value).

    Produced by Predicate Grounder. Owns filter-column resolution,
    operator mapping, and value normalization.
    """

    model_config = ConfigDict(strict=True)

    resolved_column: str | None = None
    """Physical column the predicate targets, or None if unresolved."""

    operator: str | None = None
    """Canonical operator (e.g. '==', '!=', '>', '<', 'in', 'not_in')."""

    value: str | None = None
    """Normalized predicate value, or None if unresolved."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Confidence in the overall predicate resolution."""

    evidence: list[ScoredCandidate] = Field(default_factory=list)
    """Scored candidates considered during column resolution."""
