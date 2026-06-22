"""Shared Candidate Generation Layer for the grounding pipeline.

Produces deterministically scored candidate columns for semantic references.
Used by both Column Grounder and Predicate Grounder to ensure consistent,
auditable scoring across all grounding operations.

Scoring dimensions (from grounding/scoring.py):
- Token overlap: Jaccard similarity of tokenized names
- Value-concept matching: reference text vs. actual column values
- Semantic-type alignment: reference kind vs. column role compatibility
- Column-name similarity: normalized Levenshtein distance

Guarantees:
- Identical inputs → identical scores (deterministic, no randomness)
- Exposes positive + negative evidence per candidate

Requirements: 7.1, 7.2, 7.3, 7.4
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from finflow_agent.grounding.evidence import ScoredCandidate
from finflow_agent.grounding.preflight_loader import DataFrameProfile
from finflow_agent.grounding.schema_service import (
    ColumnSemanticType,
    SchemaInferenceResult,
)
from finflow_agent.grounding.scoring import (
    compute_name_similarity,
    compute_semantic_type_alignment,
    compute_token_overlap,
    compute_value_concept_match,
)
from finflow_agent.models.draft import SemanticColumnReference


# ---------------------------------------------------------------------------
# Scoring Weights Configuration
# ---------------------------------------------------------------------------


class ScoringWeights(BaseModel):
    """Configurable weights for each scoring dimension.

    All weights are combined as a weighted average (normalized by sum).
    Default: equal weight across all four dimensions.
    """

    model_config = ConfigDict(strict=True)

    token_overlap: float = Field(default=0.25, ge=0.0, le=1.0)
    value_concept: float = Field(default=0.25, ge=0.0, le=1.0)
    semantic_type: float = Field(default=0.25, ge=0.0, le=1.0)
    name_similarity: float = Field(default=0.25, ge=0.0, le=1.0)

    @property
    def total(self) -> float:
        """Sum of all weights (used for normalization)."""
        return (
            self.token_overlap
            + self.value_concept
            + self.semantic_type
            + self.name_similarity
        )


# ---------------------------------------------------------------------------
# Candidate Generator
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """Shared Candidate Generation Layer for both Column Grounder and Predicate Grounder.

    Produces deterministically scored candidates for a semantic reference by
    evaluating all columns in the schema against four scoring dimensions.

    Guarantees (Req 7.3):
    - Identical inputs → identical scores (pure deterministic, no randomness)
    - Exposes positive + negative evidence per candidate (Req 7.4)

    Requirements: 7.1, 7.2, 7.3, 7.4
    """

    def __init__(self, weights: ScoringWeights | None = None) -> None:
        """Initialize with optional custom scoring weights.

        Args:
            weights: Custom scoring weights. Defaults to equal weights (0.25 each).
        """
        self._weights = weights or ScoringWeights()

    @property
    def weights(self) -> ScoringWeights:
        """Current scoring weights."""
        return self._weights

    def generate_candidates(
        self,
        reference: SemanticColumnReference,
        schema_result: SchemaInferenceResult,
        profile: DataFrameProfile,
    ) -> list[ScoredCandidate]:
        """Generate scored column candidates for a semantic reference.

        For each column in the schema, computes four scoring dimensions and
        combines them with configurable weights. Results are sorted by
        total_score descending.

        Scoring dimensions:
        - Token overlap: shared tokens between reference text and column name
        - Value-concept matching: reference values present in column data
        - Semantic-type alignment: reference kind vs. inferred column role
        - Column-name similarity: normalized edit distance

        Guarantees:
        - Identical inputs → identical scores (Req 7.3)
        - Exposes positive + negative evidence per candidate (Req 7.4)

        Args:
            reference: The semantic column reference to resolve.
            schema_result: Schema inference result with column roles.
            profile: DataFrameProfile with column statistics and values.

        Returns:
            List of ScoredCandidate sorted by total_score descending.
        """
        candidates: list[ScoredCandidate] = []

        for column_type in schema_result.columns:
            column_name = column_type.column_name

            # Compute individual scoring dimensions
            token_score = compute_token_overlap(
                reference.reference_text, column_name
            )

            # Gather column values for value-concept matching
            column_values = self._get_column_values(column_name, profile)
            value_score = compute_value_concept_match(
                reference.reference_text, column_values
            )

            type_score = compute_semantic_type_alignment(
                reference.reference_kind.value, column_type.inferred_role.value
            )

            name_score = compute_name_similarity(
                reference.reference_text, column_name
            )

            # Combine scores with weighted average
            total_score = self._compute_weighted_score(
                token_score, value_score, type_score, name_score
            )

            # Collect evidence
            positive_evidence, negative_evidence = self._collect_evidence(
                reference=reference,
                column_name=column_name,
                column_type=column_type,
                token_score=token_score,
                value_score=value_score,
                type_score=type_score,
                name_score=name_score,
            )

            candidates.append(
                ScoredCandidate(
                    column_name=column_name,
                    total_score=total_score,
                    token_overlap_score=token_score,
                    value_concept_score=value_score,
                    semantic_type_score=type_score,
                    name_similarity_score=name_score,
                    positive_evidence=positive_evidence,
                    negative_evidence=negative_evidence,
                )
            )

        # Sort by total_score descending (stable sort for determinism)
        candidates.sort(key=lambda c: c.total_score, reverse=True)

        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_column_values(column_name: str, profile: DataFrameProfile) -> list[str]:
        """Extract known values for a column from the DataFrameProfile.

        Gathers frequent_values, representative_values, and random_distinct_values
        as string representations for value-concept matching.
        """
        for col_profile in profile.columns:
            if col_profile.column == column_name:
                values: list[str] = []
                for val in col_profile.frequent_values:
                    values.append(str(val))
                for val in col_profile.representative_values:
                    str_val = str(val)
                    if str_val not in values:
                        values.append(str_val)
                for val in col_profile.random_distinct_values:
                    str_val = str(val)
                    if str_val not in values:
                        values.append(str_val)
                return values
        return []

    def _compute_weighted_score(
        self,
        token_score: float,
        value_score: float,
        type_score: float,
        name_score: float,
    ) -> float:
        """Compute weighted average of scoring dimensions.

        Normalizes by the sum of weights to allow non-unit-sum configurations.
        Clamps result to [0.0, 1.0].
        """
        weight_sum = self._weights.total
        if weight_sum == 0.0:
            return 0.0

        raw = (
            self._weights.token_overlap * token_score
            + self._weights.value_concept * value_score
            + self._weights.semantic_type * type_score
            + self._weights.name_similarity * name_score
        ) / weight_sum

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _collect_evidence(
        reference: SemanticColumnReference,
        column_name: str,
        column_type: ColumnSemanticType,
        token_score: float,
        value_score: float,
        type_score: float,
        name_score: float,
    ) -> tuple[list[str], list[str]]:
        """Collect positive and negative evidence for a candidate.

        Evidence provides human-readable explanations for observability (Req 7.4).

        Returns:
            Tuple of (positive_evidence, negative_evidence).
        """
        positive: list[str] = []
        negative: list[str] = []

        # Token overlap evidence
        if token_score > 0.5:
            positive.append(
                f"Strong token overlap ({token_score:.2f}): "
                f"reference '{reference.reference_text}' shares tokens with '{column_name}'"
            )
        elif token_score > 0.0:
            positive.append(
                f"Partial token overlap ({token_score:.2f}): "
                f"some shared tokens between '{reference.reference_text}' and '{column_name}'"
            )
        else:
            negative.append(
                f"No token overlap: '{reference.reference_text}' and '{column_name}' "
                f"share no tokens"
            )

        # Value-concept evidence
        if value_score > 0.5:
            positive.append(
                f"Value-concept match ({value_score:.2f}): "
                f"reference values found in column '{column_name}'"
            )
        elif value_score > 0.0:
            positive.append(
                f"Partial value match ({value_score:.2f}): "
                f"some reference tokens overlap with column values"
            )
        else:
            negative.append(
                f"No value-concept match: reference text not found in column values"
            )

        # Semantic-type evidence
        if type_score > 0.6:
            positive.append(
                f"Semantic-type aligned ({type_score:.2f}): "
                f"reference kind '{reference.reference_kind.value}' aligns with "
                f"column role '{column_type.inferred_role.value}'"
            )
        elif type_score <= 0.3:
            negative.append(
                f"Semantic-type misaligned ({type_score:.2f}): "
                f"reference kind '{reference.reference_kind.value}' does not align with "
                f"column role '{column_type.inferred_role.value}'"
            )

        # Name similarity evidence
        if name_score > 0.7:
            positive.append(
                f"High name similarity ({name_score:.2f}): "
                f"'{reference.reference_text}' is similar to '{column_name}'"
            )
        elif name_score > 0.3:
            positive.append(
                f"Moderate name similarity ({name_score:.2f}): "
                f"'{reference.reference_text}' partially matches '{column_name}'"
            )
        else:
            negative.append(
                f"Low name similarity ({name_score:.2f}): "
                f"'{reference.reference_text}' differs significantly from '{column_name}'"
            )

        return positive, negative


__all__ = [
    "CandidateGenerator",
    "ScoringWeights",
]
