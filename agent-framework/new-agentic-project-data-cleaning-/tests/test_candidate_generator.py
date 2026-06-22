"""Unit tests for grounding/candidate_generator.py.

Tests cover:
- CandidateGenerator.generate_candidates(): scoring, sorting, evidence
- ScoringWeights: configuration and normalization
- Determinism guarantee: identical inputs → identical scores
- Evidence collection: positive + negative evidence per candidate
"""

import pytest

from finflow_agent.grounding.candidate_generator import (
    CandidateGenerator,
    ScoringWeights,
)
from finflow_agent.grounding.evidence import ScoredCandidate
from finflow_agent.grounding.schema_service import (
    ColumnRole,
    ColumnSemanticType,
    SchemaInferenceResult,
)
from finflow_agent.models.draft import ReferenceKind, SemanticColumnReference
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.tools.dataframe_profile import ColumnProfile, DataFrameProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_provenance(text: str) -> list[PromptSpanProvenance]:
    return [PromptSpanProvenance(start_offset=0, end_offset=len(text), source_text=text)]


def _make_reference(text: str, kind: ReferenceKind = ReferenceKind.SEMANTIC_CONCEPT) -> SemanticColumnReference:
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=kind,
        provenance=_make_provenance(text),
    )


def _make_schema(*columns: tuple[str, ColumnRole, float]) -> SchemaInferenceResult:
    return SchemaInferenceResult(
        columns=[
            ColumnSemanticType(
                column_name=name,
                inferred_role=role,
                confidence=conf,
                evidence=[f"role={role.value}"],
            )
            for name, role, conf in columns
        ]
    )


def _make_profile(*columns: tuple[str, str, list]) -> DataFrameProfile:
    """Create a profile with specified columns.

    Args:
        columns: tuples of (column_name, dtype, frequent_values)
    """
    col_profiles = []
    for col_name, dtype, freq_values in columns:
        col_profiles.append(
            ColumnProfile(
                column=col_name,
                normalized_name=col_name.lower(),
                pandas_dtype=dtype,
                null_count=0,
                non_null_count=100,
                distinct_count=len(freq_values) if freq_values else 10,
                frequent_values=freq_values,
                representative_values=freq_values[:2] if freq_values else [],
                random_distinct_values=[],
                semantic_guess="categorical" if freq_values else "unknown",
                confidence=0.8,
            )
        )
    return DataFrameProfile(
        row_count=100,
        column_count=len(columns),
        columns=col_profiles,
        duplicate_row_count=0,
    )


# ---------------------------------------------------------------------------
# CandidateGenerator Tests
# ---------------------------------------------------------------------------


class TestCandidateGenerator:
    """Tests for the CandidateGenerator class."""

    def test_returns_scored_candidates_for_all_columns(self) -> None:
        """Should return one ScoredCandidate per column in schema."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("amount", ColumnRole.MEASURE, 0.85),
        )
        profile = _make_profile(
            ("payment_method", "object", ["paypal", "cash"]),
            ("amount", "float64", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        assert len(candidates) == 2
        assert all(isinstance(c, ScoredCandidate) for c in candidates)

    def test_sorted_by_total_score_descending(self) -> None:
        """Candidates should be sorted by total_score in descending order."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("transaction_id", ColumnRole.IDENTIFIER, 0.95),
            ("amount", ColumnRole.MEASURE, 0.85),
        )
        profile = _make_profile(
            ("payment_method", "object", ["paypal", "cash"]),
            ("transaction_id", "int64", []),
            ("amount", "float64", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        scores = [c.total_score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_payment_method(self) -> None:
        """'payment method' reference should score highest on 'payment_method' column."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("transaction_id", ColumnRole.IDENTIFIER, 0.95),
            ("amount", ColumnRole.MEASURE, 0.85),
        )
        profile = _make_profile(
            ("payment_method", "object", ["paypal", "cash"]),
            ("transaction_id", "int64", []),
            ("amount", "float64", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        assert candidates[0].column_name == "payment_method"
        assert candidates[0].total_score > candidates[1].total_score

    def test_determinism_identical_inputs_identical_scores(self) -> None:
        """Req 7.3: identical inputs must produce identical scores."""
        gen = CandidateGenerator()
        ref = _make_reference("total amount")
        schema = _make_schema(
            ("amount", ColumnRole.MEASURE, 0.85),
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("status", ColumnRole.BOOLEAN, 0.9),
        )
        profile = _make_profile(
            ("amount", "float64", ["100.0", "250.5"]),
            ("payment_method", "object", ["paypal", "cash"]),
            ("status", "bool", ["true", "false"]),
        )

        candidates1 = gen.generate_candidates(ref, schema, profile)
        candidates2 = gen.generate_candidates(ref, schema, profile)

        assert len(candidates1) == len(candidates2)
        for c1, c2 in zip(candidates1, candidates2):
            assert c1.column_name == c2.column_name
            assert c1.total_score == c2.total_score
            assert c1.token_overlap_score == c2.token_overlap_score
            assert c1.value_concept_score == c2.value_concept_score
            assert c1.semantic_type_score == c2.semantic_type_score
            assert c1.name_similarity_score == c2.name_similarity_score

    def test_scores_in_valid_range(self) -> None:
        """All scores should be in [0.0, 1.0]."""
        gen = CandidateGenerator()
        ref = _make_reference("some reference")
        schema = _make_schema(
            ("col_a", ColumnRole.TEXT, 0.6),
            ("col_b", ColumnRole.UNKNOWN, 0.3),
        )
        profile = _make_profile(
            ("col_a", "object", ["hello"]),
            ("col_b", "object", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        for c in candidates:
            assert 0.0 <= c.total_score <= 1.0
            assert 0.0 <= c.token_overlap_score <= 1.0
            assert 0.0 <= c.value_concept_score <= 1.0
            assert 0.0 <= c.semantic_type_score <= 1.0
            assert 0.0 <= c.name_similarity_score <= 1.0

    def test_evidence_is_populated(self) -> None:
        """Req 7.4: each candidate should have positive or negative evidence."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("amount", ColumnRole.MEASURE, 0.85),
        )
        profile = _make_profile(
            ("payment_method", "object", ["paypal", "cash"]),
            ("amount", "float64", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        for c in candidates:
            # Every candidate should have at least some evidence
            assert len(c.positive_evidence) + len(c.negative_evidence) > 0

    def test_positive_evidence_for_strong_match(self) -> None:
        """Strong matching candidate should have positive evidence."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(("payment_method", ColumnRole.CATEGORICAL, 0.9))
        profile = _make_profile(("payment_method", "object", ["paypal"]))

        candidates = gen.generate_candidates(ref, schema, profile)

        assert len(candidates) == 1
        # Token overlap is 1.0, name similarity is high → positive evidence
        assert len(candidates[0].positive_evidence) > 0

    def test_negative_evidence_for_weak_match(self) -> None:
        """Weakly matching candidate should have negative evidence."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = _make_schema(("xyz_unrelated", ColumnRole.IDENTIFIER, 0.5))
        profile = _make_profile(("xyz_unrelated", "int64", []))

        candidates = gen.generate_candidates(ref, schema, profile)

        assert len(candidates) == 1
        assert len(candidates[0].negative_evidence) > 0

    def test_empty_schema_returns_empty_list(self) -> None:
        """No columns in schema → no candidates."""
        gen = CandidateGenerator()
        ref = _make_reference("payment method")
        schema = SchemaInferenceResult(columns=[])
        profile = DataFrameProfile(
            row_count=0, column_count=0, columns=[], duplicate_row_count=0
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        assert candidates == []

    def test_value_concept_match_boosts_score(self) -> None:
        """Value-implied reference matching column values should boost score."""
        gen = CandidateGenerator()
        ref = _make_reference("paypal", kind=ReferenceKind.VALUE_IMPLIED)
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("status", ColumnRole.BOOLEAN, 0.9),
        )
        profile = _make_profile(
            ("payment_method", "object", ["paypal", "cash", "credit"]),
            ("status", "bool", ["active", "inactive"]),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        # payment_method should rank higher due to value-concept match
        pm_candidate = next(c for c in candidates if c.column_name == "payment_method")
        status_candidate = next(c for c in candidates if c.column_name == "status")
        assert pm_candidate.value_concept_score > status_candidate.value_concept_score


# ---------------------------------------------------------------------------
# ScoringWeights Tests
# ---------------------------------------------------------------------------


class TestScoringWeights:
    """Tests for the ScoringWeights configuration."""

    def test_default_equal_weights(self) -> None:
        """Default weights should be 0.25 each."""
        w = ScoringWeights()
        assert w.token_overlap == 0.25
        assert w.value_concept == 0.25
        assert w.semantic_type == 0.25
        assert w.name_similarity == 0.25
        assert w.total == 1.0

    def test_custom_weights(self) -> None:
        """Custom weights should be respected."""
        w = ScoringWeights(token_overlap=0.4, value_concept=0.3, semantic_type=0.2, name_similarity=0.1)
        assert w.token_overlap == 0.4
        assert w.total == pytest.approx(1.0)

    def test_custom_weights_affect_scoring(self) -> None:
        """Custom weights should change candidate ranking."""
        ref = _make_reference("payment method")
        schema = _make_schema(
            ("payment_method", ColumnRole.CATEGORICAL, 0.9),
            ("amount", ColumnRole.MEASURE, 0.85),
        )
        profile = _make_profile(
            ("payment_method", "object", []),
            ("amount", "float64", []),
        )

        # Default weights
        gen_default = CandidateGenerator()
        cands_default = gen_default.generate_candidates(ref, schema, profile)

        # Name-heavy weights
        gen_name = CandidateGenerator(weights=ScoringWeights(
            token_overlap=0.0, value_concept=0.0, semantic_type=0.0, name_similarity=1.0
        ))
        cands_name = gen_name.generate_candidates(ref, schema, profile)

        # payment_method should still be the top with name-heavy weights
        assert cands_name[0].column_name == "payment_method"
        # But scores should differ from defaults
        assert cands_default[0].total_score != cands_name[0].total_score

    def test_zero_weights_produce_zero_score(self) -> None:
        """All-zero weights should produce zero total score."""
        gen = CandidateGenerator(weights=ScoringWeights(
            token_overlap=0.0, value_concept=0.0, semantic_type=0.0, name_similarity=0.0
        ))
        ref = _make_reference("test")
        schema = _make_schema(("col", ColumnRole.TEXT, 0.5))
        profile = _make_profile(("col", "object", []))

        candidates = gen.generate_candidates(ref, schema, profile)

        assert len(candidates) == 1
        assert candidates[0].total_score == 0.0


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestCandidateGeneratorEdgeCases:
    """Edge case tests for CandidateGenerator."""

    def test_explicit_name_reference_exact_match(self) -> None:
        """Explicit name reference that exactly matches a column name."""
        gen = CandidateGenerator()
        ref = _make_reference("amount", kind=ReferenceKind.EXPLICIT_NAME)
        schema = _make_schema(
            ("amount", ColumnRole.MEASURE, 0.9),
            ("payment_method", ColumnRole.CATEGORICAL, 0.8),
        )
        profile = _make_profile(
            ("amount", "float64", []),
            ("payment_method", "object", []),
        )

        candidates = gen.generate_candidates(ref, schema, profile)

        assert candidates[0].column_name == "amount"
        assert candidates[0].name_similarity_score == 1.0

    def test_generic_reference_low_type_alignment(self) -> None:
        """Generic references should get lower semantic-type alignment scores."""
        gen = CandidateGenerator()
        ref = _make_reference("field", kind=ReferenceKind.GENERIC_REFERENCE)
        schema = _make_schema(("amount", ColumnRole.MEASURE, 0.9))
        profile = _make_profile(("amount", "float64", []))

        candidates = gen.generate_candidates(ref, schema, profile)

        # Generic reference → low semantic-type alignment (0.3 baseline)
        assert candidates[0].semantic_type_score <= 0.4

    def test_column_not_in_profile_gets_zero_value_score(self) -> None:
        """Column missing from profile should get 0 value-concept score."""
        gen = CandidateGenerator()
        ref = _make_reference("amount")
        schema = _make_schema(("amount", ColumnRole.MEASURE, 0.9))
        # Profile has no 'amount' column
        profile = _make_profile(("other_col", "object", ["x"]))

        candidates = gen.generate_candidates(ref, schema, profile)

        assert candidates[0].value_concept_score == 0.0
