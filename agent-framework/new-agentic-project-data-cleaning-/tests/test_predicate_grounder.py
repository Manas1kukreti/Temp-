"""Unit tests for PredicateGrounder.

Tests deterministic resolution, operator mapping, value normalization,
and LLM fallback with post-LLM verification.
"""

import asyncio

import pytest

from finflow_agent.grounding.evidence import (
    GroundingConfig,
    PredicateGroundingResult,
    ScoredCandidate,
)
from finflow_agent.grounding.predicate_grounder import (
    PredicateGrounder,
    _map_operator,
    _normalize_value,
)
from finflow_agent.models.draft import (
    ReferenceKind,
    SemanticColumnReference,
    UnresolvedPredicate,
)
from finflow_agent.models.provenance import PromptSpanProvenance


def _make_provenance():
    return PromptSpanProvenance(start_offset=0, end_offset=5, source_text="price")


def _make_predicate(ref_text="price", operator="gt", value=100):
    prov = _make_provenance()
    return UnresolvedPredicate(
        field_ref=SemanticColumnReference(
            reference_text=ref_text,
            reference_kind=ReferenceKind.EXPLICIT_NAME,
            provenance=[prov],
        ),
        operator=operator,
        value=value,
        provenance=[prov],
    )


def _make_candidate(name="unit_price", score=0.9, dtype="float64"):
    evidence = [f"dtype:{dtype}"] if dtype else []
    return ScoredCandidate(
        column_name=name,
        total_score=score,
        token_overlap_score=score * 0.9,
        value_concept_score=score * 0.8,
        semantic_type_score=score * 0.85,
        name_similarity_score=score * 0.95,
        positive_evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Operator mapping tests
# ---------------------------------------------------------------------------


class TestMapOperator:
    def test_equality_aliases(self):
        assert _map_operator("eq") == "=="
        assert _map_operator("equals") == "=="
        assert _map_operator("equal") == "=="
        assert _map_operator("==") == "=="
        assert _map_operator("=") == "=="

    def test_inequality_aliases(self):
        assert _map_operator("ne") == "!="
        assert _map_operator("neq") == "!="
        assert _map_operator("!=") == "!="
        assert _map_operator("<>") == "!="

    def test_comparison_aliases(self):
        assert _map_operator("gt") == ">"
        assert _map_operator("gte") == ">="
        assert _map_operator("lt") == "<"
        assert _map_operator("lte") == "<="
        assert _map_operator("ge") == ">="
        assert _map_operator("le") == "<="

    def test_set_operators(self):
        assert _map_operator("in") == "in"
        assert _map_operator("not_in") == "not_in"
        assert _map_operator("nin") == "not_in"

    def test_null_operators(self):
        assert _map_operator("is_null") == "is_null"
        assert _map_operator("isnull") == "is_null"
        assert _map_operator("not_null") == "not_null"
        assert _map_operator("notnull") == "not_null"

    def test_case_insensitive(self):
        assert _map_operator("EQUALS") == "=="
        assert _map_operator("GT") == ">"
        assert _map_operator("In") == "in"

    def test_unknown_passthrough(self):
        assert _map_operator("unknown_op") == "unknown_op"
        assert _map_operator("CUSTOM") == "custom"

    def test_whitespace_trimming(self):
        assert _map_operator("  eq  ") == "=="
        assert _map_operator(" gt ") == ">"


# ---------------------------------------------------------------------------
# Value normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeValue:
    def test_none_returns_none(self):
        assert _normalize_value(None, "int64") is None
        assert _normalize_value(None, None) is None

    def test_numeric_int(self):
        assert _normalize_value(42, "int64") == "42"
        assert _normalize_value(0, "float64") == "0"

    def test_numeric_float(self):
        assert _normalize_value(3.14, "float64") == "3.14"

    def test_numeric_string(self):
        assert _normalize_value("42", "int64") == "42"
        assert _normalize_value("3.14", "float64") == "3.14"

    def test_non_numeric_string_for_numeric_dtype(self):
        # Non-numeric string stays as-is when dtype is numeric
        assert _normalize_value("abc", "int64") == "abc"

    def test_string_dtype_trimmed(self):
        assert _normalize_value("  hello  ", "object") == "hello"
        assert _normalize_value("  hello  ", "string") == "hello"

    def test_list_values(self):
        result = _normalize_value([1, 2, 3], "int64")
        assert result == "1,2,3"

    def test_list_string_values(self):
        result = _normalize_value(["a", "b", "c"], "object")
        assert result == "a,b,c"

    def test_no_dtype(self):
        # Without dtype, values are just stringified
        assert _normalize_value(42, None) == "42"
        assert _normalize_value("hello", None) == "hello"


# ---------------------------------------------------------------------------
# Deterministic resolution tests
# ---------------------------------------------------------------------------


class TestDeterministicResolution:
    @pytest.mark.asyncio
    async def test_resolves_above_threshold_with_clear_margin(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        candidate = _make_candidate("unit_price", 0.9)
        config = GroundingConfig()

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate]},
            config=config,
        )

        assert len(results) == 1
        r = results[0]
        assert r.resolved_column == "unit_price"
        assert r.operator == ">"
        assert r.value == "100"
        assert r.confidence == 0.9

    @pytest.mark.asyncio
    async def test_does_not_resolve_below_threshold(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        # Score below default threshold (0.75)
        candidate = _make_candidate("unit_price", 0.5)
        config = GroundingConfig(llm_fallback_enabled=False)

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate]},
            config=config,
        )

        assert len(results) == 1
        r = results[0]
        assert r.resolved_column is None
        assert r.operator == ">"

    @pytest.mark.asyncio
    async def test_does_not_resolve_within_ambiguity_margin(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        # Two close candidates - within ambiguity margin
        candidate1 = _make_candidate("unit_price", 0.85)
        candidate2 = _make_candidate("total_price", 0.80)
        config = GroundingConfig(llm_fallback_enabled=False)

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate1, candidate2]},
            config=config,
        )

        assert len(results) == 1
        r = results[0]
        # Margin is 0.05 which is <= ambiguity_margin (0.1), so not resolved
        assert r.resolved_column is None

    @pytest.mark.asyncio
    async def test_resolves_with_clear_margin(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        # Two candidates with clear margin (> 0.1)
        candidate1 = _make_candidate("unit_price", 0.9)
        candidate2 = _make_candidate("total_price", 0.6)
        config = GroundingConfig()

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate1, candidate2]},
            config=config,
        )

        assert len(results) == 1
        r = results[0]
        assert r.resolved_column == "unit_price"
        assert r.confidence == 0.9

    @pytest.mark.asyncio
    async def test_no_candidates_returns_unresolved(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        config = GroundingConfig()

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={},
            config=config,
        )

        assert len(results) == 1
        r = results[0]
        assert r.resolved_column is None
        assert r.confidence == 0.0

    @pytest.mark.asyncio
    async def test_multiple_predicates(self):
        grounder = PredicateGrounder()
        pred1 = _make_predicate("price", "gt", 100)
        pred2 = _make_predicate("name", "eq", "Alice")

        c1 = _make_candidate("unit_price", 0.9, "float64")
        c2 = _make_candidate("customer_name", 0.88, "object")

        config = GroundingConfig()

        results = await grounder.ground(
            predicates=[pred1, pred2],
            candidates_by_ref={"price": [c1], "name": [c2]},
            config=config,
        )

        assert len(results) == 2
        assert results[0].resolved_column == "unit_price"
        assert results[0].operator == ">"
        assert results[1].resolved_column == "customer_name"
        assert results[1].operator == "=="


# ---------------------------------------------------------------------------
# LLM fallback tests
# ---------------------------------------------------------------------------


class TestLLMFallback:
    @pytest.mark.asyncio
    async def test_fallback_disabled_returns_unresolved(self):
        grounder = PredicateGrounder()
        predicate = _make_predicate()
        candidate = _make_candidate("unit_price", 0.6)
        config = GroundingConfig(llm_fallback_enabled=False)

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate]},
            config=config,
        )

        assert results[0].resolved_column is None

    @pytest.mark.asyncio
    async def test_no_resolver_returns_unresolved(self):
        # No resolver provided, so fallback cannot run
        grounder = PredicateGrounder(resolver=None)
        predicate = _make_predicate()
        candidate = _make_candidate("unit_price", 0.6)
        config = GroundingConfig(llm_fallback_enabled=True)

        results = await grounder.ground(
            predicates=[predicate],
            candidates_by_ref={"price": [candidate]},
            config=config,
        )

        assert results[0].resolved_column is None


# ---------------------------------------------------------------------------
# Package export test
# ---------------------------------------------------------------------------


class TestPackageExport:
    def test_predicate_grounder_exported_from_package(self):
        from finflow_agent.grounding import PredicateGrounder as PG

        assert PG is PredicateGrounder
