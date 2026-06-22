"""Unit tests for grounding/scoring.py deterministic scoring utilities.

Tests cover:
- tokenize(): camelCase, underscores, spaces, mixed, empty
- compute_token_overlap(): Jaccard similarity
- compute_value_concept_match(): exact/partial/no match
- compute_semantic_type_alignment(): reference kind vs column role
- compute_name_similarity(): normalized Levenshtein
"""

import pytest

from finflow_agent.grounding.scoring import (
    compute_name_similarity,
    compute_semantic_type_alignment,
    compute_token_overlap,
    compute_value_concept_match,
    tokenize,
)


# ---------------------------------------------------------------------------
# tokenize() tests
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the tokenize function."""

    def test_underscore_split(self) -> None:
        assert tokenize("payment_method") == ["payment", "method"]

    def test_camel_case_split(self) -> None:
        assert tokenize("paymentMethod") == ["payment", "method"]

    def test_space_split(self) -> None:
        assert tokenize("total amount") == ["total", "amount"]

    def test_mixed_separators(self) -> None:
        assert tokenize("payment_methodName here") == [
            "payment",
            "method",
            "name",
            "here",
        ]

    def test_uppercase_acronym(self) -> None:
        assert tokenize("XMLParser") == ["xml", "parser"]

    def test_all_uppercase(self) -> None:
        assert tokenize("HTTP") == ["http"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_single_word(self) -> None:
        assert tokenize("amount") == ["amount"]

    def test_multiple_underscores(self) -> None:
        assert tokenize("a__b___c") == ["a", "b", "c"]

    def test_leading_trailing_spaces(self) -> None:
        assert tokenize("  hello world  ") == ["hello", "world"]

    def test_lowercases_all_tokens(self) -> None:
        result = tokenize("PaymentMethod")
        assert all(t == t.lower() for t in result)

    def test_complex_camel(self) -> None:
        assert tokenize("getHTTPResponseCode") == ["get", "http", "response", "code"]


# ---------------------------------------------------------------------------
# compute_token_overlap() tests
# ---------------------------------------------------------------------------


class TestComputeTokenOverlap:
    """Tests for Jaccard similarity of token sets."""

    def test_identical_tokens(self) -> None:
        assert compute_token_overlap("payment method", "payment_method") == 1.0

    def test_partial_overlap(self) -> None:
        # tokens: {"amount"} vs {"total", "amount"} → 1/2 = 0.5
        assert compute_token_overlap("amount", "total_amount") == 0.5

    def test_no_overlap(self) -> None:
        assert compute_token_overlap("foo", "bar") == 0.0

    def test_both_empty(self) -> None:
        assert compute_token_overlap("", "") == 0.0

    def test_one_empty(self) -> None:
        assert compute_token_overlap("", "payment") == 0.0
        assert compute_token_overlap("payment", "") == 0.0

    def test_camel_vs_underscore(self) -> None:
        # Both tokenize to the same set
        assert compute_token_overlap("paymentMethod", "payment_method") == 1.0

    def test_result_in_range(self) -> None:
        score = compute_token_overlap("total revenue", "monthly_revenue_amount")
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        # Same inputs always produce same output
        for _ in range(10):
            assert compute_token_overlap("test input", "test_column") == compute_token_overlap("test input", "test_column")


# ---------------------------------------------------------------------------
# compute_value_concept_match() tests
# ---------------------------------------------------------------------------


class TestComputeValueConceptMatch:
    """Tests for value-concept matching."""

    def test_exact_match(self) -> None:
        assert compute_value_concept_match("paypal", ["paypal", "cash", "credit"]) == 1.0

    def test_exact_match_case_insensitive(self) -> None:
        assert compute_value_concept_match("PayPal", ["paypal", "cash"]) == 1.0

    def test_no_match(self) -> None:
        assert compute_value_concept_match("bitcoin", ["paypal", "cash"]) == 0.0

    def test_empty_reference(self) -> None:
        assert compute_value_concept_match("", ["paypal", "cash"]) == 0.0

    def test_empty_values(self) -> None:
        assert compute_value_concept_match("paypal", []) == 0.0

    def test_token_subset_in_value(self) -> None:
        # "credit card" tokens are subset of "credit card" value tokens
        score = compute_value_concept_match("credit card", ["credit card", "debit"])
        assert score == 1.0  # Exact match (full string)

    def test_partial_token_match(self) -> None:
        # "credit" token is a subset of "credit card" value tokens → 0.8 (all ref tokens in one value)
        score = compute_value_concept_match("credit", ["credit card", "debit"])
        assert score == 0.8

    def test_partial_token_overlap_across_values(self) -> None:
        # "fast credit" → tokens {"fast", "credit"}
        # "credit card" has {"credit", "card"}, "debit" has {"debit"}
        # Not all tokens in one value, but "credit" matches across values → partial (1/2 * 0.6 = 0.3)
        score = compute_value_concept_match("fast credit", ["credit card", "debit"])
        assert 0.0 < score <= 0.6

    def test_result_in_range(self) -> None:
        score = compute_value_concept_match("test", ["test_value", "other"])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_semantic_type_alignment() tests
# ---------------------------------------------------------------------------


class TestComputeSemanticTypeAlignment:
    """Tests for semantic type alignment scoring."""

    def test_explicit_name_any_role(self) -> None:
        assert compute_semantic_type_alignment("explicit_name", "measure") == 0.5
        assert compute_semantic_type_alignment("explicit_name", "identifier") == 0.5
        assert compute_semantic_type_alignment("explicit_name", "text") == 0.5

    def test_semantic_concept_measure(self) -> None:
        assert compute_semantic_type_alignment("semantic_concept", "measure") == 0.7

    def test_semantic_concept_dimension(self) -> None:
        assert compute_semantic_type_alignment("semantic_concept", "dimension") == 0.7

    def test_semantic_concept_categorical(self) -> None:
        assert compute_semantic_type_alignment("semantic_concept", "categorical") == 0.7

    def test_semantic_concept_other(self) -> None:
        assert compute_semantic_type_alignment("semantic_concept", "identifier") == 0.4

    def test_value_implied_categorical(self) -> None:
        assert compute_semantic_type_alignment("value_implied", "categorical") == 0.8

    def test_value_implied_dimension(self) -> None:
        assert compute_semantic_type_alignment("value_implied", "dimension") == 0.8

    def test_value_implied_other(self) -> None:
        assert compute_semantic_type_alignment("value_implied", "text") == 0.3

    def test_generic_reference_any(self) -> None:
        assert compute_semantic_type_alignment("generic_reference", "measure") == 0.3
        assert compute_semantic_type_alignment("generic_reference", "text") == 0.3

    def test_unknown_reference_kind(self) -> None:
        assert compute_semantic_type_alignment("nonexistent_kind", "measure") == 0.2

    def test_result_in_range(self) -> None:
        for kind in ["explicit_name", "semantic_concept", "value_implied", "generic_reference", "column_group"]:
            for role in ["measure", "dimension", "categorical", "identifier", "temporal", "text", "boolean", "unknown"]:
                score = compute_semantic_type_alignment(kind, role)
                assert 0.0 <= score <= 1.0, f"Out of range for {kind}/{role}: {score}"


# ---------------------------------------------------------------------------
# compute_name_similarity() tests
# ---------------------------------------------------------------------------


class TestComputeNameSimilarity:
    """Tests for normalized Levenshtein similarity."""

    def test_exact_match(self) -> None:
        assert compute_name_similarity("payment_method", "payment_method") == 1.0

    def test_case_insensitive(self) -> None:
        assert compute_name_similarity("Payment_Method", "payment_method") == 1.0

    def test_completely_different(self) -> None:
        assert compute_name_similarity("abc", "xyz") == 0.0

    def test_partial_similarity(self) -> None:
        score = compute_name_similarity("payment", "payments")
        # "payment" (7) vs "payments" (8) → distance=1, max_len=8 → 1-(1/8) = 0.875
        assert score == pytest.approx(0.875)

    def test_both_empty(self) -> None:
        assert compute_name_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert compute_name_similarity("", "payment") == 0.0
        assert compute_name_similarity("payment", "") == 0.0

    def test_result_in_range(self) -> None:
        score = compute_name_similarity("hello", "world")
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        for _ in range(10):
            assert compute_name_similarity("test", "testing") == compute_name_similarity("test", "testing")

    def test_symmetric(self) -> None:
        # Levenshtein distance is symmetric
        score1 = compute_name_similarity("abc", "abcd")
        score2 = compute_name_similarity("abcd", "abc")
        assert score1 == pytest.approx(score2)
