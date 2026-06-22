"""Unit tests for grounding/llm_adapter.py.

Validates the LLM adapter protocol types, constraints, retry policies,
exceptions, and default constraint configurations.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 18.1
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

import pytest
from pydantic import ValidationError

from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    LLMConstraint,
    LLMProviderError,
    LLMResponse,
    LLMValidationError,
    RetryPolicy,
)


class TestLLMCallSite:
    """Tests for LLMCallSite enum."""

    def test_all_call_sites_defined(self):
        expected = {
            "extraction",
            "repair",
            "schema_inference",
            "column_grounding",
            "predicate_grounding",
            "coverage_shadow",
        }
        assert {s.value for s in LLMCallSite} == expected

    def test_is_str_enum(self):
        assert isinstance(LLMCallSite.EXTRACTION, str)
        assert LLMCallSite.EXTRACTION == "extraction"


class TestRetryPolicy:
    """Tests for RetryPolicy model."""

    def test_default_values(self):
        rp = RetryPolicy()
        assert rp.max_retries == 2
        assert rp.base_delay_seconds == 1.0
        assert rp.max_delay_seconds == 10.0
        assert rp.backoff_factor == 2.0
        assert rp.retryable_errors == ["timeout", "rate_limit", "server_error"]

    def test_custom_values(self):
        rp = RetryPolicy(
            max_retries=5,
            base_delay_seconds=0.5,
            max_delay_seconds=30.0,
            backoff_factor=3.0,
            retryable_errors=["timeout"],
        )
        assert rp.max_retries == 5
        assert rp.backoff_factor == 3.0

    def test_rejects_negative_retries(self):
        with pytest.raises(ValidationError):
            RetryPolicy(max_retries=-1)

    def test_rejects_backoff_below_one(self):
        with pytest.raises(ValidationError):
            RetryPolicy(backoff_factor=0.5)

    def test_rejects_negative_delay(self):
        with pytest.raises(ValidationError):
            RetryPolicy(base_delay_seconds=-1.0)


class TestLLMConstraint:
    """Tests for LLMConstraint model."""

    def test_default_values(self):
        c = LLMConstraint()
        assert c.output_schema is None
        assert c.allowed_operations is None
        assert c.max_tokens is None
        assert c.temperature == 0.0
        assert isinstance(c.retry_policy, RetryPolicy)

    def test_with_output_schema(self):
        schema = {"type": "object", "properties": {"col": {"type": "string"}}}
        c = LLMConstraint(output_schema=schema)
        assert c.output_schema == schema

    def test_with_allowed_operations(self):
        c = LLMConstraint(allowed_operations=["add", "replace", "remove"])
        assert c.allowed_operations == ["add", "replace", "remove"]

    def test_custom_retry_policy(self):
        rp = RetryPolicy(max_retries=0)
        c = LLMConstraint(retry_policy=rp)
        assert c.retry_policy.max_retries == 0


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_minimal_response(self):
        resp = LLMResponse(
            content='{"col": "amount"}',
            call_site=LLMCallSite.COLUMN_GROUNDING,
            latency_ms=150.5,
        )
        assert resp.content == '{"col": "amount"}'
        assert resp.call_site == LLMCallSite.COLUMN_GROUNDING
        assert resp.latency_ms == 150.5
        assert resp.parsed is None
        assert resp.retries_used == 0

    def test_response_with_parsed(self):
        resp = LLMResponse(
            content='{"col": "amount"}',
            parsed={"col": "amount"},
            call_site=LLMCallSite.EXTRACTION,
            latency_ms=200.0,
            retries_used=1,
        )
        assert resp.parsed == {"col": "amount"}
        assert resp.retries_used == 1


class TestExceptions:
    """Tests for LLM exception types."""

    def test_provider_error_attributes(self):
        err = LLMProviderError(
            "Connection timed out", error_type="timeout", call_site="extraction"
        )
        assert str(err) == "Connection timed out"
        assert err.error_type == "timeout"
        assert err.call_site == "extraction"

    def test_validation_error_attributes(self):
        err = LLMValidationError(
            "Invalid JSON response",
            call_site="repair",
            raw_content="{not valid json",
        )
        assert str(err) == "Invalid JSON response"
        assert err.call_site == "repair"
        assert err.raw_content == "{not valid json"

    def test_validation_error_optional_raw_content(self):
        err = LLMValidationError("Schema mismatch", call_site="schema_inference")
        assert err.raw_content is None

    def test_provider_error_is_exception(self):
        with pytest.raises(LLMProviderError):
            raise LLMProviderError("fail", error_type="rate_limit", call_site="repair")

    def test_validation_error_is_exception(self):
        with pytest.raises(LLMValidationError):
            raise LLMValidationError("fail", call_site="extraction")


class TestDefaultConstraints:
    """Tests for DEFAULT_CONSTRAINTS configuration."""

    def test_all_call_sites_have_constraints(self):
        for site in LLMCallSite:
            assert site in DEFAULT_CONSTRAINTS
            assert isinstance(DEFAULT_CONSTRAINTS[site], LLMConstraint)

    def test_extraction_constraint(self):
        c = DEFAULT_CONSTRAINTS[LLMCallSite.EXTRACTION]
        assert c.output_schema is not None
        assert c.max_tokens == 4096
        assert c.temperature == 0.0
        assert c.retry_policy.max_retries == 2

    def test_repair_constraint_allowed_operations(self):
        c = DEFAULT_CONSTRAINTS[LLMCallSite.REPAIR]
        assert c.allowed_operations == ["add", "replace", "remove"]
        assert c.retry_policy.max_retries == 1

    def test_coverage_shadow_no_retries(self):
        """Coverage shadow failures should be silent (0 retries)."""
        c = DEFAULT_CONSTRAINTS[LLMCallSite.COVERAGE_SHADOW]
        assert c.retry_policy.max_retries == 0

    def test_grounding_constraints_match(self):
        """Column and predicate grounding should have similar constraints."""
        col = DEFAULT_CONSTRAINTS[LLMCallSite.COLUMN_GROUNDING]
        pred = DEFAULT_CONSTRAINTS[LLMCallSite.PREDICATE_GROUNDING]
        assert col.max_tokens == pred.max_tokens
        assert col.temperature == pred.temperature
        assert col.retry_policy.max_retries == pred.retry_policy.max_retries
