"""LLM adapter protocol for FinFlow's semantic pipeline.

Defines the SemanticResolver protocol, LLM call site classification, constraint
models, retry policies, and exception types. Each LLM call site has explicit
bounded behavior, constraint definitions, and failure semantics.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 18.1
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Raised on LLM provider failures (timeout, rate limit, server error).

    These are infrastructure-level failures that may be retried according to
    the bounded retry policy. They map to interpretation_failed status, never
    needs_clarification.

    Requirements: 18.1
    """

    def __init__(self, message: str, *, error_type: str, call_site: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.call_site = call_site


class LLMValidationError(Exception):
    """Raised when LLM response fails validation (invalid JSON, schema mismatch).

    These are non-retryable failures: the provider responded but the content
    does not satisfy the declared output constraints.

    Requirements: 18.1
    """

    def __init__(
        self, message: str, *, call_site: str, raw_content: str | None = None
    ) -> None:
        super().__init__(message)
        self.call_site = call_site
        self.raw_content = raw_content


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LLMCallSite(str, Enum):
    """Identifies which pipeline component is making the LLM call.

    Used for metrics, tracing, constraint lookup, and retry policy selection.
    Each call site has distinct output expectations and failure semantics.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """

    EXTRACTION = "extraction"
    REPAIR = "repair"
    SCHEMA_INFERENCE = "schema_inference"
    COLUMN_GROUNDING = "column_grounding"
    PREDICATE_GROUNDING = "predicate_grounding"
    COVERAGE_SHADOW = "coverage_shadow"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """Bounded retry configuration per LLM call site.

    Implements exponential backoff with configurable bounds. Non-retryable
    errors (invalid_json, schema_validation) fail immediately without retry.

    Requirements: 18.1
    """

    model_config = ConfigDict(strict=True)

    max_retries: int = Field(default=2, ge=0)
    base_delay_seconds: float = Field(default=1.0, ge=0.0)
    max_delay_seconds: float = Field(default=10.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    retryable_errors: list[str] = Field(
        default_factory=lambda: ["timeout", "rate_limit", "server_error"]
    )


class LLMConstraint(BaseModel):
    """Defines constraints for a specific LLM call site.

    Each call site declares its output expectations, allowed operations,
    retry behavior, and generation parameters. This ensures LLM usage is
    auditable and bounded.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """

    model_config = ConfigDict(strict=True)

    output_schema: dict[str, Any] | None = None  # Expected output JSON schema
    allowed_operations: list[str] | None = None  # For repair: only add/replace/remove
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    max_tokens: int | None = None
    temperature: float = 0.0


class LLMResponse(BaseModel):
    """Response from an LLM call.

    Contains both the raw content and optionally parsed JSON if the call site
    declared an output schema. Includes latency and retry metadata for
    observability.
    """

    model_config = ConfigDict(strict=True)

    content: str
    parsed: dict[str, Any] | None = None  # Parsed JSON if output_schema was provided
    call_site: LLMCallSite
    latency_ms: float
    retries_used: int = 0


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SemanticResolver(Protocol):
    """LLM adapter with explicit constraints per call site.

    Implementations of this protocol wrap a specific LLM provider and enforce
    the constraint and retry policies defined for each call site. All LLM
    interactions in the pipeline go through this protocol.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 18.1
    """

    async def call(
        self,
        messages: list[dict[str, str]],
        *,
        call_site: LLMCallSite,
        constraint: LLMConstraint,
        timeout: float = 30.0,
    ) -> LLMResponse:
        """Make an LLM call with site-specific constraints.

        Args:
            messages: Chat-format messages for the LLM.
            call_site: Identifies which component is calling (for metrics/tracing).
            constraint: Defines output schema, allowed operations, retry policy.
            timeout: Maximum wall-clock time for the call in seconds.

        Returns:
            LLMResponse with content and optional parsed output.

        Raises:
            LLMProviderError: On infrastructure failures (timeout, rate limit).
            LLMValidationError: On response validation failures (bad JSON, schema mismatch).
        """
        ...


# ---------------------------------------------------------------------------
# Default Constraints per Call Site
# ---------------------------------------------------------------------------


DEFAULT_CONSTRAINTS: dict[LLMCallSite, LLMConstraint] = {
    LLMCallSite.EXTRACTION: LLMConstraint(
        output_schema={
            "type": "object",
            "description": "SemanticIntentDraft JSON structure",
        },
        retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=1.0),
        max_tokens=4096,
        temperature=0.0,
    ),
    LLMCallSite.REPAIR: LLMConstraint(
        output_schema={
            "type": "array",
            "items": {"type": "object"},
            "description": "List of SemanticPatch operations",
        },
        allowed_operations=["add", "replace", "remove"],
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.5),
        max_tokens=2048,
        temperature=0.0,
    ),
    LLMCallSite.SCHEMA_INFERENCE: LLMConstraint(
        output_schema={
            "type": "object",
            "description": "Column role and semantic type proposals",
        },
        retry_policy=RetryPolicy(max_retries=2, base_delay_seconds=1.0),
        max_tokens=2048,
        temperature=0.0,
    ),
    LLMCallSite.COLUMN_GROUNDING: LLMConstraint(
        output_schema={
            "type": "object",
            "properties": {"selected_column": {"type": "string"}},
            "description": "Selected physical column for standalone reference",
        },
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.5),
        max_tokens=512,
        temperature=0.0,
    ),
    LLMCallSite.PREDICATE_GROUNDING: LLMConstraint(
        output_schema={
            "type": "object",
            "properties": {"selected_column": {"type": "string"}},
            "description": "Selected physical column for filter predicate",
        },
        retry_policy=RetryPolicy(max_retries=1, base_delay_seconds=0.5),
        max_tokens=512,
        temperature=0.0,
    ),
    LLMCallSite.COVERAGE_SHADOW: LLMConstraint(
        output_schema={
            "type": "object",
            "description": "Shadow coverage comparison result",
        },
        retry_policy=RetryPolicy(max_retries=0, base_delay_seconds=0.0),
        max_tokens=2048,
        temperature=0.0,
    ),
}
