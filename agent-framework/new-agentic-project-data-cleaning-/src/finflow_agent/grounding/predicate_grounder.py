"""Predicate Grounder for FinFlow's semantic pipeline.

Resolves complete filter predicates (field + operator + value) to physical
columns, owning filter-column resolution, operator mapping, and value
normalization. Does NOT delegate filter-column decisions to Column Grounder.

Uses the Candidate Generation Layer for column scoring and applies LLM
fallback (constrained to existing physical columns) when deterministic
scoring is insufficient. Post-LLM verification is required before accepting
any LLM selection.

Requirements: 2.5, 8.5
"""

from __future__ import annotations

import logging
from typing import Any

from finflow_agent.grounding.evidence import (
    GroundingConfig,
    PredicateGroundingResult,
    ScoredCandidate,
)
from finflow_agent.grounding.llm_adapter import (
    DEFAULT_CONSTRAINTS,
    LLMCallSite,
    SemanticResolver,
)
from finflow_agent.grounding.verification import verify_llm_selection
from finflow_agent.models.draft import UnresolvedPredicate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Operator mapping: raw operator aliases → canonical form
# ---------------------------------------------------------------------------

_OPERATOR_MAP: dict[str, str] = {
    # Equality
    "eq": "==",
    "equals": "==",
    "equal": "==",
    "==": "==",
    "=": "==",
    # Inequality
    "ne": "!=",
    "neq": "!=",
    "not_equal": "!=",
    "not_equals": "!=",
    "!=": "!=",
    "<>": "!=",
    # Greater than
    "gt": ">",
    "greater_than": ">",
    "greater": ">",
    ">": ">",
    # Greater than or equal
    "gte": ">=",
    "ge": ">=",
    "greater_equal": ">=",
    "greater_than_or_equal": ">=",
    ">=": ">=",
    # Less than
    "lt": "<",
    "less_than": "<",
    "less": "<",
    "<": "<",
    # Less than or equal
    "lte": "<=",
    "le": "<=",
    "less_equal": "<=",
    "less_than_or_equal": "<=",
    "<=": "<=",
    # Set membership
    "in": "in",
    "contains": "in",
    "one_of": "in",
    "not_in": "not_in",
    "not in": "not_in",
    "nin": "not_in",
    # Null checks
    "is_null": "is_null",
    "isnull": "is_null",
    "null": "is_null",
    "not_null": "not_null",
    "notnull": "not_null",
    "is_not_null": "not_null",
}

# Dtypes grouped for value normalization
_NUMERIC_DTYPES = {"int64", "float64", "int32", "float32", "number", "int", "float"}
_STRING_DTYPES = {"object", "string", "str", "category"}
_DATETIME_DTYPES = {"datetime64", "datetime64[ns]", "datetime", "date"}


class PredicateGrounder:
    """Resolves complete filter predicates including column, operator, and value.

    Owns: filter column resolution, operator mapping, value normalization.
    Does NOT delegate filter-column decisions to Column Grounder.
    Uses Candidate Generation Layer for column scoring.
    LLM fallback constrained to existing physical columns.
    Post-LLM verification required before accepting.

    Requirements: 2.5, 8.5
    """

    def __init__(self, resolver: SemanticResolver | None = None) -> None:
        """Initialize the Predicate Grounder.

        Args:
            resolver: Optional LLM adapter for fallback resolution. If None,
                LLM fallback is effectively disabled regardless of config.
        """
        self._resolver = resolver

    async def ground(
        self,
        predicates: list[UnresolvedPredicate],
        candidates_by_ref: dict[str, list[ScoredCandidate]],
        config: GroundingConfig,
    ) -> list[PredicateGroundingResult]:
        """Resolve filter predicates including column, operator, and value.

        For each predicate:
        1. Get column candidates from candidates_by_ref using field_ref.reference_text
        2. If top candidate is above confidence_threshold with clear margin → resolve deterministically
        3. Map operator to canonical form
        4. Normalize value based on column dtype
        5. If below threshold and llm_fallback_enabled → call LLM fallback
        6. Apply post-LLM verification (including operator and value shape checks)
        7. If verification fails → return unresolved result

        Args:
            predicates: List of UnresolvedPredicate from the draft.
            candidates_by_ref: Mapping of reference_text → scored candidates
                from the Candidate Generation Layer.
            config: Grounding configuration (thresholds, flags).

        Returns:
            List of PredicateGroundingResult, one per input predicate.
        """
        results: list[PredicateGroundingResult] = []

        for predicate in predicates:
            result = await self._resolve_predicate(predicate, candidates_by_ref, config)
            results.append(result)

        return results

    async def _resolve_predicate(
        self,
        predicate: UnresolvedPredicate,
        candidates_by_ref: dict[str, list[ScoredCandidate]],
        config: GroundingConfig,
    ) -> PredicateGroundingResult:
        """Resolve a single predicate to a physical column with operator and value."""
        ref_text = predicate.field_ref.reference_text
        candidates = candidates_by_ref.get(ref_text, [])

        # Sort candidates by total_score descending
        sorted_candidates = sorted(
            candidates, key=lambda c: c.total_score, reverse=True
        )

        # Map operator to canonical form
        canonical_operator = _map_operator(predicate.operator)

        # Attempt deterministic resolution
        if sorted_candidates:
            leader = sorted_candidates[0]
            runner_up = sorted_candidates[1] if len(sorted_candidates) >= 2 else None
            runner_up_score = runner_up.total_score if runner_up else 0.0
            margin = leader.total_score - runner_up_score

            if (
                leader.total_score >= config.confidence_threshold
                and margin > config.ambiguity_margin
            ):
                # Deterministic resolution: clear winner above threshold
                normalized_value = _normalize_value(
                    predicate.value, self._get_dtype_for_candidate(leader)
                )
                logger.debug(
                    "Predicate '%s' resolved deterministically to '%s' (score=%.3f, margin=%.3f)",
                    ref_text,
                    leader.column_name,
                    leader.total_score,
                    margin,
                )
                return PredicateGroundingResult(
                    resolved_column=leader.column_name,
                    operator=canonical_operator,
                    value=normalized_value,
                    confidence=leader.total_score,
                    evidence=sorted_candidates,
                )

        # Deterministic resolution insufficient — try LLM fallback
        if config.llm_fallback_enabled and self._resolver is not None and sorted_candidates:
            return await self._llm_fallback(
                predicate=predicate,
                candidates=sorted_candidates,
                canonical_operator=canonical_operator,
                config=config,
            )

        # No resolution possible
        logger.debug(
            "Predicate '%s' unresolved: no candidates or LLM fallback disabled",
            ref_text,
        )
        return PredicateGroundingResult(
            resolved_column=None,
            operator=canonical_operator,
            value=None,
            confidence=0.0,
            evidence=sorted_candidates,
        )

    async def _llm_fallback(
        self,
        predicate: UnresolvedPredicate,
        candidates: list[ScoredCandidate],
        canonical_operator: str,
        config: GroundingConfig,
    ) -> PredicateGroundingResult:
        """Invoke LLM fallback constrained to existing physical columns.

        The LLM can only select from columns already present in the candidate set.
        Post-LLM verification is applied before accepting.

        Requirements: 8.5, 8.7, 8.8, 8.9
        """
        assert self._resolver is not None

        # Build the permitted column set from candidates
        permitted_columns = {c.column_name for c in candidates}

        # Construct the LLM prompt
        candidate_descriptions = [
            f"- {c.column_name} (score: {c.total_score:.3f})"
            for c in candidates[:10]  # Limit context window usage
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a column-resolution assistant. Given a filter predicate "
                    "reference and candidate columns, select the best physical column. "
                    "You MUST select from the provided candidates only. "
                    "Respond with JSON: {\"selected_column\": \"<column_name>\"}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Filter predicate reference: '{predicate.field_ref.reference_text}'\n"
                    f"Operator: {predicate.operator}\n"
                    f"Value: {predicate.value}\n\n"
                    f"Candidate columns:\n"
                    + "\n".join(candidate_descriptions)
                ),
            },
        ]

        constraint = DEFAULT_CONSTRAINTS[LLMCallSite.PREDICATE_GROUNDING]

        try:
            response = await self._resolver.call(
                messages,
                call_site=LLMCallSite.PREDICATE_GROUNDING,
                constraint=constraint,
            )
        except Exception:
            logger.warning(
                "LLM fallback failed for predicate '%s', returning unresolved",
                predicate.field_ref.reference_text,
                exc_info=True,
            )
            return PredicateGroundingResult(
                resolved_column=None,
                operator=canonical_operator,
                value=None,
                confidence=0.0,
                evidence=candidates,
            )

        # Extract selected column from LLM response
        llm_selection: str | None = None
        if response.parsed and isinstance(response.parsed.get("selected_column"), str):
            llm_selection = response.parsed["selected_column"]
        elif response.content:
            # Attempt to parse from raw content
            import json

            try:
                parsed = json.loads(response.content)
                if isinstance(parsed.get("selected_column"), str):
                    llm_selection = parsed["selected_column"]
            except (json.JSONDecodeError, TypeError):
                pass

        if llm_selection is None:
            logger.warning(
                "LLM returned no valid selection for predicate '%s'",
                predicate.field_ref.reference_text,
            )
            return PredicateGroundingResult(
                resolved_column=None,
                operator=canonical_operator,
                value=None,
                confidence=0.0,
                evidence=candidates,
            )

        # Build dtype map from candidates for verification
        dtype_map = self._build_dtype_map(candidates)

        # Post-LLM verification (Req 8.7)
        verification = verify_llm_selection(
            llm_selection=llm_selection,
            candidates=candidates,
            permitted_columns=permitted_columns,
            operator=canonical_operator,
            value=predicate.value,
            dtype_map=dtype_map,
            is_destructive=False,  # Filter predicates are not destructive
            ambiguity_margin=config.ambiguity_margin,
        )

        if verification.passed:
            # Find the selected candidate to get its score
            selected_candidate = next(
                (c for c in candidates if c.column_name == llm_selection), None
            )
            confidence = selected_candidate.total_score if selected_candidate else 0.5
            normalized_value = _normalize_value(
                predicate.value,
                dtype_map.get(llm_selection),
            )
            logger.debug(
                "Predicate '%s' resolved via LLM fallback to '%s'",
                predicate.field_ref.reference_text,
                llm_selection,
            )
            return PredicateGroundingResult(
                resolved_column=llm_selection,
                operator=canonical_operator,
                value=normalized_value,
                confidence=confidence,
                evidence=candidates,
            )

        # Verification failed — return unresolved (Req 8.8)
        logger.info(
            "Post-LLM verification failed for predicate '%s' → LLM selected '%s' "
            "(candidate_exists=%s, operator_compatible=%s, value_shape=%s, "
            "dtype_compatible=%s, in_permitted=%s, tie_breaking=%s)",
            predicate.field_ref.reference_text,
            llm_selection,
            verification.candidate_exists,
            verification.operator_compatible,
            verification.value_shape_valid,
            verification.dtype_compatible,
            verification.in_permitted_set,
            verification.tie_breaking_result.value,
        )
        return PredicateGroundingResult(
            resolved_column=None,
            operator=canonical_operator,
            value=None,
            confidence=0.0,
            evidence=candidates,
        )

    @staticmethod
    def _get_dtype_for_candidate(candidate: ScoredCandidate) -> str | None:
        """Extract dtype hint from candidate evidence if available.

        Looks at positive evidence for dtype information.
        """
        for ev in candidate.positive_evidence:
            if ev.startswith("dtype:"):
                return ev.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _build_dtype_map(candidates: list[ScoredCandidate]) -> dict[str, str]:
        """Build a column→dtype mapping from candidate evidence."""
        dtype_map: dict[str, str] = {}
        for candidate in candidates:
            for ev in candidate.positive_evidence:
                if ev.startswith("dtype:"):
                    dtype_map[candidate.column_name] = ev.split(":", 1)[1].strip()
                    break
        return dtype_map


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def _map_operator(raw_operator: str) -> str:
    """Normalize a raw operator string to its canonical form.

    Supports common aliases for equality, inequality, comparison,
    set membership, and null checks.

    Args:
        raw_operator: The raw operator string from the draft.

    Returns:
        Canonical operator string (e.g., '==', '>', 'in', 'is_null').
        Returns the input lowered if no mapping is found.
    """
    normalized = raw_operator.strip().lower()
    return _OPERATOR_MAP.get(normalized, normalized)


def _normalize_value(value: Any, dtype: str | None) -> str | None:
    """Normalize a predicate value based on the target column dtype.

    Converts the value to a string representation appropriate for the
    target dtype. Handles numeric coercion, string trimming, and list
    values for set operators.

    Args:
        value: The raw predicate value (may be str, int, float, list, None).
        dtype: The target column dtype string, or None if unknown.

    Returns:
        Normalized string representation of the value, or None if value is None.
    """
    if value is None:
        return None

    # Handle list values (for 'in' / 'not_in' operators)
    if isinstance(value, (list, tuple, set)):
        normalized_items = [
            _normalize_scalar(item, dtype) for item in value
        ]
        return ",".join(normalized_items)

    return _normalize_scalar(value, dtype)


def _normalize_scalar(value: Any, dtype: str | None) -> str:
    """Normalize a single scalar value based on dtype.

    Args:
        value: A scalar value to normalize.
        dtype: Target column dtype string, or None.

    Returns:
        Normalized string representation.
    """
    if value is None:
        return ""

    if dtype is not None:
        dtype_lower = dtype.lower()

        if dtype_lower in _NUMERIC_DTYPES:
            # Attempt numeric coercion
            try:
                num = float(value)
                # Return integer form if it's a whole number
                if num == int(num):
                    return str(int(num))
                return str(num)
            except (ValueError, TypeError):
                # Not a valid numeric — return as-is
                return str(value).strip()

        if dtype_lower in _STRING_DTYPES:
            return str(value).strip()

        if dtype_lower in _DATETIME_DTYPES:
            # Return as string — leave datetime parsing to execution layer
            return str(value).strip()

    # No dtype info or unknown dtype — return string representation
    return str(value).strip()
