"""Post-LLM deterministic verification and tie-breaking for the grounding layer.

After an LLM fallback returns a column selection, the pipeline applies
deterministic verification checks before accepting the result. This ensures
that LLM outputs are constrained to physically valid, compatible selections
and that ambiguous situations route to clarification rather than silent
acceptance.

Key components:
- TieBreakingPolicy: Enum describing the disposition of a tie-breaking decision.
- PostLLMVerification: Model collecting all verification check results.
- apply_tie_breaking_policy(): Deterministic tie-breaking rules.
- verify_llm_selection(): Full verification of an LLM selection against evidence.

Requirements: 8.7, 8.8, 8.9
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from finflow_agent.grounding.evidence import ScoredCandidate


class TieBreakingPolicy(str, Enum):
    """Disposition of a tie-breaking decision between LLM and deterministic scoring.

    RESOLVE: LLM agrees with a strong deterministic leader — accept the result.
    CLARIFY: Ambiguous situation — route to user clarification.
    REJECT:  LLM conflicts with deterministic evidence — reject the selection.
    """

    RESOLVE = "resolve"
    CLARIFY = "clarify"
    REJECT = "reject"


class PostLLMVerification(BaseModel):
    """Result of post-LLM deterministic verification checks.

    All boolean fields represent individual verification gates. The overall
    verification passes only when all gates are True and the tie-breaking
    result is RESOLVE.
    """

    model_config = ConfigDict(strict=True)

    candidate_exists: bool
    """The LLM-selected column exists in the candidate set."""

    operator_compatible: bool
    """The operator is compatible with the selected column's type."""

    value_shape_valid: bool
    """The value has a valid shape for the operator and column."""

    dtype_compatible: bool
    """The value's dtype is compatible with the selected column."""

    in_permitted_set: bool
    """The selected column is within the permitted column set."""

    tie_breaking_result: TieBreakingPolicy
    """Result of the tie-breaking policy evaluation."""

    @property
    def passed(self) -> bool:
        """All verification checks passed and tie-breaking resolved."""
        return all([
            self.candidate_exists,
            self.operator_compatible,
            self.value_shape_valid,
            self.dtype_compatible,
            self.in_permitted_set,
            self.tie_breaking_result == TieBreakingPolicy.RESOLVE,
        ])


# ---------------------------------------------------------------------------
# Tie-breaking policy
# ---------------------------------------------------------------------------


def apply_tie_breaking_policy(
    llm_selection: str,
    deterministic_leader: ScoredCandidate | None,
    runner_up: ScoredCandidate | None,
    ambiguity_margin: float,
    is_destructive: bool,
) -> TieBreakingPolicy:
    """Apply deterministic tie-breaking rules after LLM fallback.

    Rules (evaluated in order):
    1. No deterministic leader at all → CLARIFY (insufficient evidence).
    2. Destructive operation with close candidates (margin ≤ ambiguity_margin)
       → CLARIFY regardless of LLM agreement.
    3. LLM agrees with deterministic leader AND margin > ambiguity_margin
       → RESOLVE (strong agreement with strong leader).
    4. LLM agrees with deterministic leader but margin ≤ ambiguity_margin
       → CLARIFY (LLM is the only evidence breaking a close tie).
    5. LLM conflicts with deterministic leader → CLARIFY.

    Args:
        llm_selection: Column name selected by the LLM.
        deterministic_leader: Highest-scored candidate from deterministic scoring,
            or None if no candidates were scored.
        runner_up: Second-highest scored candidate, or None if fewer than 2 candidates.
        ambiguity_margin: Threshold below which scores are considered too close.
        is_destructive: Whether the operation is destructive (drop, rename, etc.).

    Returns:
        TieBreakingPolicy indicating the disposition.
    """
    # Rule 1: No deterministic leader — cannot resolve without evidence
    if deterministic_leader is None:
        return TieBreakingPolicy.CLARIFY

    # Compute margin between leader and runner-up
    runner_up_score = runner_up.total_score if runner_up else 0.0
    margin = deterministic_leader.total_score - runner_up_score

    # Rule 2: Destructive operation with close candidates → always clarify
    if is_destructive and margin <= ambiguity_margin:
        return TieBreakingPolicy.CLARIFY

    # Rule 3 & 4: LLM agrees with deterministic leader
    if llm_selection == deterministic_leader.column_name:
        if margin > ambiguity_margin:
            return TieBreakingPolicy.RESOLVE
        else:
            # Close tie — LLM alone isn't enough to break it
            return TieBreakingPolicy.CLARIFY

    # Rule 5: LLM conflicts with deterministic leader
    return TieBreakingPolicy.CLARIFY


# ---------------------------------------------------------------------------
# Full LLM selection verification
# ---------------------------------------------------------------------------

# Operators grouped by the value shapes they accept
_COMPARISON_OPERATORS = {"==", "!=", ">", "<", ">=", "<="}
_SET_OPERATORS = {"in", "not_in"}
_UNARY_OPERATORS = {"is_null", "not_null"}
_ALL_OPERATORS = _COMPARISON_OPERATORS | _SET_OPERATORS | _UNARY_OPERATORS

# Dtype compatibility: which dtypes support comparison operators
_NUMERIC_DTYPES = {"int64", "float64", "int32", "float32", "number", "int", "float"}
_STRING_DTYPES = {"object", "string", "str", "category"}
_DATETIME_DTYPES = {"datetime64", "datetime64[ns]", "datetime", "date"}
_ORDERABLE_DTYPES = _NUMERIC_DTYPES | _DATETIME_DTYPES


def _check_operator_compatible(operator: str | None, dtype: str | None) -> bool:
    """Check if the operator is compatible with the column dtype."""
    if operator is None:
        # No operator to validate (e.g., projection/sort context)
        return True

    if operator in _UNARY_OPERATORS:
        # Unary operators work with any dtype
        return True

    if operator in {"==", "!=", "in", "not_in"}:
        # Equality and set membership work with any dtype
        return True

    # Ordering operators (>, <, >=, <=) require orderable types
    if operator in {">", "<", ">=", "<="}:
        if dtype is None:
            # Cannot confirm compatibility without dtype
            return False
        return dtype.lower() in _ORDERABLE_DTYPES

    # Unknown operator — fail safe
    return False


def _check_value_shape_valid(operator: str | None, value: Any) -> bool:
    """Check that the value has the correct shape for the operator."""
    if operator is None:
        # No operator means no value constraint
        return True

    if operator in _UNARY_OPERATORS:
        # Unary operators don't use a value — any value (including None) is fine
        return True

    if operator in _SET_OPERATORS:
        # Set operators require an iterable (list, tuple, set)
        return isinstance(value, (list, tuple, set))

    # Comparison operators require a scalar (not a collection)
    if operator in _COMPARISON_OPERATORS:
        return not isinstance(value, (list, tuple, set, dict))

    # Unknown operator — permissive
    return True


def _check_dtype_compatible(value: Any, dtype: str | None) -> bool:
    """Check that the value is compatible with the column dtype."""
    if dtype is None or value is None:
        # No dtype info or no value — cannot disprove compatibility
        return True

    dtype_lower = dtype.lower()

    if dtype_lower in _NUMERIC_DTYPES:
        # For numeric columns, value should be numeric or a numeric string
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        if isinstance(value, (list, tuple, set)):
            # For set operators — check all elements
            return all(
                isinstance(v, (int, float))
                or (isinstance(v, str) and _is_numeric_string(v))
                for v in value
            )
        return False

    # String and other dtypes — anything is acceptable as it can be coerced
    return True


def _is_numeric_string(s: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def verify_llm_selection(
    llm_selection: str,
    candidates: list[ScoredCandidate],
    permitted_columns: set[str],
    operator: str | None,
    value: Any,
    dtype_map: dict[str, str],
    is_destructive: bool,
    ambiguity_margin: float,
) -> PostLLMVerification:
    """Perform full post-LLM deterministic verification of a selection.

    Runs all verification checks against the LLM's column selection and
    returns a PostLLMVerification result. Any check failure means the
    overall verification fails and the pipeline should route to
    Clarification rather than accepting the LLM result.

    Args:
        llm_selection: Column name selected by the LLM fallback.
        candidates: Scored candidates from the Candidate Generation Layer.
        permitted_columns: Set of physical column names the LLM is allowed
            to select from.
        operator: The canonical operator (e.g., '==', '>', 'in'), or None
            for non-predicate contexts.
        value: The predicate value to apply, or None for non-predicate contexts.
        dtype_map: Mapping of column_name → dtype string for the dataset.
        is_destructive: Whether the operation is destructive.
        ambiguity_margin: Threshold for the tie-breaking policy.

    Returns:
        PostLLMVerification with all check results populated.
    """
    # 1. candidate_exists: LLM selection must be in the candidate set
    candidate_names = {c.column_name for c in candidates}
    candidate_exists = llm_selection in candidate_names

    # 2. in_permitted_set: LLM selection must be in the permitted columns
    in_permitted_set = llm_selection in permitted_columns

    # 3. operator_compatible: operator must be valid for the column's dtype
    column_dtype = dtype_map.get(llm_selection)
    operator_compatible = _check_operator_compatible(operator, column_dtype)

    # 4. value_shape_valid: value shape must match operator expectations
    value_shape_valid = _check_value_shape_valid(operator, value)

    # 5. dtype_compatible: value must be compatible with column dtype
    dtype_compatible = _check_dtype_compatible(value, column_dtype)

    # 6. tie_breaking_result: apply tie-breaking policy
    # Sort candidates by total_score descending to find leader and runner-up
    sorted_candidates = sorted(candidates, key=lambda c: c.total_score, reverse=True)
    deterministic_leader = sorted_candidates[0] if len(sorted_candidates) >= 1 else None
    runner_up = sorted_candidates[1] if len(sorted_candidates) >= 2 else None

    tie_breaking_result = apply_tie_breaking_policy(
        llm_selection=llm_selection,
        deterministic_leader=deterministic_leader,
        runner_up=runner_up,
        ambiguity_margin=ambiguity_margin,
        is_destructive=is_destructive,
    )

    return PostLLMVerification(
        candidate_exists=candidate_exists,
        operator_compatible=operator_compatible,
        value_shape_valid=value_shape_valid,
        dtype_compatible=dtype_compatible,
        in_permitted_set=in_permitted_set,
        tie_breaking_result=tie_breaking_result,
    )
