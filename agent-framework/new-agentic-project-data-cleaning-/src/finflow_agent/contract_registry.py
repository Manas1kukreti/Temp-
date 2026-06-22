"""Contract Registry — single source of truth for canonical operators and action kinds.

This module defines the shared contract between the semantic compiler and execution engine.
It provides:
- CanonicalOperator enum: all valid filter operators
- ActionKind enum: all valid action kinds
- Exception hierarchy for contract violations
- Semantic-to-canonical mapping dictionaries
- Validation and resolution functions
- Coverage check functions for import-time handler verification

The module has zero internal dependencies (only stdlib) to avoid circular imports.
"""

from enum import Enum
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CanonicalOperator(str, Enum):
    """All valid canonical filter operators.

    Each member's value is the string token used in serialized canonical intents.
    Using str mixin ensures JSON serialization produces the string value directly
    and that CanonicalOperator.EQ == "eq" evaluates to True.
    """

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class ActionKind(str, Enum):
    """All valid canonical action kinds.

    Each member's value is the string token used in serialized canonical intents.
    """

    CLEAN = "clean"
    PROJECT_COLUMNS = "project_columns"
    DROP_COLUMNS = "drop_columns"
    RENAME_COLUMNS = "rename_columns"
    FILTER_ROWS = "filter_rows"
    SORT_ROWS = "sort_rows"
    LIMIT_ROWS = "limit_rows"
    CALCULATE = "calculate"
    VISUALIZE = "visualize"
    REPORT = "report"


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------


class ContractViolationError(Exception):
    """Base class for contract violations."""

    pass


class InvalidOperatorError(ContractViolationError):
    """Raised when an operator string is not in CanonicalOperator."""

    pass


class InvalidActionKindError(ContractViolationError):
    """Raised when an action kind string is not in ActionKind."""

    pass


class UnmappedSemanticTypeError(ContractViolationError):
    """Raised when a semantic type has no canonical mapping."""

    pass


# ---------------------------------------------------------------------------
# Semantic Mapping Dictionaries
# ---------------------------------------------------------------------------

# Maps semantic relation operator strings to canonical operators.
# This is the single place where semantic-to-canonical operator mapping is defined.
SEMANTIC_RELATION_TO_OPERATOR: Dict[str, CanonicalOperator] = {
    "eq": CanonicalOperator.EQ,
    "neq": CanonicalOperator.NEQ,
    "gt": CanonicalOperator.GT,
    "gte": CanonicalOperator.GTE,
    "lt": CanonicalOperator.LT,
    "lte": CanonicalOperator.LTE,
    "contains": CanonicalOperator.CONTAINS,
    "not_contains": CanonicalOperator.NOT_CONTAINS,
    "starts_with": CanonicalOperator.STARTS_WITH,
    "ends_with": CanonicalOperator.ENDS_WITH,
    "between": CanonicalOperator.BETWEEN,
    "in": CanonicalOperator.IN,
    "not_in": CanonicalOperator.NOT_IN,
    "is_null": CanonicalOperator.IS_NULL,
    "is_not_null": CanonicalOperator.IS_NOT_NULL,
}

# Maps semantic operation type strings to canonical action kinds.
SEMANTIC_OPERATION_TO_ACTION_KIND: Dict[str, ActionKind] = {
    "clean": ActionKind.CLEAN,
    "project_columns": ActionKind.PROJECT_COLUMNS,
    "drop_columns": ActionKind.DROP_COLUMNS,
    "rename_columns": ActionKind.RENAME_COLUMNS,
    "filter_rows": ActionKind.FILTER_ROWS,
    "sort_rows": ActionKind.SORT_ROWS,
    "limit_rows": ActionKind.LIMIT_ROWS,
    "calculate": ActionKind.CALCULATE,
    "visualize": ActionKind.VISUALIZE,
    "report": ActionKind.REPORT,
}


# ---------------------------------------------------------------------------
# Validation & Resolution Functions
# ---------------------------------------------------------------------------


def validate_operator(operator: str) -> CanonicalOperator:
    """Validate and return the canonical operator for a string token.

    Raises InvalidOperatorError with the invalid value and valid options if not found.
    """
    try:
        return CanonicalOperator(operator)
    except ValueError:
        valid = [op.value for op in CanonicalOperator]
        raise InvalidOperatorError(
            f"Invalid canonical operator: {operator!r}. "
            f"Valid operators: {valid}"
        )


def validate_action_kind(kind: str) -> ActionKind:
    """Validate and return the canonical action kind for a string token.

    Raises InvalidActionKindError with the invalid value and valid options if not found.
    """
    try:
        return ActionKind(kind)
    except ValueError:
        valid = [ak.value for ak in ActionKind]
        raise InvalidActionKindError(
            f"Invalid canonical action kind: {kind!r}. "
            f"Valid action kinds: {valid}"
        )


def resolve_semantic_operator(relation_operator: str) -> CanonicalOperator:
    """Map a semantic relation operator to its canonical operator.

    Raises UnmappedSemanticTypeError if no mapping exists.
    """
    result = SEMANTIC_RELATION_TO_OPERATOR.get(relation_operator)
    if result is None:
        raise UnmappedSemanticTypeError(
            f"Unmapped semantic relation operator: {relation_operator!r}. "
            f"Add a registration entry in SEMANTIC_RELATION_TO_OPERATOR in contract_registry.py."
        )
    return result


def resolve_semantic_operation_type(operation_type: str) -> ActionKind:
    """Map a semantic operation type to its canonical action kind.

    Raises UnmappedSemanticTypeError if no mapping exists.
    """
    result = SEMANTIC_OPERATION_TO_ACTION_KIND.get(operation_type)
    if result is None:
        raise UnmappedSemanticTypeError(
            f"Unmapped semantic operation type: {operation_type!r}. "
            f"Add a registration entry in SEMANTIC_OPERATION_TO_ACTION_KIND in contract_registry.py."
        )
    return result


# ---------------------------------------------------------------------------
# Coverage Check Functions
# ---------------------------------------------------------------------------


def check_operator_handler_coverage(handler_registry: Dict[str, Any]) -> None:
    """Verify every CanonicalOperator has a handler and no unknown handlers exist.

    Called at module import time by the execution engine.
    Raises ImportError identifying any unhandled or unknown operators.
    """
    enum_values = {op.value for op in CanonicalOperator}
    handler_values = set(handler_registry.keys())

    missing = enum_values - handler_values
    unknown = handler_values - enum_values

    errors = []
    if missing:
        errors.append(
            f"Filter handler registry is missing handlers for operators: {sorted(missing)}. "
            f"Add handler functions for these operators in filter_handlers.py."
        )
    if unknown:
        errors.append(
            f"Filter handler registry contains unknown operators: {sorted(unknown)}. "
            f"Remove them or add corresponding enum members to CanonicalOperator."
        )
    if errors:
        raise ImportError(" ".join(errors))


def check_action_kind_coverage(action_handler_registry: Dict[str, Any]) -> None:
    """Verify every ActionKind has an execution path and no unknown handlers exist.

    Called at module import time by the execution engine.
    Raises ImportError identifying any unhandled or unknown action kinds.
    """
    enum_values = {ak.value for ak in ActionKind}
    handler_values = set(action_handler_registry.keys())

    missing = enum_values - handler_values
    unknown = handler_values - enum_values

    errors = []
    if missing:
        errors.append(
            f"Action handler registry is missing execution paths for: {sorted(missing)}. "
            f"Add handler functions for these action kinds in the executor module."
        )
    if unknown:
        errors.append(
            f"Action handler registry contains unknown action kinds: {sorted(unknown)}. "
            f"Remove them or add corresponding enum members to ActionKind."
        )
    if errors:
        raise ImportError(" ".join(errors))
