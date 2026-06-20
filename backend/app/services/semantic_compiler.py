"""Deterministic semantic-to-canonical compiler.

Maps validated, grounded semantic operations into canonical intent actions.
The LLM never touches this layer — all mappings are deterministic.

Mapping examples:
- exclude_columns → drop_columns canonical action
- select_columns → project_columns canonical action
- filter with between → typed filter predicate
- clean → clean canonical action
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.semantic_models import (
    ColumnGroundingResult,
    GroundedSemanticIntent,
    RelationOperator,
    SemanticIntent,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
)

logger = logging.getLogger(__name__)

SEMANTIC_COMPILER_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Operation mapping: semantic → canonical
# ---------------------------------------------------------------------------

# Maps SemanticOperationType to canonical action kind
_OPERATION_TO_CANONICAL: dict[SemanticOperationType, str] = {
    SemanticOperationType.clean: "clean",
    SemanticOperationType.select_columns: "project_columns",
    SemanticOperationType.exclude_columns: "drop_columns",
    SemanticOperationType.filter: "filter_rows",
    SemanticOperationType.sort: "sort_rows",
    SemanticOperationType.limit: "limit_rows",
    SemanticOperationType.visualize: "visualize",
    SemanticOperationType.aggregate: "calculate",
    SemanticOperationType.rename_columns: "rename_columns",
    SemanticOperationType.deduplicate: "clean",
    SemanticOperationType.export: "report",
    SemanticOperationType.format: "report",
}

# Maps RelationOperator to canonical filter operator
_RELATION_TO_CANONICAL: dict[str, str] = {
    RelationOperator.equals.value: "eq",
    RelationOperator.not_equals.value: "neq",
    RelationOperator.greater_than.value: "gt",
    RelationOperator.greater_than_or_equal.value: "gte",
    RelationOperator.less_than.value: "lt",
    RelationOperator.less_than_or_equal.value: "lte",
    RelationOperator.contains.value: "contains",
    RelationOperator.not_contains.value: "contains",  # inverted in mode
    RelationOperator.between.value: "gte",  # decomposed into two conditions
    RelationOperator.in_.value: "eq",  # decomposed into OR conditions
    RelationOperator.not_in.value: "neq",  # decomposed into AND conditions
    RelationOperator.is_null.value: "eq",
    RelationOperator.is_not_null.value: "neq",
    RelationOperator.matches.value: "contains",
    RelationOperator.belongs_to.value: "eq",
}


class SemanticCompilationError(Exception):
    """Raised when semantic intent cannot be compiled to canonical actions."""
    pass


def compile_semantic_to_canonical(
    grounded_intent: GroundedSemanticIntent,
    *,
    output_format: str = "xlsx",
) -> dict[str, Any]:
    """Compile a grounded semantic intent into a canonical intent dict.

    This is the deterministic bridge from semantic understanding to
    the existing canonical intent system.

    Raises SemanticCompilationError if compilation is not possible.
    """
    if not grounded_intent.all_resolved:
        unresolved = grounded_intent.unresolved_references
        raise SemanticCompilationError(
            f"Cannot compile: unresolved column references: {unresolved}"
        )

    intent = grounded_intent.intent
    grounding_map = _build_grounding_map(grounded_intent.grounding_results)

    actions: list[dict[str, Any]] = []
    evidence: list[str] = []
    assumptions: list[str] = []

    for task in intent.tasks:
        canonical_action = _compile_task(task, grounding_map)
        if canonical_action is not None:
            actions.append(canonical_action)
            evidence.append(f"Compiled {task.operation.type.value} task '{task.task_id}'.")

    if not actions:
        raise SemanticCompilationError("No compilable tasks in semantic intent.")

    return {
        "actions": actions,
        "output_format": output_format,
        "evidence": evidence,
        "assumptions": assumptions,
    }


def _build_grounding_map(results: list[ColumnGroundingResult]) -> dict[str, str]:
    """Build a lookup from user_term → resolved_column."""
    mapping: dict[str, str] = {}
    for r in results:
        if r.resolved_column:
            mapping[r.user_term.strip().lower()] = r.resolved_column
    return mapping


def _resolve_user_term(user_term: str, grounding_map: dict[str, str]) -> str | None:
    """Resolve a user term to an actual column name."""
    return grounding_map.get(user_term.strip().lower())


def _compile_task(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any] | None:
    """Compile a single semantic task to a canonical action dict."""
    op_type = task.operation.type

    if op_type == SemanticOperationType.clean:
        return _compile_clean(task)
    elif op_type == SemanticOperationType.deduplicate:
        return _compile_deduplicate(task)
    elif op_type == SemanticOperationType.select_columns:
        return _compile_select_columns(task, grounding_map)
    elif op_type == SemanticOperationType.exclude_columns:
        return _compile_exclude_columns(task, grounding_map)
    elif op_type == SemanticOperationType.filter:
        return _compile_filter(task, grounding_map)
    elif op_type == SemanticOperationType.sort:
        return _compile_sort(task, grounding_map)
    elif op_type == SemanticOperationType.limit:
        return _compile_limit(task)
    elif op_type == SemanticOperationType.visualize:
        return _compile_visualize(task, grounding_map)
    elif op_type == SemanticOperationType.aggregate:
        return _compile_aggregate(task, grounding_map)
    elif op_type == SemanticOperationType.rename_columns:
        return _compile_rename(task, grounding_map)
    elif op_type in (SemanticOperationType.export, SemanticOperationType.format):
        return _compile_report(task)
    else:
        logger.warning("Unsupported semantic operation: %s", op_type.value)
        return None


def _compile_clean(task: SemanticTask) -> dict[str, Any]:
    """Compile a clean task."""
    operations = task.parameters.get("operations", [])
    return {
        "kind": "clean",
        "mode": "explicit" if operations else "safe_default",
        "operations": operations,
    }


def _compile_deduplicate(task: SemanticTask) -> dict[str, Any]:
    """Compile a deduplicate task as a clean with deduplicate operation."""
    return {
        "kind": "clean",
        "mode": "explicit",
        "operations": [{"name": "deduplicate", "parameters": {}}],
    }


def _compile_select_columns(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile a select_columns task → project_columns canonical action."""
    requested_fields = []
    for inp in task.inputs:
        if inp.kind == "column_reference":
            resolved = _resolve_user_term(inp.user_term, grounding_map)
            requested_fields.append({
                "raw_reference": inp.user_term,
                "resolved_column": resolved,
                "resolution_method": "semantic_extraction" if resolved else None,
            })
    return {
        "kind": "project_columns",
        "requested_fields": requested_fields,
    }


def _compile_exclude_columns(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile an exclude_columns task → drop_columns canonical action."""
    requested_fields = []
    for inp in task.inputs:
        if inp.kind == "column_reference":
            resolved = _resolve_user_term(inp.user_term, grounding_map)
            requested_fields.append({
                "raw_reference": inp.user_term,
                "resolved_column": resolved,
                "resolution_method": "semantic_extraction" if resolved else None,
            })
    return {
        "kind": "drop_columns",
        "requested_fields": requested_fields,
    }


def _compile_filter(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile a filter task → filter_rows canonical action."""
    conditions = []
    logic = task.parameters.get("logic", "and")
    mode = task.parameters.get("mode", "keep")

    # Handle single predicate
    predicate = task.parameters.get("predicate")
    if predicate and isinstance(predicate, dict):
        compiled_conditions = _compile_predicate(predicate, grounding_map)
        conditions.extend(compiled_conditions)

    # Handle multiple predicates
    predicates = task.parameters.get("predicates")
    if predicates and isinstance(predicates, list):
        for p in predicates:
            if isinstance(p, dict):
                compiled_conditions = _compile_predicate(p, grounding_map)
                conditions.extend(compiled_conditions)

    # Handle simple input-based filter (column + value in parameters)
    if not conditions and task.inputs:
        for inp in task.inputs:
            if inp.kind == "column_reference":
                value = task.parameters.get("value")
                operator = task.parameters.get("operator", "equals")
                if value is not None:
                    resolved = _resolve_user_term(inp.user_term, grounding_map)
                    canonical_op = _RELATION_TO_CANONICAL.get(operator, "eq")
                    conditions.append({
                        "field": {
                            "raw_reference": inp.user_term,
                            "resolved_column": resolved,
                            "resolution_method": "semantic_extraction" if resolved else None,
                        },
                        "operator": canonical_op,
                        "value": value,
                    })

    return {
        "kind": "filter_rows",
        "mode": mode,
        "conditions": conditions,
        "logic": logic,
    }


def _compile_predicate(
    predicate: dict[str, Any],
    grounding_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Compile a semantic predicate into canonical filter conditions.

    Some operators (between, in) decompose into multiple conditions.
    """
    left = predicate.get("left", {})
    operator = str(predicate.get("operator", "equals"))
    right = predicate.get("right", {})

    # Get the column reference
    user_term = left.get("user_term", "") if isinstance(left, dict) else ""
    resolved = _resolve_user_term(user_term, grounding_map) if user_term else None
    field = {
        "raw_reference": user_term,
        "resolved_column": resolved,
        "resolution_method": "semantic_extraction" if resolved else None,
    }

    # Handle between → decompose into gte + lte
    if operator == RelationOperator.between.value:
        minimum = None
        maximum = None
        if isinstance(right, dict):
            minimum = right.get("minimum")
            maximum = right.get("maximum")
            if minimum is None and "values" in right:
                values = right["values"]
                if isinstance(values, list) and len(values) >= 2:
                    minimum = values[0]
                    maximum = values[1]
        conditions = []
        if minimum is not None:
            conditions.append({"field": field, "operator": "gte", "value": minimum})
        if maximum is not None:
            conditions.append({"field": field, "operator": "lte", "value": maximum})
        return conditions if conditions else [{"field": field, "operator": "gte", "value": 0}]

    # Handle in → single condition with value as list
    if operator in (RelationOperator.in_.value, RelationOperator.belongs_to.value):
        values = []
        if isinstance(right, dict):
            values = right.get("values", [])
            if not values and right.get("value") is not None:
                values = [right["value"]]
        # For canonical format, use multiple eq conditions with OR logic
        # Or a single condition with list value
        return [{"field": field, "operator": "eq", "value": values}]

    # Handle not_in
    if operator == RelationOperator.not_in.value:
        values = []
        if isinstance(right, dict):
            values = right.get("values", [])
        return [{"field": field, "operator": "neq", "value": values}]

    # Handle is_null / is_not_null
    if operator == RelationOperator.is_null.value:
        return [{"field": field, "operator": "eq", "value": None}]
    if operator == RelationOperator.is_not_null.value:
        return [{"field": field, "operator": "neq", "value": None}]

    # Standard single-condition operators
    canonical_op = _RELATION_TO_CANONICAL.get(operator, "eq")
    value = None
    if isinstance(right, dict):
        value = right.get("value")
        if value is None:
            value = right.get("user_term")  # for column comparisons

    return [{"field": field, "operator": canonical_op, "value": value}]


def _compile_sort(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile a sort task."""
    sort_keys = []
    direction = task.parameters.get("direction", "asc")
    for inp in task.inputs:
        if inp.kind == "column_reference":
            resolved = _resolve_user_term(inp.user_term, grounding_map)
            sort_keys.append({
                "column": {
                    "raw_reference": inp.user_term,
                    "resolved_column": resolved,
                    "resolution_method": "semantic_extraction" if resolved else None,
                },
                "direction": direction,
            })
    return {
        "kind": "sort_rows",
        "sort_keys": sort_keys,
    }


def _compile_limit(task: SemanticTask) -> dict[str, Any]:
    """Compile a limit task."""
    limit = task.parameters.get("limit", task.parameters.get("count", 10))
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    return {
        "kind": "limit_rows",
        "limit": max(0, limit),
    }


def _compile_visualize(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile a visualize task."""
    fields = []
    for inp in task.inputs:
        if inp.kind == "column_reference":
            resolved = _resolve_user_term(inp.user_term, grounding_map)
            fields.append({
                "raw_reference": inp.user_term,
                "resolved_column": resolved,
                "resolution_method": "semantic_extraction" if resolved else None,
            })
    return {
        "kind": "visualize",
        "chart_type": task.parameters.get("chart_type"),
        "fields": fields,
    }


def _compile_aggregate(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile an aggregate task."""
    operations = task.parameters.get("operations", [])
    return {
        "kind": "calculate",
        "operations": operations,
    }


def _compile_rename(
    task: SemanticTask,
    grounding_map: dict[str, str],
) -> dict[str, Any]:
    """Compile a rename task."""
    mapping = []
    renames = task.parameters.get("renames", [])
    for rename in renames:
        if isinstance(rename, dict):
            source_term = rename.get("from", "")
            target_name = rename.get("to", "")
            resolved = _resolve_user_term(source_term, grounding_map)
            mapping.append({
                "source": {
                    "raw_reference": source_term,
                    "resolved_column": resolved,
                    "resolution_method": "semantic_extraction" if resolved else None,
                },
                "target_name": target_name,
            })
    return {
        "kind": "rename_columns",
        "mapping": mapping,
    }


def _compile_report(task: SemanticTask) -> dict[str, Any]:
    """Compile a report/export task."""
    return {
        "kind": "report",
        "sections": task.parameters.get("sections", []),
    }
