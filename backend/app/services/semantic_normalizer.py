"""Deterministic semantic normalization.

Normalizes equivalent semantic forms into a common canonical representation.
This runs AFTER the LLM extraction and BEFORE grounding.

Examples of equivalences handled:
- "return everything except X" / "hide X" / "not include X" → exclude_columns
- "only X and Y" / "just X and Y" → select_columns
- "between A and B" / "range A to B" → filter with between operator
- "in A, B" / "belongs to A or B" → filter with in operator
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.semantic_models import (
    FilterPredicate,
    RelationOperator,
    SemanticIntent,
    SemanticOperation,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
)

logger = logging.getLogger(__name__)

SEMANTIC_NORMALIZER_VERSION = "1.0"


def normalize_semantic_intent(intent: SemanticIntent) -> SemanticIntent:
    """Normalize a raw semantic intent into canonical form.

    This function:
    1. Merges duplicate tasks with the same operation on the same columns.
    2. Normalizes operator synonyms into canonical RelationOperator values.
    3. Normalizes task dependencies (ensures valid ordering).
    4. Assigns task_ids if missing.

    Returns a new SemanticIntent (does not mutate the input).
    """
    tasks = [_normalize_task(task) for task in intent.tasks]
    tasks = _merge_compatible_tasks(tasks)
    tasks = _assign_task_ids(tasks)
    tasks = _fix_dependencies(tasks)

    return SemanticIntent(
        goals=intent.goals,
        tasks=tasks,
        outputs=intent.outputs,
        constraints=intent.constraints,
        ambiguities=intent.ambiguities,
        unsupported_requirements=intent.unsupported_requirements,
    )


def _normalize_task(task: SemanticTask) -> SemanticTask:
    """Normalize a single task's operation and parameters."""
    # Normalize filter predicates
    parameters = dict(task.parameters)
    if task.operation.type == SemanticOperationType.filter and "predicate" in parameters:
        parameters["predicate"] = _normalize_predicate(parameters["predicate"])

    # Normalize multiple predicates if present
    if task.operation.type == SemanticOperationType.filter and "predicates" in parameters:
        parameters["predicates"] = [
            _normalize_predicate(p) for p in parameters["predicates"]
        ]

    return SemanticTask(
        task_id=task.task_id,
        operation=task.operation,
        inputs=task.inputs,
        parameters=parameters,
        depends_on=task.depends_on,
        confidence=task.confidence,
    )


def _normalize_predicate(predicate: dict[str, Any] | FilterPredicate) -> dict[str, Any]:
    """Normalize a filter predicate's operator to canonical form."""
    if isinstance(predicate, FilterPredicate):
        data = predicate.model_dump()
    elif isinstance(predicate, dict):
        data = dict(predicate)
    else:
        return predicate

    operator = data.get("operator", "")
    normalized_op = _normalize_operator(operator)
    data["operator"] = normalized_op

    # Normalize the right-hand side for between/in/not_in
    right = data.get("right")
    if right and isinstance(right, dict):
        if normalized_op == RelationOperator.between.value:
            # Ensure range_value kind
            if right.get("kind") != "range_value":
                if "minimum" in right and "maximum" in right:
                    right["kind"] = "range_value"
                elif "values" in right and len(right["values"]) == 2:
                    right = {
                        "kind": "range_value",
                        "minimum": right["values"][0],
                        "maximum": right["values"][1],
                    }
            data["right"] = right

        elif normalized_op in (RelationOperator.in_.value, RelationOperator.not_in.value):
            # Ensure list_value kind
            if right.get("kind") != "list_value":
                if "values" in right:
                    right["kind"] = "list_value"
                elif "value" in right and isinstance(right["value"], list):
                    right = {"kind": "list_value", "values": right["value"]}
            data["right"] = right

    return data


def _normalize_operator(operator: str) -> str:
    """Map operator synonyms to canonical RelationOperator values."""
    operator = str(operator).strip().lower()

    # Direct enum values pass through
    try:
        return RelationOperator(operator).value
    except ValueError:
        pass

    # Synonym mapping
    synonyms: dict[str, str] = {
        "eq": RelationOperator.equals.value,
        "=": RelationOperator.equals.value,
        "==": RelationOperator.equals.value,
        "equal": RelationOperator.equals.value,
        "equal_to": RelationOperator.equals.value,
        "is": RelationOperator.equals.value,
        "neq": RelationOperator.not_equals.value,
        "!=": RelationOperator.not_equals.value,
        "<>": RelationOperator.not_equals.value,
        "not_equal": RelationOperator.not_equals.value,
        "not_equal_to": RelationOperator.not_equals.value,
        "gt": RelationOperator.greater_than.value,
        ">": RelationOperator.greater_than.value,
        "above": RelationOperator.greater_than.value,
        "more_than": RelationOperator.greater_than.value,
        "gte": RelationOperator.greater_than_or_equal.value,
        ">=": RelationOperator.greater_than_or_equal.value,
        "at_least": RelationOperator.greater_than_or_equal.value,
        "not_less_than": RelationOperator.greater_than_or_equal.value,
        "lt": RelationOperator.less_than.value,
        "<": RelationOperator.less_than.value,
        "below": RelationOperator.less_than.value,
        "less_than": RelationOperator.less_than.value,
        "lte": RelationOperator.less_than_or_equal.value,
        "<=": RelationOperator.less_than_or_equal.value,
        "at_most": RelationOperator.less_than_or_equal.value,
        "not_more_than": RelationOperator.less_than_or_equal.value,
        "range": RelationOperator.between.value,
        "in_range": RelationOperator.between.value,
        "within": RelationOperator.between.value,
        "one_of": RelationOperator.in_.value,
        "belongs_to": RelationOperator.in_.value,
        "member_of": RelationOperator.in_.value,
        "not_one_of": RelationOperator.not_in.value,
        "does_not_contain": RelationOperator.not_contains.value,
        "null": RelationOperator.is_null.value,
        "missing": RelationOperator.is_null.value,
        "empty": RelationOperator.is_null.value,
        "not_null": RelationOperator.is_not_null.value,
        "present": RelationOperator.is_not_null.value,
        "not_empty": RelationOperator.is_not_null.value,
        "like": RelationOperator.matches.value,
        "regex": RelationOperator.matches.value,
    }

    return synonyms.get(operator, operator)


def _merge_compatible_tasks(tasks: list[SemanticTask]) -> list[SemanticTask]:
    """Merge tasks with identical operations and compatible parameters.

    For example, two separate exclude_columns tasks for different columns
    can be merged into one.
    """
    merged: list[SemanticTask] = []
    exclude_tasks: list[SemanticTask] = []
    select_tasks: list[SemanticTask] = []

    for task in tasks:
        if task.operation.type == SemanticOperationType.exclude_columns:
            exclude_tasks.append(task)
        elif task.operation.type == SemanticOperationType.select_columns:
            select_tasks.append(task)
        else:
            merged.append(task)

    # Merge all exclude_columns into one
    if exclude_tasks:
        all_inputs: list[SemanticReference] = []
        for t in exclude_tasks:
            all_inputs.extend(t.inputs)
        # Deduplicate by user_term
        seen_terms: set[str] = set()
        deduped_inputs: list[SemanticReference] = []
        for inp in all_inputs:
            term = inp.user_term.strip().lower()
            if term not in seen_terms:
                seen_terms.add(term)
                deduped_inputs.append(inp)
        merged.append(SemanticTask(
            task_id=exclude_tasks[0].task_id,
            operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
            inputs=deduped_inputs,
            parameters={},
            depends_on=exclude_tasks[0].depends_on,
            confidence=max((t.confidence or 0.0) for t in exclude_tasks),
        ))

    # Merge all select_columns into one
    if select_tasks:
        all_inputs = []
        for t in select_tasks:
            all_inputs.extend(t.inputs)
        seen_terms = set()
        deduped_inputs = []
        for inp in all_inputs:
            term = inp.user_term.strip().lower()
            if term not in seen_terms:
                seen_terms.add(term)
                deduped_inputs.append(inp)
        merged.append(SemanticTask(
            task_id=select_tasks[0].task_id,
            operation=SemanticOperation(type=SemanticOperationType.select_columns),
            inputs=deduped_inputs,
            parameters={},
            depends_on=select_tasks[0].depends_on,
            confidence=max((t.confidence or 0.0) for t in select_tasks),
        ))

    return merged


def _assign_task_ids(tasks: list[SemanticTask]) -> list[SemanticTask]:
    """Ensure every task has a unique task_id."""
    result: list[SemanticTask] = []
    seen_ids: set[str] = set()
    counter = 1

    for task in tasks:
        task_id = task.task_id.strip() if task.task_id else ""
        if not task_id or task_id in seen_ids:
            task_id = f"task_{counter}"
            while task_id in seen_ids:
                counter += 1
                task_id = f"task_{counter}"
        seen_ids.add(task_id)
        result.append(SemanticTask(
            task_id=task_id,
            operation=task.operation,
            inputs=task.inputs,
            parameters=task.parameters,
            depends_on=task.depends_on,
            confidence=task.confidence,
        ))
        counter += 1

    return result


def _fix_dependencies(tasks: list[SemanticTask]) -> list[SemanticTask]:
    """Remove invalid dependency references."""
    valid_ids = {t.task_id for t in tasks}
    result: list[SemanticTask] = []
    for task in tasks:
        valid_deps = [d for d in task.depends_on if d in valid_ids and d != task.task_id]
        result.append(SemanticTask(
            task_id=task.task_id,
            operation=task.operation,
            inputs=task.inputs,
            parameters=task.parameters,
            depends_on=valid_deps,
            confidence=task.confidence,
        ))
    return result
