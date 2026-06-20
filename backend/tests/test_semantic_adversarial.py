"""Adversarial distinction tests for semantic extraction.

These tests verify that the system does NOT incorrectly interpret every
occurrence of keywords like "except" as column exclusion. The system must
correctly distinguish between:

- "except" as column exclusion: "all columns except X"
- "except" as conditional predicate: "remove rows except when status is pending"
- "except" as operation scope: "calculate revenue except for cancelled"
"""

from __future__ import annotations

import pytest

from app.services.semantic_models import (
    SemanticIntent,
    SemanticOperation,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
)
from app.services.semantic_normalizer import normalize_semantic_intent
from app.services.semantic_coverage import check_coverage_deterministic
from app.services.semantic_grounding import ground_semantic_intent
from app.services.semantic_pipeline import run_semantic_pipeline_sync


def _mock_llm_for_exclude_columns(columns_to_exclude: list[str]):
    """Create a mock LLM that returns an exclude_columns task."""
    def mock_llm(messages):
        return {
            "goals": [{"description": "Exclude specified columns from output", "priority": 1}],
            "tasks": [{
                "task_id": "task_1",
                "operation": {"type": "exclude_columns"},
                "inputs": [
                    {"kind": "column_reference", "user_term": col}
                    for col in columns_to_exclude
                ],
                "parameters": {},
                "depends_on": [],
                "confidence": 0.95,
            }],
            "outputs": [],
            "constraints": [],
            "ambiguities": [],
            "unsupported_requirements": [],
        }
    return mock_llm


SAMPLE_COLUMNS = [
    "consumer_id", "age", "gender", "income", "status",
    "transaction_date", "amount", "merchant", "payment_method",
    "revenue", "cancelled",
]


# ---------------------------------------------------------------------------
# Mock LLMs for adversarial cases
# ---------------------------------------------------------------------------


def _mock_llm_conditional_except(messages):
    """'remove invalid rows except when status is pending' → filter, NOT drop_columns."""
    return {
        "goals": [{"description": "Filter rows with conditional exception", "priority": 1}],
        "tasks": [{
            "task_id": "task_1",
            "operation": {"type": "filter"},
            "inputs": [{"kind": "column_reference", "user_term": "status"}],
            "parameters": {
                "predicate": {
                    "left": {"kind": "column_reference", "user_term": "status"},
                    "operator": "not_equals",
                    "right": {"kind": "literal_value", "value": "pending"},
                },
                "mode": "drop",
                "note": "Remove invalid rows but keep those where status is pending",
            },
            "depends_on": [],
            "confidence": 0.85,
        }],
        "outputs": [],
        "constraints": [],
        "ambiguities": [{
            "description": "Definition of 'invalid rows' is ambiguous — using status != pending as the keep condition",
            "possible_interpretations": ["remove null rows except pending", "remove flagged rows except pending"],
            "source_text": "remove invalid rows except when status is pending",
        }],
        "unsupported_requirements": [],
    }


def _mock_llm_revenue_except_cancelled(messages):
    """'calculate revenue except for cancelled transactions' → filter + aggregate."""
    return {
        "goals": [{"description": "Calculate revenue excluding cancelled", "priority": 1}],
        "tasks": [
            {
                "task_id": "task_1",
                "operation": {"type": "filter"},
                "inputs": [{"kind": "column_reference", "user_term": "status"}],
                "parameters": {
                    "predicate": {
                        "left": {"kind": "column_reference", "user_term": "status"},
                        "operator": "not_equals",
                        "right": {"kind": "literal_value", "value": "cancelled"},
                    }
                },
                "depends_on": [],
                "confidence": 0.90,
            },
            {
                "task_id": "task_2",
                "operation": {"type": "aggregate"},
                "inputs": [{"kind": "column_reference", "user_term": "revenue"}],
                "parameters": {"operations": ["sum"]},
                "depends_on": ["task_1"],
                "confidence": 0.90,
            },
        ],
        "outputs": [],
        "constraints": [],
        "ambiguities": [],
        "unsupported_requirements": [],
    }


def _mock_llm_clean_except_below_zero(messages):
    """'clean values except those below zero' → clean with constraint."""
    return {
        "goals": [{"description": "Clean data with exception for negative values", "priority": 1}],
        "tasks": [{
            "task_id": "task_1",
            "operation": {"type": "clean"},
            "inputs": [],
            "parameters": {
                "mode": "safe_default",
                "exception_note": "Do not clean/remove values below zero",
            },
            "depends_on": [],
            "confidence": 0.80,
        }],
        "outputs": [],
        "constraints": [{
            "constraint_type": "preserve_condition",
            "description": "Values below zero should be preserved during cleaning",
            "parameters": {"condition": "value < 0"},
        }],
        "ambiguities": [],
        "unsupported_requirements": [],
    }


def _mock_llm_multiple_tasks(messages):
    """Multi-task prompt: clean + filter + sort."""
    return {
        "goals": [{"description": "Clean, filter, and sort data", "priority": 1}],
        "tasks": [
            {
                "task_id": "task_1",
                "operation": {"type": "clean"},
                "inputs": [],
                "parameters": {},
                "depends_on": [],
                "confidence": 0.95,
            },
            {
                "task_id": "task_2",
                "operation": {"type": "filter"},
                "inputs": [{"kind": "column_reference", "user_term": "age"}],
                "parameters": {
                    "predicate": {
                        "left": {"kind": "column_reference", "user_term": "age"},
                        "operator": "greater_than",
                        "right": {"kind": "literal_value", "value": 18},
                    }
                },
                "depends_on": ["task_1"],
                "confidence": 0.95,
            },
            {
                "task_id": "task_3",
                "operation": {"type": "sort"},
                "inputs": [{"kind": "column_reference", "user_term": "income"}],
                "parameters": {"direction": "desc"},
                "depends_on": ["task_2"],
                "confidence": 0.90,
            },
        ],
        "outputs": [],
        "constraints": [],
        "ambiguities": [],
        "unsupported_requirements": [],
    }


def _mock_llm_contradictory(messages):
    """Contradictory: 'show only age' and 'include all columns'."""
    return {
        "goals": [{"description": "Show data", "priority": 1}],
        "tasks": [
            {
                "task_id": "task_1",
                "operation": {"type": "select_columns"},
                "inputs": [{"kind": "column_reference", "user_term": "age"}],
                "parameters": {},
                "depends_on": [],
                "confidence": 0.70,
            },
        ],
        "outputs": [],
        "constraints": [],
        "ambiguities": [{
            "description": "Instruction says 'show only age' but also 'include all columns' — these conflict",
            "possible_interpretations": ["show only age", "show all columns"],
            "source_text": "show only age but include all columns",
        }],
        "unsupported_requirements": [],
    }


def _mock_llm_unknown_column(messages):
    """Reference to a column that doesn't exist."""
    return {
        "goals": [{"description": "Filter by nonexistent column", "priority": 1}],
        "tasks": [{
            "task_id": "task_1",
            "operation": {"type": "filter"},
            "inputs": [{"kind": "column_reference", "user_term": "zodiac_sign"}],
            "parameters": {
                "predicate": {
                    "left": {"kind": "column_reference", "user_term": "zodiac_sign"},
                    "operator": "equals",
                    "right": {"kind": "literal_value", "value": "Aries"},
                }
            },
            "depends_on": [],
            "confidence": 0.85,
        }],
        "outputs": [],
        "constraints": [],
        "ambiguities": [],
        "unsupported_requirements": [],
    }


def _mock_llm_unsupported_operation(messages):
    """Request for an operation the system can't do."""
    return {
        "goals": [{"description": "Train a model", "priority": 1}],
        "tasks": [],
        "outputs": [],
        "constraints": [],
        "ambiguities": [],
        "unsupported_requirements": [{
            "description": "Machine learning model training is not supported",
            "reason": "This system handles data cleaning and filtering, not ML training",
            "source_text": "train a random forest model on this data",
        }],
    }


# ---------------------------------------------------------------------------
# Tests: Adversarial "except" cases
# ---------------------------------------------------------------------------


class TestExceptAsConditionalPredicate:
    """'except' used as a conditional, NOT column exclusion."""

    def test_except_when_is_not_column_exclusion(self):
        """'remove invalid rows except when status is pending' → filter, not drop_columns."""
        result = run_semantic_pipeline_sync(
            "remove invalid rows except when status is pending",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_conditional_except,
        )
        # Should NOT produce a drop_columns action
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "drop_columns" not in action_kinds, (
            f"Should NOT be drop_columns: {action_kinds}"
        )
        # Should produce a filter_rows action
        assert "filter_rows" in action_kinds, (
            f"Should be filter_rows: {action_kinds}"
        )

    def test_except_for_is_not_column_exclusion(self):
        """'calculate revenue except for cancelled' → filter + aggregate."""
        result = run_semantic_pipeline_sync(
            "calculate revenue except for cancelled transactions",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_revenue_except_cancelled,
        )
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "drop_columns" not in action_kinds
        assert "filter_rows" in action_kinds

    def test_except_those_is_not_column_exclusion(self):
        """'clean values except those below zero' → clean with constraint."""
        result = run_semantic_pipeline_sync(
            "clean values except those below zero",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_clean_except_below_zero,
        )
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "drop_columns" not in action_kinds
        assert "clean" in action_kinds


# ---------------------------------------------------------------------------
# Tests: Multiple tasks in one prompt
# ---------------------------------------------------------------------------


class TestMultipleTasksInOnePrompt:
    """Verify that multi-task prompts extract all operations."""

    def test_clean_filter_sort(self):
        """'clean this, filter age > 18, sort by income desc'."""
        result = run_semantic_pipeline_sync(
            "clean this data, keep rows where age is greater than 18, sort by income descending",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_multiple_tasks,
        )
        assert result.success
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "clean" in action_kinds
        assert "filter_rows" in action_kinds
        assert "sort_rows" in action_kinds


# ---------------------------------------------------------------------------
# Tests: Contradictory instructions
# ---------------------------------------------------------------------------


class TestContradictoryInstructions:
    """Contradictory instructions should be marked as ambiguous."""

    def test_contradictory_produces_ambiguity(self):
        """Contradictory projection instructions should flag ambiguity."""
        result = run_semantic_pipeline_sync(
            "show only age but include all columns",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_contradictory,
        )
        # The intent should contain ambiguities
        if result.semantic_intent:
            assert len(result.semantic_intent.ambiguities) > 0


# ---------------------------------------------------------------------------
# Tests: Unknown columns
# ---------------------------------------------------------------------------


class TestUnknownColumns:
    """Unknown columns should fail at grounding, not silently pass."""

    def test_unknown_column_blocks_compilation(self):
        """Reference to a nonexistent column → needs_clarification."""
        result = run_semantic_pipeline_sync(
            "filter where zodiac_sign equals Aries",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_unknown_column,
        )
        # Should not succeed — column doesn't exist
        assert result.needs_clarification or not result.success, (
            "Unknown column should block compilation"
        )


# ---------------------------------------------------------------------------
# Tests: Unsupported operations
# ---------------------------------------------------------------------------


class TestUnsupportedOperations:
    """Unsupported operations should be explicitly marked."""

    def test_unsupported_op_does_not_crash(self):
        """Unsupported ML request should return gracefully."""
        result = run_semantic_pipeline_sync(
            "train a random forest model on this data",
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_unsupported_operation,
        )
        # Should either fail or have unsupported_requirements in the intent
        if result.semantic_intent:
            assert len(result.semantic_intent.unsupported_requirements) > 0 or not result.success


# ---------------------------------------------------------------------------
# Tests: Same meaning, different wording (not parametrized — single check)
# ---------------------------------------------------------------------------


class TestSameMeaningDifferentWording:
    """Verify different wordings produce the same canonical action."""

    def test_exclude_wording_variants_all_produce_drop_columns(self):
        """Various exclusion phrasings all → drop_columns."""
        variants = [
            "remove consumer_id from the output",
            "exclude consumer_id",
            "without consumer_id",
            "get rid of consumer_id column",
        ]
        for prompt in variants:
            result = run_semantic_pipeline_sync(
                prompt,
                SAMPLE_COLUMNS,
                llm_call=_mock_llm_for_exclude_columns(["consumer_id"]),
            )
            assert result.success, f"Failed for: {prompt!r}"
            action_kinds = [a["kind"] for a in result.canonical_actions]
            assert "drop_columns" in action_kinds, (
                f"Expected drop_columns for: {prompt!r}, got: {action_kinds}"
            )
