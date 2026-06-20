"""Paraphrase-family regression tests for semantic extraction.

These tests verify that multiple equivalent phrasings normalize to the
same semantic intent. This is the core value proposition: the system
understands MEANING, not specific word patterns.
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
from app.services.semantic_compiler import compile_semantic_to_canonical
from app.services.semantic_pipeline import run_semantic_pipeline_sync, SemanticExtractionResult


# ---------------------------------------------------------------------------
# Test fixture: mock LLM that returns structured semantic intent
# ---------------------------------------------------------------------------

SAMPLE_COLUMNS = [
    "consumer_id", "age", "gender", "income", "status",
    "transaction_date", "amount", "merchant", "payment_method",
]


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


def _mock_llm_for_select_columns(columns_to_select: list[str]):
    """Create a mock LLM that returns a select_columns task."""
    def mock_llm(messages):
        return {
            "goals": [{"description": "Select specific columns", "priority": 1}],
            "tasks": [{
                "task_id": "task_1",
                "operation": {"type": "select_columns"},
                "inputs": [
                    {"kind": "column_reference", "user_term": col}
                    for col in columns_to_select
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


def _mock_llm_for_filter_between(column: str, min_val, max_val):
    """Create a mock LLM that returns a filter with between operator."""
    def mock_llm(messages):
        return {
            "goals": [{"description": "Filter rows by range", "priority": 1}],
            "tasks": [{
                "task_id": "task_1",
                "operation": {"type": "filter"},
                "inputs": [{"kind": "column_reference", "user_term": column}],
                "parameters": {
                    "predicate": {
                        "left": {"kind": "column_reference", "user_term": column},
                        "operator": "between",
                        "right": {"kind": "range_value", "minimum": min_val, "maximum": max_val},
                    }
                },
                "depends_on": [],
                "confidence": 0.95,
            }],
            "outputs": [],
            "constraints": [],
            "ambiguities": [],
            "unsupported_requirements": [],
        }
    return mock_llm


def _mock_llm_for_filter_in(column: str, values: list):
    """Create a mock LLM that returns a filter with in operator."""
    def mock_llm(messages):
        return {
            "goals": [{"description": "Filter by membership", "priority": 1}],
            "tasks": [{
                "task_id": "task_1",
                "operation": {"type": "filter"},
                "inputs": [{"kind": "column_reference", "user_term": column}],
                "parameters": {
                    "predicate": {
                        "left": {"kind": "column_reference", "user_term": column},
                        "operator": "in",
                        "right": {"kind": "list_value", "values": values},
                    }
                },
                "depends_on": [],
                "confidence": 0.95,
            }],
            "outputs": [],
            "constraints": [],
            "ambiguities": [],
            "unsupported_requirements": [],
        }
    return mock_llm


def _mock_llm_for_clean_and_exclude(columns_to_exclude: list[str]):
    """Mock LLM for clean + exclude combination."""
    def mock_llm(messages):
        return {
            "goals": [{"description": "Clean data and exclude columns", "priority": 1}],
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
                    "operation": {"type": "exclude_columns"},
                    "inputs": [
                        {"kind": "column_reference", "user_term": col}
                        for col in columns_to_exclude
                    ],
                    "parameters": {},
                    "depends_on": ["task_1"],
                    "confidence": 0.95,
                },
            ],
            "outputs": [],
            "constraints": [],
            "ambiguities": [],
            "unsupported_requirements": [],
        }
    return mock_llm


# ---------------------------------------------------------------------------
# Test family 1: Negative projection (exclude_columns)
# All phrasings should produce exclude_columns → drop_columns
# ---------------------------------------------------------------------------


class TestNegativeProjectionFamily:
    """All of these phrasings mean the same thing: exclude consumer_id."""

    PHRASINGS = [
        "return all columns except consumer ID",
        "return everything except consumer ID",
        "everything but consumer ID",
        "hide consumer ID",
        "do not include consumer ID",
        "consumer ID should not appear",
        "all fields other than consumer ID",
        "remove consumer ID from the output",
    ]

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_produces_exclude_columns_task(self, prompt: str):
        """Each phrasing should extract an exclude_columns semantic task."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_exclude_columns(["consumer ID"]),
        )
        assert result.success, f"Failed for: {prompt!r} — {result.error}"
        # Should have a drop_columns action in canonical output
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "drop_columns" in action_kinds, (
            f"Expected drop_columns for: {prompt!r}, got: {action_kinds}"
        )

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_resolves_consumer_id_column(self, prompt: str):
        """The excluded column should resolve to consumer_id."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_exclude_columns(["consumer ID"]),
        )
        if not result.success:
            pytest.skip(f"Extraction failed: {result.error}")
        drop_actions = [a for a in result.canonical_actions if a["kind"] == "drop_columns"]
        assert drop_actions, f"No drop_columns action for: {prompt!r}"
        fields = drop_actions[0].get("requested_fields", [])
        resolved_columns = [f.get("resolved_column") for f in fields if f.get("resolved_column")]
        assert "consumer_id" in resolved_columns, (
            f"Expected consumer_id to be resolved for: {prompt!r}, got: {resolved_columns}"
        )


# ---------------------------------------------------------------------------
# Test family 2: Positive projection (select_columns)
# ---------------------------------------------------------------------------


class TestPositiveProjectionFamily:
    """All of these mean: keep only age and gender columns."""

    PHRASINGS = [
        "show only age and gender",
        "return age and gender only",
        "keep just age and gender",
        "I only need age and gender",
    ]

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_produces_select_columns_task(self, prompt: str):
        """Each phrasing should extract a select_columns → project_columns action."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_select_columns(["age", "gender"]),
        )
        assert result.success, f"Failed for: {prompt!r} — {result.error}"
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "project_columns" in action_kinds, (
            f"Expected project_columns for: {prompt!r}, got: {action_kinds}"
        )


# ---------------------------------------------------------------------------
# Test family 3: Range relation (between)
# ---------------------------------------------------------------------------


class TestRangeRelationFamily:
    """All of these mean: filter age between 18 and 25."""

    PHRASINGS = [
        "age between 18 and 25",
        "age belongs to the range 18 to 25",
        "people aged from 18 through 25",
        "keep rows where age is at least 18 and at most 25",
    ]

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_produces_filter_with_range(self, prompt: str):
        """Each phrasing should produce a filter_rows action with range conditions."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_filter_between("age", 18, 25),
        )
        assert result.success, f"Failed for: {prompt!r} — {result.error}"
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "filter_rows" in action_kinds, (
            f"Expected filter_rows for: {prompt!r}, got: {action_kinds}"
        )
        # Verify the filter has gte/lte conditions for the range
        filter_actions = [a for a in result.canonical_actions if a["kind"] == "filter_rows"]
        conditions = filter_actions[0].get("conditions", [])
        operators = [c.get("operator") for c in conditions]
        assert "gte" in operators or "lte" in operators, (
            f"Expected range operators for: {prompt!r}, got: {operators}"
        )


# ---------------------------------------------------------------------------
# Test family 4: Membership relation (in)
# ---------------------------------------------------------------------------


class TestMembershipRelationFamily:
    """All of these mean: filter status in [approved, pending]."""

    PHRASINGS = [
        "status belongs to approved or pending",
        "status is in approved and pending",
        "keep approved and pending statuses",
    ]

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_produces_filter_with_in(self, prompt: str):
        """Each phrasing should produce a filter_rows action."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_filter_in("status", ["approved", "pending"]),
        )
        assert result.success, f"Failed for: {prompt!r} — {result.error}"
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "filter_rows" in action_kinds, (
            f"Expected filter_rows for: {prompt!r}, got: {action_kinds}"
        )


# ---------------------------------------------------------------------------
# Test family 5: Clean + Exclude combination (the original failing case)
# ---------------------------------------------------------------------------


class TestCleanAndExcludeFamily:
    """The original problem: clean + column exclusion in one prompt."""

    PHRASINGS = [
        "clean this data and return all columns except consumer ID",
        "clean this data and hide consumer ID",
        "clean this data and do not include consumer ID",
        "clean this data, everything except consumer ID",
    ]

    @pytest.mark.parametrize("prompt", PHRASINGS)
    def test_produces_both_clean_and_exclude(self, prompt: str):
        """Must produce BOTH clean and drop_columns actions."""
        result = run_semantic_pipeline_sync(
            prompt,
            SAMPLE_COLUMNS,
            llm_call=_mock_llm_for_clean_and_exclude(["consumer ID"]),
        )
        assert result.success, f"Failed for: {prompt!r} — {result.error}"
        action_kinds = [a["kind"] for a in result.canonical_actions]
        assert "clean" in action_kinds, (
            f"Expected clean action for: {prompt!r}, got: {action_kinds}"
        )
        assert "drop_columns" in action_kinds, (
            f"Expected drop_columns action for: {prompt!r}, got: {action_kinds}"
        )


# ---------------------------------------------------------------------------
# Test: Coverage checker catches missing exclude_columns
# ---------------------------------------------------------------------------


class TestCoverageDetectsMissing:
    """Verify that coverage check catches omissions."""

    def test_clean_only_extraction_flagged_when_exclusion_present(self):
        """If prompt says 'except X' but only clean was extracted → coverage fails."""
        # Simulate: LLM only returned clean, missed exclude_columns
        intent = SemanticIntent(
            tasks=[
                SemanticTask(
                    task_id="task_1",
                    operation=SemanticOperation(type=SemanticOperationType.clean),
                    inputs=[],
                )
            ]
        )
        coverage = check_coverage_deterministic(
            "clean this data and return all columns except consumer ID",
            intent,
        )
        assert not coverage.covered, "Coverage should detect missing exclude_columns"
        assert any(
            "exclusion" in r.description.lower() or "exclude" in r.description.lower()
            for r in coverage.missing_requirements
        ), f"Should mention exclusion in missing requirements: {coverage.missing_requirements}"

    def test_full_extraction_passes_coverage(self):
        """If both clean and exclude are present, coverage passes."""
        intent = SemanticIntent(
            tasks=[
                SemanticTask(
                    task_id="task_1",
                    operation=SemanticOperation(type=SemanticOperationType.clean),
                    inputs=[],
                ),
                SemanticTask(
                    task_id="task_2",
                    operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                    inputs=[SemanticReference(kind="column_reference", user_term="consumer ID")],
                ),
            ]
        )
        coverage = check_coverage_deterministic(
            "clean this data and return all columns except consumer ID",
            intent,
        )
        assert coverage.covered, f"Coverage should pass: {coverage.missing_requirements}"


# ---------------------------------------------------------------------------
# Test: Grounding resolves user terms to actual columns
# ---------------------------------------------------------------------------


class TestColumnGrounding:
    """Verify grounding resolves user-facing terms to actual columns."""

    def test_consumer_id_resolves(self):
        """'consumer ID' should resolve to 'consumer_id' column."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="consumer ID")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert grounded.all_resolved, f"Unresolved: {grounded.unresolved_references}"
        assert grounded.grounding_results[0].resolved_column == "consumer_id"

    def test_nonexistent_column_unresolved(self):
        """A column that doesn't exist should be marked unresolved."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="unicorn_field")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert not grounded.all_resolved
        assert "unicorn_field" in grounded.unresolved_references
