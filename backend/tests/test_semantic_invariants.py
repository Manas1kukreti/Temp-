"""Semantic invariant tests.

These tests verify structural invariants of the semantic extraction system:
- Every user requirement must be represented
- Every semantic reference must be grounded or marked unresolved
- Every semantic task must map to a supported canonical action
- Unresolved tasks cannot compile
- Coverage failure blocks execution
"""

from __future__ import annotations

import pytest

from app.services.semantic_models import (
    CoverageResult,
    MissingRequirement,
    SemanticIntent,
    SemanticOperation,
    SemanticOperationType,
    SemanticReference,
    SemanticTask,
    SemanticGoal,
    OutputRequirement,
    UnsupportedRequirement,
)
from app.services.semantic_normalizer import normalize_semantic_intent
from app.services.semantic_coverage import check_coverage_deterministic
from app.services.semantic_grounding import ground_semantic_intent
from app.services.semantic_compiler import (
    SemanticCompilationError,
    compile_semantic_to_canonical,
)
from app.services.semantic_pipeline import run_semantic_pipeline_sync


SAMPLE_COLUMNS = [
    "consumer_id", "age", "gender", "income", "status",
    "transaction_date", "amount", "merchant",
]


# ---------------------------------------------------------------------------
# Invariant 1: Every user-requested action must be represented
# ---------------------------------------------------------------------------


class TestEveryRequirementRepresented:
    """Coverage verification must detect omissions."""

    def test_missing_filter_detected(self):
        """If prompt has filter intent but no filter task → coverage fails."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.clean),
                inputs=[],
            )]
        )
        coverage = check_coverage_deterministic(
            "clean data and filter where age greater than 30",
            intent,
        )
        assert not coverage.covered
        assert any("filter" in r.description.lower() for r in coverage.missing_requirements)

    def test_missing_clean_detected(self):
        """If prompt has clean intent but no clean task → coverage fails."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.filter),
                inputs=[SemanticReference(kind="column_reference", user_term="age")],
                parameters={"predicate": {"left": {"kind": "column_reference", "user_term": "age"}, "operator": "greater_than", "right": {"kind": "literal_value", "value": 30}}},
            )]
        )
        coverage = check_coverage_deterministic(
            "clean this data and filter where age greater than 30",
            intent,
        )
        assert not coverage.covered
        assert any("clean" in r.description.lower() for r in coverage.missing_requirements)

    def test_complete_intent_passes_coverage(self):
        """If all requirements are represented → coverage passes."""
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
            "clean this data and exclude consumer ID",
            intent,
        )
        assert coverage.covered


# ---------------------------------------------------------------------------
# Invariant 2: Every reference must be grounded or marked unresolved
# ---------------------------------------------------------------------------


class TestEveryReferenceGrounded:
    """Grounding must resolve or explicitly mark every reference."""

    def test_existing_column_resolves(self):
        """Known column resolves with high confidence."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="age")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert grounded.all_resolved
        assert grounded.grounding_results[0].resolved_column == "age"
        assert grounded.grounding_results[0].confidence >= 0.95

    def test_unknown_column_marked_unresolved(self):
        """Unknown column is explicitly marked, not silently dropped."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="nonexistent_field")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert not grounded.all_resolved
        assert "nonexistent_field" in grounded.unresolved_references

    def test_similar_column_resolves_with_lower_confidence(self):
        """Column with similar name resolves via fuzzy match."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="consumer ID")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert grounded.all_resolved
        assert grounded.grounding_results[0].resolved_column == "consumer_id"


# ---------------------------------------------------------------------------
# Invariant 3: Every task maps to a supported canonical action
# ---------------------------------------------------------------------------


class TestEveryTaskMapsToCanonical:
    """All supported semantic operations must compile to canonical actions."""

    @pytest.mark.parametrize("op_type,expected_kind", [
        (SemanticOperationType.clean, "clean"),
        (SemanticOperationType.exclude_columns, "drop_columns"),
        (SemanticOperationType.select_columns, "project_columns"),
        (SemanticOperationType.filter, "filter_rows"),
        (SemanticOperationType.sort, "sort_rows"),
        (SemanticOperationType.limit, "limit_rows"),
    ])
    def test_operation_compiles(self, op_type, expected_kind):
        """Each semantic operation type maps to its canonical action."""
        inputs = []
        parameters = {}
        if op_type in (SemanticOperationType.exclude_columns, SemanticOperationType.select_columns):
            inputs = [SemanticReference(kind="column_reference", user_term="age")]
        elif op_type == SemanticOperationType.filter:
            inputs = [SemanticReference(kind="column_reference", user_term="age")]
            parameters = {
                "predicate": {
                    "left": {"kind": "column_reference", "user_term": "age"},
                    "operator": "greater_than",
                    "right": {"kind": "literal_value", "value": 18},
                }
            }
        elif op_type == SemanticOperationType.sort:
            inputs = [SemanticReference(kind="column_reference", user_term="age")]
            parameters = {"direction": "asc"}
        elif op_type == SemanticOperationType.limit:
            parameters = {"limit": 10}

        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=op_type),
                inputs=inputs,
                parameters=parameters,
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        if grounded.all_resolved:
            compiled = compile_semantic_to_canonical(grounded)
            action_kinds = [a["kind"] for a in compiled["actions"]]
            assert expected_kind in action_kinds


# ---------------------------------------------------------------------------
# Invariant 4: Unresolved tasks cannot compile
# ---------------------------------------------------------------------------


class TestUnresolvedBlocksCompilation:
    """Unresolved column references must prevent compilation."""

    def test_unresolved_reference_raises(self):
        """Compilation fails when references are unresolved."""
        intent = SemanticIntent(
            tasks=[SemanticTask(
                task_id="task_1",
                operation=SemanticOperation(type=SemanticOperationType.exclude_columns),
                inputs=[SemanticReference(kind="column_reference", user_term="nonexistent_xyz")],
            )]
        )
        grounded = ground_semantic_intent(intent, SAMPLE_COLUMNS)
        assert not grounded.all_resolved

        with pytest.raises(SemanticCompilationError, match="unresolved"):
            compile_semantic_to_canonical(grounded)


# ---------------------------------------------------------------------------
# Invariant 5: Coverage failure blocks execution
# ---------------------------------------------------------------------------


class TestCoverageFailureBlocksExecution:
    """If coverage fails and repair fails, the pipeline returns needs_clarification."""

    def test_coverage_failure_returns_needs_clarification(self):
        """Pipeline with coverage gap → needs_clarification status."""
        # Mock LLM that only returns clean, missing the exclude
        def mock_llm_only_clean(messages):
            return {
                "goals": [],
                "tasks": [{
                    "task_id": "task_1",
                    "operation": {"type": "clean"},
                    "inputs": [],
                    "parameters": {},
                    "depends_on": [],
                    "confidence": 0.95,
                }],
                "outputs": [],
                "constraints": [],
                "ambiguities": [],
                "unsupported_requirements": [],
            }

        result = run_semantic_pipeline_sync(
            "clean this data and return all columns except consumer ID",
            SAMPLE_COLUMNS,
            llm_call=mock_llm_only_clean,
        )
        # Should fail because coverage detects the missing exclude_columns
        assert result.needs_clarification or not result.success


# ---------------------------------------------------------------------------
# Invariant 6: Normalization is idempotent
# ---------------------------------------------------------------------------


class TestNormalizationIdempotent:
    """Normalizing twice should produce the same result."""

    def test_double_normalize_same_result(self):
        """normalize(normalize(x)) == normalize(x)."""
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
                    inputs=[SemanticReference(kind="column_reference", user_term="age")],
                ),
            ]
        )
        first = normalize_semantic_intent(intent)
        second = normalize_semantic_intent(first)

        assert len(first.tasks) == len(second.tasks)
        for t1, t2 in zip(first.tasks, second.tasks):
            assert t1.operation.type == t2.operation.type
            assert len(t1.inputs) == len(t2.inputs)
