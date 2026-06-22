"""Unit tests for patch application logic.

Validates that SemanticPatch operations produce a new draft revision (N → N+1)
while leaving the original draft immutable.

Requirements: 4.4, 15.4, 17.1
"""

import os
import sys
import copy

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.models.draft import (
    DraftAction,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    ReferenceKind,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    UnresolvedPredicate,
)
from finflow_agent.models.patches import PatchOp, SemanticPatch
from finflow_agent.models.provenance import PromptSpanProvenance, SchemaEvidenceProvenance
from finflow_agent.pipeline.patch_applicator import (
    PatchApplicationError,
    apply_patches,
    _parse_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance():
    """Create a minimal valid PromptSpanProvenance."""
    return PromptSpanProvenance(
        type="prompt_span",
        start_offset=0,
        end_offset=5,
        source_text="hello",
    )


def _make_column_ref(text: str = "amount", resolved: str | None = None):
    """Create a minimal SemanticColumnReference."""
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=ReferenceKind.EXPLICIT_NAME,
        resolved_column=resolved,
        provenance=[_make_provenance()],
    )


def _make_draft(revision: int = 1) -> SemanticIntentDraft:
    """Create a minimal valid SemanticIntentDraft for testing."""
    return SemanticIntentDraft(
        raw_prompt="show me the amount column",
        draft_revision=revision,
        actions=[
            ProjectAction(
                type="project",
                columns=[_make_column_ref("amount")],
                provenance=[_make_provenance()],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Path parsing tests
# ---------------------------------------------------------------------------


class TestPathParsing:
    """Tests for JSON-path-like path parsing."""

    def test_simple_field(self):
        assert _parse_path("actions") == ["actions"]

    def test_indexed_field(self):
        assert _parse_path("actions[0]") == ["actions", 0]

    def test_nested_path(self):
        assert _parse_path("actions[0].columns[1].resolved_column") == [
            "actions", 0, "columns", 1, "resolved_column"
        ]

    def test_ambiguities_path(self):
        assert _parse_path("ambiguities[0]") == ["ambiguities", 0]

    def test_empty_path_raises(self):
        with pytest.raises(PatchApplicationError, match="Empty patch path"):
            _parse_path("")

    def test_invalid_segment_raises(self):
        with pytest.raises(PatchApplicationError, match="Invalid path segment"):
            _parse_path("123invalid")


# ---------------------------------------------------------------------------
# Revision increment tests (Req 17.1)
# ---------------------------------------------------------------------------


class TestRevisionIncrement:
    """Tests that patch application produces N+1 revision."""

    def test_revision_increments_by_one(self):
        """Applying patches produces draft at revision N+1."""
        draft = _make_draft(revision=1)
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[0].columns[0].resolved_column",
            value="physical_amount",
            reason="Grounding resolved column",
            source_failure="unresolved_column",
            provenance=[],
        )

        result = apply_patches(draft, [patch])

        assert result.draft_revision == 2
        assert draft.draft_revision == 1  # Original unchanged

    def test_multiple_revision_increments(self):
        """Successive applications increment revision each time."""
        draft = _make_draft(revision=3)
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="resolution_status",
            value="resolved",
            reason="All references resolved",
            source_failure="status_update",
            provenance=[],
        )

        result = apply_patches(draft, [patch])
        assert result.draft_revision == 4

    def test_empty_patch_list_still_increments(self):
        """Even an empty patch list produces a new revision."""
        draft = _make_draft(revision=5)
        result = apply_patches(draft, [])
        assert result.draft_revision == 6


# ---------------------------------------------------------------------------
# Immutability tests (Req 17.1)
# ---------------------------------------------------------------------------


class TestOriginalImmutability:
    """Tests that the original draft is never mutated."""

    def test_original_unchanged_after_replace(self):
        """Original draft is not modified by REPLACE patch."""
        draft = _make_draft(revision=1)
        original_dump = draft.model_dump(mode="json")

        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[0].columns[0].resolved_column",
            value="resolved_col",
            reason="test",
            source_failure="test_failure",
            provenance=[],
        )

        _ = apply_patches(draft, [patch])

        assert draft.model_dump(mode="json") == original_dump

    def test_original_unchanged_after_add(self):
        """Original draft is not modified by ADD patch."""
        draft = _make_draft(revision=2)
        original_actions_count = len(draft.actions)

        patch = SemanticPatch(
            operation=PatchOp.ADD,
            path="ambiguities",
            value={
                "element_path": "actions[0].columns[0]",
                "candidates": ["col_a", "col_b"],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 6,
                        "source_text": "amount",
                    }
                ],
            },
            reason="test add",
            source_failure="test_failure",
            provenance=[],
        )

        result = apply_patches(draft, [patch])

        # Original still has empty ambiguities
        assert len(draft.ambiguities) == 0
        # New draft has the added ambiguity
        assert len(result.ambiguities) == 1


# ---------------------------------------------------------------------------
# Operation tests (ADD, REPLACE, REMOVE)
# ---------------------------------------------------------------------------


class TestPatchOperations:
    """Tests for individual patch operations."""

    def test_replace_resolved_column(self):
        """REPLACE sets value at path."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[0].columns[0].resolved_column",
            value="physical_amount",
            reason="Column grounded",
            source_failure="unresolved_ref",
            provenance=[],
        )

        result = apply_patches(draft, [patch])
        # Access the resolved column from the project action
        action = result.actions[0]
        assert action.type == "project"
        assert action.columns[0].resolved_column == "physical_amount"

    def test_add_to_ambiguities_list(self):
        """ADD appends to a list field."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.ADD,
            path="ambiguities",
            value={
                "element_path": "actions[0]",
                "candidates": ["filter", "project"],
                "provenance": [
                    {
                        "type": "prompt_span",
                        "start_offset": 0,
                        "end_offset": 4,
                        "source_text": "show",
                    }
                ],
            },
            reason="Operation ambiguous",
            source_failure="ambiguity_detected",
            provenance=[],
        )

        result = apply_patches(draft, [patch])
        assert len(result.ambiguities) == 1
        assert result.ambiguities[0].candidates == ["filter", "project"]

    def test_remove_from_list(self):
        """REMOVE deletes element at index from a list."""
        # Create draft with two columns in the project action
        draft = SemanticIntentDraft(
            raw_prompt="show amount and name",
            actions=[
                ProjectAction(
                    type="project",
                    columns=[
                        _make_column_ref("amount"),
                        _make_column_ref("name"),
                    ],
                    provenance=[_make_provenance()],
                )
            ],
        )

        patch = SemanticPatch(
            operation=PatchOp.REMOVE,
            path="actions[0].columns[1]",
            reason="Duplicate column",
            source_failure="duplicate_detection",
            provenance=[],
        )

        result = apply_patches(draft, [patch])
        assert len(result.actions[0].columns) == 1
        assert result.actions[0].columns[0].reference_text == "amount"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestPatchErrors:
    """Tests for error cases in patch application."""

    def test_invalid_path_raises_error(self):
        """Invalid path raises PatchApplicationError."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="nonexistent_field.foo",
            value="bar",
            reason="test",
            source_failure="test",
            provenance=[],
        )

        with pytest.raises(PatchApplicationError):
            apply_patches(draft, [patch])

    def test_out_of_range_index_on_replace_raises(self):
        """Out-of-range index on REPLACE raises PatchApplicationError."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[99]",
            value={"type": "project", "columns": [], "provenance": []},
            reason="test",
            source_failure="test",
            provenance=[],
        )

        with pytest.raises(PatchApplicationError):
            apply_patches(draft, [patch])

    def test_add_without_value_raises(self):
        """ADD with None value raises PatchApplicationError."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.ADD,
            path="ambiguities",
            value=None,
            reason="test",
            source_failure="test",
            provenance=[],
        )

        with pytest.raises(PatchApplicationError, match="ADD operation requires a value"):
            apply_patches(draft, [patch])

    def test_replace_nonexistent_key_raises(self):
        """REPLACE on non-existent dict key raises PatchApplicationError."""
        draft = _make_draft()
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[0].nonexistent_key",
            value="bar",
            reason="test",
            source_failure="test",
            provenance=[],
        )

        with pytest.raises(PatchApplicationError):
            apply_patches(draft, [patch])


# ---------------------------------------------------------------------------
# ProvenanceRef in patches test (Req 15.4)
# ---------------------------------------------------------------------------


class TestPatchProvenance:
    """Tests that patches include ProvenanceRef for new/modified elements."""

    def test_patch_carries_provenance(self):
        """Patches adding/modifying elements can include ProvenanceRef (Req 15.4)."""
        draft = _make_draft()
        schema_prov = SchemaEvidenceProvenance(
            type="schema_evidence",
            schema_fingerprint="abc123",
            column="amount",
            evidence=["numeric column", "matches reference"],
        )
        patch = SemanticPatch(
            operation=PatchOp.REPLACE,
            path="actions[0].columns[0].resolved_column",
            value="physical_amount",
            reason="Schema evidence grounding",
            source_failure="unresolved_ref",
            provenance=[schema_prov],
        )

        # The patch itself carries provenance
        assert len(patch.provenance) == 1
        assert patch.provenance[0].type == "schema_evidence"

        # Application still works
        result = apply_patches(draft, [patch])
        assert result.actions[0].columns[0].resolved_column == "physical_amount"
