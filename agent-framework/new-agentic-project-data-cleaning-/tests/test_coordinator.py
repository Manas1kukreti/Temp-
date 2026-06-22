"""Unit tests for IntentResolutionCoordinator.

Validates operation classification finalization, boolean scope validation,
clarification patch application, and ambiguity preservation.

Requirements: 2.1, 2.2, 21.3
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.models.draft import (
    AmbiguityMarker,
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
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.pipeline.coordinator import (
    IntentResolutionCoordinator,
    OperationAmbiguityError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance(start: int = 0, end: int = 5, text: str = "hello"):
    """Create a minimal valid PromptSpanProvenance."""
    return PromptSpanProvenance(
        type="prompt_span",
        start_offset=start,
        end_offset=end,
        source_text=text,
    )


def _make_column_ref(text: str = "amount", resolved: str | None = None):
    """Create a minimal SemanticColumnReference."""
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=ReferenceKind.EXPLICIT_NAME,
        resolved_column=resolved,
        provenance=[_make_provenance()],
    )


def _make_simple_project_draft() -> SemanticIntentDraft:
    """Create a simple draft with one project action and no ambiguity."""
    return SemanticIntentDraft(
        raw_prompt="show me the amount column",
        actions=[
            ProjectAction(
                type="project",
                columns=[_make_column_ref("amount")],
                provenance=[_make_provenance(0, 25, "show me the amount column")],
            )
        ],
        ambiguities=[],
    )


def _make_filter_draft() -> SemanticIntentDraft:
    """Create a draft with a filter action and logical groups."""
    return SemanticIntentDraft(
        raw_prompt="filter rows where amount > 100 and status = active",
        actions=[
            FilterAction(
                type="filter",
                logical_groups=[
                    LogicalGroup(
                        operator="and",
                        predicates=[
                            UnresolvedPredicate(
                                field_ref=_make_column_ref("amount"),
                                operator="gt",
                                value=100,
                                provenance=[_make_provenance(0, 30, "amount > 100")],
                            ),
                            UnresolvedPredicate(
                                field_ref=_make_column_ref("status"),
                                operator="eq",
                                value="active",
                                provenance=[_make_provenance(31, 50, "status = active")],
                            ),
                        ],
                        provenance=[_make_provenance(0, 50, "amount > 100 and status = active")],
                    )
                ],
                provenance=[_make_provenance(0, 50, "filter rows where amount > 100 and status = active")],
            )
        ],
        ambiguities=[],
    )


def _make_ambiguous_draft() -> SemanticIntentDraft:
    """Create a draft with operation-type ambiguity markers."""
    return SemanticIntentDraft(
        raw_prompt="remove the amount column",
        actions=[
            ProjectAction(
                type="project",
                columns=[_make_column_ref("amount")],
                provenance=[_make_provenance(0, 24, "remove the amount column")],
            )
        ],
        ambiguities=[
            AmbiguityMarker(
                element_path="actions[0].type",
                candidates=["project", "drop"],
                provenance=[_make_provenance(0, 6, "remove")],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Tests: Basic resolution (no ambiguity)
# ---------------------------------------------------------------------------


class TestBasicResolution:
    """Test coordinator with unambiguous drafts."""

    def test_resolve_single_action_no_ambiguity(self):
        """Single action with no ambiguity markers finalizes as-is."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_simple_project_draft()

        result = coordinator.resolve(draft)

        # Should produce a new revision
        assert result.draft_revision == draft.draft_revision + 1
        # Actions preserved
        assert len(result.actions) == 1
        assert result.actions[0].type == "project"
        # Resolution history should have coordinator's record
        assert len(result.resolution_history) > 0
        last_record = result.resolution_history[-1]
        assert last_record.decision_owner == "IntentResolutionCoordinator"
        assert last_record.stage == "intent_resolution"

    def test_resolve_filter_with_valid_logical_groups(self):
        """Filter action with well-formed logical groups passes validation."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_filter_draft()

        result = coordinator.resolve(draft)

        assert result.draft_revision == draft.draft_revision + 1
        assert len(result.actions) == 1
        assert result.actions[0].type == "filter"

    def test_resolve_preserves_original_draft_immutability(self):
        """Original draft is not mutated by resolve."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_simple_project_draft()
        original_revision = draft.draft_revision

        coordinator.resolve(draft)

        assert draft.draft_revision == original_revision


# ---------------------------------------------------------------------------
# Tests: Operation ambiguity handling
# ---------------------------------------------------------------------------


class TestOperationAmbiguity:
    """Test coordinator behavior with operation-type ambiguities."""

    def test_raises_operation_ambiguity_error_when_unresolved(self):
        """Unresolved operation ambiguity raises OperationAmbiguityError."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_ambiguous_draft()

        with pytest.raises(OperationAmbiguityError) as exc_info:
            coordinator.resolve(draft)

        assert "actions[0].type" in exc_info.value.ambiguous_paths
        assert "project" in exc_info.value.candidates
        assert "drop" in exc_info.value.candidates

    def test_resolved_ambiguity_via_clarification_patch(self):
        """Ambiguity resolved by clarification patch succeeds."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_ambiguous_draft()

        # Patch that removes the ambiguity marker
        patches = [
            SemanticPatch(
                operation=PatchOp.REMOVE,
                path="ambiguities[0]",
                reason="User clarified operation is drop",
                source_failure="operation_ambiguity",
            )
        ]

        result = coordinator.resolve(draft, clarification_patches=patches)

        # Should succeed (ambiguity was patched out)
        assert result.draft_revision > draft.draft_revision
        # Ambiguities list should be empty after patch
        assert len(result.ambiguities) == 0

    def test_non_operation_ambiguity_does_not_block(self):
        """Ambiguity markers for columns (not operation type) don't block."""
        coordinator = IntentResolutionCoordinator()
        draft = SemanticIntentDraft(
            raw_prompt="show me the data",
            actions=[
                ProjectAction(
                    type="project",
                    columns=[_make_column_ref("data")],
                    provenance=[_make_provenance(0, 16, "show me the data")],
                )
            ],
            ambiguities=[
                AmbiguityMarker(
                    element_path="actions[0].columns[0].resolved_column",
                    candidates=["amount", "total", "balance"],
                    provenance=[_make_provenance(12, 16, "data")],
                )
            ],
        )

        # Column ambiguity is not the coordinator's concern — should pass
        result = coordinator.resolve(draft)
        assert result.draft_revision == draft.draft_revision + 1


# ---------------------------------------------------------------------------
# Tests: Logical group validation (boolean scope)
# ---------------------------------------------------------------------------


class TestBooleanScopeValidation:
    """Test coordinator's boolean scope validation (Req 2.2)."""

    def test_valid_and_group_passes(self):
        """Valid AND logical group passes validation."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_filter_draft()

        result = coordinator.resolve(draft)
        assert result.actions[0].type == "filter"

    def test_valid_or_group_passes(self):
        """Valid OR logical group passes validation."""
        coordinator = IntentResolutionCoordinator()
        draft = SemanticIntentDraft(
            raw_prompt="filter where status is active or pending",
            actions=[
                FilterAction(
                    type="filter",
                    logical_groups=[
                        LogicalGroup(
                            operator="or",
                            predicates=[
                                UnresolvedPredicate(
                                    field_ref=_make_column_ref("status"),
                                    operator="eq",
                                    value="active",
                                    provenance=[_make_provenance(0, 20, "status is active")],
                                ),
                                UnresolvedPredicate(
                                    field_ref=_make_column_ref("status"),
                                    operator="eq",
                                    value="pending",
                                    provenance=[_make_provenance(21, 40, "or pending")],
                                ),
                            ],
                            provenance=[_make_provenance(0, 40, "status is active or pending")],
                        )
                    ],
                    provenance=[_make_provenance(0, 40, "filter where status is active or pending")],
                )
            ],
        )

        result = coordinator.resolve(draft)
        assert result.actions[0].type == "filter"
        # Boolean scope (OR grouping) preserved
        assert result.actions[0].logical_groups[0].operator == "or"


# ---------------------------------------------------------------------------
# Tests: Resolution record (audit trail)
# ---------------------------------------------------------------------------


class TestResolutionRecord:
    """Test that the coordinator records its decisions."""

    def test_resolution_record_added_on_success(self):
        """Successful resolution adds audit record to resolution_history."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_simple_project_draft()
        assert len(draft.resolution_history) == 0

        result = coordinator.resolve(draft)

        assert len(result.resolution_history) == 1
        record = result.resolution_history[0]
        assert record.decision_owner == "IntentResolutionCoordinator"
        assert record.stage == "intent_resolution"
        assert record.resolution == "operation_classification_finalized"
        assert record.confidence == 1.0

    def test_resolution_evidence_includes_action_info(self):
        """Resolution record evidence includes action count and types."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_filter_draft()

        result = coordinator.resolve(draft)

        record = result.resolution_history[-1]
        assert "action_count=1" in record.evidence
        assert any("filter" in e for e in record.evidence)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test coordinator edge case behavior."""

    def test_multiple_actions_no_ambiguity(self):
        """Multiple unambiguous actions finalize normally."""
        coordinator = IntentResolutionCoordinator()
        draft = SemanticIntentDraft(
            raw_prompt="sort by amount and show only name",
            actions=[
                ProjectAction(
                    type="project",
                    columns=[_make_column_ref("name")],
                    provenance=[_make_provenance(20, 34, "show only name")],
                ),
                FilterAction(
                    type="filter",
                    logical_groups=[
                        LogicalGroup(
                            operator="and",
                            predicates=[
                                UnresolvedPredicate(
                                    field_ref=_make_column_ref("amount"),
                                    operator="gt",
                                    value=0,
                                    provenance=[_make_provenance(0, 14, "sort by amount")],
                                ),
                            ],
                            provenance=[_make_provenance(0, 14, "sort by amount")],
                        )
                    ],
                    provenance=[_make_provenance(0, 14, "sort by amount")],
                ),
            ],
        )

        result = coordinator.resolve(draft)
        assert result.draft_revision == draft.draft_revision + 1
        assert len(result.actions) == 2

    def test_empty_clarification_patches_is_noop(self):
        """Passing empty clarification patches list doesn't affect draft."""
        coordinator = IntentResolutionCoordinator()
        draft = _make_simple_project_draft()

        result = coordinator.resolve(draft, clarification_patches=[])

        assert result.draft_revision == draft.draft_revision + 1
