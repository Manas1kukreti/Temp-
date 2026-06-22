"""Unit tests for the Canonicalizer pipeline boundary enforcer.

Tests verify that the Canonicalizer:
1. Rejects drafts with resolution_status != RESOLVED
2. Rejects drafts with unresolved ambiguities
3. Rejects drafts with unresolved column references
4. Rejects drafts without data_snapshot_ref
5. Rejects drafts without resolution_origin
6. Accepts fully-resolved drafts and delegates to CanonicalIntent.from_resolved_draft()

Requirements: 21.2, 21.4, 6.2
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from finflow_agent.models.canonical import CanonicalIntent, CanonicalizeError
from finflow_agent.models.draft import (
    AmbiguityMarker,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    ReferenceKind,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
    UnresolvedPredicate,
)
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.models.snapshot import DataSnapshotRef
from finflow_agent.pipeline.canonicalizer import Canonicalizer


def _make_provenance():
    """Helper: create a minimal valid PromptSpanProvenance."""
    return PromptSpanProvenance(
        start_offset=0, end_offset=5, source_text="hello"
    )


def _make_resolved_ref(text: str = "amount", column: str = "amount") -> SemanticColumnReference:
    """Helper: create a resolved SemanticColumnReference."""
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=ReferenceKind.EXPLICIT_NAME,
        resolved_column=column,
        confidence=0.95,
        provenance=[_make_provenance()],
    )


def _make_unresolved_ref(text: str = "amount") -> SemanticColumnReference:
    """Helper: create an unresolved SemanticColumnReference (resolved_column=None)."""
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=ReferenceKind.SEMANTIC_CONCEPT,
        resolved_column=None,
        confidence=None,
        provenance=[_make_provenance()],
    )


def _make_snapshot_ref() -> DataSnapshotRef:
    """Helper: create a valid DataSnapshotRef."""
    return DataSnapshotRef(
        file_id="file-123",
        content_hash="sha256:abc123",
        byte_size=1024,
        storage_version="1",
        profile_id="profile-456",
        structural_schema_fingerprint="fp-struct",
        profile_fingerprint="fp-profile",
    )


def _make_fully_resolved_draft() -> SemanticIntentDraft:
    """Helper: create a fully-resolved draft that should pass canonicalization."""
    return SemanticIntentDraft(
        raw_prompt="show me the amount column",
        actions=[
            ProjectAction(
                columns=[_make_resolved_ref("amount", "amount")],
                provenance=[_make_provenance()],
            ),
        ],
        ambiguities=[],
        resolution_status=ResolutionStatus.RESOLVED,
        resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
        data_snapshot_ref=_make_snapshot_ref(),
    )


class TestCanonicalizerSuccess:
    """Tests for successful canonicalization."""

    def test_fully_resolved_draft_produces_canonical_intent(self):
        canonicalizer = Canonicalizer()
        draft = _make_fully_resolved_draft()

        result = canonicalizer.canonicalize(draft)

        assert isinstance(result, CanonicalIntent)
        assert result.resolution_status == "resolved"
        assert result.source_draft_id == draft.draft_id
        assert result.source_draft_revision == draft.draft_revision

    def test_canonical_intent_contains_resolved_actions(self):
        canonicalizer = Canonicalizer()
        draft = _make_fully_resolved_draft()

        result = canonicalizer.canonicalize(draft)

        assert len(result.actions) == 1
        assert result.actions[0].type == "project"
        assert result.actions[0].columns == ["amount"]

    def test_canonical_intent_preserves_data_snapshot_ref(self):
        canonicalizer = Canonicalizer()
        draft = _make_fully_resolved_draft()

        result = canonicalizer.canonicalize(draft)

        assert result.data_snapshot_ref == draft.data_snapshot_ref

    def test_canonical_intent_preserves_resolution_origin(self):
        canonicalizer = Canonicalizer()
        draft = _make_fully_resolved_draft()

        result = canonicalizer.canonicalize(draft)

        assert result.resolution_origin == ResolutionOrigin.AUTOMATIC_GROUNDING


class TestCanonicalizerResolutionStatusCheck:
    """Tests for resolution_status != RESOLVED rejection."""

    @pytest.mark.parametrize(
        "status",
        [
            ResolutionStatus.PENDING,
            ResolutionStatus.NEEDS_CLARIFICATION,
            ResolutionStatus.INTERPRETATION_FAILED,
            ResolutionStatus.UNSUPPORTED,
            ResolutionStatus.INVALID,
        ],
    )
    def test_rejects_non_resolved_status(self, status):
        canonicalizer = Canonicalizer()
        # Construct the draft directly with the desired non-resolved status
        draft = SemanticIntentDraft(
            raw_prompt="show me the amount column",
            actions=[
                ProjectAction(
                    columns=[_make_resolved_ref("amount", "amount")],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=status,
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            data_snapshot_ref=_make_snapshot_ref(),
        )

        with pytest.raises(CanonicalizeError, match="resolution_status"):
            canonicalizer.canonicalize(draft)


class TestCanonicalizerAmbiguityCheck:
    """Tests for unresolved ambiguity rejection."""

    def test_rejects_draft_with_ambiguities(self):
        canonicalizer = Canonicalizer()
        draft_data = _make_fully_resolved_draft().model_dump()
        draft_data["ambiguities"] = [
            {
                "element_path": "actions[0].columns[0]",
                "candidates": ["amount", "total_amount"],
                "provenance": [
                    {"type": "prompt_span", "start_offset": 0, "end_offset": 6, "source_text": "amount"}
                ],
            }
        ]
        draft = SemanticIntentDraft.model_validate(draft_data)

        with pytest.raises(CanonicalizeError, match="ambiguity"):
            canonicalizer.canonicalize(draft)


class TestCanonicalizerColumnResolutionCheck:
    """Tests for unresolved column reference rejection."""

    def test_rejects_unresolved_project_column(self):
        canonicalizer = Canonicalizer()
        draft = SemanticIntentDraft(
            raw_prompt="show me the amount column",
            actions=[
                ProjectAction(
                    columns=[_make_unresolved_ref("amount")],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            data_snapshot_ref=_make_snapshot_ref(),
        )

        with pytest.raises(CanonicalizeError, match="unresolved column"):
            canonicalizer.canonicalize(draft)

    def test_rejects_unresolved_filter_column(self):
        canonicalizer = Canonicalizer()
        draft = SemanticIntentDraft(
            raw_prompt="filter where amount > 100",
            actions=[
                FilterAction(
                    logical_groups=[
                        LogicalGroup(
                            operator="and",
                            predicates=[
                                UnresolvedPredicate(
                                    field_ref=_make_unresolved_ref("amount"),
                                    operator="gt",
                                    value=100,
                                    provenance=[_make_provenance()],
                                ),
                            ],
                            provenance=[_make_provenance()],
                        ),
                    ],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            data_snapshot_ref=_make_snapshot_ref(),
        )

        with pytest.raises(CanonicalizeError, match="unresolved column"):
            canonicalizer.canonicalize(draft)

    def test_rejects_unresolved_sort_key(self):
        canonicalizer = Canonicalizer()
        draft = SemanticIntentDraft(
            raw_prompt="sort by amount",
            actions=[
                SortAction(
                    keys=[_make_unresolved_ref("amount")],
                    directions=["asc"],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            data_snapshot_ref=_make_snapshot_ref(),
        )

        with pytest.raises(CanonicalizeError, match="unresolved column"):
            canonicalizer.canonicalize(draft)


class TestCanonicalizerDataSnapshotCheck:
    """Tests for missing data_snapshot_ref rejection."""

    def test_rejects_missing_data_snapshot_ref(self):
        canonicalizer = Canonicalizer()
        draft = SemanticIntentDraft(
            raw_prompt="show me the amount column",
            actions=[
                ProjectAction(
                    columns=[_make_resolved_ref("amount", "amount")],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_origin=ResolutionOrigin.AUTOMATIC_GROUNDING,
            data_snapshot_ref=None,
        )

        with pytest.raises(CanonicalizeError, match="data_snapshot_ref"):
            canonicalizer.canonicalize(draft)


class TestCanonicalizerResolutionOriginCheck:
    """Tests for missing resolution_origin rejection."""

    def test_rejects_missing_resolution_origin(self):
        canonicalizer = Canonicalizer()
        draft = SemanticIntentDraft(
            raw_prompt="show me the amount column",
            actions=[
                ProjectAction(
                    columns=[_make_resolved_ref("amount", "amount")],
                    provenance=[_make_provenance()],
                ),
            ],
            ambiguities=[],
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_origin=None,
            data_snapshot_ref=_make_snapshot_ref(),
        )

        with pytest.raises(CanonicalizeError, match="resolution_origin"):
            canonicalizer.canonicalize(draft)
