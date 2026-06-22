"""Unit tests for ClarificationDraftPatcher — draft-patching extension.

Tests cover:
- Stale revision rejection (Req 17.2)
- Duplicate idempotency key rejection (Req 17.4)
- Stage-resume policy mapping (Req 9.3)
- Producing new immutable draft revision (Req 9.2, 17.1, 17.3)
- Recording ClarificationProvenance (Req 15.5)
"""

import pytest

from finflow_agent.models.draft import (
    FilterAction,
    LogicalGroup,
    ProjectAction,
    ReferenceKind,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    UnresolvedPredicate,
)
from finflow_agent.models.provenance import PromptSpanProvenance
from finflow_agent.pipeline.clarification_service import (
    ClarificationDraftPatcher,
    ClarificationResponse,
    DuplicateIdempotencyKeyError,
    PatchType,
    StageResumeDirective,
    StaleRevisionError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_provenance(text: str = "test") -> PromptSpanProvenance:
    """Helper to create a minimal PromptSpanProvenance."""
    return PromptSpanProvenance(start_offset=0, end_offset=len(text), source_text=text)


def _make_column_ref(
    text: str = "amount",
    kind: ReferenceKind = ReferenceKind.EXPLICIT_NAME,
    resolved: str | None = None,
) -> SemanticColumnReference:
    """Helper to create a SemanticColumnReference."""
    return SemanticColumnReference(
        reference_text=text,
        reference_kind=kind,
        resolved_column=resolved,
        provenance=[_make_provenance(text)],
    )


def _make_draft(revision: int = 1) -> SemanticIntentDraft:
    """Create a minimal draft with an unresolved project action."""
    return SemanticIntentDraft(
        raw_prompt="show amount column",
        draft_revision=revision,
        actions=[
            ProjectAction(
                columns=[_make_column_ref("amount")],
                provenance=[_make_provenance("show amount column")],
            )
        ],
        extraction_provenance=[_make_provenance("show amount column")],
    )


def _make_filter_draft(revision: int = 1) -> SemanticIntentDraft:
    """Create a draft with a filter action containing an unresolved predicate."""
    return SemanticIntentDraft(
        raw_prompt="filter where amount > 100",
        draft_revision=revision,
        actions=[
            FilterAction(
                logical_groups=[
                    LogicalGroup(
                        operator="and",
                        predicates=[
                            UnresolvedPredicate(
                                field_ref=_make_column_ref("amount"),
                                operator="gt",
                                value=100,
                                provenance=[_make_provenance("amount > 100")],
                            )
                        ],
                        provenance=[_make_provenance("filter where")],
                    )
                ],
                provenance=[_make_provenance("filter where amount > 100")],
            )
        ],
        extraction_provenance=[_make_provenance("filter where amount > 100")],
    )


def _make_response(
    patch_type: PatchType = PatchType.COLUMN,
    idempotency_key: str = "key-1",
    selected_value: str = "total_amount",
) -> ClarificationResponse:
    """Create a ClarificationResponse."""
    return ClarificationResponse(
        question_id="q-001",
        response_id="r-001",
        selected_value=selected_value,
        idempotency_key=idempotency_key,
        patch_type=patch_type,
    )


# ---------------------------------------------------------------------------
# Tests: Stale Revision Rejection (Req 17.2)
# ---------------------------------------------------------------------------


class TestStaleRevisionRejection:
    """Tests for stale revision check on patch_draft."""

    def test_raises_when_expected_revision_is_lower(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft(revision=3)
        response = _make_response()

        with pytest.raises(StaleRevisionError) as exc_info:
            patcher.patch_draft(draft, response, expected_revision=2)

        assert exc_info.value.expected == 2
        assert exc_info.value.actual == 3

    def test_raises_when_expected_revision_is_higher(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft(revision=1)
        response = _make_response()

        with pytest.raises(StaleRevisionError) as exc_info:
            patcher.patch_draft(draft, response, expected_revision=5)

        assert exc_info.value.expected == 5
        assert exc_info.value.actual == 1

    def test_succeeds_when_expected_revision_matches(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft(revision=3)
        response = _make_response()

        new_draft, directive = patcher.patch_draft(draft, response, expected_revision=3)
        assert new_draft.draft_revision == 4


# ---------------------------------------------------------------------------
# Tests: Idempotency Key Enforcement (Req 17.4)
# ---------------------------------------------------------------------------


class TestIdempotencyKeyEnforcement:
    """Tests for duplicate idempotency key rejection."""

    def test_first_use_of_key_succeeds(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(idempotency_key="unique-key-1")

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)
        assert new_draft.draft_revision == 2

    def test_duplicate_key_raises_error(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response1 = _make_response(idempotency_key="dup-key")

        patcher.patch_draft(draft, response1, expected_revision=1)

        # Second attempt with same key should fail
        draft2 = _make_draft(revision=2)
        response2 = _make_response(idempotency_key="dup-key")

        with pytest.raises(DuplicateIdempotencyKeyError) as exc_info:
            patcher.patch_draft(draft2, response2, expected_revision=2)

        assert exc_info.value.idempotency_key == "dup-key"

    def test_different_keys_both_succeed(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()

        response1 = _make_response(idempotency_key="key-a")
        new_draft1, _ = patcher.patch_draft(draft, response1, expected_revision=1)

        response2 = _make_response(idempotency_key="key-b")
        new_draft2, _ = patcher.patch_draft(new_draft1, response2, expected_revision=2)

        assert new_draft2.draft_revision == 3

    def test_seen_keys_tracked(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(idempotency_key="tracked-key")

        patcher.patch_draft(draft, response, expected_revision=1)
        assert "tracked-key" in patcher.seen_idempotency_keys


# ---------------------------------------------------------------------------
# Tests: Stage-Resume Policy (Req 9.3)
# ---------------------------------------------------------------------------


class TestStageResumePolicy:
    """Tests for stage-resume directive based on patch_type."""

    def test_column_returns_grounding(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(patch_type=PatchType.COLUMN)

        _, directive = patcher.patch_draft(draft, response, expected_revision=1)
        assert directive == StageResumeDirective.GROUNDING

    def test_operation_returns_validation_and_grounding(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(
            patch_type=PatchType.OPERATION, idempotency_key="op-key"
        )

        _, directive = patcher.patch_draft(draft, response, expected_revision=1)
        assert directive == StageResumeDirective.VALIDATION_AND_GROUNDING

    def test_value_returns_predicate_grounding(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_filter_draft()
        response = _make_response(
            patch_type=PatchType.VALUE, idempotency_key="val-key"
        )

        _, directive = patcher.patch_draft(draft, response, expected_revision=1)
        assert directive == StageResumeDirective.PREDICATE_GROUNDING

    def test_prompt_returns_extraction(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(
            patch_type=PatchType.PROMPT,
            idempotency_key="prompt-key",
            selected_value="show the total_amount column please",
        )

        _, directive = patcher.patch_draft(draft, response, expected_revision=1)
        assert directive == StageResumeDirective.EXTRACTION


# ---------------------------------------------------------------------------
# Tests: New Immutable Draft Revision (Req 9.2, 17.1, 17.3)
# ---------------------------------------------------------------------------


class TestDraftRevisionImmutability:
    """Tests for producing a new immutable draft revision."""

    def test_original_draft_not_mutated(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft(revision=5)
        response = _make_response()

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=5)

        # Original must remain at revision 5
        assert draft.draft_revision == 5
        assert new_draft.draft_revision == 6

    def test_new_draft_has_incremented_revision(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft(revision=1)
        response = _make_response()

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)
        assert new_draft.draft_revision == 2

    def test_resolution_origin_set_to_user_clarification(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response()

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)
        assert new_draft.resolution_origin == ResolutionOrigin.USER_CLARIFICATION


# ---------------------------------------------------------------------------
# Tests: ClarificationProvenance Recording (Req 15.5)
# ---------------------------------------------------------------------------


class TestClarificationProvenanceRecording:
    """Tests for recording ClarificationProvenance in the patched draft."""

    def test_provenance_added_to_extraction_provenance(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(
            idempotency_key="prov-key",
            selected_value="resolved_col",
        )

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)

        # Find ClarificationProvenance in extraction_provenance
        clarification_provs = [
            p for p in new_draft.extraction_provenance
            if p.type == "clarification"
        ]
        assert len(clarification_provs) == 1
        prov = clarification_provs[0]
        assert prov.question_id == "q-001"
        assert prov.response_id == "r-001"
        assert prov.selected_value == "resolved_col"

    def test_provenance_is_clarification_type_not_prompt_span(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response()

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)

        clarification_provs = [
            p for p in new_draft.extraction_provenance
            if p.type == "clarification"
        ]
        # Must use ClarificationProvenance, not synthetic PromptSpanProvenance
        assert len(clarification_provs) >= 1
        for prov in clarification_provs:
            assert prov.type == "clarification"


# ---------------------------------------------------------------------------
# Tests: Column Patch Behavior
# ---------------------------------------------------------------------------


class TestColumnPatchBehavior:
    """Tests for column-type patches resolving column references."""

    def test_column_patch_resolves_unresolved_references(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(
            patch_type=PatchType.COLUMN,
            selected_value="total_amount",
        )

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)

        # The unresolved column should now be resolved
        project_action = new_draft.actions[0]
        assert project_action.type == "project"
        col_ref = project_action.columns[0]
        assert col_ref.resolved_column == "total_amount"


# ---------------------------------------------------------------------------
# Tests: Prompt Patch Behavior
# ---------------------------------------------------------------------------


class TestPromptPatchBehavior:
    """Tests for prompt-type patches replacing raw_prompt."""

    def test_prompt_patch_replaces_raw_prompt(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        new_prompt = "show me the total revenue column"
        response = _make_response(
            patch_type=PatchType.PROMPT,
            idempotency_key="prompt-test",
            selected_value=new_prompt,
        )

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)
        assert new_draft.raw_prompt == new_prompt

    def test_prompt_patch_clears_actions(self):
        patcher = ClarificationDraftPatcher()
        draft = _make_draft()
        response = _make_response(
            patch_type=PatchType.PROMPT,
            idempotency_key="prompt-clear",
            selected_value="rewritten prompt",
        )

        new_draft, _ = patcher.patch_draft(draft, response, expected_revision=1)
        assert new_draft.actions == []
        assert new_draft.ambiguities == []
        assert new_draft.ignored_spans == []
