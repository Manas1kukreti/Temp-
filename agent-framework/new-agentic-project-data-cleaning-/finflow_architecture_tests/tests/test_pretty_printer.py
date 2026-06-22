"""Unit tests for SemanticIntentDraft pretty-printer.

Validates requirement 14.7: pretty-printer formats draft objects into
human-readable structured text for debugging and logging.
"""

import pytest

from finflow_agent.models import (
    AmbiguityMarker,
    DropAction,
    FilterAction,
    LogicalGroup,
    ProjectAction,
    PromptSpanProvenance,
    ReferenceKind,
    RenameAction,
    ResolutionOrigin,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
    UnresolvedPredicate,
    pretty_print_draft,
)


def _make_prov(start: int = 0, end: int = 10, text: str = "some text"):
    return PromptSpanProvenance(start_offset=start, end_offset=end, source_text=text)


def _make_col(name: str = "col1", kind: ReferenceKind = ReferenceKind.EXPLICIT_NAME, resolved: str | None = "col1"):
    return SemanticColumnReference(
        reference_text=name,
        reference_kind=kind,
        resolved_column=resolved,
        provenance=[_make_prov()],
    )


class TestPrettyPrintDraftBasicOutput:
    """Tests for basic draft formatting."""

    def test_contains_header(self):
        draft = SemanticIntentDraft(
            draft_id="test-id-1",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "=== SemanticIntentDraft ===" in result

    def test_contains_draft_id(self):
        draft = SemanticIntentDraft(
            draft_id="my-unique-id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "my-unique-id" in result

    def test_contains_revision(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            draft_revision=5,
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "Revision: 5" in result

    def test_contains_status(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
            resolution_status=ResolutionStatus.NEEDS_CLARIFICATION,
        )
        result = pretty_print_draft(draft)
        assert "Status:   needs_clarification" in result

    def test_contains_origin(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
            resolution_origin=ResolutionOrigin.SEMANTIC_REPAIR,
        )
        result = pretty_print_draft(draft)
        assert "Origin:   semantic_repair" in result

    def test_origin_none_shows_none(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "Origin:   none" in result

    def test_contains_prompt(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="show me columns",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert '"show me columns"' in result


class TestPrettyPrintDraftActions:
    """Tests for action formatting."""

    def test_project_action(self):
        col = _make_col("payment_method", resolved="payment_method")
        action = ProjectAction(columns=[col], provenance=[_make_prov()])
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="show payment",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "Actions (1):" in result
        assert "[0] project:" in result
        assert "payment_method" in result
        assert "explicit_name" in result

    def test_filter_action(self):
        col = _make_col("status")
        pred = UnresolvedPredicate(
            field_ref=col,
            operator="eq",
            value="active",
            provenance=[_make_prov()],
        )
        group = LogicalGroup(
            operator="and",
            predicates=[pred],
            provenance=[_make_prov()],
        )
        action = FilterAction(logical_groups=[group], provenance=[_make_prov()])
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="filter status",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "[0] filter:" in result
        assert "Group 0 (and):" in result
        assert "status" in result
        assert "'active'" in result

    def test_drop_action(self):
        col = _make_col("unused_col")
        action = DropAction(columns=[col], provenance=[_make_prov()])
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="drop column",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "[0] drop:" in result
        assert "unused_col" in result

    def test_sort_action(self):
        col = _make_col("date")
        action = SortAction(keys=[col], directions=["desc"], provenance=[_make_prov()])
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="sort by date",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "[0] sort:" in result
        assert "date" in result
        assert "desc" in result

    def test_rename_action(self):
        col = _make_col("old_name")
        action = RenameAction(
            mappings=[(col, "new_name")],
            provenance=[_make_prov()],
        )
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="rename column",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "[0] rename:" in result
        assert "old_name" in result
        assert "'new_name'" in result

    def test_empty_actions_shows_none(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "Actions (0):" in result
        assert "    none" in result


class TestPrettyPrintDraftAmbiguities:
    """Tests for ambiguity formatting."""

    def test_ambiguities_with_candidates(self):
        amb = AmbiguityMarker(
            element_path="actions[0].columns[0]",
            candidates=["col_a", "col_b"],
            provenance=[_make_prov()],
        )
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
            ambiguities=[amb],
        )
        result = pretty_print_draft(draft)
        assert "Ambiguities (1):" in result
        assert "actions[0].columns[0]" in result
        assert "col_a, col_b" in result

    def test_no_ambiguities_shows_none(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "Ambiguities (0):" in result


class TestPrettyPrintDraftIgnoredSpans:
    """Tests for ignored span formatting."""

    def test_ignored_spans_displayed(self):
        span = PromptSpanProvenance(start_offset=5, end_offset=10, source_text="noise")
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello noise world",
            actions=[],
            ignored_spans=[span],
        )
        result = pretty_print_draft(draft)
        assert "Ignored Spans (1):" in result
        assert "[5:10]" in result
        assert '"noise"' in result

    def test_no_ignored_spans_shows_none(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "Ignored Spans (0):" in result


class TestPrettyPrintDraftReturnType:
    """Tests for return type and general formatting."""

    def test_returns_string(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert isinstance(result, str)

    def test_multiline_output(self):
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="hello",
            actions=[],
        )
        result = pretty_print_draft(draft)
        assert "\n" in result
        lines = result.split("\n")
        assert len(lines) > 5

    def test_unresolved_column_shows_unresolved(self):
        col = _make_col("vague_ref", kind=ReferenceKind.GENERIC_REFERENCE, resolved=None)
        action = ProjectAction(columns=[col], provenance=[_make_prov()])
        draft = SemanticIntentDraft(
            draft_id="id",
            raw_prompt="show the column",
            actions=[action],
        )
        result = pretty_print_draft(draft)
        assert "unresolved" in result
