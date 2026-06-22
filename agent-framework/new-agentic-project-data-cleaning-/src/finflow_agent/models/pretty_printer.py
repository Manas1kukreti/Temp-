"""Pretty-printer for SemanticIntentDraft objects.

Formats draft objects into human-readable structured text for debugging and
logging use cases. Output is multi-line, indented, and suitable for inclusion
in log messages or interactive debugging sessions.

Requirements: 14.7
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finflow_agent.models.draft import SemanticIntentDraft

from finflow_agent.models.draft import (
    DraftAction,
    FilterAction,
    ProjectAction,
    DropAction,
    SortAction,
    RenameAction,
    SemanticColumnReference,
)
from finflow_agent.models.provenance import (
    PromptSpanProvenance,
    ClarificationProvenance,
    SchemaEvidenceProvenance,
)


def _format_provenance_summary(provenance: list) -> str:
    """Produce a compact one-line summary of provenance references."""
    parts: list[str] = []
    for p in provenance:
        if isinstance(p, PromptSpanProvenance):
            parts.append(f"prompt_span[{p.start_offset}:{p.end_offset}]")
        elif isinstance(p, ClarificationProvenance):
            parts.append(f"clarification[{p.question_id}]")
        elif isinstance(p, SchemaEvidenceProvenance):
            parts.append(f"schema_evidence[{p.column}]")
        else:
            parts.append("unknown")
    return ", ".join(parts) if parts else "none"


def _format_column_ref(col: SemanticColumnReference) -> str:
    """Format a single SemanticColumnReference compactly."""
    resolved = col.resolved_column if col.resolved_column else "unresolved"
    return f"{col.reference_text} ({col.reference_kind.value}, resolved={resolved})"


def _format_action(index: int, action: DraftAction) -> str:
    """Format a single DraftAction into indented lines."""
    lines: list[str] = []

    if isinstance(action, FilterAction):
        lines.append(f"    [{index}] filter:")
        for gi, group in enumerate(action.logical_groups):
            lines.append(f"        Group {gi} ({group.operator}):")
            for pi, pred in enumerate(group.predicates):
                col_str = _format_column_ref(pred.field_ref)
                neg = "NOT " if pred.negated else ""
                lines.append(
                    f"          Predicate {pi}: {neg}{col_str} {pred.operator} {pred.value!r}"
                )
        lines.append(f"        Provenance: {_format_provenance_summary(action.provenance)}")

    elif isinstance(action, ProjectAction):
        lines.append(f"    [{index}] project:")
        cols = ", ".join(_format_column_ref(c) for c in action.columns)
        lines.append(f"        Columns: {cols}")
        lines.append(f"        Provenance: {_format_provenance_summary(action.provenance)}")

    elif isinstance(action, DropAction):
        lines.append(f"    [{index}] drop:")
        cols = ", ".join(_format_column_ref(c) for c in action.columns)
        lines.append(f"        Columns: {cols}")
        lines.append(f"        Provenance: {_format_provenance_summary(action.provenance)}")

    elif isinstance(action, SortAction):
        lines.append(f"    [{index}] sort:")
        keys_str = ", ".join(
            f"{_format_column_ref(k)} {d}"
            for k, d in zip(action.keys, action.directions)
        )
        lines.append(f"        Keys: {keys_str}")
        lines.append(f"        Provenance: {_format_provenance_summary(action.provenance)}")

    elif isinstance(action, RenameAction):
        lines.append(f"    [{index}] rename:")
        mappings_str = ", ".join(
            f"{_format_column_ref(src)} -> {new_name!r}"
            for src, new_name in action.mappings
        )
        lines.append(f"        Mappings: {mappings_str}")
        lines.append(f"        Provenance: {_format_provenance_summary(action.provenance)}")

    else:
        lines.append(f"    [{index}] unknown_action_type")

    return "\n".join(lines)


def pretty_print_draft(draft: "SemanticIntentDraft") -> str:
    """Format a SemanticIntentDraft into human-readable structured text.

    Output includes:
    - Draft ID and revision
    - Resolution status and origin
    - Raw prompt
    - Actions with their types, columns, and provenance summary
    - Ambiguity markers with candidates
    - Ignored spans

    Args:
        draft: The SemanticIntentDraft object to format.

    Returns:
        A multi-line, indented string suitable for logging or debugging.
    """
    lines: list[str] = []

    lines.append("=== SemanticIntentDraft ===")
    lines.append(f"  ID:       {draft.draft_id}")
    lines.append(f"  Revision: {draft.draft_revision}")
    lines.append(f"  Status:   {draft.resolution_status.value}")
    origin_str = draft.resolution_origin.value if draft.resolution_origin else "none"
    lines.append(f"  Origin:   {origin_str}")
    lines.append(f'  Prompt:   "{draft.raw_prompt}"')

    # Actions section
    lines.append("")
    action_count = len(draft.actions)
    lines.append(f"  Actions ({action_count}):")
    if action_count == 0:
        lines.append("    none")
    else:
        for i, action in enumerate(draft.actions):
            lines.append(_format_action(i, action))

    # Ambiguities section
    lines.append("")
    amb_count = len(draft.ambiguities)
    lines.append(f"  Ambiguities ({amb_count}):")
    if amb_count == 0:
        lines.append("    none")
    else:
        for i, amb in enumerate(draft.ambiguities):
            lines.append(f"    [{i}] path: {amb.element_path}")
            lines.append(f"        Candidates: {', '.join(amb.candidates)}")
            lines.append(
                f"        Provenance: {_format_provenance_summary(amb.provenance)}"
            )

    # Ignored spans section
    lines.append("")
    span_count = len(draft.ignored_spans)
    lines.append(f"  Ignored Spans ({span_count}):")
    if span_count == 0:
        lines.append("    none")
    else:
        for i, span in enumerate(draft.ignored_spans):
            lines.append(
                f"    [{i}] [{span.start_offset}:{span.end_offset}] "
                f'"{span.source_text}"'
            )

    return "\n".join(lines)
