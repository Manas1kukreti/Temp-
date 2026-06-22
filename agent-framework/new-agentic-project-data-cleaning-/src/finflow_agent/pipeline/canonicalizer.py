"""Canonicalizer: pipeline-level boundary enforcement for draft-to-canonical promotion.

Wraps CanonicalIntent.from_resolved_draft() with additional pre-canonicalization
checks that ensure the pipeline has completed semantic validation, clarification,
and grounding before allowing promotion.

The Canonicalizer verifies:
- Draft resolution_status == RESOLVED
- No unresolved ambiguities remain
- All column references in actions have resolved_column set
- data_snapshot_ref is set (grounding completed against profiled data)
- resolution_origin is set (workflow path recorded)

Requirements: 21.2, 21.4, 6.2
"""

from __future__ import annotations

from finflow_agent.models.canonical import CanonicalIntent, CanonicalizeError
from finflow_agent.models.draft import (
    DraftAction,
    FilterAction,
    ProjectAction,
    DropAction,
    RenameAction,
    ResolutionStatus,
    SemanticColumnReference,
    SemanticIntentDraft,
    SortAction,
)


class Canonicalizer:
    """Pipeline-level boundary enforcer for draft-to-canonical promotion.

    Applies pre-canonicalization checks beyond what the CanonicalIntent model
    factory does, ensuring the full pipeline (validation, clarification,
    grounding) has completed before allowing construction of a CanonicalIntent.

    Requirements: 21.2, 21.4, 6.2
    """

    def canonicalize(self, draft: SemanticIntentDraft) -> CanonicalIntent:
        """Convert a fully-resolved draft to CanonicalIntent.

        Applies pre-canonicalization checks beyond what the model factory does:
        1. resolution_status must be RESOLVED
        2. ambiguities list must be empty (all clarified)
        3. All column references in actions must have resolved_column set
        4. data_snapshot_ref must be set
        5. resolution_origin must be set

        If all checks pass, delegates to CanonicalIntent.from_resolved_draft().

        Args:
            draft: The SemanticIntentDraft to promote to canonical form.

        Returns:
            A fully-resolved, frozen CanonicalIntent.

        Raises:
            CanonicalizeError: If any pre-canonicalization check fails.
        """
        self._verify_resolution_status(draft)
        self._verify_no_ambiguities(draft)
        self._verify_all_columns_resolved(draft)
        self._verify_data_snapshot_ref(draft)
        self._verify_resolution_origin(draft)

        return CanonicalIntent.from_resolved_draft(draft)

    def _verify_resolution_status(self, draft: SemanticIntentDraft) -> None:
        """Verify draft.resolution_status == RESOLVED."""
        if draft.resolution_status != ResolutionStatus.RESOLVED:
            raise CanonicalizeError(
                f"Cannot canonicalize: draft resolution_status is "
                f"'{draft.resolution_status.value}', expected 'resolved'. "
                f"Semantic validation, clarification, and grounding must complete "
                f"before canonicalization."
            )

    def _verify_no_ambiguities(self, draft: SemanticIntentDraft) -> None:
        """Verify all ambiguities have been resolved (list is empty)."""
        if draft.ambiguities:
            ambiguity_paths = [a.element_path for a in draft.ambiguities]
            raise CanonicalizeError(
                f"Cannot canonicalize: {len(draft.ambiguities)} unresolved "
                f"ambiguity marker(s) remain. Ambiguous elements: "
                f"{ambiguity_paths}. All ambiguities must be resolved via "
                f"clarification before canonicalization."
            )

    def _verify_all_columns_resolved(self, draft: SemanticIntentDraft) -> None:
        """Verify all column references in actions have resolved_column set."""
        unresolved: list[str] = []

        for i, action in enumerate(draft.actions):
            refs = _extract_column_references(action)
            for ref in refs:
                if ref.resolved_column is None:
                    unresolved.append(
                        f"actions[{i}].{action.type}: "
                        f"reference_text='{ref.reference_text}', "
                        f"kind='{ref.reference_kind.value}'"
                    )

        if unresolved:
            raise CanonicalizeError(
                f"Cannot canonicalize: {len(unresolved)} unresolved column "
                f"reference(s) found. No unresolved active execution references "
                f"may remain. Unresolved: {unresolved}"
            )

    def _verify_data_snapshot_ref(self, draft: SemanticIntentDraft) -> None:
        """Verify data_snapshot_ref is set (grounding executed against profiled data)."""
        if draft.data_snapshot_ref is None:
            raise CanonicalizeError(
                "Cannot canonicalize: data_snapshot_ref is not set. "
                "Preflight data loading and grounding must complete before "
                "canonicalization."
            )

    def _verify_resolution_origin(self, draft: SemanticIntentDraft) -> None:
        """Verify resolution_origin is set (workflow path recorded)."""
        if draft.resolution_origin is None:
            raise CanonicalizeError(
                "Cannot canonicalize: resolution_origin is not set. "
                "The workflow path that produced the resolution must be "
                "recorded before canonicalization."
            )


def _extract_column_references(action: DraftAction) -> list[SemanticColumnReference]:
    """Extract all SemanticColumnReferences from a draft action.

    Supports all action types in the discriminated union.
    """
    if isinstance(action, FilterAction):
        refs: list[SemanticColumnReference] = []
        for group in action.logical_groups:
            for predicate in group.predicates:
                refs.append(predicate.field_ref)
        return refs
    elif isinstance(action, ProjectAction):
        return list(action.columns)
    elif isinstance(action, DropAction):
        return list(action.columns)
    elif isinstance(action, SortAction):
        return list(action.keys)
    elif isinstance(action, RenameAction):
        return [col_ref for col_ref, _new_name in action.mappings]
    else:
        return []
